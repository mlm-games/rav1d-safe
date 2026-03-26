//! Row-slice intra prediction functions.
//!
//! Row-slice equivalents of ipred functions from `ipred.rs`. These write to
//! `&mut [&mut [BD::Pixel]]` row slices instead of PicOffset. The edge buffer
//! (topleft) is unchanged — it's already a separate scratch array.

#![forbid(unsafe_code)]

use crate::include::common::bitdepth::AsPrimitive;
use crate::include::common::bitdepth::BitDepth;
use crate::include::common::bitdepth::BPC;
use crate::include::common::intops::apply_sign;
use crate::src::internal::SCRATCH_AC_TXTP_LEN;
use crate::src::internal::SCRATCH_EDGE_LEN;
use std::ffi::c_int;
use std::ffi::c_uint;

// === DC generation helpers (unchanged from ipred.rs) ===

fn dc_gen_top<BD: BitDepth>(
    topleft: &[BD::Pixel; SCRATCH_EDGE_LEN],
    offset: usize,
    width: c_int,
) -> c_uint {
    let mut dc = width as u32 >> 1;
    for i in 0..width as usize {
        dc += topleft[offset + 1 + i].as_::<c_uint>();
    }
    dc >> width.trailing_zeros()
}

fn dc_gen_left<BD: BitDepth>(
    topleft: &[BD::Pixel; SCRATCH_EDGE_LEN],
    offset: usize,
    height: c_int,
) -> c_uint {
    let mut dc = height as u32 >> 1;
    for i in 0..height as usize {
        dc += topleft[offset - (1 + i)].as_::<c_uint>();
    }
    dc >> height.trailing_zeros()
}

fn dc_gen<BD: BitDepth>(
    topleft: &[BD::Pixel; SCRATCH_EDGE_LEN],
    offset: usize,
    width: c_int,
    height: c_int,
) -> c_uint {
    let (multiplier_1x2, multiplier_1x4, base_shift) = match BD::BPC {
        BPC::BPC8 => (0x5556u32, 0x3334u32, 16u32),
        BPC::BPC16 => (0xAAABu32, 0x6667u32, 17u32),
    };

    let mut dc = (width + height >> 1) as u32;
    for i in 0..width as usize {
        dc += topleft[offset + i + 1].as_::<c_uint>();
    }
    for i in 0..height as usize {
        dc += topleft[offset - (1 + i)].as_::<c_uint>();
    }

    if width != height {
        dc = dc.wrapping_mul(if width * 2 == height || height * 2 == width {
            multiplier_1x2
        } else {
            multiplier_1x4
        });
        dc >>= base_shift;
    } else {
        dc >>= (width + height).trailing_zeros();
    }
    dc
}

// === Prediction modes ===

/// DC fill (splat a constant value).
#[inline(never)]
pub fn splat_dc_rows<BD: BitDepth>(
    dst_rows: &mut [&mut [BD::Pixel]],
    dst_x: usize,
    width: usize,
    height: usize,
    dc: c_int,
    bd: BD,
) {
    assert!(dc <= bd.bitdepth_max().as_::<c_int>());
    let dc = dc.as_::<BD::Pixel>();
    for y in 0..height {
        let dst = &mut dst_rows[y][dst_x..dst_x + width];
        dst.fill(dc);
    }
}

/// DC prediction from top + left edges.
pub fn dc_pred_rows<BD: BitDepth>(
    dst_rows: &mut [&mut [BD::Pixel]],
    dst_x: usize,
    topleft: &[BD::Pixel; SCRATCH_EDGE_LEN],
    offset: usize,
    width: c_int,
    height: c_int,
    bd: BD,
) {
    let dc = dc_gen::<BD>(topleft, offset, width, height);
    splat_dc_rows::<BD>(dst_rows, dst_x, width as usize, height as usize, dc as c_int, bd);
}

/// DC prediction from top edge only.
pub fn dc_top_pred_rows<BD: BitDepth>(
    dst_rows: &mut [&mut [BD::Pixel]],
    dst_x: usize,
    topleft: &[BD::Pixel; SCRATCH_EDGE_LEN],
    offset: usize,
    width: c_int,
    height: c_int,
    bd: BD,
) {
    let dc = dc_gen_top::<BD>(topleft, offset, width);
    splat_dc_rows::<BD>(dst_rows, dst_x, width as usize, height as usize, dc as c_int, bd);
}

/// DC prediction from left edge only.
pub fn dc_left_pred_rows<BD: BitDepth>(
    dst_rows: &mut [&mut [BD::Pixel]],
    dst_x: usize,
    topleft: &[BD::Pixel; SCRATCH_EDGE_LEN],
    offset: usize,
    width: c_int,
    height: c_int,
    bd: BD,
) {
    let dc = dc_gen_left::<BD>(topleft, offset, height);
    splat_dc_rows::<BD>(dst_rows, dst_x, width as usize, height as usize, dc as c_int, bd);
}

/// DC_128 prediction (no edges available).
pub fn dc_128_pred_rows<BD: BitDepth>(
    dst_rows: &mut [&mut [BD::Pixel]],
    dst_x: usize,
    width: c_int,
    height: c_int,
    bd: BD,
) {
    let dc = bd.bitdepth_max().as_::<c_int>() + 1 >> 1;
    splat_dc_rows::<BD>(dst_rows, dst_x, width as usize, height as usize, dc, bd);
}

/// Vertical prediction (copy top row to all rows).
#[inline(never)]
pub fn v_pred_rows<BD: BitDepth>(
    dst_rows: &mut [&mut [BD::Pixel]],
    dst_x: usize,
    topleft: &[BD::Pixel; SCRATCH_EDGE_LEN],
    offset: usize,
    width: usize,
    height: usize,
) {
    let top = &topleft[offset + 1..offset + 1 + width];
    for y in 0..height {
        dst_rows[y][dst_x..dst_x + width].copy_from_slice(top);
    }
}

/// Horizontal prediction (replicate left pixel across each row).
#[inline(never)]
pub fn h_pred_rows<BD: BitDepth>(
    dst_rows: &mut [&mut [BD::Pixel]],
    dst_x: usize,
    topleft: &[BD::Pixel; SCRATCH_EDGE_LEN],
    offset: usize,
    width: usize,
    height: usize,
) {
    for y in 0..height {
        let left = topleft[offset - (1 + y)];
        dst_rows[y][dst_x..dst_x + width].fill(left);
    }
}

/// Paeth prediction.
#[inline(never)]
pub fn paeth_pred_rows<BD: BitDepth>(
    dst_rows: &mut [&mut [BD::Pixel]],
    dst_x: usize,
    topleft: &[BD::Pixel; SCRATCH_EDGE_LEN],
    offset: usize,
    width: usize,
    height: usize,
) {
    let tl = topleft[offset].as_::<i32>();
    for y in 0..height {
        let left = topleft[offset - (y + 1)].as_::<i32>();
        let dst = &mut dst_rows[y][dst_x..dst_x + width];
        for x in 0..width {
            let top = topleft[offset + 1 + x].as_::<i32>();
            let base = top + left - tl;
            let dt = (top - base).abs();
            let dl = (left - base).abs();
            let dtl = (tl - base).abs();
            dst[x] = if dt <= dl && dt <= dtl {
                top
            } else if dl <= dtl {
                left
            } else {
                tl
            }
            .as_::<BD::Pixel>();
        }
    }
}

/// Smooth prediction (bilinear blend of all 4 edges).
#[inline(never)]
pub fn smooth_pred_rows<BD: BitDepth>(
    dst_rows: &mut [&mut [BD::Pixel]],
    dst_x: usize,
    topleft: &[BD::Pixel; SCRATCH_EDGE_LEN],
    offset: usize,
    width: usize,
    height: usize,
) {
    use crate::src::tables::dav1d_sm_weights;

    let right = topleft[offset + width].as_::<u32>();
    let bottom = topleft[offset - height as usize].as_::<u32>();

    for y in 0..height {
        let left = topleft[offset - (y + 1)].as_::<u32>();
        let w_ver = dav1d_sm_weights[height + y] as u32;
        let dst = &mut dst_rows[y][dst_x..dst_x + width];
        for x in 0..width {
            let top = topleft[offset + 1 + x].as_::<u32>();
            let w_hor = dav1d_sm_weights[width + x] as u32;
            let pred = w_ver * top + (256 - w_ver) * bottom
                + w_hor * left + (256 - w_hor) * right;
            dst[x] = ((pred + 256) >> 9).as_::<BD::Pixel>();
        }
    }
}

/// Smooth vertical prediction (blend top and bottom only).
#[inline(never)]
pub fn smooth_v_pred_rows<BD: BitDepth>(
    dst_rows: &mut [&mut [BD::Pixel]],
    dst_x: usize,
    topleft: &[BD::Pixel; SCRATCH_EDGE_LEN],
    offset: usize,
    width: usize,
    height: usize,
) {
    use crate::src::tables::dav1d_sm_weights;

    let bottom = topleft[offset - height as usize].as_::<u32>();

    for y in 0..height {
        let w_ver = dav1d_sm_weights[height + y] as u32;
        let dst = &mut dst_rows[y][dst_x..dst_x + width];
        for x in 0..width {
            let top = topleft[offset + 1 + x].as_::<u32>();
            dst[x] = ((w_ver * top + (256 - w_ver) * bottom + 128) >> 8)
                .as_::<BD::Pixel>();
        }
    }
}

/// Smooth horizontal prediction (blend left and right only).
#[inline(never)]
pub fn smooth_h_pred_rows<BD: BitDepth>(
    dst_rows: &mut [&mut [BD::Pixel]],
    dst_x: usize,
    topleft: &[BD::Pixel; SCRATCH_EDGE_LEN],
    offset: usize,
    width: usize,
    height: usize,
) {
    use crate::src::tables::dav1d_sm_weights;

    let right = topleft[offset + width].as_::<u32>();

    for y in 0..height {
        let left = topleft[offset - (y + 1)].as_::<u32>();
        let dst = &mut dst_rows[y][dst_x..dst_x + width];
        for x in 0..width {
            let w_hor = dav1d_sm_weights[width + x] as u32;
            dst[x] = ((w_hor * left + (256 - w_hor) * right + 128) >> 8)
                .as_::<BD::Pixel>();
        }
    }
}

/// CFL (chroma-from-luma) prediction.
#[inline(never)]
pub fn cfl_pred_rows<BD: BitDepth>(
    dst_rows: &mut [&mut [BD::Pixel]],
    dst_x: usize,
    width: usize,
    height: usize,
    dc: c_int,
    ac: &[i16; SCRATCH_AC_TXTP_LEN],
    alpha: c_int,
    bd: BD,
) {
    let ac = &ac[..width * height];
    for y in 0..height {
        let dst = &mut dst_rows[y][dst_x..dst_x + width];
        for x in 0..width {
            let diff = alpha * ac[y * width + x] as c_int;
            dst[x] = bd.iclip_pixel(dc + apply_sign(diff.abs() + 32 >> 6, diff));
        }
    }
}

/// Palette prediction.
#[inline(never)]
pub fn pal_pred_rows<BD: BitDepth>(
    dst_rows: &mut [&mut [BD::Pixel]],
    dst_x: usize,
    pal: &[BD::Pixel; 8],
    idx: &[u8],
    width: usize,
    height: usize,
) {
    for y in 0..height {
        let dst = &mut dst_rows[y][dst_x..dst_x + width];
        for x in (0..width).step_by(2) {
            let i = idx[y * (width / 2) + x / 2];
            dst[x] = pal[(i & 7) as usize];
            if x + 1 < width {
                dst[x + 1] = pal[((i >> 4) & 7) as usize];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::include::common::bitdepth::BitDepth8;

    type BD = BitDepth8;
    const OFF: usize = 128; // center of SCRATCH_EDGE_LEN

    fn make_topleft(top: &[u8], left: &[u8], tl: u8) -> [u8; SCRATCH_EDGE_LEN] {
        let mut buf = [0u8; SCRATCH_EDGE_LEN];
        buf[OFF] = tl;
        for (i, &v) in top.iter().enumerate() {
            buf[OFF + 1 + i] = v;
        }
        for (i, &v) in left.iter().enumerate() {
            buf[OFF - 1 - i] = v;
        }
        buf
    }

    #[test]
    fn test_v_pred_copies_top_row() {
        let topleft = make_topleft(&[10, 20, 30, 40], &[0; 4], 0);
        let mut dst_buf = vec![0u8; 16];
        let mut dst_rows: Vec<&mut [u8]> = dst_buf.chunks_mut(4).collect();

        v_pred_rows::<BD>(&mut dst_rows, 0, &topleft, OFF, 4, 4);

        for row in &dst_rows {
            assert_eq!(*row, [10, 20, 30, 40]);
        }
    }

    #[test]
    fn test_h_pred_replicates_left() {
        let topleft = make_topleft(&[0; 4], &[10, 20, 30, 40], 0);
        let mut dst_buf = vec![0u8; 16];
        let mut dst_rows: Vec<&mut [u8]> = dst_buf.chunks_mut(4).collect();

        h_pred_rows::<BD>(&mut dst_rows, 0, &topleft, OFF, 4, 4);

        assert_eq!(dst_rows[0], &[10, 10, 10, 10]);
        assert_eq!(dst_rows[1], &[20, 20, 20, 20]);
        assert_eq!(dst_rows[2], &[30, 30, 30, 30]);
        assert_eq!(dst_rows[3], &[40, 40, 40, 40]);
    }

    #[test]
    fn test_paeth_pred_basic() {
        // With top=100, left=100, tl=100 → all predictions should be 100
        let topleft = make_topleft(&[100; 4], &[100; 4], 100);
        let mut dst_buf = vec![0u8; 16];
        let mut dst_rows: Vec<&mut [u8]> = dst_buf.chunks_mut(4).collect();

        paeth_pred_rows::<BD>(&mut dst_rows, 0, &topleft, OFF, 4, 4);

        for row in &dst_rows {
            for &px in row.iter() {
                assert_eq!(px, 100);
            }
        }
    }

    #[test]
    fn test_dc_128_pred() {
        let mut dst_buf = vec![0u8; 16];
        let mut dst_rows: Vec<&mut [u8]> = dst_buf.chunks_mut(4).collect();

        dc_128_pred_rows::<BD>(&mut dst_rows, 0, 4, 4, BD::new(()));

        for row in &dst_rows {
            for &px in row.iter() {
                assert_eq!(px, 128); // (255 + 1) >> 1 = 128
            }
        }
    }

    #[test]
    fn test_dc_pred_averages_edges() {
        // Top = [100, 100, 100, 100], Left = [100, 100, 100, 100]
        // DC should be 100
        let topleft = make_topleft(&[100; 4], &[100; 4], 100);
        let mut dst_buf = vec![0u8; 16];
        let mut dst_rows: Vec<&mut [u8]> = dst_buf.chunks_mut(4).collect();

        dc_pred_rows::<BD>(&mut dst_rows, 0, &topleft, OFF, 4, 4, BD::new(()));

        for row in &dst_rows {
            for &px in row.iter() {
                assert_eq!(px, 100);
            }
        }
    }
}
