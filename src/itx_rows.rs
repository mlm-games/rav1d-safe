//! Row-slice inverse transform function.
//!
//! Row-slice equivalent of `inv_txfm_add` from `itx.rs`. The transform logic
//! is identical — only the pixel read/write changes from PicOffset to row slices.

#![forbid(unsafe_code)]

use crate::include::common::bitdepth::AsPrimitive;
use crate::include::common::bitdepth::BitDepth;
use crate::include::common::intops::iclip;
use crate::src::itx::Itx1dFn;
use std::cmp;

/// Add inverse-transformed residual to prediction (row-slice version of `inv_txfm_add`).
///
/// `dst_rows[0..h]`: destination rows, reads prediction and writes reconstructed pixels
/// at columns `dst_x..dst_x+w`.
/// `coeff[0..sh*sw]`: quantized residual coefficients (zeroed after use).
#[inline(never)]
pub fn inv_txfm_add_rows<BD: BitDepth>(
    dst_rows: &mut [&mut [BD::Pixel]],
    dst_x: usize,
    coeff: &mut [BD::Coef],
    eob: i32,
    w: usize,
    h: usize,
    shift: u8,
    first_1d_fn: Itx1dFn,
    second_1d_fn: Itx1dFn,
    has_dc_only: bool,
    bd: BD,
) {
    let bitdepth_max = bd.bitdepth_max().as_::<i32>();

    assert!(w >= 4 && w <= 64);
    assert!(h >= 4 && h <= 64);
    assert!(eob >= 0);

    let is_rect2 = w * 2 == h || h * 2 == w;
    let rnd = 1 << shift >> 1;

    // DC-only fast path
    if eob < has_dc_only as i32 {
        let mut dc = coeff[0].as_::<i32>();
        coeff[0] = 0.as_();
        if is_rect2 {
            dc = dc * 181 + 128 >> 8;
        }
        dc = dc * 181 + 128 >> 8;
        dc = dc + rnd >> shift;
        dc = dc * 181 + 128 + 2048 >> 12;
        for y in 0..h {
            let dst = &mut dst_rows[y][dst_x..dst_x + w];
            for x in 0..w {
                dst[x] = bd.iclip_pixel(dst[x].as_::<i32>() + dc);
            }
        }
        return;
    }

    let sh = cmp::min(h, 32);
    let sw = cmp::min(w, 32);

    let coeff = &mut coeff[..sh * sw];

    let row_clip_min;
    let col_clip_min;
    if BD::BITDEPTH == 8 {
        row_clip_min = i16::MIN as i32;
        col_clip_min = i16::MIN as i32;
    } else {
        row_clip_min = (!bitdepth_max) << 7;
        col_clip_min = (!bitdepth_max) << 5;
    }
    let row_clip_max = !row_clip_min;
    let col_clip_max = !col_clip_min;

    let mut tmp = [0i32; 64 * 64];
    let mut c = &mut tmp[..];
    for y in 0..sh {
        if is_rect2 {
            for x in 0..sw {
                c[x] = coeff[y + x * sh].as_::<i32>() * 181 + 128 >> 8;
            }
        } else {
            for x in 0..sw {
                c[x] = coeff[y + x * sh].as_();
            }
        }
        first_1d_fn(c, 1.try_into().unwrap(), row_clip_min, row_clip_max);
        c = &mut c[w..];
    }

    coeff.fill(0.into());
    for i in 0..w * sh {
        tmp[i] = iclip(tmp[i] + rnd >> shift, col_clip_min, col_clip_max);
    }

    for x in 0..w {
        second_1d_fn(
            &mut tmp[x..],
            w.try_into().unwrap(),
            col_clip_min,
            col_clip_max,
        );
    }

    // Final pixel write — the only difference from itx.rs
    for y in 0..h {
        let dst = &mut dst_rows[y][dst_x..dst_x + w];
        for x in 0..w {
            dst[x] = bd.iclip_pixel(dst[x].as_::<i32>() + (tmp[y * w + x] + 8 >> 4));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::include::common::bitdepth::BitDepth8;
    use crate::src::itx_1d::rav1d_inv_dct4_1d_c;

    type BD = BitDepth8;

    #[test]
    fn test_inv_txfm_add_dc_only() {
        // 4×4 block, DC=0 → no change to dst
        let mut coeff: Vec<i16> = vec![0; 16];
        let mut dst_buf = vec![128u8; 16];
        let mut dst_rows: Vec<&mut [u8]> = dst_buf.chunks_mut(4).collect();

        inv_txfm_add_rows::<BD>(
            &mut dst_rows,
            0,
            &mut coeff,
            0,   // eob=0 → DC path
            4,   // w
            4,   // h
            0,   // shift
            rav1d_inv_dct4_1d_c,
            rav1d_inv_dct4_1d_c,
            true, // has_dc_only
            BD::new(()),
        );

        // DC=0 → all pixels remain 128
        for row in &dst_rows {
            for &px in row.iter() {
                assert_eq!(px, 128);
            }
        }
    }

    #[test]
    fn test_inv_txfm_add_dc_nonzero() {
        // 4×4 block with non-zero DC
        let mut coeff: Vec<i16> = vec![0; 16];
        coeff[0] = 64; // positive DC
        let mut dst_buf = vec![100u8; 16];
        let mut dst_rows: Vec<&mut [u8]> = dst_buf.chunks_mut(4).collect();

        inv_txfm_add_rows::<BD>(
            &mut dst_rows,
            0,
            &mut coeff,
            0,
            4,
            4,
            0,
            rav1d_inv_dct4_1d_c,
            rav1d_inv_dct4_1d_c,
            true,
            BD::new(()),
        );

        // All pixels should increase from DC contribution
        for row in &dst_rows {
            for &px in row.iter() {
                assert!(px > 100, "pixel should increase with positive DC, got {px}");
            }
        }

        // Coefficient should be zeroed
        assert_eq!(coeff[0], 0);
    }
}
