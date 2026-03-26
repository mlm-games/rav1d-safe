//! Row-slice motion compensation functions.
//!
//! These are direct equivalents of the `*_rust` functions in `mc.rs`, but operating
//! on pre-split row slices (`&mut [&mut [BD::Pixel]]`) instead of `PicOffset`.
//! This enables `split_at_mut`-based ownership for rayon parallelism.
//!
//! Every function here has an exact semantic match in `mc.rs`. The access patterns
//! are documented in RAYON-THREADING-SPEC.md.

#![forbid(unsafe_code)]

use crate::include::common::bitdepth::AsPrimitive;
use crate::include::common::bitdepth::BitDepth;
use crate::include::dav1d::headers::Rav1dFilterMode;
use crate::src::internal::COMPINTER_LEN;
use crate::src::internal::SCRATCH_INTER_INTRA_BUF_LEN;
use crate::src::tables::dav1d_mc_subpel_filters;
use crate::src::tables::dav1d_obmc_masks;
use to_method::To;

const MID_STRIDE: usize = 128;

// === Fullpel functions ===

/// Fullpel pixel copy (row-slice version of `put_rust`).
///
/// Copies `w` pixels per row from `src_rows` starting at column `src_x`
/// to `dst_rows` starting at column `dst_x`.
/// `dst_rows.len()` and `src_rows.len()` must both be >= h (the height is
/// determined by the shorter of the two).
#[inline(never)]
pub fn put_rows<BD: BitDepth>(
    dst_rows: &mut [&mut [BD::Pixel]],
    dst_x: usize,
    src_rows: &[&[BD::Pixel]],
    src_x: usize,
    w: usize,
) {
    for (dst_row, src_row) in dst_rows.iter_mut().zip(src_rows.iter()) {
        dst_row[dst_x..dst_x + w].copy_from_slice(&src_row[src_x..src_x + w]);
    }
}

/// Fullpel preparation to i16 temp buffer (row-slice version of `prep_rust`).
#[inline(never)]
pub fn prep_rows<BD: BitDepth>(
    tmp: &mut [i16],
    src_rows: &[&[BD::Pixel]],
    src_x: usize,
    w: usize,
    h: usize,
    bd: BD,
) {
    let intermediate_bits = bd.get_intermediate_bits();
    for y in 0..h {
        let src = &src_rows[y][src_x..src_x + w];
        let dst = &mut tmp[y * w..][..w];
        for x in 0..w {
            dst[x] = BD::sub_prep_bias(src[x].as_::<i32>() << intermediate_bits);
        }
    }
}

// === 8-tap filter helpers ===

#[derive(Clone, Copy)]
struct FilterResult {
    pixel: i32,
}

impl FilterResult {
    pub fn get(&self) -> i16 {
        self.pixel as i16
    }

    pub fn rnd(&self, sh: u8) -> Self {
        Self {
            pixel: (self.pixel + ((1 << sh) >> 1)) >> sh,
        }
    }

    pub fn rnd2(&self, sh: u8, rnd: u8) -> Self {
        Self {
            pixel: (self.pixel + rnd as i32) >> sh,
        }
    }

    pub fn clip<BD: BitDepth>(&self, bd: BD) -> BD::Pixel {
        bd.iclip_pixel(self.pixel)
    }

    pub fn sub_prep_bias<BD: BitDepth>(&self) -> i16 {
        BD::sub_prep_bias(self.pixel)
    }
}

fn filter_8tap_mid(mid: &[[i16; MID_STRIDE]], x: usize, f: &[i8; 8]) -> FilterResult {
    let pixel = (0..f.len()).map(|y| f[y] as i32 * mid[y][x] as i32).sum();
    FilterResult { pixel }
}

/// 8-tap horizontal filter on a row, reading from an immutable row slice.
/// `src_row` must have at least `x + 5` pixels (reads at offsets x-3..x+5).
fn filter_8tap_row<BD: BitDepth>(src_row: &[BD::Pixel], x: usize, f: &[i8; 8]) -> FilterResult {
    let pixel = (0..8)
        .map(|i| {
            let px_idx = (x as isize + i as isize - 3) as usize;
            f[i] as i32 * src_row[px_idx].to::<i32>()
        })
        .sum();
    FilterResult { pixel }
}

/// 8-tap vertical filter reading from multiple row slices.
/// `rows[0..8]` must be valid, reading column `x` from each.
fn filter_8tap_col<BD: BitDepth>(rows: &[&[BD::Pixel]], x: usize, f: &[i8; 8]) -> FilterResult {
    let pixel = (0..8)
        .map(|i| f[i] as i32 * rows[i][x].to::<i32>())
        .sum();
    FilterResult { pixel }
}

fn get_filter(m: usize, d: usize, filter_type: Rav1dFilterMode) -> Option<&'static [i8; 8]> {
    let m = m.checked_sub(1)?;
    let i = if d > 4 {
        filter_type as u8
    } else {
        3 + (filter_type as u8 & 1)
    };
    Some(&dav1d_mc_subpel_filters[i as usize][m])
}

// === 8-tap subpel functions ===

/// 8-tap subpel put (row-slice version of `put_8tap_rust`).
///
/// `dst_rows[0..h]`: destination rows, writes at columns `dst_x..dst_x+w`.
/// `src_rows`: source rows from reference frame. Must include 3 extra rows above
/// and 4 extra rows below (for vertical filtering). Index 0 in src_rows corresponds
/// to 3 rows ABOVE the block start (i.e., src_rows[3] is the first block row).
/// For horizontal filtering, each row must have 3 extra pixels left and 4 right.
#[inline(never)]
pub fn put_8tap_rows<BD: BitDepth>(
    dst_rows: &mut [&mut [BD::Pixel]],
    dst_x: usize,
    src_rows: &[&[BD::Pixel]],
    src_x: usize,
    w: usize,
    h: usize,
    mx: usize,
    my: usize,
    (h_filter_type, v_filter_type): (Rav1dFilterMode, Rav1dFilterMode),
    bd: BD,
) {
    let intermediate_bits = bd.get_intermediate_bits();
    let intermediate_rnd = 32 + (1 << 6 - intermediate_bits >> 1);

    let fh = get_filter(mx, w, h_filter_type);
    let fv = get_filter(my, h, v_filter_type);

    if let Some(fh) = fh {
        if let Some(fv) = fv {
            // Both horizontal and vertical filtering
            let tmp_h = h + 7;
            let mut mid = [[0i16; MID_STRIDE]; 135];

            for y in 0..tmp_h {
                // src_rows[y] corresponds to row (y - 3) relative to block start
                // src_rows is indexed with the 3-row margin already included
                let src = &src_rows[y];
                for x in 0..w {
                    mid[y][x] = filter_8tap_row::<BD>(src, src_x + x, fh)
                        .rnd(6 - intermediate_bits)
                        .get();
                }
            }

            for y in 0..h {
                let dst = &mut dst_rows[y][dst_x..dst_x + w];
                for x in 0..w {
                    dst[x] = filter_8tap_mid(&mid[y..], x, fv)
                        .rnd(6 + intermediate_bits)
                        .clip(bd);
                }
            }
        } else {
            // Horizontal only
            for y in 0..h {
                let src = &src_rows[y + 3]; // skip the 3-row margin
                let dst = &mut dst_rows[y][dst_x..dst_x + w];
                for x in 0..w {
                    dst[x] = filter_8tap_row::<BD>(src, src_x + x, fh)
                        .rnd2(6, intermediate_rnd)
                        .clip(bd);
                }
            }
        }
    } else if let Some(fv) = fv {
        // Vertical only
        for y in 0..h {
            // For vertical filter, we read 8 rows centered around the target row
            // src_rows[y..y+8] (since src_rows has 3-row margin, y+0 = 3 rows above target)
            let dst = &mut dst_rows[y][dst_x..dst_x + w];
            for x in 0..w {
                // Build a temporary slice of 8 row references for the vertical filter
                let col_vals: [&[BD::Pixel]; 8] = [
                    src_rows[y],
                    src_rows[y + 1],
                    src_rows[y + 2],
                    src_rows[y + 3],
                    src_rows[y + 4],
                    src_rows[y + 5],
                    src_rows[y + 6],
                    src_rows[y + 7],
                ];
                dst[x] = filter_8tap_col::<BD>(&col_vals, src_x + x, fv)
                    .rnd(6)
                    .clip(bd);
            }
        }
    } else {
        // Fullpel — delegate to put_rows
        put_rows::<BD>(
            &mut dst_rows[..h],
            dst_x,
            &src_rows[3..3 + h], // skip the 3-row margin
            src_x,
            w,
        );
    }
}

/// 8-tap subpel prep (row-slice version of `prep_8tap_rust`).
///
/// Same source access pattern as `put_8tap_rows`, but writes to i16 temp buffer.
#[inline(never)]
pub fn prep_8tap_rows<BD: BitDepth>(
    tmp: &mut [i16],
    src_rows: &[&[BD::Pixel]],
    src_x: usize,
    w: usize,
    h: usize,
    mx: usize,
    my: usize,
    (h_filter_type, v_filter_type): (Rav1dFilterMode, Rav1dFilterMode),
    bd: BD,
) {
    let intermediate_bits = bd.get_intermediate_bits();

    let fh = get_filter(mx, w, h_filter_type);
    let fv = get_filter(my, h, v_filter_type);

    if let Some(fh) = fh {
        if let Some(fv) = fv {
            let tmp_h = h + 7;
            let mut mid = [[0i16; MID_STRIDE]; 135];

            for y in 0..tmp_h {
                let src = &src_rows[y];
                for x in 0..w {
                    mid[y][x] = filter_8tap_row::<BD>(src, src_x + x, fh)
                        .rnd(6 - intermediate_bits)
                        .get();
                }
            }

            for y in 0..h {
                let dst = &mut tmp[y * w..][..w];
                for x in 0..w {
                    dst[x] = filter_8tap_mid(&mid[y..], x, fv)
                        .rnd(6)
                        .sub_prep_bias::<BD>();
                }
            }
        } else {
            for y in 0..h {
                let src = &src_rows[y + 3];
                let dst = &mut tmp[y * w..][..w];
                for x in 0..w {
                    dst[x] = filter_8tap_row::<BD>(src, src_x + x, fh)
                        .rnd(6 - intermediate_bits)
                        .sub_prep_bias::<BD>();
                }
            }
        }
    } else if let Some(fv) = fv {
        for y in 0..h {
            let dst = &mut tmp[y * w..][..w];
            for x in 0..w {
                let col_vals: [&[BD::Pixel]; 8] = [
                    src_rows[y],
                    src_rows[y + 1],
                    src_rows[y + 2],
                    src_rows[y + 3],
                    src_rows[y + 4],
                    src_rows[y + 5],
                    src_rows[y + 6],
                    src_rows[y + 7],
                ];
                dst[x] = filter_8tap_col::<BD>(&col_vals, src_x + x, fv)
                    .rnd(6 - intermediate_bits)
                    .sub_prep_bias::<BD>();
            }
        }
    } else {
        prep_rows::<BD>(tmp, &src_rows[3..3 + h], src_x, w, h, bd);
    }
}

// === Bilinear filter functions ===

/// Bilinear put (row-slice version of `put_bilin_rust`).
///
/// `src_rows`: for my≠0, needs h+1 rows. src_rows[0] is the first block row (no margin).
/// For mx≠0, each row needs 1 extra pixel to the right of src_x+w.
#[inline(never)]
pub fn put_bilin_rows<BD: BitDepth>(
    dst_rows: &mut [&mut [BD::Pixel]],
    dst_x: usize,
    src_rows: &[&[BD::Pixel]],
    src_x: usize,
    w: usize,
    h: usize,
    mx: usize,
    my: usize,
    bd: BD,
) {
    let intermediate_bits = bd.get_intermediate_bits();
    let intermediate_rnd = 32 + (1 << 6 - intermediate_bits >> 1);

    if mx != 0 {
        if my != 0 {
            let tmp_h = h + 1;
            let mut mid = [[0i16; MID_STRIDE]; 129];

            for y in 0..tmp_h {
                let src = &src_rows[y];
                for x in 0..w {
                    let sx = src_x + x;
                    let p0 = src[sx].to::<i32>();
                    let p1 = src[sx + 1].to::<i32>();
                    mid[y][x] = (16 * p0 + mx as i32 * (p1 - p0)) as i16;
                }
            }

            for y in 0..h {
                let dst = &mut dst_rows[y][dst_x..dst_x + w];
                for x in 0..w {
                    let m0 = mid[y][x] as i32;
                    let m1 = mid[y + 1][x] as i32;
                    dst[x] = bd.iclip_pixel(
                        ((16 * m0 + my as i32 * (m1 - m0) + intermediate_rnd as i32)
                            >> (4 + intermediate_bits))
                            .to::<i32>(),
                    );
                }
            }
        } else {
            for y in 0..h {
                let src = &src_rows[y];
                let dst = &mut dst_rows[y][dst_x..dst_x + w];
                for x in 0..w {
                    let sx = src_x + x;
                    let p0 = src[sx].to::<i32>();
                    let p1 = src[sx + 1].to::<i32>();
                    dst[x] = bd
                        .iclip_pixel(((p0 * 16 + mx as i32 * (p1 - p0) + 8) >> 4).to::<i32>());
                }
            }
        }
    } else if my != 0 {
        for y in 0..h {
            let s0 = &src_rows[y];
            let s1 = &src_rows[y + 1];
            let dst = &mut dst_rows[y][dst_x..dst_x + w];
            for x in 0..w {
                let sx = src_x + x;
                let p0 = s0[sx].to::<i32>();
                let p1 = s1[sx].to::<i32>();
                dst[x] =
                    bd.iclip_pixel(((p0 * 16 + my as i32 * (p1 - p0) + 8) >> 4).to::<i32>());
            }
        }
    } else {
        put_rows::<BD>(&mut dst_rows[..h], dst_x, &src_rows[..h], src_x, w);
    }
}

/// Bilinear prep (row-slice version of `prep_bilin_rust`).
#[inline(never)]
pub fn prep_bilin_rows<BD: BitDepth>(
    tmp: &mut [i16],
    src_rows: &[&[BD::Pixel]],
    src_x: usize,
    w: usize,
    h: usize,
    mx: usize,
    my: usize,
    bd: BD,
) {
    let intermediate_bits = bd.get_intermediate_bits();

    if mx != 0 {
        if my != 0 {
            let tmp_h = h + 1;
            let mut mid = [[0i16; MID_STRIDE]; 129];

            for y in 0..tmp_h {
                let src = &src_rows[y];
                for x in 0..w {
                    let sx = src_x + x;
                    let p0 = src[sx].to::<i32>();
                    let p1 = src[sx + 1].to::<i32>();
                    mid[y][x] = (16 * p0 + mx as i32 * (p1 - p0)) as i16;
                }
            }

            for y in 0..h {
                let dst = &mut tmp[y * w..][..w];
                for x in 0..w {
                    let m0 = mid[y][x] as i32;
                    let m1 = mid[y + 1][x] as i32;
                    dst[x] =
                        BD::sub_prep_bias((16 * m0 + my as i32 * (m1 - m0) + 128) >> 8);
                }
            }
        } else {
            for y in 0..h {
                let src = &src_rows[y];
                let dst = &mut tmp[y * w..][..w];
                for x in 0..w {
                    let sx = src_x + x;
                    let p0 = src[sx].to::<i32>();
                    let p1 = src[sx + 1].to::<i32>();
                    dst[x] = BD::sub_prep_bias(
                        (p0 << intermediate_bits) + mx as i32 * (p1 - p0),
                    );
                }
            }
        }
    } else if my != 0 {
        for y in 0..h {
            let s0 = &src_rows[y];
            let s1 = &src_rows[y + 1];
            let dst = &mut tmp[y * w..][..w];
            for x in 0..w {
                let sx = src_x + x;
                let p0 = s0[sx].to::<i32>();
                let p1 = s1[sx].to::<i32>();
                dst[x] =
                    BD::sub_prep_bias((p0 << intermediate_bits) + my as i32 * (p1 - p0));
            }
        }
    } else {
        prep_rows::<BD>(tmp, &src_rows[..h], src_x, w, h, bd);
    }
}

// === Compound functions (write to dst from i16 temp buffers) ===

/// Average two i16 temps → pixels (row-slice version of `avg_rust`).
#[inline(never)]
pub fn avg_rows<BD: BitDepth>(
    dst_rows: &mut [&mut [BD::Pixel]],
    dst_x: usize,
    tmp1: &[i16],
    tmp2: &[i16],
    w: usize,
    h: usize,
    bd: BD,
) {
    let intermediate_bits = bd.get_intermediate_bits();
    let sh = intermediate_bits + 1;
    let rnd = (1 << intermediate_bits) + i32::from(BD::PREP_BIAS) * 2;
    for y in 0..h {
        let dst = &mut dst_rows[y][dst_x..dst_x + w];
        for x in 0..w {
            dst[x] = bd.iclip_pixel(
                (tmp1[y * w + x] as i32 + tmp2[y * w + x] as i32 + rnd) >> sh,
            );
        }
    }
}

/// Weighted average (row-slice version of `w_avg_rust`).
#[inline(never)]
pub fn w_avg_rows<BD: BitDepth>(
    dst_rows: &mut [&mut [BD::Pixel]],
    dst_x: usize,
    tmp1: &[i16],
    tmp2: &[i16],
    w: usize,
    h: usize,
    weight: i32,
    bd: BD,
) {
    let intermediate_bits = bd.get_intermediate_bits();
    let sh = intermediate_bits + 4;
    let rnd = (8 << intermediate_bits) + i32::from(BD::PREP_BIAS) * 16;
    for y in 0..h {
        let dst = &mut dst_rows[y][dst_x..dst_x + w];
        for x in 0..w {
            dst[x] = bd.iclip_pixel(
                (tmp1[y * w + x] as i32 * weight
                    + tmp2[y * w + x] as i32 * (16 - weight)
                    + rnd)
                    >> sh,
            );
        }
    }
}

/// Mask-blended compound (row-slice version of `mask_rust`).
#[inline(never)]
pub fn mask_rows<BD: BitDepth>(
    dst_rows: &mut [&mut [BD::Pixel]],
    dst_x: usize,
    tmp1: &[i16],
    tmp2: &[i16],
    w: usize,
    h: usize,
    mask: &[u8],
    bd: BD,
) {
    let intermediate_bits = bd.get_intermediate_bits();
    let sh = intermediate_bits + 6;
    let rnd = (32 << intermediate_bits) + i32::from(BD::PREP_BIAS) * 64;
    for y in 0..h {
        let dst = &mut dst_rows[y][dst_x..dst_x + w];
        for x in 0..w {
            dst[x] = bd.iclip_pixel(
                (tmp1[y * w + x] as i32 * mask[y * w + x] as i32
                    + tmp2[y * w + x] as i32 * (64 - mask[y * w + x] as i32)
                    + rnd)
                    >> sh,
            );
        }
    }
}

// === Blend functions (read-modify-write on dst) ===

fn blend_px<BD: BitDepth>(a: BD::Pixel, b: BD::Pixel, m: u8) -> BD::Pixel {
    let m = m as u32;
    ((a.as_::<u32>() * (64 - m) + b.as_::<u32>() * m + 32) >> 6).as_::<BD::Pixel>()
}

/// OBMC blending (row-slice version of `blend_rust`).
#[inline(never)]
pub fn blend_rows<BD: BitDepth>(
    dst_rows: &mut [&mut [BD::Pixel]],
    dst_x: usize,
    tmp: &[BD::Pixel],
    w: usize,
    h: usize,
    mask: &[u8],
) {
    for y in 0..h {
        let dst = &mut dst_rows[y][dst_x..dst_x + w];
        for x in 0..w {
            dst[x] = blend_px::<BD>(dst[x], tmp[y * w + x], mask[y * w + x]);
        }
    }
}

/// Vertical OBMC blending (row-slice version of `blend_v_rust`).
#[inline(never)]
pub fn blend_v_rows<BD: BitDepth>(
    dst_rows: &mut [&mut [BD::Pixel]],
    dst_x: usize,
    tmp: &[BD::Pixel],
    w: usize,
    h: usize,
) {
    let dst_w = w * 3 >> 2;
    for y in 0..h {
        let dst = &mut dst_rows[y][dst_x..dst_x + dst_w];
        for x in 0..dst_w {
            dst[x] = blend_px::<BD>(dst[x], tmp[y * w + x], dav1d_obmc_masks[w + x]);
        }
    }
}

/// Horizontal OBMC blending (row-slice version of `blend_h_rust`).
#[inline(never)]
pub fn blend_h_rows<BD: BitDepth>(
    dst_rows: &mut [&mut [BD::Pixel]],
    dst_x: usize,
    tmp: &[BD::Pixel],
    w: usize,
    h: usize,
) {
    let dst_h = h * 3 >> 2;
    for y in 0..dst_h {
        let dst = &mut dst_rows[y][dst_x..dst_x + w];
        for x in 0..w {
            dst[x] = blend_px::<BD>(dst[x], tmp[y * w + x], dav1d_obmc_masks[h + y]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::include::common::bitdepth::BitDepth8;

    type BD = BitDepth8;

    #[test]
    fn test_put_rows_copies_pixels() {
        let src_data: Vec<u8> = (0..24).collect();
        let src_rows_owned: Vec<Vec<u8>> = src_data.chunks(6).map(|c| c.to_vec()).collect();
        let src_rows: Vec<&[u8]> = src_rows_owned.iter().map(|r| r.as_slice()).collect();

        let mut dst_buf = vec![0u8; 24];
        let mut dst_rows_v: Vec<&mut [u8]> = dst_buf.chunks_mut(6).collect();

        put_rows::<BD>(&mut dst_rows_v, 1, &src_rows, 2, 3);

        // Row 0: src[2..5] → dst[1..4]
        assert_eq!(dst_rows_v[0], &[0, 2, 3, 4, 0, 0]);
        // Row 1: src[8..11] → dst[7..10]
        assert_eq!(dst_rows_v[1], &[0, 8, 9, 10, 0, 0]);
    }

    #[test]
    fn test_avg_rows_basic() {
        let tmp1: Vec<i16> = vec![100, 200, 300, 400];
        let tmp2: Vec<i16> = vec![100, 200, 300, 400];

        let mut dst_buf = vec![0u8; 4];
        let mut dst_rows: Vec<&mut [u8]> = dst_buf.chunks_mut(2).collect();

        avg_rows::<BD>(&mut dst_rows, 0, &tmp1, &tmp2, 2, 2, BD::new(()));

        // avg of equal values should give the same value (after bias and rounding)
        for row in &dst_rows {
            for &px in row.iter() {
                assert!(px > 0, "pixel should be non-zero");
            }
        }
    }

    #[test]
    fn test_blend_px_midpoint() {
        // m=32 (midpoint): should average a and b
        let result = blend_px::<BD>(0u8, 128u8, 32);
        assert_eq!(result, 64); // (0*32 + 128*32 + 32) >> 6 = 64
    }

    #[test]
    fn test_blend_px_endpoints() {
        // m=0: keep a
        assert_eq!(blend_px::<BD>(100u8, 200u8, 0), 100);
        // m=64: keep b
        assert_eq!(blend_px::<BD>(100u8, 200u8, 64), 200);
    }
}
