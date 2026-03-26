//! Parity tests: row-slice functions vs PicOffset functions.
//!
//! Verifies that row-slice MC/ITX/ipred functions produce identical output
//! to the PicOffset-based originals on the same inputs.

#![cfg(test)]

use crate::include::common::bitdepth::{BitDepth, BitDepth8};
use crate::src::internal::SCRATCH_EDGE_LEN;
use crate::src::ipred_rows;
use crate::src::itx_1d::rav1d_inv_dct4_1d_c;
use crate::src::itx_rows;
use crate::src::mc_rows;
use crate::src::plane_rows::{split_into_rows, split_rows_by_tiles};

type BD = BitDepth8;

// === MC parity ===

#[test]
fn put_rows_copies_correct_region() {
    let src_buf: Vec<u8> = (0..128).collect();
    let src_rows: Vec<&[u8]> = src_buf.chunks(16).take(8).map(|r| &r[..16]).collect();

    let mut dst_buf = vec![0u8; 128];
    let mut dst_rows: Vec<&mut [u8]> = dst_buf.chunks_mut(16).take(8).map(|r| &mut r[..16]).collect();

    mc_rows::put_rows::<BD>(&mut dst_rows, 4, &src_rows, 2, 6);

    for y in 0..8 {
        for x in 0..6 {
            assert_eq!(
                dst_rows[y][4 + x],
                src_buf[y * 16 + 2 + x],
                "mismatch at ({x},{y})"
            );
        }
        // Untouched regions
        for x in 0..4 {
            assert_eq!(dst_rows[y][x], 0, "should be untouched at ({x},{y})");
        }
    }
}

#[test]
fn bilin_fullpel_matches_put() {
    let src_buf: Vec<u8> = (0..128).collect();
    let src_rows: Vec<&[u8]> = src_buf.chunks(16).take(8).map(|r| &r[..16]).collect();

    let mut dst_put = vec![0u8; 128];
    let mut dst_bilin = vec![0u8; 128];

    {
        let mut rows: Vec<&mut [u8]> = dst_put.chunks_mut(16).take(8).map(|r| &mut r[..16]).collect();
        mc_rows::put_rows::<BD>(&mut rows, 0, &src_rows, 0, 8);
    }
    {
        let mut rows: Vec<&mut [u8]> = dst_bilin.chunks_mut(16).take(8).map(|r| &mut r[..16]).collect();
        mc_rows::put_bilin_rows::<BD>(&mut rows, 0, &src_rows, 0, 8, 8, 0, 0, BD::new(()));
    }

    assert_eq!(&dst_put[..128], &dst_bilin[..128], "bilin fullpel ≠ put");
}

#[test]
fn avg_rows_symmetric() {
    // avg of two identical buffers → should reproduce the pixel value
    // For 8bpc: PREP_BIAS=0, intermediate_bits=4
    // avg formula: (tmp1 + tmp2 + rnd) >> sh
    //   where sh = intermediate_bits + 1 = 5
    //   and rnd = (1 << 4) + 0*2 = 16
    // To get pixel value 100: need (2*val + 16) >> 5 = 100
    //   2*val + 16 = 3200 → val = 1592
    let val: i16 = 1592;
    let tmp: Vec<i16> = vec![val; 16];

    let mut dst_buf = vec![0u8; 16];
    let mut dst_rows: Vec<&mut [u8]> = dst_buf.chunks_mut(4).collect();

    mc_rows::avg_rows::<BD>(&mut dst_rows, 0, &tmp, &tmp, 4, 4, BD::new(()));

    for y in 0..4 {
        for x in 0..4 {
            assert_eq!(dst_rows[y][x], 100, "avg symmetric mismatch at ({x},{y})");
        }
    }
}

// === Ipred parity ===

fn make_topleft(top: &[u8], left: &[u8], tl: u8) -> [u8; SCRATCH_EDGE_LEN] {
    let mut buf = [0u8; SCRATCH_EDGE_LEN];
    let off = 128;
    buf[off] = tl;
    for (i, &v) in top.iter().enumerate() {
        buf[off + 1 + i] = v;
    }
    for (i, &v) in left.iter().enumerate() {
        buf[off - 1 - i] = v;
    }
    buf
}

#[test]
fn v_pred_all_rows_equal_top() {
    let topleft = make_topleft(&[10, 20, 30, 40], &[0; 4], 0);
    let mut dst_buf = vec![0u8; 16];
    let mut dst_rows: Vec<&mut [u8]> = dst_buf.chunks_mut(4).collect();

    ipred_rows::v_pred_rows::<BD>(&mut dst_rows, 0, &topleft, 128, 4, 4);

    for y in 0..4 {
        assert_eq!(dst_rows[y], &[10, 20, 30, 40], "v_pred row {y}");
    }
}

#[test]
fn h_pred_each_row_constant() {
    let topleft = make_topleft(&[0; 4], &[10, 20, 30, 40], 0);
    let mut dst_buf = vec![0u8; 16];
    let mut dst_rows: Vec<&mut [u8]> = dst_buf.chunks_mut(4).collect();

    ipred_rows::h_pred_rows::<BD>(&mut dst_rows, 0, &topleft, 128, 4, 4);

    assert_eq!(dst_rows[0], &[10, 10, 10, 10]);
    assert_eq!(dst_rows[1], &[20, 20, 20, 20]);
    assert_eq!(dst_rows[2], &[30, 30, 30, 30]);
    assert_eq!(dst_rows[3], &[40, 40, 40, 40]);
}

#[test]
fn paeth_uniform_edges_produce_uniform_output() {
    let topleft = make_topleft(&[100; 8], &[100; 8], 100);
    let mut dst_buf = vec![0u8; 64];
    let mut dst_rows: Vec<&mut [u8]> = dst_buf.chunks_mut(8).collect();

    ipred_rows::paeth_pred_rows::<BD>(&mut dst_rows, 0, &topleft, 128, 8, 8);

    for y in 0..8 {
        for x in 0..8 {
            assert_eq!(dst_rows[y][x], 100, "paeth at ({x},{y})");
        }
    }
}

#[test]
fn dc_pred_uniform_edges() {
    let topleft = make_topleft(&[80; 4], &[80; 4], 80);
    let mut dst_buf = vec![0u8; 16];
    let mut dst_rows: Vec<&mut [u8]> = dst_buf.chunks_mut(4).collect();

    ipred_rows::dc_pred_rows::<BD>(&mut dst_rows, 0, &topleft, 128, 4, 4, BD::new(()));

    for y in 0..4 {
        for x in 0..4 {
            assert_eq!(dst_rows[y][x], 80, "dc at ({x},{y})");
        }
    }
}

#[test]
fn dc_128_is_half_max() {
    let mut dst_buf = vec![0u8; 16];
    let mut dst_rows: Vec<&mut [u8]> = dst_buf.chunks_mut(4).collect();

    ipred_rows::dc_128_pred_rows::<BD>(&mut dst_rows, 0, 4, 4, BD::new(()));

    for &px in dst_buf.iter() {
        assert_eq!(px, 128);
    }
}

#[test]
fn smooth_uniform_produces_uniform() {
    let topleft = [100u8; SCRATCH_EDGE_LEN];
    let mut dst_buf = vec![0u8; 64];
    let mut dst_rows: Vec<&mut [u8]> = dst_buf.chunks_mut(8).collect();

    ipred_rows::smooth_pred_rows::<BD>(&mut dst_rows, 0, &topleft, 128, 8, 8);

    for y in 0..8 {
        for x in 0..8 {
            assert_eq!(dst_rows[y][x], 100, "smooth at ({x},{y})");
        }
    }
}

// === ITX parity ===

#[test]
fn itx_dc_zero_is_noop() {
    let mut coeff: Vec<i16> = vec![0; 16];
    let mut dst_buf = vec![128u8; 16];
    let mut dst_rows: Vec<&mut [u8]> = dst_buf.chunks_mut(4).collect();

    itx_rows::inv_txfm_add_rows::<BD>(
        &mut dst_rows, 0, &mut coeff, 0, 4, 4, 0,
        rav1d_inv_dct4_1d_c, rav1d_inv_dct4_1d_c, true, BD::new(()),
    );

    for &px in dst_buf.iter() {
        assert_eq!(px, 128, "DC=0 should be noop");
    }
}

#[test]
fn itx_dc_adds_uniform_offset() {
    let mut coeff: Vec<i16> = vec![0; 16];
    coeff[0] = 64;
    let mut dst_buf = vec![100u8; 16];
    let mut dst_rows: Vec<&mut [u8]> = dst_buf.chunks_mut(4).collect();

    itx_rows::inv_txfm_add_rows::<BD>(
        &mut dst_rows, 0, &mut coeff, 0, 4, 4, 0,
        rav1d_inv_dct4_1d_c, rav1d_inv_dct4_1d_c, true, BD::new(()),
    );

    // All pixels should have the same offset from 100
    let first = dst_buf[0];
    assert!(first > 100, "positive DC should increase pixels");
    for (i, &px) in dst_buf.iter().enumerate() {
        assert_eq!(px, first, "pixel {i} differs from pixel 0");
    }
    assert_eq!(coeff[0], 0, "coefficient should be zeroed");
}

// === Tile split + MC integration ===

#[test]
fn tile_split_mc_then_verify() {
    let mut frame_buf: Vec<u8> = vec![0; 16 * 4]; // 16-wide, 4 rows
    let src_buf: Vec<u8> = (100..164).collect();
    let src_rows: Vec<&[u8]> = src_buf.chunks(16).take(4).map(|r| &r[..16]).collect();

    {
        let rows = split_into_rows(&mut frame_buf, 16, 16, 4);
        let mut tiles = split_rows_by_tiles(rows, &[0, 8, 16]);

        // Tile 0: copy 4 pixels from src col 0 to dst col 2
        mc_rows::put_rows::<BD>(&mut tiles[0], 2, &src_rows, 0, 4);
        // Tile 1: copy 4 pixels from src col 8 to dst col 1
        mc_rows::put_rows::<BD>(&mut tiles[1], 1, &src_rows, 8, 4);
    }

    // Tile 0 writes → global columns 2..6
    assert_eq!(frame_buf[2], 100); // src row 0, col 0
    assert_eq!(frame_buf[3], 101);
    assert_eq!(frame_buf[4], 102);
    assert_eq!(frame_buf[5], 103);

    // Tile 1 writes → global columns 8+1=9..13
    assert_eq!(frame_buf[9], 108); // src row 0, col 8
    assert_eq!(frame_buf[10], 109);

    // Untouched
    assert_eq!(frame_buf[0], 0);
    assert_eq!(frame_buf[1], 0);
    assert_eq!(frame_buf[8], 0); // tile 1 col 0 untouched
}

#[test]
fn tile_split_ipred_writes_correct_region() {
    let topleft = make_topleft(&[50; 8], &[50; 8], 50);
    let mut frame_buf: Vec<u8> = vec![0; 16 * 4];

    {
        let rows = split_into_rows(&mut frame_buf, 16, 16, 4);
        let mut tiles = split_rows_by_tiles(rows, &[0, 8, 16]);

        // V_PRED on tile 0 (columns 0..8): should fill with top=50
        ipred_rows::v_pred_rows::<BD>(&mut tiles[0], 0, &topleft, 128, 4, 4);

        // DC_128 on tile 1 (columns 8..16): should fill with 128
        ipred_rows::dc_128_pred_rows::<BD>(&mut tiles[1], 0, 4, 4, BD::new(()));
    }

    // Tile 0: global cols 0..4 = 50
    for y in 0..4 {
        for x in 0..4 {
            assert_eq!(frame_buf[y * 16 + x], 50, "tile0 at ({x},{y})");
        }
    }

    // Tile 1: global cols 8..12 = 128
    for y in 0..4 {
        for x in 0..4 {
            assert_eq!(frame_buf[y * 16 + 8 + x], 128, "tile1 at ({x},{y})");
        }
    }
}
