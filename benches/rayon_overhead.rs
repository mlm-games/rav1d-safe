//! Micro-benchmark: rayon pipeline overhead.
//!
//! Measures the cost of the split/drain/spawn/join cycle for tile-parallel
//! reconstruction, compared to sequential processing of the same work.
//!
//! Run: cargo bench --bench rayon_overhead

use divan::Bencher;
use rav1d_safe::src::plane_rows::{split_into_rows, split_rows_by_tiles};

fn main() {
    divan::main();
}

// Simulated "work" per tile: fill each row with a value
fn tile_work(rows: &mut [&mut [u8]], tile_idx: usize, _sby: usize) {
    let val = (tile_idx as u8).wrapping_add(1);
    for row in rows.iter_mut() {
        for px in row.iter_mut() {
            *px = val;
        }
    }
}

// === 4K frame (3840×2160), 4 tiles, SB64 ===

const W_4K: usize = 3840;
const H_4K: usize = 2160;
const SB: usize = 64;
const N_TILES: usize = 4;

#[divan::bench]
fn sequential_4k_4tiles(bencher: Bencher) {
    let tile_w = W_4K / N_TILES;
    let boundaries: Vec<usize> = (0..=N_TILES).map(|i| i * tile_w).collect();

    bencher
        .with_inputs(|| vec![0u8; W_4K * SB])
        .bench_local_values(|mut buf| {
            let rows = split_into_rows(&mut buf, W_4K, W_4K, SB);
            let mut tiles = split_rows_by_tiles(rows, &boundaries);
            for (tile_idx, strip) in tiles.iter_mut().enumerate() {
                tile_work(strip, tile_idx, 0);
            }
        });
}

#[divan::bench]
fn rayon_4k_4tiles(bencher: Bencher) {
    let tile_w = W_4K / N_TILES;
    let boundaries: Vec<usize> = (0..=N_TILES).map(|i| i * tile_w).collect();

    bencher
        .with_inputs(|| vec![0u8; W_4K * SB])
        .bench_local_values(|mut buf| {
            let rows = split_into_rows(&mut buf, W_4K, W_4K, SB);
            let mut tiles = split_rows_by_tiles(rows, &boundaries);
            let tile_vec: Vec<(usize, Vec<&mut [u8]>)> =
                tiles.drain(..).enumerate().collect();
            rayon::scope(|s| {
                for (tile_idx, mut strip) in tile_vec {
                    s.spawn(move |_| {
                        tile_work(&mut strip, tile_idx, 0);
                    });
                }
            });
        });
}

// === Split overhead only (no work) ===

#[divan::bench]
fn split_only_4k(bencher: Bencher) {
    let tile_w = W_4K / N_TILES;
    let boundaries: Vec<usize> = (0..=N_TILES).map(|i| i * tile_w).collect();

    bencher
        .with_inputs(|| vec![0u8; W_4K * SB])
        .bench_local_values(|mut buf| {
            let rows = split_into_rows(&mut buf, W_4K, W_4K, SB);
            let _tiles = split_rows_by_tiles(rows, &boundaries);
        });
}

#[divan::bench]
fn split_drain_4k(bencher: Bencher) {
    let tile_w = W_4K / N_TILES;
    let boundaries: Vec<usize> = (0..=N_TILES).map(|i| i * tile_w).collect();

    bencher
        .with_inputs(|| vec![0u8; W_4K * SB])
        .bench_local_values(|mut buf| {
            let rows = split_into_rows(&mut buf, W_4K, W_4K, SB);
            let mut tiles = split_rows_by_tiles(rows, &boundaries);
            let _tile_vec: Vec<(usize, Vec<&mut [u8]>)> =
                tiles.drain(..).enumerate().collect();
        });
}

// === 1-tile (no splitting needed) ===

#[divan::bench]
fn sequential_4k_1tile(bencher: Bencher) {
    bencher
        .with_inputs(|| vec![0u8; W_4K * SB])
        .bench_local_values(|mut buf| {
            let mut rows = split_into_rows(&mut buf, W_4K, W_4K, SB);
            tile_work(&mut rows, 0, 0);
        });
}

// === Batched: 4 SB rows per rayon scope (amortize spawn cost) ===

#[divan::bench]
fn pipeline_rayon_batched4_4k(bencher: Bencher) {
    let num_sb = (H_4K + SB - 1) / SB;
    let tile_w = W_4K / N_TILES;
    let boundaries: Vec<usize> = (0..=N_TILES).map(|i| i * tile_w).collect();
    let sb_batch = 4;

    bencher
        .with_inputs(|| vec![0u8; W_4K * H_4K])
        .bench_local_values(|mut buf| {
            let buf_len = buf.len();
            let num_batches = (num_sb + sb_batch - 1) / sb_batch;

            for batch in 0..num_batches {
                let row_start = batch * sb_batch * SB;
                let row_end = ((batch + 1) * sb_batch * SB).min(H_4K);
                let nrows = row_end - row_start;
                let start = row_start * W_4K;
                let end = (start + nrows * W_4K).min(buf_len);
                let batch_buf = &mut buf[start..end];

                let rows = split_into_rows(batch_buf, W_4K, W_4K, nrows);
                let mut tiles = split_rows_by_tiles(rows, &boundaries);
                let tile_vec: Vec<(usize, Vec<&mut [u8]>)> =
                    tiles.drain(..).enumerate().collect();
                rayon::scope(|s| {
                    for (tile_idx, mut strip) in tile_vec {
                        s.spawn(move |_| {
                            tile_work(&mut strip, tile_idx, 0);
                        });
                    }
                });
            }
        });
}

// === Full pipeline: 34 SB rows × 4 tiles ===

#[divan::bench]
fn pipeline_sequential_4k(bencher: Bencher) {
    let num_sb = (H_4K + SB - 1) / SB;
    let tile_w = W_4K / N_TILES;
    let boundaries: Vec<usize> = (0..=N_TILES).map(|i| i * tile_w).collect();

    bencher
        .with_inputs(|| vec![0u8; W_4K * H_4K])
        .bench_local_values(|mut buf| {
            let buf_len = buf.len();
            for sby in 0..num_sb {
                let row_start = sby * SB;
                let row_end = (row_start + SB).min(H_4K);
                let nrows = row_end - row_start;
                let start = row_start * W_4K;
                let end = (start + nrows * W_4K).min(buf_len);
                let sby_buf = &mut buf[start..end];

                let rows = split_into_rows(sby_buf, W_4K, W_4K, nrows);
                let mut tiles = split_rows_by_tiles(rows, &boundaries);
                for (tile_idx, strip) in tiles.iter_mut().enumerate() {
                    tile_work(strip, tile_idx, sby);
                }
            }
        });
}

#[divan::bench]
fn pipeline_rayon_4k(bencher: Bencher) {
    let num_sb = (H_4K + SB - 1) / SB;
    let tile_w = W_4K / N_TILES;
    let boundaries: Vec<usize> = (0..=N_TILES).map(|i| i * tile_w).collect();

    bencher
        .with_inputs(|| vec![0u8; W_4K * H_4K])
        .bench_local_values(|mut buf| {
            let buf_len = buf.len();
            for sby in 0..num_sb {
                let row_start = sby * SB;
                let row_end = (row_start + SB).min(H_4K);
                let nrows = row_end - row_start;
                let start = row_start * W_4K;
                let end = (start + nrows * W_4K).min(buf_len);
                let sby_buf = &mut buf[start..end];

                let rows = split_into_rows(sby_buf, W_4K, W_4K, nrows);
                let mut tiles = split_rows_by_tiles(rows, &boundaries);
                let tile_vec: Vec<(usize, Vec<&mut [u8]>)> =
                    tiles.drain(..).enumerate().collect();
                rayon::scope(|s| {
                    for (tile_idx, mut strip) in tile_vec {
                        s.spawn(move |_| {
                            tile_work(&mut strip, tile_idx, sby);
                        });
                    }
                });
            }
        });
}
