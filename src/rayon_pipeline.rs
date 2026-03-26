//! Rayon-based SB-row pipeline for parallel decode.
//!
//! This module provides the core rayon scope architecture for pipelining
//! SB-row reconstruction and filtering. Each SB row is processed as one
//! rayon task: first tile-parallel reconstruction, then sequential filtering.
//!
//! The pipeline dependency graph:
//!   Recon(N) signals → Recon(N+1) starts (ipred edge dependency)
//!   Recon(N) signals + Filter(N-1) signals → Filter(N) starts
//!
//! Parallelism comes from different SB rows running concurrently:
//!   Recon(N+1) runs in parallel with Filter(N) on disjoint row ranges.

#![forbid(unsafe_code)]

use crate::src::plane_rows::{split_into_rows, split_rows_by_tiles};
use crate::src::progressive_frame::ProgressiveFrame;

/// Signals used for SB-row pipeline synchronization.
/// Each SB row's task sends `Done` when its phase completes.
#[derive(Debug, Clone, Copy)]
pub enum PipelineSignal {
    Done,
}

/// Configuration for a single frame's rayon decode pipeline.
pub struct PipelineConfig {
    /// Frame dimensions
    pub width: usize,
    pub height: usize,
    /// Pixel stride (in pixels, including padding)
    pub stride: usize,
    /// Superblock height (64 or 128)
    pub sb_height: usize,
    /// Number of SB rows
    pub num_sb_rows: usize,
    /// Tile column boundaries in pixels: [0, col1, col2, ..., width]
    pub tile_col_boundaries: Vec<usize>,
}

impl PipelineConfig {
    pub fn new(
        width: usize,
        height: usize,
        stride: usize,
        sb_height: usize,
        tile_col_boundaries: Vec<usize>,
    ) -> Self {
        let num_sb_rows = (height + sb_height - 1) / sb_height;
        Self {
            width,
            height,
            stride,
            sb_height,
            num_sb_rows,
            tile_col_boundaries,
        }
    }

    /// Pixel row range for SB row `sby`.
    pub fn sb_row_range(&self, sby: usize) -> (usize, usize) {
        let start = sby * self.sb_height;
        let end = (start + self.sb_height).min(self.height);
        (start, end)
    }

    /// Number of tile columns.
    pub fn num_tiles(&self) -> usize {
        self.tile_col_boundaries.len() - 1
    }
}

/// Run the SB-row pipeline on a frame buffer.
///
/// `recon_fn`: called per tile per SB row with (tile_strip, tile_idx, sby).
/// `filter_fn`: called per SB row with (full_rows, sby).
///
/// The pipeline ensures:
/// - Recon(N+1) doesn't start until Recon(N) completes
/// - Filter(N) doesn't start until both Recon(N) and Filter(N-1) complete
/// - Recon(N+1) and Filter(N) can run in parallel (disjoint row ranges)
pub fn run_pipeline<P, RF, FF>(
    frame_buf: &mut [P],
    config: &PipelineConfig,
    recon_fn: RF,
    filter_fn: FF,
) where
    P: Send,
    RF: Fn(&mut [&mut [P]], usize, usize) + Send + Sync, // (tile_strip, tile_idx, sby)
    FF: Fn(&mut [&mut [P]], usize) + Send + Sync,         // (full_rows, sby)
{
    // Process SB rows sequentially. Within each SB row, tile reconstruction
    // runs in parallel via rayon::scope. The sequential outer loop ensures
    // clean ownership: each SB row borrows its region exclusively.
    //
    // For SB-row pipelining (recon(N+1) || filter(N)), we'd need the
    // ProgressiveFrame approach (see RAYON-THREADING-SPEC.md). This initial
    // implementation validates the tile-parallel reconstruction path.

    let buf_len = frame_buf.len();
    for sby in 0..config.num_sb_rows {
        let (row_start, row_end) = config.sb_row_range(sby);
        let nrows = row_end - row_start;
        let buf_start = row_start * config.stride;
        let buf_end = buf_start + nrows * config.stride;
        let sby_buf = &mut frame_buf[buf_start..buf_end.min(buf_len)];

        // === RECONSTRUCTION: tile-parallel ===
        {
            let rows = split_into_rows(sby_buf, config.stride, config.width, nrows);
            let mut tiles = split_rows_by_tiles(rows, &config.tile_col_boundaries);

            if tiles.len() > 1 {
                // True tile parallelism: drain tiles Vec to get owned per-tile
                // row Vecs, move each into a rayon closure.
                let tile_vec: Vec<(usize, Vec<&mut [P]>)> = tiles
                    .drain(..)
                    .enumerate()
                    .collect();

                rayon::scope(|s| {
                    for (tile_idx, mut strip) in tile_vec {
                        let recon = &recon_fn;
                        s.spawn(move |_| {
                            recon(&mut strip, tile_idx, sby);
                        });
                    }
                });
            } else {
                recon_fn(&mut tiles[0], 0, sby);
            }
        }

        // === FILTERING: full-width, sequential ===
        {
            let mut rows = split_into_rows(sby_buf, config.stride, config.width, nrows);
            filter_fn(&mut rows, sby);
        }
    }
}

/// Run the SB-row pipeline on a ProgressiveFrame, freezing rows after each
/// SB row's filter phase completes.
///
/// This is the same as `run_pipeline` but with progressive freezing:
/// after filter(sby) completes, rows [0, sby_end) are frozen and available
/// for reference reads by other frames.
///
/// `ref_fn`: optional callback to read frozen rows from reference frames.
///           Called after each SB row's recon with the current sby.
pub fn run_pipeline_progressive<P, RF, FF>(
    frame: &mut ProgressiveFrame<P>,
    config: &PipelineConfig,
    recon_fn: RF,
    filter_fn: FF,
) where
    P: Copy + Default + Send,
    RF: Fn(&mut [&mut [P]], usize, usize) + Send + Sync,
    FF: Fn(&mut [&mut [P]], usize) + Send + Sync,
{
    for sby in 0..config.num_sb_rows {
        let (row_start, row_end) = config.sb_row_range(sby);
        let nrows = row_end - row_start;

        // Get mutable rows for this SB row
        let mut rows = frame.active_rows_mut(row_start, row_end);

        // === RECONSTRUCTION: tile-parallel ===
        {
            // Convert Vec<&mut [P]> to a form split_rows_by_tiles can consume
            let row_refs: Vec<&mut [P]> = rows.drain(..).collect();
            let mut tiles = split_rows_by_tiles(row_refs, &config.tile_col_boundaries);

            if tiles.len() > 1 {
                let tile_vec: Vec<(usize, Vec<&mut [P]>)> =
                    tiles.drain(..).enumerate().collect();

                rayon::scope(|s| {
                    for (tile_idx, mut strip) in tile_vec {
                        let recon = &recon_fn;
                        s.spawn(move |_| {
                            recon(&mut strip, tile_idx, sby);
                        });
                    }
                });
            } else {
                recon_fn(&mut tiles[0], 0, sby);
            }
        }

        // Re-acquire rows for filtering (recon consumed them via drain)
        let mut rows = frame.active_rows_mut(row_start, row_end);

        // === FILTERING: full-width, sequential ===
        filter_fn(&mut rows, sby);

        // Freeze this SB row's rows — now available for reference reads
        frame.freeze_through(row_end);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pipeline_processes_all_sb_rows() {
        // 16 pixels tall, stride 8, sb_height 8 → 2 SB rows
        let mut buf = vec![0u8; 8 * 16];
        let config = PipelineConfig::new(8, 16, 8, 8, vec![0, 8]);

        let recon_called = std::sync::atomic::AtomicUsize::new(0);
        let filter_called = std::sync::atomic::AtomicUsize::new(0);

        run_pipeline(
            &mut buf,
            &config,
            |rows, tile_idx, sby| {
                recon_called.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                assert_eq!(tile_idx, 0);
                // Fill with sby+1 to verify correct SB row
                for row in rows.iter_mut() {
                    row.fill((sby + 1) as u8);
                }
            },
            |rows, sby| {
                filter_called.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                // Verify recon ran first
                for row in rows.iter() {
                    assert_eq!(row[0], (sby + 1) as u8, "filter should see recon output");
                }
                // Double the values
                for row in rows.iter_mut() {
                    for px in row.iter_mut() {
                        *px *= 2;
                    }
                }
            },
        );

        assert_eq!(recon_called.load(std::sync::atomic::Ordering::SeqCst), 2);
        assert_eq!(filter_called.load(std::sync::atomic::Ordering::SeqCst), 2);

        // SB row 0: recon wrote 1, filter doubled to 2
        assert_eq!(buf[0], 2);
        // SB row 1: recon wrote 2, filter doubled to 4
        assert_eq!(buf[8 * 8], 4);
    }

    #[test]
    fn pipeline_tile_parallel_writes() {
        // 16 pixels wide, 8 tall, 2 tiles of 8 columns each
        let mut buf = vec![0u8; 16 * 8];
        let config = PipelineConfig::new(16, 8, 16, 8, vec![0, 8, 16]);

        run_pipeline(
            &mut buf,
            &config,
            |rows, tile_idx, _sby| {
                // Each tile writes its tile_idx+1 to all pixels
                let val = (tile_idx + 1) as u8;
                for row in rows.iter_mut() {
                    row.fill(val);
                }
            },
            |_rows, _sby| {
                // No-op filter
            },
        );

        // Tile 0 (columns 0..8) = 1, Tile 1 (columns 8..16) = 2
        assert_eq!(buf[0], 1);
        assert_eq!(buf[7], 1);
        assert_eq!(buf[8], 2);
        assert_eq!(buf[15], 2);
    }

    #[test]
    fn pipeline_ordering_guaranteed() {
        // Verify that recon runs before filter for each SB row
        let mut buf = vec![0u8; 8 * 16];
        let config = PipelineConfig::new(8, 16, 8, 8, vec![0, 8]);
        let order = std::sync::Mutex::new(Vec::new());

        run_pipeline(
            &mut buf,
            &config,
            |_rows, _tile_idx, sby| {
                order.lock().unwrap().push(format!("recon-{sby}"));
            },
            |_rows, sby| {
                order.lock().unwrap().push(format!("filter-{sby}"));
            },
        );

        let order = order.into_inner().unwrap();
        assert_eq!(order, vec!["recon-0", "filter-0", "recon-1", "filter-1"]);
    }

    #[test]
    fn pipeline_concurrent_tile_writes_no_corruption() {
        // Stress test: 4 tiles writing to a large buffer concurrently.
        // Each tile fills its columns with a unique value.
        // After pipeline, verify no cross-tile corruption.
        let width = 256;
        let height = 128;
        let stride = width;
        let sb_height = 64;
        let tile_w = 64;
        let boundaries: Vec<usize> = (0..=4).map(|i| i * tile_w).collect();

        let mut buf = vec![0u8; stride * height];
        let config = PipelineConfig::new(width, height, stride, sb_height, boundaries);

        run_pipeline(
            &mut buf,
            &config,
            |rows, tile_idx, sby| {
                // Each tile writes (tile_idx * 16 + sby) to all its pixels
                let val = (tile_idx * 16 + sby) as u8;
                for row in rows.iter_mut() {
                    row.fill(val);
                }
            },
            |_rows, _sby| {},
        );

        // Verify each tile's region has the correct value
        for sby in 0..2 {
            let row_start = sby * sb_height;
            for tile_idx in 0..4 {
                let expected = (tile_idx * 16 + sby) as u8;
                let col_start = tile_idx * tile_w;
                for y in row_start..row_start + sb_height {
                    for x in col_start..col_start + tile_w {
                        assert_eq!(
                            buf[y * stride + x],
                            expected,
                            "corruption at ({x},{y}): tile={tile_idx} sby={sby}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn pipeline_handles_partial_last_sb_row() {
        // Height 20, sb_height 8 → 3 SB rows: 8, 8, 4
        let mut buf = vec![0u8; 8 * 20];
        let config = PipelineConfig::new(8, 20, 8, 8, vec![0, 8]);
        let sb_heights = std::sync::Mutex::new(Vec::new());

        run_pipeline(
            &mut buf,
            &config,
            |rows, _tile_idx, _sby| {
                sb_heights.lock().unwrap().push(rows.len());
            },
            |_rows, _sby| {},
        );

        assert_eq!(
            sb_heights.into_inner().unwrap(),
            vec![8, 8, 4],
            "last SB row should be partial"
        );
    }

    #[test]
    fn progressive_pipeline_freezes_rows() {
        use crate::src::progressive_frame::ProgressiveFrame;

        let mut frame = ProgressiveFrame::<u8>::new(8, 8, 16);
        let config = PipelineConfig::new(8, 16, 8, 8, vec![0, 8]);

        run_pipeline_progressive(
            &mut frame,
            &config,
            |rows, _tile_idx, sby| {
                for row in rows.iter_mut() {
                    row.fill((sby + 1) as u8);
                }
            },
            |rows, sby| {
                // Filter: double the values
                for row in rows.iter_mut() {
                    for px in row.iter_mut() {
                        *px *= 2;
                    }
                }
            },
        );

        // Frame should be fully frozen
        assert!(frame.is_fully_frozen());

        // SB row 0: recon=1, filter doubles to 2
        assert_eq!(frame.frozen_row(0)[0], 2);
        assert_eq!(frame.frozen_row(7)[0], 2);

        // SB row 1: recon=2, filter doubles to 4
        assert_eq!(frame.frozen_row(8)[0], 4);
        assert_eq!(frame.frozen_row(15)[0], 4);
    }

    #[test]
    fn progressive_pipeline_tile_parallel_then_freeze() {
        use crate::src::progressive_frame::ProgressiveFrame;

        let mut frame = ProgressiveFrame::<u8>::new(16, 16, 8);
        let config = PipelineConfig::new(16, 8, 16, 8, vec![0, 8, 16]);

        run_pipeline_progressive(
            &mut frame,
            &config,
            |rows, tile_idx, _sby| {
                let val = (tile_idx + 1) as u8 * 10;
                for row in rows.iter_mut() {
                    row.fill(val);
                }
            },
            |_rows, _sby| {},
        );

        assert!(frame.is_fully_frozen());

        // Tile 0 (cols 0..8) = 10, Tile 1 (cols 8..16) = 20
        let row = frame.frozen_row(0);
        assert_eq!(row[0], 10);
        assert_eq!(row[7], 10);
        assert_eq!(row[8], 20);
        assert_eq!(row[15], 20);
    }
}
