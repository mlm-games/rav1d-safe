//! Per-tile plane row slices for DisjointMut-free reconstruction.
//!
//! `TilePlaneRows` provides per-row mutable pixel slices for one tile's
//! column range of one plane. Created from the raw frame buffer via
//! `split_at_mut` — no DisjointMut involved.
//!
//! This is the type that `call_rows` methods accept for pixel output.

#![forbid(unsafe_code)]

/// Per-row pixel slices for one tile's column range of one plane.
///
/// Created by splitting the frame buffer at tile column boundaries.
/// Each row covers `[tile_col_start, tile_col_end)` pixels.
pub struct TilePlaneRows<'a, Pixel> {
    /// One mutable slice per row, covering this tile's column range.
    pub rows: Vec<&'a mut [Pixel]>,
    /// Tile column start in frame pixel coordinates.
    pub col_start: usize,
    /// Tile column end in frame pixel coordinates.
    pub col_end: usize,
    /// First pixel row of this SB row in the frame.
    pub row_start: usize,
}

impl<'a, Pixel> TilePlaneRows<'a, Pixel> {
    /// Number of rows.
    pub fn height(&self) -> usize {
        self.rows.len()
    }

    /// Tile width in pixels.
    pub fn width(&self) -> usize {
        self.col_end - self.col_start
    }

    /// Get a mutable sub-slice for a block at (bx, by) in 4-pixel units,
    /// with block dimensions (bw, bh) in pixels.
    ///
    /// Returns `&mut [&mut [Pixel]]` covering the block's rows,
    /// where each inner slice starts at the block's column within the tile.
    ///
    /// `bx` is in frame-absolute 4-pixel units. It's converted to tile-relative.
    /// Get the tile-relative column offset for a block at absolute position bx (4-px units).
    pub fn tile_col(&self, bx_abs: usize) -> usize {
        bx_abs * 4 - self.col_start
    }

    /// Get the row index for a block at absolute position by (4-px units).
    pub fn tile_row(&self, by_abs: usize) -> usize {
        by_abs * 4 - self.row_start
    }

    /// Get mutable row slices for a block's rows.
    pub fn rows_for_block(&mut self, by_abs: usize, bh: usize) -> &mut [&'a mut [Pixel]] {
        let frame_row = by_abs * 4 - self.row_start;
        &mut self.rows[frame_row..frame_row + bh]
    }
}

/// All 3 planes for one tile within one SB row.
pub struct TilePixelRows<'a, Pixel> {
    pub y: TilePlaneRows<'a, Pixel>,
    pub u: Option<TilePlaneRows<'a, Pixel>>,
    pub v: Option<TilePlaneRows<'a, Pixel>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_rows_returns_correct_subslice() {
        let mut buf = vec![0u8; 32 * 4]; // 32-wide, 4 rows
        let mut rows: Vec<&mut [u8]> = buf.chunks_mut(32).collect();

        // Tile covers columns 8..24 (16 pixels wide)
        let tile_rows: Vec<&mut [u8]> = rows.iter_mut()
            .map(|r| &mut r[8..24])
            .collect();

        let mut tile = TilePlaneRows {
            rows: tile_rows,
            col_start: 8,
            col_end: 24,
            row_start: 0,
        };

        // Block at (3, 0) in 4-px units = pixel (12, 0)
        assert_eq!(tile.tile_col(3), 4); // 12 - 8 = 4
        assert_eq!(tile.tile_row(0), 0);

        let block = tile.rows_for_block(0, 2);
        assert_eq!(block.len(), 2);
        assert_eq!(block[0].len(), 16); // full tile width
    }
}
