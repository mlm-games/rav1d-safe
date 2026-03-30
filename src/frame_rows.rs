//! Frame row-slice management for safe parallel decode.
//!
//! `FramePlaneRows` provides pre-split row slices for all 3 planes (Y, U, V)
//! of a frame buffer. This replaces `PicOffset` + `DisjointMut` for pixel access
//! during reconstruction.
//!
//! The row slices are created from the flat frame buffer before SB-row processing
//! and split by tile columns for parallel reconstruction.

#![forbid(unsafe_code)]

use crate::src::plane_rows::{split_into_rows, split_rows_by_tiles};

/// Pre-split row slices for one plane of a frame.
pub struct PlaneRowsMut<'a, P> {
    rows: Vec<&'a mut [P]>,
    width: usize,
}

impl<'a, P> PlaneRowsMut<'a, P> {
    /// Split a flat pixel buffer into per-row mutable slices.
    pub fn from_buf(buf: &'a mut [P], stride: usize, width: usize, height: usize) -> Self {
        let rows = split_into_rows(buf, stride, width, height);
        Self { rows, width }
    }

    /// Number of rows.
    pub fn height(&self) -> usize {
        self.rows.len()
    }

    /// Width in pixels.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Get a mutable reference to a single row.
    pub fn row_mut(&mut self, y: usize) -> &mut [P] {
        self.rows[y]
    }

    /// Get an immutable reference to a single row.
    pub fn row(&self, y: usize) -> &[P] {
        self.rows[y]
    }

    /// Get mutable row slices for a range of rows.
    pub fn rows_mut(&mut self, start: usize, end: usize) -> &mut [&'a mut [P]] {
        &mut self.rows[start..end]
    }

    /// Get immutable row slices for a range of rows.
    pub fn rows_ref(&self, start: usize, end: usize) -> Vec<&[P]> {
        self.rows[start..end].iter().map(|r| &**r).collect()
    }

    /// Read a single pixel.
    pub fn pixel(&self, x: usize, y: usize) -> &P {
        &self.rows[y][x]
    }

    /// Write a single pixel.
    pub fn set_pixel(&mut self, x: usize, y: usize, val: P) {
        self.rows[y][x] = val;
    }

    /// Consume and split by tile column boundaries.
    /// Returns one `TilePlaneRows` per tile.
    pub fn split_tiles(self, boundaries: &[usize]) -> Vec<TilePlaneRows<'a, P>> {
        let tile_row_vecs = split_rows_by_tiles(self.rows, boundaries);
        tile_row_vecs
            .into_iter()
            .enumerate()
            .map(|(i, rows)| {
                let tile_width = boundaries[i + 1] - boundaries[i];
                TilePlaneRows {
                    rows,
                    width: tile_width,
                    col_offset: boundaries[i],
                }
            })
            .collect()
    }
}

/// Row slices for one tile's column range of one plane.
pub struct TilePlaneRows<'a, P> {
    pub rows: Vec<&'a mut [P]>,
    pub width: usize,
    /// Column offset of this tile within the full frame.
    pub col_offset: usize,
}

impl<'a, P> TilePlaneRows<'a, P> {
    /// Number of rows.
    pub fn height(&self) -> usize {
        self.rows.len()
    }

    /// Tile width in pixels.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Column offset of this tile within the full frame.
    pub fn col_offset(&self) -> usize {
        self.col_offset
    }

    /// Get a mutable reference to a row within this tile.
    /// Column indices are relative to the tile (0..tile_width).
    pub fn row_mut(&mut self, y: usize) -> &mut [P] {
        self.rows[y]
    }

    /// Get an immutable reference to a row within this tile.
    pub fn row(&self, y: usize) -> &[P] {
        self.rows[y]
    }

    /// Get multiple mutable rows for DSP function calls.
    pub fn rows_mut(&mut self, start: usize, end: usize) -> &mut [&'a mut [P]] {
        &mut self.rows[start..end]
    }

    /// Read a single pixel (tile-relative coordinates).
    pub fn pixel(&self, x: usize, y: usize) -> &P {
        &self.rows[y][x]
    }

    /// Write a single pixel (tile-relative coordinates).
    pub fn set_pixel(&mut self, x: usize, y: usize, val: P) {
        self.rows[y][x] = val;
    }
}

/// All 3 planes of a frame, pre-split into row slices.
pub struct FramePlanesMut<'a, P> {
    pub y: PlaneRowsMut<'a, P>,
    pub u: PlaneRowsMut<'a, P>,
    pub v: PlaneRowsMut<'a, P>,
}

/// All 3 planes for one tile's column range.
pub struct TileFramePlanes<'a, P> {
    pub y: TilePlaneRows<'a, P>,
    pub u: TilePlaneRows<'a, P>,
    pub v: TilePlaneRows<'a, P>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plane_rows_basic() {
        let mut buf: Vec<u8> = (0..32).collect();
        let plane = PlaneRowsMut::from_buf(&mut buf, 8, 6, 4);
        assert_eq!(plane.height(), 4);
        assert_eq!(plane.width(), 6);
        assert_eq!(plane.row(0), &[0, 1, 2, 3, 4, 5]);
        assert_eq!(plane.row(1), &[8, 9, 10, 11, 12, 13]);
    }

    #[test]
    fn plane_rows_split_tiles() {
        let mut buf: Vec<u8> = (0..32).collect();
        let plane = PlaneRowsMut::from_buf(&mut buf, 8, 8, 4);
        let tiles = plane.split_tiles(&[0, 4, 8]);

        assert_eq!(tiles.len(), 2);
        assert_eq!(tiles[0].width(), 4);
        assert_eq!(tiles[0].col_offset(), 0);
        assert_eq!(tiles[0].row(0), &[0, 1, 2, 3]);

        assert_eq!(tiles[1].width(), 4);
        assert_eq!(tiles[1].col_offset(), 4);
        assert_eq!(tiles[1].row(0), &[4, 5, 6, 7]);
    }

    #[test]
    fn tile_write_visible_in_frame() {
        let mut buf: Vec<u8> = vec![0; 32];
        {
            let plane = PlaneRowsMut::from_buf(&mut buf, 8, 8, 4);
            let mut tiles = plane.split_tiles(&[0, 4, 8]);

            tiles[0].set_pixel(2, 1, 42);
            tiles[1].set_pixel(1, 0, 99);
        }
        // Tile 0, row 1, col 2 → global position row 1, col 2 → buf[8+2]
        assert_eq!(buf[10], 42);
        // Tile 1, row 0, col 1 → global position row 0, col 5 → buf[5]
        assert_eq!(buf[5], 99);
    }
}
