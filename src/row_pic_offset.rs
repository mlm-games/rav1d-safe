//! PicOffset-compatible API backed by row slices instead of DisjointMut.
//!
//! `RowPicOffset` provides the same offset+stride arithmetic as `PicOffset`,
//! but resolves to `&[BD::Pixel]` / `&mut [BD::Pixel]` via row slice indexing
//! instead of DisjointMut guards.
//!
//! This enables incremental migration: recon.rs code can switch from PicOffset
//! to RowPicOffset one function at a time without changing the arithmetic.

#![forbid(unsafe_code)]

use crate::include::common::bitdepth::BitDepth;
use std::ffi::c_int;

/// A position within a plane's row slices.
///
/// Equivalent to `PicOffset { data, offset }` but backed by row slices.
/// The offset is decomposed into (row, col) for direct row-slice indexing.
#[derive(Clone, Copy)]
pub struct RowPicPos {
    pub row: usize,
    pub col: usize,
    /// Stride in pixels (positive only — negative stride is handled by row reversal).
    pub stride: usize,
}

impl RowPicPos {
    /// Create a position from pixel coordinates.
    pub fn new(row: usize, col: usize, stride: usize) -> Self {
        Self { row, col, stride }
    }

    /// Create from block coordinates (4-pixel units).
    pub fn from_block(bx: c_int, by: c_int, stride: usize) -> Self {
        Self {
            row: (by * 4) as usize,
            col: (bx * 4) as usize,
            stride,
        }
    }

    /// Advance by `dy` rows and `dx` columns.
    pub fn offset(self, dy: isize, dx: isize) -> Self {
        Self {
            row: (self.row as isize + dy) as usize,
            col: (self.col as isize + dx) as usize,
            stride: self.stride,
        }
    }

    /// Add a pixel offset (compatible with PicOffset arithmetic).
    /// `off` is in pixels, decomposed into row/col via stride.
    pub fn add_offset(self, off: isize) -> Self {
        let stride = self.stride as isize;
        let row_delta = off / stride;
        let col_delta = off % stride;
        Self {
            row: (self.row as isize + row_delta) as usize,
            col: (self.col as isize + col_delta) as usize,
            stride: self.stride,
        }
    }

    /// Read a pixel from immutable row slices.
    pub fn index<'a, BD: BitDepth>(&self, rows: &'a [&[BD::Pixel]]) -> &'a BD::Pixel {
        &rows[self.row][self.col]
    }

    /// Read a row slice from immutable row slices.
    pub fn slice<'a, BD: BitDepth>(&self, rows: &'a [&[BD::Pixel]], w: usize) -> &'a [BD::Pixel] {
        &rows[self.row][self.col..self.col + w]
    }

    /// Write a pixel to mutable row slices.
    pub fn index_mut<'a, BD: BitDepth>(&self, rows: &'a mut [&mut [BD::Pixel]]) -> &'a mut BD::Pixel {
        &mut rows[self.row][self.col]
    }

    /// Get a mutable row slice.
    pub fn slice_mut<'a, BD: BitDepth>(&self, rows: &'a mut [&mut [BD::Pixel]], w: usize) -> &'a mut [BD::Pixel] {
        &mut rows[self.row][self.col..self.col + w]
    }
}

impl std::ops::Add<isize> for RowPicPos {
    type Output = Self;
    fn add(self, off: isize) -> Self {
        self.add_offset(off)
    }
}

impl std::ops::AddAssign<isize> for RowPicPos {
    fn add_assign(&mut self, off: isize) {
        *self = self.add_offset(off);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::include::common::bitdepth::BitDepth8;
    type BD = BitDepth8;

    #[test]
    fn from_block_coordinates() {
        let pos = RowPicPos::from_block(5, 3, 128);
        assert_eq!(pos.row, 12); // 3 * 4
        assert_eq!(pos.col, 20); // 5 * 4
    }

    #[test]
    fn offset_moves_position() {
        let pos = RowPicPos::new(10, 20, 128);
        let moved = pos.offset(3, -5);
        assert_eq!(moved.row, 13);
        assert_eq!(moved.col, 15);
    }

    #[test]
    fn read_pixel_via_index() {
        let row0: Vec<u8> = (0..16).collect();
        let row1: Vec<u8> = (16..32).collect();
        let rows: Vec<&[u8]> = vec![&row0, &row1];

        let pos = RowPicPos::new(1, 3, 16);
        assert_eq!(*pos.index::<BD>(&rows), 19); // row1[3] = 16+3
    }

    #[test]
    fn write_pixel_via_index_mut() {
        let mut row0 = vec![0u8; 16];
        let mut row1 = vec![0u8; 16];
        let mut rows: Vec<&mut [u8]> = vec![&mut row0, &mut row1];

        let pos = RowPicPos::new(0, 5, 16);
        *pos.index_mut::<BD>(&mut rows) = 42;
        assert_eq!(rows[0][5], 42);
    }

    #[test]
    fn add_offset_with_stride() {
        let pos = RowPicPos::new(0, 0, 16);
        // Adding 19 with stride 16 = 1 row + 3 cols
        let moved = pos.add_offset(19);
        assert_eq!(moved.row, 1);
        assert_eq!(moved.col, 3);
    }

    #[test]
    fn slice_reads_correct_range() {
        let row0: Vec<u8> = (0..16).collect();
        let rows: Vec<&[u8]> = vec![&row0];

        let pos = RowPicPos::new(0, 4, 16);
        let s = pos.slice::<BD>(&rows, 5);
        assert_eq!(s, &[4, 5, 6, 7, 8]);
    }
}
