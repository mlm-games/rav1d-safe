//! Per-tile pixel buffers for safe parallel reconstruction.
//!
//! Each tile gets its own separate pixel buffer for its column range.
//! After reconstruction, the tile buffers are composited back into the
//! frame's Rav1dPictureDataComponent. This avoids DisjointMut overlap
//! between concurrent tiles.
//!
//! The copy-back cost is small: ~0.2ms for a 4K frame (8MB memcpy at ~40GB/s).

#![forbid(unsafe_code)]

use crate::include::common::bitdepth::BitDepth;
use crate::include::dav1d::headers::Rav1dPixelLayout;
use crate::include::dav1d::picture::Rav1dPictureDataComponent;
use crate::src::strided::Strided as _;
use std::ffi::c_int;

/// Pixel buffer for one tile column's worth of one plane for one SB row.
pub struct TilePlaneBuf {
    /// Pixel data (row-major, width = tile_width * pixel_size).
    pub buf: Vec<u8>,
    /// Byte stride per row.
    pub byte_stride: usize,
    /// Tile width in pixels.
    pub tile_pixel_width: usize,
    /// Number of rows (SB height, clamped to frame boundary).
    pub height: usize,
    /// Column offset of this tile in the frame (in pixels).
    pub frame_col_offset: usize,
    /// Row offset of this SB row in the frame (in pixels).
    pub frame_row_offset: usize,
}

impl TilePlaneBuf {
    /// Create a new tile plane buffer.
    pub fn new(
        tile_pixel_width: usize,
        height: usize,
        pixel_size: usize,
        frame_col_offset: usize,
        frame_row_offset: usize,
    ) -> Self {
        let byte_stride = tile_pixel_width * pixel_size;
        let buf = vec![0u8; byte_stride * height];
        Self {
            buf,
            byte_stride,
            tile_pixel_width,
            height,
            frame_col_offset,
            frame_row_offset,
        }
    }

    /// Get mutable 8bpc row slices.
    pub fn rows_mut_8bpc(&mut self) -> Vec<&mut [u8]> {
        self.buf
            .chunks_mut(self.byte_stride)
            .take(self.height)
            .map(|r| &mut r[..self.tile_pixel_width])
            .collect()
    }

    /// Copy tile pixels back to the frame's picture component.
    ///
    /// Copies each row of the tile buffer into the corresponding position
    /// in the frame's Rav1dPictureDataComponent.
    pub fn copy_to_frame<BD: BitDepth>(&self, frame_plane: &Rav1dPictureDataComponent) {
        let pixel_size = std::mem::size_of::<BD::Pixel>();
        let frame_byte_stride = frame_plane.stride().unsigned_abs() as usize;
        let tile_byte_width = self.tile_pixel_width * pixel_size;
        let col_byte_offset = self.frame_col_offset * pixel_size;

        for y in 0..self.height {
            let frame_row = self.frame_row_offset + y;
            let src_start = y * self.byte_stride;
            let src = &self.buf[src_start..src_start + tile_byte_width];

            // Write to frame via DisjointMut (single-threaded copy-back, no overlap)
            let dst_start = frame_row * frame_byte_stride + col_byte_offset;
            let mut guard = frame_plane.slice_mut::<BD, _>(
                (dst_start / pixel_size.., ..tile_byte_width / pixel_size),
            );
            let dst_bytes = zerocopy::IntoBytes::as_mut_bytes(&mut *guard);
            dst_bytes.copy_from_slice(src);
        }
    }

    /// Copy frame pixels into the tile buffer (for read-modify-write operations
    /// like ITX residual addition).
    pub fn copy_from_frame<BD: BitDepth>(&mut self, frame_plane: &Rav1dPictureDataComponent) {
        let pixel_size = std::mem::size_of::<BD::Pixel>();
        let frame_byte_stride = frame_plane.stride().unsigned_abs() as usize;
        let tile_byte_width = self.tile_pixel_width * pixel_size;
        let col_byte_offset = self.frame_col_offset * pixel_size;

        for y in 0..self.height {
            let frame_row = self.frame_row_offset + y;
            let dst_start = y * self.byte_stride;
            let dst = &mut self.buf[dst_start..dst_start + tile_byte_width];

            let src_start = frame_row * frame_byte_stride + col_byte_offset;
            let guard = frame_plane.slice::<BD, _>(
                (src_start / pixel_size.., ..tile_byte_width / pixel_size),
            );
            let src_bytes = zerocopy::IntoBytes::as_bytes(&*guard);
            dst.copy_from_slice(src_bytes);
        }
    }
}

/// All 3 planes for one tile within one SB row.
pub struct TilePixelBufs {
    pub y: TilePlaneBuf,
    pub u: Option<TilePlaneBuf>,
    pub v: Option<TilePlaneBuf>,
}

impl TilePixelBufs {
    /// Create tile buffers for an SB row.
    pub fn new(
        tile_col_start: c_int, // in 4-pixel units
        tile_col_end: c_int,   // in 4-pixel units
        sby_row_start: usize,  // first pixel row of this SB row
        sb_height: usize,      // pixels
        frame_height: usize,   // total frame height in pixels
        layout: Rav1dPixelLayout,
        bpc: u8,
    ) -> Self {
        let pixel_size = if bpc > 8 { 2 } else { 1 };
        let tile_width = (tile_col_end - tile_col_start) as usize * 4;
        let nrows = sb_height.min(frame_height - sby_row_start);
        let col_offset = tile_col_start as usize * 4;

        let ss_hor = (layout != Rav1dPixelLayout::I444) as usize;
        let ss_ver = (layout == Rav1dPixelLayout::I420) as usize;
        let has_chroma = layout != Rav1dPixelLayout::I400;

        let y = TilePlaneBuf::new(tile_width, nrows, pixel_size, col_offset, sby_row_start);

        let (u, v) = if has_chroma {
            let uv_width = tile_width >> ss_hor;
            let uv_rows = nrows >> ss_ver;
            let uv_col = col_offset >> ss_hor;
            let uv_row = sby_row_start >> ss_ver;
            (
                Some(TilePlaneBuf::new(uv_width, uv_rows, pixel_size, uv_col, uv_row)),
                Some(TilePlaneBuf::new(uv_width, uv_rows, pixel_size, uv_col, uv_row)),
            )
        } else {
            (None, None)
        };

        Self { y, u, v }
    }

    /// Copy all tile buffers back to the frame.
    pub fn copy_to_frame<BD: BitDepth>(
        &self,
        frame_data: &[Rav1dPictureDataComponent; 3],
    ) {
        self.y.copy_to_frame::<BD>(&frame_data[0]);
        if let Some(u) = &self.u {
            u.copy_to_frame::<BD>(&frame_data[1]);
        }
        if let Some(v) = &self.v {
            v.copy_to_frame::<BD>(&frame_data[2]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tile_plane_buf_basic() {
        let mut tile = TilePlaneBuf::new(8, 4, 1, 16, 0);
        assert_eq!(tile.buf.len(), 8 * 4);
        let rows = tile.rows_mut_8bpc();
        assert_eq!(rows.len(), 4);
        assert_eq!(rows[0].len(), 8);
    }

    #[test]
    fn tile_pixel_bufs_420() {
        let bufs = TilePixelBufs::new(
            0, 16, // tile columns 0..64 pixels
            0, 64, 240,
            Rav1dPixelLayout::I420,
            8,
        );
        assert_eq!(bufs.y.tile_pixel_width, 64);
        assert_eq!(bufs.y.height, 64);
        assert_eq!(bufs.u.as_ref().unwrap().tile_pixel_width, 32);
        assert_eq!(bufs.u.as_ref().unwrap().height, 32);
    }
}
