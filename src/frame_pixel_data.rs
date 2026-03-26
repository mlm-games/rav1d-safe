//! Frame pixel data with both DisjointMut-wrapped and raw row-slice access.
//!
//! During the transition from PicOffset to row slices, we need both access paths:
//! - PicOffset (via Rav1dPictureDataComponent) for filter stages
//! - Row slices (via PlaneRowsMut) for reconstruction
//!
//! This module provides `FramePixelData` which wraps the per-plane `Vec<u8>` buffers
//! and provides row-slice access without going through DisjointMut.
//!
//! IMPORTANT: The Vec<u8> and the Rav1dPictureDataComponent point to the SAME memory
//! (in the c-ffi path). In the safe path, Rav1dPictureDataComponent copies the data
//! into a PicBuf. This means modifications via row slices are NOT visible through
//! PicOffset in the safe path. For the rayon integration, we need to either:
//! 1. Use the row-slice path exclusively (no PicOffset) for reconstruction
//! 2. Or use c-ffi mode where both point to the same memory
//!
//! The end goal is option 1: row slices only, no DisjointMut, no PicOffset.

#![forbid(unsafe_code)]

use crate::include::common::bitdepth::BitDepth;
use crate::src::frame_rows::PlaneRowsMut;
use std::ffi::c_int;

/// Per-plane pixel data that can be split into row slices.
///
/// Each plane is stored as a contiguous `Vec<u8>` with known stride and dimensions.
/// Row slices are created on demand via `rows_mut()`.
pub struct PlanePixelBuf {
    /// Raw pixel buffer (u8 for 8bpc, pairs of u8 for 16bpc).
    pub buf: Vec<u8>,
    /// Byte stride per row (includes alignment padding).
    pub byte_stride: usize,
    /// Width in pixels.
    pub width: usize,
    /// Height in rows.
    pub height: usize,
}

impl PlanePixelBuf {
    /// Create a new zero-initialized plane buffer.
    pub fn new(byte_stride: usize, width: usize, height: usize) -> Self {
        let buf = vec![0u8; byte_stride * height];
        Self {
            buf,
            byte_stride,
            width,
            height,
        }
    }

    /// Create from an existing Vec.
    pub fn from_vec(buf: Vec<u8>, byte_stride: usize, width: usize, height: usize) -> Self {
        assert!(buf.len() >= byte_stride * height.saturating_sub(1) + width);
        Self {
            buf,
            byte_stride,
            width,
            height,
        }
    }

    /// Pixel stride for a given bit depth.
    pub fn pixel_stride<BD: BitDepth>(&self) -> usize {
        self.byte_stride / std::mem::size_of::<BD::Pixel>()
    }

    /// Create mutable row slices for 8bpc pixels.
    pub fn rows_mut_8bpc(&mut self) -> PlaneRowsMut<'_, u8> {
        PlaneRowsMut::from_buf(
            &mut self.buf,
            self.byte_stride,
            self.width,
            self.height,
        )
    }

    /// Create mutable row slices for a range of rows (8bpc).
    pub fn rows_mut_range_8bpc(
        &mut self,
        start_row: usize,
        end_row: usize,
    ) -> PlaneRowsMut<'_, u8> {
        let start = start_row * self.byte_stride;
        let nrows = end_row - start_row;
        PlaneRowsMut::from_buf(
            &mut self.buf[start..],
            self.byte_stride,
            self.width,
            nrows,
        )
    }
}

/// Frame pixel data for all 3 planes with row-slice access.
pub struct FramePixelData {
    pub y: PlanePixelBuf,
    pub u: PlanePixelBuf,
    pub v: PlanePixelBuf,
}

impl FramePixelData {
    /// Create a new frame with the given dimensions.
    pub fn new(
        width: c_int,
        height: c_int,
        layout: crate::include::dav1d::headers::Rav1dPixelLayout,
        bpc: u8,
    ) -> Self {
        use crate::include::dav1d::headers::Rav1dPixelLayout;

        let hbd = (bpc > 8) as usize;
        let pixel_size = 1 + hbd;
        let aligned_w = ((width as usize + 127) & !127) * pixel_size;
        let has_chroma = layout != Rav1dPixelLayout::I400;
        let ss_hor = (layout != Rav1dPixelLayout::I444) as usize;
        let ss_ver = (layout == Rav1dPixelLayout::I420) as usize;

        let mut y_stride = aligned_w;
        let mut uv_stride = if has_chroma { aligned_w >> ss_hor } else { 0 };
        // Avoid 1024-byte alignment issues (same as alloc_picture_data)
        if y_stride & 1023 == 0 {
            y_stride += 64;
        }
        if uv_stride & 1023 == 0 && has_chroma {
            uv_stride += 64;
        }

        let w = width as usize;
        let h = height as usize;
        let uw = if has_chroma { w >> ss_hor } else { 0 };
        let uh = if has_chroma { h >> ss_ver } else { 0 };

        Self {
            y: PlanePixelBuf::new(y_stride, w * pixel_size, h),
            u: PlanePixelBuf::new(uv_stride, uw * pixel_size, uh),
            v: PlanePixelBuf::new(uv_stride, uw * pixel_size, uh),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::include::dav1d::headers::Rav1dPixelLayout;

    #[test]
    fn frame_pixel_data_dimensions() {
        let fpd = FramePixelData::new(320, 240, Rav1dPixelLayout::I420, 8);
        assert_eq!(fpd.y.width, 320);
        assert_eq!(fpd.y.height, 240);
        assert_eq!(fpd.u.width, 160); // 320/2
        assert_eq!(fpd.u.height, 120); // 240/2
    }

    #[test]
    fn plane_rows_mut_8bpc() {
        let mut plane = PlanePixelBuf::new(64, 32, 4);
        {
            let mut rows = plane.rows_mut_8bpc();
            rows.row_mut(0)[0] = 42;
            rows.row_mut(1)[31] = 99;
        }
        assert_eq!(plane.buf[0], 42);
        assert_eq!(plane.buf[64 + 31], 99);
    }

    #[test]
    fn plane_rows_range() {
        let mut plane = PlanePixelBuf::new(16, 8, 8);
        {
            let mut rows = plane.rows_mut_range_8bpc(2, 5);
            assert_eq!(rows.height(), 3);
            rows.row_mut(0)[0] = 77; // row 2 of the plane
        }
        assert_eq!(plane.buf[2 * 16], 77);
    }
}
