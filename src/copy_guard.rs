//! Copy-buffer guard for safe multithreaded pixel access.
//!
//! `CopyGuard<P>` copies pixels from a picture plane into a compact buffer
//! (stride = w, no gaps), lets SIMD operate on the buffer, then copies
//! the results back on drop via per-row guards. Generic over pixel type.

#![forbid(unsafe_code)]

use crate::include::common::bitdepth::BitDepth;
use crate::include::dav1d::picture::Rav1dPictureDataComponentOffset;
use crate::src::strided::Strided as _;
use core::ops::{Deref, DerefMut};
use zerocopy::{FromBytes, Immutable, IntoBytes};

pub type PicOffset<'a> = Rav1dPictureDataComponentOffset<'a>;

/// A compact w×h pixel buffer that writes back to the picture on drop.
/// Generic over pixel type P (u8 for 8bpc, u16 for 16bpc).
pub struct CopyGuard<'a, P: Copy + FromBytes + IntoBytes + Immutable> {
    buf: Vec<P>,
    pic: PicOffset<'a>,
    w: usize,
    h: usize,
    pixel_stride: isize, // in pixels, not bytes
}

impl<'a, P: Copy + FromBytes + IntoBytes + Immutable> CopyGuard<'a, P> {
    /// Create by copying w×h pixels from the picture into a compact buffer.
    pub fn new<BD: BitDepth<Pixel = P>>(pic: PicOffset<'a>, w: usize, h: usize) -> Self {
        let pixel_stride = pic.data.pixel_stride::<BD>();
        let pixel_size = core::mem::size_of::<P>();
        let byte_stride = pic.stride();

        let mut buf = vec![P::new_zeroed(); w * h];

        // Copy-in: read each row via narrow per-row guard.
        // pic.offset is in PIXEL units. byte_stride is in BYTES.
        // slice_bytes expects BYTE offset and length.
        let base_byte_offset = pic.offset * pixel_size;
        for y in 0..h {
            let row_byte_offset = base_byte_offset.wrapping_add_signed(y as isize * byte_stride);
            let guard = pic.data.slice_bytes(row_byte_offset, w * pixel_size);
            let src_pixels: &[P] = FromBytes::ref_from_bytes(&*guard).unwrap();
            buf[y * w..(y + 1) * w].copy_from_slice(&src_pixels[..w]);
        }

        Self {
            buf,
            pic,
            w,
            h,
            pixel_stride,
        }
    }

    /// The pixel stride of the compact buffer (= w, contiguous).
    pub fn compact_stride(&self) -> usize {
        self.w
    }
}

impl<P: Copy + FromBytes + IntoBytes + Immutable> Deref for CopyGuard<'_, P> {
    type Target = [P];
    fn deref(&self) -> &[P] {
        &self.buf
    }
}

impl<P: Copy + FromBytes + IntoBytes + Immutable> DerefMut for CopyGuard<'_, P> {
    fn deref_mut(&mut self) -> &mut [P] {
        &mut self.buf
    }
}

impl<P: Copy + FromBytes + IntoBytes + Immutable> Drop for CopyGuard<'_, P> {
    fn drop(&mut self) {
        let pixel_size = core::mem::size_of::<P>();
        let byte_stride = self.pic.stride();
        // Copy-out: write each modified row back via narrow per-row guard.
        let base_byte_offset = self.pic.offset * pixel_size;
        for y in 0..self.h {
            let row_byte_offset =
                base_byte_offset.wrapping_add_signed(y as isize * byte_stride);
            let mut guard =
                self.pic
                    .data
                    .slice_mut_bytes(row_byte_offset, self.w * pixel_size);
            let dst_pixels: &mut [P] = FromBytes::mut_from_bytes(&mut *guard).unwrap();
            dst_pixels[..self.w].copy_from_slice(&self.buf[y * self.w..(y + 1) * self.w]);
        }
    }
}
