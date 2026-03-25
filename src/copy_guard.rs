//! Copy-buffer guard for safe multithreaded pixel access.
//!
//! `CopyGuard` copies pixels from a picture plane into a compact buffer
//! (no stride gaps), lets SIMD operate on the compact buffer, then copies
//! the results back on drop via per-row guards.
//!
//! This is sound because:
//! - The compact buffer is owned (no aliasing)
//! - Picture access uses narrow per-row guards (w bytes, no stride gaps)
//! - Per-row guards are acquired and released immediately (no held borrows)
//! - Two tile threads can hold CopyGuards for non-overlapping columns
//!   because their per-row guards cover disjoint byte ranges

#![forbid(unsafe_code)]

use crate::include::common::bitdepth::BitDepth;
use crate::include::dav1d::picture::Rav1dPictureDataComponentOffset;
use crate::src::strided::Strided as _;
use core::ops::{Deref, DerefMut};
use zerocopy::IntoBytes;

pub type PicOffset<'a> = Rav1dPictureDataComponentOffset<'a>;

/// A compact w×h pixel buffer that writes back to the picture on drop.
pub struct CopyGuard<'a> {
    buf: Vec<u8>,
    pic: PicOffset<'a>,
    w_bytes: usize,
    h: usize,
    byte_stride: isize,
}

impl<'a> CopyGuard<'a> {
    /// Create by copying w×h pixels from the picture into a compact buffer.
    pub fn new<BD: BitDepth>(pic: PicOffset<'a>, w: usize, h: usize) -> Self {
        let pixel_size = core::mem::size_of::<BD::Pixel>();
        let w_bytes = w * pixel_size;
        let byte_stride = pic.stride();
        let abs_byte_stride = byte_stride.unsigned_abs();

        let mut buf = vec![0u8; w_bytes * h];

        // Copy-in: read each row via narrow per-row guard
        for y in 0..h {
            let row_byte_off = y as isize * byte_stride;
            let row_start = if row_byte_off >= 0 {
                pic.offset + row_byte_off as usize
            } else {
                pic.offset - (-row_byte_off) as usize
            };
            let guard = pic.data.slice_bytes(row_start, w_bytes);
            let dst_start = y * w_bytes;
            buf[dst_start..dst_start + w_bytes].copy_from_slice(&*guard);
        }

        Self {
            buf,
            pic,
            w_bytes,
            h,
            byte_stride,
        }
    }

    /// The byte stride of the compact buffer (= w_bytes, contiguous).
    pub fn compact_stride(&self) -> usize {
        self.w_bytes
    }
}

impl Deref for CopyGuard<'_> {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        &self.buf
    }
}

impl DerefMut for CopyGuard<'_> {
    fn deref_mut(&mut self) -> &mut [u8] {
        &mut self.buf
    }
}

impl Drop for CopyGuard<'_> {
    fn drop(&mut self) {
        // Copy-out: write each modified row back via narrow per-row guard
        for y in 0..self.h {
            let src_start = y * self.w_bytes;
            let src = &self.buf[src_start..src_start + self.w_bytes];
            let row_byte_off = y as isize * self.byte_stride;
            let row_start = if row_byte_off >= 0 {
                self.pic.offset + row_byte_off as usize
            } else {
                self.pic.offset - (-row_byte_off) as usize
            };
            let mut guard = self.pic.data.slice_mut_bytes(row_start, self.w_bytes);
            guard.copy_from_slice(src);
        }
    }
}
