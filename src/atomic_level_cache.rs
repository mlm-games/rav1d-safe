//! Atomic level cache for safe concurrent loopfilter access.
//!
//! Replaces `DisjointMut<Vec<u8>>` for the loopfilter level cache (`f.lf.level`).
//! Uses `AtomicU8` for lock-free concurrent reads and writes between tile threads.

#![forbid(unsafe_code)]

use core::sync::atomic::{AtomicU8, Ordering};

/// Thread-safe level cache backed by `AtomicU8` entries.
/// Each entry is 4 bytes: `[Y_filter0, Y_filter1, UV_filter0, UV_filter1]`.
#[derive(Default)]
pub struct AtomicLevelCache {
    data: Vec<AtomicU8>,
}

impl AtomicLevelCache {
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }

    /// Resize the cache, zero-filling new entries.
    pub fn resize(&mut self, len: usize) {
        self.data.resize_with(len, || AtomicU8::new(0));
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Write 2 bytes at `offset`.
    #[inline]
    pub fn write2(&self, offset: usize, values: [u8; 2]) {
        self.data[offset].store(values[0], Ordering::Relaxed);
        self.data[offset + 1].store(values[1], Ordering::Relaxed);
    }

    /// Read 4 bytes at `offset` into `[u8; 4]`.
    #[inline]
    pub fn read4(&self, offset: usize) -> [u8; 4] {
        [
            self.data[offset].load(Ordering::Relaxed),
            self.data[offset + 1].load(Ordering::Relaxed),
            self.data[offset + 2].load(Ordering::Relaxed),
            self.data[offset + 3].load(Ordering::Relaxed),
        ]
    }

    /// Read a single byte.
    #[inline]
    pub fn read1(&self, offset: usize) -> u8 {
        self.data[offset].load(Ordering::Relaxed)
    }

    /// Raw pointer for FFI (asm feature only).
    /// SAFETY: AtomicU8 has the same layout as u8.
    #[cfg(feature = "asm")]
    #[inline]
    pub fn as_mut_ptr(&self) -> *mut u8 {
        self.data.as_ptr() as *mut u8
    }
}
