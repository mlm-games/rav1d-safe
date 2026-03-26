//! Progressive frame buffer for safe cross-SB-row and cross-frame access.
//!
//! `ProgressiveFrame` wraps a contiguous pixel buffer with a monotonically
//! advancing freeze boundary. Rows below the boundary are immutable (available
//! for reference reads by other frames). Rows at and above the boundary can
//! be written by the owning decoder.
//!
//! This enables frame-level parallelism: Frame N+1 can read Frame N's frozen
//! rows while Frame N continues filtering later rows.
//!
//! The freeze boundary enforces a temporal partition:
//!   [0, frozen_through) — immutable, readable by any thread
//!   [frozen_through, height) — mutable, writable by owner only
//!
//! Soundness: Release/Acquire ordering on the atomic boundary ensures that
//! all writes before freeze_through are visible to readers after the boundary.

#![forbid(unsafe_code)]

use std::sync::atomic::{AtomicUsize, Ordering};

/// A frame buffer with a progressive freeze boundary.
///
/// Rows are frozen top-to-bottom as decoding + filtering completes.
/// Once frozen, a row is never written again.
pub struct ProgressiveFrame<P> {
    /// Contiguous pixel buffer (stride-aligned).
    buf: Vec<P>,
    /// Pixels per row (including padding to stride alignment).
    stride: usize,
    /// Active pixels per row (≤ stride).
    width: usize,
    /// Total rows in the frame.
    height: usize,
    /// Rows [0, frozen_through) are immutable.
    /// Monotonically non-decreasing.
    frozen_through: AtomicUsize,
}

impl<P: Copy + Default> ProgressiveFrame<P> {
    /// Create a new frame with all rows unfrozen.
    pub fn new(stride: usize, width: usize, height: usize) -> Self {
        assert!(stride >= width);
        Self {
            buf: vec![P::default(); stride * height],
            stride,
            width,
            height,
            frozen_through: AtomicUsize::new(0),
        }
    }

    /// Create from an existing buffer.
    pub fn from_buf(buf: Vec<P>, stride: usize, width: usize, height: usize) -> Self {
        assert!(stride >= width);
        assert!(buf.len() >= stride * height.saturating_sub(1) + width);
        Self {
            buf,
            stride,
            width,
            height,
            frozen_through: AtomicUsize::new(0),
        }
    }
}

impl<P> ProgressiveFrame<P> {
    /// Frame width in pixels.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Frame height in rows.
    pub fn height(&self) -> usize {
        self.height
    }

    /// Stride in pixels (including padding).
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// Current freeze boundary (rows [0, boundary) are immutable).
    pub fn frozen_through(&self) -> usize {
        self.frozen_through.load(Ordering::Acquire)
    }

    /// Is the entire frame frozen?
    pub fn is_fully_frozen(&self) -> bool {
        self.frozen_through() >= self.height
    }

    /// Advance the freeze boundary. Must be monotonically non-decreasing.
    ///
    /// After this call, rows [0, through) are guaranteed visible to all
    /// threads that subsequently call `frozen_row`.
    ///
    /// # Panics
    ///
    /// Panics if `through` is less than the current boundary or exceeds height.
    pub fn freeze_through(&self, through: usize) {
        assert!(through <= self.height, "cannot freeze past frame height");
        let prev = self.frozen_through.load(Ordering::Relaxed);
        assert!(
            through >= prev,
            "freeze boundary must advance: {through} < {prev}"
        );
        self.frozen_through.store(through, Ordering::Release);
    }

    /// Freeze the entire frame (called after all filters complete).
    pub fn freeze_all(&self) {
        self.freeze_through(self.height);
    }

    /// Read a frozen row (immutable access).
    ///
    /// Returns a slice of `width` pixels for the given row.
    ///
    /// # Panics
    ///
    /// Panics if the row is not yet frozen.
    pub fn frozen_row(&self, y: usize) -> &[P] {
        let frozen = self.frozen_through.load(Ordering::Acquire);
        assert!(
            y < frozen,
            "row {y} not yet frozen (boundary={frozen})"
        );
        let start = y * self.stride;
        &self.buf[start..start + self.width]
    }

    /// Get immutable row slices for a range of frozen rows.
    ///
    /// # Panics
    ///
    /// Panics if any row in the range is not yet frozen.
    pub fn frozen_rows(&self, start_row: usize, end_row: usize) -> Vec<&[P]> {
        let frozen = self.frozen_through.load(Ordering::Acquire);
        assert!(
            end_row <= frozen,
            "rows {start_row}..{end_row} not all frozen (boundary={frozen})"
        );
        (start_row..end_row)
            .map(|y| {
                let start = y * self.stride;
                &self.buf[start..start + self.width]
            })
            .collect()
    }

    /// Get mutable access to unfrozen rows.
    ///
    /// Returns mutable row slices for rows [start_row, end_row).
    /// All requested rows must be at or above the freeze boundary.
    ///
    /// # Safety note
    ///
    /// This requires `&mut self`, which the borrow checker enforces is exclusive.
    /// In the rayon pipeline, this is called within the SB row's task which has
    /// exclusive ownership of the frame during that phase.
    ///
    /// # Panics
    ///
    /// Panics if any requested row is frozen.
    pub fn active_rows_mut(&mut self, start_row: usize, end_row: usize) -> Vec<&mut [P]> {
        let frozen = *self.frozen_through.get_mut();
        assert!(
            start_row >= frozen,
            "cannot mutate frozen row {start_row} (boundary={frozen})"
        );
        assert!(end_row <= self.height);
        self.buf
            .chunks_mut(self.stride)
            .skip(start_row)
            .take(end_row - start_row)
            .map(|r| &mut r[..self.width])
            .collect()
    }

    /// Get the raw buffer (for passing to existing code that needs flat access).
    pub fn buf(&self) -> &[P] {
        &self.buf
    }

    /// Get the raw buffer mutably.
    pub fn buf_mut(&mut self) -> &mut [P] {
        &mut self.buf
    }

    /// Consume the frame and return the buffer.
    pub fn into_buf(self) -> Vec<P> {
        self.buf
    }
}

// ProgressiveFrame is Send if P is Send (the AtomicUsize is inherently Send+Sync).
// We don't impl Sync because frozen_row(&self) and active_rows_mut(&mut self)
// are exclusive by &self/&mut self, but frozen_row from one thread and
// active_rows_mut from another would require Sync + UnsafeCell.
// For now, ProgressiveFrame is used with &mut in the owning task
// and frozen_rows via a shared reference (after the task gives up &mut).

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_frame_is_unfrozen() {
        let frame = ProgressiveFrame::<u8>::new(8, 6, 4);
        assert_eq!(frame.frozen_through(), 0);
        assert!(!frame.is_fully_frozen());
    }

    #[test]
    fn freeze_advances_monotonically() {
        let frame = ProgressiveFrame::<u8>::new(8, 6, 4);
        frame.freeze_through(2);
        assert_eq!(frame.frozen_through(), 2);
        frame.freeze_through(4);
        assert_eq!(frame.frozen_through(), 4);
        assert!(frame.is_fully_frozen());
    }

    #[test]
    #[should_panic(expected = "must advance")]
    fn freeze_cannot_go_backwards() {
        let frame = ProgressiveFrame::<u8>::new(8, 6, 4);
        frame.freeze_through(3);
        frame.freeze_through(2); // panic
    }

    #[test]
    fn frozen_row_returns_correct_data() {
        let mut frame = ProgressiveFrame::<u8>::new(8, 6, 4);
        // Write some data to row 0
        {
            let mut rows = frame.active_rows_mut(0, 4);
            rows[0][0] = 42;
            rows[0][5] = 99;
        }
        frame.freeze_through(2);

        let row = frame.frozen_row(0);
        assert_eq!(row[0], 42);
        assert_eq!(row[5], 99);
        assert_eq!(row.len(), 6); // width, not stride
    }

    #[test]
    #[should_panic(expected = "not yet frozen")]
    fn frozen_row_panics_for_unfrozen() {
        let frame = ProgressiveFrame::<u8>::new(8, 6, 4);
        frame.frozen_row(0); // panic — nothing frozen yet
    }

    #[test]
    fn active_rows_mut_works_above_boundary() {
        let mut frame = ProgressiveFrame::<u8>::new(8, 6, 4);
        frame.freeze_through(2);
        // Can get mutable rows 2..4
        let rows = frame.active_rows_mut(2, 4);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].len(), 6); // width
    }

    #[test]
    #[should_panic(expected = "cannot mutate frozen")]
    fn active_rows_panics_for_frozen() {
        let mut frame = ProgressiveFrame::<u8>::new(8, 6, 4);
        frame.freeze_through(2);
        frame.active_rows_mut(1, 3); // panic — row 1 is frozen
    }

    #[test]
    fn frozen_rows_batch() {
        let mut frame = ProgressiveFrame::<u8>::new(8, 6, 4);
        {
            let rows = frame.active_rows_mut(0, 4);
            for (y, row) in rows.into_iter().enumerate() {
                row.fill(y as u8 * 10);
            }
        }
        frame.freeze_through(4);

        let rows = frame.frozen_rows(0, 4);
        assert_eq!(rows.len(), 4);
        assert_eq!(rows[0][0], 0);
        assert_eq!(rows[1][0], 10);
        assert_eq!(rows[2][0], 20);
        assert_eq!(rows[3][0], 30);
    }

    #[test]
    fn progressive_freeze_workflow() {
        // Simulate the frame decode workflow:
        // 1. Write SB row 0, freeze it
        // 2. Write SB row 1, freeze it
        // 3. Read frozen rows from "another frame"
        let mut frame = ProgressiveFrame::<u8>::new(16, 12, 8);

        // SB row 0: rows 0..4
        {
            let rows = frame.active_rows_mut(0, 4);
            for row in rows {
                row.fill(100);
            }
        }
        frame.freeze_through(4);

        // Verify frozen rows are readable
        for y in 0..4 {
            assert_eq!(frame.frozen_row(y)[0], 100);
        }

        // SB row 1: rows 4..8
        {
            let rows = frame.active_rows_mut(4, 8);
            for row in rows {
                row.fill(200);
            }
        }
        frame.freeze_through(8);
        assert!(frame.is_fully_frozen());

        // All rows frozen
        for y in 0..4 {
            assert_eq!(frame.frozen_row(y)[0], 100);
        }
        for y in 4..8 {
            assert_eq!(frame.frozen_row(y)[0], 200);
        }
    }
}
