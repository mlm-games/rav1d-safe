//! Borrow instrumentation for DisjointMut.
//!
//! Records 1D borrow statistics (sizes, concurrency, contention) into thread-local
//! counters. Zero-overhead when the `instrument` feature is off.
//!
//! # Usage
//!
//! ```ignore
//! // After decoding:
//! rav1d_disjoint_mut::instrument::flush_thread_stats();
//! let report = rav1d_disjoint_mut::instrument::collect_report();
//! eprintln!("{}", report.summary());
//! ```

use std::cell::Cell;
use std::sync::Mutex;

// ============================================================================
// Constants
// ============================================================================

/// Number of log2 histogram buckets: [0,1), [1,2), [2,4), ... [2^21, inf)
const NUM_BUCKETS: usize = 23;

/// Map a byte size to a log2 bucket index.
/// Bucket 0 = size 0 (empty), bucket 1 = [1,2), bucket 2 = [2,4), etc.
#[inline]
fn size_to_bucket(size: usize) -> usize {
    if size == 0 {
        return 0;
    }
    // bit_length - 1 gives log2 floor, +1 for the zero bucket
    let bits = usize::BITS - size.leading_zeros();
    (bits as usize).min(NUM_BUCKETS - 1)
}

// ============================================================================
// Per-bucket stats
// ============================================================================

#[derive(Clone, Copy, Default)]
struct BucketStats {
    mut_count: u64,
    immut_count: u64,
    mut_bytes: u64,
    immut_bytes: u64,
}

// ============================================================================
// Thread-local stats
// ============================================================================

#[derive(Clone)]
struct ThreadStats {
    buckets: [BucketStats; NUM_BUCKETS],
    mut_borrows: u64,
    immut_borrows: u64,
    empty_borrows: u64,
    removes: u64,
    peak_concurrent: u32,
    contention_events: u64,
    overflow_events: u64,
}

impl Default for ThreadStats {
    fn default() -> Self {
        Self {
            buckets: [BucketStats::default(); NUM_BUCKETS],
            mut_borrows: 0,
            immut_borrows: 0,
            empty_borrows: 0,
            removes: 0,
            peak_concurrent: 0,
            contention_events: 0,
            overflow_events: 0,
        }
    }
}

thread_local! {
    static THREAD_STATS: Cell<ThreadStats> = const { Cell::new(ThreadStats {
        buckets: [BucketStats { mut_count: 0, immut_count: 0, mut_bytes: 0, immut_bytes: 0 }; NUM_BUCKETS],
        mut_borrows: 0,
        immut_borrows: 0,
        empty_borrows: 0,
        removes: 0,
        peak_concurrent: 0,
        contention_events: 0,
        overflow_events: 0,
    }) };
}

/// Access thread-local stats, mutate via closure.
#[inline]
fn with_stats(f: impl FnOnce(&mut ThreadStats)) {
    THREAD_STATS.with(|cell| {
        let mut stats = cell.take();
        f(&mut stats);
        cell.set(stats);
    });
}

// ============================================================================
// Recording functions (called from BorrowTracker)
// ============================================================================

/// Record a mutable borrow of `size` bytes with `concurrent` active borrows.
#[inline]
pub(crate) fn record_mut_borrow(size: usize, concurrent: u32) {
    with_stats(|s| {
        s.mut_borrows += 1;
        let bucket = size_to_bucket(size);
        s.buckets[bucket].mut_count += 1;
        s.buckets[bucket].mut_bytes += size as u64;
        if concurrent > s.peak_concurrent {
            s.peak_concurrent = concurrent;
        }
    });
}

/// Record an immutable borrow of `size` bytes with `concurrent` active borrows.
#[inline]
pub(crate) fn record_immut_borrow(size: usize, concurrent: u32) {
    with_stats(|s| {
        s.immut_borrows += 1;
        let bucket = size_to_bucket(size);
        s.buckets[bucket].immut_count += 1;
        s.buckets[bucket].immut_bytes += size as u64;
        if concurrent > s.peak_concurrent {
            s.peak_concurrent = concurrent;
        }
    });
}

/// Record an empty-range borrow (start >= end, skipped).
#[inline]
pub(crate) fn record_empty_borrow() {
    with_stats(|s| {
        s.empty_borrows += 1;
    });
}

/// Record a borrow removal.
#[inline]
pub(crate) fn record_remove() {
    with_stats(|s| {
        s.removes += 1;
    });
}

/// Record a lock contention event (lock_slow entered).
#[inline]
pub(crate) fn record_contention() {
    with_stats(|s| {
        s.contention_events += 1;
    });
}

/// Record a borrow overflow (>64 concurrent borrows spilled to Vec).
#[inline]
pub(crate) fn record_overflow() {
    with_stats(|s| {
        s.overflow_events += 1;
    });
}

// ============================================================================
// Global registry
// ============================================================================

static GLOBAL_REGISTRY: Mutex<Vec<ThreadStats>> = Mutex::new(Vec::new());

/// Flush this thread's stats into the global registry. Call from each thread
/// before it exits, or from the main thread after joining workers.
pub fn flush_thread_stats() {
    THREAD_STATS.with(|cell| {
        let stats = cell.take();
        // Only flush if there's actual data
        if stats.mut_borrows > 0 || stats.immut_borrows > 0 || stats.empty_borrows > 0 {
            GLOBAL_REGISTRY.lock().unwrap().push(stats.clone());
        }
        // Reset thread-local
        cell.set(ThreadStats::default());
    });
}

/// Collect all flushed thread stats into an aggregated report.
///
/// Call `flush_thread_stats()` from each thread first. This drains the registry.
pub fn collect_report() -> BorrowReport {
    let entries = {
        let mut registry = GLOBAL_REGISTRY.lock().unwrap();
        core::mem::take(&mut *registry)
    };

    let mut report = BorrowReport::default();
    for ts in &entries {
        for i in 0..NUM_BUCKETS {
            report.buckets[i].mut_count += ts.buckets[i].mut_count;
            report.buckets[i].immut_count += ts.buckets[i].immut_count;
            report.buckets[i].mut_bytes += ts.buckets[i].mut_bytes;
            report.buckets[i].immut_bytes += ts.buckets[i].immut_bytes;
        }
        report.mut_borrows += ts.mut_borrows;
        report.immut_borrows += ts.immut_borrows;
        report.empty_borrows += ts.empty_borrows;
        report.removes += ts.removes;
        if ts.peak_concurrent > report.peak_concurrent {
            report.peak_concurrent = ts.peak_concurrent;
        }
        report.contention_events += ts.contention_events;
        report.overflow_events += ts.overflow_events;
    }
    report.thread_count = entries.len() as u32;
    report
}

/// Reset the global registry (for testing).
pub fn reset_global_stats() {
    GLOBAL_REGISTRY.lock().unwrap().clear();
}

// ============================================================================
// Report
// ============================================================================

/// Aggregated borrow statistics across all threads.
#[derive(Clone, Default)]
pub struct BorrowReport {
    pub buckets: [BucketStats; NUM_BUCKETS],
    pub mut_borrows: u64,
    pub immut_borrows: u64,
    pub empty_borrows: u64,
    pub removes: u64,
    pub peak_concurrent: u32,
    pub contention_events: u64,
    pub overflow_events: u64,
    pub thread_count: u32,
}

// Make BucketStats fields public for the report
impl BucketStats {
    pub fn mut_count(&self) -> u64 {
        self.mut_count
    }
    pub fn immut_count(&self) -> u64 {
        self.immut_count
    }
    pub fn mut_bytes(&self) -> u64 {
        self.mut_bytes
    }
    pub fn immut_bytes(&self) -> u64 {
        self.immut_bytes
    }
}

impl BorrowReport {
    /// Human-readable summary of borrow statistics.
    pub fn summary(&self) -> String {
        use std::fmt::Write;
        let mut out = String::with_capacity(2048);

        let total_borrows = self.mut_borrows + self.immut_borrows;
        let _ = writeln!(out, "=== DisjointMut Borrow Report ===");
        let _ = writeln!(out, "Threads reporting: {}", self.thread_count);
        let _ = writeln!(
            out,
            "Total borrows: {} (mut: {}, immut: {}, empty: {})",
            total_borrows, self.mut_borrows, self.immut_borrows, self.empty_borrows
        );
        let _ = writeln!(out, "Total removes: {}", self.removes);
        let _ = writeln!(out, "Peak concurrent borrows: {}", self.peak_concurrent);
        let _ = writeln!(out, "Lock contention events: {}", self.contention_events);
        let _ = writeln!(out, "Overflow events (>64 slots): {}", self.overflow_events);
        let _ = writeln!(out);

        // Size histogram
        let _ = writeln!(
            out,
            "{:<20} {:>10} {:>10} {:>12} {:>12}",
            "Size Range", "Mut Count", "Imm Count", "Mut Bytes", "Imm Bytes"
        );
        let _ = writeln!(out, "{:-<66}", "");
        for i in 0..NUM_BUCKETS {
            let b = &self.buckets[i];
            if b.mut_count == 0 && b.immut_count == 0 {
                continue;
            }
            let range = if i == 0 {
                "0 (empty)".to_string()
            } else if i == 1 {
                "1".to_string()
            } else if i < NUM_BUCKETS - 1 {
                format!("[{}, {})", 1usize << (i - 1), 1usize << i)
            } else {
                format!("[{}, inf)", 1usize << (i - 1))
            };
            let _ = writeln!(
                out,
                "{:<20} {:>10} {:>10} {:>12} {:>12}",
                range,
                b.mut_count,
                b.immut_count,
                format_bytes(b.mut_bytes),
                format_bytes(b.immut_bytes),
            );
        }

        out
    }
}

fn format_bytes(bytes: u64) -> String {
    if bytes < 1024 {
        format!("{}B", bytes)
    } else if bytes < 1024 * 1024 {
        format!("{:.1}KB", bytes as f64 / 1024.0)
    } else if bytes < 1024 * 1024 * 1024 {
        format!("{:.1}MB", bytes as f64 / (1024.0 * 1024.0))
    } else {
        format!("{:.1}GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_size_to_bucket() {
        assert_eq!(size_to_bucket(0), 0);
        assert_eq!(size_to_bucket(1), 1);
        assert_eq!(size_to_bucket(2), 2);
        assert_eq!(size_to_bucket(3), 2);
        assert_eq!(size_to_bucket(4), 3);
        assert_eq!(size_to_bucket(7), 3);
        assert_eq!(size_to_bucket(8), 4);
        assert_eq!(size_to_bucket(1024), 11);
        assert_eq!(size_to_bucket(1 << 21), 22);
        assert_eq!(size_to_bucket(1 << 30), 22); // clamped
    }

    #[test]
    fn test_record_and_report() {
        // Reset any stale state
        reset_global_stats();

        record_mut_borrow(100, 1);
        record_mut_borrow(200, 2);
        record_immut_borrow(50, 3);
        record_empty_borrow();
        record_remove();
        record_contention();
        record_overflow();

        flush_thread_stats();
        let report = collect_report();

        assert_eq!(report.mut_borrows, 2);
        assert_eq!(report.immut_borrows, 1);
        assert_eq!(report.empty_borrows, 1);
        assert_eq!(report.removes, 1);
        assert_eq!(report.peak_concurrent, 3);
        assert_eq!(report.contention_events, 1);
        assert_eq!(report.overflow_events, 1);
        assert_eq!(report.thread_count, 1);

        let summary = report.summary();
        assert!(summary.contains("Total borrows: 3"));
    }
}
