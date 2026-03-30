//! Regression test for DisjointMut overlap panic in tile threading.
//!
//! Crafted AV1 bitstream (extracted from fuzz corpus AVIF container) with
//! dimensions that cause tile thread loopfilter to access overlapping regions
//! of the pixel buffer, triggering DisjointMut's runtime borrow checker.
//!
//! The overlap is non-deterministic (~40% repro rate per attempt) due to thread
//! scheduling, so we run multiple attempts to increase the chance of triggering.
//!
//! The panic occurs in a rav1d-worker thread, not the calling thread, so
//! `catch_unwind` cannot intercept it. Instead we spawn decode in a child
//! thread and join with a timeout to detect panics or deadlocks.
//!
//! **Requires `--release`** — debug mode is too slow for decode tests.
//!
//! Run: cargo test --release --test tile_threading_overlap -- --ignored --nocapture

#[cfg(debug_assertions)]
compile_error!("tile_threading_overlap tests require release mode: cargo test --release");

use rav1d_safe::src::managed::{Decoder, Settings};
use std::time::Duration;

const OBU: &[u8] = include_bytes!("crash_vectors/disjoint_mut_tile_overlap.obu");

/// Outcome of a single decode attempt.
#[derive(Debug)]
enum DecodeOutcome {
    /// Decode completed (success or graceful error).
    Ok,
    /// Worker thread panicked (DisjointMut overlap) causing deadlock or join failure.
    WorkerPanic,
    /// Decode did not complete within the timeout (likely deadlocked after worker panic).
    Timeout,
}

/// Attempt a single decode with the given thread count.
///
/// The entire decode runs in a spawned thread so that worker-thread panics
/// can be detected via join timeout rather than hanging the test runner.
fn try_decode_with_threads(threads: u32, timeout: Duration) -> DecodeOutcome {
    let handle = std::thread::spawn(move || {
        let mut settings = Settings::default();
        settings.threads = threads;
        // max_frame_delay=1 disables frame threading, isolating tile threading.
        settings.max_frame_delay = 1;
        let mut decoder = Decoder::with_settings(settings).expect("create decoder");

        match decoder.decode(OBU) {
            Ok(Some(frame)) => {
                eprintln!(
                    "  threads={threads}: decoded {}x{} @ {}bpc",
                    frame.width(),
                    frame.height(),
                    frame.bit_depth()
                );
            }
            Ok(None) => match decoder.flush() {
                Ok(frames) => {
                    eprintln!("  threads={threads}: flushed {} frames", frames.len());
                }
                Err(e) => {
                    eprintln!("  threads={threads}: flush error: {e:?}");
                }
            },
            Err(e) => {
                eprintln!("  threads={threads}: decode error: {e:?}");
            }
        }
        // Explicit drop to join worker threads — this is where we detect worker panics.
        drop(decoder);
    });

    // Poll the join handle with a timeout.
    let start = std::time::Instant::now();
    loop {
        if handle.is_finished() {
            return match handle.join() {
                Ok(()) => DecodeOutcome::Ok,
                Err(_) => DecodeOutcome::WorkerPanic,
            };
        }
        if start.elapsed() >= timeout {
            return DecodeOutcome::Timeout;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
}

/// Single-threaded decode should complete without panic or deadlock.
#[test]
#[ignore]
fn single_threaded_no_panic() {
    eprintln!("OBU size: {} bytes", OBU.len());
    let outcome = try_decode_with_threads(1, Duration::from_secs(30));
    match outcome {
        DecodeOutcome::Ok => eprintln!("Single-threaded decode completed without panic."),
        other => panic!("Single-threaded decode failed unexpectedly: {other:?}"),
    }
}

/// Multi-threaded tile decode must not trigger DisjointMut overlap panic.
///
/// Previously, the loop filter V-pass at the bottom of sbrow N would
/// read/write pixels extending into the top rows of sbrow N+1. This
/// conflicted with concurrent TileReconstruction for sbrow N+1, causing
/// DisjointMut to (correctly) detect overlapping borrows and panic on a
/// worker thread, which deadlocked the decoder.
///
/// Fixed by adding a deblock progress barrier in check_tile: reconstruction
/// of sbrow N now waits until DeblockRows for sbrow N-1 completes.
#[test]
#[ignore]
fn multi_threaded_tile_overlap() {
    eprintln!("OBU size: {} bytes", OBU.len());

    let mut panic_or_timeout = 0;
    let mut ok_count = 0;
    let attempts = 10;
    // Per-attempt timeout: the decode itself is fast (~10ms for 700x400),
    // but a worker panic can deadlock the decoder drop, so we allow 10s.
    let timeout = Duration::from_secs(10);

    for attempt in 0..attempts {
        for threads in [4, 8] {
            eprintln!("Attempt {}/{attempts}, threads={threads}", attempt + 1);
            match try_decode_with_threads(threads, timeout) {
                DecodeOutcome::Ok => {
                    ok_count += 1;
                    eprintln!("  ok");
                }
                DecodeOutcome::WorkerPanic => {
                    panic_or_timeout += 1;
                    eprintln!("  WORKER PANIC (DisjointMut overlap)");
                }
                DecodeOutcome::Timeout => {
                    panic_or_timeout += 1;
                    eprintln!("  TIMEOUT (likely deadlocked after worker panic)");
                }
            }
        }
        // If we see any failures, stop early — the fix has regressed.
        if panic_or_timeout > 0 {
            break;
        }
    }

    let total = panic_or_timeout + ok_count;
    eprintln!("\nResults: {panic_or_timeout} failures, {ok_count} ok out of {total} attempts");

    assert_eq!(
        panic_or_timeout, 0,
        "DisjointMut overlap should not occur — deblock progress barrier prevents \
         concurrent reconstruction and loop filter V-pass on the same pixel rows"
    );
}
