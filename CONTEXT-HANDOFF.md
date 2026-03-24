# Context Handoff: DisjointMut Strided Guard Support

## Goal
Enable multi-threaded tile decoding in rav1d-safe by adding strided guard support to DisjointMut, so SIMD functions can lock only the pixels they actually access within a strided region.

## Problem
`strided_slice_mut(w=8, h=8)` at stride=768 creates a contiguous guard `[offset..offset+5384]` — covering 5384 bytes for 64 actually-used pixels. Two tile threads working on adjacent SB rows create overlapping guards because the stride gap includes pixels from other columns.

## What's Already Fixed
- `f.lf.level` loopfilter level cache overlap (commit e2de9f1) — stack-gathered entries
- Scalar paths work perfectly with tile threading (narrow per-row guards)

## What's Needed
Add a `StridedBorrow` variant to `BorrowTracker` in `crates/rav1d-disjoint-mut/src/lib.rs` that tracks `(start, w, h, stride)` and checks overlap against the actual w-byte ranges at each stride interval, not the full contiguous span.

### Key Files
- `crates/rav1d-disjoint-mut/src/lib.rs` — BorrowTracker, overlap checking (lines 1115-1337)
- `include/dav1d/picture.rs:610` — `strided_slice_mut` creates the guards
- `include/dav1d/picture.rs:640` — `strided_slice` (immutable version)
- `src/safe_simd/mc.rs` — 15 `full_guard` calls that could use strided guards instead
- `benches/threading_bench.rs` — benchmark to measure impact
- `tests/reproduce_overlap.rs` — reproducer test (panics with SIMD tile threading)
- `tests/reproduce_overlap_scalar.rs` — scalar tile threading (passes, confirms fix target)

### Overlap Check Algorithm
Current: contiguous range `[start..end)` overlap check — `a.start < b.end && b.start < a.end`

Needed: strided ranges. Two strided borrows `A(start_a, w_a, h_a, stride_a)` and `B(start_b, w_b, h_b, stride_b)` overlap iff any row range from A overlaps any row range from B:
- A's row i: `[start_a + i*stride_a, start_a + i*stride_a + w_a)`
- B's row j: `[start_b + j*stride_b, start_b + j*stride_b + w_b)`

For the common case (same stride): rows overlap when `i*stride + w_a > j*stride` and `j*stride + w_b > i*stride`, which simplifies to checking if the row indices overlap in the vertical dimension.

For same-stride, the check reduces to:
1. Do the vertical row ranges overlap? (row_start_a..row_start_a+h_a vs row_start_b..row_start_b+h_b)
2. Do the horizontal pixel ranges overlap? (col_start_a..col_start_a+w_a vs col_start_b..col_start_b+w_b)

This is a 2D rectangle overlap test when strides are equal, which they always are for the same picture plane.

### Prior Art Search
No existing strided guard implementation found in zen/. The DisjointMut crate has:
- `PicBuf` (line 1680) — owned buffer, no strided tracking
- `instrument.rs` — 1D borrow size histograms, no strided support
- `pixel_access.rs:540` — `strided_slice_from_ptr` (unsafe raw pointer helper, no guards)
- `picture.rs:610` — `strided_slice_mut` creates contiguous guards via `(h-1)*stride+w`

The `BorrowSlots` uses parallel arrays (SoA layout) for O(popcount) scanning:
```
starts: [usize; 64], ends: [usize; 64], mutable: [bool; 64], occupied: u64
```

### Implementation Plan
Add `strides: [usize; 64]` and `widths: [usize; 64]` to `BorrowSlots`. A stride of 0 means "contiguous range" (current behavior, backwards compatible). When stride > 0, the borrow covers rows `[start + i*stride .. start + i*stride + width)` for `i` in `0..((end - start - width) / stride + 1)`.

Overlap check: when both borrows have the same non-zero stride, decompose into 2D rectangle overlap:
- Vertical: `row_a_start..row_a_end` vs `row_b_start..row_b_end` (in stride units)
- Horizontal: `col_a_start..col_a_start+w_a` vs `col_b_start..col_b_start+w_b`
This is O(1) — no per-row iteration needed.

When strides differ or one is 0: fall back to contiguous range check (conservative, may false-positive).

### New API
```rust
// In picture.rs, replace:
self.data.slice_mut::<BD, _>((self.offset.., ..total))
// With:
self.data.strided_slice_mut_tracked::<BD>(self.offset, w, h, stride)
// Which registers (start, start + (h-1)*stride + w, mutable, stride, w)
```

### Performance Constraint
Guard acquisition is on every DSP function call (~1000s per frame). The overlap check must be fast — ideally O(1) per existing borrow, not O(h_a * h_b). The 2D rectangle test for same-stride is O(1).

### Benchmark Baseline
```
Single-threaded, checked mode, x86_64 AVX2:
  2K: SIMD 65ms, Scalar 96ms (1.47x)
  4K: SIMD 233ms, Scalar 351ms (1.50x)
  8K: SIMD 1071ms, Scalar 1697ms (1.58x)
```

After implementing strided guards, run `cargo bench --bench threading_bench` and compare. Then add tile-threaded SIMD configs to the benchmark.

### Testing
1. `cargo test --release --test reproduce_overlap -- --ignored` — must pass (currently panics)
2. `cargo test --release --test integration_decode -- --ignored` — 784/784 conformance vectors
3. `cargo bench --bench threading_bench` — no regression vs baseline
