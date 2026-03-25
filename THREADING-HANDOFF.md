# Threading Handoff

## Status

Single-threaded: 784/784, correct, production-ready.
Multi-threaded: blocked on DisjointMut guard conflicts between tile threads.

## Experiment Branches (pushed)

### experiment/narrow-guards-threading
- Strided 2D overlap tracking in disjoint-mut
- **UNSOUND**: stride-gap aliasing (safe code can write to stride gaps that the tracker declared unaccessed)
- 17% single-threaded overhead (LTO codegen layout), 0% with feature flag
- Tile threading: passes all thread counts (768×512 and 4K/8K photos)
- Frame threading: passes
- AtomicLevelCache for lf_mask race
- Benchmark: 2t gives 1.4-1.55x speedup on photos

### experiment/copy-buffer-threading  
- CopyGuard: compact w×h buffer, per-row writeback on drop
- **Sound**: no unsafe, no disjoint-mut changes
- 28% single-threaded overhead, 0% with feature flag
- Tile threading: passes (768×512 OBU)
- Blocks on 4K+: immut strided_slice conflicts with mut CopyGuard writeback
- AtomicLevelCache for lf_mask race

## Root Cause

DisjointMut tracks borrows as contiguous byte ranges. Two tile threads accessing
non-overlapping pixel columns in the same row have guards whose byte ranges
overlap in the stride gaps. The tracker can't distinguish "I access columns 0-15"
from "I access bytes 0 through stride-1."

Additionally, immutable reads (strided_slice) hold wide multi-row guards that
conflict with mutable per-row writebacks from CopyGuard on adjacent SB rows.

## Approaches Evaluated

| Approach | Sound? | Overhead | 4K+ | Complexity |
|----------|--------|----------|-----|------------|
| Strided tracking | NO (stride gaps) | 17% | works | medium |
| unsafe strided tracking | yes (audited) | 17% | works | medium |
| Copy-buffer (writes) | yes | 28% | blocks on reads | high |
| Copy-buffer (reads+writes) | yes | untested | stride propagation nightmare | very high |
| Scheduler-owned guards | yes | ~0% | should work | high (new architecture) |
| unchecked feature | yes (trusted) | 0% | needs Default fix | trivial |

## Recommended Next Step: Scheduler-Owned Guards

Instead of each DSP function acquiring its own DisjointMut guard (bottom-up),
the thread scheduler pre-acquires guards for each SB row task (top-down):

1. Before `rav1d_decode_tile_sbrow`: scheduler acquires mutable guard for
   the SB row's pixel region (Y + UV planes)
2. Pre-acquired `&mut [BD::Pixel]` slice passed down through the call chain
3. DSP functions index into the pre-acquired slice — zero guard overhead
4. After SB row completes: guard dropped

### Why this works
- Scheduler already ensures non-overlapping SB row assignments
- Guard acquisition cost: O(1) per SB row instead of O(N) per DSP call
- No stride-gap aliasing: the guard covers the exact SB row region
- No immut+mut conflict: reads from adjacent SBs use narrow immut guards
  acquired at the scheduler level, released before the writing thread starts

### What needs investigation
- Trace every picture access in TileReconstruction task type
- Trace every picture access in DeblockRows, Cdef, LoopRestoration tasks
- Determine how to pass pre-acquired guards through PicOffset abstraction
- Handle reference frame reads (different DisjointMut instance)
- Handle negative strides (guard base offset)
- Handle cross-SB reads for loopfilter/CDEF/ipred boundary access

### Quick win: fix DisjointMut Default + unchecked
The `unchecked` feature already disables tracking for DisjointMut created via
`dm_new()`, but `#[derive(Default)]` uses `DisjointMut::new()` (always tracked).
One-line fix in disjoint-mut crate: make Default respect the `unchecked` feature.
Then `--features unchecked` enables threading with zero overhead.

This is the FASTEST path to working multithreading but requires `unchecked` 
(which also disables bounds checking in SIMD dispatch). A separate `mt-unchecked`
feature could enable untracked DisjointMut without disabling bounds checks.
