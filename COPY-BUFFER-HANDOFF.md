# Copy-Buffer Threading Handoff

## Approach

Safe multithreading via `CopyGuard` — copies w×h pixels to a compact buffer
(stride=w, no gaps), SIMD operates on the buffer, per-row writeback on drop.

**Sound without unsafe. Zero disjoint-mut changes. Zero overhead without `mt` feature.**

## What's Done

### Infrastructure
- `src/copy_guard.rs` — `CopyGuard` type, `forbid(unsafe_code)`
- `with_dst!` macro in `mc.rs` — abstracts guard acquisition per `cfg(feature = "mt")`
- `mt` feature flag in `Cargo.toml`
- `slice_bytes` / `slice_mut_bytes` on `Rav1dPictureDataComponent` (mt-gated)
- `benches/thread_scaling.rs` — zenbench 1t/2t/3t/4t comparison

### Converted MC dispatch (mc.rs)
All use `with_dst!` macro — drop-in replacement for `full_guard_mut`:
- `avg_dispatch` ✅
- `w_avg_dispatch` ✅
- `mask_dispatch` ✅
- `blend_dispatch` ✅
- `blend_dir_dispatch` ✅
- `mc_put_dispatch` 8bpc ✅
- `mc_put_dispatch` 16bpc ❌ (complex u16 casting)
- `warp8x8_dispatch` ❌ (always 8×8, needs w=8 h=8)

### Loopfilter (loopfilter.rs)
- Level cache gather: per-entry guards behind `cfg(mt)` ✅
- Picture data guard: SB boundary fallback to scalar ✅

### Conformance
- Without mt: 784/784, 14.8s (matches main)
- With mt, single-threaded: 784/784, 17.2s
- Tile threading unit test (768×512 OBU): PASS at 0/2/4/8 threads

## What Remains

Every DSP dispatch function that holds a wide mutable guard on picture data
must be converted. The pattern is identical for each — mechanically apply
`with_dst!` or equivalent copy-buffer approach.

### Remaining `full_guard_mut` on picture data

| File | Function | Lines | Notes |
|------|----------|-------|-------|
| `mc.rs` | `mc_put_dispatch` 16bpc | ~12026 | u16 casting, `dst_stride/2` |
| `mc.rs` | `warp8x8_dispatch` | ~12694 | fixed 8×8, has src guard |
| `cdef.rs` | CDEF filter dispatch | multiple | reads + writes pixels |
| `ipred.rs` | intra prediction dispatch | ~4921 | writes predicted pixels |
| `ipred_arm.rs` | ARM intra prediction | ~1048 | same pattern |
| `loopfilter.rs` | picture data guard | ~1511 | has SB fallback but still wide |
| `looprestoration.rs` | Wiener/SGR dispatch | multiple | reads + writes filtered pixels |
| `filmgrain.rs` | film grain overlay | multiple | reads + writes with grain |
| `filmgrain_arm.rs` | ARM film grain | ~1482+ | same pattern |

### How to Convert Each

1. Find `dst.full_guard_mut::<BD>()` (or `dst.strided_slice_mut`)
2. Wrap the body in `with_dst!(dst, w, h, BD, |dst_bytes, dst_stride_u| { ... });`
3. Replace `&mut dst_bytes[dst_offset..]` → `dst_bytes`
4. Replace `dst_stride as usize` → `dst_stride_u`
5. For functions that also have `src.full_guard()`: src is immutable, keep as-is
   (immutable guards don't cause aliasing UB)
6. For 16bpc: the u16 reinterpretation needs `dst_stride_u / 2`

### Level cache overlaps

The loopfilter level cache (`f.lf.level`) has the same overlap pattern.
Per-entry gather (already done behind `cfg(mt)`) fixes this for the
loopfilter SIMD path. The scalar loopfilter path also reads the level
cache and may need similar treatment.

### src guard for frame threading

Frame threading (not tile threading) additionally requires narrow immutable
src guards when reading reference frames. The `narrow-guards` branch solved
this with `narrow_src_guard` — a clamped strided immutable guard. The
copy-buffer approach can use the same technique (immutable strided guards
are sound — `&[u8]` aliasing is fine).

## Performance

| Config | 768×512 OBU | Notes |
|--------|-------------|-------|
| Without mt | 14.8s (784 vectors) | Matches main |
| With mt, 1t | 17.2s | +16% from copy-buffer + LTO layout |
| With mt, tile threading | PASS unit test | Bench crashes on CDEF overlap |

The 16% mt overhead is from:
- Copy-buffer memcpy (~3% per-frame)
- Per-entry level cache gather (~5%)
- SB boundary scalar fallback (~3%)
- LTO codegen layout changes (~5%)

The `narrow-guards` branch achieved similar overhead (14-17%) but required
strided tracking in disjoint-mut, which had a soundness hole (stride gap
aliasing). The copy-buffer approach is provably sound.

## Key Design Decisions

1. **`with_dst!` macro** — avoids type-level enum for guard (CopyGuard vs
   DisjointMutGuard have different types). The macro provides `dst_bytes`
   and `dst_stride_u` to the body, hiding the guard type.

2. **Per-row writeback on Drop** — CopyGuard acquires h narrow mutable
   guards (one per row, w bytes each) during Drop. Each guard is
   acquired and released immediately. No stride gaps, no aliasing.

3. **SB boundary fallback** — loopfilter reaches span many SB rows.
   Copy-buffer for the full reach would be too large. Instead, fall
   back to scalar (per-row guards) when the reach crosses a 64-row
   SB boundary. Scalar loopfilter handles this natively.

4. **Feature flag** — `cfg(feature = "mt")` eliminates ALL copy-buffer
   code at compile time. Zero codegen impact when disabled.
