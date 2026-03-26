# rav1d-rayon: Sound Multithreaded Decode via Rayon Scopes

## Goal

Replace DisjointMut-based threading with rayon scoped parallelism and `split_at_mut`
ownership splitting. No `unsafe`, no `unchecked`, no runtime borrow tracking on
picture planes. The borrow checker proves non-overlap at compile time.

## Problem

rav1d's C-inherited design: all threads share raw pointers to a flat frame buffer.
DisjointMut provides runtime tracking, but stride gaps and cross-SB MC blocks cause
both false-positive and real overlaps. `unchecked` disables tracking entirely.

## Core Type Change

```rust
// BEFORE: offset into flat DisjointMut buffer
struct PicOffset<'a> { data: &'a DisjointMut<PicBuf>, offset: usize }

// AFTER: pre-split row slices, no DisjointMut for picture planes
// Each row is exactly width pixels — no stride padding, no gaps
type RowSlices<'a> = &'a mut [&'a mut [u8]];
```

Frame buffer pre-split before decode:
```rust
let rows: Vec<&mut [u8]> = frame_buf.chunks_mut(stride)
    .take(height)
    .map(|row| &mut row[..width * pixel_size])
    .collect();
```

## AV1 Task Pipeline per SB Row

| Phase | Writes | Reads (cur frame) | Reads (other) | Row Reach |
|-------|--------|--------------------|---------------|-----------|
| Reconstruction | Y/U/V [sby*64..(sby+1)*64] | ipred top edge (1 row) | ref frames (immutable) | SB + MC overflow |
| backup_ipred_edge | ipred_edge scratch | last row of SB | — | 1 row |
| DeblockCols | Y/U/V ±1px | Y/U/V ±4px | lf.level | ±4 rows |
| DeblockRows + CopyLPF | Y/U/V ±1px, cdef_line_buf | Y/U/V ±3px | lf.level | ±3 rows |
| CDEF | Y/U/V [SB rows] | cdef_line_buf (boundary backup) | — | SB only |
| LoopRestoration | Y/U/V [SB rows] | lr_lpf_line (pre-filter snapshot) | — | SB only |

## Dependencies

```
Recon(N) ──→ Deblock(N) ──→ CDEF(N) ──→ LR(N)
    │              │
    ↓              ↓ copy_lpf(N)
Recon(N+1)         CDEF(N+1) waits for copy_lpf(N)
```

Recon(N+1) starts when Recon(N) completes. Filters pipeline behind.

## Cross-SB Boundary Access

**MC overflow**: Blocks can span SB boundaries (24-row block at row 105 writes
through row 128). Solution: extend thread's write zone by overflow rows from
the next SB chunk. Return overflow before next thread starts.

**Filter reach**: Deblock ±3 rows, CDEF from backup buffers. Solution: copy
boundary rows to scratch before releasing SB row ownership.

**Ipred edge**: Reads 1 row from SB above. Already handled by backup_ipred_edge.

## Rayon Scope Architecture

```rust
fn decode_frame(frame_buf: &mut [u8], ...) {
    let pixel_rows: Vec<&mut [u8]> = pre_split(frame_buf);

    pool.scope(|s| {
        // Phase 1: Reconstruction (parallel, pipelined via channels)
        let mut prev_done = None;
        for sby in 0..num_sb {
            let (tx, rx) = channel();
            let chunk = take_sb_chunk(&mut pixel_rows, sby, overflow);
            let wait = prev_done.take();
            s.spawn(move |_| {
                if let Some(rx) = wait { rx.recv().unwrap(); }
                reconstruct_sb_row(chunk, sby, ref_frames, scratch);
                copy_boundary(chunk, sby, scratch);
                tx.send(()).unwrap();
            });
            prev_done = Some(rx);
        }
    });

    pool.scope(|s| {
        // Phase 2: Filters (sequential per SB, pipelined)
        let mut prev_done = None;
        for sby in 0..num_sb {
            let (tx, rx) = channel();
            let rows = filter_region(&mut pixel_rows, sby, margin);
            let wait = prev_done.take();
            s.spawn(move |_| {
                if let Some(rx) = wait { rx.recv().unwrap(); }
                deblock(rows, sby);
                backup2lines(rows, sby, &cdef_scratch);
                cdef(rows, sby, &cdef_scratch);
                lr(rows, sby, &lr_scratch);
                tx.send(()).unwrap();
            });
            prev_done = Some(rx);
        }
    });
}
```

## DSP Function Interface

```rust
// Current inner SIMD: flat &mut [u8] + offset + stride
fn avg_inner(dst: &mut [u8], offset: usize, stride: usize, ...);

// New: row slices directly
fn avg_inner(dst_rows: &mut [&mut [u8]], x: usize, ...);
```

SIMD within a row is unchanged — each row slice is contiguous.

## Reference Frames

After decode, freeze: `Arc::new(frame_buf)`. MC reads via `&[u8]`.
Frame threading: N+1 reads N's fully-filtered rows via atomic per-row flags.

## Level Cache

Per-SB-row level buffer (restructure to eliminate cross-row access) for full
soundness. Or: keep DisjointMut for level cache only (minimal scope).

## Migration Path

### Phase 1: Scalar validation
- `FrameRows` type alongside existing `PicOffset`
- Row-slice scalar MC, ITX, ipred, loopfilter with `#[inline(always)]`
- Validate: 766/768 at scalar CPU level

### Phase 2: autoversion SIMD
- `#[autoversion(avx2, neon)]` on scalar row-slice functions
- Validate: conformance at all CPU levels

### Phase 3: Hand-tuned critical paths
- Profile, port top ~5 dispatch functions with hand SIMD
- Target: match current PicOffset SIMD performance

### Phase 4: Remove PicOffset
- Delete DisjointMut from picture planes
- Remove `unchecked` dependency for threading

## Verification

1. Scalar conformance: 766/768
2. SIMD conformance: 766/768 at all CPU levels
3. Threading: 6/6 tests pass WITHOUT unchecked
4. zenbench 4K: 1t ≤249ms, 2t ~125ms, 4t ~66ms
5. Miri: no UB in single-threaded decode
6. No `unchecked` required for any threading mode
