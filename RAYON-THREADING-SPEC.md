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

// AFTER: pre-split row slices typed by pixel depth
// BD::Pixel is u8 for 8bpc, u16 for 10/12bpc
// Each row is exactly `width` pixels — no stride padding, no gaps
// Alignment: inherited from the aligned frame allocation (64-byte aligned base,
// stride is a multiple of 64). Row slice start = base + row * stride, which
// preserves alignment. The [BD::Pixel] element type additionally guarantees
// natural alignment (1-byte for u8, 2-byte for u16).
type RowSlices<'a, BD: BitDepth> = &'a mut [&'a mut [BD::Pixel]];
```

Frame buffer pre-split before decode:
```rust
// Frame allocated with 64-byte alignment (AlignedVec64), stride is 64-aligned.
// BD::Pixel determines element type and natural alignment.
fn split_frame<BD: BitDepth>(
    frame_buf: &mut [BD::Pixel],
    stride: usize,   // in pixels (= byte_stride / size_of::<BD::Pixel>())
    width: usize,    // in pixels
    height: usize,
) -> Vec<&mut [BD::Pixel]> {
    frame_buf.chunks_mut(stride)
        .take(height)
        .map(|row| &mut row[..width])
        .collect()
}
// Each resulting row slice:
// - Points into the aligned frame buffer (alignment preserved)
// - Is exactly `width` pixels (no stride gap at the end)
// - Type [BD::Pixel] ensures natural pixel alignment
// - Independently borrowable via split_at_mut
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
// Current inner SIMD: flat &mut [u8] + byte offset + byte stride
fn avg_inner(dst: &mut [u8], offset: usize, stride: usize, ...);

// New: pixel-typed row slices + column offset
fn avg_inner<BD: BitDepth>(
    dst_rows: &mut [&mut [BD::Pixel]],  // h rows, each ≥ x+w pixels
    x: usize,                           // column offset in pixels
    ...,
);
```

SIMD within a row is unchanged — each row slice is contiguous in memory.
The `BD::Pixel` type (u8 or u16) gives natural alignment and eliminates
manual `pixel_size` multiplication. SIMD functions cast `&mut [BD::Pixel]`
to `&mut [u8]` for byte-level intrinsic operations (safe via zerocopy
`IntoBytes`/`FromBytes`).

For reference frame reads (immutable):
```rust
fn mc_put_inner<BD: BitDepth>(
    dst_rows: &mut [&mut [BD::Pixel]],  // mutable destination rows
    dst_x: usize,
    src_rows: &[&[BD::Pixel]],          // immutable reference frame rows
    src_x: usize,
    ...,
);
```

## Reference Frames

After decode completes all filters, the frame buffer is frozen for reference use:
```rust
// During decode: Vec<BD::Pixel> owned mutably, split into row slices
// After decode: freeze into Arc for immutable sharing across frames
let ref_frame: Arc<Vec<BD::Pixel>> = Arc::new(frame_buf);

// Next frame's MC reads: &[BD::Pixel] row slices (immutable, no DisjointMut)
let ref_rows: Vec<&[BD::Pixel]> = ref_frame.chunks(stride)
    .take(height)
    .map(|row| &row[..width])
    .collect();
```
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

## Appendix A: PicOffset Call Chain Analysis

### PicOffset Type

`PicOffset<'a>` = `WithOffset<&'a Rav1dPictureDataComponent>` = `{ data, offset }`.
Offset is in **pixels** (not bytes). Stride is in pixels via `pixel_stride::<BD>()`.
PicOffset is `Copy` — passed by value everywhere, arithmetic creates new offsets.

### Creation Points (recon.rs)

**Y plane:**
```rust
let y_dst = cur_data[0].with_offset::<BD>()
    + 4 * (t.b.y as isize * stride + t.b.x as isize);
// 4* because block coords are 4-pixel units
```

**UV planes:**
```rust
let uv_dst = cur_data[1 + pl].with_offset::<BD>()
    + 4 * ((t.b.x >> ss_hor) as isize + (t.b.y >> ss_ver) as isize * uv_stride);
```

**Loopfilter:**
```rust
let p: [PicOffset; 3] = f.cur.lf_offsets::<BD>(y); // Y, U, V at row y
```

### Call Depth Map

```
recon.rs: creates PicOffset, passes to DSP
  ├── mc.rs: .call() → *_direct() → *_dispatch() → *_dispatch_inner()
  │     └── safe_simd/mc.rs: acquires guard, extracts &mut [u8], calls SIMD
  ├── itx.rs: .call() → inv_txfm_add()
  │     └── safe_simd/itx.rs: acquires guard, runs SIMD on pixels
  ├── ipred.rs: .call() → intra_pred_*()
  │     └── safe_simd/ipred.rs: acquires guard, fills prediction block
  ├── loopfilter.rs: loopfilter_sb() → loopfilter_sb_dispatch()
  │     └── safe_simd/loopfilter.rs: acquires pixel guard + level guard
  └── cdef_apply.rs: rav1d_cdef_brow() → backup2lines() + cdef_filter()
        └── safe_simd/cdef.rs: acquires guard, applies filter
```

### Pixel Access Patterns

**Per-row slice** (most common):
```rust
for y in 0..h {
    let row = dst + y as isize * stride;
    let pixels = &mut *row.slice_mut::<BD>(w);  // guard acquired
    // ... process pixels[0..w]
}  // guard dropped
```

**Per-pixel index** (filter_8tap, edge access):
```rust
*(src + x + y * stride).index::<BD>()  // single-pixel guard
```

**Block guard** (SIMD dispatch):
```rust
let (guard, base) = dst.narrow_guard_mut::<BD>(w, h);
let bytes = guard.as_mut_bytes();
simd_fn(&mut bytes[base..], stride, ...);
```

### Row-Slice Replacement Strategy

Every PicOffset access follows one of three patterns:
1. **Row loop**: `for y in 0..h { dst + y*stride → slice(w) }` → `dst_rows[y][x..x+w]`
2. **Pixel index**: `(dst + x + y*stride).index()` → `dst_rows[y][x]`
3. **Block guard**: `dst.narrow_guard_mut(w,h)` → `&mut dst_rows[y_start..y_end]`

All three map naturally to `&mut [&mut [BD::Pixel]]` row slices.

## Appendix B: Functions to Port (Phase 1 Scalar)

### Must Port (reconstruction hot path)

| Function | File | Pattern | Priority |
|----------|------|---------|----------|
| `put_rust` | mc.rs:56 | row loop, copy | P0 |
| `prep_rust` | mc.rs:67 | row loop, shift | P0 |
| `put_8tap_rust` | mc.rs:139 | pixel index, filter | P0 |
| `put_bilin_rust` | mc.rs:398 | pixel index, filter | P0 |
| `avg_rust` | mc.rs:482 | row loop, avg | P0 |
| `w_avg_rust` | mc.rs:497 | row loop, weighted avg | P0 |
| `mask_rust` | mc.rs:514 | row loop, mask blend | P0 |
| `inv_txfm_add` | itx.rs:68 | row loop, coeff add | P0 |
| `dc_pred` | ipred.rs | row loop, fill | P1 |
| `v_pred` | ipred.rs | row loop, copy top | P1 |
| `h_pred` | ipred.rs | row loop, fill left | P1 |
| `paeth_pred` | ipred.rs | pixel index, predict | P1 |

### Must Port (filter path)

| Function | File | Pattern | Priority |
|----------|------|---------|----------|
| `loop_filter_4_8bpc` | loopfilter.rs | pixel index, ±3 rows | P1 |
| `backup2lines` | cdef_apply.rs | row copy, 2 rows | P1 |
| `cdef_filter_block` | cdef.rs | pixel index, ±2 | P2 |
| `sgr_filter` | looprestoration.rs | pixel index, window | P2 |
| `wiener_filter` | looprestoration.rs | pixel index, 7-tap | P2 |

### Unchanged (no picture access)

- `msac.rs` — entropy decoding, no pixel access
- `decode.rs` — bitstream parsing, creates PicOffset but doesn't access pixels directly
- `refmvs.rs` — motion vector management

## Appendix C: Implementation Notes

### FrameRows Type

```rust
/// Pre-split frame buffer for one plane.
/// Each row is exactly `width` pixels — no stride padding.
/// Alignment: rows point into a 64-byte-aligned allocation.
pub struct PlaneRows<'a, Pixel> {
    rows: Vec<&'a mut [Pixel]>,
    width: usize,
    height: usize,
}

impl<'a, Pixel> PlaneRows<'a, Pixel> {
    /// Split a flat buffer into row slices.
    pub fn from_buf(buf: &'a mut [Pixel], stride: usize, width: usize, height: usize) -> Self {
        let rows = buf.chunks_mut(stride)
            .take(height)
            .map(|row| &mut row[..width])
            .collect();
        Self { rows, width, height }
    }

    /// Get mutable rows for an SB row (sby * sb_height .. (sby+1) * sb_height).
    pub fn sb_rows(&mut self, sby: usize, sb_height: usize) -> &mut [&mut [Pixel]] {
        let start = sby * sb_height;
        let end = (start + sb_height).min(self.height);
        &mut self.rows[start..end]
    }

    /// Split into two at a row boundary (for split_at_mut ownership transfer).
    pub fn split_at_row(&mut self, row: usize) -> (&mut [&mut [Pixel]], &mut [&mut [Pixel]]) {
        self.rows.split_at_mut(row)
    }
}
```

### Block Access Within Row Slices

```rust
/// Access a w×h block starting at (x, y) within row slices.
fn block_rows<'a, P>(
    rows: &'a mut [&'a mut [P]],
    x: usize,
    y: usize,
    w: usize,
    h: usize,
) -> impl Iterator<Item = &'a mut [P]> {
    rows[y..y+h].iter_mut().map(move |row| &mut row[x..x+w])
}
```

### Negative Stride Handling

rav1d uses negative strides for bottom-to-top picture layout. With row slices,
this becomes row index reversal:
```rust
fn row_index(y: usize, height: usize, negative_stride: bool) -> usize {
    if negative_stride { height - 1 - y } else { y }
}
```
The row slices themselves are always left-to-right (positive). Only the Y index
is flipped.

### Chroma Subsampling

For YUV420: chroma has half width and half height.
```rust
let y_rows = PlaneRows::from_buf(&mut y_buf, y_stride, width, height);
let u_rows = PlaneRows::from_buf(&mut u_buf, uv_stride, width >> ss_hor, height >> ss_ver);
let v_rows = PlaneRows::from_buf(&mut v_buf, uv_stride, width >> ss_hor, height >> ss_ver);
```

### First Implementation Target

Start with `put_rust` — the simplest MC function (pixel copy):

```rust
// Current:
fn put_rust<BD: BitDepth>(dst: PicOffset, src: PicOffset, w: usize, h: usize) {
    for y in 0..h {
        let src = src + y as isize * src.pixel_stride::<BD>();
        let dst = dst + y as isize * dst.pixel_stride::<BD>();
        BD::pixel_copy(&mut *dst.slice_mut::<BD>(w), &*src.slice::<BD>(w), w);
    }
}

// Row-slice version:
fn put_rows<BD: BitDepth>(
    dst_rows: &mut [&mut [BD::Pixel]],
    dst_x: usize,
    src_rows: &[&[BD::Pixel]],
    src_x: usize,
    w: usize,
) {
    for (dst_row, src_row) in dst_rows.iter_mut().zip(src_rows.iter()) {
        dst_row[dst_x..dst_x+w].copy_from_slice(&src_row[src_x..src_x+w]);
    }
}
```
