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

---

# Data Ownership Architecture

This section documents every data access pattern in the decode pipeline — the exact
rows, columns, and buffers each function reads and writes. This is the ground truth
for proving that rayon scoped ownership can replace DisjointMut.

## Frame Buffer Layout

### Allocation

```
Rav1dPictureData.data: [Rav1dPictureDataComponent; 3]  // Y, U, V
```

Each plane is a separate contiguous buffer (not interleaved):
- **Base alignment**: 64-byte (AlignedVec64)
- **Stride**: in bytes, always a multiple of 64; converted to pixels via
  `pixel_stride::<BD>() = byte_stride / size_of::<BD::Pixel>()`
- **Negative stride**: supported (buffer pointer adjusted so row 0 is at
  `base + (height - 1) * |stride|`, stride is negative isize)

### Per-Plane Dimensions

| Plane | Width (pixels) | Height (pixels) | Stride (pixels) |
|-------|----------------|-----------------|------------------|
| Y     | `f.bw << 2`   | `f.bh << 2`     | `y_stride`       |
| U     | `(f.bw << 2) >> ss_hor` | `(f.bh << 2) >> ss_ver` | `uv_stride` |
| V     | `(f.bw << 2) >> ss_hor` | `(f.bh << 2) >> ss_ver` | `uv_stride` |

Subsampling: `ss_hor = (layout != I444)`, `ss_ver = (layout == I420)`.

### SB Grid

- SB size: 64×64 (sb128=0) or 128×128 (sb128=1)
- SB rows: `f.sbh` rows, SB row `sby` covers pixel rows `[sby * sb_step, min((sby+1) * sb_step, height))`
- `sb_step` = 16 (in 4-pixel block units) for SB64, 32 for SB128
- Pixel row range for SB row `sby`: `[sby << (6 + sb128), min((sby + 1) << (6 + sb128), f.cur.p.h))`

### PicOffset Formula

All pixel access goes through `PicOffset { data, offset }` where offset is in pixels:
```rust
// Block (bx, by) in 4-pixel units → pixel offset:
let offset = base_offset + 4 * (by as isize * pixel_stride + bx as isize);
// For negative stride: base_offset positions row 0 at the buffer end
```

Row-slice equivalent: `rows[4*by + y][4*bx + x]`

---

## Phase 1: Reconstruction — Exact Access Patterns

### 1.1 Motion Compensation (MC)

All MC functions: **dst** is current frame (write), **src** is reference frame (read-only).

#### put_rust — Fullpel pixel copy
- **Write**: `dst[0..h][0..w]` — `h` rows, `w` pixels per row
- **Read src**: `src[0..h][0..w]` — same dimensions, no overflow
- **Block sizes**: 2–128 pixels in each dimension

#### prep_rust — Fullpel preparation to i16 temp
- **Write**: `tmp[0..h*w]` (i16 temp buffer, not picture plane)
- **Read src**: `src[0..h][0..w]` — no overflow
- **No picture write** (output to temp buffer only)

#### put_8tap_rust — Subpel 8-tap filtered put
Three conditional paths based on `mx` (horizontal subpel) and `my` (vertical subpel):

| Condition | Read Region (src) | Intermediate | Write Region (dst) |
|-----------|--------------------|--------------|---------------------|
| mx≠0, my≠0 | `src[-3..h+4][-3..w+4]` | `mid[0..h+7][0..w]` i16 | `dst[0..h][0..w]` |
| mx≠0, my=0 | `src[0..h][-3..w+4]` | none | `dst[0..h][0..w]` |
| mx=0, my≠0 | `src[-3..h+4][0..w]` | none | `dst[0..h][0..w]` |
| mx=0, my=0 | delegates to `put_rust` | — | — |

**Read overflow**: **3 rows above, 4 rows below** (vertical); **3 pixels left, 4 pixels right** (horizontal).
Filter is 8 taps at positions [-3, -2, -1, 0, 1, 2, 3, 4] relative to the integer position.

The intermediate buffer `mid` is stack-allocated: `[[i16; 128]; 135]` (MID_STRIDE=128).

#### prep_8tap_rust — Subpel 8-tap filtered prep
- Identical read pattern to `put_8tap_rust`
- **Write**: `tmp[0..h*w]` i16 temp buffer (no picture write)

#### put_bilin_rust — Bilinear 2-tap filtered put

| Condition | Read Region (src) | Write Region (dst) |
|-----------|--------------------|--------------------|
| mx≠0, my≠0 | `src[0..h+1][0..w+1]` | `dst[0..h][0..w]` |
| mx≠0, my=0 | `src[0..h][0..w+1]` | `dst[0..h][0..w]` |
| mx=0, my≠0 | `src[0..h+1][0..w]` | `dst[0..h][0..w]` |
| mx=0, my=0 | delegates to `put_rust` | — |

**Read overflow**: **1 pixel right** (horizontal); **1 row below** (vertical).
Intermediate buffer for both-subpel case: `[[i16; 128]; 129]`.

#### prep_bilin_rust — Bilinear prep
- Identical read pattern to `put_bilin_rust`
- **Write**: `tmp[0..h*w]` i16 buffer

#### put_8tap_scaled_rust — Scaled 8-tap
- **Read src**: Variable — depends on `dx`, `dy` scale factors (8.10 fixed-point)
  - Intermediate buffer: `[[i16; 128]; 263]` (256 + 7 rows)
  - Horizontal reads: `src[y-3..y+5]` per intermediate row, columns vary per pixel via `dx`
  - Vertical reads: up to 8 intermediate rows per output row, varying via `dy`
- **Write**: `dst[0..h][0..w]`
- **Max read overflow**: 3 above, 4 below, 3 left, variable right (up to `src_w`)

#### prep_8tap_scaled_rust — Scaled 8-tap prep
- Same read pattern as scaled put; output to i16 temp buffer

#### put_bilin_scaled_rust / prep_bilin_scaled_rust — Scaled bilinear
- Read overflow: 1 row/pixel in each direction (scaled)
- Intermediate: `[[i16; 128]; 257]` (256 + 1)

#### avg_rust — Average two i16 temps → pixels
- **Write**: `dst[0..h][0..w]` (picture plane)
- **Read**: `tmp1[0..h*w]`, `tmp2[0..h*w]` (i16 temp buffers, not picture)
- **No src picture read**

#### w_avg_rust — Weighted average
- Same as `avg_rust` plus `weight: i32` parameter

#### mask_rust — Mask-blended compound
- **Write**: `dst[0..h][0..w]`
- **Read**: `tmp1[0..h*w]`, `tmp2[0..h*w]`, `mask[0..h*w]` (u8 weights)

#### w_mask_rust — Weighted mask compound (dual output)
- **Write 1**: `dst[0..h][0..w]` (picture plane)
- **Write 2**: `mask[0..(w>>ss_hor)*(h>>ss_ver)]` (segmentation mask buffer)
- **Read**: `tmp1[0..h*w]`, `tmp2[0..h*w]`

#### blend_rust — OBMC blending (read-modify-write)
- **Read+Write**: `dst[0..h][0..w]` (reads existing prediction, blends with tmp)
- **Read**: `tmp[0..h*w]` (BD::Pixel temp), `mask[0..h*w]` (OBMC weights)

#### blend_v_rust — Vertical OBMC
- **Read+Write**: `dst[0..h][0..w*3/4]` (only writes leftmost 3/4 of block)
- **Read**: `tmp[0..h*(w*3/4)]`, `dav1d_obmc_masks[w..]` (constant table)

#### blend_h_rust — Horizontal OBMC
- **Read+Write**: `dst[0..h*3/4][0..w]` (only writes top 3/4 of block)
- **Read**: `tmp[0..(h*3/4)*w]`, `dav1d_obmc_masks[h..]`

#### emu_edge_rust — Edge extension for boundary blocks
- **Read src (reference frame)**: Clipped to `[max(0,y)..min(y+bh,ih)][max(0,x)..min(x+bw,iw)]`
  - Extends edge pixels for out-of-bounds regions
- **Write**: `dst[0..bh][0..bw]` — stack buffer `[BD::Pixel; 320 × 263]`, NOT picture plane
- **Not a picture write** — output fed to MC functions as source

#### resize_rust — Super-resolution horizontal resize
- **Read src**: `src[0..h][0..src_w]` — each output pixel reads 8 source pixels via
  resize filter, clipped to `[0, src_w-1]`
- **Write**: `dst[0..h][0..dst_w]` (picture or scratch buffer)
- No vertical overflow (one source row per output row)

#### warp_affine_8x8_rust — Affine warp (fixed 8×8 block)
- **Write**: `dst[0..8][0..8]` (always 8×8)
- **Read src**: Per-pixel affine position, but bounded by:
  - Intermediate: `mid[0..15][0..8]` (15 rows × 8 cols)
  - Source reads: `src[-3..12][-3..12]` worst case (15 rows, extends ±3 for 8-tap warp filter)
  - Filter taps from `dav1d_mc_warp_filter` indexed by fractional affine position
- **Read overflow**: ±3 pixels/rows from block bounds in both directions

#### warp_affine_8x8t_rust — Affine warp prep
- Same read pattern, output to `tmp[0..8*8]` i16 buffer

### MC Access Summary

| Function | Src Overflow (rows) | Src Overflow (cols) | Dst Rows | Dst Cols |
|----------|-------|-------|----------|----------|
| put/prep_rust | 0 | 0 | h | w |
| put/prep_8tap | -3..+4 | -3..+4 | h | w |
| put/prep_bilin | 0..+1 | 0..+1 | h | w |
| put/prep_8tap_scaled | -3..+4 (var) | -3..+var | h | w |
| avg/w_avg/mask | N/A (temp) | N/A (temp) | h | w |
| blend | 0 | 0 | h | w (R+W) |
| blend_v | 0 | 0 | h | w*3/4 (R+W) |
| blend_h | 0 | 0 | h*3/4 | w (R+W) |
| emu_edge | clipped | clipped | bh | bw (temp) |
| warp_affine | -3..+12 | -3..+12 | 8 | 8 |

**Key for rayon**: MC only writes to the *current* SB row's pixel region in dst.
All src reads are from *reference frames* (immutable `Arc`). The emu_edge and
intermediate buffers are per-thread stack allocations.

### 1.2 Inverse Transform (ITX)

#### inv_txfm_add — Add residual to prediction
- **Read+Write dst**: `dst[0..h][0..w]` — reads prediction, adds transformed coefficients
- **Read coeff**: `coeff[0..min(h,32)*min(w,32)]` (quantized residual, zeroed after use)
- **Internal temp**: `tmp[0..64*64]` i32 stack buffer
- **Block sizes**: w,h ∈ {4, 8, 16, 32, 64}
- **DC-only fast path** (eob=0): reads `coeff[0]`, adds constant to all `h×w` dst pixels

**No cross-block or cross-SB access.** ITX is completely self-contained within the
transform block.

### 1.3 Intra Prediction (ipred)

All ipred functions: write `dst[0..h][0..w]`, read from `topleft[offset ± ...]` edge buffer.
The edge buffer is a 257-element scratch array prepared by `rav1d_prepare_intra_edges`.

#### Edge Buffer Layout (topleft_out, center at offset 128)

```
 Indices:  ... [off-2h] ... [off-h] ... [off-1] [off] [off+1] ... [off+w] ... [off+2w]
 Content:  ... bottom-left ext ... left col ... corner  top row ... top-right ext ...
```

#### Per-Mode Edge Access

| Mode | Top row | Left col | Top-left | Extended edges | dst self-read |
|------|---------|----------|----------|----------------|---------------|
| DC_PRED | `[off+1..off+1+w]` | `[off-1..off-h]` | no | no | no |
| TOP_DC | `[off+1..off+1+w]` | no | no | no | no |
| LEFT_DC | no | `[off-1..off-h]` | no | no | no |
| DC_128 | no | no | no | no | no |
| V_PRED | `[off+1..off+1+w]` | no | no | no | no |
| H_PRED | no | `[off-1..off-h]` | no | no | no |
| PAETH | `[off+1..off+1+w]` | `[off-1..off-h]` | `[off]` | no | no |
| SMOOTH | `[off+1..off+1+w]` | `[off-1..off-h]` | no | `[off+w]`, `[off-h]` | no |
| SMOOTH_V | `[off+1..off+1+w]` | no | no | `[off-h]` | no |
| SMOOTH_H | no | `[off-1..off-h]` | no | `[off+w]` | no |
| Z1 | `[off+1..off+1+w+h]` | no | no | upsampled/filtered | no |
| Z2 | `[off+1..off+1+w]` | `[off-1..off-h]` | `[off]` | upsampled/filtered | no |
| Z3 | no | `[off-1..off-w-h]` | no | upsampled/filtered | no |
| FILTER | `[off+1..off+1+w]` | `[off-1..off-2]` | `[off]` | no | yes (prev 4×2 blocks) |
| CFL_PRED | no | no | no | no | no (reads AC buf) |
| PAL_PRED | no | no | no | no | no (reads pal+idx) |

#### rav1d_prepare_intra_edges — Edge gathering
- **Read from picture** (current frame, already reconstructed neighbors):
  - Top edge: `dst[-stride + 0..w]` (row above block)
  - Left edge: `dst[-1 + y*stride]` for y=0..h (column left of block)
  - Top-left corner: `dst[-stride - 1]`
  - Top-right extension: `dst[-stride + w..w + overflow]`
  - Bottom-left extension: `dst[h*stride - 1 + y*stride]` for y=0..overflow
- **Or from ipred_edge cache**: reads pre-backed-up edge from `f.ipred_edge` when
  at SB boundary (the row above is in a different SB row)
- **Write**: `topleft_out[0..257]` scratch buffer (per-block, stack or per-thread)

**Cross-SB read**: When block is at the top of an SB row (`by == sby * sb_step`),
the top edge comes from `f.ipred_edge` (backed up by previous SB row), NOT from
the live picture buffer. This is the only cross-SB read in reconstruction.

#### cfl_ac_rust — Extract luma AC for chroma-from-luma
- **Read src (luma plane)**: `y_src[0..(h-h_pad)*(1<<ss_ver)][0..(w-w_pad)*(1<<ss_hor)]`
  - For 4:2:0: reads 2×2 luma blocks per chroma pixel
  - For 4:2:2: reads 2×1 luma blocks
  - For 4:4:4: reads 1:1
- **Write**: `ac[0..w*h]` i16 scratch (not picture)
- **Cross-plane read**: reads *luma* plane while writing *chroma* plane

#### pal_pred_rust — Palette prediction
- **Write**: `dst[0..h][0..w]` (picture)
- **Read**: `pal[0..8]` (palette table), `idx[0..w*h/2]` (packed indices)
- No picture read

### 1.4 backup_ipred_edge — Edge backup for next SB row

Called once per SB row after all blocks are reconstructed.

- **Read from picture (Y)**: row `(sby + 1) * sb_step * 4 - 1` (last row of SB), all tile columns
- **Read from picture (U,V)**: same row, subsampled
- **Write**: `f.ipred_edge[plane * ipred_edge_off + sby_off + col_start..col_end]`

**Scope**: reads exactly 1 row from the current SB's bottom edge; writes to a
dedicated scratch buffer. Does NOT read from adjacent SB rows.

---

## Phase 2: Post-Reconstruction Filters — Exact Access Patterns

### Orchestration (per SB row `sby`)

```
rav1d_filter_sbrow_deblock_cols(sby)    // vertical edge deblocking
rav1d_filter_sbrow_deblock_rows(sby)    // horizontal edge deblocking + copy_lpf
rav1d_filter_sbrow_cdef(sby)            // CDEF (if enabled)
rav1d_filter_sbrow_resize(sby)          // super-resolution (if enabled)
rav1d_filter_sbrow_lr(sby)              // loop restoration (if enabled)
```

Film grain is applied separately, per 32-row strip, after all filter stages.

### 2.1 Deblock (Loopfilter)

#### Pixel row range for SB row `sby`

```
y_start = sby << (6 + sb128)          // first pixel row
y_end   = min((sby + 1) << (6 + sb128), frame_height)
// SB64: 64 rows, SB128: 128 rows per SB
```

Deblocking processes block edges within and at the boundaries of this SB row.

#### Filter reach per filter width

| Filter Width | Read Range | Write Range | Notes |
|-------------|------------|-------------|-------|
| 4 | ±2 pixels from edge | ±1 pixel from edge | p1..q1 read, p0/q0 written |
| 6 | ±3 pixels from edge | ±2 pixels from edge | |
| 8 | ±4 pixels from edge | ±3 pixels from edge | |
| 14 | ±7 pixels from edge | ±6 pixels from edge | Wide filter for Y only |
| 16 | ±8 pixels from edge | ±7 pixels from edge | Wide filter for Y only |

#### Cross-SB boundary access

- **Vertical edges (DeblockCols)**: Filter is applied to column edges. Reads ±7 pixels
  horizontally from the edge. All reads are within the same row — **no cross-SB-row access**.

- **Horizontal edges (DeblockRows)**: Filter is applied to row edges. At the top of
  SB row `sby`, the edge is between row `sby*sb_step*4 - 1` (previous SB) and row
  `sby*sb_step*4` (current SB). The filter reads **up to 7 rows above** the edge
  (into the previous SB row) and **up to 7 rows below**.

  **Cross-SB read**: DeblockRows reads up to 7 pixel rows from SB row `sby-1`.
  **Cross-SB write**: DeblockRows writes up to 6 pixel rows into SB row `sby-1`.

#### Level cache access

- **Buffer**: `f.lf.level` — flat `Vec<u8>`, indexed by 4-pixel block position
- **Index**: `level_offset + (bx4 + by4 * b4_stride)` where `b4_stride = (f.bw + 31) & !31`
- **Range per SB row**: reads blocks in `[sby*sbsz, (sby+1)*sbsz)` range vertically,
  plus up to 4 blocks (1 block = 4 pixels) into the previous row for edge classification
- **Cross-SB read**: reads level values from previous SB row (`by4 - 4`) for edge
  strength determination at horizontal SB boundaries

#### Mask access

- **Buffer**: `f.lf.mask: Vec<Av1Filter>` — indexed by `sb128x` (SB column in 128-pixel units)
- **Per SB**: `filter_y[2][32][2]` (column/row × 32 4-pixel rows × 2 u16 words)
- **Per SB**: `filter_uv[2][32][2]` (same for chroma)
- **Per SB**: `noskip_mask[16][2]` (which 8×8 blocks have non-zero coefficients)
- **Read-only** during filter application; written during entropy decode

### 2.2 copy_lpf — Backup loop-filtered rows for CDEF/LR

Called within `rav1d_filter_sbrow_deblock_rows`, after deblocking.

#### What it copies

For each plane, copies **4 rows** of loop-filtered pixels into backup buffers,
centered around the SB boundary:

- **Y**: copies rows at offset `-8` from SB boundary (for `sby > 0`) through
  `+3` below boundary. Specifically:
  - `y_stripe = sby << (6 + sb128) - offset_y` where `offset_y = 8 * (sby != 0)`
  - Copies stripe of height up to `row_h - y_stripe` where `row_h = min((sby+1) << (6+sb128), h-1)`
- **UV**: same pattern, subsampled

#### Destination buffers

Two separate backup destinations (double-buffered by SB row):

1. **`f.lf.lr_line_buf`** — for loop restoration
   - Offset: `f.lf.lr_lpf_line[plane]` (advanced by `tt_off * stride` per SB row when threaded)
   - Contains: 4 rows of loop-filtered pixels per plane per SB row

2. **`f.lf.cdef_line_buf`** — for CDEF (only when resize + threading)
   - Offset: `f.lf.cdef_lpf_line[plane]` + per-SB offset
   - Contains: same 4 rows, at coded resolution (pre-super-resolution)

#### Read from picture

- Reads deblocked pixel rows from `src[plane]` at the SB boundary region
- Range: `[y_stripe - offset_y, min(y_stripe + stripe_height, frame_height)]`
- Width: full frame width (`f.bw << 2`)

#### Backup strategy function (backup_lpf)

Copies rows from the picture to the lr/cdef line buffers. For each stripe:
- First stripe for this SB: `stripe_h = (64 >> ss_ver) - offset_y` (shorter)
- Subsequent stripes: `stripe_h = 64 >> ss_ver`
- Copies `min(4, remaining)` rows per stripe to the backup buffer

### 2.3 CDEF (Constrained Directional Enhancement Filter)

#### Processing unit

CDEF operates on 8×8 blocks (luma) or 4×4/4×8 blocks (chroma depending on subsampling).

#### Pixel row range

```
by_start = sby * sbsz        // in 4-pixel units
by_end   = min(by_start + sbsz, f.bh)
// Iterates by += 2 (8-pixel steps) within this range
```

#### backup2lines — Pre-CDEF boundary backup

Called once per SB row before CDEF filtering.

- **Read from picture (Y)**: rows 6 and 7 relative to SB row start (the 2 bottom
  rows of the first 8-row stripe). With negative stride: row indices adjusted.
  - Exact: `src[0] + (6 + strides_adj) * y_stride`, length = `2 * |y_stride|`
- **Read from picture (UV)**: rows 2 and 3 (I420) or rows 6 and 7 (I422/I444)
- **Write**: `f.lf.cdef_line_buf` at `dst_off[plane]` — double-buffered via `!tf` toggle
- **Copies**: exactly **2 contiguous rows** per plane (2 × |stride| bytes)

#### backup2x8 — Per-block column edge backup

Called per 8×8 block being filtered.

- **Read from picture**: 2 pixels wide × 8 (or 4 for chroma) pixels tall
  - Left edge: columns `[block_x - 2, block_x - 1]` × 8 rows
  - Right edge: columns `[block_x + 8, block_x + 9]` × 8 rows
- **Write**: `lr_bak: [[[BD::Pixel; 2]; 8]; 3]` — tiny stack buffer per block

#### CDEF filter kernel

Builds a padded 12×12 temporary buffer per 8×8 block:

```
Padding layout for 8×8 luma block (TMP_STRIDE=12):
         2px pad     8px center    2px pad
      ┌──────────┬──────────────┬──────────┐
      │          │   top 2 rows │          │  ← from cdef_line_buf (prev SB)
      ├──────────┼──────────────┼──────────┤     or replicated edge
      │          │              │          │
      │  2px     │  8×8 center  │  2px     │  ← from picture (current SB)
      │  left    │  block data  │  right   │
      │          │              │          │
      ├──────────┼──────────────┼──────────┤
      │          │ bottom 2 rows│          │  ← from cdef_line_buf or lr_line_buf
      └──────────┴──────────────┴──────────┘     or replicated edge
```

- **Top padding source**: `cdef_line_buf` (backed up from *previous* SB row's bottom rows)
  - Toggle index: `!tf` (opposite of current SB row's write buffer)
- **Bottom padding source**: `cdef_line_buf` or `lr_line_buf` depending on context
  - If at start of SB row and have_tt: from `lr_line_buf` (loop-filtered backup)
  - Otherwise: from `cdef_line_buf` (next SB row's top backup)
- **Left/right padding**: from `lr_bak` (backed up by `backup2x8`) or replicated edge
- **Center**: direct read from picture

**Write**: in-place to picture — `dst[0..8][0..8]` per block (overwrites the block)

#### CDEF cross-SB access summary

| Source | What | Cross-SB? |
|--------|------|-----------|
| Top 2 rows | `cdef_line_buf[!tf]` | Yes — reads prev SB row's backup |
| Center 8×8 | picture | No — current SB only |
| Bottom 2 rows | `cdef_line_buf[tf]` or `lr_line_buf` | Depends on pipeline position |
| Left/right 2 cols | `lr_bak` (stack) or picture | No — same SB row (adjacent block) |

**Key for rayon**: CDEF reads the picture only within the current SB row. Cross-SB
data comes from pre-copied backup buffers. The backup buffers are separate allocations
with their own ownership — no picture-plane cross-SB aliasing.

### 2.4 Loop Restoration (LR)

#### Processing unit

LR operates on "restoration units" — typically 64 to 256 pixels wide, one
full stripe height (64 pixels luma, 32 for I420 chroma).

#### Stripe geometry

```
stripe_h = 64 pixels (luma), 32 pixels (I420 chroma), 64 pixels (I422/I444 chroma)
y_start  = sby << (6 + sb128) - offset_y    // offset_y = 8 * (sby > 0)
row_h    = min((sby + 1) << (6 + sb128) - 8 * not_last, h)
```

The stripe extends 8 rows beyond the SB boundary (overlap between adjacent SB rows'
filter regions), but the overlap region is handled via the backup buffer.

#### Padding buffer layout

Per restoration unit, a padded buffer is assembled:

```
REST_UNIT_STRIDE = 390 pixels (= 256 * 3/2 + 3 + 3)

Buffer: [BD::Pixel; (64 + 3 + 3) * REST_UNIT_STRIDE]  (70 rows × 390 cols)

       3px left pad   unit_w center   3px right pad
      ┌────────────┬────────────────┬────────────────┐
      │            │  3 rows above  │                │  ← from lr_line_buf (prev SB)
      ├────────────┼────────────────┼────────────────┤
      │            │                │                │
      │  3px pad   │  stripe_h rows │  3px pad       │  ← from picture (current SB)
      │            │   (center)     │                │
      │            │                │                │
      ├────────────┼────────────────┼────────────────┤
      │            │  3 rows below  │                │  ← from lr_line_buf (next SB)
      └────────────┴────────────────┴────────────────┘
```

- **Above rows**: from `lr_line_buf` (backed up by `copy_lpf` of previous SB row)
- **Below rows**: from `lr_line_buf` (backed up by `copy_lpf` of current SB row)
- **Left/right pad**: from saved `left` array or edge replication
- **Center**: direct read from picture

#### Wiener filter kernel

- **Horizontal pass**: 7-tap (or 5-tap if `filter[0][0]==0`)
  - Reads: 3 pixels left, center, 3 pixels right per pixel per row
  - Output: intermediate i16 buffer `hor[(64+6) * REST_UNIT_STRIDE]`
- **Vertical pass**: 7-tap
  - Reads: 3 rows above, center, 3 rows below per pixel per column
  - From the `hor` intermediate buffer
  - **Write**: directly to picture at restoration unit position

#### SGR (Self-Guided Recovery) filter

- **Box filtering**: computes 3×3 or 5×5 box sums over the padded buffer
  - Uses `sumsq[(64+4) * REST_UNIT_STRIDE]` and `sum[(64+4) * REST_UNIT_STRIDE]`
- **Guided filter**: applies self-guided filter using box statistics
- **Write**: directly to picture at restoration unit position

#### LR cross-SB access summary

| Source | What | Cross-SB? |
|--------|------|-----------|
| 3 rows above | `lr_line_buf` (prev SB's copy_lpf) | Yes — reads backup |
| stripe_h center | picture | No — current SB row |
| 3 rows below | `lr_line_buf` (current SB's copy_lpf) | No — own backup |
| Left edge | saved `left[stripe_h][4]` array | No |

**Key for rayon**: LR reads the picture only within the current SB row's stripe.
All cross-SB data comes from `lr_line_buf`, which is a separate allocation.

### 2.5 Film Grain

Applied per 32-row strip after all other filters complete.

#### Processing unit

- Block size: 32×32 pixels (FG_BLOCK_SIZE=32)
- Strips: `rows = ceil(height / 32)`, each strip is `min(32, remaining)` rows

#### Film grain application (fgy_32x32xn / fguv_32x32xn)

- **Read**: `in_data[plane]` — source picture (may be same as or different from output)
  - Reads strip: `[row * 32, min((row + 1) * 32, height))` rows, full width
- **Write**: `out_data[plane]` — destination picture
  - Same row range, full width
- **Read (chroma)**: `in_data[0]` — luma plane (for chroma scaling from luma)
  - I420: reads `2×2` luma pixels per chroma pixel
  - I422: reads `2×1` luma pixels
  - I444: reads `1×1` luma pixels

#### Grain table access

- `grain_lut[plane]: [i8; 73][82]` — pre-computed per-plane
- `scaling[plane]: [u8; 256 or 4096]` — scaling lookup by pixel value
- Both are read-only, computed once per frame

#### Overlap blending

First 2 rows/columns of each 32×32 block are blended with the previous block's
grain using a 2×2 weight matrix. This is within the same strip — no cross-strip access.

#### Film grain cross-SB access

**None.** Film grain is applied per strip with no data dependencies between strips.
Each strip reads its own rows from `in` and writes to `out`. The grain LUT and scaling
tables are frame-constant read-only data.

**Key for rayon**: Film grain strips are trivially parallel — no cross-strip data flow.

---

## Scratch Buffer Inventory

### Per-Thread Stack Allocations (reconstruction)

| Buffer | Type | Size | Used By |
|--------|------|------|---------|
| `mid` (8tap) | `[[i16; 128]; 135]` | 34,560 bytes | put_8tap, prep_8tap |
| `mid` (bilin) | `[[i16; 128]; 129]` | 33,024 bytes | put_bilin, prep_bilin |
| `mid` (scaled) | `[[i16; 128]; 263]` | 67,328 bytes | put_8tap_scaled |
| `mid` (warp) | `[[i32; 8]; 15]` | 480 bytes | warp_affine |
| `tmp1, tmp2` | `[i16; 16384]` | 32,768 bytes each | compound inter (avg, w_avg, mask) |
| `seg_mask` | `[u8; 16384]` | 16,384 bytes | w_mask |
| `emu_edge` | `[BD::Pixel; 84160]` | 84,160-168,320 bytes | emu_edge boundary blocks |
| `topleft_out` | `[BD::Pixel; 257]` | 257-514 bytes | ipred edge buffer |
| `ac` | `[i16; 1024]` | 2,048 bytes | CFL AC extraction |
| `coeff` | `[BD::Coef; 4096]` | 4,096-8,192 bytes | ITX coefficient buffer |
| `itx_tmp` | `[i32; 4096]` | 16,384 bytes | ITX intermediate |

All of these are per-thread / per-block — no sharing between SB rows.

### Per-Frame Shared Buffers

| Buffer | Type | Size | Access Pattern |
|--------|------|------|----------------|
| `f.ipred_edge` | `DisjointMut<AlignedVec64<u8>>` | `3 * ipred_edge_off` bytes | Write: backup_ipred_edge(sby); Read: prepare_intra_edges(sby+1) |
| `f.lf.level` | `DisjointMut<Vec<u8>>` | `b4_stride * (f.bh + 31)` bytes | Write: decode; Read: deblock |
| `f.lf.mask` | `Vec<Av1Filter>` | `f.sb128w * f.sbh` entries | Write: decode; Read: deblock, CDEF |
| `f.lf.cdef_line_buf` | `DisjointMut<AlignedVec64<u8>>` | `2 * 3 * |stride| * 2` bytes | Double-buffered per SB row |
| `f.lf.lr_line_buf` | `DisjointMut<AlignedVec64<u8>>` | per-SB-row offsets | Offset-indexed per plane per SB row |
| `f.lf.lr_mask` | `Vec<Av1Restoration>` | per-SB entries | Read-only during LR |

### Per-Frame Read-Only (reference frames)

| Buffer | Type | Access Pattern |
|--------|------|----------------|
| `f.refp[0..7].p.data` | `Arc<Rav1dPictureData>` | MC reads via immutable row slices |
| `f.ref_mvs[0..7]` | `DisjointMutArcSlice<RefMvsTemporalBlock>` | MV lookup, read-only |

---

## Ownership Transfer Model

### Per-SB-Row Exclusive Regions

For SB row `sby`, the exclusive pixel write region is:

```
Y:  rows [sby * (64 << sb128), min((sby + 1) * (64 << sb128), height))
UV: rows [sby * (64 << sb128) >> ss_ver, min((...) >> ss_ver, uv_height))
```

Plus **overflow rows** for MC blocks that span the SB boundary (up to 24 rows
for a 128-tall block starting at the bottom of the SB).

### Cross-SB Reads During Reconstruction

| Source | Reader | What | Resolution |
|--------|--------|------|------------|
| Previous SB's bottom row | `prepare_intra_edges` | ipred top edge | Via `f.ipred_edge` scratch (already separate) |
| Reference frames | all MC functions | motion-compensated pixels | Via immutable `Arc` row slices |

**No direct cross-SB picture read during reconstruction.** All cross-SB data arrives
through dedicated scratch buffers or immutable reference frames.

### Cross-SB Reads During Filtering

| Stage | Reader | What | Resolution |
|-------|--------|------|------------|
| Deblock (rows) | `loop_filter_sb128_rust` | ±7 rows across SB edge | Overlap region — see below |
| CDEF | `cdef_filter_block` | 2 rows above/below | Via `cdef_line_buf` (separate allocation) |
| LR | `wiener_rust` / `sgr` | 3 rows above/below | Via `lr_line_buf` (separate allocation) |

### The Deblock Overlap Problem

Deblock is the **only filter that directly reads/writes across SB boundaries** in the
picture plane. Horizontal deblocking at the SB boundary reads up to 7 rows into the
previous SB row and writes up to 6 rows into it.

**Resolution options:**

1. **Sequential deblocking**: Deblock SB row N finishes before Deblock SB row N+1 starts.
   Since N+1's deblock only touches the top of its region and N's deblock only touches
   the bottom of its region (and they overlap by ≤7 rows at the boundary), sequential
   execution is sufficient. The rayon channel-based pipeline already enforces this.

2. **Split deblock region**: Deblock SB row N owns rows `[N_start - 7, N_end]`. The
   `-7` rows are "borrowed back" from SB row N-1 after N-1's deblock completes.
   This maps to `split_at_mut` at `N_start - 7`.

3. **Copy boundary rows**: Copy the 7 boundary rows to a scratch buffer before
   releasing SB row N-1's ownership. Deblock N reads from the scratch copy.

Option 1 (sequential pipeline) is simplest and matches the current `channel`-based
architecture.

### Ownership Timeline (per SB row)

```
Time →
SB row N:  [recon ────────────][deblock ──][copy_lpf][CDEF ──][LR ──]
SB row N+1:                    [recon ────────────][deblock ...
                                                         ↑
                                                  waits for N's copy_lpf
```

**Owned rows during each phase:**

| Phase | SB row N owns | Notes |
|-------|---------------|-------|
| Recon | rows `[N_start, N_end + overflow)` | overflow ≤ 24 rows for large MC blocks |
| Deblock | rows `[N_start - 7, N_end]` | reads/writes into prev SB row's tail |
| copy_lpf | rows `[N_start - 8, N_end]` | reads deblocked rows, writes to lr/cdef buffers |
| CDEF | rows `[N_start, N_end)` | cross-SB via cdef_line_buf only |
| LR | rows `[N_start - 8, N_end - 8*not_last)` | cross-SB via lr_line_buf only |

### Rayon Split Strategy

```rust
// Before decode:
let all_rows: Vec<&mut [BD::Pixel]> = split_frame(buf, stride, width, height);

// Per SB row: split off the needed region
let (above, rest) = all_rows.split_at_mut(sb_start);
let (my_rows, below) = rest.split_at_mut(sb_height + overflow);

// Recon gets my_rows mutably, ref_frames as &[&[BD::Pixel]] immutably
// After recon, return overflow rows to below via channel

// For deblock: borrow 7 rows from above via channel (after prev SB's deblock done)
// For copy_lpf: read from my_rows, write to lr/cdef buffers (separate allocs)
// For CDEF/LR: my_rows only (cross-SB via buffers)
```

---

## Complete Function Inventory

### Reconstruction Functions (write to picture)

| Function | File | Writes | Reads (picture) | Reads (other) | Cross-SB |
|----------|------|--------|------------------|---------------|----------|
| `put_rust` | mc.rs:56 | dst `h×w` | — | src (ref) `h×w` | No |
| `prep_rust` | mc.rs:67 | tmp (i16) | — | src (ref) `h×w` | No |
| `put_8tap_rust` | mc.rs:139 | dst `h×w` | — | src (ref) `(h+7)×(w+7)` | No |
| `prep_8tap_rust` | mc.rs:266 | tmp (i16) | — | src (ref) `(h+7)×(w+7)` | No |
| `put_bilin_rust` | mc.rs:397 | dst `h×w` | — | src (ref) `(h+1)×(w+1)` | No |
| `prep_bilin_rust` | mc.rs:505 | tmp (i16) | — | src (ref) `(h+1)×(w+1)` | No |
| `put_8tap_scaled_rust` | mc.rs:207 | dst `h×w` | — | src (ref) variable | No |
| `prep_8tap_scaled_rust` | mc.rs:329 | tmp (i16) | — | src (ref) variable | No |
| `put_bilin_scaled_rust` | mc.rs:460 | dst `h×w` | — | src (ref) variable | No |
| `prep_bilin_scaled_rust` | mc.rs:562 | tmp (i16) | — | src (ref) variable | No |
| `avg_rust` | mc.rs:1022 | dst `h×w` | — | tmp1, tmp2 (i16) | No |
| `w_avg_rust` | mc.rs:1046 | dst `h×w` | — | tmp1, tmp2 (i16) | No |
| `mask_rust` | mc.rs:1072 | dst `h×w` | — | tmp1, tmp2, mask | No |
| `w_mask_rust` | mc.rs:1157 | dst `h×w` + mask | — | tmp1, tmp2 | No |
| `blend_rust` | mc.rs:1105 | dst `h×w` (R+W) | dst `h×w` | tmp, mask | No |
| `blend_v_rust` | mc.rs:1121 | dst `h×(3w/4)` (R+W) | dst | tmp, obmc table | No |
| `blend_h_rust` | mc.rs:1139 | dst `(3h/4)×w` (R+W) | dst | tmp, obmc table | No |
| `emu_edge_rust` | mc.rs:1326 | scratch buf | — | ref (clipped) | No |
| `resize_rust` | mc.rs:1397 | dst `h×dst_w` | — | src `h×src_w` | No |
| `warp_affine_8x8_rust` | mc.rs:1226 | dst `8×8` | — | src (ref) `15×12` | No |
| `warp_affine_8x8t_rust` | mc.rs:1276 | tmp (i16) | — | src (ref) `15×12` | No |
| `inv_txfm_add` | itx.rs:68 | dst `h×w` (R+W) | dst `h×w` | coeff | No |
| `splat_dc` | ipred.rs:489 | dst `h×w` | — | — | No |
| `ipred_v_rust` | ipred.rs:749 | dst `h×w` | — | topleft (scratch) | No |
| `ipred_h_rust` | ipred.rs:794 | dst `h×w` | — | topleft (scratch) | No |
| `ipred_paeth_rust` | ipred.rs:839 | dst `h×w` | — | topleft | No |
| `ipred_smooth_rust` | ipred.rs:898 | dst `h×w` | — | topleft | No |
| `ipred_smooth_v_rust` | ipred.rs:950 | dst `h×w` | — | topleft | No |
| `ipred_smooth_h_rust` | ipred.rs:998 | dst `h×w` | — | topleft | No |
| `ipred_z1_rust` | ipred.rs:1145 | dst `h×w` | — | topleft (extended) | No |
| `ipred_z2_rust` | ipred.rs:1229 | dst `h×w` | — | topleft (extended) | No |
| `ipred_z3_rust` | ipred.rs:1365 | dst `h×w` | — | topleft (extended) | No |
| `ipred_filter_rust` | ipred.rs:1494 | dst `h×w` (4×2 blocks) | dst (prev blocks) | topleft | No |
| `cfl_ac_rust` | ipred.rs:1586 | ac (i16 buf) | luma plane | — | No* |
| `cfl_pred` | ipred.rs:514 | dst `h×w` | — | ac (i16 buf) | No |
| `pal_pred_rust` | ipred.rs:1682 | dst `h×w` | — | pal, idx | No |
| `prepare_intra_edges` | ipred_prepare.rs:165 | topleft `[257]` | dst (neighbors) | ipred_edge cache | **Indirect** |
| `backup_ipred_edge` | recon.rs:3834 | ipred_edge buf | cur frame (1 row) | — | **Write** |

\* `cfl_ac_rust` reads the *luma* plane while the current block is on *chroma*. Both
are within the same SB row — the luma block was already reconstructed before the
chroma block.

### Filter Functions (read+write picture in place)

| Function | File | Writes | Reads (picture) | Reads (buffers) | Cross-SB |
|----------|------|--------|------------------|-----------------|----------|
| `loop_filter_sb128_rust` | loopfilter.rs | ±6px from edge | ±7px from edge | lf.level, lf.mask | **Yes** (deblock) |
| `backup2lines` | cdef_apply.rs:42 | cdef_line_buf | rows 6-7 of SB | — | No |
| `backup2x8` | cdef_apply.rs:82 | lr_bak (stack) | 2×8 edge pixels | — | No |
| `cdef_filter_block` | cdef.rs | 8×8 block | 8×8 block | cdef_line_buf, lr_bak | **Via buffer** |
| `padding` (LR) | looprestoration.rs:331 | padded buf | stripe rows | lr_line_buf, left arr | **Via buffer** |
| `wiener_rust` | looprestoration.rs:554 | unit region | — | padded buf (stack) | No |
| `sgr_filter/selfguided` | looprestoration.rs | unit region | — | padded buf (stack) | No |
| `fgy_32x32xn` | filmgrain.rs | out `32×w` | in `32×w` | grain_lut, scaling | No |
| `fguv_32x32xn` | filmgrain.rs | out UV strip | in UV strip | luma plane, grain_lut | No |
| `copy_lpf` (backup_lpf) | lf_apply.rs:150 | lr_line_buf, cdef_line_buf | boundary rows | — | No |

### Functions with NO Picture Access

| Function | File | Notes |
|----------|------|-------|
| `msac_*` | msac.rs | Entropy decoding only |
| `decode_*` | decode.rs | Bitstream parsing; creates PicOffset but delegates pixel access |
| `refmvs_*` | refmvs.rs | Motion vector management (separate buffer) |
| `pal_idx_finish` | pal.rs | Palette index SIMD (operates on index buffer, not picture) |
| `splat_mv` | refmvs.rs | MV buffer SIMD (not picture) |

---

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

---

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

## Appendix D: Key Constants

| Constant | Value | Used By |
|----------|-------|---------|
| `COMPINTER_LEN` | 128 × 128 = 16,384 | tmp1/tmp2 compound buffers |
| `SEG_MASK_LEN` | 128 × 128 = 16,384 | w_mask segmentation mask |
| `SCRATCH_INTER_INTRA_BUF_LEN` | 64 × 64 = 4,096 | blend inter-intra buffer |
| `SCRATCH_LAP_LEN` | 128 × 32 = 4,096 | blend_v/blend_h OBMC buffer |
| `EMU_EDGE_LEN` | 320 × 263 = 84,160 | edge extension scratch |
| `SCRATCH_EDGE_LEN` | 257 | ipred edge buffer |
| `SCRATCH_AC_TXTP_LEN` | 32 × 32 = 1,024 | CFL AC buffer |
| `MID_STRIDE` | 128 | 8tap/bilin intermediate width |
| `REST_UNIT_STRIDE` | 390 (= 256*3/2 + 6) | LR padding buffer width |
| `FG_BLOCK_SIZE` | 32 | Film grain processing unit |
| `GRAIN_WIDTH` | 82 | Film grain LUT width |
| `GRAIN_HEIGHT` | 73 | Film grain LUT height |
| `TMP_STRIDE` | 12 | CDEF padding buffer stride |
