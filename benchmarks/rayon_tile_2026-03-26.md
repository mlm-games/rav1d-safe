# Rayon Tile Parallelism Benchmark Results

Date: 2026-03-26
Branch: rayon-threading (commit after 6f368a5)
CPU: AMD (WSL2), 32 cores
Test images: plasma:fractal encoded via avifenc -s4 -q50 (1080p) / -s6 -q40 (4K)

## Results (ms/decode)

| Config | 1080p 2T | 1080p 4T | 4K 4T | 4K 8T |
|--------|----------|----------|-------|-------|
| ASM 1T | 10.4 | 9.4 | 21.3 | 21.7 |
| ASM rayon | **8.6** | **6.6** | **17.0** | **16.6** |
| Unchecked 1T | 56.0 | 55.4 | 179.9 | 176.5 |
| Unchecked rayon | **38.2** | **27.4** | **73.6** | **71.1** |
| Checked 1T | 58.4 | 58.5 | 182.8 | 205.6 |
| Checked rayon | **45.3** | **51.4** | **104.8** | **175.1** |

## Speedups (rayon vs sequential, same config)

| Config | 1080p 2T | 1080p 4T | 4K 4T | 4K 8T |
|--------|----------|----------|-------|-------|
| ASM | 1.21× | 1.42× | 1.25× | 1.31× |
| Unchecked | 1.47× | 2.02× | 2.44× | 2.48× |
| Checked | 1.29× | 1.14× | 1.74× | 1.17× |

## Analysis

- **Unchecked + rayon** delivers the best speedups (up to 2.48×) because SIMD runs on tile threads
- **Checked + rayon** is limited by force_scalar: reconstruction runs scalar on tile threads,
  only filter stages use SIMD (single-threaded). Best at 4K 4-tile (1.74×).
- **Checked 4K 8-tile (1.17×)** is poor: 8 scalar tile threads create excessive DisjointMut
  guard tracking overhead that overwhelms parallelism gain.
- **ASM + rayon** gives modest speedup (1.25-1.42×) because ASM kernels are so fast that
  the sequential filter phase dominates.

## Path to checked + SIMD tile parallelism

The force_scalar workaround disables SIMD on tile worker threads to prevent DisjointMut
guard overlap. To enable SIMD in checked mode:

1. Refactor ~50 SIMD inner functions to accept `&mut [&mut [BD::Pixel]]` per-row slices
   instead of `&mut [u8]` + offset + stride
2. Dispatch layer splits frame rows by tile columns via split_at_mut
3. Each tile gets per-row slices — borrow checker proves non-overlap
4. No DisjointMut needed for pixel access
5. Full SIMD performance with forbid(unsafe_code)
