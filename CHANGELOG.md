# Changelog

All notable changes to the `rav1d-safe` crate are documented in this file. Format follows [Keep a Changelog](https://keepachangelog.com/). `rav1d-safe` is a fork of [rav1d](https://github.com/memorysafety/rav1d), which is itself a Rust port of [dav1d](https://code.videolan.org/videolan/dav1d); this fork adds archmage-based SIMD dispatch and removes the C FFI path. Entries below cover only changes made in this fork — upstream rav1d and dav1d release notes remain the canonical record for the shared decoder core. This file was backfilled from git history on 2026-04-15; the `[0.5.4]` date reflects the commit date of tag `v0.5.4` rather than the crates.io publish date.

## [Unreleased]

### QUEUED BREAKING CHANGES
<!-- Breaking changes that will ship together in the next major (or minor for 0.x) release.
     Add items here as you discover them. Do NOT ship these piecemeal — batch them. -->

### Changed
- Replace blanket `#![allow(clippy::all)]` with a targeted lint policy across 27 files: 22 specific lint allows (each documented with warning count and rationale) cover pervasive C-port patterns such as `precedence`, `too_many_arguments`, `unnecessary_cast`, `identity_op`, and `needless_range_loop`, while ~100 warnings for the remaining enabled lints were fixed in place (db99f94, #7)
- Add crate-level allows for seven additional clippy lints that fire on CI's clippy 1.87+ (`duplicated_attributes`, `manual_is_multiple_of`, `let_and_return`, `unnecessary_map_on_constructor`, `clone_on_copy`, `option_map_unit_fn`, `unnecessary_lazy_evaluations`) — all pervasive C-port patterns not worth fixing individually (8c6621c, #7)

### Fixed
- Restore `MsacAsmContext` visibility for asm builds: the lint-audit refactor had accidentally gated the type behind `#[cfg(not(asm_msac))]`, breaking the `asm`-feature CI job; the erroneous cfg gate is removed and the manual `Default` impl (needed because the conditionally-compiled `symbol_adapt16` fn-pointer field doesn't derive `Default`) is reinstated (96dde32, #7)

### Tests
- `CpuLevel` doctest in `src/managed.rs` builds `Settings` via `Settings::default()` plus field mutation instead of a bare struct expression, avoiding E0639 on `#[non_exhaustive]` structs across the crate boundary that doctests compile against (008a811)

### Internal
- Ignore the `.workongoing` coordination marker file (008a811)

## [0.5.4] - 2026-04-10

Patch release focused on concurrency safety, parser hardening, and fuzz coverage.

### Fixed
- CDEF tile threading race
- MV parsing overflow guard
- `wrapping_sub` in `read_golomb`

### Tests
- AV1 fuzz dictionary expansion
