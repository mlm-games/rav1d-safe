//! Measure DisjointMut tracking + bounds-check overhead.
//!
//! Run with each feature set and compare via saved baselines:
//!
//!   # 1. Build and save "checked" baseline (clean build!)
//!   cargo clean
//!   cargo bench --bench checked_vs_unchecked \
//!     --no-default-features --features "bitdepth_8,bitdepth_16" \
//!     -- --save-baseline=checked
//!
//!   # 2. Build and compare "unchecked" against saved baseline
//!   cargo clean
//!   cargo bench --bench checked_vs_unchecked \
//!     --no-default-features --features "bitdepth_8,bitdepth_16,unchecked" \
//!     -- --baseline=checked
//!
//! The baselines are stored per-testbed so hardware differences don't corrupt results.

use rav1d_safe::src::managed::{self, Decoder, Settings};
use std::path::PathBuf;
use std::sync::OnceLock;
use zenbench::prelude::*;

fn bench_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("test-vectors")
        .join("bench")
}

fn load_obu(filename: &str) -> Option<Vec<u8>> {
    let path = bench_dir().join(filename);
    let data = std::fs::read(&path).ok()?;
    let parser = zenavif_parse::AvifParser::from_bytes(&data).ok()?;
    let obu = parser.primary_data().ok()?.into_owned();
    let mut s = Settings::default();
    s.frame_size_limit = 8192 * 8192;
    let mut dec = Decoder::with_settings(s).ok()?;
    dec.decode(&obu).ok()?.map(|_| obu)
}

fn obu_4k() -> &'static [u8] {
    static CACHE: OnceLock<Vec<u8>> = OnceLock::new();
    CACHE.get_or_init(|| load_obu("photo_4k.avif").expect("photo_4k.avif missing"))
}

fn bench_decode(suite: &mut Suite) {
    let mode = if managed::is_unchecked() { "unchecked" } else { "checked" };
    eprintln!("rav1d-safe build mode: {mode}");

    let obu = obu_4k();

    // Use a fixed group name so baselines are comparable across feature flags.
    // The mode label is just for display — the benchmark name must match.
    suite.group("decode_4k_1t", |g| {
        g.throughput(Throughput::Bytes(obu.len() as u64));
        g.config().max_time(std::time::Duration::from_secs(30));

        g.bench("decode", |b| {
            b.iter(|| {
                let mut s = Settings::default();
                s.frame_size_limit = 8192 * 8192;
                let mut dec = Decoder::with_settings(s).unwrap();
                dec.decode(obu).unwrap().unwrap();
            })
        });
    });
}

zenbench::main!(bench_decode);
