//! Thread scaling benchmark: measures decode time at 1-4 tile threads.
//!
//! Run: cargo bench --bench thread_scaling --features mt

use rav1d_safe::src::managed::{Decoder, Settings};
use std::path::PathBuf;
use std::sync::OnceLock;
use zenbench::black_box;

static OBU_DATA: OnceLock<Option<Vec<u8>>> = OnceLock::new();

fn get_obu() -> Option<&'static [u8]> {
    OBU_DATA
        .get_or_init(|| {
            let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("tests/crash_vectors/kodim03_yuv420_8bpc.obu");
            std::fs::read(&path).ok()
        })
        .as_deref()
}

zenbench::main!(|suite| {
    let obu = match get_obu() {
        Some(o) => o,
        None => {
            eprintln!("skipping: OBU test vector not found");
            return;
        }
    };
    suite.compare("tile_decode_768x512", |group| {
        group.throughput(zenbench::Throughput::Bytes(obu.len() as u64));

        for threads in [1u32, 2, 3, 4] {
            let label = format!("{threads}t");
            group.bench(&label, move |b| {
                b.iter(|| {
                    let settings = Settings {
                        threads,
                        max_frame_delay: 1,
                        ..Default::default()
                    };
                    let mut decoder = Decoder::with_settings(settings).expect("decoder");
                    let frame = decoder.decode(obu).expect("decode");
                    black_box(frame)
                })
            });
        }
    });
});
