//! Quick 4K photo decode at various thread counts.
//! Run: cargo test --release --features mt --test test_4k_threading -- --ignored --nocapture

use rav1d_safe::src::managed::{Decoder, Settings};

fn load_4k_obu() -> Option<Vec<u8>> {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("test-vectors/bench/photo_4k.avif");
    if !path.exists() {
        return None;
    }
    let data = std::fs::read(&path).ok()?;
    let parser = zenavif_parse::AvifParser::from_bytes(&data).ok()?;
    Some(parser.primary_data().ok()?.to_vec())
}

#[test]
#[ignore]
fn test_4k_thread_scaling() {
    let obu = match load_4k_obu() {
        Some(o) => o,
        None => {
            eprintln!("4K photo not found, skipping");
            return;
        }
    };
    eprintln!("4K OBU: {} bytes", obu.len());

    for threads in [1u32, 2, 3, 4] {
        let start = std::time::Instant::now();
        let n_iters = 3;
        for _ in 0..n_iters {
            let settings = Settings {
                threads,
                max_frame_delay: 1,
                ..Default::default()
            };
            let mut decoder = Decoder::with_settings(settings).expect("decoder");
            let frame = decoder.decode(&obu).expect("decode");
            let f = frame.expect("frame");
            assert!(f.width() > 0);
            std::hint::black_box(&f);
        }
        let elapsed = start.elapsed();
        let ms_per = elapsed.as_millis() as f64 / n_iters as f64;
        eprintln!("  {threads}t: {ms_per:.1}ms/decode ({n_iters} iters)");
    }
}
