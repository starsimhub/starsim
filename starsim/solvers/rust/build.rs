//! Compile in a hash of the crate's source (`source_hash` in the module), which `starsim/solvers/backends/rust.py` checks against the source beside it,
//! so that a library built from older source isn't used silently. The same as `zlib.crc32()` over the files, in the order of `sources()` there.

use std::fs;
use std::path::{Path, PathBuf};

fn sources(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            sources(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

fn crc32(mut crc: u32, bytes: &[u8]) -> u32 {
    crc = !crc;
    for &byte in bytes {
        crc ^= byte as u32;
        for _ in 0..8 {
            crc = (crc >> 1) ^ (0xEDB88320 & (crc & 1).wrapping_neg());
        }
    }
    !crc
}

fn main() {
    if std::env::var("CARGO_CFG_TARGET_OS").unwrap() == "macos" { // Python's symbols are found when it loads the library, so `cargo build` can link it (as maturin and setuptools-rust do themselves)
        println!("cargo:rustc-cdylib-link-arg=-undefined");
        println!("cargo:rustc-cdylib-link-arg=dynamic_lookup");
    }
    println!("cargo:rerun-if-changed=src");
    println!("cargo:rerun-if-changed=Cargo.toml");
    let mut files = vec![PathBuf::from("Cargo.toml")];
    sources(Path::new("src"), &mut files);
    files.sort_by_key(|path| path.to_string_lossy().replace('\\', "/"));
    let hash = files.iter().fold(0, |crc, path| crc32(crc, &fs::read(path).unwrap()));
    println!("cargo:rustc-env=SOURCE_HASH={hash}");
}
