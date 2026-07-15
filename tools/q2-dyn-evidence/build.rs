use std::fs;
use std::path::{Path, PathBuf};

use serde::Serialize;
use sha2::{Digest, Sha256};

#[derive(Serialize)]
struct HashRecord {
    path: String,
    sha256: String,
}

fn main() {
    let manifest_dir = PathBuf::from(
        std::env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is set by Cargo"),
    );
    let repo_root = manifest_dir
        .parent()
        .and_then(Path::parent)
        .expect("helper lives at tools/q2-dyn-evidence")
        .to_path_buf();
    let helper = helper_inputs(&repo_root);
    let lattice = lattice_inputs(&repo_root);
    println!(
        "cargo:rustc-env=Q2_DYN_HELPER_SOURCE_CLOSURE_SHA256={}",
        closure_sha256(&repo_root, &helper)
    );
    println!(
        "cargo:rustc-env=Q2_LATTICE_SOURCE_CLOSURE_SHA256={}",
        closure_sha256(&repo_root, &lattice)
    );
    println!("cargo:rerun-if-env-changed=Q2_LATTICE_CRATE_COMMIT");
    for path in helper.iter().chain(&lattice) {
        println!("cargo:rerun-if-changed={}", path.display());
    }
}

fn helper_inputs(root: &Path) -> Vec<PathBuf> {
    let base = root.join("tools/q2-dyn-evidence");
    let mut inputs = vec![
        base.join("Cargo.lock"),
        base.join("Cargo.toml"),
        base.join("README.md"),
        base.join("build.rs"),
    ];
    collect_rs(&base.join("src"), &mut inputs);
    canonical_inputs(inputs)
}

fn lattice_inputs(root: &Path) -> Vec<PathBuf> {
    let base = root.join("crates/q2-lattice");
    let mut inputs = vec![base.join("Cargo.toml")];
    collect_rs(&base.join("src"), &mut inputs);
    canonical_inputs(inputs)
}

fn collect_rs(directory: &Path, output: &mut Vec<PathBuf>) {
    let mut entries: Vec<_> = fs::read_dir(directory)
        .unwrap_or_else(|error| panic!("cannot read {}: {error}", directory.display()))
        .map(|entry| entry.expect("source directory entry is readable").path())
        .collect();
    entries.sort();
    for path in entries {
        if path.is_dir() {
            collect_rs(&path, output);
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            output.push(path);
        }
    }
}

fn canonical_inputs(mut inputs: Vec<PathBuf>) -> Vec<PathBuf> {
    inputs.sort();
    inputs.dedup();
    for path in &inputs {
        assert!(
            path.is_file(),
            "source-closure input missing: {}",
            path.display()
        );
    }
    inputs
}

fn closure_sha256(root: &Path, inputs: &[PathBuf]) -> String {
    let records: Vec<_> = inputs
        .iter()
        .map(|path| HashRecord {
            path: path
                .strip_prefix(root)
                .expect("closure input is below repository root")
                .to_string_lossy()
                .replace('\\', "/"),
            sha256: hex(&Sha256::digest(fs::read(path).unwrap_or_else(|error| {
                panic!("cannot read {}: {error}", path.display())
            }))),
        })
        .collect();
    hex(&Sha256::digest(
        serde_json::to_vec(&records).expect("source-closure records serialize"),
    ))
}

fn hex(bytes: &[u8]) -> String {
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write;
        write!(&mut output, "{byte:02x}").expect("writing to String cannot fail");
    }
    output
}
