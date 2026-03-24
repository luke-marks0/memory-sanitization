use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn find_nvcc() -> Option<String> {
    if let Ok(explicit) = env::var("NVCC") {
        if !explicit.trim().is_empty() {
            return Some(explicit);
        }
    }
    let path = env::var_os("PATH")?;
    for entry in env::split_paths(&path) {
        let candidate = entry.join("nvcc");
        if candidate.is_file() {
            return Some(candidate.to_string_lossy().into_owned());
        }
    }
    None
}

fn cargo_home() -> Option<PathBuf> {
    env::var_os("CARGO_HOME")
        .map(PathBuf::from)
        .or_else(|| env::var_os("HOME").map(|home| PathBuf::from(home).join(".cargo")))
}

fn find_blake3_c_dir() -> Option<PathBuf> {
    let registry_src = cargo_home()?.join("registry").join("src");
    let mut candidates = Vec::new();
    for registry in fs::read_dir(registry_src).ok()? {
        let registry = registry.ok()?;
        if !registry.file_type().ok()?.is_dir() {
            continue;
        }
        for entry in fs::read_dir(registry.path()).ok()? {
            let entry = entry.ok()?;
            if !entry.file_type().ok()?.is_dir() {
                continue;
            }
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name.starts_with("blake3-") {
                let candidate = entry.path().join("c");
                if candidate.is_dir() {
                    candidates.push(candidate);
                }
            }
        }
    }
    candidates.sort();
    candidates.pop()
}

fn compile_blake3_internal2_batch() {
    println!("cargo:rerun-if-changed=src/blake3_internal2_batch.c");
    println!("cargo:rustc-check-cfg=cfg(pose_blake3_internal2_batch_available)");

    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    if target_arch != "x86_64" && target_arch != "x86" {
        return;
    }

    let Some(blake3_c_dir) = find_blake3_c_dir() else {
        println!("cargo:warning=blake3 C sources not found; host internal_label_2 batch kernel disabled");
        return;
    };

    let mut build = cc::Build::new();
    if env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default() != "msvc" {
        build.flag("-std=c11");
    }
    build.include(&blake3_c_dir);
    build.file(blake3_c_dir.join("blake3.c"));
    build.file(blake3_c_dir.join("blake3_dispatch.c"));
    build.file(blake3_c_dir.join("blake3_portable.c"));
    build.file(Path::new("src").join("blake3_internal2_batch.c"));
    build.compile("pose_blake3_internal2_batch");

    println!("cargo:rustc-cfg=pose_blake3_internal2_batch_available");
}

fn main() {
    compile_blake3_internal2_batch();
    println!("cargo:rerun-if-changed=src/hbm_inplace.cu");
    println!("cargo:rustc-check-cfg=cfg(pose_cuda_hbm_available)");

    let Some(nvcc) = find_nvcc() else {
        println!("cargo:warning=nvcc not found; CUDA HBM in-place engine disabled");
        return;
    };

    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR not set"));
    let object_path = out_dir.join("hbm_inplace.o");
    let archive_path = out_dir.join("libpose_hbm_cuda.a");
    let source_path = Path::new("src").join("hbm_inplace.cu");

    let status = Command::new(&nvcc)
        .args([
            "-O3",
            "-std=c++17",
            "-Xcompiler",
            "-fPIC",
            "-c",
        ])
        .arg(&source_path)
        .arg("-o")
        .arg(&object_path)
        .status()
        .expect("failed to run nvcc");
    if !status.success() {
        panic!("nvcc failed to compile {}", source_path.display());
    }

    let status = Command::new("ar")
        .args(["crus"])
        .arg(&archive_path)
        .arg(&object_path)
        .status()
        .expect("failed to run ar");
    if !status.success() {
        panic!("ar failed to archive {}", archive_path.display());
    }

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    for candidate in ["/usr/local/cuda/lib64", "/usr/local/cuda-12.4/lib64", "/usr/local/cuda-12/lib64"] {
        if Path::new(candidate).is_dir() {
            println!("cargo:rustc-link-search=native={candidate}");
        }
    }
    println!("cargo:rustc-link-lib=static=pose_hbm_cuda");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rustc-cfg=pose_cuda_hbm_available");
}
