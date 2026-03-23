use std::env;
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

fn main() {
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
