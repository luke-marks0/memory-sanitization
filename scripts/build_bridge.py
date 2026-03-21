#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BRIDGE_MANIFEST = REPO_ROOT / "rust" / "pose_filecoin_bridge" / "Cargo.toml"
VENV_BIN = REPO_ROOT / ".venv" / "bin"
CARGO_BIN = Path.home() / ".cargo" / "bin"
MATURIN = VENV_BIN / "maturin"
BACKEND_ENV = "POSE_FILECOIN_BRIDGE_BACKEND"
AUTO_BACKEND = "auto"
VALID_BACKENDS = {AUTO_BACKEND, "cpu", "opencl", "cuda", "cuda-opencl"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the vendored Filecoin Python bridge with an explicit upstream "
            "backend selection."
        )
    )
    parser.add_argument(
        "--backend",
        choices=tuple(sorted(VALID_BACKENDS)),
        default=os.environ.get(BACKEND_ENV, AUTO_BACKEND),
        help=(
            "Bridge backend to compile. Defaults to auto-detection, or the "
            f"{BACKEND_ENV} environment variable when set."
        ),
    )
    return parser.parse_args()


def _tool_exists(command: str, env: dict[str, str]) -> bool:
    return shutil.which(command, path=env["PATH"]) is not None


def _has_opencl_dev_files(env: dict[str, str]) -> bool:
    if _tool_exists("pkg-config", env):
        result = subprocess.run(
            ("pkg-config", "--exists", "OpenCL"),
            check=False,
            env=env,
        )
        if result.returncode == 0:
            return True
    return (Path("/usr/include/CL/cl.h")).exists() or (Path("/usr/local/include/CL/cl.h")).exists()


def detect_backend(env: dict[str, str]) -> str:
    if _tool_exists("nvcc", env):
        # Match upstream's preferred runtime behavior when both GPU stacks are available,
        # but still allow CUDA-only machines to build successfully.
        if _has_opencl_dev_files(env):
            return "cuda-opencl"
        return "cuda"
    if _has_opencl_dev_files(env):
        return "opencl"
    return "cpu"


def resolve_backend(requested_backend: str, env: dict[str, str]) -> str:
    backend = requested_backend
    if backend == AUTO_BACKEND:
        backend = detect_backend(env)

    if backend in {"cuda", "cuda-opencl"} and not _tool_exists("nvcc", env):
        raise SystemExit(
            "CUDA backend requested but `nvcc` was not found in PATH. Install the CUDA "
            f"toolkit or choose `{BACKEND_ENV}=opencl` / `{BACKEND_ENV}=cpu`."
        )

    if backend in {"opencl", "cuda-opencl"} and not _has_opencl_dev_files(env):
        raise SystemExit(
            "OpenCL backend requested but OpenCL development files were not found. "
            "Install `ocl-icd-opencl-dev` or choose "
            f"`{BACKEND_ENV}=cuda` / `{BACKEND_ENV}=cpu`."
        )

    return backend


def maturin_backend_args(backend: str) -> list[str]:
    if backend == "cpu":
        return ["--no-default-features", "--features", "cpu"]
    return ["--no-default-features", "--features", backend]


def main() -> int:
    args = parse_args()
    env = os.environ.copy()
    env.pop("CONDA_PREFIX", None)
    env["VIRTUAL_ENV"] = str(REPO_ROOT / ".venv")
    env["PATH"] = f"{VENV_BIN}:{CARGO_BIN}:{env.get('PATH', '')}"
    backend = resolve_backend(args.backend, env)

    if not MATURIN.exists():
        raise SystemExit(
            "maturin is not installed in the project environment. Run `uv sync --extra dev` first."
        )

    subprocess.run(
        [
            sys.executable,
            "scripts/run_upstream_rust_tests.py",
            "--bootstrap-only",
            "--skip-system-deps",
        ],
        cwd=REPO_ROOT,
        check=True,
        env=env,
    )
    print(f"Building pose_filecoin_bridge with backend `{backend}`")
    subprocess.run(
        [
            str(MATURIN),
            "develop",
            "--manifest-path",
            str(BRIDGE_MANIFEST),
            *maturin_backend_args(backend),
        ],
        cwd=REPO_ROOT,
        check=True,
        env=env,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
