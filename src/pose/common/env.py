from __future__ import annotations

import platform
import shutil
import sys
from pathlib import Path


def capture_environment() -> dict[str, str]:
    repo_root = Path(__file__).resolve().parents[3]
    rust_toolchain = (repo_root / "rust-toolchain.toml").read_text(encoding="utf-8")
    payload = {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "cargo": shutil.which("cargo") or "missing",
        "uv": shutil.which("uv") or "missing",
        "rust_toolchain": rust_toolchain,
    }
    return payload
