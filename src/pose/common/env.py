from __future__ import annotations

import platform
import shutil
import sys


def capture_environment() -> dict[str, str]:
    return {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "cargo": shutil.which("cargo") or "missing",
        "uv": shutil.which("uv") or "missing",
        "rust_toolchain": "pinned-via-rust-toolchain.toml",
    }

