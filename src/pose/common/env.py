from __future__ import annotations

import platform
import shutil
import sys
from pathlib import Path


def capture_environment() -> dict[str, str]:
    repo_root = Path(__file__).resolve().parents[3]
    payload = {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "uv": shutil.which("uv") or "missing",
    }
    cargo = shutil.which("cargo")
    if cargo is not None:
        payload["cargo"] = cargo
    return payload
