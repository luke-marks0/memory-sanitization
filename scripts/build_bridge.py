#!/usr/bin/env python
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BRIDGE_MANIFEST = REPO_ROOT / "rust" / "pose_filecoin_bridge" / "Cargo.toml"
VENV_BIN = REPO_ROOT / ".venv" / "bin"
CARGO_BIN = Path.home() / ".cargo" / "bin"
MATURIN = VENV_BIN / "maturin"


def main() -> int:
    env = os.environ.copy()
    env.pop("CONDA_PREFIX", None)
    env["VIRTUAL_ENV"] = str(REPO_ROOT / ".venv")
    env["PATH"] = f"{VENV_BIN}:{CARGO_BIN}:{env.get('PATH', '')}"

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
    subprocess.run(
        [
            str(MATURIN),
            "develop",
            "--manifest-path",
            str(BRIDGE_MANIFEST),
        ],
        cwd=REPO_ROOT,
        check=True,
        env=env,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
