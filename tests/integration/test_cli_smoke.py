from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root / "src")
    return subprocess.run(
        [sys.executable, "-m", "pose.cli.main", *args],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_prover_inspect_succeeds() -> None:
    result = run_cli("prover", "inspect")
    assert result.returncode == 0
    assert "supports_real_filecoin_reference" in result.stdout


def test_bench_matrix_succeeds() -> None:
    result = run_cli("bench", "matrix", "--profiles", "bench_profiles/")
    assert result.returncode == 0
    assert "dev-small" in result.stdout
