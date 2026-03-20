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


def test_verifier_verify_record_succeeds_for_valid_bootstrap_artifact(tmp_path: Path) -> None:
    payload = {
        "success": False,
        "verdict": "PROTOCOL_ERROR",
        "session_id": "test-session",
        "profile_name": "dev-small",
        "timings_ms": {
            "discover": 0,
            "region_leasing": 0,
            "allocation": 0,
            "data_generation": 0,
            "seal_pre_commit_phase1": 0,
            "seal_pre_commit_phase2": 0,
            "seal_commit_phase1": 0,
            "seal_commit_phase2": 0,
            "object_serialization": 0,
            "copy_to_host": 0,
            "copy_to_hbm": 0,
            "outer_tree_build": 0,
            "inner_verify": 0,
            "challenge_response": 0,
            "outer_verify": 0,
            "cleanup": 0,
            "total": 0,
        },
    }
    record_path = tmp_path / "result.json"
    record_path.write_text(__import__("json").dumps(payload), encoding="utf-8")

    result = run_cli("verifier", "verify-record", str(record_path))
    assert result.returncode == 0
    assert '"status": "valid"' in result.stdout
