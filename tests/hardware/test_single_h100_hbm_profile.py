from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    root = _repo_root()
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


def _require_single_h100() -> None:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
            capture_output=True,
            check=False,
        )
    except FileNotFoundError:
        pytest.skip("nvidia-smi is unavailable on this host")
    if completed.returncode != 0:
        pytest.skip("nvidia-smi is unavailable on this host")
    gpu_names = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if len(gpu_names) != 1:
        pytest.skip(f"single-H100 hardware test requires exactly one visible GPU, found {len(gpu_names)}")
    if "H100" not in gpu_names[0]:
        pytest.skip(f"single-H100 hardware test requires an H100, found {gpu_names[0]!r}")


def test_single_h100_hbm_profile_calibrates_on_target_hardware() -> None:
    _require_single_h100()

    result = _run_cli("verifier", "calibrate", "--profile", "single-h100-hbm-max")

    payload = json.loads(result.stdout)
    assert payload["profile"]["name"] == "single-h100-hbm-max"
    assert int(payload["planning"]["gpu_covered_bytes_by_device"]["0"]) > 0
    assert any("untargeted_local_host_tier=" in note for note in payload["planning"]["claim_notes"])
    if payload["status"] == "calibrated":
        assert result.returncode == 0, result.stderr
        assert payload["q_bound"] < payload["gamma"]
        return

    assert result.returncode == 1, result.stderr
    assert payload["status"] == "calibration-invalid"
    assert any("M' / m >= 1" in note for note in payload["notes"])


def test_single_h100_hbm_profile_runs_on_target_hardware() -> None:
    _require_single_h100()

    result = _run_cli("verifier", "run", "--profile", "single-h100-hbm-max")

    payload = json.loads(result.stdout)
    assert payload["profile_name"] == "single-h100-hbm-max"
    assert payload["coverage_fraction"] >= 0.9
    if payload["verdict"] == "SUCCESS":
        assert result.returncode == 0, result.stderr
        assert payload["host_covered_bytes"] == 0
        assert int(payload["gpu_covered_bytes_by_device"]["0"]) > 0
        assert payload["q_bound"] < payload["gamma"]
        assert any("GPU HBM regions" in note for note in payload["operational_claim_notes"])
        return

    assert result.returncode == 1, result.stderr
    assert payload["verdict"] == "CALIBRATION_INVALID"
    assert any("untargeted_local_host_tier=" in note for note in payload["notes"])
    assert any("M' / m >= 1" in note for note in payload["notes"])


def test_single_h100_hbm_small_profile_runs_in_development_mode() -> None:
    _require_single_h100()

    result = _run_cli("verifier", "run", "--profile", "single-h100-hbm-small")

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["verdict"] == "SUCCESS"
    assert payload["profile_name"] == "single-h100-hbm-small"
    assert payload["host_covered_bytes"] == 0
    assert int(payload["gpu_covered_bytes_by_device"]["0"]) > 0
    assert payload["q_bound"] < payload["gamma"]
    assert any("development_only_not_for_production=true" in note for note in payload["claim_notes"])
    assert any("GPU HBM regions" in note for note in payload["operational_claim_notes"])


def test_single_h100_hybrid_small_profile_runs_in_development_mode() -> None:
    _require_single_h100()

    result = _run_cli("verifier", "run", "--profile", "single-h100-hybrid-small")

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["verdict"] == "SUCCESS"
    assert payload["profile_name"] == "single-h100-hybrid-small"
    assert payload["host_covered_bytes"] > 0
    assert int(payload["gpu_covered_bytes_by_device"]["0"]) > 0
    assert payload["coverage_fraction"] == 1.0
    assert payload["q_bound"] < payload["gamma"]
    assert any("development_only_not_for_production=true" in note for note in payload["claim_notes"])
    assert any("mixed host/HBM regions" in note for note in payload["operational_claim_notes"])
