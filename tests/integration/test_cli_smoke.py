from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from pose.graphs import build_graph_descriptor
from pose.protocol.result_schema import bootstrap_result


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
    assert '"protocol": "graph-based PoSE-DB"' in result.stdout
    assert "supports_real_filecoin_reference" not in result.stdout


def test_bench_matrix_succeeds() -> None:
    result = run_cli("bench", "matrix", "--profiles", "bench_profiles/")
    assert result.returncode == 0
    assert "dev-small" in result.stdout
    assert "dev-small-ex-post" not in result.stdout


def test_verifier_calibrate_succeeds_for_dev_small_profile() -> None:
    result = run_cli("verifier", "calibrate", "--profile", "dev-small")
    assert result.returncode in {0, 1}
    assert '"status": "' in result.stdout
    assert '"q_bound"' in result.stdout


def test_verifier_verify_record_succeeds_for_valid_bootstrap_artifact(tmp_path: Path) -> None:
    payload = bootstrap_result("dev-small").to_dict()
    payload["session_id"] = "test-session"
    record_path = tmp_path / "result.json"
    record_path.write_text(json.dumps(payload), encoding="utf-8")

    result = run_cli("verifier", "verify-record", str(record_path))
    assert result.returncode == 0
    assert '"status": "valid"' in result.stdout


def test_verifier_verify_record_rejects_missing_required_fields(tmp_path: Path) -> None:
    payload = bootstrap_result("dev-small").to_dict()
    payload.pop("profile_name")
    record_path = tmp_path / "invalid-result.json"
    record_path.write_text(json.dumps(payload), encoding="utf-8")

    result = run_cli("verifier", "verify-record", str(record_path))
    assert result.returncode == 2
    assert "profile_name" in result.stderr


def test_verifier_run_succeeds_for_host_plan_file(tmp_path: Path) -> None:
    descriptor = build_graph_descriptor(
        label_count_m=8,
        graph_parameter_n=2,
        gamma=4,
        hash_backend="blake3-xof",
        label_width_bits=256,
    )
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text(
        f"""
session_plan:
  session_id: cli-plan-session
  session_seed_hex: "3333333333333333333333333333333333333333333333333333333333333333"
  profile_name: dev-small
  graph_family: pose-db-drg-v1
  graph_parameter_n: 2
  label_count_m: 8
  gamma: 4
  label_width_bits: 256
  hash_backend: blake3-xof
  graph_descriptor_digest: {descriptor.digest}
  challenge_policy:
    rounds_r: 4
    target_success_bound: 1.0e-9
    sample_with_replacement: true
  deadline_policy:
    response_deadline_us: 500000
    session_timeout_ms: 60000
  cleanup_policy:
    zeroize: true
    verify_zeroization: false
  adversary_model: general
  attacker_budget_bytes_assumed: 16
  q_bound: 3
  claim_notes:
    - cli-smoke
  regions:
    - region_id: host-0
      region_type: host
      usable_bytes: 256
      slot_count: 8
      covered_bytes: 256
      slack_bytes: 0
retain_session: false
""".strip(),
        encoding="utf-8",
    )

    result = run_cli("verifier", "run", "--plan", str(plan_path))
    assert result.returncode == 0
    assert '"verdict": "SUCCESS"' in result.stdout
    assert '"graph_family": "pose-db-drg-v1"' in result.stdout


def test_verifier_rechallenge_succeeds_for_retained_host_plan_file(tmp_path: Path) -> None:
    descriptor = build_graph_descriptor(
        label_count_m=8,
        graph_parameter_n=2,
        gamma=4,
        hash_backend="blake3-xof",
        label_width_bits=256,
    )
    plan_path = tmp_path / "retained-plan.yaml"
    plan_path.write_text(
        f"""
session_plan:
  session_id: cli-retained-session
  session_seed_hex: "5555555555555555555555555555555555555555555555555555555555555555"
  profile_name: dev-small
  graph_family: pose-db-drg-v1
  graph_parameter_n: 2
  label_count_m: 8
  gamma: 4
  label_width_bits: 256
  hash_backend: blake3-xof
  graph_descriptor_digest: {descriptor.digest}
  challenge_policy:
    rounds_r: 4
    target_success_bound: 1.0e-9
    sample_with_replacement: true
  deadline_policy:
    response_deadline_us: 500000
    session_timeout_ms: 60000
  cleanup_policy:
    zeroize: true
    verify_zeroization: false
  adversary_model: general
  attacker_budget_bytes_assumed: 16
  q_bound: 3
  claim_notes:
    - cli-retained
  regions:
    - region_id: host-0
      region_type: host
      usable_bytes: 256
      slot_count: 8
      covered_bytes: 256
      slack_bytes: 0
retain_session: true
""".strip(),
        encoding="utf-8",
    )

    run_result = run_cli("verifier", "run", "--plan", str(plan_path))
    assert run_result.returncode == 0
    retained_payload = json.loads(run_result.stdout)
    assert retained_payload["cleanup_status"] == "RETAINED_FOR_RECHALLENGE"

    rechallenge_result = run_cli(
        "verifier",
        "rechallenge",
        "--session-id",
        retained_payload["session_id"],
        "--release",
    )
    assert rechallenge_result.returncode == 0
    released_payload = json.loads(rechallenge_result.stdout)
    assert released_payload["run_class"] == "rechallenge"
    assert released_payload["verdict"] == "SUCCESS"
    assert released_payload["cleanup_status"] == "ZEROIZED_AND_RELEASED"
