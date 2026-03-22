from __future__ import annotations

from pathlib import Path

import pytest

from pose.benchmarks.profiles import BenchmarkProfile
from pose.common.errors import ProtocolError
from pose.graphs import build_graph_descriptor
from pose.verifier.service import VerifierService


def test_run_plan_file_executes_host_pose_db_plan(tmp_path: Path) -> None:
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
  session_id: planned-session
  session_seed_hex: "1111111111111111111111111111111111111111111111111111111111111111"
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
    - unit-test-plan
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

    result = VerifierService().run_plan_file(plan_path)

    assert result.verdict == "SUCCESS"
    assert result.success is True
    assert result.session_id == "planned-session"
    assert result.graph_family == "pose-db-drg-v1"
    assert result.rounds_r == 4
    assert result.accepted_rounds == 4
    assert result.host_covered_bytes == 256
    assert result.covered_bytes == 256
    assert result.cleanup_status == "ZEROIZED_AND_RELEASED"
    assert any("reference graph and label semantics are active" in note for note in result.notes)


def test_run_plan_file_rejects_invalid_q_gamma_margin(tmp_path: Path) -> None:
    descriptor = build_graph_descriptor(
        label_count_m=8,
        graph_parameter_n=2,
        gamma=4,
        hash_backend="blake3-xof",
        label_width_bits=256,
    )
    plan_path = tmp_path / "invalid-plan.yaml"
    plan_path.write_text(
        f"""
session_plan:
  session_id: invalid-session
  session_seed_hex: "2222222222222222222222222222222222222222222222222222222222222222"
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
  attacker_budget_bytes_assumed: 4096
  q_bound: 4
  claim_notes: []
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

    result = VerifierService().run_plan_file(plan_path)

    assert result.verdict == "CALIBRATION_INVALID"
    assert result.success is False
    assert "q=4 and gamma=4" in result.notes[-1]


def test_run_plan_file_rejects_invalid_soundness_ratio(tmp_path: Path) -> None:
    descriptor = build_graph_descriptor(
        label_count_m=8,
        graph_parameter_n=2,
        gamma=4,
        hash_backend="blake3-xof",
        label_width_bits=256,
    )
    plan_path = tmp_path / "invalid-soundness.yaml"
    plan_path.write_text(
        f"""
session_plan:
  session_id: invalid-soundness
  session_seed_hex: "4444444444444444444444444444444444444444444444444444444444444444"
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
  attacker_budget_bytes_assumed: 4096
  q_bound: 3
  claim_notes: []
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

    result = VerifierService().run_plan_file(plan_path)

    assert result.verdict == "CALIBRATION_INVALID"
    assert result.success is False
    assert "M' / m >= 1" in result.notes[-1]


def test_run_session_executes_host_profile_with_slot_planning(monkeypatch) -> None:
    profile = BenchmarkProfile.from_dict(
        {
            "name": "dev-small",
            "benchmark_class": "cold",
            "target_devices": {"host": True, "gpus": []},
            "reserve_policy": {"host_bytes": 256, "per_gpu_bytes": 0},
            "host_target_fraction": 1.0,
            "per_gpu_target_fraction": 0.0,
            "w_bits": 256,
            "graph_family": "pose-db-drg-v1",
            "hash_backend": "blake3-xof",
            "adversary_model": "general",
            "attacker_budget_bytes_assumed": 16,
            "challenge_policy": {
                "rounds_r": 4,
                "target_success_bound": 1.0e-9,
                "sample_with_replacement": True,
            },
            "deadline_policy": {"response_deadline_us": 500000, "session_timeout_ms": 60000},
            "calibration_policy": {
                "lookup_samples": 32,
                "hash_measurement_rounds": 1,
                "hashes_per_round": 64,
                "transport_overhead_us": 100,
                "serialization_overhead_us": 50,
                "safety_margin_fraction": 0.25,
            },
            "cleanup_policy": {"zeroize": True, "verify_zeroization": False},
            "repetition_count": 1,
        }
    )
    descriptor = build_graph_descriptor(
        label_count_m=8,
        graph_parameter_n=2,
        gamma=4,
        hash_backend="blake3-xof",
        label_width_bits=256,
    )
    monkeypatch.setattr(
        "pose.verifier.service.calibrate_profile",
        lambda _profile, persist_artifact=True: {
            "status": "calibrated",
            "artifact_path": "/tmp/test-calibration.json",
            "planning": {
                "profile_name": "dev-small",
                "graph_family": "pose-db-drg-v1",
                "hash_backend": "blake3-xof",
                "graph_parameter_n": 2,
                "graph_descriptor_digest": descriptor.digest,
                "label_count_m": 8,
                "gamma": 4,
                "w_bits": 256,
                "w_bytes": 32,
                "base_attacker_budget_bytes_assumed": 16,
                "effective_attacker_budget_bytes_assumed": 80,
                "untargeted_local_tier_bytes_auto_included": 64,
                "covered_bytes": 256,
                "slack_bytes": 0,
                "total_usable_bytes": 256,
                "host_total_bytes": 256,
                "host_budget_bytes": 256,
                "host_usable_bytes": 256,
                "host_covered_bytes": 256,
                "gpu_total_bytes_by_device": {},
                "gpu_budget_bytes_by_device": {},
                "gpu_usable_bytes_by_device": {},
                "gpu_covered_bytes_by_device": {},
                "regions": [
                    {
                        "region_id": "host-0",
                        "region_type": "host",
                        "total_bytes": 256,
                        "budget_bytes": 256,
                        "usable_bytes": 256,
                        "slot_count": 8,
                        "covered_bytes": 256,
                        "slack_bytes": 0,
                        "gpu_device": None,
                    }
                ],
            },
            "q_bound": 3,
            "rounds_r": 4,
            "soundness": {
                "soundness_model": "random-oracle + distant-attacker + calibrated q<gamma",
            },
            "notes": [],
        },
    )

    result = VerifierService().run_session(profile)

    assert result.verdict == "SUCCESS"
    assert result.success is True
    assert result.graph_family == "pose-db-drg-v1"
    assert result.covered_bytes == 256
    assert result.accepted_rounds == 4
    assert result.attacker_budget_bytes_assumed == 80
    assert result.scratch_peak_bytes >= 0
    assert result.round_trip_p50_us > 0
    assert result.round_trip_p95_us >= result.round_trip_p50_us
    assert result.round_trip_p99_us >= result.round_trip_p95_us
    assert result.max_round_trip_us >= result.round_trip_p99_us
    assert result.formal_claim_notes
    assert result.operational_claim_notes
    assert any("profile-driven slot planning via dev-small" in note for note in result.notes)


def test_run_session_can_retain_and_rechallenge_host_profile(monkeypatch) -> None:
    profile = BenchmarkProfile.from_dict(
        {
            "name": "dev-small",
            "benchmark_class": "cold",
            "target_devices": {"host": True, "gpus": []},
            "reserve_policy": {"host_bytes": 256, "per_gpu_bytes": 0},
            "host_target_fraction": 1.0,
            "per_gpu_target_fraction": 0.0,
            "w_bits": 256,
            "graph_family": "pose-db-drg-v1",
            "hash_backend": "blake3-xof",
            "adversary_model": "general",
            "attacker_budget_bytes_assumed": 16,
            "challenge_policy": {
                "rounds_r": 4,
                "target_success_bound": 1.0e-9,
                "sample_with_replacement": True,
            },
            "deadline_policy": {"response_deadline_us": 500000, "session_timeout_ms": 60000},
            "calibration_policy": {
                "lookup_samples": 32,
                "hash_measurement_rounds": 1,
                "hashes_per_round": 64,
                "transport_overhead_us": 100,
                "serialization_overhead_us": 50,
                "safety_margin_fraction": 0.25,
            },
            "cleanup_policy": {"zeroize": True, "verify_zeroization": False},
            "repetition_count": 1,
        }
    )
    descriptor = build_graph_descriptor(
        label_count_m=8,
        graph_parameter_n=2,
        gamma=4,
        hash_backend="blake3-xof",
        label_width_bits=256,
    )
    monkeypatch.setattr(
        "pose.verifier.service.calibrate_profile",
        lambda _profile, persist_artifact=True: {
            "status": "calibrated",
            "artifact_path": "/tmp/test-calibration-retain.json",
            "planning": {
                "profile_name": "dev-small",
                "graph_family": "pose-db-drg-v1",
                "hash_backend": "blake3-xof",
                "graph_parameter_n": 2,
                "graph_descriptor_digest": descriptor.digest,
                "label_count_m": 8,
                "gamma": 4,
                "w_bits": 256,
                "w_bytes": 32,
                "covered_bytes": 256,
                "slack_bytes": 0,
                "total_usable_bytes": 256,
                "host_total_bytes": 256,
                "host_budget_bytes": 256,
                "host_usable_bytes": 256,
                "host_covered_bytes": 256,
                "gpu_total_bytes_by_device": {},
                "gpu_budget_bytes_by_device": {},
                "gpu_usable_bytes_by_device": {},
                "gpu_covered_bytes_by_device": {},
                "regions": [
                    {
                        "region_id": "host-0",
                        "region_type": "host",
                        "total_bytes": 256,
                        "budget_bytes": 256,
                        "usable_bytes": 256,
                        "slot_count": 8,
                        "covered_bytes": 256,
                        "slack_bytes": 0,
                        "gpu_device": None,
                    }
                ],
            },
            "q_bound": 3,
            "rounds_r": 4,
            "soundness": {
                "soundness_model": "random-oracle + distant-attacker + calibrated q<gamma",
            },
            "notes": [],
        },
    )

    service = VerifierService()
    retained = service.run_session(profile, retain_session=True)
    try:
        assert retained.verdict == "SUCCESS"
        assert retained.success is True
        assert retained.cleanup_status == "RETAINED_FOR_RECHALLENGE"
        assert retained.resident_socket_path
        assert retained.resident_process_id > 0
        rechallenge = service.rechallenge(retained.session_id, release=True)
    finally:
        if retained.cleanup_status == "RETAINED_FOR_RECHALLENGE":
            try:
                service.rechallenge(retained.session_id, release=True)
            except ProtocolError:
                pass

    assert rechallenge.verdict == "SUCCESS"
    assert rechallenge.success is True
    assert rechallenge.run_class == "rechallenge"
    assert rechallenge.cleanup_status == "ZEROIZED_AND_RELEASED"


def test_run_session_rejects_profile_below_coverage_threshold(monkeypatch) -> None:
    profile = BenchmarkProfile.from_dict(
        {
            "name": "dev-threshold",
            "benchmark_class": "cold",
            "target_devices": {"host": True, "gpus": []},
            "reserve_policy": {"host_bytes": 256, "per_gpu_bytes": 0},
            "host_target_fraction": 0.5,
            "per_gpu_target_fraction": 0.0,
            "w_bits": 256,
            "graph_family": "pose-db-drg-v1",
            "hash_backend": "blake3-xof",
            "adversary_model": "general",
            "attacker_budget_bytes_assumed": 16,
            "challenge_policy": {
                "rounds_r": 4,
                "target_success_bound": 1.0e-9,
                "sample_with_replacement": True,
            },
            "deadline_policy": {"response_deadline_us": 500000, "session_timeout_ms": 60000},
            "calibration_policy": {
                "lookup_samples": 32,
                "hash_measurement_rounds": 1,
                "hashes_per_round": 64,
                "transport_overhead_us": 100,
                "serialization_overhead_us": 50,
                "safety_margin_fraction": 0.25,
            },
            "cleanup_policy": {"zeroize": True, "verify_zeroization": False},
            "repetition_count": 1,
            "coverage_threshold": 1.0,
        }
    )
    descriptor = build_graph_descriptor(
        label_count_m=4,
        graph_parameter_n=1,
        gamma=2,
        hash_backend="blake3-xof",
        label_width_bits=256,
    )
    monkeypatch.setattr(
        "pose.verifier.service.calibrate_profile",
        lambda _profile, persist_artifact=True: {
            "status": "calibrated",
            "artifact_path": "/tmp/test-calibration-threshold.json",
            "planning": {
                "profile_name": "dev-threshold",
                "graph_family": "pose-db-drg-v1",
                "hash_backend": "blake3-xof",
                "graph_parameter_n": 1,
                "graph_descriptor_digest": descriptor.digest,
                "label_count_m": 4,
                "gamma": 2,
                "w_bits": 256,
                "w_bytes": 32,
                "covered_bytes": 128,
                "slack_bytes": 128,
                "total_usable_bytes": 256,
                "host_total_bytes": 256,
                "host_budget_bytes": 256,
                "host_usable_bytes": 256,
                "host_covered_bytes": 128,
                "gpu_total_bytes_by_device": {},
                "gpu_budget_bytes_by_device": {},
                "gpu_usable_bytes_by_device": {},
                "gpu_covered_bytes_by_device": {},
                "regions": [
                    {
                        "region_id": "host-0",
                        "region_type": "host",
                        "total_bytes": 256,
                        "budget_bytes": 256,
                        "usable_bytes": 256,
                        "slot_count": 4,
                        "covered_bytes": 128,
                        "slack_bytes": 128,
                        "gpu_device": None,
                    }
                ],
            },
            "q_bound": 1,
            "rounds_r": 4,
            "soundness": {
                "soundness_model": "random-oracle + distant-attacker + calibrated q<gamma",
            },
            "notes": [],
        },
    )

    result = VerifierService().run_session(profile)

    assert result.verdict == "COVERAGE_BELOW_THRESHOLD"
    assert result.success is False
    assert result.coverage_fraction == 0.5
    assert "coverage_threshold" in result.notes[-1]


def test_rechallenge_rejects_unknown_resident_session() -> None:
    with pytest.raises(ProtocolError, match="Unknown retained PoSE-DB session"):
        VerifierService().rechallenge("resident-session")
