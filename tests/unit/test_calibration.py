from __future__ import annotations

from pathlib import Path

from pose.benchmarks.calibration import calibrate_profile, prepare_calibration
from pose.benchmarks.profiles import load_profile
from pose.verifier.slot_planning import plan_slot_layout


class _FakeCudaRuntime:
    def device_count(self) -> int:
        return 2

    def mem_get_info(self, device: int) -> tuple[int, int]:
        if device == 0:
            return (256, 512)
        return (128, 256)


def _write_profile(path: Path) -> None:
    path.write_text(
        """
name: test-calibration
benchmark_class: cold
target_devices:
  host: true
  gpus: []
reserve_policy:
  host_bytes: 1048576
  per_gpu_bytes: 0
host_target_fraction: 1.0
per_gpu_target_fraction: 0.0
w_bits: 256
graph_family: pose-db-drg-v1
hash_backend: blake3-xof
adversary_model: general
attacker_budget_bytes_assumed: 16384
challenge_policy:
  rounds_r: 64
  target_success_bound: 1.0e-9
  sample_with_replacement: true
deadline_policy:
  response_deadline_us: 2500
  session_timeout_ms: 60000
calibration_policy:
  lookup_samples: 32
  hash_measurement_rounds: 1
  hashes_per_round: 64
  transport_overhead_us: 100
  serialization_overhead_us: 50
  safety_margin_fraction: 0.25
cleanup_policy:
  zeroize: true
  verify_zeroization: false
repetition_count: 1
transport_mode: grpc
coverage_threshold: 1.0
""".strip(),
        encoding="utf-8",
    )


def test_prepare_calibration_writes_valid_artifact(monkeypatch, tmp_path: Path) -> None:
    profile_path = tmp_path / "profile.yaml"
    _write_profile(profile_path)
    artifact_dir = tmp_path / "artifacts"

    monkeypatch.setattr("pose.benchmarks.calibration.calibration_root", lambda: artifact_dir)
    monkeypatch.setattr(
        "pose.benchmarks.calibration.plan_slot_layout",
        lambda profile: plan_slot_layout(profile, detected_host_bytes=1_048_576),
    )
    monkeypatch.setattr("pose.benchmarks.calibration.get_cuda_runtime", lambda: _FakeCudaRuntime())
    monkeypatch.setattr(
        "pose.benchmarks.calibration._measure_host_lookup_latency_us",
        lambda **kwargs: {"mean": 10.0, "p50": 10.0, "p95": 12.0, "p99": 14.0, "max": 14.0},
    )
    monkeypatch.setattr(
        "pose.benchmarks.calibration._measure_grpc_fast_phase_transport_us",
        lambda *_args, **_kwargs: {
            "fast_phase_round_trip_us": {"mean": 90.0, "p50": 88.0, "p95": 100.0, "p99": 110.0, "max": 110.0},
            "fast_phase_prover_lookup_round_trip_us": {"mean": 8.0, "p50": 8.0, "p95": 10.0, "p99": 12.0, "max": 12.0},
            "fast_phase_transport_overhead_us": {"mean": 82.0, "p50": 80.0, "p95": 90.0, "p99": 98.0, "max": 98.0},
        },
    )
    monkeypatch.setattr(
        "pose.benchmarks.calibration._measure_hash_evaluations_per_second",
        lambda **kwargs: 200_000.0,
    )

    payload = prepare_calibration(str(profile_path))

    assert payload["status"] == "calibrated"
    assert payload["q_bound"] > 0
    assert payload["q_bound"] < payload["gamma"]
    assert Path(str(payload["artifact_path"])).exists()
    assert payload["planning"]["label_count_m"] == 32_768
    assert any("untargeted_local_gpu_tier=" in note for note in payload["notes"])
    assert any("untargeted_local_gpu_tier=" in note for note in payload["planning"]["claim_notes"])
    assert payload["planning"]["base_attacker_budget_bytes_assumed"] == 16_384
    assert payload["planning"]["untargeted_local_tier_bytes_auto_included"] == 384
    assert payload["planning"]["effective_attacker_budget_bytes_assumed"] == 16_768
    assert payload["soundness"]["attacker_budget_bits_assumed"] == 16_768 * 8
    assert payload["measurements"]["fast_phase_transport_overhead_us"]["p95"] == 90.0
    assert payload["measurements"]["effective_transport_overhead_p95_us"] == 240.0
    assert payload["soundness"]["soundness_model"] == "random-oracle + distant-attacker + calibrated q<gamma"


def test_prepare_calibration_reports_invalid_q_margin(monkeypatch, tmp_path: Path) -> None:
    profile_path = tmp_path / "profile.yaml"
    _write_profile(profile_path)
    artifact_dir = tmp_path / "artifacts"

    monkeypatch.setattr("pose.benchmarks.calibration.calibration_root", lambda: artifact_dir)
    monkeypatch.setattr(
        "pose.benchmarks.calibration.plan_slot_layout",
        lambda profile: plan_slot_layout(profile, detected_host_bytes=1_048_576),
    )
    monkeypatch.setattr("pose.benchmarks.calibration.get_cuda_runtime", lambda: _FakeCudaRuntime())
    monkeypatch.setattr(
        "pose.benchmarks.calibration._measure_host_lookup_latency_us",
        lambda **kwargs: {"mean": 1.0, "p50": 1.0, "p95": 1.0, "p99": 1.0, "max": 1.0},
    )
    monkeypatch.setattr(
        "pose.benchmarks.calibration._measure_grpc_fast_phase_transport_us",
        lambda *_args, **_kwargs: {
            "fast_phase_round_trip_us": {"mean": 2.0, "p50": 2.0, "p95": 2.0, "p99": 2.0, "max": 2.0},
            "fast_phase_prover_lookup_round_trip_us": {"mean": 1.0, "p50": 1.0, "p95": 1.0, "p99": 1.0, "max": 1.0},
            "fast_phase_transport_overhead_us": {"mean": 1.0, "p50": 1.0, "p95": 1.0, "p99": 1.0, "max": 1.0},
        },
    )
    monkeypatch.setattr(
        "pose.benchmarks.calibration._measure_hash_evaluations_per_second",
        lambda **kwargs: 10_000_000_000.0,
    )

    payload = prepare_calibration(str(profile_path))

    assert payload["status"] == "calibration-invalid"
    assert any("q_bound must be strictly less than gamma" in note for note in payload["notes"])
    assert Path(str(payload["artifact_path"])).exists()


def test_calibrate_profile_persists_for_ad_hoc_profile_object(monkeypatch, tmp_path: Path) -> None:
    profile_path = tmp_path / "profile.yaml"
    _write_profile(profile_path)
    artifact_dir = tmp_path / "artifacts"

    monkeypatch.setattr("pose.benchmarks.calibration.calibration_root", lambda: artifact_dir)
    monkeypatch.setattr(
        "pose.benchmarks.calibration.plan_slot_layout",
        lambda profile: plan_slot_layout(profile, detected_host_bytes=1_048_576),
    )
    monkeypatch.setattr("pose.benchmarks.calibration.get_cuda_runtime", lambda: _FakeCudaRuntime())
    monkeypatch.setattr(
        "pose.benchmarks.calibration._measure_host_lookup_latency_us",
        lambda **kwargs: {"mean": 10.0, "p50": 10.0, "p95": 12.0, "p99": 14.0, "max": 14.0},
    )
    monkeypatch.setattr(
        "pose.benchmarks.calibration._measure_grpc_fast_phase_transport_us",
        lambda *_args, **_kwargs: {
            "fast_phase_round_trip_us": {"mean": 90.0, "p50": 88.0, "p95": 100.0, "p99": 110.0, "max": 110.0},
            "fast_phase_prover_lookup_round_trip_us": {"mean": 8.0, "p50": 8.0, "p95": 10.0, "p99": 12.0, "max": 12.0},
            "fast_phase_transport_overhead_us": {"mean": 82.0, "p50": 80.0, "p95": 90.0, "p99": 98.0, "max": 98.0},
        },
    )
    monkeypatch.setattr(
        "pose.benchmarks.calibration._measure_hash_evaluations_per_second",
        lambda **kwargs: 200_000.0,
    )

    payload = calibrate_profile(load_profile(str(profile_path)), persist_artifact=True)

    assert payload["status"] == "calibrated"
    assert Path(str(payload["artifact_path"])).exists()
    assert any("untargeted_local_gpu_tier=" in note for note in payload["planning"]["claim_notes"])
    assert payload["planning"]["effective_attacker_budget_bytes_assumed"] == 16_768
