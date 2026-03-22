from __future__ import annotations

from pathlib import Path

from pose.benchmarks.calibration import prepare_calibration
from pose.verifier.slot_planning import plan_slot_layout


class _HugeUntargetedGpuRuntime:
    def device_count(self) -> int:
        return 1

    def mem_get_info(self, device: int) -> tuple[int, int]:
        del device
        return (1_000_000, 1_000_000)


def _write_profile(path: Path) -> None:
    path.write_text(
        """
name: bad-memory-accounting
benchmark_class: cold
target_devices:
  host: true
  gpus: []
reserve_policy:
  host_bytes: 65536
  per_gpu_bytes: 0
host_target_fraction: 1.0
per_gpu_target_fraction: 0.0
w_bits: 256
graph_family: pose-db-drg-v1
hash_backend: blake3-xof
adversary_model: general
attacker_budget_bytes_assumed: 16384
challenge_policy:
  rounds_r: 8
  target_success_bound: 0.0
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


def test_prepare_calibration_rejects_hidden_untargeted_tier_when_m_accounting_breaks_soundness(
    monkeypatch,
    tmp_path: Path,
) -> None:
    profile_path = tmp_path / "profile.yaml"
    artifact_dir = tmp_path / "artifacts"
    _write_profile(profile_path)

    monkeypatch.setattr("pose.benchmarks.calibration.calibration_root", lambda: artifact_dir)
    monkeypatch.setattr(
        "pose.benchmarks.calibration.plan_slot_layout",
        lambda profile: plan_slot_layout(profile, detected_host_bytes=65_536),
    )
    monkeypatch.setattr("pose.benchmarks.calibration.get_cuda_runtime", lambda: _HugeUntargetedGpuRuntime())
    monkeypatch.setattr(
        "pose.benchmarks.calibration._measure_host_lookup_latency_us",
        lambda **_kwargs: {"mean": 10.0, "p50": 10.0, "p95": 12.0, "p99": 14.0, "max": 14.0},
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
        lambda **_kwargs: 200_000.0,
    )

    payload = prepare_calibration(str(profile_path))

    assert payload["status"] == "calibration-invalid"
    assert payload["planning"]["effective_attacker_budget_bytes_assumed"] > payload["planning"]["base_attacker_budget_bytes_assumed"]
    assert any("untargeted_local_gpu_tier=" in note for note in payload["planning"]["claim_notes"])
    assert any("M' / m >= 1" in note for note in payload["notes"])
