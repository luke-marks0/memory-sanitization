from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from pose.benchmarks.calibration import (
    _measure_gpu_lookup_latency_us,
    _measure_grpc_fast_phase_transport_us,
    _transport_measurement_plan,
    calibrate_profile,
    prepare_calibration,
)
from pose.benchmarks.profiles import BenchmarkProfile, load_profile
from pose.protocol.messages import CleanupPolicy, LeaseRecord
from pose.verifier.slot_planning import plan_slot_layout


class _FakeCudaRuntime:
    def device_count(self) -> int:
        return 2

    def mem_get_info(self, device: int) -> tuple[int, int]:
        if device == 0:
            return (256, 512)
        return (128, 256)


class _SingleGpuRuntime:
    def device_count(self) -> int:
        return 1

    def mem_get_info(self, device: int) -> tuple[int, int]:
        assert device == 0
        return (1_048_576, 2_097_152)


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


def _write_hbm_profile(path: Path) -> None:
    path.write_text(
        """
name: test-hbm-calibration
benchmark_class: cold
target_devices:
  host: false
  gpus: [0]
reserve_policy:
  host_bytes: 1048576
  per_gpu_bytes: 0
host_target_fraction: 0.0
per_gpu_target_fraction: 1.0
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
coverage_threshold: 0.9
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
    monkeypatch.setattr("pose.benchmarks.calibration.detect_host_memory_bytes", lambda: 1_048_576)
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
    monkeypatch.setattr("pose.benchmarks.calibration.detect_host_memory_bytes", lambda: 1_048_576)
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
    monkeypatch.setattr("pose.benchmarks.calibration.detect_host_memory_bytes", lambda: 1_048_576)
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


def test_transport_measurement_plan_uses_gpu_region_for_hbm_only_profile(tmp_path: Path) -> None:
    profile_path = tmp_path / "hbm-profile.yaml"
    _write_hbm_profile(profile_path)
    profile = load_profile(str(profile_path))
    layout = plan_slot_layout(profile, detected_gpu_bytes_by_device={0: (1_048_576, 2_097_152)})

    measurement_plan = _transport_measurement_plan(profile, measurement_region=layout.regions[0])

    assert len(measurement_plan.regions) == 1
    assert measurement_plan.regions[0].region_id == "gpu-0"
    assert measurement_plan.regions[0].region_type == "gpu"
    assert measurement_plan.regions[0].gpu_device == 0


def test_prepare_calibration_uses_gpu_lookup_for_hbm_only_profile(monkeypatch, tmp_path: Path) -> None:
    profile_path = tmp_path / "hbm-profile.yaml"
    _write_hbm_profile(profile_path)
    artifact_dir = tmp_path / "artifacts"

    monkeypatch.setattr("pose.benchmarks.calibration.calibration_root", lambda: artifact_dir)
    monkeypatch.setattr(
        "pose.benchmarks.calibration.plan_slot_layout",
        lambda profile: plan_slot_layout(
            profile,
            detected_gpu_bytes_by_device={0: (1_048_576, 2_097_152)},
        ),
    )
    monkeypatch.setattr("pose.benchmarks.calibration.detect_host_memory_bytes", lambda: 65_536)
    monkeypatch.setattr("pose.benchmarks.calibration.get_cuda_runtime", lambda: _SingleGpuRuntime())
    monkeypatch.setattr(
        "pose.benchmarks.calibration._measure_host_lookup_latency_us",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("host lookup should not be used for HBM-only profiles")),
    )
    monkeypatch.setattr(
        "pose.benchmarks.calibration._measure_gpu_lookup_latency_us",
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

    assert payload["status"] == "calibrated"
    assert payload["planning"]["gpu_covered_bytes_by_device"] == {"0": 1_048_576}
    assert payload["measurements"]["resident_lookup_latency_us"]["p95"] == 12.0
    assert any("untargeted_local_host_tier=" in note for note in payload["planning"]["claim_notes"])


def test_gpu_lookup_measurement_uses_direct_gpu_lease_reads(monkeypatch) -> None:
    class _FakeGpuLease:
        def __init__(self, usable_bytes: int) -> None:
            self.record = LeaseRecord(
                region_id="gpu-0",
                region_type="gpu",
                usable_bytes=usable_bytes,
                lease_handle="cuda-ipc:0:ZmFrZQ==",
                lease_expiry="2099-01-01T00:00:00+00:00",
                cleanup_policy=CleanupPolicy(zeroize=True, verify_zeroization=False),
            )
            self._buffer = bytearray(usable_bytes)
            self.read_calls: list[tuple[int, int]] = []

        def write(self, payload: bytes) -> None:
            self._buffer[: len(payload)] = payload

        def read(self, length: int | None = None, offset: int = 0) -> bytes:
            requested = self.record.usable_bytes if length is None else length
            self.read_calls.append((requested, offset))
            return bytes(self._buffer[offset : offset + requested])

    captured: dict[str, object] = {}
    fake_lease = _FakeGpuLease(4096)

    def _fake_create_gpu_lease(**kwargs):
        captured["create_gpu_lease"] = kwargs
        return fake_lease

    def _fake_release_gpu_lease(lease, **kwargs):
        captured["release_gpu_lease"] = (lease, kwargs)
        return "ZEROIZED_AND_RELEASED"

    monkeypatch.setattr("pose.benchmarks.calibration.create_gpu_lease", _fake_create_gpu_lease)
    monkeypatch.setattr("pose.benchmarks.calibration.release_gpu_lease", _fake_release_gpu_lease)

    payload = _measure_gpu_lookup_latency_us(device=0, w_bytes=32, sample_count=8)

    assert captured["create_gpu_lease"]["device"] == 0
    assert len(fake_lease.read_calls) == 8
    assert all(call[0] == 32 for call in fake_lease.read_calls)
    assert captured["release_gpu_lease"][0] is fake_lease
    assert payload["p95"] >= 0.0


def test_grpc_transport_measurement_uses_gpu_lease_for_hbm_only_profile(monkeypatch) -> None:
    profile = BenchmarkProfile.from_dict(
        {
            "name": "transport-gpu",
            "benchmark_class": "cold",
            "target_devices": {"host": False, "gpus": [0]},
            "reserve_policy": {"host_bytes": 0, "per_gpu_bytes": 0},
            "host_target_fraction": 0.0,
            "per_gpu_target_fraction": 1.0,
            "w_bits": 256,
            "graph_family": "pose-db-drg-v1",
            "hash_backend": "blake3-xof",
            "adversary_model": "general",
            "attacker_budget_bytes_assumed": 16_384,
            "challenge_policy": {
                "rounds_r": 64,
                "target_success_bound": 1.0e-9,
                "sample_with_replacement": True,
            },
            "deadline_policy": {"response_deadline_us": 2_500, "session_timeout_ms": 60_000},
            "calibration_policy": {
                "lookup_samples": 16,
                "hash_measurement_rounds": 1,
                "hashes_per_round": 64,
                "transport_overhead_us": 100,
                "serialization_overhead_us": 50,
                "safety_margin_fraction": 0.25,
            },
            "cleanup_policy": {"zeroize": True, "verify_zeroization": False},
            "repetition_count": 1,
            "transport_mode": "grpc",
            "coverage_threshold": 0.9,
        }
    )
    layout = plan_slot_layout(profile, detected_gpu_bytes_by_device={0: (1_048_576, 2_097_152)})

    class _FakeProcess:
        pid = 1234

        def poll(self) -> int:
            return 0

        def terminate(self) -> None:
            return None

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            return 0

        def kill(self) -> None:
            return None

    captured: dict[str, object] = {}
    fake_lease = SimpleNamespace(
        record=LeaseRecord(
            region_id="gpu-0",
            region_type="gpu",
            usable_bytes=8 * profile.w_bytes,
            lease_handle="cuda-ipc:0:ZmFrZQ==",
            lease_expiry="2099-01-01T00:00:00+00:00",
            cleanup_policy=CleanupPolicy(zeroize=True, verify_zeroization=False),
        ),
    )

    class _FakeFastPhaseClient:
        def __init__(self, _socket_path: str) -> None:
            return None

        def __enter__(self) -> "_FakeFastPhaseClient":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            del exc_type, exc, tb

        def run_round(
            self,
            *,
            session_id: str,
            round_index: int,
            challenge_index: int,
        ) -> dict[str, object]:
            del session_id, round_index
            return {
                "region_id": "gpu-0",
                "challenge_index": challenge_index,
                "label_bytes": b"",
                "round_trip_us": 200,
                "prover_lookup_round_trip_us": 20,
            }

    def _fake_create_gpu_lease(**kwargs):
        captured["create_gpu_lease"] = kwargs
        return fake_lease

    def _fake_release_gpu_lease(lease, **kwargs):
        captured["release_gpu_lease"] = (lease, kwargs)
        return "ZEROIZED_AND_RELEASED"

    monkeypatch.setattr(
        "pose.benchmarks.calibration.start_ephemeral_prover_server",
        lambda **_kwargs: _FakeProcess(),
    )
    monkeypatch.setattr(
        "pose.benchmarks.calibration.create_host_lease",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("host lease should not be used for HBM-only profiles")),
    )
    monkeypatch.setattr(
        "pose.benchmarks.calibration.create_gpu_lease",
        _fake_create_gpu_lease,
    )
    monkeypatch.setattr("pose.benchmarks.calibration.plan_session", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "pose.benchmarks.calibration.lease_regions",
        lambda _socket_path, _session_id, leases: captured.setdefault("leased_region", leases[0]),
    )
    monkeypatch.setattr("pose.benchmarks.calibration.seed_session", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("pose.benchmarks.calibration.materialize_labels", lambda *_args, **_kwargs: ({}, {}))
    monkeypatch.setattr("pose.benchmarks.calibration.prepare_fast_phase", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("pose.benchmarks.calibration.finalize_session", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "pose.benchmarks.calibration.cleanup_session",
        lambda *_args, **_kwargs: "ZEROIZED_AND_RELEASED",
    )
    monkeypatch.setattr(
        "pose.benchmarks.calibration.release_host_lease",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("host lease should not be released for HBM-only profiles")),
    )
    monkeypatch.setattr(
        "pose.benchmarks.calibration.release_gpu_lease",
        _fake_release_gpu_lease,
    )
    monkeypatch.setattr("pose.benchmarks.calibration.FastPhaseClient", _FakeFastPhaseClient)

    payload = _measure_grpc_fast_phase_transport_us(
        profile,
        measurement_region=layout.regions[0],
        sample_count=8,
    )

    leased_region = captured["leased_region"]
    assert captured["create_gpu_lease"]["device"] == 0
    assert leased_region.region_type == "gpu"
    assert leased_region.gpu_device == 0
    assert payload["fast_phase_prover_lookup_round_trip_us"]["mean"] == 20.0
    assert payload["fast_phase_transport_overhead_us"]["mean"] == 180.0
