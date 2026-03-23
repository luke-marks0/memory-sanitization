from __future__ import annotations

from pathlib import Path

from pose.benchmarks.calibration import prepare_calibration
from pose.common.sandbox import (
    ProverSandboxPolicy,
    sandbox_claim_notes,
    sandboxed_child_environment,
    sandboxed_command,
)
from pose.verifier.grpc_client import start_ephemeral_prover_server
from pose.verifier.slot_planning import plan_slot_layout


def _write_sandboxed_profile(path: Path) -> None:
    path.write_text(
        """
name: sandboxed-host
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
attacker_budget_bytes_assumed: 65536
challenge_policy:
  rounds_r: 16
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
prover_sandbox:
  mode: process_budget_dev
  process_memory_max_bytes: 1048576
  require_no_visible_gpus: true
  memlock_max_bytes: 0
  file_size_max_bytes: 0
""".strip(),
        encoding="utf-8",
    )


def _write_hbm_sandboxed_profile(path: Path, *, require_no_visible_gpus: bool = False) -> None:
    require_no_visible_gpus_literal = "true" if require_no_visible_gpus else "false"
    path.write_text(
        f"""
name: sandboxed-hbm
benchmark_class: cold
target_devices:
  host: false
  gpus: [0]
reserve_policy:
  host_bytes: 0
  per_gpu_bytes: 1048576
host_target_fraction: 0.0
per_gpu_target_fraction: 1.0
w_bits: 256
graph_family: pose-db-drg-v1
hash_backend: blake3-xof
adversary_model: general
attacker_budget_bytes_assumed: 65536
challenge_policy:
  rounds_r: 16
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
prover_sandbox:
  mode: process_budget_dev
  process_memory_max_bytes: 1073741824
  require_no_visible_gpus: {require_no_visible_gpus_literal}
  memlock_max_bytes: 0
  file_size_max_bytes: 0
""".strip(),
        encoding="utf-8",
    )


def _write_hybrid_sandboxed_profile(path: Path, *, require_no_visible_gpus: bool = False) -> None:
    require_no_visible_gpus_literal = "true" if require_no_visible_gpus else "false"
    path.write_text(
        f"""
name: sandboxed-hybrid
benchmark_class: cold
target_devices:
  host: true
  gpus: [0]
reserve_policy:
  host_bytes: 1048576
  per_gpu_bytes: 1048576
host_target_fraction: 1.0
per_gpu_target_fraction: 1.0
w_bits: 256
graph_family: pose-db-drg-v1
hash_backend: blake3-xof
adversary_model: general
attacker_budget_bytes_assumed: 65536
challenge_policy:
  rounds_r: 16
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
prover_sandbox:
  mode: process_budget_dev
  process_memory_max_bytes: 1073741824
  require_no_visible_gpus: {require_no_visible_gpus_literal}
  memlock_max_bytes: 0
  file_size_max_bytes: 0
""".strip(),
        encoding="utf-8",
    )


def test_sandbox_claim_notes_mark_development_only_mode() -> None:
    notes = sandbox_claim_notes(
        ProverSandboxPolicy(
            mode="process_budget_dev",
            process_memory_max_bytes=1_048_576,
            require_no_visible_gpus=True,
            memlock_max_bytes=0,
            file_size_max_bytes=0,
        )
    )

    assert "prover_sandbox_mode=process_budget_dev" in notes
    assert "development_only_attacker_budget_override=true" in notes
    assert "development_only_not_for_production=true" in notes
    assert "prover_sandbox_process_memory_max_bytes=1048576" in notes
    assert "prover_sandbox_hidden_gpu_tiers=all" in notes


def test_sandboxed_child_environment_hides_gpus() -> None:
    env = sandboxed_child_environment({"PYTHONPATH": "src"}, require_no_visible_gpus=True)

    assert env["PYTHONDONTWRITEBYTECODE"] == "1"
    assert env["CUDA_VISIBLE_DEVICES"] == ""
    assert env["NVIDIA_VISIBLE_DEVICES"] == "void"


def test_sandboxed_command_wraps_with_prlimit_and_as_limit() -> None:
    command = sandboxed_command(
        ["python", "-m", "pose.cli.main"],
        process_memory_max_bytes=1_048_576,
        memlock_max_bytes=0,
        file_size_max_bytes=0,
    )

    assert command[0].endswith("prlimit")
    assert "--as=1048576" in command
    assert "--memlock=0" in command
    assert "--fsize=0" in command
    assert command[-3:] == ["python", "-m", "pose.cli.main"]


def test_prepare_calibration_uses_process_budget_dev_override(monkeypatch, tmp_path: Path) -> None:
    profile_path = tmp_path / "profile.yaml"
    _write_sandboxed_profile(profile_path)
    artifact_dir = tmp_path / "artifacts"

    monkeypatch.setattr("pose.benchmarks.calibration.calibration_root", lambda: artifact_dir)
    monkeypatch.setattr(
        "pose.benchmarks.calibration.plan_slot_layout",
        lambda profile: plan_slot_layout(profile, detected_host_bytes=1_048_576),
    )
    monkeypatch.setattr(
        "pose.benchmarks.calibration.get_cuda_runtime",
        lambda: (_ for _ in ()).throw(AssertionError("GPU runtime should not be queried in process_budget_dev mode")),
    )
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
        lambda **kwargs: 100_000.0,
    )

    payload = prepare_calibration(str(profile_path))

    assert payload["status"] == "calibrated"
    assert payload["planning"]["base_attacker_budget_bytes_assumed"] == 65_536
    assert payload["planning"]["effective_attacker_budget_bytes_assumed"] == 65_536
    assert payload["planning"]["untargeted_local_tier_bytes_auto_included"] == 0
    assert "development_only_not_for_production=true" in payload["planning"]["claim_notes"]


def test_prepare_calibration_uses_process_budget_dev_override_for_hbm_only_profile(
    monkeypatch,
    tmp_path: Path,
) -> None:
    profile_path = tmp_path / "profile.yaml"
    _write_hbm_sandboxed_profile(profile_path)
    artifact_dir = tmp_path / "artifacts"

    monkeypatch.setattr("pose.benchmarks.calibration.calibration_root", lambda: artifact_dir)
    monkeypatch.setattr(
        "pose.benchmarks.calibration.plan_slot_layout",
        lambda profile: plan_slot_layout(
            profile,
            detected_gpu_bytes_by_device={0: (1_048_576, 2_097_152)},
        ),
    )
    monkeypatch.setattr(
        "pose.benchmarks.calibration.detect_host_memory_bytes",
        lambda: (_ for _ in ()).throw(AssertionError("host memory should not be auto-included in process_budget_dev mode")),
    )
    monkeypatch.setattr(
        "pose.benchmarks.calibration.get_cuda_runtime",
        lambda: (_ for _ in ()).throw(AssertionError("GPU runtime should not be queried for untargeted tiers in process_budget_dev mode")),
    )
    monkeypatch.setattr(
        "pose.benchmarks.calibration._measure_gpu_lookup_latency_us",
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
        lambda **kwargs: 100_000.0,
    )

    payload = prepare_calibration(str(profile_path))

    assert payload["status"] == "calibrated"
    assert payload["planning"]["base_attacker_budget_bytes_assumed"] == 65_536
    assert payload["planning"]["effective_attacker_budget_bytes_assumed"] == 65_536
    assert payload["planning"]["untargeted_local_tier_bytes_auto_included"] == 0
    assert payload["planning"]["gpu_covered_bytes_by_device"] == {"0": 1_048_576}
    assert "development_only_not_for_production=true" in payload["planning"]["claim_notes"]


def test_prepare_calibration_rejects_hbm_process_budget_dev_when_target_gpu_hidden(
    monkeypatch,
    tmp_path: Path,
) -> None:
    profile_path = tmp_path / "profile.yaml"
    _write_hbm_sandboxed_profile(profile_path, require_no_visible_gpus=True)
    artifact_dir = tmp_path / "artifacts"

    monkeypatch.setattr("pose.benchmarks.calibration.calibration_root", lambda: artifact_dir)
    monkeypatch.setattr(
        "pose.benchmarks.calibration.plan_slot_layout",
        lambda profile: plan_slot_layout(
            profile,
            detected_gpu_bytes_by_device={0: (1_048_576, 2_097_152)},
        ),
    )
    monkeypatch.setattr(
        "pose.benchmarks.calibration._measure_gpu_lookup_latency_us",
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
        lambda **kwargs: 100_000.0,
    )

    payload = prepare_calibration(str(profile_path))

    assert payload["status"] == "calibration-invalid"
    assert any(
        "GPU-targeted process_budget_dev prover sandbox mode must keep targeted GPUs visible" in note
        for note in payload["notes"]
    )


def test_prepare_calibration_uses_process_budget_dev_override_for_hybrid_profile(
    monkeypatch,
    tmp_path: Path,
) -> None:
    profile_path = tmp_path / "profile.yaml"
    _write_hybrid_sandboxed_profile(profile_path)
    artifact_dir = tmp_path / "artifacts"

    monkeypatch.setattr("pose.benchmarks.calibration.calibration_root", lambda: artifact_dir)
    monkeypatch.setattr(
        "pose.benchmarks.calibration.plan_slot_layout",
        lambda profile: plan_slot_layout(
            profile,
            detected_host_bytes=1_048_576,
            detected_gpu_bytes_by_device={0: (1_048_576, 2_097_152)},
        ),
    )
    monkeypatch.setattr(
        "pose.benchmarks.calibration.detect_host_memory_bytes",
        lambda: (_ for _ in ()).throw(AssertionError("host memory should not be auto-included in process_budget_dev mode")),
    )
    monkeypatch.setattr(
        "pose.benchmarks.calibration.get_cuda_runtime",
        lambda: (_ for _ in ()).throw(AssertionError("GPU runtime should not be queried for untargeted tiers in process_budget_dev mode")),
    )
    monkeypatch.setattr(
        "pose.benchmarks.calibration._measure_resident_lookup_latency_us_by_region",
        lambda **kwargs: {
            "host-0": {"mean": 9.0, "p50": 9.0, "p95": 10.0, "p99": 11.0, "max": 11.0},
            "gpu-0": {"mean": 10.0, "p50": 10.0, "p95": 12.0, "p99": 14.0, "max": 14.0},
        },
    )
    monkeypatch.setattr(
        "pose.benchmarks.calibration._measure_grpc_fast_phase_transport_us_by_region",
        lambda *_args, **_kwargs: {
            "host-0": {
                "fast_phase_round_trip_us": {"mean": 85.0, "p50": 84.0, "p95": 90.0, "p99": 95.0, "max": 95.0},
                "fast_phase_prover_lookup_round_trip_us": {"mean": 7.0, "p50": 7.0, "p95": 8.0, "p99": 9.0, "max": 9.0},
                "fast_phase_transport_overhead_us": {"mean": 78.0, "p50": 77.0, "p95": 82.0, "p99": 86.0, "max": 86.0},
            },
            "gpu-0": {
                "fast_phase_round_trip_us": {"mean": 95.0, "p50": 94.0, "p95": 101.0, "p99": 111.0, "max": 111.0},
                "fast_phase_prover_lookup_round_trip_us": {"mean": 8.0, "p50": 8.0, "p95": 10.0, "p99": 12.0, "max": 12.0},
                "fast_phase_transport_overhead_us": {"mean": 87.0, "p50": 86.0, "p95": 91.0, "p99": 99.0, "max": 99.0},
            },
        },
    )
    monkeypatch.setattr(
        "pose.benchmarks.calibration._measure_hash_evaluations_per_second",
        lambda **kwargs: 100_000.0,
    )

    payload = prepare_calibration(str(profile_path))

    assert payload["status"] == "calibrated"
    assert payload["planning"]["effective_attacker_budget_bytes_assumed"] == 65_536
    assert payload["planning"]["untargeted_local_tier_bytes_auto_included"] == 0
    assert payload["measurements"]["resident_lookup_latency_us"]["p95"] == 12.0
    assert payload["measurements"]["resident_lookup_latency_us_by_region"]["host-0"]["p95"] == 10.0
    assert payload["measurements"]["fast_phase_transport_overhead_us"]["p95"] == 91.0
    assert "development_only_not_for_production=true" in payload["planning"]["claim_notes"]


def test_start_ephemeral_prover_server_applies_process_budget_environment(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    class _FakeProcess:
        returncode = 0

        def poll(self):
            return None

        def terminate(self):
            return None

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured["env"] = dict(kwargs["env"])
        return _FakeProcess()

    monkeypatch.setattr("pose.verifier.grpc_client.subprocess.Popen", fake_popen)
    monkeypatch.setattr(
        "pose.verifier.grpc_client.discover",
        lambda _socket_path: {"protocol_version": "v1", "capabilities": []},
    )

    process = start_ephemeral_prover_server(
        socket_path=str(tmp_path / "prover.sock"),
        prover_sandbox=ProverSandboxPolicy(
            mode="process_budget_dev",
            process_memory_max_bytes=1_048_576,
            require_no_visible_gpus=True,
            memlock_max_bytes=0,
            file_size_max_bytes=0,
        ),
    )

    assert process is not None
    assert str(captured["command"][0]).endswith("prlimit")
    assert "--as=1048576" in captured["command"]
    env = captured["env"]
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"
    assert env["CUDA_VISIBLE_DEVICES"] == ""
    assert env["NVIDIA_VISIBLE_DEVICES"] == "void"


def test_start_ephemeral_prover_server_keeps_target_gpus_visible_for_hbm_dev_profile(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("NVIDIA_VISIBLE_DEVICES", "all")

    class _FakeProcess:
        returncode = 0

        def poll(self):
            return None

        def terminate(self):
            return None

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured["env"] = dict(kwargs["env"])
        return _FakeProcess()

    monkeypatch.setattr("pose.verifier.grpc_client.subprocess.Popen", fake_popen)
    monkeypatch.setattr(
        "pose.verifier.grpc_client.discover",
        lambda _socket_path: {"protocol_version": "v1", "capabilities": []},
    )

    process = start_ephemeral_prover_server(
        socket_path=str(tmp_path / "prover.sock"),
        prover_sandbox=ProverSandboxPolicy(
            mode="process_budget_dev",
            process_memory_max_bytes=1_048_576,
            require_no_visible_gpus=False,
            memlock_max_bytes=0,
            file_size_max_bytes=0,
        ),
    )

    assert process is not None
    assert str(captured["command"][0]).endswith("prlimit")
    assert "--as=1048576" in captured["command"]
    env = captured["env"]
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"
    assert env["CUDA_VISIBLE_DEVICES"] == "0"
    assert env["NVIDIA_VISIBLE_DEVICES"] == "all"
