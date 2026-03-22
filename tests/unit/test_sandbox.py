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
