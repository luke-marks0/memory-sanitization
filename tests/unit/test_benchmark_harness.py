from __future__ import annotations

import json
from pathlib import Path

from pose.benchmarks.harness import run_benchmark
from pose.protocol.result_schema import bootstrap_result


def test_run_benchmark_archives_required_artifacts(
    monkeypatch,
    tmp_path: Path,
) -> None:
    call_count = 0
    calibration_source = tmp_path / "source-calibration.json"
    calibration_source.write_text('{"status":"calibrated"}\n', encoding="utf-8")

    def fake_run_session(self, profile, *, retain_session=False, session_plan=None):
        nonlocal call_count
        call_count += 1
        result = bootstrap_result(profile.name)
        result.verdict = "SUCCESS"
        result.success = True
        result.run_class = str(profile.benchmark_class)
        result.response_ms = 20 + call_count
        result.coverage_fraction = 0.92
        result.q_bound = 4096 + call_count
        result.gamma = 8192
        result.reported_success_bound = 1.0e-9
        result.max_round_trip_us = 150 + call_count
        result.gpu_devices = [0]
        result.gpu_usable_bytes_by_device = {"0": 2000}
        result.gpu_covered_bytes_by_device = {"0": 1800 + call_count}
        result.timings_ms["copy_to_hbm"] = 7 + call_count
        result.timings_ms["fast_phase_total"] = result.response_ms
        result.timings_ms["total"] = 50 + call_count
        result.notes.append(f"calibration_artifact={calibration_source}")
        return result

    monkeypatch.setattr(
        "pose.benchmarks.harness.VerifierService.run_session",
        fake_run_session,
    )

    payload = run_benchmark("single-h100-hbm-max", output_dir=tmp_path)

    assert payload["status"] == "benchmark-archived"
    archive = payload["archive"]
    run_root = Path(str(archive["run_directory"]))
    assert run_root.exists()
    assert Path(str(archive["summary_path"])).exists()
    assert Path(str(archive["log_path"])).exists()
    assert Path(str(archive["environment_path"])).exists()
    assert Path(str(archive["toolchains_path"])).exists()
    assert Path(str(archive["gpu_inventory_path"])).exists()
    assert len(archive["calibration_artifact_paths"]) == 3
    assert all(Path(path).exists() for path in archive["calibration_artifact_paths"])
    assert (run_root / "manifest.json").exists()
    assert len(list(run_root.glob("run-*.result.json"))) == 3

    summary = payload["summary"]
    assert summary["result_count"] == 3
    assert summary["success_rate"] == 1.0
    assert summary["deadline_miss_rate"] == 0.0
    assert summary["coverage_fraction"]["mean"] == 0.92
    assert summary["q_bound"]["mean"] > 4096.0
    assert summary["gamma"]["mean"] == 8192.0
    assert summary["reported_success_bound"]["mean"] == 1.0e-9
    assert summary["max_round_trip_us"]["p95"] >= 151.0
    assert summary["timings_ms"]["copy_to_hbm"]["mean"] > 0.0
    assert summary["timings_ms"]["fast_phase_total"]["p95"] >= 21.0
    assert summary["per_device_hbm_coverage_bytes"]["0"]["mean"] > 1800.0
    assert summary["verifier_cpu_time_ms"]["mean"] >= 0.0

    benchmark_log = (run_root / "benchmark.log").read_text(encoding="utf-8")
    assert "coverage_fraction=0.920000" in benchmark_log
    assert "q_bound=4097" in benchmark_log
    assert "gamma=8192" in benchmark_log

    second_run = json.loads((run_root / "run-002.result.json").read_text(encoding="utf-8"))
    assert second_run["q_bound"] == 4098
    assert second_run["gamma"] == 8192
    assert second_run["reported_success_bound"] == 1.0e-9


def test_run_benchmark_accepts_profile_paths(
    monkeypatch,
    tmp_path: Path,
) -> None:
    profile_path = tmp_path / "path-profile.yaml"
    profile_path.write_text(
        """
name: path-profile
benchmark_class: cold
target_devices:
  host: true
  gpus: []
reserve_policy:
  host_bytes: 131072
  per_gpu_bytes: 0
host_target_fraction: 1.0
per_gpu_target_fraction: 0.0
w_bits: 256
graph_family: pose-db-drg-v1
hash_backend: blake3-xof
adversary_model: general
attacker_budget_bytes_assumed: 16384
challenge_policy:
  rounds_r: 4
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

    monkeypatch.setattr(
        "pose.benchmarks.harness.VerifierService.run_session",
        lambda self, profile, *, retain_session=False, session_plan=None: bootstrap_result(profile.name),
    )

    payload = run_benchmark(str(profile_path), output_dir=tmp_path / "out")

    assert payload["status"] == "benchmark-archived"
    assert payload["plan"]["profile"]["name"] == "path-profile"
