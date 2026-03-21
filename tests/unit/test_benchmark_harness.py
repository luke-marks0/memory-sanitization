from __future__ import annotations

from pathlib import Path

from pose.benchmarks.harness import run_benchmark
from pose.protocol.result_schema import bootstrap_result


def test_run_benchmark_archives_required_artifacts(
    monkeypatch,
    tmp_path: Path,
) -> None:
    call_count = 0

    def fake_run_session(self, profile, *, retain_session=False, session_plan=None):
        nonlocal call_count
        call_count += 1
        result = bootstrap_result(profile.name)
        result.verdict = "SUCCESS"
        result.success = True
        result.run_class = str(profile.benchmark_class)
        result.response_ms = 20 + call_count
        result.coverage_fraction = 0.92
        result.real_porep_ratio = 1.0
        result.gpu_devices = [0]
        result.gpu_usable_bytes_by_device = {"0": 2000}
        result.gpu_covered_bytes_by_device = {"0": 1800 + call_count}
        result.timings_ms["copy_to_hbm"] = 7 + call_count
        result.timings_ms["challenge_response"] = result.response_ms
        result.timings_ms["total"] = 50 + call_count
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
    assert Path(str(archive["upstream_path"])).exists()
    assert Path(str(archive["gpu_inventory_path"])).exists()
    assert (run_root / "manifest.json").exists()
    assert len(list(run_root.glob("run-*.result.json"))) == 3

    summary = payload["summary"]
    assert summary["result_count"] == 3
    assert summary["success_rate"] == 1.0
    assert summary["deadline_miss_rate"] == 0.0
    assert summary["timings_ms"]["copy_to_hbm"]["mean"] > 0.0
    assert summary["timings_ms"]["challenge_response"]["p95"] >= 21.0
    assert summary["per_device_hbm_coverage_bytes"]["0"]["mean"] > 1800.0
    assert summary["verifier_cpu_time_ms"]["mean"] >= 0.0
