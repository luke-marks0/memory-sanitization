from __future__ import annotations

from pose.benchmarks.summarize import summarize_session_results
from pose.protocol.result_schema import bootstrap_result


def test_benchmark_summary_reports_hbm_timing_and_coverage_metrics() -> None:
    result = bootstrap_result("single-h100-hbm-max")
    result.verdict = "SUCCESS"
    result.success = True
    result.gpu_devices = [0]
    result.gpu_covered_bytes_by_device = {"0": 123456}
    result.gpu_usable_bytes_by_device = {"0": 234567}
    result.coverage_fraction = 0.91
    result.real_porep_ratio = 1.0
    result.response_ms = 17
    result.timings_ms["copy_to_hbm"] = 12
    result.timings_ms["challenge_response"] = 17
    result.timings_ms["outer_tree_build"] = 9
    result.timings_ms["total"] = 40

    summary = summarize_session_results([result], verifier_cpu_times_ms=[11])

    assert summary["success_rate"] == 1.0
    assert summary["coverage_fraction"]["mean"] == 0.91
    assert summary["real_porep_ratio"]["mean"] == 1.0
    assert summary["timings_ms"]["copy_to_hbm"]["mean"] == 12.0
    assert summary["timings_ms"]["challenge_response"]["mean"] == 17.0
    assert summary["per_device_hbm_coverage_bytes"]["0"]["mean"] == 123456.0
    assert summary["verifier_cpu_time_ms"]["mean"] == 11.0
