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
    result.slack_bytes = 4096
    result.scratch_peak_bytes = 2048
    result.declared_stage_copy_bytes = 0
    result.q_bound = 4096
    result.gamma = 8192
    result.attacker_budget_bytes_assumed = 33554432
    result.soundness_model = "random-oracle + distant-attacker + calibrated q<gamma"
    result.reported_success_bound = 1e-9
    result.round_trip_p50_us = 120
    result.round_trip_p95_us = 150
    result.round_trip_p99_us = 165
    result.max_round_trip_us = 170
    result.timings_ms["copy_to_hbm"] = 12
    result.timings_ms["fast_phase_total"] = 17
    result.timings_ms["verifier_check_total"] = 9
    result.timings_ms["total"] = 40

    summary = summarize_session_results([result], verifier_cpu_times_ms=[11])

    assert summary["success_rate"] == 1.0
    assert summary["coverage_fraction"]["mean"] == 0.91
    assert summary["slack_bytes"]["mean"] == 4096.0
    assert summary["scratch_peak_bytes"]["mean"] == 2048.0
    assert summary["declared_stage_copy_bytes"]["mean"] == 0.0
    assert summary["q_bound"]["mean"] == 4096.0
    assert summary["q_over_gamma"]["mean"] == 0.5
    assert summary["gamma"]["mean"] == 8192.0
    assert summary["attacker_budget_bytes_assumed"]["mean"] == 33554432.0
    assert summary["timings_ms"]["copy_to_hbm"]["mean"] == 12.0
    assert summary["timings_ms"]["fast_phase_total"]["mean"] == 17.0
    assert summary["per_device_hbm_coverage_bytes"]["0"]["mean"] == 123456.0
    assert summary["verifier_cpu_time_ms"]["mean"] == 11.0
    assert summary["soundness_models"]["random-oracle + distant-attacker + calibrated q<gamma"] == 1
    assert summary["reported_success_bound"]["mean"] == 1e-9
    assert summary["round_trip_p50_us"]["mean"] == 120.0
    assert summary["round_trip_p95_us"]["mean"] == 150.0
    assert summary["round_trip_p99_us"]["mean"] == 165.0
    assert summary["max_round_trip_us"]["mean"] == 170.0


def test_benchmark_summary_reports_host_q_margin_and_fast_phase_latency() -> None:
    result = bootstrap_result("dev-small")
    result.verdict = "SUCCESS"
    result.success = True
    result.coverage_fraction = 1.0
    result.host_covered_bytes = 131072
    result.host_usable_bytes = 131072
    result.covered_bytes = 131072
    result.q_bound = 512
    result.gamma = 1024
    result.round_trip_p50_us = 200
    result.round_trip_p95_us = 260
    result.round_trip_p99_us = 400
    result.max_round_trip_us = 450
    result.timings_ms["label_generation"] = 17
    result.timings_ms["fast_phase_total"] = 9
    result.timings_ms["total"] = 44

    summary = summarize_session_results([result], verifier_cpu_times_ms=[6])

    assert summary["coverage_fraction"]["mean"] == 1.0
    assert summary["q_bound"]["mean"] == 512.0
    assert summary["q_over_gamma"]["mean"] == 0.5
    assert summary["timings_ms"]["label_generation"]["mean"] == 17.0
    assert summary["timings_ms"]["fast_phase_total"]["mean"] == 9.0
    assert summary["timings_ms"]["total"]["mean"] == 44.0
    assert summary["round_trip_p95_us"]["mean"] == 260.0
    assert summary["max_round_trip_us"]["mean"] == 450.0
