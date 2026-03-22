from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

from pose.protocol.codec import load_json_file
from pose.protocol.result_schema import SessionResult


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] + ((ordered[upper] - ordered[lower]) * weight)


def _series_summary(values: Sequence[int | float]) -> dict[str, float]:
    if not values:
        return {
            "mean": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "p99": 0.0,
        }
    numeric = [float(value) for value in values]
    return {
        "mean": sum(numeric) / len(numeric),
        "p50": _percentile(numeric, 0.50),
        "p95": _percentile(numeric, 0.95),
        "p99": _percentile(numeric, 0.99),
    }


def summarize_session_results(
    results: Sequence[SessionResult],
    *,
    verifier_cpu_times_ms: Sequence[int] | None = None,
) -> dict[str, object]:
    verdicts: dict[str, int] = {}
    timings: dict[str, list[int]] = {}
    per_device_hbm_coverage: dict[str, list[int]] = {}

    success_count = 0
    deadline_miss_count = 0
    coverage_fractions: list[float] = []
    slack_bytes: list[int] = []
    scratch_peak_bytes: list[int] = []
    declared_stage_copy_bytes: list[int] = []
    q_bounds: list[int] = []
    q_over_gamma: list[float] = []
    gammas: list[int] = []
    attacker_budgets: list[int] = []
    success_bounds: list[float] = []
    round_trip_p50_us: list[int] = []
    round_trip_p95_us: list[int] = []
    round_trip_p99_us: list[int] = []
    max_round_trip_us: list[int] = []
    soundness_models: dict[str, int] = {}

    for result in results:
        verdicts[result.verdict] = verdicts.get(result.verdict, 0) + 1
        success_count += int(result.success)
        deadline_miss_count += int(result.verdict == "DEADLINE_MISS")
        coverage_fractions.append(float(result.coverage_fraction))
        slack_bytes.append(int(result.slack_bytes))
        scratch_peak_bytes.append(int(result.scratch_peak_bytes))
        declared_stage_copy_bytes.append(int(result.declared_stage_copy_bytes))
        q_bounds.append(int(result.q_bound))
        gammas.append(int(result.gamma))
        q_over_gamma.append((int(result.q_bound) / float(result.gamma)) if int(result.gamma) else 0.0)
        attacker_budgets.append(int(result.attacker_budget_bytes_assumed))
        success_bounds.append(float(result.reported_success_bound))
        round_trip_p50_us.append(int(result.round_trip_p50_us))
        round_trip_p95_us.append(int(result.round_trip_p95_us))
        round_trip_p99_us.append(int(result.round_trip_p99_us))
        max_round_trip_us.append(int(result.max_round_trip_us))
        soundness_models[result.soundness_model] = soundness_models.get(result.soundness_model, 0) + 1

        for key, value in result.timings_ms.items():
            timings.setdefault(key, []).append(int(value))
        for device, value in result.gpu_covered_bytes_by_device.items():
            per_device_hbm_coverage.setdefault(str(device), []).append(int(value))

    total = len(results)
    return {
        "result_count": total,
        "success_rate": (success_count / total) if total else 0.0,
        "deadline_miss_rate": (deadline_miss_count / total) if total else 0.0,
        "verdict_counts": verdicts,
        "coverage_fraction": _series_summary(coverage_fractions),
        "slack_bytes": _series_summary(slack_bytes),
        "scratch_peak_bytes": _series_summary(scratch_peak_bytes),
        "declared_stage_copy_bytes": _series_summary(declared_stage_copy_bytes),
        "q_bound": _series_summary(q_bounds),
        "q_over_gamma": _series_summary(q_over_gamma),
        "gamma": _series_summary(gammas),
        "attacker_budget_bytes_assumed": _series_summary(attacker_budgets),
        "soundness_models": soundness_models,
        "reported_success_bound": _series_summary(success_bounds),
        "round_trip_p50_us": _series_summary(round_trip_p50_us),
        "round_trip_p95_us": _series_summary(round_trip_p95_us),
        "round_trip_p99_us": _series_summary(round_trip_p99_us),
        "max_round_trip_us": _series_summary(max_round_trip_us),
        "timings_ms": {
            key: _series_summary(values)
            for key, values in sorted(timings.items())
        },
        "per_device_hbm_coverage_bytes": {
            device: _series_summary(values)
            for device, values in sorted(per_device_hbm_coverage.items())
        },
        "verifier_cpu_time_ms": _series_summary(verifier_cpu_times_ms or ()),
    }


def summarize_results(paths: list[str]) -> dict[str, object]:
    results = [
        SessionResult.from_dict(load_json_file(Path(path_str)))
        for path_str in paths
    ]
    return summarize_session_results(results)
