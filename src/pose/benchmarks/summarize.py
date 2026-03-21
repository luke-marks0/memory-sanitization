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
    cpu_fallback_run_count = 0
    deadline_miss_count = 0
    coverage_fractions: list[float] = []
    real_porep_ratios: list[float] = []
    rechallenge_response_ms: list[int] = []
    cpu_fallback_events: list[str] = []
    seen_cpu_fallback_events: set[str] = set()

    for result in results:
        verdicts[result.verdict] = verdicts.get(result.verdict, 0) + 1
        success_count += int(result.success)
        cpu_fallback_run_count += int(result.cpu_fallback_detected)
        deadline_miss_count += int(result.verdict == "TIMEOUT")
        coverage_fractions.append(float(result.coverage_fraction))
        real_porep_ratios.append(float(result.real_porep_ratio))
        if result.run_class == "rechallenge":
            rechallenge_response_ms.append(int(result.response_ms))
        for event in result.cpu_fallback_events:
            if event in seen_cpu_fallback_events:
                continue
            seen_cpu_fallback_events.add(event)
            cpu_fallback_events.append(str(event))

        for key, value in result.timings_ms.items():
            timings.setdefault(key, []).append(int(value))
        for device, value in result.gpu_covered_bytes_by_device.items():
            per_device_hbm_coverage.setdefault(str(device), []).append(int(value))

    total = len(results)
    return {
        "result_count": total,
        "success_rate": (success_count / total) if total else 0.0,
        "cpu_fallback": {
            "detected_run_count": cpu_fallback_run_count,
            "detected_run_rate": (cpu_fallback_run_count / total) if total else 0.0,
            "unique_events": cpu_fallback_events,
        },
        "deadline_miss_rate": (deadline_miss_count / total) if total else 0.0,
        "verdict_counts": verdicts,
        "coverage_fraction": _series_summary(coverage_fractions),
        "real_porep_ratio": _series_summary(real_porep_ratios),
        "timings_ms": {
            key: _series_summary(values)
            for key, values in sorted(timings.items())
        },
        "per_device_hbm_coverage_bytes": {
            device: _series_summary(values)
            for device, values in sorted(per_device_hbm_coverage.items())
        },
        "verifier_cpu_time_ms": _series_summary(verifier_cpu_times_ms or ()),
        "rechallenge_performance": (
            _series_summary(rechallenge_response_ms) if rechallenge_response_ms else None
        ),
    }


def summarize_results(paths: list[str]) -> dict[str, object]:
    results = [
        SessionResult.from_dict(load_json_file(Path(path_str)))
        for path_str in paths
    ]
    return summarize_session_results(results)
