from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

REQUIRED_TIMING_KEYS = (
    "discover",
    "region_leasing",
    "allocation",
    "graph_construction",
    "challenge_schedule_prep",
    "expected_response_prep",
    "label_generation",
    "copy_to_host",
    "copy_to_hbm",
    "stage_buffer_cleanup",
    "fast_phase_total",
    "verifier_check_total",
    "cleanup",
    "total",
)


def empty_timings() -> dict[str, int]:
    return {key: 0 for key in REQUIRED_TIMING_KEYS}


@dataclass
class TimingTracker:
    values: dict[str, int] = field(default_factory=empty_timings)
    _starts: dict[str, float] = field(default_factory=dict)

    def start(self, key: str) -> None:
        self._starts[key] = perf_counter()

    def stop(self, key: str) -> int:
        started = self._starts.pop(key)
        duration_ms = int((perf_counter() - started) * 1000)
        self.values[key] = duration_ms
        return duration_ms
