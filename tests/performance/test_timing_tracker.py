from __future__ import annotations

from time import sleep

from pose.common.timing import TimingTracker, empty_timings


def test_empty_timing_map_contains_total() -> None:
    assert "total" in empty_timings()


def test_timing_tracker_records_elapsed_time() -> None:
    tracker = TimingTracker()
    tracker.start("discover")
    sleep(0.002)
    assert tracker.stop("discover") >= 1

