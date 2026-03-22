from __future__ import annotations

import os
from pathlib import Path

from pose.common.errors import ResourceFailure


def _physical_memory_bytes() -> int:
    try:
        pages = int(os.sysconf("SC_PHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, ValueError):
        pages = 0
        page_size = 0
    if pages > 0 and page_size > 0:
        return pages * page_size

    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                parts = line.split()
                if len(parts) >= 2:
                    return int(parts[1]) * 1024
    raise ResourceFailure("Unable to detect total host memory bytes")


def _cgroup_memory_limit_bytes() -> int | None:
    candidates = (
        Path("/sys/fs/cgroup/memory.max"),
        Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
    )
    for path in candidates:
        if not path.exists():
            continue
        raw = path.read_text(encoding="utf-8").strip()
        if not raw or raw == "max":
            continue
        try:
            value = int(raw)
        except ValueError:
            continue
        if value <= 0 or value >= 2**60:
            continue
        return value
    return None


def detect_host_memory_bytes() -> int:
    detected = _physical_memory_bytes()
    cgroup_limit = _cgroup_memory_limit_bytes()
    if cgroup_limit is not None:
        detected = min(detected, cgroup_limit)
    return detected
