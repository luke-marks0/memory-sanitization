from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HostRegionConfig:
    usable_bytes: int
    mlock: bool = False
    huge_pages: bool = False
    numa_node: int | None = None

