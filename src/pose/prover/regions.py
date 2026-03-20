from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RegionDescriptor:
    region_id: str
    region_type: str
    usable_bytes: int
    gpu_device: int | None = None

