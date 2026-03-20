from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GpuRegionConfig:
    device: int
    usable_bytes: int
    cuda_ipc_required: bool = True

