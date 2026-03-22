from __future__ import annotations

from pose.common.gpu_lease import get_cuda_runtime


def detect_gpu_memory_bytes(device: int) -> tuple[int, int]:
    return get_cuda_runtime().mem_get_info(device)
