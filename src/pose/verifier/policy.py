from __future__ import annotations

import math


def compute_challenge_count(epsilon: float, lambda_bits: int, cap: int | None = None) -> int:
    if not 0 < epsilon < 1:
        raise ValueError("epsilon must be between 0 and 1")
    raw = math.ceil(math.log(2 ** (-lambda_bits)) / math.log(1 - epsilon))
    if cap is None:
        return raw
    return min(raw, cap)


def default_real_porep_threshold() -> float:
    return 0.99

