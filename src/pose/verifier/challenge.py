from __future__ import annotations

import math
import random


def challenge_count_for_policy(
    *,
    total_leaves: int,
    epsilon: float,
    lambda_bits: int,
    max_challenges: int,
) -> int:
    if total_leaves <= 0 or max_challenges <= 0:
        return 0
    if not 0 < epsilon < 1:
        raise ValueError(f"epsilon must be between 0 and 1, got {epsilon}")
    if lambda_bits <= 0:
        raise ValueError(f"lambda_bits must be positive, got {lambda_bits}")

    required = math.ceil(math.log(2 ** (-lambda_bits)) / math.log(1 - epsilon))
    return min(total_leaves, max_challenges, max(1, required))


def sample_leaf_indices(total_leaves: int, count: int, seed: str) -> list[int]:
    if total_leaves <= 0 or count <= 0:
        return []
    count = min(total_leaves, count)
    generator = random.Random(seed)
    return sorted(generator.sample(range(total_leaves), count))
