from __future__ import annotations

import random


def sample_leaf_indices(total_leaves: int, count: int, seed: str) -> list[int]:
    if total_leaves <= 0 or count <= 0:
        return []
    count = min(total_leaves, count)
    generator = random.Random(seed)
    return sorted(generator.sample(range(total_leaves), count))

