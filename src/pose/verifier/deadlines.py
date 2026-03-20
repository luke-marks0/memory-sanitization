from __future__ import annotations


def default_response_deadline_ms(leaf_size: int, challenge_count: int) -> int:
    baseline = 500
    leaf_component = max(1, leaf_size // 4096)
    return baseline + (leaf_component * max(1, challenge_count))

