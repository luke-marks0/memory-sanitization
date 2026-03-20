from __future__ import annotations


def estimate_payload_layout(
    unit_size: int,
    usable_bytes: int,
    leaf_size: int,
) -> dict[str, int]:
    if unit_size <= 0 or leaf_size <= 0:
        raise ValueError("unit_size and leaf_size must be positive")
    unit_count = usable_bytes // unit_size
    covered_bytes = unit_count * unit_size
    slack_bytes = usable_bytes - covered_bytes
    tail_filler_bytes = min(slack_bytes, leaf_size)
    return {
        "unit_count": unit_count,
        "covered_bytes": covered_bytes + tail_filler_bytes,
        "real_porep_bytes": covered_bytes,
        "tail_filler_bytes": tail_filler_bytes,
        "slack_bytes": slack_bytes,
    }

