from __future__ import annotations

import secrets

from pose.common.errors import ProtocolError


def sample_challenge_indices(
    *,
    label_count_m: int,
    rounds_r: int,
    sample_with_replacement: bool,
) -> list[int]:
    if not sample_with_replacement:
        raise ProtocolError("PoSE-DB runtime requires uniform sampling with replacement.")
    if label_count_m <= 0:
        raise ProtocolError(f"Session plan label_count_m must be positive, got {label_count_m}")
    if rounds_r <= 0:
        raise ProtocolError(f"rounds_r must be positive, got {rounds_r}")
    return [secrets.randbelow(label_count_m) for _ in range(rounds_r)]
