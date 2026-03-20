from __future__ import annotations

from pose.common.hashing import sha256_bytes


def deterministic_tail_filler(
    session_nonce: str,
    region_id: str,
    session_plan_root: str,
    length: int,
) -> bytes:
    if length <= 0:
        return b""
    seed = sha256_bytes(
        "|".join((session_nonce, region_id, session_plan_root)).encode("utf-8")
    )
    repeats = (length // len(seed)) + 1
    return (seed * repeats)[:length]


def build_region_payload(
    serialized_units: list[bytes],
    session_nonce: str,
    region_id: str,
    session_plan_root: str,
    tail_filler_bytes: int,
) -> bytes:
    payload = b"".join(serialized_units)
    return payload + deterministic_tail_filler(
        session_nonce=session_nonce,
        region_id=region_id,
        session_plan_root=session_plan_root,
        length=tail_filler_bytes,
    )
