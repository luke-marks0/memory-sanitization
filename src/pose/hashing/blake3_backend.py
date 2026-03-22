from __future__ import annotations

from blake3 import blake3


def hash_xof(data: bytes, *, length: int) -> bytes:
    return blake3(data).digest(length=length)
