from __future__ import annotations

import hashlib


def hash_xof(data: bytes, *, length: int) -> bytes:
    return hashlib.shake_256(data).digest(length)
