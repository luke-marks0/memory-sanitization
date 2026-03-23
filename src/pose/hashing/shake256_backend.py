from __future__ import annotations

from collections.abc import Iterable
import hashlib


def hash_xof(data: bytes, *, length: int) -> bytes:
    return hashlib.shake_256(data).digest(length)


def hash_xof_parts(
    parts: Iterable[bytes | bytearray | memoryview],
    *,
    length: int,
) -> bytes:
    hasher = hashlib.shake_256()
    for part in parts:
        hasher.update(part)
    return hasher.digest(length)
