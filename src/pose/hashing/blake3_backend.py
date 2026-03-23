from __future__ import annotations

from collections.abc import Iterable

from blake3 import blake3


def hash_xof(data: bytes, *, length: int) -> bytes:
    return blake3(data).digest(length=length)


def hash_xof_parts(
    parts: Iterable[bytes | bytearray | memoryview],
    *,
    length: int,
) -> bytes:
    hasher = blake3()
    for part in parts:
        hasher.update(part)
    return hasher.digest(length=length)
