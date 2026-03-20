from __future__ import annotations


def zeroized(length: int) -> bytes:
    return bytes(length)


def cleanup_status(zeroize: bool) -> str:
    return "ZEROIZED_AND_RELEASED" if zeroize else "RELEASED_WITHOUT_ZEROIZE"

