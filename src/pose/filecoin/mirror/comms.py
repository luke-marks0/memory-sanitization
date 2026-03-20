from __future__ import annotations

from pose.common.hashing import sha256_hex


def assemble_commitment(parts: list[bytes]) -> str:
    return sha256_hex(b"".join(parts))

