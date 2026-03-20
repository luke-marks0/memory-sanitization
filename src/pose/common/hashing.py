from __future__ import annotations

import hashlib


def sha256_bytes(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def merkle_leaf_hash(index: int, leaf: bytes) -> bytes:
    return sha256_bytes(index.to_bytes(8, "big") + leaf)


def merkle_parent_hash(left: bytes, right: bytes) -> bytes:
    return sha256_bytes(left + right)

