from __future__ import annotations


def open_leaf(payload: bytes, leaf_index: int, leaf_size: int) -> bytes:
    start = leaf_index * leaf_size
    end = start + leaf_size
    return payload[start:end]

