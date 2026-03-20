from __future__ import annotations

from pose.common.hashing import sha256_hex


def derive_label(replica_id_hex: str, layer: int, node: int, parents: list[int]) -> str:
    payload = "|".join(
        (
            replica_id_hex,
            str(layer),
            str(node),
            ",".join(str(parent) for parent in parents),
        )
    )
    return sha256_hex(payload.encode("ascii"))

