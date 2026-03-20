from __future__ import annotations

from pose.common.hashing import sha256_hex


def derive_replica_id(
    prover_id: bytes,
    sector_id: int,
    ticket: bytes,
    porep_id: bytes,
) -> str:
    material = b"|".join(
        (
            prover_id,
            str(sector_id).encode("ascii"),
            ticket,
            porep_id,
        )
    )
    return sha256_hex(material)

