from __future__ import annotations

from dataclasses import dataclass

from pose.common.errors import ProtocolError
from pose.common.merkle import MerkleOpening, verify_opening


@dataclass(frozen=True)
class OuterChallengeOpening:
    region_id: str
    session_manifest_root: str
    leaf_index: int
    leaf: bytes
    sibling_hashes: tuple[bytes, ...]

    def to_merkle_opening(self) -> MerkleOpening:
        return MerkleOpening(
            leaf_index=self.leaf_index,
            leaf=self.leaf,
            sibling_hashes=self.sibling_hashes,
        )


def verify_outer_challenge_response(
    *,
    expected_region_id: str,
    expected_session_manifest_root: str,
    expected_indices: list[int],
    root: bytes,
    leaf_size: int,
    openings: list[OuterChallengeOpening],
) -> bool:
    if len(openings) != len(expected_indices):
        return False
    if len(set(expected_indices)) != len(expected_indices):
        return False

    seen_indices: set[int] = set()
    for expected_index, opening in zip(expected_indices, openings, strict=True):
        if opening.region_id != expected_region_id:
            return False
        if opening.session_manifest_root != expected_session_manifest_root:
            return False
        if opening.leaf_index != expected_index:
            return False
        if opening.leaf_index in seen_indices:
            return False
        seen_indices.add(opening.leaf_index)
        if len(opening.leaf) != leaf_size:
            return False
        if not verify_opening(root, opening.to_merkle_opening()):
            return False
    return True


def decode_opening_payload(payload: dict[str, object]) -> OuterChallengeOpening:
    try:
        sibling_hashes = tuple(
            bytes.fromhex(str(item)) for item in payload["sibling_hashes_hex"]  # type: ignore[index]
        )
        return OuterChallengeOpening(
            region_id=str(payload["region_id"]),
            session_manifest_root=str(payload["session_manifest_root"]),
            leaf_index=int(payload["leaf_index"]),
            leaf=bytes.fromhex(str(payload["leaf_hex"])),
            sibling_hashes=sibling_hashes,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ProtocolError(f"Malformed outer opening payload: {error}") from error
