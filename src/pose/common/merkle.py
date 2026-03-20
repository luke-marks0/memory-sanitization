from __future__ import annotations

from dataclasses import dataclass

from pose.common.errors import ProtocolError
from pose.common.hashing import merkle_leaf_hash, merkle_parent_hash


@dataclass(frozen=True)
class MerkleOpening:
    leaf_index: int
    leaf: bytes
    sibling_hashes: tuple[bytes, ...]


@dataclass(frozen=True)
class MerkleCommitment:
    leaf_size: int
    leaf_count: int
    root: bytes
    levels: tuple[tuple[bytes, ...], ...]

    @property
    def root_hex(self) -> str:
        return self.root.hex()

    def sibling_hashes(self, leaf_index: int) -> tuple[bytes, ...]:
        if not 0 <= leaf_index < self.leaf_count:
            raise ProtocolError(f"Merkle leaf index out of range: {leaf_index}")

        siblings: list[bytes] = []
        index = leaf_index
        for level in self.levels[:-1]:
            sibling_index = index ^ 1
            sibling_hash = level[sibling_index] if sibling_index < len(level) else level[index]
            siblings.append(sibling_hash)
            index //= 2
        return tuple(siblings)

    def opening(self, leaf_index: int, leaf: bytes) -> MerkleOpening:
        if len(leaf) != self.leaf_size:
            raise ProtocolError(
                f"Merkle opening leaf length mismatch: expected {self.leaf_size}, got {len(leaf)}"
            )
        return MerkleOpening(
            leaf_index=leaf_index,
            leaf=leaf,
            sibling_hashes=self.sibling_hashes(leaf_index),
        )


def _split_leaves(payload: bytes, leaf_size: int) -> tuple[bytes, ...]:
    if leaf_size <= 0:
        raise ProtocolError(f"Merkle leaf size must be positive: {leaf_size}")
    if not payload:
        raise ProtocolError("Merkle payload must not be empty")
    if len(payload) % leaf_size != 0:
        raise ProtocolError(
            "Merkle payload length must be a multiple of the leaf size: "
            f"{len(payload)} % {leaf_size} != 0"
        )
    return tuple(
        payload[offset : offset + leaf_size]
        for offset in range(0, len(payload), leaf_size)
    )


def commit_payload(payload: bytes, leaf_size: int) -> MerkleCommitment:
    leaves = _split_leaves(payload, leaf_size)
    current_level = [merkle_leaf_hash(index, leaf) for index, leaf in enumerate(leaves)]
    levels: list[tuple[bytes, ...]] = [tuple(current_level)]

    while len(current_level) > 1:
        next_level: list[bytes] = []
        for index in range(0, len(current_level), 2):
            left = current_level[index]
            right = current_level[index + 1] if index + 1 < len(current_level) else left
            next_level.append(merkle_parent_hash(left, right))
        current_level = next_level
        levels.append(tuple(current_level))

    return MerkleCommitment(
        leaf_size=leaf_size,
        leaf_count=len(leaves),
        root=current_level[0],
        levels=tuple(levels),
    )


def verify_opening(root: bytes, opening: MerkleOpening) -> bool:
    current = merkle_leaf_hash(opening.leaf_index, opening.leaf)
    index = opening.leaf_index
    for sibling_hash in opening.sibling_hashes:
        if index % 2 == 0:
            current = merkle_parent_hash(current, sibling_hash)
        else:
            current = merkle_parent_hash(sibling_hash, current)
        index //= 2
    return current == root
