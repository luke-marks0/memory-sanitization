from __future__ import annotations

from collections.abc import Sequence

from pose.common.errors import ProtocolError
from pose.hashing.blake3_backend import hash_xof as blake3_xof
from pose.hashing.encoding import (
    encode_graph_descriptor_input,
    encode_internal_label_input,
    encode_source_label_input,
)
from pose.hashing.shake256_backend import hash_xof as shake256_xof

DEFAULT_HASH_BACKEND = "blake3-xof"
SUPPORTED_HASH_BACKENDS = frozenset((DEFAULT_HASH_BACKEND, "shake256"))


def normalize_hash_backend(hash_backend: str | None) -> str:
    candidate = DEFAULT_HASH_BACKEND if hash_backend is None else str(hash_backend).strip().lower()
    if not candidate:
        return DEFAULT_HASH_BACKEND
    if candidate not in SUPPORTED_HASH_BACKENDS:
        supported = ", ".join(sorted(SUPPORTED_HASH_BACKENDS))
        raise ProtocolError(f"Unsupported hash backend: {hash_backend!r}. Expected one of: {supported}")
    return candidate


def validate_hash_backend(hash_backend: str | None) -> str:
    return normalize_hash_backend(hash_backend)


def hash_xof(data: bytes, *, hash_backend: str | None = None, length: int) -> bytes:
    backend = normalize_hash_backend(hash_backend)
    if length <= 0:
        raise ProtocolError(f"Hash output length must be positive, got {length}")
    if backend == DEFAULT_HASH_BACKEND:
        return blake3_xof(data, length=length)
    if backend == "shake256":
        return shake256_xof(data, length=length)
    raise ProtocolError(f"Unsupported hash backend dispatch: {backend}")


def hash_xof_hex(data: bytes, *, hash_backend: str | None = None, length: int) -> str:
    backend = normalize_hash_backend(hash_backend)
    return f"{backend}:{hash_xof(data, hash_backend=backend, length=length).hex()}"


def graph_descriptor_oracle_bytes(
    *,
    graph_family: str,
    label_count_m: int,
    graph_parameter_n: int,
    gamma: int,
    hash_backend: str | None = None,
    label_width_bits: int,
    output_bytes: int = 32,
    node_ordering_version: str = "pose-db-node-order/v1",
    challenge_set_ordering_version: str = "pose-db-challenge-order/v1",
) -> bytes:
    backend = normalize_hash_backend(hash_backend)
    return hash_xof(
        encode_graph_descriptor_input(
            graph_family=graph_family,
            label_count_m=label_count_m,
            graph_parameter_n=graph_parameter_n,
            gamma=gamma,
            node_ordering_version=node_ordering_version,
            challenge_set_ordering_version=challenge_set_ordering_version,
            hash_backend=backend,
            label_width_bits=label_width_bits,
        ),
        hash_backend=backend,
        length=output_bytes,
    )


def source_label_bytes(
    *,
    session_seed: bytes | str,
    graph_descriptor_digest: bytes | str,
    node_index: int,
    hash_backend: str | None = None,
    output_bytes: int,
) -> bytes:
    backend = normalize_hash_backend(hash_backend)
    return hash_xof(
        encode_source_label_input(
            session_seed=session_seed,
            graph_descriptor_digest=graph_descriptor_digest,
            node_index=node_index,
        ),
        hash_backend=backend,
        length=output_bytes,
    )


def internal_label_bytes(
    *,
    session_seed: bytes | str,
    graph_descriptor_digest: bytes | str,
    node_index: int,
    predecessor_labels: Sequence[bytes],
    hash_backend: str | None = None,
    output_bytes: int,
) -> bytes:
    backend = normalize_hash_backend(hash_backend)
    return hash_xof(
        encode_internal_label_input(
            session_seed=session_seed,
            graph_descriptor_digest=graph_descriptor_digest,
            node_index=node_index,
            predecessor_labels=predecessor_labels,
        ),
        hash_backend=backend,
        length=output_bytes,
    )
