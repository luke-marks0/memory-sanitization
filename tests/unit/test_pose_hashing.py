from __future__ import annotations

import pytest

from pose.common.errors import ProtocolError
from pose.hashing import (
    DEFAULT_HASH_BACKEND,
    SUPPORTED_HASH_BACKENDS,
    encode_graph_descriptor_input,
    encode_internal_label_input,
    encode_source_label_input,
    graph_descriptor_oracle_bytes,
    hash_xof,
    internal_label_bytes,
    normalize_hash_backend,
    source_label_bytes,
)


def test_hash_backend_normalization_supports_default_and_shake256() -> None:
    assert DEFAULT_HASH_BACKEND in SUPPORTED_HASH_BACKENDS
    assert normalize_hash_backend(None) == "blake3-xof"
    assert normalize_hash_backend("") == "blake3-xof"
    assert normalize_hash_backend("SHAKE256") == "shake256"


def test_hash_backend_normalization_rejects_unknown_backend() -> None:
    with pytest.raises(ProtocolError, match="Unsupported hash backend"):
        normalize_hash_backend("sha256")


def test_hash_backends_emit_requested_xof_length() -> None:
    payload = b"pose-db-random-oracle"

    blake3_output = hash_xof(payload, hash_backend="blake3-xof", length=48)
    shake_output = hash_xof(payload, hash_backend="shake256", length=48)

    assert len(blake3_output) == 48
    assert len(shake_output) == 48
    assert blake3_output != shake_output


def test_domain_separated_encodings_are_deterministic_and_distinct() -> None:
    descriptor_input = encode_graph_descriptor_input(
        graph_family="pose-db-drg-v1",
        label_count_m=16,
        graph_parameter_n=4,
        gamma=8,
        hash_backend="blake3-xof",
        label_width_bits=256,
    )
    source_input = encode_source_label_input(
        session_seed=b"\x11" * 32,
        graph_descriptor_digest="sha256:descriptor",
        node_index=3,
    )
    internal_input = encode_internal_label_input(
        session_seed=b"\x11" * 32,
        graph_descriptor_digest="sha256:descriptor",
        node_index=3,
        predecessor_labels=[b"\x22" * 32, b"\x33" * 32],
    )

    assert descriptor_input == encode_graph_descriptor_input(
        graph_family="pose-db-drg-v1",
        label_count_m=16,
        graph_parameter_n=4,
        gamma=8,
        hash_backend="blake3-xof",
        label_width_bits=256,
    )
    assert len({descriptor_input, source_input, internal_input}) == 3


def test_label_hash_helpers_bind_backend_and_inputs() -> None:
    source = source_label_bytes(
        session_seed=b"\xaa" * 32,
        graph_descriptor_digest="sha256:test-descriptor",
        node_index=7,
        hash_backend="blake3-xof",
        output_bytes=32,
    )
    source_wrong_index = source_label_bytes(
        session_seed=b"\xaa" * 32,
        graph_descriptor_digest="sha256:test-descriptor",
        node_index=8,
        hash_backend="blake3-xof",
        output_bytes=32,
    )
    internal = internal_label_bytes(
        session_seed=b"\xaa" * 32,
        graph_descriptor_digest="sha256:test-descriptor",
        node_index=7,
        predecessor_labels=[b"\xbb" * 32],
        hash_backend="blake3-xof",
        output_bytes=32,
    )

    assert source != source_wrong_index
    assert source != internal


def test_graph_descriptor_oracle_binds_hash_backend() -> None:
    blake3_descriptor = graph_descriptor_oracle_bytes(
        graph_family="pose-db-drg-v1",
        label_count_m=16,
        graph_parameter_n=4,
        gamma=8,
        hash_backend="blake3-xof",
        label_width_bits=256,
    )
    shake_descriptor = graph_descriptor_oracle_bytes(
        graph_family="pose-db-drg-v1",
        label_count_m=16,
        graph_parameter_n=4,
        gamma=8,
        hash_backend="shake256",
        label_width_bits=256,
    )

    assert len(blake3_descriptor) == 32
    assert len(shake_descriptor) == 32
    assert blake3_descriptor != shake_descriptor
