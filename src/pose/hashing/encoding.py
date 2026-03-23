from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence

from pose.common.errors import ProtocolError

GRAPH_DESCRIPTOR_DOMAIN = "pose-db/graph-descriptor/v1"
SOURCE_LABEL_DOMAIN = "pose-db/label/source/v1"
INTERNAL_LABEL_DOMAIN = "pose-db/label/internal/v1"
_DOMAIN_NAMESPACE = b"pose-db"
_ENCODING_VERSION = 1
_LENGTH_BYTES = 4
_ENCODING_VERSION_BYTES = _ENCODING_VERSION.to_bytes(4, "big", signed=False)


def _encode_u32(value: int) -> bytes:
    if value < 0 or value >= 2**32:
        raise ProtocolError(f"Expected uint32-compatible value, got {value}")
    return int(value).to_bytes(4, "big", signed=False)


def _encode_u64(value: int) -> bytes:
    if value < 0 or value >= 2**64:
        raise ProtocolError(f"Expected uint64-compatible value, got {value}")
    return int(value).to_bytes(8, "big", signed=False)


def _field_bytes(value: bytes | str, *, field_name: str) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return value.encode("utf-8")
    raise ProtocolError(f"{field_name} must be bytes or str, got {type(value).__name__}")


def _length_prefixed(value: bytes) -> bytes:
    if len(value) >= 2 ** (_LENGTH_BYTES * 8):
        raise ProtocolError(f"Field too large for {_LENGTH_BYTES}-byte length prefix: {len(value)}")
    return len(value).to_bytes(_LENGTH_BYTES, "big", signed=False) + value


def _field_bytes_like(
    value: bytes | bytearray | memoryview | str,
    *,
    field_name: str,
) -> bytes | bytearray | memoryview:
    if isinstance(value, (bytes, bytearray)):
        return value
    if isinstance(value, memoryview):
        return value if value.ndim == 1 and value.format == "B" else value.cast("B")
    if isinstance(value, str):
        return value.encode("utf-8")
    raise ProtocolError(
        f"{field_name} must be bytes-like or str, got {type(value).__name__}"
    )


def _length_prefix(length: int) -> bytes:
    if length < 0 or length >= 2 ** (_LENGTH_BYTES * 8):
        raise ProtocolError(f"Field too large for {_LENGTH_BYTES}-byte length prefix: {length}")
    return int(length).to_bytes(_LENGTH_BYTES, "big", signed=False)


def _iter_domain_separated_message_parts(
    domain: str,
    *,
    field_count: int,
) -> Iterator[bytes]:
    if not domain:
        raise ProtocolError("Domain tag must be a non-empty string.")
    domain_bytes = domain.encode("ascii")
    yield _DOMAIN_NAMESPACE
    yield _ENCODING_VERSION_BYTES
    yield _length_prefix(len(domain_bytes))
    yield domain_bytes
    yield _encode_u32(field_count)


def _iter_length_prefixed_field_parts(
    value: bytes | bytearray | memoryview,
) -> Iterator[bytes | bytearray | memoryview]:
    yield _length_prefix(len(value))
    yield value


def iter_source_label_input_parts(
    *,
    session_seed: bytes | bytearray | memoryview | str,
    graph_descriptor_digest: bytes | bytearray | memoryview | str,
    node_index: int,
) -> Iterator[bytes | bytearray | memoryview]:
    session_seed_bytes = _field_bytes_like(session_seed, field_name="session_seed")
    descriptor_bytes = _field_bytes_like(
        graph_descriptor_digest,
        field_name="graph_descriptor_digest",
    )
    node_index_bytes = _encode_u64(node_index)
    yield from _iter_domain_separated_message_parts(SOURCE_LABEL_DOMAIN, field_count=3)
    yield from _iter_length_prefixed_field_parts(session_seed_bytes)
    yield from _iter_length_prefixed_field_parts(descriptor_bytes)
    yield from _iter_length_prefixed_field_parts(node_index_bytes)


def iter_internal_label_input_parts(
    *,
    session_seed: bytes | bytearray | memoryview | str,
    graph_descriptor_digest: bytes | bytearray | memoryview | str,
    node_index: int,
    predecessor_labels: Iterable[bytes | bytearray | memoryview],
    predecessor_count: int | None = None,
) -> Iterator[bytes | bytearray | memoryview]:
    session_seed_bytes = _field_bytes_like(session_seed, field_name="session_seed")
    descriptor_bytes = _field_bytes_like(
        graph_descriptor_digest,
        field_name="graph_descriptor_digest",
    )
    if predecessor_count is None:
        if isinstance(predecessor_labels, Sequence):
            labels_iterable: Iterable[bytes | bytearray | memoryview] = predecessor_labels
            predecessor_count = len(predecessor_labels)
        else:
            labels_tuple = tuple(predecessor_labels)
            labels_iterable = labels_tuple
            predecessor_count = len(labels_tuple)
    else:
        labels_iterable = predecessor_labels
    node_index_bytes = _encode_u64(node_index)
    predecessor_count_bytes = _encode_u32(predecessor_count)
    yield from _iter_domain_separated_message_parts(
        INTERNAL_LABEL_DOMAIN,
        field_count=4 + predecessor_count,
    )
    yield from _iter_length_prefixed_field_parts(session_seed_bytes)
    yield from _iter_length_prefixed_field_parts(descriptor_bytes)
    yield from _iter_length_prefixed_field_parts(node_index_bytes)
    yield from _iter_length_prefixed_field_parts(predecessor_count_bytes)
    for label_bytes in labels_iterable:
        normalized = _field_bytes_like(label_bytes, field_name="predecessor_label")
        yield from _iter_length_prefixed_field_parts(normalized)


def _join_parts(parts: Iterable[bytes | bytearray | memoryview]) -> bytes:
    return b"".join(
        part if isinstance(part, bytes) else bytes(part)
        for part in parts
    )


def encode_domain_separated_message(domain: str, fields: Sequence[bytes]) -> bytes:
    if not domain:
        raise ProtocolError("Domain tag must be a non-empty string.")
    payload = bytearray()
    payload.extend(_DOMAIN_NAMESPACE)
    payload.extend(_encode_u32(_ENCODING_VERSION))
    payload.extend(_length_prefixed(domain.encode("ascii")))
    payload.extend(_encode_u32(len(fields)))
    for field in fields:
        payload.extend(_length_prefixed(field))
    return bytes(payload)


def encode_graph_descriptor_input(
    *,
    graph_family: str,
    label_count_m: int,
    graph_parameter_n: int,
    gamma: int,
    node_ordering_version: str = "pose-db-node-order/v1",
    challenge_set_ordering_version: str = "pose-db-challenge-order/v1",
    hash_backend: str,
    label_width_bits: int,
) -> bytes:
    return encode_domain_separated_message(
        GRAPH_DESCRIPTOR_DOMAIN,
        (
            _field_bytes(graph_family, field_name="graph_family"),
            _encode_u64(label_count_m),
            _encode_u64(graph_parameter_n),
            _encode_u64(gamma),
            _field_bytes(node_ordering_version, field_name="node_ordering_version"),
            _field_bytes(
                challenge_set_ordering_version,
                field_name="challenge_set_ordering_version",
            ),
            _field_bytes(hash_backend, field_name="hash_backend"),
            _encode_u64(label_width_bits),
        ),
    )


def encode_source_label_input(
    *,
    session_seed: bytes | str,
    graph_descriptor_digest: bytes | str,
    node_index: int,
) -> bytes:
    return encode_domain_separated_message(
        SOURCE_LABEL_DOMAIN,
        (
            _field_bytes(session_seed, field_name="session_seed"),
            _field_bytes(graph_descriptor_digest, field_name="graph_descriptor_digest"),
            _encode_u64(node_index),
        ),
    )


def encode_internal_label_input(
    *,
    session_seed: bytes | str,
    graph_descriptor_digest: bytes | str,
    node_index: int,
    predecessor_labels: Sequence[bytes],
) -> bytes:
    return encode_domain_separated_message(
        INTERNAL_LABEL_DOMAIN,
        (
            _field_bytes(session_seed, field_name="session_seed"),
            _field_bytes(graph_descriptor_digest, field_name="graph_descriptor_digest"),
            _encode_u64(node_index),
            _encode_u32(len(predecessor_labels)),
            *predecessor_labels,
        ),
    )
