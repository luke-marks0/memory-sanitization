from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from struct import pack_into

from pose.common.errors import ProtocolError
from pose.hashing.blake3_backend import (
    hash_xof as blake3_xof,
    hash_xof_parts as blake3_xof_parts,
)
from pose.hashing.encoding import (
    INTERNAL_LABEL_DOMAIN,
    SOURCE_LABEL_DOMAIN,
    _DOMAIN_NAMESPACE,
    _ENCODING_VERSION_BYTES,
    _encode_u32,
    _encode_u64,
    _field_bytes_like,
    _length_prefix,
    encode_graph_descriptor_input,
    encode_internal_label_input,
    encode_source_label_input,
)
from pose.hashing.shake256_backend import (
    hash_xof as shake256_xof,
    hash_xof_parts as shake256_xof_parts,
)

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


def hash_xof_parts(
    parts: Iterable[bytes | bytearray | memoryview],
    *,
    hash_backend: str | None = None,
    length: int,
) -> bytes:
    backend = normalize_hash_backend(hash_backend)
    if length <= 0:
        raise ProtocolError(f"Hash output length must be positive, got {length}")
    if backend == DEFAULT_HASH_BACKEND:
        return blake3_xof_parts(parts, length=length)
    if backend == "shake256":
        return shake256_xof_parts(parts, length=length)
    raise ProtocolError(f"Unsupported hash backend dispatch: {backend}")


def hash_xof_hex(data: bytes, *, hash_backend: str | None = None, length: int) -> str:
    backend = normalize_hash_backend(hash_backend)
    return f"{backend}:{hash_xof(data, hash_backend=backend, length=length).hex()}"


@dataclass
class LabelOracleContext:
    hash_backend: str
    output_bytes: int
    session_seed_bytes: bytes
    graph_descriptor_digest_bytes: bytes
    source_prefix: bytes
    internal_prefixes: dict[int, bytes] = field(default_factory=dict)
    predecessor_count_fields: dict[int, bytes] = field(default_factory=dict)
    label_length_prefix: bytes = b""
    source_payload: bytearray = field(default_factory=bytearray, repr=False)
    source_node_index_offset: int = 0
    internal_payloads: dict[int, bytearray] = field(default_factory=dict, repr=False)
    internal_node_index_offsets: dict[int, int] = field(default_factory=dict, repr=False)
    internal_label_offsets: dict[int, tuple[int, ...]] = field(default_factory=dict, repr=False)
    _hash_fn: Callable[[bytes | bytearray | memoryview], bytes] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.output_bytes <= 0:
            raise ProtocolError(f"Hash output length must be positive, got {self.output_bytes}")
        backend = normalize_hash_backend(self.hash_backend)
        self.hash_backend = backend
        if backend == DEFAULT_HASH_BACKEND:
            self._hash_fn = lambda payload: blake3_xof(payload, length=self.output_bytes)
        else:
            self._hash_fn = lambda payload: shake256_xof(payload, length=self.output_bytes)
        self.source_payload = bytearray(self.source_prefix)
        self.source_node_index_offset = len(self.source_payload)
        self.source_payload.extend(b"\x00" * 8)

    @classmethod
    def create(
        cls,
        *,
        session_seed: bytes | bytearray | memoryview | str,
        graph_descriptor_digest: bytes | bytearray | memoryview | str,
        hash_backend: str | None = None,
        output_bytes: int,
        max_predecessor_count: int = 0,
    ) -> "LabelOracleContext":
        seed_bytes = bytes(_field_bytes_like(session_seed, field_name="session_seed"))
        descriptor_bytes = bytes(
            _field_bytes_like(
                graph_descriptor_digest,
                field_name="graph_descriptor_digest",
            )
        )
        source_prefix = cls._build_source_prefix(
            session_seed=seed_bytes,
            graph_descriptor_digest=descriptor_bytes,
        )
        context = cls(
            hash_backend=normalize_hash_backend(hash_backend),
            output_bytes=output_bytes,
            session_seed_bytes=seed_bytes,
            graph_descriptor_digest_bytes=descriptor_bytes,
            source_prefix=source_prefix,
            label_length_prefix=_length_prefix(output_bytes),
        )
        for predecessor_count in range(max(0, max_predecessor_count) + 1):
            context._ensure_internal_prefix(predecessor_count)
        return context

    @staticmethod
    def _domain_prefix(domain: str, *, field_count: int) -> bytes:
        domain_bytes = domain.encode("ascii")
        payload = bytearray()
        payload.extend(_DOMAIN_NAMESPACE)
        payload.extend(_ENCODING_VERSION_BYTES)
        payload.extend(_length_prefix(len(domain_bytes)))
        payload.extend(domain_bytes)
        payload.extend(_encode_u32(field_count))
        return bytes(payload)

    @classmethod
    def _build_source_prefix(
        cls,
        *,
        session_seed: bytes,
        graph_descriptor_digest: bytes,
    ) -> bytes:
        payload = bytearray(cls._domain_prefix(SOURCE_LABEL_DOMAIN, field_count=3))
        payload.extend(_length_prefix(len(session_seed)))
        payload.extend(session_seed)
        payload.extend(_length_prefix(len(graph_descriptor_digest)))
        payload.extend(graph_descriptor_digest)
        payload.extend(_length_prefix(8))
        return bytes(payload)

    @classmethod
    def _build_internal_prefix(
        cls,
        *,
        session_seed: bytes,
        graph_descriptor_digest: bytes,
        predecessor_count: int,
    ) -> tuple[bytes, bytes]:
        payload = bytearray(cls._domain_prefix(INTERNAL_LABEL_DOMAIN, field_count=4 + predecessor_count))
        payload.extend(_length_prefix(len(session_seed)))
        payload.extend(session_seed)
        payload.extend(_length_prefix(len(graph_descriptor_digest)))
        payload.extend(graph_descriptor_digest)
        payload.extend(_length_prefix(8))
        predecessor_count_field = _length_prefix(4) + _encode_u32(predecessor_count)
        return bytes(payload), predecessor_count_field

    def _ensure_internal_prefix(self, predecessor_count: int) -> None:
        if predecessor_count in self.internal_prefixes:
            return
        prefix, predecessor_count_field = self._build_internal_prefix(
            session_seed=self.session_seed_bytes,
            graph_descriptor_digest=self.graph_descriptor_digest_bytes,
            predecessor_count=predecessor_count,
        )
        self.internal_prefixes[predecessor_count] = prefix
        self.predecessor_count_fields[predecessor_count] = predecessor_count_field
        payload = bytearray(prefix)
        node_index_offset = len(payload)
        payload.extend(b"\x00" * 8)
        payload.extend(predecessor_count_field)
        label_offsets: list[int] = []
        for _ in range(predecessor_count):
            payload.extend(self.label_length_prefix)
            label_offsets.append(len(payload))
            payload.extend(b"\x00" * self.output_bytes)
        self.internal_payloads[predecessor_count] = payload
        self.internal_node_index_offsets[predecessor_count] = node_index_offset
        self.internal_label_offsets[predecessor_count] = tuple(label_offsets)

    @staticmethod
    def _pack_node_index(payload: bytearray, *, offset: int, node_index: int) -> None:
        if node_index < 0 or node_index >= 2**64:
            raise ProtocolError(f"Expected uint64-compatible value, got {node_index}")
        pack_into(">Q", payload, offset, int(node_index))

    def source_label(self, *, node_index: int) -> bytes:
        self._pack_node_index(
            self.source_payload,
            offset=self.source_node_index_offset,
            node_index=node_index,
        )
        return self._hash_fn(self.source_payload)

    def internal_label_1(
        self,
        *,
        node_index: int,
        predecessor0: bytes | bytearray | memoryview,
    ) -> bytes:
        self._ensure_internal_prefix(1)
        normalized0 = _field_bytes_like(predecessor0, field_name="predecessor_label")
        if len(normalized0) != self.output_bytes:
            return self._internal_label_generic(
                node_index=node_index,
                predecessor_labels=(normalized0,),
                predecessor_count=1,
            )
        payload = self.internal_payloads[1]
        self._pack_node_index(
            payload,
            offset=self.internal_node_index_offsets[1],
            node_index=node_index,
        )
        label0_offset = self.internal_label_offsets[1][0]
        payload[label0_offset : label0_offset + self.output_bytes] = normalized0
        return self._hash_fn(payload)

    def internal_label_2(
        self,
        *,
        node_index: int,
        predecessor0: bytes | bytearray | memoryview,
        predecessor1: bytes | bytearray | memoryview,
    ) -> bytes:
        self._ensure_internal_prefix(2)
        normalized0 = _field_bytes_like(predecessor0, field_name="predecessor_label")
        normalized1 = _field_bytes_like(predecessor1, field_name="predecessor_label")
        if len(normalized0) != self.output_bytes or len(normalized1) != self.output_bytes:
            return self._internal_label_generic(
                node_index=node_index,
                predecessor_labels=(normalized0, normalized1),
                predecessor_count=2,
            )
        payload = self.internal_payloads[2]
        self._pack_node_index(
            payload,
            offset=self.internal_node_index_offsets[2],
            node_index=node_index,
        )
        label0_offset, label1_offset = self.internal_label_offsets[2]
        payload[label0_offset : label0_offset + self.output_bytes] = normalized0
        payload[label1_offset : label1_offset + self.output_bytes] = normalized1
        return self._hash_fn(payload)

    def _internal_label_generic(
        self,
        *,
        node_index: int,
        predecessor_labels: Iterable[bytes | bytearray | memoryview],
        predecessor_count: int,
    ) -> bytes:
        prefix = self.internal_prefixes.get(predecessor_count)
        predecessor_count_field = self.predecessor_count_fields.get(predecessor_count)
        if prefix is None or predecessor_count_field is None:
            self._ensure_internal_prefix(predecessor_count)
            prefix = self.internal_prefixes[predecessor_count]
            predecessor_count_field = self.predecessor_count_fields[predecessor_count]
        payload = bytearray(prefix)
        payload.extend(_encode_u64(node_index))
        payload.extend(predecessor_count_field)
        for label_bytes in predecessor_labels:
            normalized = _field_bytes_like(label_bytes, field_name="predecessor_label")
            if len(normalized) == self.output_bytes:
                payload.extend(self.label_length_prefix)
            else:
                payload.extend(_length_prefix(len(normalized)))
            payload.extend(normalized)
        return self._hash_fn(payload)

    def internal_label(
        self,
        *,
        node_index: int,
        predecessor_labels: Iterable[bytes | bytearray | memoryview],
        predecessor_count: int | None = None,
    ) -> bytes:
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
        if predecessor_count == 1 and isinstance(labels_iterable, Sequence):
            return self.internal_label_1(
                node_index=node_index,
                predecessor0=labels_iterable[0],
            )
        if predecessor_count == 2 and isinstance(labels_iterable, Sequence):
            return self.internal_label_2(
                node_index=node_index,
                predecessor0=labels_iterable[0],
                predecessor1=labels_iterable[1],
            )
        return self._internal_label_generic(
            node_index=node_index,
            predecessor_labels=labels_iterable,
            predecessor_count=predecessor_count,
        )


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


def source_label_bytes_accelerated(
    *,
    session_seed: bytes | str,
    graph_descriptor_digest: bytes | str,
    node_index: int,
    hash_backend: str | None = None,
    output_bytes: int,
) -> bytes:
    return LabelOracleContext.create(
        session_seed=session_seed,
        graph_descriptor_digest=graph_descriptor_digest,
        hash_backend=hash_backend,
        output_bytes=output_bytes,
    ).source_label(
        node_index=node_index,
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


def internal_label_bytes_accelerated(
    *,
    session_seed: bytes | str,
    graph_descriptor_digest: bytes | str,
    node_index: int,
    predecessor_labels: Iterable[bytes | bytearray | memoryview],
    predecessor_count: int | None = None,
    hash_backend: str | None = None,
    output_bytes: int,
) -> bytes:
    context = LabelOracleContext.create(
        session_seed=session_seed,
        graph_descriptor_digest=graph_descriptor_digest,
        hash_backend=hash_backend,
        output_bytes=output_bytes,
        max_predecessor_count=predecessor_count or 0,
    )
    return context.internal_label(
        node_index=node_index,
        predecessor_labels=predecessor_labels,
        predecessor_count=predecessor_count,
    )
