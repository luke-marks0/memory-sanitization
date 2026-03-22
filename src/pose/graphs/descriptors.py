from __future__ import annotations

from dataclasses import dataclass

from pose.common.cbor import canonical_cbor_dumps
from pose.common.errors import ProtocolError
from pose.common.hashing import sha256_hex
from pose.hashing import normalize_hash_backend

GRAPH_FAMILY = "pose-db-drg-v1"
NODE_ORDERING_VERSION = "pose-db-node-order/v1"
CHALLENGE_SET_ORDERING_VERSION = "pose-db-challenge-order/v1"


def expected_graph_parameter_n(label_count_m: int) -> int:
    if label_count_m <= 0:
        raise ProtocolError(f"label_count_m must be positive, got {label_count_m}")
    return max(0, (label_count_m - 1).bit_length() - 1)


def gamma_for_graph_parameter_n(graph_parameter_n: int) -> int:
    if graph_parameter_n < 0:
        raise ProtocolError(f"graph_parameter_n must be non-negative, got {graph_parameter_n}")
    return 1 << graph_parameter_n


def validate_label_width_bits(label_width_bits: int) -> int:
    if label_width_bits < 128:
        raise ProtocolError(f"label_width_bits must be at least 128, got {label_width_bits}")
    if label_width_bits % 8 != 0:
        raise ProtocolError(f"label_width_bits must be byte-aligned, got {label_width_bits}")
    return label_width_bits


@dataclass(frozen=True)
class GraphDescriptor:
    graph_family: str
    label_count_m: int
    graph_parameter_n: int
    gamma: int
    hash_backend: str
    label_width_bits: int
    node_ordering_version: str = NODE_ORDERING_VERSION
    challenge_set_ordering_version: str = CHALLENGE_SET_ORDERING_VERSION

    def to_cbor_object(self) -> dict[str, object]:
        return {
            "challenge_set_ordering_version": self.challenge_set_ordering_version,
            "gamma": self.gamma,
            "graph_family": self.graph_family,
            "graph_parameter_n": self.graph_parameter_n,
            "hash_backend": self.hash_backend,
            "label_count_m": self.label_count_m,
            "label_width_bits": self.label_width_bits,
            "node_ordering_version": self.node_ordering_version,
        }

    @property
    def digest(self) -> str:
        return f"sha256:{sha256_hex(canonical_cbor_dumps(self.to_cbor_object()))}"


def build_graph_descriptor(
    *,
    label_count_m: int,
    graph_parameter_n: int | None = None,
    gamma: int | None = None,
    hash_backend: str,
    label_width_bits: int,
    graph_family: str = GRAPH_FAMILY,
    node_ordering_version: str = NODE_ORDERING_VERSION,
    challenge_set_ordering_version: str = CHALLENGE_SET_ORDERING_VERSION,
) -> GraphDescriptor:
    normalized_hash_backend = normalize_hash_backend(hash_backend)
    validated_label_width_bits = validate_label_width_bits(label_width_bits)
    computed_n = expected_graph_parameter_n(label_count_m) if graph_parameter_n is None else int(graph_parameter_n)
    computed_gamma = gamma_for_graph_parameter_n(computed_n)
    if gamma is not None and int(gamma) != computed_gamma:
        raise ProtocolError(
            f"gamma must equal 2^graph_parameter_n for pose-db-drg-v1, got gamma={gamma} and n={computed_n}"
        )
    if computed_n != expected_graph_parameter_n(label_count_m):
        raise ProtocolError(
            "graph_parameter_n must be the smallest integer such that 2^(n+1) >= label_count_m "
            f"for pose-db-drg-v1, got n={computed_n} and m={label_count_m}"
        )
    if graph_family != GRAPH_FAMILY:
        raise ProtocolError(f"Unsupported graph family: {graph_family!r}")
    return GraphDescriptor(
        graph_family=graph_family,
        label_count_m=int(label_count_m),
        graph_parameter_n=computed_n,
        gamma=computed_gamma,
        hash_backend=normalized_hash_backend,
        label_width_bits=validated_label_width_bits,
        node_ordering_version=node_ordering_version,
        challenge_set_ordering_version=challenge_set_ordering_version,
    )
