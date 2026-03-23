from __future__ import annotations

from collections.abc import Sequence

from pose.graphs.construction import PoseDbGraph
from pose.hashing import (
    LabelOracleContext,
    internal_label_bytes,
    source_label_bytes,
)

DEFAULT_LABEL_ENGINE = "accelerated"
SUPPORTED_LABEL_ENGINES = frozenset(("reference", DEFAULT_LABEL_ENGINE))


def _session_seed_bytes(session_seed: bytes | str) -> bytes:
    if isinstance(session_seed, bytes):
        return session_seed
    return bytes.fromhex(session_seed)


def normalize_label_engine(label_engine: str | None) -> str:
    candidate = DEFAULT_LABEL_ENGINE if label_engine is None else str(label_engine).strip().lower()
    if candidate == "auto":
        return DEFAULT_LABEL_ENGINE
    if candidate not in SUPPORTED_LABEL_ENGINES:
        supported = ", ".join(sorted(SUPPORTED_LABEL_ENGINES))
        raise ValueError(
            f"Unsupported label engine: {label_engine!r}. Expected one of: {supported}"
        )
    return candidate


def _compute_node_labels_reference(
    graph: PoseDbGraph,
    *,
    session_seed: bytes | str,
) -> tuple[bytes, ...]:
    seed = _session_seed_bytes(session_seed)
    output_bytes = graph.label_width_bits // 8
    labels: list[bytes] = []

    def consume(predecessor_count: int, predecessor0: int, predecessor1: int) -> None:
        node_index = len(labels)
        if predecessor_count == 0:
            labels.append(
                source_label_bytes(
                    session_seed=seed,
                    graph_descriptor_digest=graph.graph_descriptor_digest,
                    node_index=node_index,
                    hash_backend=graph.hash_backend,
                    output_bytes=output_bytes,
                )
            )
            return
        labels.append(
            internal_label_bytes(
                session_seed=seed,
                graph_descriptor_digest=graph.graph_descriptor_digest,
                node_index=node_index,
                predecessor_labels=(
                    [labels[predecessor0]]
                    if predecessor_count == 1
                    else [labels[predecessor0], labels[predecessor1]]
                ),
                hash_backend=graph.hash_backend,
                output_bytes=output_bytes,
            )
        )

    graph.visit_predecessor_specs(consume)
    return tuple(labels)


def _compute_all_labels_buffer_accelerated(
    graph: PoseDbGraph,
    *,
    session_seed: bytes | str,
) -> tuple[bytearray, memoryview]:
    seed = _session_seed_bytes(session_seed)
    output_bytes = graph.label_width_bits // 8
    label_oracle = LabelOracleContext.create(
        session_seed=seed,
        graph_descriptor_digest=graph.graph_descriptor_digest,
        hash_backend=graph.hash_backend,
        output_bytes=output_bytes,
        max_predecessor_count=graph.max_predecessor_count,
    )
    labels_buffer = bytearray(graph.node_count * output_bytes)
    labels_view = memoryview(labels_buffer)
    node_index = 0

    def consume(predecessor_count: int, predecessor0: int, predecessor1: int) -> None:
        nonlocal node_index
        if predecessor_count == 0:
            label_bytes = label_oracle.source_label(
                node_index=node_index,
            )
        elif predecessor_count == 1:
            start0 = predecessor0 * output_bytes
            label_bytes = label_oracle.internal_label_1(
                node_index=node_index,
                predecessor0=labels_view[start0 : start0 + output_bytes],
            )
        else:
            start0 = predecessor0 * output_bytes
            start1 = predecessor1 * output_bytes
            label_bytes = label_oracle.internal_label_2(
                node_index=node_index,
                predecessor0=labels_view[start0 : start0 + output_bytes],
                predecessor1=labels_view[start1 : start1 + output_bytes],
            )
        offset = node_index * output_bytes
        labels_view[offset : offset + output_bytes] = label_bytes
        node_index += 1

    graph.visit_predecessor_specs(consume)
    return labels_buffer, labels_view


def compute_node_labels(
    graph: PoseDbGraph,
    *,
    session_seed: bytes | str,
    label_engine: str | None = None,
) -> tuple[bytes, ...]:
    engine = normalize_label_engine(label_engine)
    if engine == "reference":
        return _compute_node_labels_reference(graph, session_seed=session_seed)
    output_bytes = graph.label_width_bits // 8
    _labels_buffer, labels_view = _compute_all_labels_buffer_accelerated(
        graph,
        session_seed=session_seed,
    )
    return tuple(
        bytes(labels_view[node_index * output_bytes : (node_index + 1) * output_bytes])
        for node_index in range(graph.node_count)
    )


def compute_challenge_labels(
    graph: PoseDbGraph,
    *,
    session_seed: bytes | str,
    challenge_indices: Sequence[int] | None = None,
    label_engine: str | None = None,
) -> tuple[bytes, ...]:
    engine = normalize_label_engine(label_engine)
    if engine == "reference":
        node_labels = _compute_node_labels_reference(graph, session_seed=session_seed)
        if challenge_indices is None:
            return tuple(node_labels[node] for node in graph.challenge_set)
        return tuple(node_labels[graph.challenge_node(challenge_index)] for challenge_index in challenge_indices)

    output_bytes = graph.label_width_bits // 8
    _labels_buffer, labels_view = _compute_all_labels_buffer_accelerated(
        graph,
        session_seed=session_seed,
    )
    if challenge_indices is None:
        challenge_nodes = graph.challenge_set
    else:
        challenge_nodes = tuple(graph.challenge_node(challenge_index) for challenge_index in challenge_indices)
    return tuple(
        bytes(labels_view[node * output_bytes : (node + 1) * output_bytes])
        for node in challenge_nodes
    )


def compute_label_array(
    graph: PoseDbGraph,
    *,
    session_seed: bytes | str,
    label_engine: str | None = None,
) -> bytes:
    engine = normalize_label_engine(label_engine)
    if engine == "reference":
        return b"".join(
            compute_challenge_labels(
                graph,
                session_seed=session_seed,
                label_engine="reference",
            )
        )

    output_bytes = graph.label_width_bits // 8
    _labels_buffer, labels_view = _compute_all_labels_buffer_accelerated(
        graph,
        session_seed=session_seed,
    )
    challenge_label_array = bytearray(graph.label_count_m * output_bytes)
    challenge_view = memoryview(challenge_label_array)
    for challenge_index, node in enumerate(graph.challenge_set):
        source_start = node * output_bytes
        source_end = source_start + output_bytes
        dest_start = challenge_index * output_bytes
        challenge_view[dest_start : dest_start + output_bytes] = labels_view[source_start:source_end]
    return bytes(challenge_label_array)
