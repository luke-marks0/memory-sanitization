from __future__ import annotations

from collections.abc import Sequence

from pose.graphs.construction import PoseDbGraph
from pose.hashing import internal_label_bytes, source_label_bytes


def _session_seed_bytes(session_seed: bytes | str) -> bytes:
    if isinstance(session_seed, bytes):
        return session_seed
    return bytes.fromhex(session_seed)


def compute_node_labels(graph: PoseDbGraph, *, session_seed: bytes | str) -> tuple[bytes, ...]:
    seed = _session_seed_bytes(session_seed)
    output_bytes = graph.label_width_bits // 8
    labels: list[bytes] = []
    for node_index, predecessor_nodes in enumerate(graph.predecessors):
        if not predecessor_nodes:
            labels.append(
                source_label_bytes(
                    session_seed=seed,
                    graph_descriptor_digest=graph.graph_descriptor_digest,
                    node_index=node_index,
                    hash_backend=graph.hash_backend,
                    output_bytes=output_bytes,
                )
            )
            continue
        labels.append(
            internal_label_bytes(
                session_seed=seed,
                graph_descriptor_digest=graph.graph_descriptor_digest,
                node_index=node_index,
                predecessor_labels=[labels[predecessor] for predecessor in predecessor_nodes],
                hash_backend=graph.hash_backend,
                output_bytes=output_bytes,
            )
        )
    return tuple(labels)


def compute_challenge_labels(
    graph: PoseDbGraph,
    *,
    session_seed: bytes | str,
    challenge_indices: Sequence[int] | None = None,
) -> tuple[bytes, ...]:
    node_labels = compute_node_labels(graph, session_seed=session_seed)
    if challenge_indices is None:
        return tuple(node_labels[node] for node in graph.challenge_set)
    return tuple(node_labels[graph.challenge_node(challenge_index)] for challenge_index in challenge_indices)


def compute_label_array(graph: PoseDbGraph, *, session_seed: bytes | str) -> bytes:
    return b"".join(compute_challenge_labels(graph, session_seed=session_seed))
