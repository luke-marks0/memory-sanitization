from __future__ import annotations

import pytest

import pose.graphs.labeling as labeling
from pose.graphs import build_pose_db_graph, compute_challenge_labels, compute_label_array, compute_node_labels


@pytest.mark.parametrize("hash_backend", ["blake3-xof", "shake256"])
def test_accelerated_label_engine_matches_reference_graph_outputs(hash_backend: str) -> None:
    graph = build_pose_db_graph(
        label_count_m=17,
        hash_backend=hash_backend,
        label_width_bits=256,
    )
    session_seed = "55" * 32
    challenge_indices = [0, 3, 7, 16, 3]

    reference_node_labels = compute_node_labels(
        graph,
        session_seed=session_seed,
        label_engine="reference",
    )
    accelerated_node_labels = compute_node_labels(
        graph,
        session_seed=session_seed,
        label_engine="accelerated",
    )
    reference_challenge_labels = compute_challenge_labels(
        graph,
        session_seed=session_seed,
        challenge_indices=challenge_indices,
        label_engine="reference",
    )
    accelerated_challenge_labels = compute_challenge_labels(
        graph,
        session_seed=session_seed,
        challenge_indices=challenge_indices,
        label_engine="accelerated",
    )
    reference_label_array = compute_label_array(
        graph,
        session_seed=session_seed,
        label_engine="reference",
    )
    accelerated_label_array = compute_label_array(
        graph,
        session_seed=session_seed,
        label_engine="accelerated",
    )

    assert accelerated_node_labels == reference_node_labels
    assert accelerated_challenge_labels == reference_challenge_labels
    assert accelerated_label_array == reference_label_array
    assert compute_node_labels(graph, session_seed=session_seed) == accelerated_node_labels
    assert compute_challenge_labels(graph, session_seed=session_seed) == compute_challenge_labels(
        graph,
        session_seed=session_seed,
        label_engine="accelerated",
    )
    assert compute_label_array(graph, session_seed=session_seed) == accelerated_label_array
    assert accelerated_label_array == b"".join(
        compute_challenge_labels(
            graph,
            session_seed=session_seed,
            label_engine="accelerated",
        )
    )


@pytest.mark.parametrize("hash_backend", ["blake3-xof", "shake256"])
def test_streaming_challenge_label_path_matches_reference(
    hash_backend: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = build_pose_db_graph(
        label_count_m=17,
        hash_backend=hash_backend,
        label_width_bits=256,
    )
    session_seed = "66" * 32
    challenge_indices = [0, 3, 7, 16, 3]

    monkeypatch.setattr(labeling, "_STREAMING_CHALLENGE_LABEL_THRESHOLD_BYTES", 0)

    assert compute_challenge_labels(
        graph,
        session_seed=session_seed,
        challenge_indices=challenge_indices,
        label_engine="accelerated",
    ) == compute_challenge_labels(
        graph,
        session_seed=session_seed,
        challenge_indices=challenge_indices,
        label_engine="reference",
    )
    assert compute_label_array(
        graph,
        session_seed=session_seed,
        label_engine="accelerated",
    ) == compute_label_array(
        graph,
        session_seed=session_seed,
        label_engine="reference",
    )
