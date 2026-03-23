from __future__ import annotations

import itertools
import json
from pathlib import Path

import pytest

from pose.common.errors import ProtocolError
from pose.graphs.construction import _build_pose_db_graph_uncached
from pose.graphs import (
    build_graph_descriptor,
    build_pose_db_graph,
    clear_pose_db_graph_cache,
    compute_challenge_labels,
    compute_label_array,
    compute_node_labels,
    expected_graph_parameter_n,
    gamma_for_graph_parameter_n,
    pose_db_graph_cache_info,
    validate_label_width_bits,
)


FIXTURE_PATH = Path(__file__).with_name("fixtures") / "pose_db_reference_vectors.json"


def _reference_vectors() -> dict[str, object]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _longest_path_lengths_after_removal(
    predecessors: tuple[tuple[int, ...], ...],
    removed_nodes: frozenset[int],
) -> tuple[int, ...]:
    lengths: list[int] = []
    for node_index, node_predecessors in enumerate(predecessors):
        if node_index in removed_nodes:
            lengths.append(-1)
            continue
        remaining_predecessors = [lengths[predecessor] for predecessor in node_predecessors if predecessor not in removed_nodes]
        lengths.append(0 if not remaining_predecessors else 1 + max(remaining_predecessors))
    return tuple(lengths)


@pytest.mark.parametrize(
    ("label_count_m", "expected_n", "expected_gamma"),
    [
        (1, 0, 1),
        (2, 0, 1),
        (3, 1, 2),
        (4, 1, 2),
        (5, 2, 4),
        (8, 2, 4),
    ],
)
def test_arbitrary_m_selects_the_smallest_valid_graph_parameter(
    label_count_m: int,
    expected_n: int,
    expected_gamma: int,
) -> None:
    graph = build_pose_db_graph(
        label_count_m=label_count_m,
        hash_backend="blake3-xof",
        label_width_bits=128,
    )

    assert expected_graph_parameter_n(label_count_m) == expected_n
    assert gamma_for_graph_parameter_n(expected_n) == expected_gamma
    assert graph.label_count_m == label_count_m
    assert graph.graph_parameter_n == expected_n
    assert graph.gamma == expected_gamma
    assert len(graph.challenge_set) == label_count_m
    assert [graph.challenge_node(index) for index in range(label_count_m)] == list(graph.challenge_set)

    with pytest.raises(ProtocolError, match="outside"):
        graph.challenge_node(label_count_m)


def test_label_width_validation_rules() -> None:
    assert validate_label_width_bits(128) == 128
    assert validate_label_width_bits(256) == 256

    with pytest.raises(ProtocolError, match="at least 128"):
        validate_label_width_bits(120)

    with pytest.raises(ProtocolError, match="byte-aligned"):
        validate_label_width_bits(130)


@pytest.mark.parametrize("hash_backend", ["blake3-xof", "shake256"])
@pytest.mark.parametrize("label_engine", ["reference", "accelerated"])
def test_reference_vectors_are_stable(hash_backend: str, label_engine: str) -> None:
    payload = _reference_vectors()
    shared = payload["shared_graph"]
    expected = payload["cases"][hash_backend]
    graph = build_pose_db_graph(
        label_count_m=shared["label_count_m"],
        graph_parameter_n=shared["graph_parameter_n"],
        gamma=shared["gamma"],
        hash_backend=hash_backend,
        label_width_bits=shared["label_width_bits"],
    )
    descriptor = build_graph_descriptor(
        label_count_m=shared["label_count_m"],
        graph_parameter_n=shared["graph_parameter_n"],
        gamma=shared["gamma"],
        hash_backend=hash_backend,
        label_width_bits=shared["label_width_bits"],
    )
    challenge_labels = compute_challenge_labels(
        graph,
        session_seed=payload["seed_hex"],
        label_engine=label_engine,
    )
    node_labels = compute_node_labels(
        graph,
        session_seed=payload["seed_hex"],
        label_engine=label_engine,
    )

    assert descriptor.digest == expected["graph_descriptor_digest"]
    assert graph.graph_descriptor_digest == expected["graph_descriptor_digest"]
    assert graph.node_count == shared["node_count"]
    assert list(graph.challenge_set) == shared["challenge_set"]
    assert [list(predecessors) for predecessors in graph.predecessors] == shared["predecessors"]
    assert [label.hex() for label in node_labels[:10]] == expected["first_ten_node_labels_hex"]
    assert [label.hex() for label in challenge_labels] == expected["challenge_labels_hex"]
    assert compute_label_array(
        graph,
        session_seed=payload["seed_hex"],
        label_engine=label_engine,
    ).hex() == "".join(
        expected["challenge_labels_hex"]
    )


@pytest.mark.parametrize("label_count_m", [1, 2, 3, 5, 8, 17, 33, 64])
def test_formula_driven_graph_matches_explicit_reference_builder(label_count_m: int) -> None:
    descriptor = build_graph_descriptor(
        label_count_m=label_count_m,
        hash_backend="blake3-xof",
        label_width_bits=256,
    )
    implicit_graph = build_pose_db_graph(
        label_count_m=label_count_m,
        hash_backend="blake3-xof",
        label_width_bits=256,
    )
    explicit_graph = _build_pose_db_graph_uncached(descriptor)

    assert implicit_graph.node_count == explicit_graph.node_count
    assert implicit_graph.challenge_set == explicit_graph.challenge_set
    assert implicit_graph.predecessors == explicit_graph.predecessors
    assert implicit_graph.max_predecessor_count == explicit_graph.max_predecessor_count


@pytest.mark.parametrize("hash_backend", ["blake3-xof", "shake256"])
def test_accelerated_label_engine_matches_reference(hash_backend: str) -> None:
    graph = build_pose_db_graph(
        label_count_m=17,
        hash_backend=hash_backend,
        label_width_bits=256,
    )
    session_seed = "ab" * 32
    challenge_indices = [0, 4, 9, 16, 4]

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

    assert accelerated_node_labels == reference_node_labels
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


def test_small_graphs_are_exhaustively_depth_robust() -> None:
    for label_count_m in range(1, 5):
        graph = build_pose_db_graph(
            label_count_m=label_count_m,
            hash_backend="blake3-xof",
            label_width_bits=128,
        )
        node_indices = range(graph.node_count)
        for removed_count in range(label_count_m):
            for removed_nodes in itertools.combinations(node_indices, removed_count):
                remaining_lengths = _longest_path_lengths_after_removal(
                    graph.predecessors,
                    frozenset(removed_nodes),
                )
                surviving_challenges = [
                    challenge_node
                    for challenge_node in graph.challenge_set
                    if challenge_node not in removed_nodes and remaining_lengths[challenge_node] >= graph.gamma
                ]
                assert len(surviving_challenges) >= label_count_m - removed_count


@pytest.mark.parametrize("label_count_m", [5, 8, 17, 33, 129])
def test_large_graphs_preserve_topological_and_challenge_invariants(label_count_m: int) -> None:
    graph = build_pose_db_graph(
        label_count_m=label_count_m,
        hash_backend="blake3-xof",
        label_width_bits=256,
    )
    longest_paths = graph.longest_path_lengths()

    assert len(graph.challenge_set) == label_count_m
    assert len(set(graph.challenge_set)) == label_count_m
    assert all(0 <= challenge_node < graph.node_count for challenge_node in graph.challenge_set)
    assert all(longest_paths[challenge_node] >= graph.gamma for challenge_node in graph.challenge_set)

    for node_index, predecessors in enumerate(graph.predecessors):
        assert tuple(sorted(predecessors)) == predecessors
        assert all(predecessor < node_index for predecessor in predecessors)


def test_graph_bucket_size_and_challenge_prefix_are_stable_within_one_n_bucket() -> None:
    bucket_graphs = [
        build_pose_db_graph(
            label_count_m=label_count_m,
            hash_backend="blake3-xof",
            label_width_bits=256,
        )
        for label_count_m in range(5, 9)
    ]

    node_counts = {graph.node_count for graph in bucket_graphs}
    assert len(node_counts) == 1
    for smaller, larger in zip(bucket_graphs, bucket_graphs[1:]):
        assert tuple(larger.challenge_set[: smaller.label_count_m]) == smaller.challenge_set


def test_identical_graph_descriptors_reuse_cached_graph_objects() -> None:
    clear_pose_db_graph_cache()
    try:
        first = build_pose_db_graph(
            label_count_m=33,
            hash_backend="blake3-xof",
            label_width_bits=256,
        )
        first_info = pose_db_graph_cache_info()
        second = build_pose_db_graph(
            label_count_m=33,
            hash_backend="blake3-xof",
            label_width_bits=256,
        )
        second_info = pose_db_graph_cache_info()

        assert first is second
        assert first_info.misses == 1
        assert first_info.hits == 0
        assert second_info.misses == 1
        assert second_info.hits == 1
    finally:
        clear_pose_db_graph_cache()


def test_distinct_graph_descriptors_do_not_alias_cache_entries() -> None:
    clear_pose_db_graph_cache()
    try:
        first = build_pose_db_graph(
            label_count_m=33,
            hash_backend="blake3-xof",
            label_width_bits=256,
        )
        second = build_pose_db_graph(
            label_count_m=33,
            hash_backend="shake256",
            label_width_bits=256,
        )
        cache_info = pose_db_graph_cache_info()

        assert first is not second
        assert first.graph_descriptor_digest != second.graph_descriptor_digest
        assert cache_info.misses == 2
        assert cache_info.hits == 0
    finally:
        clear_pose_db_graph_cache()
