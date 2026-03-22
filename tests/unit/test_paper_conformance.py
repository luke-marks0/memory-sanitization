from __future__ import annotations

import math
from dataclasses import replace

import pytest

from pose.common.errors import ProtocolError
from pose.graphs import GRAPH_FAMILY, build_pose_db_graph, expected_graph_parameter_n, gamma_for_graph_parameter_n
from pose.protocol.messages import ChallengePolicy, CleanupPolicy, DeadlinePolicy, RegionPlan, SessionPlan
from pose.verifier.service import _sample_challenge_indices
from pose.verifier.soundness import assess_soundness, derive_rounds_for_target


def _session_plan(session_seed_hex: str = "11" * 32) -> SessionPlan:
    graph = build_pose_db_graph(
        label_count_m=8,
        graph_parameter_n=2,
        gamma=4,
        hash_backend="blake3-xof",
        label_width_bits=256,
    )
    return SessionPlan(
        session_id="paper-conformance",
        session_seed_hex=session_seed_hex,
        profile_name="dev-small",
        graph_family=graph.descriptor.graph_family,
        graph_parameter_n=graph.graph_parameter_n,
        label_count_m=graph.label_count_m,
        gamma=graph.gamma,
        label_width_bits=graph.label_width_bits,
        hash_backend=graph.hash_backend,
        graph_descriptor_digest=graph.graph_descriptor_digest,
        challenge_policy=ChallengePolicy(
            rounds_r=12,
            target_success_bound=1.0e-9,
            sample_with_replacement=True,
        ),
        deadline_policy=DeadlinePolicy(response_deadline_us=2_500, session_timeout_ms=60_000),
        cleanup_policy=CleanupPolicy(zeroize=True, verify_zeroization=False),
        regions=[
            RegionPlan(
                region_id="host-0",
                region_type="host",
                usable_bytes=256,
                slot_count=8,
                covered_bytes=256,
                slack_bytes=0,
            )
        ],
        adversary_model="general",
        attacker_budget_bytes_assumed=16,
        q_bound=3,
        claim_notes=["paper-conformance"],
    )


@pytest.mark.parametrize(
    ("label_count_m", "expected_n", "expected_gamma"),
    [
        (1, 0, 1),
        (2, 0, 1),
        (3, 1, 2),
        (4, 1, 2),
        (5, 2, 4),
        (8, 2, 4),
        (9, 3, 8),
        (16, 3, 8),
        (17, 4, 16),
    ],
)
def test_paper_conformance_graph_family_uses_smallest_valid_n_for_arbitrary_m(
    label_count_m: int,
    expected_n: int,
    expected_gamma: int,
) -> None:
    graph = build_pose_db_graph(
        label_count_m=label_count_m,
        hash_backend="blake3-xof",
        label_width_bits=256,
    )

    assert graph.descriptor.graph_family == GRAPH_FAMILY
    assert graph.graph_parameter_n == expected_n
    assert graph.gamma == expected_gamma
    assert expected_graph_parameter_n(label_count_m) == expected_n
    assert gamma_for_graph_parameter_n(expected_n) == expected_gamma
    assert label_count_m <= 2 ** (expected_n + 1)
    if expected_n > 0:
        assert label_count_m > 2 ** expected_n


def test_paper_conformance_small_graph_has_canonical_node_and_challenge_ordering() -> None:
    graph = build_pose_db_graph(
        label_count_m=3,
        graph_parameter_n=1,
        gamma=2,
        hash_backend="blake3-xof",
        label_width_bits=256,
    )

    assert graph.challenge_set == (7, 10, 18)
    assert graph.predecessors == (
        (),
        (0,),
        (1,),
        (0,),
        (2,),
        (3, 4),
        (3, 4),
        (5,),
        (6,),
        (7, 8),
        (9,),
        (),
        (11,),
        (12,),
        (11,),
        (13,),
        (14, 15),
        (14, 15),
        (16,),
        (17,),
        (18, 19),
        (20,),
    )
    assert graph.longest_path_lengths() == (
        0,
        1,
        2,
        1,
        3,
        4,
        4,
        5,
        5,
        6,
        7,
        0,
        1,
        2,
        1,
        3,
        4,
        4,
        5,
        5,
        6,
        7,
    )


def test_paper_conformance_challenge_schedule_is_deterministic_and_sampled_with_replacement() -> None:
    baseline = _session_plan("11" * 32)
    changed = _session_plan("22" * 32)

    baseline_schedule = _sample_challenge_indices(baseline)
    repeated_schedule = _sample_challenge_indices(baseline)
    changed_schedule = _sample_challenge_indices(changed)

    assert baseline_schedule == repeated_schedule
    assert baseline_schedule != changed_schedule
    assert len(baseline_schedule) == baseline.rounds_r
    assert all(0 <= item < baseline.label_count_m for item in baseline_schedule)
    assert len(set(baseline_schedule)) < len(baseline_schedule)


def test_paper_conformance_challenge_schedule_requires_sampling_with_replacement() -> None:
    without_replacement = replace(
        _session_plan(),
        challenge_policy=replace(_session_plan().challenge_policy, sample_with_replacement=False),
    )

    with pytest.raises(ProtocolError, match="sampling with replacement"):
        _sample_challenge_indices(without_replacement)


def test_paper_conformance_general_model_formula_matches_hand_computation() -> None:
    label_count_m = 8
    rounds_r = 4
    q_bound = 3
    label_width_bits = 256
    attacker_budget_bytes_assumed = 16

    assessment = assess_soundness(
        label_count_m=label_count_m,
        rounds_r=rounds_r,
        q_bound=q_bound,
        gamma=4,
        label_width_bits=label_width_bits,
        attacker_budget_bytes_assumed=attacker_budget_bytes_assumed,
        adversary_model="general",
        target_success_bound=1.0e-9,
    )

    expected_w0_bits = math.floor(label_width_bits - math.log2(label_count_m) - math.log2(q_bound))
    expected_m_prime = math.ceil((attacker_budget_bytes_assumed * 8) / expected_w0_bits)
    expected_ratio = expected_m_prime / label_count_m
    expected_bound = (expected_ratio ** rounds_r) + (2.0 ** (-expected_w0_bits))

    assert assessment.w0_bits == expected_w0_bits
    assert assessment.effective_label_budget_m_prime == expected_m_prime
    assert assessment.ratio_m_prime_over_m == expected_ratio
    assert assessment.reported_success_bound == pytest.approx(expected_bound)


def test_paper_conformance_graph_restricted_round_derivation_matches_hand_computation() -> None:
    label_count_m = 64
    q_bound = 7
    label_width_bits = 256
    attacker_budget_bytes_assumed = 32
    target_success_bound = 1.0e-9

    assessment = derive_rounds_for_target(
        label_count_m=label_count_m,
        q_bound=q_bound,
        gamma=32,
        label_width_bits=label_width_bits,
        attacker_budget_bytes_assumed=attacker_budget_bytes_assumed,
        adversary_model="graph-restricted",
        target_success_bound=target_success_bound,
    )

    expected_w0_bits = label_width_bits
    expected_m_prime = math.ceil((attacker_budget_bytes_assumed * 8) / expected_w0_bits)
    expected_ratio = expected_m_prime / label_count_m
    additive_term = 2.0 ** (-expected_w0_bits)
    expected_rounds = math.ceil(math.log(target_success_bound - additive_term) / math.log(expected_ratio))
    expected_bound = (expected_ratio ** expected_rounds) + additive_term

    assert assessment.w0_bits == expected_w0_bits
    assert assessment.effective_label_budget_m_prime == expected_m_prime
    assert assessment.rounds_r == expected_rounds
    assert assessment.reported_success_bound == pytest.approx(expected_bound)
    assert assessment.target_met is True
