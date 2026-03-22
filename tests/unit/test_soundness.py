from __future__ import annotations

import math

import pytest

from pose.common.errors import ProtocolError
from pose.verifier.soundness import (
    assess_soundness,
    derive_rounds_for_target,
    normalize_adversary_model,
    soundness_model_label,
)


def test_assess_soundness_uses_conservative_rounding_for_general_model() -> None:
    assessment = assess_soundness(
        label_count_m=8,
        rounds_r=4,
        q_bound=3,
        gamma=4,
        label_width_bits=256,
        attacker_budget_bytes_assumed=16,
        adversary_model="general",
        target_success_bound=1.0e-9,
    )

    assert assessment.adversary_model == "general"
    assert assessment.w0_bits == 251
    assert assessment.effective_label_budget_m_prime == 1
    assert assessment.ratio_m_prime_over_m == 0.125
    assert assessment.reported_success_bound == pytest.approx((1 / 8) ** 4 + 2.0 ** (-251))
    assert assessment.target_met is False


def test_assess_soundness_matches_graph_restricted_formula() -> None:
    assessment = assess_soundness(
        label_count_m=64,
        rounds_r=5,
        q_bound=7,
        gamma=32,
        label_width_bits=256,
        attacker_budget_bytes_assumed=32,
        adversary_model="graph-restricted",
        target_success_bound=1.0e-6,
    )

    assert assessment.adversary_model == "graph_restricted"
    assert assessment.w0_bits == 256
    assert assessment.effective_label_budget_m_prime == 1
    assert assessment.ratio_m_prime_over_m == 1 / 64
    assert assessment.reported_success_bound == pytest.approx((1 / 64) ** 5 + 2.0 ** (-256))
    assert assessment.target_met is True


def test_assess_soundness_rejects_non_meaningful_ratio() -> None:
    with pytest.raises(ProtocolError, match="M' / m >= 1"):
        assess_soundness(
            label_count_m=8,
            rounds_r=4,
            q_bound=3,
            gamma=4,
            label_width_bits=256,
            attacker_budget_bytes_assumed=4096,
            adversary_model="general",
            target_success_bound=1.0e-9,
        )


def test_assess_soundness_rejects_when_additive_term_exceeds_target() -> None:
    with pytest.raises(ProtocolError, match="additive 2\\^\\(-w0\\) term already exceeds"):
        assess_soundness(
            label_count_m=64,
            rounds_r=4,
            q_bound=7,
            gamma=32,
            label_width_bits=32,
            attacker_budget_bytes_assumed=1,
            adversary_model="graph_restricted",
            target_success_bound=1.0e-12,
        )


def test_derive_rounds_for_target_returns_smallest_valid_round_count() -> None:
    assessment = derive_rounds_for_target(
        label_count_m=64,
        q_bound=7,
        gamma=32,
        label_width_bits=256,
        attacker_budget_bytes_assumed=32,
        adversary_model="graph_restricted",
        target_success_bound=1.0e-9,
    )

    assert assessment.target_met is True
    assert assessment.rounds_r > 0
    assert assessment.reported_success_bound <= 1.0e-9

    previous_bound = assess_soundness(
        label_count_m=64,
        rounds_r=assessment.rounds_r - 1,
        q_bound=7,
        gamma=32,
        label_width_bits=256,
        attacker_budget_bytes_assumed=32,
        adversary_model="graph_restricted",
        target_success_bound=1.0e-9,
    ).reported_success_bound
    assert previous_bound > 1.0e-9


def test_model_helpers_normalize_and_label_supported_models() -> None:
    assert normalize_adversary_model("graph-restricted") == "graph_restricted"
    assert soundness_model_label("graph-restricted") == "random-oracle + graph-restricted + calibrated q<gamma"
    assert soundness_model_label("general") == "random-oracle + distant-attacker + calibrated q<gamma"

    with pytest.raises(ProtocolError, match="Unsupported adversary model"):
        normalize_adversary_model("unknown")
