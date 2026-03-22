from __future__ import annotations

import math
from dataclasses import dataclass

from pose.common.errors import ProtocolError


@dataclass(frozen=True)
class SoundnessAssessment:
    adversary_model: str
    attacker_budget_bits_assumed: int
    label_count_m: int
    rounds_r: int
    q_bound: int
    gamma: int
    label_width_bits: int
    w0_bits: int
    effective_label_budget_m_prime: int
    ratio_m_prime_over_m: float
    additive_term: float
    reported_success_bound: float
    target_success_bound: float
    target_met: bool


def normalize_adversary_model(adversary_model: str) -> str:
    candidate = str(adversary_model).strip().lower().replace("-", "_")
    if candidate == "general":
        return "general"
    if candidate == "graph_restricted":
        return "graph_restricted"
    raise ProtocolError(
        f"Unsupported adversary model: {adversary_model!r}. Expected 'general' or 'graph_restricted'."
    )


def soundness_model_label(adversary_model: str) -> str:
    normalized_model = normalize_adversary_model(adversary_model)
    if normalized_model == "graph_restricted":
        return "random-oracle + graph-restricted + calibrated q<gamma"
    return "random-oracle + distant-attacker + calibrated q<gamma"


def _conservative_w0_bits(
    *,
    label_count_m: int,
    q_bound: int,
    label_width_bits: int,
    adversary_model: str,
) -> int:
    if label_count_m <= 0:
        raise ProtocolError(f"label_count_m must be positive, got {label_count_m}")
    if q_bound <= 0:
        raise ProtocolError(f"q_bound must be positive, got {q_bound}")
    if label_width_bits <= 0:
        raise ProtocolError(f"label_width_bits must be positive, got {label_width_bits}")
    normalized_model = normalize_adversary_model(adversary_model)
    if normalized_model == "graph_restricted":
        return label_width_bits
    return math.floor(label_width_bits - math.log2(label_count_m) - math.log2(q_bound))


def assess_soundness(
    *,
    label_count_m: int,
    rounds_r: int,
    q_bound: int,
    gamma: int,
    label_width_bits: int,
    attacker_budget_bytes_assumed: int,
    adversary_model: str,
    target_success_bound: float = 0.0,
) -> SoundnessAssessment:
    if rounds_r <= 0:
        raise ProtocolError(f"rounds_r must be positive, got {rounds_r}")
    if gamma <= 0:
        raise ProtocolError(f"gamma must be positive, got {gamma}")
    if attacker_budget_bytes_assumed < 0:
        raise ProtocolError(
            f"attacker_budget_bytes_assumed must be non-negative, got {attacker_budget_bytes_assumed}"
        )

    normalized_model = normalize_adversary_model(adversary_model)
    w0_bits = _conservative_w0_bits(
        label_count_m=label_count_m,
        q_bound=q_bound,
        label_width_bits=label_width_bits,
        adversary_model=normalized_model,
    )
    if w0_bits <= 0:
        raise ProtocolError(
            "Soundness parameters are invalid because the conservative w0 is non-positive: "
            f"w0={w0_bits}, w_bits={label_width_bits}, m={label_count_m}, q={q_bound}."
        )

    attacker_budget_bits_assumed = attacker_budget_bytes_assumed * 8
    effective_label_budget_m_prime = math.ceil(attacker_budget_bits_assumed / w0_bits)
    ratio_m_prime_over_m = effective_label_budget_m_prime / label_count_m
    if ratio_m_prime_over_m >= 1.0:
        raise ProtocolError(
            "Soundness parameters are invalid because M' / m >= 1: "
            f"M'={effective_label_budget_m_prime}, m={label_count_m}."
        )

    additive_term = 2.0 ** (-w0_bits)
    if target_success_bound > 0.0 and additive_term > target_success_bound:
        raise ProtocolError(
            "Soundness parameters are invalid because the additive 2^(-w0) term already exceeds the "
            f"target success bound: 2^(-{w0_bits})={additive_term} > {target_success_bound}."
        )

    reported_success_bound = (ratio_m_prime_over_m ** rounds_r) + additive_term
    return SoundnessAssessment(
        adversary_model=normalized_model,
        attacker_budget_bits_assumed=attacker_budget_bits_assumed,
        label_count_m=label_count_m,
        rounds_r=rounds_r,
        q_bound=q_bound,
        gamma=gamma,
        label_width_bits=label_width_bits,
        w0_bits=w0_bits,
        effective_label_budget_m_prime=effective_label_budget_m_prime,
        ratio_m_prime_over_m=ratio_m_prime_over_m,
        additive_term=additive_term,
        reported_success_bound=reported_success_bound,
        target_success_bound=target_success_bound,
        target_met=(target_success_bound <= 0.0 or reported_success_bound <= target_success_bound),
    )


def derive_rounds_for_target(
    *,
    label_count_m: int,
    q_bound: int,
    gamma: int,
    label_width_bits: int,
    attacker_budget_bytes_assumed: int,
    adversary_model: str,
    target_success_bound: float,
) -> SoundnessAssessment:
    if target_success_bound <= 0.0:
        raise ProtocolError(f"target_success_bound must be positive, got {target_success_bound}")

    base_assessment = assess_soundness(
        label_count_m=label_count_m,
        rounds_r=1,
        q_bound=q_bound,
        gamma=gamma,
        label_width_bits=label_width_bits,
        attacker_budget_bytes_assumed=attacker_budget_bytes_assumed,
        adversary_model=adversary_model,
        target_success_bound=target_success_bound,
    )
    if base_assessment.ratio_m_prime_over_m == 0.0:
        return base_assessment

    target_for_ratio_term = target_success_bound - base_assessment.additive_term
    if target_for_ratio_term <= 0.0:
        raise ProtocolError(
            "Soundness parameters are invalid because the additive term leaves no room for the multiplicative term."
        )

    derived_rounds = max(
        1,
        math.ceil(math.log(target_for_ratio_term) / math.log(base_assessment.ratio_m_prime_over_m)),
    )
    return assess_soundness(
        label_count_m=label_count_m,
        rounds_r=derived_rounds,
        q_bound=q_bound,
        gamma=gamma,
        label_width_bits=label_width_bits,
        attacker_budget_bytes_assumed=attacker_budget_bytes_assumed,
        adversary_model=adversary_model,
        target_success_bound=target_success_bound,
    )
