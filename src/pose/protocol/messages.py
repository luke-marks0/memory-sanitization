from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ChallengePolicy:
    epsilon: float
    lambda_bits: int
    max_challenges: int


@dataclass(frozen=True)
class DeadlinePolicy:
    response_deadline_ms: int
    session_timeout_ms: int


@dataclass(frozen=True)
class CleanupPolicy:
    zeroize: bool
    verify_zeroization: bool


@dataclass(frozen=True)
class RegionPlan:
    region_id: str
    region_type: str
    usable_bytes: int
    gpu_device: int | None = None


@dataclass(frozen=True)
class LeaseRecord:
    region_id: str
    region_type: str
    usable_bytes: int
    lease_handle: str
    lease_expiry: str
    cleanup_policy: CleanupPolicy


@dataclass(frozen=True)
class SessionPlan:
    session_id: str
    nonce: str
    profile_name: str
    porep_unit_profile: str
    challenge_leaf_size: int
    challenge_policy: ChallengePolicy
    deadline_policy: DeadlinePolicy
    cleanup_policy: CleanupPolicy
    regions: list[RegionPlan] = field(default_factory=list)

