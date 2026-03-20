from __future__ import annotations

from dataclasses import dataclass, field

from pose.common.hashing import sha256_hex
from pose.filecoin.porep_unit import canonical_cbor_dumps


@dataclass(frozen=True)
class ChallengePolicy:
    epsilon: float
    lambda_bits: int
    max_challenges: int

    def to_cbor_object(self) -> dict[str, int | float]:
        return {
            "epsilon": self.epsilon,
            "lambda_bits": self.lambda_bits,
            "max_challenges": self.max_challenges,
        }


@dataclass(frozen=True)
class DeadlinePolicy:
    response_deadline_ms: int
    session_timeout_ms: int

    def to_cbor_object(self) -> dict[str, int]:
        return {
            "response_deadline_ms": self.response_deadline_ms,
            "session_timeout_ms": self.session_timeout_ms,
        }


@dataclass(frozen=True)
class CleanupPolicy:
    zeroize: bool
    verify_zeroization: bool

    def to_cbor_object(self) -> dict[str, bool]:
        return {
            "verify_zeroization": self.verify_zeroization,
            "zeroize": self.zeroize,
        }


@dataclass(frozen=True)
class RegionPlan:
    region_id: str
    region_type: str
    usable_bytes: int
    gpu_device: int | None = None

    def to_cbor_object(self) -> dict[str, int | str | None]:
        return {
            "gpu_device": self.gpu_device,
            "region_id": self.region_id,
            "region_type": self.region_type,
            "usable_bytes": self.usable_bytes,
        }


@dataclass(frozen=True)
class LeaseRecord:
    region_id: str
    region_type: str
    usable_bytes: int
    lease_handle: str
    lease_expiry: str
    cleanup_policy: CleanupPolicy

    def to_cbor_object(self) -> dict[str, object]:
        return {
            "cleanup_policy": self.cleanup_policy.to_cbor_object(),
            "lease_expiry": self.lease_expiry,
            "lease_handle": self.lease_handle,
            "region_id": self.region_id,
            "region_type": self.region_type,
            "usable_bytes": self.usable_bytes,
        }


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

    def to_cbor_object(self) -> dict[str, object]:
        return {
            "challenge_leaf_size": self.challenge_leaf_size,
            "challenge_policy": self.challenge_policy.to_cbor_object(),
            "cleanup_policy": self.cleanup_policy.to_cbor_object(),
            "deadline_policy": self.deadline_policy.to_cbor_object(),
            "nonce": self.nonce,
            "porep_unit_profile": self.porep_unit_profile,
            "profile_name": self.profile_name,
            "regions": [region.to_cbor_object() for region in self.regions],
            "session_id": self.session_id,
        }

    @property
    def plan_root_hex(self) -> str:
        return sha256_hex(canonical_cbor_dumps(self.to_cbor_object()))
