from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pose.common.cbor import canonical_cbor_dumps
from pose.common.hashing import sha256_hex
from pose.hashing import normalize_hash_backend


@dataclass(frozen=True)
class ChallengePolicy:
    rounds_r: int
    target_success_bound: float = 0.0
    sample_with_replacement: bool = True

    def to_cbor_object(self) -> dict[str, int | float | bool]:
        return {
            "rounds_r": self.rounds_r,
            "sample_with_replacement": self.sample_with_replacement,
            "target_success_bound": self.target_success_bound,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ChallengePolicy":
        return cls(
            rounds_r=int(payload["rounds_r"]),
            target_success_bound=float(payload.get("target_success_bound", 0.0)),
            sample_with_replacement=bool(payload.get("sample_with_replacement", True)),
        )


@dataclass(frozen=True)
class DeadlinePolicy:
    response_deadline_us: int
    session_timeout_ms: int

    def to_cbor_object(self) -> dict[str, int]:
        return {
            "response_deadline_us": self.response_deadline_us,
            "session_timeout_ms": self.session_timeout_ms,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DeadlinePolicy":
        return cls(
            response_deadline_us=int(payload["response_deadline_us"]),
            session_timeout_ms=int(payload["session_timeout_ms"]),
        )


@dataclass(frozen=True)
class CleanupPolicy:
    zeroize: bool
    verify_zeroization: bool

    def to_cbor_object(self) -> dict[str, bool]:
        return {
            "verify_zeroization": self.verify_zeroization,
            "zeroize": self.zeroize,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CleanupPolicy":
        return cls(
            zeroize=bool(payload["zeroize"]),
            verify_zeroization=bool(payload["verify_zeroization"]),
        )


@dataclass(frozen=True)
class RegionPlan:
    region_id: str
    region_type: str
    usable_bytes: int
    slot_count: int
    covered_bytes: int
    slack_bytes: int
    gpu_device: int | None = None

    def to_cbor_object(self) -> dict[str, int | str | None]:
        return {
            "covered_bytes": self.covered_bytes,
            "gpu_device": self.gpu_device,
            "region_id": self.region_id,
            "region_type": self.region_type,
            "slack_bytes": self.slack_bytes,
            "slot_count": self.slot_count,
            "usable_bytes": self.usable_bytes,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RegionPlan":
        gpu_device = payload.get("gpu_device")
        return cls(
            region_id=str(payload["region_id"]),
            region_type=str(payload["region_type"]),
            usable_bytes=int(payload["usable_bytes"]),
            slot_count=int(payload["slot_count"]),
            covered_bytes=int(payload["covered_bytes"]),
            slack_bytes=int(payload["slack_bytes"]),
            gpu_device=None if gpu_device is None else int(gpu_device),
        )


@dataclass(frozen=True)
class LeaseRecord:
    region_id: str
    region_type: str
    usable_bytes: int
    lease_handle: str
    lease_expiry: str
    cleanup_policy: CleanupPolicy
    slot_count: int = 0
    slack_bytes: int = 0
    gpu_device: int | None = None

    def to_cbor_object(self) -> dict[str, object]:
        return {
            "cleanup_policy": self.cleanup_policy.to_cbor_object(),
            "gpu_device": self.gpu_device,
            "lease_expiry": self.lease_expiry,
            "lease_handle": self.lease_handle,
            "region_id": self.region_id,
            "region_type": self.region_type,
            "slack_bytes": self.slack_bytes,
            "slot_count": self.slot_count,
            "usable_bytes": self.usable_bytes,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LeaseRecord":
        gpu_device = payload.get("gpu_device")
        return cls(
            region_id=str(payload["region_id"]),
            region_type=str(payload["region_type"]),
            usable_bytes=int(payload["usable_bytes"]),
            slot_count=int(payload["slot_count"]),
            slack_bytes=int(payload["slack_bytes"]),
            lease_handle=str(payload["lease_handle"]),
            lease_expiry=str(payload["lease_expiry"]),
            cleanup_policy=CleanupPolicy.from_dict(dict(payload["cleanup_policy"])),
            gpu_device=None if gpu_device is None else int(gpu_device),
        )


@dataclass(frozen=True)
class SessionPlan:
    session_id: str
    session_seed_hex: str
    profile_name: str
    graph_family: str
    graph_parameter_n: int
    label_count_m: int
    gamma: int
    label_width_bits: int
    hash_backend: str
    graph_descriptor_digest: str
    challenge_policy: ChallengePolicy
    deadline_policy: DeadlinePolicy
    cleanup_policy: CleanupPolicy
    regions: list[RegionPlan] = field(default_factory=list)
    adversary_model: str = "general"
    attacker_budget_bytes_assumed: int = 0
    q_bound: int = 0
    claim_notes: list[str] = field(default_factory=list)

    def to_cbor_object(self) -> dict[str, object]:
        return {
            "adversary_model": self.adversary_model,
            "attacker_budget_bytes_assumed": self.attacker_budget_bytes_assumed,
            "challenge_policy": self.challenge_policy.to_cbor_object(),
            "claim_notes": list(self.claim_notes),
            "cleanup_policy": self.cleanup_policy.to_cbor_object(),
            "deadline_policy": self.deadline_policy.to_cbor_object(),
            "gamma": self.gamma,
            "graph_descriptor_digest": self.graph_descriptor_digest,
            "graph_family": self.graph_family,
            "graph_parameter_n": self.graph_parameter_n,
            "hash_backend": self.hash_backend,
            "label_count_m": self.label_count_m,
            "label_width_bits": self.label_width_bits,
            "profile_name": self.profile_name,
            "q_bound": self.q_bound,
            "regions": [region.to_cbor_object() for region in self.regions],
            "session_id": self.session_id,
            "session_seed_hex": self.session_seed_hex,
        }

    @property
    def plan_root_hex(self) -> str:
        return sha256_hex(canonical_cbor_dumps(self.to_cbor_object()))

    @property
    def rounds_r(self) -> int:
        return self.challenge_policy.rounds_r

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SessionPlan":
        return cls(
            session_id=str(payload["session_id"]),
            session_seed_hex=str(payload["session_seed_hex"]),
            profile_name=str(payload["profile_name"]),
            graph_family=str(payload["graph_family"]),
            graph_parameter_n=int(payload["graph_parameter_n"]),
            label_count_m=int(payload["label_count_m"]),
            gamma=int(payload["gamma"]),
            label_width_bits=int(payload["label_width_bits"]),
            hash_backend=normalize_hash_backend(str(payload["hash_backend"])),
            graph_descriptor_digest=str(payload["graph_descriptor_digest"]),
            challenge_policy=ChallengePolicy.from_dict(dict(payload["challenge_policy"])),
            deadline_policy=DeadlinePolicy.from_dict(dict(payload["deadline_policy"])),
            cleanup_policy=CleanupPolicy.from_dict(dict(payload["cleanup_policy"])),
            regions=[RegionPlan.from_dict(dict(item)) for item in payload.get("regions", [])],
            adversary_model=str(payload.get("adversary_model", "general")),
            attacker_budget_bytes_assumed=int(payload.get("attacker_budget_bytes_assumed", 0)),
            q_bound=int(payload.get("q_bound", 0)),
            claim_notes=[str(item) for item in payload.get("claim_notes", [])],
        )
