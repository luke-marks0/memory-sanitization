from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

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

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ChallengePolicy":
        return cls(
            epsilon=float(payload["epsilon"]),
            lambda_bits=int(payload["lambda_bits"]),
            max_challenges=int(payload["max_challenges"]),
        )


@dataclass(frozen=True)
class DeadlinePolicy:
    response_deadline_ms: int
    session_timeout_ms: int

    def to_cbor_object(self) -> dict[str, int]:
        return {
            "response_deadline_ms": self.response_deadline_ms,
            "session_timeout_ms": self.session_timeout_ms,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DeadlinePolicy":
        return cls(
            response_deadline_ms=int(payload["response_deadline_ms"]),
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
    gpu_device: int | None = None

    def to_cbor_object(self) -> dict[str, int | str | None]:
        return {
            "gpu_device": self.gpu_device,
            "region_id": self.region_id,
            "region_type": self.region_type,
            "usable_bytes": self.usable_bytes,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RegionPlan":
        gpu_device = payload.get("gpu_device")
        return cls(
            region_id=str(payload["region_id"]),
            region_type=str(payload["region_type"]),
            usable_bytes=int(payload["usable_bytes"]),
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

    def to_cbor_object(self) -> dict[str, object]:
        return {
            "cleanup_policy": self.cleanup_policy.to_cbor_object(),
            "lease_expiry": self.lease_expiry,
            "lease_handle": self.lease_handle,
            "region_id": self.region_id,
            "region_type": self.region_type,
            "usable_bytes": self.usable_bytes,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LeaseRecord":
        return cls(
            region_id=str(payload["region_id"]),
            region_type=str(payload["region_type"]),
            usable_bytes=int(payload["usable_bytes"]),
            lease_handle=str(payload["lease_handle"]),
            lease_expiry=str(payload["lease_expiry"]),
            cleanup_policy=CleanupPolicy.from_dict(dict(payload["cleanup_policy"])),
        )


@dataclass(frozen=True)
class SectorPlanEntry:
    region_id: str
    unit_index: int
    prover_id_hex: str
    sector_id: int
    ticket_hex: str
    seed_hex: str
    piece_bytes_hex: str | None = None
    porep_id_hex: str | None = None
    verify_after_seal: bool = True

    def to_cbor_object(self) -> dict[str, object]:
        return {
            "piece_bytes_hex": self.piece_bytes_hex,
            "porep_id_hex": self.porep_id_hex,
            "prover_id_hex": self.prover_id_hex,
            "region_id": self.region_id,
            "sector_id": self.sector_id,
            "seed_hex": self.seed_hex,
            "ticket_hex": self.ticket_hex,
            "unit_index": self.unit_index,
            "verify_after_seal": self.verify_after_seal,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SectorPlanEntry":
        return cls(
            region_id=str(payload["region_id"]),
            unit_index=int(payload["unit_index"]),
            prover_id_hex=str(payload["prover_id_hex"]),
            sector_id=int(payload["sector_id"]),
            ticket_hex=str(payload["ticket_hex"]),
            seed_hex=str(payload["seed_hex"]),
            piece_bytes_hex=None
            if payload.get("piece_bytes_hex") in (None, "")
            else str(payload["piece_bytes_hex"]),
            porep_id_hex=None
            if payload.get("porep_id_hex") in (None, "")
            else str(payload["porep_id_hex"]),
            verify_after_seal=bool(payload.get("verify_after_seal", True)),
        )

    def to_seal_request(self):
        from pose.filecoin.reference import SealRequest

        return SealRequest(
            piece_bytes=None if self.piece_bytes_hex is None else bytes.fromhex(self.piece_bytes_hex),
            prover_id_hex=self.prover_id_hex,
            sector_id=self.sector_id,
            ticket_hex=self.ticket_hex,
            seed_hex=self.seed_hex,
            porep_id_hex=self.porep_id_hex,
            verify_after_seal=self.verify_after_seal,
        )


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
    unit_count: int = 1
    regions: list[RegionPlan] = field(default_factory=list)
    sector_plan: list[SectorPlanEntry] = field(default_factory=list)

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
            "sector_plan": [item.to_cbor_object() for item in self.sector_plan],
            "session_id": self.session_id,
            "unit_count": self.unit_count,
        }

    @property
    def plan_root_hex(self) -> str:
        return sha256_hex(canonical_cbor_dumps(self.to_cbor_object()))

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SessionPlan":
        return cls(
            session_id=str(payload["session_id"]),
            nonce=str(payload["nonce"]),
            profile_name=str(payload["profile_name"]),
            porep_unit_profile=str(payload["porep_unit_profile"]),
            challenge_leaf_size=int(payload["challenge_leaf_size"]),
            challenge_policy=ChallengePolicy.from_dict(dict(payload["challenge_policy"])),
            deadline_policy=DeadlinePolicy.from_dict(dict(payload["deadline_policy"])),
            cleanup_policy=CleanupPolicy.from_dict(dict(payload["cleanup_policy"])),
            unit_count=int(payload.get("unit_count", 1)),
            regions=[RegionPlan.from_dict(dict(item)) for item in payload.get("regions", [])],
            sector_plan=[
                SectorPlanEntry.from_dict(dict(item))
                for item in payload.get("sector_plan", [])
            ],
        )
