from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pose.common.errors import ProtocolError
from pose.common.hashing import sha256_hex
from pose.common.merkle import commit_payload
from pose.filecoin.porep_unit import SerializedPoRepUnit, canonical_cbor_dumps


def deterministic_tail_filler(
    session_nonce: str,
    region_id: str,
    session_plan_root: str,
    length: int,
) -> bytes:
    from pose.common.hashing import sha256_bytes

    if length <= 0:
        return b""
    seed = sha256_bytes(
        "|".join((session_nonce, region_id, session_plan_root)).encode("utf-8")
    )
    repeats = (length // len(seed)) + 1
    return (seed * repeats)[:length]


def build_region_payload(
    serialized_units: list[bytes],
    session_nonce: str,
    region_id: str,
    session_plan_root: str,
    tail_filler_bytes: int,
) -> bytes:
    payload = b"".join(serialized_units)
    return payload + deterministic_tail_filler(
        session_nonce=session_nonce,
        region_id=region_id,
        session_plan_root=session_plan_root,
        length=tail_filler_bytes,
    )


@dataclass(frozen=True)
class RegionManifest:
    region_id: str
    region_type: str
    usable_bytes: int
    leaf_size: int
    payload_length_bytes: int
    real_porep_bytes: int
    tail_filler_bytes: int
    unit_count: int
    unit_digests_hex: tuple[str, ...]
    payload_sha256_hex: str
    merkle_root_hex: str

    @property
    def real_porep_ratio(self) -> float:
        if self.payload_length_bytes == 0:
            return 0.0
        return self.real_porep_bytes / self.payload_length_bytes

    def to_cbor_object(self) -> dict[str, Any]:
        return {
            "leaf_size": self.leaf_size,
            "merkle_root_hex": self.merkle_root_hex,
            "payload_length_bytes": self.payload_length_bytes,
            "payload_sha256_hex": self.payload_sha256_hex,
            "real_porep_bytes": self.real_porep_bytes,
            "region_id": self.region_id,
            "region_type": self.region_type,
            "tail_filler_bytes": self.tail_filler_bytes,
            "unit_count": self.unit_count,
            "unit_digests_hex": list(self.unit_digests_hex),
            "usable_bytes": self.usable_bytes,
        }

    @property
    def manifest_root_hex(self) -> str:
        return sha256_hex(canonical_cbor_dumps(self.to_cbor_object()))

    @classmethod
    def from_cbor_object(cls, payload: dict[str, Any]) -> "RegionManifest":
        return cls(
            region_id=str(payload["region_id"]),
            region_type=str(payload["region_type"]),
            usable_bytes=int(payload["usable_bytes"]),
            leaf_size=int(payload["leaf_size"]),
            payload_length_bytes=int(payload["payload_length_bytes"]),
            real_porep_bytes=int(payload["real_porep_bytes"]),
            tail_filler_bytes=int(payload["tail_filler_bytes"]),
            unit_count=int(payload["unit_count"]),
            unit_digests_hex=tuple(str(item) for item in payload["unit_digests_hex"]),
            payload_sha256_hex=str(payload["payload_sha256_hex"]),
            merkle_root_hex=str(payload["merkle_root_hex"]),
        )


@dataclass(frozen=True)
class SessionManifest:
    session_id: str
    nonce: str
    profile_name: str
    payload_profile: str
    leaf_size: int
    deadline_policy: dict[str, int]
    challenge_policy: dict[str, int | float]
    cleanup_policy: dict[str, bool]
    region_manifests: tuple[RegionManifest, ...]

    def to_cbor_object(self) -> dict[str, Any]:
        return {
            "challenge_policy": dict(self.challenge_policy),
            "cleanup_policy": dict(self.cleanup_policy),
            "deadline_policy": dict(self.deadline_policy),
            "leaf_size": self.leaf_size,
            "nonce": self.nonce,
            "payload_profile": self.payload_profile,
            "profile_name": self.profile_name,
            "region_manifests": [manifest.to_cbor_object() for manifest in self.region_manifests],
            "region_manifest_roots": [
                manifest.manifest_root_hex for manifest in self.region_manifests
            ],
            "region_roots": [manifest.merkle_root_hex for manifest in self.region_manifests],
            "region_sizes": [manifest.usable_bytes for manifest in self.region_manifests],
            "session_id": self.session_id,
        }

    @property
    def manifest_root_hex(self) -> str:
        return sha256_hex(canonical_cbor_dumps(self.to_cbor_object()))


def build_region_manifest(
    *,
    region_id: str,
    region_type: str,
    usable_bytes: int,
    leaf_size: int,
    payload: bytes,
    merkle_root_hex: str,
    units: tuple[SerializedPoRepUnit, ...],
    tail_filler_bytes: int,
) -> RegionManifest:
    return RegionManifest(
        region_id=region_id,
        region_type=region_type,
        usable_bytes=usable_bytes,
        leaf_size=leaf_size,
        payload_length_bytes=len(payload),
        real_porep_bytes=sum(len(unit.serialized_bytes) for unit in units),
        tail_filler_bytes=tail_filler_bytes,
        unit_count=len(units),
        unit_digests_hex=tuple(sha256_hex(unit.serialized_bytes) for unit in units),
        payload_sha256_hex=sha256_hex(payload),
        merkle_root_hex=merkle_root_hex,
    )


def region_manifest_matches_payload(
    manifest: RegionManifest,
    *,
    payload: bytes,
    session_nonce: str,
    session_plan_root: str,
) -> bool:
    if manifest.unit_count != len(manifest.unit_digests_hex):
        return False
    if manifest.payload_length_bytes != len(payload):
        return False
    if manifest.payload_length_bytes != manifest.usable_bytes:
        return False
    if manifest.real_porep_bytes < 0 or manifest.tail_filler_bytes < 0:
        return False
    if manifest.real_porep_bytes + manifest.tail_filler_bytes != manifest.payload_length_bytes:
        return False
    if manifest.tail_filler_bytes > min(manifest.leaf_size, 1024 * 1024):
        return False
    if manifest.payload_sha256_hex != sha256_hex(payload):
        return False

    try:
        commitment = commit_payload(payload, manifest.leaf_size)
    except ProtocolError:
        return False
    if commitment.root_hex != manifest.merkle_root_hex:
        return False

    tail = payload[manifest.real_porep_bytes :]
    expected_tail = deterministic_tail_filler(
        session_nonce=session_nonce,
        region_id=manifest.region_id,
        session_plan_root=session_plan_root,
        length=manifest.tail_filler_bytes,
    )
    return tail == expected_tail
