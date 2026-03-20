from __future__ import annotations

import json

from pose.filecoin.reference import SealArtifact
from pose.protocol.messages import (
    ChallengePolicy,
    CleanupPolicy,
    DeadlinePolicy,
    LeaseRecord,
    RegionPlan,
    SectorPlanEntry,
    SessionPlan,
)
from pose.protocol.region_payloads import RegionManifest
from pose.v1 import session_pb2

GRPC_PROTOCOL_VERSION = "pose-grpc/v1"


def _challenge_policy_to_proto(policy: ChallengePolicy) -> session_pb2.ChallengePolicy:
    return session_pb2.ChallengePolicy(
        epsilon=policy.epsilon,
        lambda_bits=policy.lambda_bits,
        max_challenges=policy.max_challenges,
    )


def _deadline_policy_to_proto(policy: DeadlinePolicy) -> session_pb2.DeadlinePolicy:
    return session_pb2.DeadlinePolicy(
        response_deadline_ms=policy.response_deadline_ms,
        session_timeout_ms=policy.session_timeout_ms,
    )


def _cleanup_policy_to_proto(policy: CleanupPolicy) -> session_pb2.CleanupPolicy:
    return session_pb2.CleanupPolicy(
        zeroize=policy.zeroize,
        verify_zeroization=policy.verify_zeroization,
    )


def region_plan_to_proto(region: RegionPlan) -> session_pb2.RegionPlan:
    payload = session_pb2.RegionPlan(
        region_id=region.region_id,
        region_type=region.region_type,
        usable_bytes=region.usable_bytes,
    )
    if region.gpu_device is not None:
        payload.gpu_device = region.gpu_device
    return payload


def sector_plan_entry_to_proto(entry: SectorPlanEntry) -> session_pb2.SectorPlanEntry:
    return session_pb2.SectorPlanEntry(
        region_id=entry.region_id,
        unit_index=entry.unit_index,
        piece_bytes_hex=entry.piece_bytes_hex or "",
        prover_id_hex=entry.prover_id_hex,
        sector_id=entry.sector_id,
        ticket_hex=entry.ticket_hex,
        seed_hex=entry.seed_hex,
        porep_id_hex=entry.porep_id_hex or "",
        verify_after_seal=entry.verify_after_seal,
    )


def session_plan_to_proto(plan: SessionPlan) -> session_pb2.SessionPlan:
    return session_pb2.SessionPlan(
        session_id=plan.session_id,
        nonce=plan.nonce,
        profile_name=plan.profile_name,
        porep_unit_profile=plan.porep_unit_profile,
        challenge_leaf_size=plan.challenge_leaf_size,
        challenge_policy=_challenge_policy_to_proto(plan.challenge_policy),
        deadline_policy=_deadline_policy_to_proto(plan.deadline_policy),
        cleanup_policy=_cleanup_policy_to_proto(plan.cleanup_policy),
        unit_count=plan.unit_count,
        regions=[region_plan_to_proto(region) for region in plan.regions],
        sector_plan=[sector_plan_entry_to_proto(entry) for entry in plan.sector_plan],
    )


def challenge_policy_from_proto(policy: session_pb2.ChallengePolicy) -> ChallengePolicy:
    return ChallengePolicy(
        epsilon=float(policy.epsilon),
        lambda_bits=int(policy.lambda_bits),
        max_challenges=int(policy.max_challenges),
    )


def deadline_policy_from_proto(policy: session_pb2.DeadlinePolicy) -> DeadlinePolicy:
    return DeadlinePolicy(
        response_deadline_ms=int(policy.response_deadline_ms),
        session_timeout_ms=int(policy.session_timeout_ms),
    )


def cleanup_policy_from_proto(policy: session_pb2.CleanupPolicy) -> CleanupPolicy:
    return CleanupPolicy(
        zeroize=bool(policy.zeroize),
        verify_zeroization=bool(policy.verify_zeroization),
    )


def region_plan_from_proto(region: session_pb2.RegionPlan) -> RegionPlan:
    gpu_device = int(region.gpu_device) if region.region_type == "gpu" else None
    return RegionPlan(
        region_id=region.region_id,
        region_type=region.region_type,
        usable_bytes=int(region.usable_bytes),
        gpu_device=gpu_device,
    )


def sector_plan_entry_from_proto(entry: session_pb2.SectorPlanEntry) -> SectorPlanEntry:
    return SectorPlanEntry(
        region_id=entry.region_id,
        unit_index=int(entry.unit_index),
        piece_bytes_hex=entry.piece_bytes_hex or None,
        prover_id_hex=entry.prover_id_hex,
        sector_id=int(entry.sector_id),
        ticket_hex=entry.ticket_hex,
        seed_hex=entry.seed_hex,
        porep_id_hex=entry.porep_id_hex or None,
        verify_after_seal=bool(entry.verify_after_seal),
    )


def session_plan_from_proto(plan: session_pb2.SessionPlan) -> SessionPlan:
    return SessionPlan(
        session_id=plan.session_id,
        nonce=plan.nonce,
        profile_name=plan.profile_name,
        porep_unit_profile=plan.porep_unit_profile,
        challenge_leaf_size=int(plan.challenge_leaf_size),
        challenge_policy=challenge_policy_from_proto(plan.challenge_policy),
        deadline_policy=deadline_policy_from_proto(plan.deadline_policy),
        cleanup_policy=cleanup_policy_from_proto(plan.cleanup_policy),
        unit_count=int(plan.unit_count),
        regions=[region_plan_from_proto(region) for region in plan.regions],
        sector_plan=[sector_plan_entry_from_proto(entry) for entry in plan.sector_plan],
    )


def lease_record_to_proto(record: LeaseRecord) -> session_pb2.LeaseRecord:
    return session_pb2.LeaseRecord(
        region_id=record.region_id,
        region_type=record.region_type,
        usable_bytes=record.usable_bytes,
        lease_handle=record.lease_handle,
        lease_expiry=record.lease_expiry,
        cleanup_policy=_cleanup_policy_to_proto(record.cleanup_policy),
    )


def lease_record_from_proto(record: session_pb2.LeaseRecord) -> LeaseRecord:
    return LeaseRecord(
        region_id=record.region_id,
        region_type=record.region_type,
        usable_bytes=int(record.usable_bytes),
        lease_handle=record.lease_handle,
        lease_expiry=record.lease_expiry,
        cleanup_policy=cleanup_policy_from_proto(record.cleanup_policy),
    )


def artifact_to_proto(region_id: str, artifact: SealArtifact) -> session_pb2.InnerProofArtifact:
    return session_pb2.InnerProofArtifact(
        region_id=region_id,
        artifact_json=json.dumps(artifact.to_bridge_payload(), sort_keys=True),
    )


def artifact_from_proto(artifact: session_pb2.InnerProofArtifact) -> tuple[str, SealArtifact]:
    payload = json.loads(artifact.artifact_json)
    return artifact.region_id, SealArtifact(**payload)


def region_manifest_to_json(manifest: RegionManifest) -> str:
    return json.dumps(
        {
            **manifest.to_cbor_object(),
            "manifest_root_hex": manifest.manifest_root_hex,
        },
        sort_keys=True,
    )


def region_manifest_from_json(payload: str) -> tuple[RegionManifest, str]:
    decoded = json.loads(payload)
    manifest_root_hex = str(decoded["manifest_root_hex"])
    return RegionManifest.from_cbor_object(decoded), manifest_root_hex
