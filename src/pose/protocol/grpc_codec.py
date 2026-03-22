from __future__ import annotations

from pose.protocol.messages import (
    ChallengePolicy,
    CleanupPolicy,
    DeadlinePolicy,
    LeaseRecord,
    RegionPlan,
    SessionPlan,
)
from pose.v1 import session_pb2

GRPC_PROTOCOL_VERSION = "pose-grpc/v1"


def _challenge_policy_to_proto(policy: ChallengePolicy) -> session_pb2.ChallengePolicy:
    return session_pb2.ChallengePolicy(
        rounds_r=policy.rounds_r,
        sample_with_replacement=policy.sample_with_replacement,
        target_success_bound=policy.target_success_bound,
    )


def _deadline_policy_to_proto(policy: DeadlinePolicy) -> session_pb2.DeadlinePolicy:
    return session_pb2.DeadlinePolicy(
        response_deadline_us=policy.response_deadline_us,
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
        slot_count=region.slot_count,
        covered_bytes=region.covered_bytes,
        slack_bytes=region.slack_bytes,
    )
    if region.gpu_device is not None:
        payload.gpu_device = region.gpu_device
    return payload


def session_plan_to_proto(plan: SessionPlan) -> session_pb2.SessionPlan:
    return session_pb2.SessionPlan(
        session_id=plan.session_id,
        session_seed_hex=plan.session_seed_hex,
        profile_name=plan.profile_name,
        graph_family=plan.graph_family,
        graph_parameter_n=plan.graph_parameter_n,
        label_count_m=plan.label_count_m,
        gamma=plan.gamma,
        label_width_bits=plan.label_width_bits,
        hash_backend=plan.hash_backend,
        graph_descriptor_digest=plan.graph_descriptor_digest,
        challenge_policy=_challenge_policy_to_proto(plan.challenge_policy),
        deadline_policy=_deadline_policy_to_proto(plan.deadline_policy),
        cleanup_policy=_cleanup_policy_to_proto(plan.cleanup_policy),
        regions=[region_plan_to_proto(region) for region in plan.regions],
        adversary_model=plan.adversary_model,
        attacker_budget_bytes_assumed=plan.attacker_budget_bytes_assumed,
        q_bound=plan.q_bound,
        claim_notes=list(plan.claim_notes),
    )


def challenge_policy_from_proto(policy: session_pb2.ChallengePolicy) -> ChallengePolicy:
    return ChallengePolicy(
        rounds_r=int(policy.rounds_r),
        target_success_bound=float(policy.target_success_bound),
        sample_with_replacement=bool(policy.sample_with_replacement),
    )


def deadline_policy_from_proto(policy: session_pb2.DeadlinePolicy) -> DeadlinePolicy:
    return DeadlinePolicy(
        response_deadline_us=int(policy.response_deadline_us),
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
        slot_count=int(region.slot_count),
        covered_bytes=int(region.covered_bytes),
        slack_bytes=int(region.slack_bytes),
        gpu_device=gpu_device,
    )


def session_plan_from_proto(plan: session_pb2.SessionPlan) -> SessionPlan:
    return SessionPlan(
        session_id=plan.session_id,
        session_seed_hex=plan.session_seed_hex,
        profile_name=plan.profile_name,
        graph_family=plan.graph_family,
        graph_parameter_n=int(plan.graph_parameter_n),
        label_count_m=int(plan.label_count_m),
        gamma=int(plan.gamma),
        label_width_bits=int(plan.label_width_bits),
        hash_backend=plan.hash_backend,
        graph_descriptor_digest=plan.graph_descriptor_digest,
        challenge_policy=challenge_policy_from_proto(plan.challenge_policy),
        deadline_policy=deadline_policy_from_proto(plan.deadline_policy),
        cleanup_policy=cleanup_policy_from_proto(plan.cleanup_policy),
        regions=[region_plan_from_proto(region) for region in plan.regions],
        adversary_model=plan.adversary_model or "general",
        attacker_budget_bytes_assumed=int(plan.attacker_budget_bytes_assumed),
        q_bound=int(plan.q_bound),
        claim_notes=[str(item) for item in plan.claim_notes],
    )


def lease_record_to_proto(record: LeaseRecord) -> session_pb2.LeaseRecord:
    payload = session_pb2.LeaseRecord(
        region_id=record.region_id,
        region_type=record.region_type,
        usable_bytes=record.usable_bytes,
        slot_count=record.slot_count,
        slack_bytes=record.slack_bytes,
        lease_handle=record.lease_handle,
        lease_expiry=record.lease_expiry,
        cleanup_policy=_cleanup_policy_to_proto(record.cleanup_policy),
    )
    if record.gpu_device is not None:
        payload.gpu_device = record.gpu_device
    return payload


def lease_record_from_proto(record: session_pb2.LeaseRecord) -> LeaseRecord:
    gpu_device = int(record.gpu_device) if record.region_type == "gpu" else None
    return LeaseRecord(
        region_id=record.region_id,
        region_type=record.region_type,
        usable_bytes=int(record.usable_bytes),
        slot_count=int(record.slot_count),
        slack_bytes=int(record.slack_bytes),
        lease_handle=record.lease_handle,
        lease_expiry=record.lease_expiry,
        cleanup_policy=cleanup_policy_from_proto(record.cleanup_policy),
        gpu_device=gpu_device,
    )
