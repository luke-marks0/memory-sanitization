from __future__ import annotations

from dataclasses import dataclass

from pose.benchmarks.profiles import BenchmarkProfile
from pose.common.errors import ProtocolError, ResourceFailure
from pose.common.gpu_lease import get_cuda_runtime
from pose.protocol.messages import (
    ChallengePolicy,
    CleanupPolicy,
    DeadlinePolicy,
    RegionPlan,
    SessionPlan,
)
from pose.verifier.host_planning import _align_down, build_region_sector_plan


@dataclass(frozen=True)
class GpuPlan:
    device: int
    total_bytes: int
    available_bytes: int
    budget_bytes: int
    usable_bytes: int
    unit_count: int
    unit_size_bytes: int


def detect_gpu_memory_bytes(device: int) -> tuple[int, int]:
    return get_cuda_runtime().mem_get_info(device)


def plan_gpu_lease(
    profile: BenchmarkProfile,
    *,
    device: int,
    unit_size_bytes: int,
    requested_unit_count: int | None = None,
    requested_gpu_bytes: int | None = None,
    detected_gpu_bytes: tuple[int, int] | None = None,
) -> GpuPlan:
    if unit_size_bytes <= 0:
        raise ResourceFailure(f"Measured gpu unit size must be positive: {unit_size_bytes}")
    if profile.leaf_size <= 0:
        raise ResourceFailure(f"Leaf size must be positive: {profile.leaf_size}")

    available_bytes, total_bytes = (
        detected_gpu_bytes if detected_gpu_bytes is not None else detect_gpu_memory_bytes(device)
    )
    if available_bytes <= 0 or total_bytes <= 0:
        raise ResourceFailure(
            f"Detected gpu memory must be positive: available={available_bytes} total={total_bytes}"
        )

    configured_budget = int(profile.reserve_policy.get("per_gpu_bytes", 0))
    budget_bytes = available_bytes if configured_budget <= 0 else min(available_bytes, configured_budget)
    if budget_bytes < unit_size_bytes and requested_gpu_bytes is None and requested_unit_count is None:
        raise ResourceFailure(
            f"GPU budget {budget_bytes} is too small for one measured unit of size {unit_size_bytes}"
        )

    if requested_unit_count is not None:
        if requested_unit_count <= 0:
            raise ResourceFailure(f"Requested unit count must be positive: {requested_unit_count}")
        unit_count = requested_unit_count
    else:
        target_fraction = float(profile.per_gpu_target_fraction)
        if target_fraction <= 0:
            raise ResourceFailure(
                f"Per-GPU target fraction must be positive for gpu profiles: {target_fraction}"
            )
        target_bytes = _align_down(int(budget_bytes * target_fraction), profile.leaf_size)
        if target_bytes == 0 and budget_bytes >= unit_size_bytes:
            target_bytes = _align_down(unit_size_bytes, profile.leaf_size)
        if target_bytes < unit_size_bytes:
            raise ResourceFailure(
                f"Planned gpu bytes {target_bytes} are too small for one measured unit of size "
                f"{unit_size_bytes}"
            )
        unit_count = target_bytes // unit_size_bytes
        if unit_count <= 0:
            raise ResourceFailure(
                f"Planned gpu bytes {target_bytes} do not fit any measured unit of size "
                f"{unit_size_bytes}"
            )

    minimum_bytes = unit_count * unit_size_bytes
    if requested_gpu_bytes is not None:
        if requested_gpu_bytes <= 0:
            raise ResourceFailure(f"GPU lease size must be positive: {requested_gpu_bytes}")
        if requested_gpu_bytes % profile.leaf_size != 0:
            raise ResourceFailure(
                f"GPU lease size {requested_gpu_bytes} must be a multiple of leaf size "
                f"{profile.leaf_size}"
            )
        if requested_gpu_bytes < minimum_bytes:
            raise ResourceFailure(
                f"GPU lease size {requested_gpu_bytes} is too small for {unit_count} unit(s) "
                f"of size {unit_size_bytes}"
            )
        usable_bytes = requested_gpu_bytes
    else:
        usable_bytes = minimum_bytes

    if usable_bytes > available_bytes:
        raise ResourceFailure(
            f"GPU lease size {usable_bytes} exceeds available bytes on device {device}: "
            f"{available_bytes}"
        )

    return GpuPlan(
        device=device,
        total_bytes=total_bytes,
        available_bytes=available_bytes,
        budget_bytes=budget_bytes,
        usable_bytes=usable_bytes,
        unit_count=unit_count,
        unit_size_bytes=unit_size_bytes,
    )


def build_gpu_session_plan(
    profile: BenchmarkProfile,
    *,
    session_id: str,
    session_nonce: str,
    device: int,
    unit_size_bytes: int,
    requested_unit_count: int | None = None,
    requested_gpu_bytes: int | None = None,
    detected_gpu_bytes: tuple[int, int] | None = None,
    region_id: str | None = None,
) -> tuple[SessionPlan, GpuPlan]:
    gpu_plan = plan_gpu_lease(
        profile,
        device=device,
        unit_size_bytes=unit_size_bytes,
        requested_unit_count=requested_unit_count,
        requested_gpu_bytes=requested_gpu_bytes,
        detected_gpu_bytes=detected_gpu_bytes,
    )
    resolved_region_id = region_id or f"gpu-{device}"
    session_plan = SessionPlan(
        session_id=session_id,
        nonce=session_nonce,
        profile_name=profile.name,
        porep_unit_profile=profile.porep_unit_profile,
        challenge_leaf_size=profile.leaf_size,
        challenge_policy=ChallengePolicy(**profile.challenge_policy),
        deadline_policy=DeadlinePolicy(**profile.deadline_policy),
        cleanup_policy=CleanupPolicy(**profile.cleanup_policy),
        unit_count=gpu_plan.unit_count,
        regions=[
            RegionPlan(
                region_id=resolved_region_id,
                region_type="gpu",
                usable_bytes=gpu_plan.usable_bytes,
                gpu_device=device,
            )
        ],
        sector_plan=build_region_sector_plan(session_id, resolved_region_id, gpu_plan.unit_count),
    )
    return session_plan, gpu_plan


def validate_single_gpu_session_plan(plan: SessionPlan) -> None:
    if plan.unit_count <= 0:
        raise ProtocolError(f"Session plan unit_count must be positive, got {plan.unit_count}")
    if len(plan.regions) != 1:
        raise ProtocolError("Phase 2 gpu sessions require exactly one planned region.")

    region = plan.regions[0]
    if region.region_type != "gpu" or region.gpu_device is None:
        raise ProtocolError("Phase 2 gpu sessions require exactly one gpu region.")
    if plan.challenge_leaf_size <= 0:
        raise ProtocolError(
            f"Session plan challenge leaf size must be positive, got {plan.challenge_leaf_size}"
        )
    if region.usable_bytes <= 0 or region.usable_bytes % plan.challenge_leaf_size != 0:
        raise ProtocolError(
            "GPU region usable_bytes must be a positive multiple of the challenge leaf size."
        )
    if len(plan.sector_plan) != plan.unit_count:
        raise ProtocolError(
            "Phase 2 gpu sessions require one explicit sector-plan entry per planned PoRep unit."
        )

    seen_indices: set[int] = set()
    for item in plan.sector_plan:
        if item.region_id != region.region_id:
            raise ProtocolError(
                f"Sector-plan entry targets unexpected region {item.region_id!r}; "
                f"expected {region.region_id!r}"
            )
        if item.unit_index in seen_indices:
            raise ProtocolError(f"Duplicate sector-plan unit index: {item.unit_index}")
        seen_indices.add(item.unit_index)
        if item.unit_index < 0:
            raise ProtocolError(f"Sector-plan unit index must be non-negative: {item.unit_index}")
        if item.sector_id <= 0:
            raise ProtocolError(f"Sector-plan sector_id must be positive: {item.sector_id}")
        if not item.prover_id_hex or not item.ticket_hex or not item.seed_hex:
            raise ProtocolError(
                "Sector-plan entries must include prover_id_hex, ticket_hex, and seed_hex."
            )

    if seen_indices != set(range(plan.unit_count)):
        raise ProtocolError(
            f"Sector-plan indices must match 0..{plan.unit_count - 1}, got {sorted(seen_indices)}"
        )
