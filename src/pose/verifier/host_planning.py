from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from pose.benchmarks.profiles import BenchmarkProfile
from pose.common.errors import ProtocolError, ResourceFailure
from pose.protocol.messages import (
    ChallengePolicy,
    CleanupPolicy,
    DeadlinePolicy,
    RegionPlan,
    SectorPlanEntry,
    SessionPlan,
)

BASE_SECTOR_ID = 4242


@dataclass(frozen=True)
class HostPlan:
    total_bytes: int
    budget_bytes: int
    usable_bytes: int
    unit_count: int
    unit_size_bytes: int


def _derive_hex(*parts: object) -> str:
    from pose.common.hashing import sha256_bytes

    seed = "|".join(str(part) for part in parts).encode("utf-8")
    output = bytearray()
    counter = 0
    while len(output) < 32:
        output.extend(sha256_bytes(seed + counter.to_bytes(4, "big")))
        counter += 1
    return bytes(output[:32]).hex()


def _physical_memory_bytes() -> int:
    try:
        pages = int(os.sysconf("SC_PHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, ValueError):
        pages = 0
        page_size = 0
    if pages > 0 and page_size > 0:
        return pages * page_size

    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                parts = line.split()
                if len(parts) >= 2:
                    return int(parts[1]) * 1024
    raise ResourceFailure("Unable to detect total host memory bytes")


def _cgroup_memory_limit_bytes() -> int | None:
    candidates = (
        Path("/sys/fs/cgroup/memory.max"),
        Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
    )
    for path in candidates:
        if not path.exists():
            continue
        raw = path.read_text(encoding="utf-8").strip()
        if not raw or raw == "max":
            continue
        try:
            value = int(raw)
        except ValueError:
            continue
        if value <= 0 or value >= 2**60:
            continue
        return value
    return None


def detect_host_memory_bytes() -> int:
    detected = _physical_memory_bytes()
    cgroup_limit = _cgroup_memory_limit_bytes()
    if cgroup_limit is not None:
        detected = min(detected, cgroup_limit)
    return detected


def _align_down(value: int, alignment: int) -> int:
    if alignment <= 0:
        raise ResourceFailure(f"Alignment must be positive: {alignment}")
    return (value // alignment) * alignment


def plan_host_lease(
    profile: BenchmarkProfile,
    *,
    unit_size_bytes: int,
    requested_unit_count: int | None = None,
    requested_host_bytes: int | None = None,
    detected_host_bytes: int | None = None,
) -> HostPlan:
    if unit_size_bytes <= 0:
        raise ResourceFailure(f"Measured unit size must be positive: {unit_size_bytes}")
    if profile.leaf_size <= 0:
        raise ResourceFailure(f"Leaf size must be positive: {profile.leaf_size}")

    total_bytes = detected_host_bytes if detected_host_bytes is not None else detect_host_memory_bytes()
    if total_bytes <= 0:
        raise ResourceFailure(f"Detected host memory must be positive: {total_bytes}")

    configured_budget = int(profile.reserve_policy.get("host_bytes", 0))
    budget_bytes = total_bytes if configured_budget <= 0 else min(total_bytes, configured_budget)
    if budget_bytes < unit_size_bytes and requested_host_bytes is None and requested_unit_count is None:
        raise ResourceFailure(
            f"Host budget {budget_bytes} is too small for one measured unit of size {unit_size_bytes}"
        )

    if requested_unit_count is not None:
        if requested_unit_count <= 0:
            raise ResourceFailure(f"Requested unit count must be positive: {requested_unit_count}")
        unit_count = requested_unit_count
    else:
        target_fraction = float(profile.host_target_fraction)
        if target_fraction <= 0:
            raise ResourceFailure(
                f"Host target fraction must be positive for host profiles: {target_fraction}"
            )
        target_bytes = _align_down(int(budget_bytes * target_fraction), profile.leaf_size)
        if target_bytes == 0 and budget_bytes >= unit_size_bytes:
            target_bytes = _align_down(unit_size_bytes, profile.leaf_size)
        if target_bytes < unit_size_bytes:
            raise ResourceFailure(
                f"Planned host bytes {target_bytes} are too small for one measured unit of size "
                f"{unit_size_bytes}"
            )
        unit_count = target_bytes // unit_size_bytes
        if unit_count <= 0:
            raise ResourceFailure(
                f"Planned host bytes {target_bytes} do not fit any measured unit of size {unit_size_bytes}"
            )

    minimum_bytes = unit_count * unit_size_bytes
    if requested_host_bytes is not None:
        if requested_host_bytes <= 0:
            raise ResourceFailure(f"Host lease size must be positive: {requested_host_bytes}")
        if requested_host_bytes % profile.leaf_size != 0:
            raise ResourceFailure(
                f"Host lease size {requested_host_bytes} must be a multiple of leaf size "
                f"{profile.leaf_size}"
            )
        if requested_host_bytes < minimum_bytes:
            raise ResourceFailure(
                f"Host lease size {requested_host_bytes} is too small for {unit_count} "
                f"unit(s) of size {unit_size_bytes}"
            )
        usable_bytes = requested_host_bytes
    else:
        usable_bytes = minimum_bytes

    return HostPlan(
        total_bytes=total_bytes,
        budget_bytes=budget_bytes,
        usable_bytes=usable_bytes,
        unit_count=unit_count,
        unit_size_bytes=unit_size_bytes,
    )


def build_host_sector_plan(session_id: str, region_id: str, unit_count: int) -> list[SectorPlanEntry]:
    if unit_count <= 0:
        raise ProtocolError(f"unit_count must be positive, got {unit_count}")
    return [
        SectorPlanEntry(
            region_id=region_id,
            unit_index=unit_index,
            prover_id_hex=_derive_hex("prover", session_id, region_id),
            sector_id=BASE_SECTOR_ID + unit_index,
            ticket_hex=_derive_hex("ticket", session_id, region_id, unit_index),
            seed_hex=_derive_hex("seed", session_id, region_id, unit_index),
        )
        for unit_index in range(unit_count)
    ]


def build_minimal_host_session_plan(
    profile: BenchmarkProfile,
    *,
    session_id: str,
    session_nonce: str,
    requested_unit_count: int | None = None,
    requested_host_bytes: int | None = None,
    detected_host_bytes: int | None = None,
    region_id: str = "host-0",
) -> tuple[SessionPlan, HostPlan]:
    host_plan = plan_host_lease(
        profile,
        unit_size_bytes=profile.leaf_size,
        requested_unit_count=requested_unit_count,
        requested_host_bytes=requested_host_bytes,
        detected_host_bytes=detected_host_bytes,
    )
    session_plan = SessionPlan(
        session_id=session_id,
        nonce=session_nonce,
        profile_name=profile.name,
        porep_unit_profile=profile.porep_unit_profile,
        challenge_leaf_size=profile.leaf_size,
        challenge_policy=ChallengePolicy(**profile.challenge_policy),
        deadline_policy=DeadlinePolicy(**profile.deadline_policy),
        cleanup_policy=CleanupPolicy(**profile.cleanup_policy),
        unit_count=host_plan.unit_count,
        regions=[RegionPlan(region_id=region_id, region_type="host", usable_bytes=host_plan.usable_bytes)],
        sector_plan=build_host_sector_plan(session_id, region_id, host_plan.unit_count),
    )
    return session_plan, host_plan


def validate_minimal_host_session_plan(plan: SessionPlan) -> None:
    if plan.porep_unit_profile != "minimal":
        raise ResourceFailure("Phase 1 host sessions only support the minimal PoRep unit profile.")
    if plan.unit_count <= 0:
        raise ProtocolError(f"Session plan unit_count must be positive, got {plan.unit_count}")
    if len(plan.regions) != 1:
        raise ProtocolError("Phase 1 host sessions require exactly one planned region.")

    region = plan.regions[0]
    if region.region_type != "host" or region.gpu_device is not None:
        raise ProtocolError("Phase 1 host sessions only support one host region.")
    if plan.challenge_leaf_size <= 0:
        raise ProtocolError(
            f"Session plan challenge leaf size must be positive, got {plan.challenge_leaf_size}"
        )
    if region.usable_bytes <= 0 or region.usable_bytes % plan.challenge_leaf_size != 0:
        raise ProtocolError(
            f"Host region usable_bytes must be a positive multiple of the challenge leaf size: "
            f"{region.usable_bytes}"
        )
    if len(plan.sector_plan) != plan.unit_count:
        raise ProtocolError(
            "Phase 1 host sessions require one explicit sector-plan entry per planned PoRep unit."
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
            raise ProtocolError("Sector-plan entries must include prover_id_hex, ticket_hex, and seed_hex.")

    expected_indices = set(range(plan.unit_count))
    if seen_indices != expected_indices:
        raise ProtocolError(
            f"Sector-plan indices must match 0..{plan.unit_count - 1}, got {sorted(seen_indices)}"
        )
