from __future__ import annotations

import secrets
from dataclasses import dataclass

from pose.benchmarks.profiles import BenchmarkProfile
from pose.common.errors import ProtocolError, ResourceFailure
from pose.graphs import build_graph_descriptor
from pose.protocol.messages import ChallengePolicy, CleanupPolicy, DeadlinePolicy, RegionPlan, SessionPlan
from pose.protocol.session_ids import generate_session_id
from pose.verifier.gpu_planning import detect_gpu_memory_bytes
from pose.verifier.host_planning import detect_host_memory_bytes


@dataclass(frozen=True)
class SlotRegionLayout:
    region_id: str
    region_type: str
    total_bytes: int
    budget_bytes: int
    usable_bytes: int
    slot_count: int
    covered_bytes: int
    slack_bytes: int
    gpu_device: int | None = None

    def to_region_plan(self) -> RegionPlan:
        return RegionPlan(
            region_id=self.region_id,
            region_type=self.region_type,
            usable_bytes=self.usable_bytes,
            slot_count=self.slot_count,
            covered_bytes=self.covered_bytes,
            slack_bytes=self.slack_bytes,
            gpu_device=self.gpu_device,
        )

    def to_dict(self) -> dict[str, int | str | None]:
        return {
            "budget_bytes": self.budget_bytes,
            "covered_bytes": self.covered_bytes,
            "gpu_device": self.gpu_device,
            "region_id": self.region_id,
            "region_type": self.region_type,
            "slack_bytes": self.slack_bytes,
            "slot_count": self.slot_count,
            "total_bytes": self.total_bytes,
            "usable_bytes": self.usable_bytes,
        }


@dataclass(frozen=True)
class SlotPlanningLayout:
    profile_name: str
    w_bits: int
    hash_backend: str
    graph_family: str
    graph_parameter_n: int
    gamma: int
    label_count_m: int
    graph_descriptor_digest: str
    regions: tuple[SlotRegionLayout, ...]

    @property
    def w_bytes(self) -> int:
        return self.w_bits // 8

    @property
    def covered_bytes(self) -> int:
        return sum(region.covered_bytes for region in self.regions)

    @property
    def slack_bytes(self) -> int:
        return sum(region.slack_bytes for region in self.regions)

    @property
    def total_usable_bytes(self) -> int:
        return sum(region.usable_bytes for region in self.regions)

    @property
    def host_total_bytes(self) -> int:
        return sum(region.total_bytes for region in self.regions if region.region_type == "host")

    @property
    def host_budget_bytes(self) -> int:
        return sum(region.budget_bytes for region in self.regions if region.region_type == "host")

    @property
    def host_usable_bytes(self) -> int:
        return sum(region.usable_bytes for region in self.regions if region.region_type == "host")

    @property
    def host_covered_bytes(self) -> int:
        return sum(region.covered_bytes for region in self.regions if region.region_type == "host")

    def to_dict(self) -> dict[str, object]:
        gpu_total_bytes_by_device = {
            str(region.gpu_device): region.total_bytes
            for region in self.regions
            if region.region_type == "gpu" and region.gpu_device is not None
        }
        gpu_budget_bytes_by_device = {
            str(region.gpu_device): region.budget_bytes
            for region in self.regions
            if region.region_type == "gpu" and region.gpu_device is not None
        }
        gpu_usable_bytes_by_device = {
            str(region.gpu_device): region.usable_bytes
            for region in self.regions
            if region.region_type == "gpu" and region.gpu_device is not None
        }
        gpu_covered_bytes_by_device = {
            str(region.gpu_device): region.covered_bytes
            for region in self.regions
            if region.region_type == "gpu" and region.gpu_device is not None
        }
        return {
            "covered_bytes": self.covered_bytes,
            "gamma": self.gamma,
            "gpu_budget_bytes_by_device": gpu_budget_bytes_by_device,
            "gpu_covered_bytes_by_device": gpu_covered_bytes_by_device,
            "gpu_total_bytes_by_device": gpu_total_bytes_by_device,
            "gpu_usable_bytes_by_device": gpu_usable_bytes_by_device,
            "graph_descriptor_digest": self.graph_descriptor_digest,
            "graph_parameter_n": self.graph_parameter_n,
            "host_budget_bytes": self.host_budget_bytes,
            "host_covered_bytes": self.host_covered_bytes,
            "host_total_bytes": self.host_total_bytes,
            "host_usable_bytes": self.host_usable_bytes,
            "label_count_m": self.label_count_m,
            "regions": [region.to_dict() for region in self.regions],
            "slack_bytes": self.slack_bytes,
            "total_usable_bytes": self.total_usable_bytes,
            "w_bits": self.w_bits,
            "w_bytes": self.w_bytes,
        }


def _budget_bytes(total_bytes: int, configured_limit: int) -> int:
    if total_bytes <= 0:
        raise ResourceFailure(f"Detected memory bytes must be positive, got {total_bytes}")
    return total_bytes if configured_limit <= 0 else min(total_bytes, configured_limit)


def _usable_bytes_from_budget(*, budget_bytes: int, fraction: float, w_bytes: int, label: str) -> int:
    if fraction <= 0.0 or fraction > 1.0:
        raise ResourceFailure(f"{label} target fraction must be in (0, 1], got {fraction}")
    if budget_bytes < w_bytes:
        raise ResourceFailure(
            f"{label} budget {budget_bytes} is too small for one label slot of {w_bytes} bytes"
        )
    usable_bytes = min(budget_bytes, int(budget_bytes * fraction))
    if usable_bytes < w_bytes:
        usable_bytes = w_bytes
    return usable_bytes


def _region_layout(
    *,
    region_id: str,
    region_type: str,
    total_bytes: int,
    budget_bytes: int,
    usable_bytes: int,
    w_bytes: int,
    gpu_device: int | None = None,
) -> SlotRegionLayout:
    slot_count = usable_bytes // w_bytes
    if slot_count <= 0:
        raise ResourceFailure(
            f"Planned region {region_id} does not fit any label slots: usable_bytes={usable_bytes}, w_bytes={w_bytes}"
        )
    covered_bytes = slot_count * w_bytes
    return SlotRegionLayout(
        region_id=region_id,
        region_type=region_type,
        total_bytes=total_bytes,
        budget_bytes=budget_bytes,
        usable_bytes=usable_bytes,
        slot_count=slot_count,
        covered_bytes=covered_bytes,
        slack_bytes=usable_bytes - covered_bytes,
        gpu_device=gpu_device,
    )


def plan_slot_layout(
    profile: BenchmarkProfile,
    *,
    detected_host_bytes: int | None = None,
    detected_gpu_bytes_by_device: dict[int, tuple[int, int]] | None = None,
) -> SlotPlanningLayout:
    regions: list[SlotRegionLayout] = []
    w_bytes = profile.w_bytes

    if bool(profile.target_devices.get("host", False)):
        total_bytes = detect_host_memory_bytes() if detected_host_bytes is None else int(detected_host_bytes)
        budget_bytes = _budget_bytes(total_bytes, int(profile.reserve_policy.get("host_bytes", 0)))
        usable_bytes = _usable_bytes_from_budget(
            budget_bytes=budget_bytes,
            fraction=float(profile.host_target_fraction),
            w_bytes=w_bytes,
            label="Host",
        )
        regions.append(
            _region_layout(
                region_id="host-0",
                region_type="host",
                total_bytes=total_bytes,
                budget_bytes=budget_bytes,
                usable_bytes=usable_bytes,
                w_bytes=w_bytes,
            )
        )

    gpu_targets = profile.target_devices.get("gpus", [])
    if not isinstance(gpu_targets, list):
        raise ProtocolError("Profile target_devices.gpus must be a list.")
    detected_gpu_bytes_by_device = detected_gpu_bytes_by_device or {}
    for device in gpu_targets:
        available_bytes, total_bytes = detected_gpu_bytes_by_device.get(int(device), detect_gpu_memory_bytes(int(device)))
        budget_bytes = _budget_bytes(int(available_bytes), int(profile.reserve_policy.get("per_gpu_bytes", 0)))
        usable_bytes = _usable_bytes_from_budget(
            budget_bytes=budget_bytes,
            fraction=float(profile.per_gpu_target_fraction),
            w_bytes=w_bytes,
            label=f"GPU {device}",
        )
        regions.append(
            _region_layout(
                region_id=f"gpu-{device}",
                region_type="gpu",
                total_bytes=int(total_bytes),
                budget_bytes=budget_bytes,
                usable_bytes=usable_bytes,
                w_bytes=w_bytes,
                gpu_device=int(device),
            )
        )

    if not regions:
        raise ProtocolError("Profile did not select any host or GPU regions.")

    label_count_m = sum(region.slot_count for region in regions)
    descriptor = build_graph_descriptor(
        label_count_m=label_count_m,
        hash_backend=profile.hash_backend,
        label_width_bits=profile.w_bits,
        graph_family=profile.graph_family,
    )
    return SlotPlanningLayout(
        profile_name=profile.name,
        w_bits=profile.w_bits,
        hash_backend=profile.hash_backend,
        graph_family=profile.graph_family,
        graph_parameter_n=descriptor.graph_parameter_n,
        gamma=descriptor.gamma,
        label_count_m=label_count_m,
        graph_descriptor_digest=descriptor.digest,
        regions=tuple(regions),
    )


def build_session_plan_from_profile(
    profile: BenchmarkProfile,
    calibration_payload: dict[str, object],
    *,
    session_id: str | None = None,
    session_seed_hex: str | None = None,
) -> SessionPlan:
    if str(calibration_payload.get("status")) != "calibrated":
        notes = calibration_payload.get("notes", [])
        detail = "; ".join(str(item) for item in notes) if isinstance(notes, list) and notes else "profile calibration failed"
        raise ProtocolError(detail)
    planning_payload = calibration_payload.get("planning")
    if not isinstance(planning_payload, dict):
        raise ProtocolError("Calibration payload is missing planning details.")
    regions_payload = planning_payload.get("regions")
    if not isinstance(regions_payload, list) or not regions_payload:
        raise ProtocolError("Calibration payload does not include any planned regions.")

    rounds_r = int(calibration_payload.get("rounds_r", 0))
    if rounds_r <= 0:
        raise ProtocolError(f"Calibration payload must provide a positive rounds_r, got {rounds_r}")

    claim_notes = ["profile-driven slot-planned execution"]
    planning_claim_notes = planning_payload.get("claim_notes", [])
    if isinstance(planning_claim_notes, list):
        claim_notes.extend(str(item) for item in planning_claim_notes if str(item).strip())
    artifact_path = str(calibration_payload.get("artifact_path", "")).strip()
    if artifact_path:
        claim_notes.append(f"calibration_artifact={artifact_path}")

    return SessionPlan(
        session_id=generate_session_id() if session_id is None else session_id,
        session_seed_hex=secrets.token_hex(32) if session_seed_hex is None else session_seed_hex,
        profile_name=profile.name,
        graph_family=str(planning_payload["graph_family"]) if "graph_family" in planning_payload else profile.graph_family,
        graph_parameter_n=int(planning_payload["graph_parameter_n"]),
        label_count_m=int(planning_payload["label_count_m"]),
        gamma=int(planning_payload["gamma"]),
        label_width_bits=profile.w_bits,
        hash_backend=profile.hash_backend,
        graph_descriptor_digest=str(planning_payload["graph_descriptor_digest"]),
        challenge_policy=ChallengePolicy(
            rounds_r=rounds_r,
            target_success_bound=profile.challenge_policy.target_success_bound,
            sample_with_replacement=profile.challenge_policy.sample_with_replacement,
        ),
        deadline_policy=DeadlinePolicy(
            response_deadline_us=profile.deadline_policy.response_deadline_us,
            session_timeout_ms=profile.deadline_policy.session_timeout_ms,
        ),
        cleanup_policy=CleanupPolicy(
            zeroize=bool(profile.cleanup_policy.get("zeroize", True)),
            verify_zeroization=bool(profile.cleanup_policy.get("verify_zeroization", False)),
        ),
        regions=[
            RegionPlan.from_dict(dict(region_payload))
            for region_payload in regions_payload
        ],
        adversary_model=profile.adversary_model,
        attacker_budget_bytes_assumed=int(
            planning_payload.get(
                "effective_attacker_budget_bytes_assumed",
                profile.attacker_budget_bytes_assumed,
            )
        ),
        q_bound=int(calibration_payload["q_bound"]),
        claim_notes=claim_notes,
    )
