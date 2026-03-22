from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory
from time import perf_counter_ns

from pose.benchmarks.profiles import BenchmarkProfile, load_profile
from pose.common.env import capture_environment
from pose.common.errors import ProtocolError, ResourceFailure
from pose.common.gpu_lease import get_cuda_runtime
from pose.common.host_lease import create_host_lease, release_host_lease
from pose.common.sandbox import (
    sandbox_claim_notes,
)
from pose.graphs import build_graph_descriptor
from pose.hashing import internal_label_bytes
from pose.protocol.codec import dump_json_file
from pose.protocol.messages import ChallengePolicy, CleanupPolicy, DeadlinePolicy, RegionPlan, SessionPlan
from pose.verifier.grpc_client import (
    FastPhaseClient,
    cleanup_session,
    finalize_session,
    lease_regions,
    materialize_labels,
    plan_session,
    prepare_fast_phase,
    seed_session,
    start_ephemeral_prover_server,
)
from pose.verifier.host_planning import detect_host_memory_bytes
from pose.verifier.slot_planning import plan_slot_layout
from pose.verifier.soundness import (
    SoundnessAssessment,
    assess_soundness,
    derive_rounds_for_target,
    soundness_model_label,
)


@dataclass(frozen=True)
class _UntargetedLocalTierAdjustment:
    added_bytes: int
    note: str


def calibration_root() -> Path:
    root = Path(__file__).resolve().parents[3] / ".pose" / "calibration"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] + ((ordered[upper] - ordered[lower]) * weight)


def _series_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "mean": sum(values) / len(values),
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
        "max": max(values),
    }


def _measure_hash_evaluations_per_second(
    *,
    hash_backend: str,
    output_bytes: int,
    graph_descriptor_digest: str,
    measurement_rounds: int,
    hashes_per_round: int,
) -> float:
    predecessor_labels = [bytes([index + 1]) * output_bytes for index in range(2)]
    session_seed = bytes.fromhex("11" * 32)
    best_rate = 0.0
    for round_index in range(max(1, measurement_rounds)):
        started = perf_counter_ns()
        for node_index in range(max(1, hashes_per_round)):
            internal_label_bytes(
                session_seed=session_seed,
                graph_descriptor_digest=graph_descriptor_digest,
                node_index=node_index + (round_index * hashes_per_round),
                predecessor_labels=predecessor_labels,
                hash_backend=hash_backend,
                output_bytes=output_bytes,
            )
        elapsed_ns = max(1, perf_counter_ns() - started)
        best_rate = max(best_rate, max(1, hashes_per_round) / (elapsed_ns / 1_000_000_000))
    return best_rate


def _measure_host_lookup_latency_us(*, w_bytes: int, sample_count: int) -> dict[str, float]:
    sample_slots = max(1, min(max(sample_count, 128), 4096))
    payload = bytearray(sample_slots * w_bytes)
    for index in range(len(payload)):
        payload[index] = index % 251
    view = memoryview(payload)
    rng = Random(0xC0DEC0DE)
    latencies_us: list[float] = []
    for _ in range(max(1, sample_count)):
        slot_index = rng.randrange(sample_slots)
        offset = slot_index * w_bytes
        started = perf_counter_ns()
        bytes(view[offset : offset + w_bytes])
        latencies_us.append((perf_counter_ns() - started) / 1000.0)
    return _series_summary(latencies_us)


def _transport_measurement_plan(profile: BenchmarkProfile) -> SessionPlan:
    descriptor = build_graph_descriptor(
        label_count_m=8,
        hash_backend=profile.hash_backend,
        label_width_bits=profile.w_bits,
        graph_family=profile.graph_family,
    )
    usable_bytes = 8 * profile.w_bytes
    return SessionPlan(
        session_id="calibration-transport-session",
        session_seed_hex="44" * 32,
        profile_name=profile.name,
        graph_family=profile.graph_family,
        graph_parameter_n=descriptor.graph_parameter_n,
        label_count_m=8,
        gamma=descriptor.gamma,
        label_width_bits=profile.w_bits,
        hash_backend=profile.hash_backend,
        graph_descriptor_digest=descriptor.digest,
        challenge_policy=ChallengePolicy(rounds_r=8, target_success_bound=0.0),
        deadline_policy=DeadlinePolicy(
            response_deadline_us=profile.deadline_policy.response_deadline_us,
            session_timeout_ms=profile.deadline_policy.session_timeout_ms,
        ),
        cleanup_policy=CleanupPolicy(zeroize=True, verify_zeroization=False),
        regions=[
            RegionPlan(
                region_id="host-0",
                region_type="host",
                usable_bytes=usable_bytes,
                slot_count=8,
                covered_bytes=usable_bytes,
                slack_bytes=0,
            )
        ],
        adversary_model=profile.adversary_model,
        attacker_budget_bytes_assumed=profile.attacker_budget_bytes_assumed,
        q_bound=1,
        claim_notes=["transport-calibration"],
    )


def _measure_grpc_fast_phase_transport_us(
    profile: BenchmarkProfile,
    *,
    sample_count: int,
) -> dict[str, dict[str, float]]:
    measurement_plan = _transport_measurement_plan(profile)
    round_trip_values: list[float] = []
    prover_lookup_values: list[float] = []
    transport_overhead_values: list[float] = []

    socket_path = ""
    process = None
    lease = None
    remote_cleaned = False
    try:
        with TemporaryDirectory(prefix="pose-db-calibration-transport-") as temp_dir:
            socket_path = str(Path(temp_dir) / "prover.sock")
            process = start_ephemeral_prover_server(
                socket_path=socket_path,
                prover_sandbox=profile.prover_sandbox,
            )
            region = measurement_plan.regions[0]
            lease = create_host_lease(
                session_id=measurement_plan.session_id,
                region_id=region.region_id,
                usable_bytes=region.usable_bytes,
                cleanup_policy=measurement_plan.cleanup_policy,
                lease_duration_ms=measurement_plan.deadline_policy.session_timeout_ms,
            )
            plan_session(socket_path, measurement_plan)
            lease_regions(
                socket_path,
                measurement_plan.session_id,
                [
                    lease.record.__class__(
                        region_id=region.region_id,
                        region_type=region.region_type,
                        usable_bytes=region.usable_bytes,
                        slot_count=region.slot_count,
                        slack_bytes=region.slack_bytes,
                        lease_handle=lease.record.lease_handle,
                        lease_expiry=lease.record.lease_expiry,
                        cleanup_policy=measurement_plan.cleanup_policy,
                        gpu_device=region.gpu_device,
                    )
                ],
            )
            seed_session(socket_path, measurement_plan.session_id)
            materialize_labels(socket_path, measurement_plan.session_id)
            prepare_fast_phase(socket_path, measurement_plan.session_id)

            rounds = max(8, min(max(sample_count, 8), 128))
            with FastPhaseClient(socket_path) as fast_phase_client:
                for round_index in range(rounds):
                    response = fast_phase_client.run_round(
                        session_id=measurement_plan.session_id,
                        round_index=round_index,
                        challenge_index=round_index % region.slot_count,
                    )
                    round_trip_us = float(response["round_trip_us"])
                    prover_lookup_us = float(response["prover_lookup_round_trip_us"])
                    round_trip_values.append(round_trip_us)
                    prover_lookup_values.append(prover_lookup_us)
                    transport_overhead_values.append(max(0.0, round_trip_us - prover_lookup_us))

            finalize_session(
                socket_path,
                session_id=measurement_plan.session_id,
                verdict="SUCCESS",
                success=True,
                retain_session=False,
            )
            cleanup_session(socket_path, measurement_plan.session_id)
            remote_cleaned = True
    finally:
        if lease is not None:
            release_host_lease(lease, zeroize=not remote_cleaned, verify_zeroization=False)
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except Exception:
                    process.kill()
                    process.wait(timeout=5)

    return {
        "fast_phase_round_trip_us": _series_summary(round_trip_values),
        "fast_phase_prover_lookup_round_trip_us": _series_summary(prover_lookup_values),
        "fast_phase_transport_overhead_us": _series_summary(transport_overhead_values),
    }


def _derive_q_bound(
    *,
    deadline_us: int,
    lookup_latency_p95_us: float,
    transport_overhead_us: int,
    serialization_overhead_us: int,
    fastest_hash_evaluations_per_second: float,
    safety_margin_fraction: float,
) -> tuple[int, float]:
    available_round_us = (
        float(deadline_us)
        - float(lookup_latency_p95_us)
        - float(transport_overhead_us)
        - float(serialization_overhead_us)
    )
    if available_round_us <= 0:
        raise ProtocolError(
            "Calibration leaves no time for local recomputation after lookup and transport overhead: "
            f"deadline_us={deadline_us}, available_round_us={available_round_us:.3f}."
        )
    hashes_per_microsecond = fastest_hash_evaluations_per_second / 1_000_000.0
    q_bound = max(1, int((hashes_per_microsecond * available_round_us * (1.0 + safety_margin_fraction)) + 0.999999999))
    return q_bound, available_round_us


def _soundness_for_profile(
    *,
    profile: BenchmarkProfile,
    layout,
    q_bound: int,
    attacker_budget_bytes_assumed: int,
) -> SoundnessAssessment:
    rounds_r = int(profile.challenge_policy.rounds_r)
    if rounds_r > 0:
        return assess_soundness(
            label_count_m=layout.label_count_m,
            rounds_r=rounds_r,
            q_bound=q_bound,
            gamma=layout.gamma,
            label_width_bits=profile.w_bits,
            attacker_budget_bytes_assumed=attacker_budget_bytes_assumed,
            adversary_model=profile.adversary_model,
            target_success_bound=profile.challenge_policy.target_success_bound,
        )
    return derive_rounds_for_target(
        label_count_m=layout.label_count_m,
        q_bound=q_bound,
        gamma=layout.gamma,
        label_width_bits=profile.w_bits,
        attacker_budget_bytes_assumed=attacker_budget_bytes_assumed,
        adversary_model=profile.adversary_model,
        target_success_bound=profile.challenge_policy.target_success_bound,
    )


def _untargeted_local_tier_adjustments(profile: BenchmarkProfile) -> list[_UntargetedLocalTierAdjustment]:
    targeted_gpu_devices = {int(device) for device in profile.target_devices.get("gpus", [])}
    adjustments: list[_UntargetedLocalTierAdjustment] = []
    if not bool(profile.target_devices.get("host", False)):
        try:
            host_total_bytes = detect_host_memory_bytes()
        except (OSError, ResourceFailure):
            host_total_bytes = 0
        if host_total_bytes > 0:
            adjustments.append(
                _UntargetedLocalTierAdjustment(
                    added_bytes=host_total_bytes,
                    note=(
                        "untargeted_local_host_tier="
                        f"included_in_attacker_budget:true,"
                        f"added_bytes:{host_total_bytes},"
                        f"total_bytes:{host_total_bytes}"
                    ),
                )
            )
    try:
        runtime = get_cuda_runtime()
        for device in range(runtime.device_count()):
            available_bytes, total_bytes = runtime.mem_get_info(device)
            if device in targeted_gpu_devices or total_bytes <= 0:
                continue
            added_bytes = max(0, int(available_bytes))
            adjustments.append(
                _UntargetedLocalTierAdjustment(
                    added_bytes=added_bytes,
                    note=(
                        "untargeted_local_gpu_tier="
                        f"device:{device},"
                        "included_in_attacker_budget:true,"
                        f"added_bytes:{added_bytes},"
                        f"available_bytes:{available_bytes},"
                        f"total_bytes:{total_bytes}"
                    ),
                )
            )
    except ResourceFailure:
        return adjustments
    return adjustments


def _sandbox_adjusted_budget(profile: BenchmarkProfile) -> tuple[int, list[str], bool]:
    sandbox_policy = profile.prover_sandbox
    if sandbox_policy.mode == "none":
        return int(profile.attacker_budget_bytes_assumed), [], False
    if sandbox_policy.mode != "process_budget_dev":
        raise ProtocolError(f"Unsupported prover sandbox mode: {sandbox_policy.mode!r}")
    if not bool(profile.target_devices.get("host", False)) or profile.target_devices.get("gpus"):
        raise ProtocolError(
            "process_budget_dev prover sandbox mode is currently supported only for host-only profiles"
        )
    if sandbox_policy.process_memory_max_bytes <= 0:
        raise ProtocolError(
            "process_budget_dev prover sandbox mode requires a positive process_memory_max_bytes"
        )
    base_budget_bytes = int(profile.attacker_budget_bytes_assumed)
    if base_budget_bytes <= 0:
        raise ProtocolError(
            "process_budget_dev prover sandbox mode requires a positive attacker_budget_bytes_assumed"
        )
    notes = sandbox_claim_notes(sandbox_policy)
    notes.extend(
        (
            f"attacker_budget_base_bytes_assumed={base_budget_bytes}",
            "attacker_budget_auto_included_bytes=0",
            f"effective_attacker_budget_bytes_assumed={base_budget_bytes}",
        )
    )
    return base_budget_bytes, notes, sandbox_policy.require_no_visible_gpus


def _effective_attacker_budget(profile: BenchmarkProfile) -> tuple[int, list[str], int]:
    base_bytes, sandbox_notes, hide_gpus = _sandbox_adjusted_budget(profile)
    if sandbox_notes:
        return base_bytes, sandbox_notes, 0
    adjustments = [] if hide_gpus else _untargeted_local_tier_adjustments(profile)
    added_bytes = sum(item.added_bytes for item in adjustments)
    effective_bytes = base_bytes + added_bytes
    notes = [item.note for item in adjustments]
    if adjustments:
        notes.extend(
            (
                f"attacker_budget_base_bytes_assumed={base_bytes}",
                f"attacker_budget_auto_included_bytes={added_bytes}",
                f"effective_attacker_budget_bytes_assumed={effective_bytes}",
            )
        )
    return effective_bytes, notes, added_bytes


def _artifact_path(profile_name: str) -> Path:
    root = calibration_root()
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return root / f"{profile_name}-{timestamp}.json"


def prepare_calibration(profile_identifier: str) -> dict[str, object]:
    profile = load_profile(profile_identifier)
    return _calibrate_profile_payload(profile, persist_artifact=True)


def _calibrate_profile_payload(
    profile: BenchmarkProfile,
    *,
    persist_artifact: bool,
) -> dict[str, object]:
    notes: list[str] = []
    payload: dict[str, object] = {
        "status": "calibration-invalid",
        "profile": profile.to_dict(),
        "notes": notes,
        "environment": capture_environment(),
    }
    try:
        layout = plan_slot_layout(profile)
        lookup_latency = _measure_host_lookup_latency_us(
            w_bytes=layout.w_bytes,
            sample_count=profile.calibration_policy.lookup_samples,
        )
        if profile.transport_mode == "grpc":
            transport_measurement = _measure_grpc_fast_phase_transport_us(
                profile,
                sample_count=profile.calibration_policy.lookup_samples,
            )
            measured_transport_overhead_p95_us = transport_measurement["fast_phase_transport_overhead_us"]["p95"]
        else:
            transport_measurement = {
                "fast_phase_round_trip_us": _series_summary([]),
                "fast_phase_prover_lookup_round_trip_us": _series_summary([]),
                "fast_phase_transport_overhead_us": _series_summary([]),
            }
            measured_transport_overhead_p95_us = 0.0
        fastest_hash_evaluations_per_second = _measure_hash_evaluations_per_second(
            hash_backend=profile.hash_backend,
            output_bytes=profile.w_bytes,
            graph_descriptor_digest=layout.graph_descriptor_digest,
            measurement_rounds=profile.calibration_policy.hash_measurement_rounds,
            hashes_per_round=profile.calibration_policy.hashes_per_round,
        )
        configured_transport_reserve_us = float(profile.calibration_policy.transport_overhead_us)
        configured_serialization_reserve_us = float(profile.calibration_policy.serialization_overhead_us)
        q_bound, available_round_us = _derive_q_bound(
            deadline_us=profile.deadline_policy.response_deadline_us,
            lookup_latency_p95_us=lookup_latency["p95"],
            transport_overhead_us=(
                measured_transport_overhead_p95_us
                + configured_transport_reserve_us
                + configured_serialization_reserve_us
            ),
            serialization_overhead_us=0,
            fastest_hash_evaluations_per_second=fastest_hash_evaluations_per_second,
            safety_margin_fraction=profile.calibration_policy.safety_margin_fraction,
        )
        effective_attacker_budget_bytes, accounting_notes, auto_included_bytes = _effective_attacker_budget(profile)
        notes.extend(accounting_notes)
        payload.update(
            {
                "planning": {
                    **layout.to_dict(),
                    "graph_family": profile.graph_family,
                    "hash_backend": profile.hash_backend,
                    "profile_name": profile.name,
                    "base_attacker_budget_bytes_assumed": int(profile.attacker_budget_bytes_assumed),
                    "effective_attacker_budget_bytes_assumed": effective_attacker_budget_bytes,
                    "untargeted_local_tier_bytes_auto_included": auto_included_bytes,
                    "claim_notes": accounting_notes,
                },
                "measurements": {
                    "available_round_us": available_round_us,
                    "fastest_hash_evaluations_per_second": fastest_hash_evaluations_per_second,
                    "fast_phase_prover_lookup_round_trip_us": transport_measurement["fast_phase_prover_lookup_round_trip_us"],
                    "fast_phase_round_trip_us": transport_measurement["fast_phase_round_trip_us"],
                    "fast_phase_transport_overhead_us": transport_measurement["fast_phase_transport_overhead_us"],
                    "resident_lookup_latency_us": lookup_latency,
                    "configured_serialization_reserve_us": configured_serialization_reserve_us,
                    "configured_transport_reserve_us": configured_transport_reserve_us,
                    "effective_transport_overhead_p95_us": (
                        measured_transport_overhead_p95_us
                        + configured_transport_reserve_us
                        + configured_serialization_reserve_us
                    ),
                    "safety_margin_fraction": profile.calibration_policy.safety_margin_fraction,
                },
                "q_bound": q_bound,
                "gamma": layout.gamma,
                "q_over_gamma": q_bound / float(layout.gamma),
            }
        )
        if q_bound >= layout.gamma:
            raise ProtocolError(
                f"Derived q_bound must be strictly less than gamma, got q={q_bound} and gamma={layout.gamma}."
            )
        soundness = _soundness_for_profile(
            profile=profile,
            layout=layout,
            q_bound=q_bound,
            attacker_budget_bytes_assumed=effective_attacker_budget_bytes,
        )
        if profile.challenge_policy.target_success_bound > 0.0 and profile.challenge_policy.rounds_r > 0 and not soundness.target_met:
            notes.append(
                "Fixed rounds_r does not meet target_success_bound; reported_success_bound is the conservative theorem bound."
            )
        payload.update(
            {
                "status": "calibrated",
                "rounds_r": soundness.rounds_r,
                "soundness": {
                    "adversary_model": soundness.adversary_model,
                    "attacker_budget_bits_assumed": soundness.attacker_budget_bits_assumed,
                    "effective_label_budget_m_prime": soundness.effective_label_budget_m_prime,
                    "ratio_m_prime_over_m": soundness.ratio_m_prime_over_m,
                    "w0_bits": soundness.w0_bits,
                    "additive_term": soundness.additive_term,
                    "reported_success_bound": soundness.reported_success_bound,
                    "target_success_bound": soundness.target_success_bound,
                    "target_met": soundness.target_met,
                    "soundness_model": soundness_model_label(soundness.adversary_model),
                },
            }
        )
    except (ProtocolError, ResourceFailure, OSError) as error:
        notes.append(str(error))
    if persist_artifact:
        artifact_path = _artifact_path(profile.name)
        payload["artifact_path"] = str(artifact_path)
        dump_json_file(artifact_path, payload)
    return payload


def calibrate_profile(
    profile: BenchmarkProfile,
    *,
    persist_artifact: bool = True,
) -> dict[str, object]:
    return _calibrate_profile_payload(profile, persist_artifact=persist_artifact)
