from __future__ import annotations

import subprocess
from dataclasses import replace
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory
from time import perf_counter

from pose.benchmarks.calibration import calibrate_profile
from pose.benchmarks.profiles import BenchmarkProfile
from pose.common.errors import ProtocolError, ResourceFailure
from pose.common.gpu_lease import GpuLease, create_gpu_lease, release_gpu_lease
from pose.common.hashing import sha256_hex
from pose.common.host_lease import HostLease, create_host_lease, release_host_lease
from pose.common.sandbox import ProverSandboxPolicy
from pose.graphs import PoseDbGraph, build_pose_db_graph, compute_challenge_labels
from pose.protocol.codec import load_json_file
from pose.protocol.messages import LeaseRecord, SessionPlan
from pose.protocol.result_schema import SessionResult, bootstrap_result
from pose.verifier.grpc_client import (
    cleanup_session,
    discover,
    finalize_session,
    lease_regions,
    materialize_labels,
    plan_session,
    prepare_fast_phase,
    FastPhaseClient,
    seed_session,
    start_ephemeral_prover_server,
)
from pose.verifier.host_planning import detect_host_memory_bytes
from pose.verifier.rechallenge import run_host_rechallenge
from pose.verifier.soundness import assess_soundness, soundness_model_label
from pose.verifier.slot_planning import build_session_plan_from_profile
from pose.verifier.session_store import (
    ResidentSessionRecord,
    load_plan_file,
    load_resident_session,
    sessions_root,
    write_resident_session,
)


LeaseHandle = HostLease | GpuLease


def _coverage_fraction_for_plan(plan: SessionPlan) -> float:
    total_usable_bytes = sum(region.usable_bytes for region in plan.regions)
    covered_bytes = sum(region.covered_bytes for region in plan.regions)
    return (covered_bytes / total_usable_bytes) if total_usable_bytes else 0.0


def _percentile_int(values: list[int], percentile: float) -> int:
    if not values:
        return 0
    ordered = sorted(int(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return int(round(ordered[lower] + ((ordered[upper] - ordered[lower]) * weight)))


def _refresh_claim_notes(result: SessionResult, session_claim_notes: list[str]) -> None:
    result.claim_notes = (
        list(result.formal_claim_notes)
        + list(result.operational_claim_notes)
        + list(session_claim_notes)
    )


def _resolve_slot(plan: SessionPlan, challenge_index: int) -> tuple[str, int]:
    if challenge_index < 0 or challenge_index >= plan.label_count_m:
        raise ProtocolError(f"Challenge index {challenge_index} is outside [0, {plan.label_count_m})")
    cursor = 0
    for region in plan.regions:
        next_cursor = cursor + region.slot_count
        if challenge_index < next_cursor:
            return region.region_id, challenge_index - cursor
        cursor = next_cursor
    raise ProtocolError(f"Challenge index {challenge_index} could not be mapped to a planned region.")


def _sample_challenge_indices(plan: SessionPlan) -> list[int]:
    if not plan.challenge_policy.sample_with_replacement:
        raise ProtocolError("PoSE-DB runtime requires uniform sampling with replacement.")
    if plan.label_count_m <= 0:
        raise ProtocolError("Session plan label_count_m must be positive.")
    schedule_rng = Random(plan.plan_root_hex)
    return [schedule_rng.randrange(plan.label_count_m) for _ in range(plan.rounds_r)]


def _graph_for_plan(plan: SessionPlan) -> PoseDbGraph:
    graph = build_pose_db_graph(
        label_count_m=plan.label_count_m,
        graph_parameter_n=plan.graph_parameter_n,
        gamma=plan.gamma,
        hash_backend=plan.hash_backend,
        label_width_bits=plan.label_width_bits,
    )
    if graph.graph_descriptor_digest != plan.graph_descriptor_digest:
        raise ProtocolError(
            "Plan graph_descriptor_digest does not match the canonical digest for the declared graph parameters: "
            f"{plan.graph_descriptor_digest!r} != {graph.graph_descriptor_digest!r}"
        )
    return graph


def _result_from_plan(plan: SessionPlan) -> SessionResult:
    result = bootstrap_result(profile_name=plan.profile_name)
    total_usable_bytes = sum(region.usable_bytes for region in plan.regions)
    covered_bytes = sum(region.covered_bytes for region in plan.regions)
    slack_bytes = sum(region.slack_bytes for region in plan.regions)
    result.session_id = plan.session_id
    result.graph_family = plan.graph_family
    result.graph_parameter_n = plan.graph_parameter_n
    result.graph_descriptor_digest = plan.graph_descriptor_digest
    result.label_width_bits = plan.label_width_bits
    result.label_count_m = plan.label_count_m
    result.gamma = plan.gamma
    result.hash_backend = plan.hash_backend
    result.session_seed_commitment = f"sha256:{sha256_hex(bytes.fromhex(plan.session_seed_hex))}"
    result.adversary_model = plan.adversary_model
    result.attacker_budget_bytes_assumed = plan.attacker_budget_bytes_assumed
    result.target_success_bound = plan.challenge_policy.target_success_bound
    result.deadline_us = plan.deadline_policy.response_deadline_us
    result.q_bound = plan.q_bound
    result.rounds_r = plan.rounds_r
    result.covered_bytes = covered_bytes
    result.slack_bytes = slack_bytes
    result.coverage_fraction = (covered_bytes / total_usable_bytes) if total_usable_bytes else 0.0
    result.formal_claim_notes = [
        "formal claim is about local storage under the configured attacker budget",
    ]
    if plan.regions and all(region.region_type == "host" for region in plan.regions):
        result.operational_claim_notes = [
            "operational claim is about verifier-leased host regions for challenged slots",
        ]
    else:
        result.operational_claim_notes = [
            "operational claim is about verifier-leased mixed host/HBM regions for challenged slots",
        ]
    _refresh_claim_notes(result, plan.claim_notes)
    detected_host_total_bytes = 0
    try:
        if any(region.region_type == "host" for region in plan.regions):
            detected_host_total_bytes = detect_host_memory_bytes()
    except (OSError, ResourceFailure):
        detected_host_total_bytes = 0
    for region in plan.regions:
        if region.region_type == "host":
            result.host_usable_bytes += region.usable_bytes
            result.host_covered_bytes += region.covered_bytes
            continue
        if region.gpu_device is None:
            raise ProtocolError(f"GPU region {region.region_id} is missing gpu_device.")
        device_key = str(region.gpu_device)
        if region.gpu_device not in result.gpu_devices:
            result.gpu_devices.append(region.gpu_device)
        result.gpu_usable_bytes_by_device[device_key] = (
            result.gpu_usable_bytes_by_device.get(device_key, 0) + region.usable_bytes
        )
        result.gpu_covered_bytes_by_device[device_key] = (
            result.gpu_covered_bytes_by_device.get(device_key, 0) + region.covered_bytes
        )
    if detected_host_total_bytes > 0:
        result.host_total_bytes = detected_host_total_bytes
    else:
        result.host_total_bytes = result.host_usable_bytes
    result.notes.append(
        "PoSE-DB reference graph and label semantics are active for plan-file execution. "
        "In-place and multi-device optimizations are still pending."
    )
    return result


def _lease_record_for_region(region, lease: LeaseHandle) -> LeaseRecord:
    return replace(
        lease.record,
        slot_count=region.slot_count,
        slack_bytes=region.slack_bytes,
        gpu_device=region.gpu_device,
    )


def _create_runtime_lease(plan: SessionPlan, region) -> LeaseHandle:
    if region.region_type == "host":
        return create_host_lease(
            session_id=plan.session_id,
            region_id=region.region_id,
            usable_bytes=region.usable_bytes,
            cleanup_policy=plan.cleanup_policy,
            lease_duration_ms=plan.deadline_policy.session_timeout_ms,
        )
    if region.region_type == "gpu" and region.gpu_device is not None:
        return create_gpu_lease(
            session_id=plan.session_id,
            region_id=region.region_id,
            device=region.gpu_device,
            usable_bytes=region.usable_bytes,
            cleanup_policy=plan.cleanup_policy,
            lease_duration_ms=plan.deadline_policy.session_timeout_ms,
        )
    raise ProtocolError(f"Unsupported planned region type: {region.region_type!r}")


def _release_runtime_lease(lease: LeaseHandle, *, zeroize: bool, verify_zeroization: bool) -> str:
    if isinstance(lease, HostLease):
        return release_host_lease(
            lease,
            zeroize=zeroize,
            verify_zeroization=verify_zeroization,
        )
    return release_gpu_lease(
        lease,
        zeroize=zeroize,
        verify_zeroization=verify_zeroization,
    )


def _release_runtime_leases(
    leases: dict[str, LeaseHandle],
    *,
    zeroize: bool,
    verify_zeroization: bool,
) -> None:
    for lease in leases.values():
        _release_runtime_lease(
            lease,
            zeroize=zeroize,
            verify_zeroization=verify_zeroization,
        )


def _close_runtime_leases(leases: dict[str, LeaseHandle]) -> None:
    for lease in leases.values():
        lease.close()


def _terminate_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _supports_host_retention(plan: SessionPlan) -> bool:
    return len(plan.regions) == 1 and all(region.region_type == "host" for region in plan.regions)


def _lease_expiry_for_leases(leases: dict[str, LeaseHandle]) -> str:
    expiries = sorted(lease.record.lease_expiry for lease in leases.values())
    return expiries[0] if expiries else ""


def _resident_record_for_success(
    *,
    session_plan: SessionPlan,
    result: SessionResult,
    socket_path: str,
    process_id: int,
    lease_expiry: str,
) -> ResidentSessionRecord:
    host_region = session_plan.regions[0]
    return ResidentSessionRecord(
        session_id=session_plan.session_id,
        profile_name=session_plan.profile_name,
        session_seed_hex=session_plan.session_seed_hex,
        session_plan_root=session_plan.graph_descriptor_digest,
        graph_family=session_plan.graph_family,
        graph_parameter_n=session_plan.graph_parameter_n,
        graph_descriptor_digest=session_plan.graph_descriptor_digest,
        label_width_bits=session_plan.label_width_bits,
        label_count_m=session_plan.label_count_m,
        gamma=session_plan.gamma,
        hash_backend=session_plan.hash_backend,
        region_id=host_region.region_id,
        region_slot_count=host_region.slot_count,
        challenge_policy={
            "rounds_r": session_plan.rounds_r,
            "sample_with_replacement": session_plan.challenge_policy.sample_with_replacement,
            "target_success_bound": session_plan.challenge_policy.target_success_bound,
        },
        deadline_us=session_plan.deadline_policy.response_deadline_us,
        cleanup_policy={
            "zeroize": session_plan.cleanup_policy.zeroize,
            "verify_zeroization": session_plan.cleanup_policy.verify_zeroization,
        },
        adversary_model=session_plan.adversary_model,
        attacker_budget_bytes_assumed=session_plan.attacker_budget_bytes_assumed,
        q_bound=session_plan.q_bound,
        host_total_bytes=result.host_total_bytes,
        host_usable_bytes=result.host_usable_bytes,
        host_covered_bytes=result.host_covered_bytes,
        covered_bytes=result.covered_bytes,
        slack_bytes=result.slack_bytes,
        coverage_fraction=result.coverage_fraction,
        scratch_peak_bytes=result.scratch_peak_bytes,
        declared_stage_copy_bytes=result.declared_stage_copy_bytes,
        formal_claim_notes=list(result.formal_claim_notes),
        operational_claim_notes=list(result.operational_claim_notes),
        claim_notes=list(session_plan.claim_notes),
        socket_path=socket_path,
        process_id=process_id,
        lease_expiry=lease_expiry,
    )


class VerifierService:
    def describe(self) -> dict[str, object]:
        return {
            "status": "pose-db-control-plane-slot-planning-partial",
            "supports_host_memory": True,
            "supports_gpu_hbm": True,
            "supports_plan_files": True,
            "supports_profile_runs": True,
            "supports_rechallenge": True,
        }

    def run_session(
        self,
        profile: BenchmarkProfile,
        *,
        retain_session: bool = False,
        session_plan: SessionPlan | None = None,
    ) -> SessionResult:
        calibration_payload = calibrate_profile(profile, persist_artifact=True)
        if str(calibration_payload.get("status")) != "calibrated":
            result = bootstrap_result(profile_name=profile.name)
            result.verdict = "CALIBRATION_INVALID"
            planning = calibration_payload.get("planning")
            if isinstance(planning, dict):
                result.graph_family = str(planning.get("graph_family", result.graph_family))
                result.graph_parameter_n = int(planning.get("graph_parameter_n", result.graph_parameter_n))
                result.graph_descriptor_digest = str(
                    planning.get("graph_descriptor_digest", result.graph_descriptor_digest)
                )
                result.label_count_m = int(planning.get("label_count_m", result.label_count_m))
                result.gamma = int(planning.get("gamma", result.gamma))
                result.hash_backend = str(planning.get("hash_backend", result.hash_backend))
                result.attacker_budget_bytes_assumed = int(
                    planning.get(
                        "effective_attacker_budget_bytes_assumed",
                        profile.attacker_budget_bytes_assumed,
                    )
                )
                result.covered_bytes = int(planning.get("covered_bytes", 0))
                result.slack_bytes = int(planning.get("slack_bytes", 0))
                total_usable_bytes = int(planning.get("total_usable_bytes", 0))
                result.coverage_fraction = (
                    result.covered_bytes / total_usable_bytes if total_usable_bytes > 0 else 0.0
                )
                result.host_total_bytes = int(planning.get("host_total_bytes", 0))
                result.host_usable_bytes = int(planning.get("host_usable_bytes", 0))
                result.host_covered_bytes = int(planning.get("host_covered_bytes", 0))
                result.claim_notes = [
                    str(item) for item in planning.get("claim_notes", []) if str(item).strip()
                ]
            if "q_bound" in calibration_payload:
                result.q_bound = int(calibration_payload["q_bound"])
            if "rounds_r" in calibration_payload:
                result.rounds_r = int(calibration_payload["rounds_r"])
            soundness = calibration_payload.get("soundness")
            if isinstance(soundness, dict):
                result.reported_success_bound = float(soundness.get("reported_success_bound", 0.0))
                result.soundness_model = str(soundness.get("soundness_model", result.soundness_model))
            result.notes.extend(str(item) for item in calibration_payload.get("notes", []))
            artifact_path = str(calibration_payload.get("artifact_path", "")).strip()
            if artifact_path:
                result.notes.append(f"calibration_artifact={artifact_path}")
            if session_plan is not None:
                result.session_id = session_plan.session_id
            return result
        planned_session = session_plan or build_session_plan_from_profile(profile, calibration_payload)
        planned_coverage_fraction = _coverage_fraction_for_plan(planned_session)
        if planned_coverage_fraction < float(profile.coverage_threshold):
            result = _result_from_plan(planned_session)
            result.verdict = "COVERAGE_BELOW_THRESHOLD"
            result.notes.append(
                "Planned covered-byte fraction is below the profile coverage_threshold: "
                f"{planned_coverage_fraction:.6f} < {float(profile.coverage_threshold):.6f}."
            )
            return result
        return self._run_session_plan(
            planned_session,
            retain_session=retain_session,
            extra_notes=[
                f"profile-driven slot planning via {profile.name}",
                f"calibration_artifact={calibration_payload.get('artifact_path', '')}",
            ],
            prover_sandbox=profile.prover_sandbox,
        )

    def run_plan_file(self, path: Path) -> SessionResult:
        plan_file = load_plan_file(path)
        return self._run_session_plan(
            plan_file.session_plan,
            retain_session=plan_file.retain_session,
            extra_notes=[],
        )

    def _run_session_plan(
        self,
        session_plan: SessionPlan,
        *,
        retain_session: bool,
        extra_notes: list[str],
        prover_sandbox: ProverSandboxPolicy | None = None,
    ) -> SessionResult:
        result = _result_from_plan(session_plan)
        total_started = perf_counter()
        for note in extra_notes:
            if note:
                result.notes.append(note)
        if retain_session and not _supports_host_retention(session_plan):
            result.verdict = "PROTOCOL_ERROR"
            result.notes.append(
                "retain_session currently requires exactly one host region and no GPU regions."
            )
            result.timings_ms["total"] = int((perf_counter() - total_started) * 1000)
            return result
        if session_plan.q_bound >= session_plan.gamma:
            result.verdict = "CALIBRATION_INVALID"
            result.notes.append(
                f"Plan q_bound must be strictly less than gamma, got q={session_plan.q_bound} and gamma={session_plan.gamma}."
            )
            result.timings_ms["total"] = int((perf_counter() - total_started) * 1000)
            return result
        try:
            soundness = assess_soundness(
                label_count_m=session_plan.label_count_m,
                rounds_r=session_plan.rounds_r,
                q_bound=session_plan.q_bound,
                gamma=session_plan.gamma,
                label_width_bits=session_plan.label_width_bits,
                attacker_budget_bytes_assumed=session_plan.attacker_budget_bytes_assumed,
                adversary_model=session_plan.adversary_model,
                target_success_bound=session_plan.challenge_policy.target_success_bound,
            )
        except ProtocolError as error:
            result.verdict = "CALIBRATION_INVALID"
            result.notes.append(str(error))
            result.timings_ms["total"] = int((perf_counter() - total_started) * 1000)
            return result
        result.soundness_model = soundness_model_label(soundness.adversary_model)
        result.reported_success_bound = soundness.reported_success_bound
        if session_plan.challenge_policy.target_success_bound > 0.0 and not soundness.target_met:
            result.notes.append(
                "Declared rounds do not meet the requested target_success_bound; "
                "reported_success_bound reflects the conservative theorem bound for the active plan."
            )

        socket_path = ""
        process: subprocess.Popen[str] | None = None
        leases: dict[str, LeaseHandle] = {}
        graph: PoseDbGraph | None = None
        remote_cleaned = False
        session_registered = False
        resident_retained = False
        temp_dir: TemporaryDirectory[str] | None = None
        try:
            if retain_session:
                socket_path = str(sessions_root() / f"{session_plan.session_id}.sock")
            else:
                temp_dir = TemporaryDirectory(prefix="pose-db-plan-")
                socket_path = str(Path(temp_dir.name) / "prover.sock")
            process = start_ephemeral_prover_server(
                socket_path=socket_path,
                prover_sandbox=prover_sandbox,
            )

            discover_started = perf_counter()
            probe = discover(socket_path)
            result.timings_ms["discover"] = int((perf_counter() - discover_started) * 1000)
            if "pose-db-fast-phase" not in probe["capabilities"]:
                raise ProtocolError("Prover does not advertise PoSE-DB fast-phase support.")

            allocation_started = perf_counter()
            try:
                for region in session_plan.regions:
                    leases[region.region_id] = _create_runtime_lease(session_plan, region)
            except Exception:
                if leases:
                    _release_runtime_leases(
                        leases,
                        zeroize=session_plan.cleanup_policy.zeroize,
                        verify_zeroization=session_plan.cleanup_policy.verify_zeroization,
                    )
                    leases.clear()
                raise
            result.timings_ms["allocation"] = int((perf_counter() - allocation_started) * 1000)

            graph_started = perf_counter()
            graph = _graph_for_plan(session_plan)
            result.timings_ms["graph_construction"] = int((perf_counter() - graph_started) * 1000)

            plan_session(socket_path, session_plan)
            session_registered = True

            leasing_started = perf_counter()
            lease_regions(
                socket_path,
                session_plan.session_id,
                [
                    _lease_record_for_region(region, leases[region.region_id])
                    for region in session_plan.regions
                ],
            )
            result.timings_ms["region_leasing"] = int((perf_counter() - leasing_started) * 1000)

            seed_session(socket_path, session_plan.session_id)
            materialization_report, worker_timings = materialize_labels(socket_path, session_plan.session_id)
            if str(materialization_report.get("graph_descriptor_digest", "")) != session_plan.graph_descriptor_digest:
                raise ProtocolError(
                    "MaterializeLabels reported a graph_descriptor_digest that does not match the session plan."
                )
            result.scratch_peak_bytes = int(materialization_report.get("scratch_peak_bytes", 0))
            regions_report = materialization_report.get("regions")
            if not isinstance(regions_report, dict):
                raise ProtocolError("MaterializeLabels did not return per-region metadata.")
            if set(regions_report) != {region.region_id for region in session_plan.regions}:
                raise ProtocolError("MaterializeLabels must report exactly one metadata entry per planned region.")
            declared_stage_copy_bytes = 0
            for region in session_plan.regions:
                region_report = regions_report.get(region.region_id)
                if not isinstance(region_report, dict):
                    raise ProtocolError(f"MaterializeLabels is missing metadata for region {region.region_id}.")
                if int(region_report.get("covered_bytes", -1)) != region.covered_bytes:
                    raise ProtocolError(
                        f"MaterializeLabels reported covered_bytes that do not match the plan for {region.region_id}."
                    )
                if int(region_report.get("slack_bytes", -1)) != region.slack_bytes:
                    raise ProtocolError(
                        f"MaterializeLabels reported slack_bytes that do not match the plan for {region.region_id}."
                    )
                region_stage_copy_bytes = int(region_report.get("declared_stage_copy_bytes", 0))
                declared_stage_copy_bytes += region_stage_copy_bytes
                if region_stage_copy_bytes != 0:
                    result.declared_stage_copy_bytes = declared_stage_copy_bytes
                    result.operational_claim_notes.append(
                        "materialization declared surviving stage copies into the fast phase totaling "
                        f"{declared_stage_copy_bytes} bytes"
                    )
                    _refresh_claim_notes(result, session_plan.claim_notes)
                    result.notes.append(
                        "Declared stage copies survive into the fast phase and are explicitly reported in the "
                        "result artifact; the current host/runtime slice rejects such sessions."
                    )
                    raise ProtocolError(
                        "Declared stage copies are not supported in the current PoSE-DB runtime slice."
                    )
            result.declared_stage_copy_bytes = declared_stage_copy_bytes
            result.formal_claim_notes.append("no surviving stage buffers declared before fast phase")
            if result.scratch_peak_bytes > 0:
                result.operational_claim_notes.append(
                    "materialization used bounded transient interpreter-estimated scratch up to "
                    f"{result.scratch_peak_bytes} bytes"
                )
            _refresh_claim_notes(result, session_plan.claim_notes)
            for key, value in worker_timings.items():
                if key in result.timings_ms:
                    result.timings_ms[key] = value

            prepare_fast_phase(socket_path, session_plan.session_id)

            schedule_started = perf_counter()
            challenge_indices = _sample_challenge_indices(session_plan)
            result.timings_ms["challenge_schedule_prep"] = int((perf_counter() - schedule_started) * 1000)

            expected_started = perf_counter()
            if graph is None:
                raise ProtocolError("Graph construction must complete before preparing expected responses.")
            expected_labels = compute_challenge_labels(
                graph,
                session_seed=session_plan.session_seed_hex,
                challenge_indices=challenge_indices,
            )
            result.timings_ms["expected_response_prep"] = int((perf_counter() - expected_started) * 1000)

            fast_phase_started = perf_counter()
            verify_started = perf_counter()
            deadline_miss = False
            wrong_response = False
            accepted_rounds = 0
            round_trip_us_values: list[int] = []
            with FastPhaseClient(socket_path) as fast_phase_client:
                for round_index, (challenge_index, expected_label) in enumerate(
                    zip(challenge_indices, expected_labels, strict=True)
                ):
                    response = fast_phase_client.run_round(
                        session_id=session_plan.session_id,
                        round_index=round_index,
                        challenge_index=challenge_index,
                    )
                    round_trip_us_values.append(int(response["round_trip_us"]))
                    result.max_round_trip_us = max(result.max_round_trip_us, int(response["round_trip_us"]))
                    response_is_valid = (
                        int(response["challenge_index"]) == challenge_index
                        and bytes(response["label_bytes"]) == expected_label
                    )
                    if not response_is_valid:
                        wrong_response = True
                        break
                    if int(response["round_trip_us"]) > session_plan.deadline_policy.response_deadline_us:
                        deadline_miss = True
                        break
                    accepted_rounds += 1
            result.timings_ms["fast_phase_total"] = int((perf_counter() - fast_phase_started) * 1000)
            result.accepted_rounds = accepted_rounds
            result.round_trip_p50_us = _percentile_int(round_trip_us_values, 0.50)
            result.round_trip_p95_us = _percentile_int(round_trip_us_values, 0.95)
            result.round_trip_p99_us = _percentile_int(round_trip_us_values, 0.99)
            result.timings_ms["verifier_check_total"] = int((perf_counter() - verify_started) * 1000)

            if wrong_response:
                result.verdict = "WRONG_RESPONSE"
            elif deadline_miss:
                result.verdict = "DEADLINE_MISS"
            else:
                result.verdict = "SUCCESS"
                result.success = True

            should_retain = bool(retain_session and result.success)
            finalize_session(
                socket_path,
                session_id=session_plan.session_id,
                verdict=result.verdict,
                success=result.success,
                retain_session=should_retain,
            )
            if should_retain:
                resident_record = _resident_record_for_success(
                    session_plan=session_plan,
                    result=result,
                    socket_path=socket_path,
                    process_id=0 if process is None else process.pid,
                    lease_expiry=_lease_expiry_for_leases(leases),
                )
                session_record_path = write_resident_session(resident_record)
                result.cleanup_status = "RETAINED_FOR_RECHALLENGE"
                result.resident_socket_path = socket_path
                result.resident_process_id = resident_record.process_id
                result.lease_expiry = resident_record.lease_expiry
                result.notes.append(f"resident_session_record={session_record_path}")
                resident_retained = True
            else:
                cleanup_started = perf_counter()
                result.cleanup_status = cleanup_session(socket_path, session_plan.session_id)
                remote_cleaned = True
                result.timings_ms["cleanup"] = int((perf_counter() - cleanup_started) * 1000)
        except OSError as error:
            result.verdict = "RESOURCE_FAILURE"
            result.success = False
            result.notes.append(str(error))
        except ResourceFailure as error:
            result.verdict = "RESOURCE_FAILURE"
            result.success = False
            result.notes.append(str(error))
        except ProtocolError as error:
            result.verdict = "PROTOCOL_ERROR"
            result.success = False
            result.notes.append(str(error))
        finally:
            if not resident_retained and not remote_cleaned and session_registered and socket_path:
                cleanup_started = perf_counter()
                try:
                    result.cleanup_status = cleanup_session(socket_path, session_plan.session_id)
                    remote_cleaned = True
                except (OSError, ProtocolError, ResourceFailure):
                    pass
                else:
                    result.timings_ms["cleanup"] = int((perf_counter() - cleanup_started) * 1000)
            if leases and not resident_retained:
                _release_runtime_leases(
                    leases,
                    zeroize=not remote_cleaned and session_plan.cleanup_policy.zeroize,
                    verify_zeroization=not remote_cleaned and session_plan.cleanup_policy.verify_zeroization,
                )
            elif leases and resident_retained:
                _close_runtime_leases(leases)
            if process is not None and not resident_retained:
                _terminate_process(process)
            if temp_dir is not None:
                temp_dir.cleanup()
        result.timings_ms["total"] = int((perf_counter() - total_started) * 1000)
        return result

    def run_placeholder(self, profile: BenchmarkProfile, note: str) -> SessionResult:
        return bootstrap_result(profile_name=profile.name, note=note)

    def rechallenge(self, session_id: str, *, release: bool = False) -> SessionResult:
        try:
            record = load_resident_session(session_id)
        except FileNotFoundError as error:
            raise ProtocolError(f"Unknown retained PoSE-DB session: {session_id}") from error
        return run_host_rechallenge(record, release=release)

    def verify_record(self, path: Path) -> SessionResult:
        payload = load_json_file(path)
        return SessionResult.from_dict(payload)
