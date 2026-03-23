from __future__ import annotations

import os
import signal
from dataclasses import replace
from datetime import UTC, datetime
from time import perf_counter

from pose.common.errors import ProtocolError, ResourceFailure
from pose.common.hashing import sha256_hex
from pose.graphs import build_pose_db_graph, compute_challenge_labels, preferred_runtime_label_engine
from pose.protocol.result_schema import SessionResult, bootstrap_result
from pose.verifier.challenges import sample_challenge_indices
from pose.verifier.grpc_client import FastPhaseClient, cleanup_session, discover, finalize_session
from pose.verifier.session_store import ResidentSessionRecord, delete_resident_session, write_resident_session
from pose.verifier.soundness import assess_soundness, soundness_model_label


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


def _refresh_claim_notes(result: SessionResult, claim_notes: list[str]) -> None:
    result.claim_notes = (
        list(result.formal_claim_notes)
        + list(result.operational_claim_notes)
        + list(claim_notes)
    )


def _terminate_resident_process(process_id: int) -> None:
    if process_id <= 0:
        return
    try:
        os.killpg(process_id, signal.SIGTERM)
    except ProcessLookupError:
        return
    except OSError:
        try:
            os.kill(process_id, signal.SIGTERM)
        except (OSError, ProcessLookupError):
            return


def _cleanup_resident_session(record: ResidentSessionRecord) -> str:
    cleanup_status = "NOT_RUN"
    try:
        cleanup_status = cleanup_session(record.socket_path, record.session_id)
    except (OSError, ProtocolError, ResourceFailure):
        cleanup_status = "SESSION_UNAVAILABLE"
    finally:
        _terminate_resident_process(record.process_id)
        try:
            if record.socket_path:
                os.unlink(record.socket_path)
        except FileNotFoundError:
            pass
        except OSError:
            pass
        delete_resident_session(record.session_id)
    return cleanup_status


def _sample_challenge_indices(record: ResidentSessionRecord) -> list[int]:
    return sample_challenge_indices(
        label_count_m=record.label_count_m,
        rounds_r=int(record.challenge_policy.get("rounds_r", 0)),
        sample_with_replacement=bool(record.challenge_policy.get("sample_with_replacement", True)),
    )


def _result_from_record(record: ResidentSessionRecord) -> SessionResult:
    result = bootstrap_result(profile_name=record.profile_name)
    result.run_class = "rechallenge"
    result.session_id = record.session_id
    result.graph_family = record.graph_family
    result.graph_parameter_n = record.graph_parameter_n
    result.graph_descriptor_digest = record.graph_descriptor_digest
    result.label_width_bits = record.label_width_bits
    result.label_count_m = record.label_count_m
    result.gamma = record.gamma
    result.hash_backend = record.hash_backend
    result.session_seed_commitment = f"sha256:{sha256_hex(bytes.fromhex(record.session_seed_hex))}"
    result.resident_socket_path = record.socket_path
    result.resident_process_id = record.process_id
    result.lease_expiry = record.lease_expiry
    result.adversary_model = record.adversary_model
    result.attacker_budget_bytes_assumed = record.attacker_budget_bytes_assumed
    result.deadline_us = record.deadline_us
    result.q_bound = record.q_bound
    result.rounds_r = int(record.challenge_policy.get("rounds_r", 0))
    result.host_total_bytes = record.host_total_bytes
    result.host_usable_bytes = record.host_usable_bytes
    result.host_covered_bytes = record.host_covered_bytes
    result.covered_bytes = record.covered_bytes
    result.slack_bytes = record.slack_bytes
    result.coverage_fraction = record.coverage_fraction
    result.scratch_peak_bytes = record.scratch_peak_bytes
    result.declared_stage_copy_bytes = record.declared_stage_copy_bytes
    result.formal_claim_notes = list(record.formal_claim_notes)
    result.operational_claim_notes = list(record.operational_claim_notes)
    _refresh_claim_notes(result, record.claim_notes)
    result.notes.append("PoSE-DB rechallenge run over a retained host session.")
    return result


def run_host_rechallenge(
    record: ResidentSessionRecord,
    *,
    release: bool = False,
) -> SessionResult:
    total_started = perf_counter()
    result = _result_from_record(record)

    try:
        soundness = assess_soundness(
            label_count_m=record.label_count_m,
            rounds_r=result.rounds_r,
            q_bound=record.q_bound,
            gamma=record.gamma,
            label_width_bits=record.label_width_bits,
            attacker_budget_bytes_assumed=record.attacker_budget_bytes_assumed,
            adversary_model=record.adversary_model,
            target_success_bound=float(record.challenge_policy.get("target_success_bound", 0.0)),
        )
        result.soundness_model = soundness_model_label(soundness.adversary_model)
        result.reported_success_bound = soundness.reported_success_bound
        result.target_success_bound = soundness.target_success_bound
    except ProtocolError as error:
        result.verdict = "CALIBRATION_INVALID"
        result.notes.append(str(error))
        result.timings_ms["total"] = int((perf_counter() - total_started) * 1000)
        return result

    if datetime.fromisoformat(record.lease_expiry) <= datetime.now(UTC):
        result.verdict = "RESOURCE_FAILURE"
        result.cleanup_status = _cleanup_resident_session(record)
        if result.cleanup_status == "NOT_RUN":
            result.cleanup_status = "LEASE_EXPIRED"
        result.notes.append("Resident session lease expired before rechallenge.")
        result.timings_ms["total"] = int((perf_counter() - total_started) * 1000)
        return result

    try:
        discover_started = perf_counter()
        probe = discover(record.socket_path)
        result.timings_ms["discover"] = int((perf_counter() - discover_started) * 1000)
        if "pose-db-fast-phase" not in list(probe.get("capabilities", [])):
            raise ProtocolError("Resident prover does not advertise PoSE-DB fast-phase support.")

        graph_started = perf_counter()
        graph = build_pose_db_graph(
            label_count_m=record.label_count_m,
            graph_parameter_n=record.graph_parameter_n,
            gamma=record.gamma,
            hash_backend=record.hash_backend,
            label_width_bits=record.label_width_bits,
        )
        result.timings_ms["graph_construction"] = int((perf_counter() - graph_started) * 1000)
        if graph.graph_descriptor_digest != record.graph_descriptor_digest:
            raise ProtocolError("Resident session graph descriptor digest does not match the canonical graph.")

        schedule_started = perf_counter()
        challenge_indices = _sample_challenge_indices(record)
        result.timings_ms["challenge_schedule_prep"] = int((perf_counter() - schedule_started) * 1000)

        expected_started = perf_counter()
        expected_labels = compute_challenge_labels(
            graph,
            session_seed=record.session_seed_hex,
            challenge_indices=challenge_indices,
            label_engine=preferred_runtime_label_engine(),
        )
        result.timings_ms["expected_response_prep"] = int((perf_counter() - expected_started) * 1000)

        fast_phase_started = perf_counter()
        verify_started = perf_counter()
        deadline_miss = False
        wrong_response = False
        accepted_rounds = 0
        round_trip_us_values: list[int] = []
        with FastPhaseClient(record.socket_path) as fast_phase_client:
            for round_index, (challenge_index, expected_label) in enumerate(
                zip(challenge_indices, expected_labels, strict=True)
            ):
                response = fast_phase_client.run_round(
                    session_id=record.session_id,
                    round_index=round_index,
                    challenge_index=challenge_index,
                )
                round_trip_us = int(response["round_trip_us"])
                round_trip_us_values.append(round_trip_us)
                result.max_round_trip_us = max(result.max_round_trip_us, round_trip_us)
                if (
                    int(response["challenge_index"]) != challenge_index
                    or bytes(response["label_bytes"]) != expected_label
                ):
                    wrong_response = True
                    break
                if round_trip_us > record.deadline_us:
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

        should_retain = bool(result.success and not release)
        finalize_session(
            record.socket_path,
            session_id=record.session_id,
            verdict=result.verdict,
            success=result.success,
            retain_session=should_retain,
        )
        if should_retain:
            write_resident_session(
                replace(
                    record,
                    scratch_peak_bytes=result.scratch_peak_bytes,
                    declared_stage_copy_bytes=result.declared_stage_copy_bytes,
                    formal_claim_notes=list(result.formal_claim_notes),
                    operational_claim_notes=list(result.operational_claim_notes),
                )
            )
            result.cleanup_status = "RETAINED_FOR_RECHALLENGE"
        else:
            cleanup_started = perf_counter()
            result.cleanup_status = _cleanup_resident_session(record)
            result.timings_ms["cleanup"] = int((perf_counter() - cleanup_started) * 1000)
            if result.cleanup_status == "SESSION_UNAVAILABLE":
                result.verdict = "CLEANUP_FAILURE"
                result.success = False
                result.notes.append("Failed to clean up retained host session after rechallenge.")
    except (OSError, ProtocolError, ResourceFailure) as error:
        result.verdict = "RESOURCE_FAILURE" if isinstance(error, OSError | ResourceFailure) else "PROTOCOL_ERROR"
        result.success = False
        result.notes.append(str(error))
        cleanup_started = perf_counter()
        if release or result.verdict != "SUCCESS":
            result.cleanup_status = _cleanup_resident_session(record)
            result.timings_ms["cleanup"] = int((perf_counter() - cleanup_started) * 1000)

    result.timings_ms["total"] = int((perf_counter() - total_started) * 1000)
    return result
