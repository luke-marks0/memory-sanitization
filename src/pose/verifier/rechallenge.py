from __future__ import annotations

import secrets
from datetime import UTC, datetime
from time import perf_counter

from pose.common.errors import ProtocolError, ResourceFailure
from pose.protocol.result_schema import SessionResult, bootstrap_result
from pose.verifier.challenge import challenge_count_for_policy, sample_leaf_indices
from pose.verifier.deadlines import response_within_deadline
from pose.verifier.grpc_client import challenge_outer, cleanup_session, finalize_session
from pose.verifier.outer import decode_opening_payload, verify_outer_challenge_response
from pose.verifier.session_store import ResidentSessionRecord, delete_resident_session


def run_host_rechallenge(
    record: ResidentSessionRecord,
    *,
    release: bool = False,
) -> SessionResult:
    total_started = perf_counter()
    result = bootstrap_result(profile_name=record.profile_name)
    result.run_class = "rechallenge"
    result.session_id = record.session_id
    result.session_nonce = record.session_nonce
    result.session_plan_root = record.session_plan_root
    result.session_manifest_root = record.session_manifest_root
    result.resident_socket_path = record.socket_path
    result.resident_process_id = record.process_id
    result.lease_expiry = record.lease_expiry
    result.challenge_leaf_size = record.challenge_leaf_size
    result.challenge_policy = dict(record.challenge_policy)
    result.deadline_ms = record.deadline_ms
    result.cleanup_policy = dict(record.cleanup_policy)
    result.host_total_bytes = record.host_total_bytes
    result.host_usable_bytes = record.host_usable_bytes
    result.host_covered_bytes = record.host_covered_bytes
    result.real_porep_bytes = record.real_porep_bytes
    result.tail_filler_bytes = record.tail_filler_bytes
    result.real_porep_ratio = record.real_porep_ratio
    result.coverage_fraction = record.coverage_fraction
    result.inner_filecoin_verified = record.inner_filecoin_verified
    result.region_roots = {record.region_id: record.region_root_hex}
    result.region_manifest_roots = {record.region_id: record.region_manifest_root}
    result.region_payload_bytes_by_region = {record.region_id: record.host_covered_bytes}
    result.notes.append("Rechallenge run over resident host payload.")

    if datetime.fromisoformat(record.lease_expiry) <= datetime.now(UTC):
        delete_resident_session(record.session_id)
        result.verdict = "RESOURCE_FAILURE"
        result.cleanup_status = "LEASE_EXPIRED"
        result.notes.append("Resident session lease expired before rechallenge.")
        result.timings_ms["total"] = int((perf_counter() - total_started) * 1000)
        return result

    total_leaves = record.host_covered_bytes // record.challenge_leaf_size
    result.challenge_count = challenge_count_for_policy(
        total_leaves=total_leaves,
        epsilon=float(record.challenge_policy["epsilon"]),
        lambda_bits=int(record.challenge_policy["lambda_bits"]),
        max_challenges=int(record.challenge_policy["max_challenges"]),
    )
    rechallenge_nonce = secrets.token_hex(8)
    challenge_indices = sample_leaf_indices(
        total_leaves,
        result.challenge_count,
        seed=f"{record.session_manifest_root}:{rechallenge_nonce}",
    )
    result.challenge_indices_by_region = {record.region_id: challenge_indices}

    try:
        opening_payloads, result.response_ms = challenge_outer(
            record.socket_path,
            session_id=record.session_id,
            region_id=record.region_id,
            session_manifest_root=record.session_manifest_root,
            challenge_indices=challenge_indices,
        )
        result.timings_ms["challenge_response"] = result.response_ms
    except (OSError, ProtocolError, ResourceFailure) as error:
        delete_resident_session(record.session_id)
        result.verdict = "RESOURCE_FAILURE"
        result.cleanup_status = "SESSION_UNAVAILABLE"
        result.notes.append(str(error))
        result.timings_ms["total"] = int((perf_counter() - total_started) * 1000)
        return result

    openings = [decode_opening_payload(payload) for payload in opening_payloads]
    outer_verify_started = perf_counter()
    result.outer_pose_verified = response_within_deadline(
        result.response_ms, result.deadline_ms
    ) and verify_outer_challenge_response(
        expected_region_id=record.region_id,
        expected_session_manifest_root=record.session_manifest_root,
        expected_indices=challenge_indices,
        root=bytes.fromhex(record.region_root_hex),
        leaf_size=record.challenge_leaf_size,
        openings=openings,
    )
    result.timings_ms["outer_verify"] = int((perf_counter() - outer_verify_started) * 1000)

    if not result.inner_filecoin_verified:
        result.verdict = "INNER_PROOF_INVALID"
    elif not response_within_deadline(result.response_ms, result.deadline_ms):
        result.verdict = "TIMEOUT"
    elif not result.outer_pose_verified:
        result.verdict = "OUTER_PROOF_INVALID"
    elif result.real_porep_ratio < 0.99:
        result.verdict = "COVERAGE_BELOW_THRESHOLD"
    else:
        result.verdict = "SUCCESS"
        result.success = True

    finalize_session(
        record.socket_path,
        session_id=record.session_id,
        verdict=result.verdict,
        success=result.success,
        retain_session=not release and result.success,
    )

    if release:
        cleanup_started = perf_counter()
        try:
            result.cleanup_status = cleanup_session(record.socket_path, record.session_id)
        except (OSError, ProtocolError, ResourceFailure) as error:
            result.cleanup_status = "CLEANUP_FAILED"
            result.verdict = "CLEANUP_FAILURE"
            result.success = False
            result.notes.append(str(error))
        finally:
            delete_resident_session(record.session_id)
            result.timings_ms["cleanup"] = int((perf_counter() - cleanup_started) * 1000)
    else:
        result.cleanup_status = "RETAINED_FOR_RECHALLENGE"

    result.timings_ms["total"] = int((perf_counter() - total_started) * 1000)
    return result
