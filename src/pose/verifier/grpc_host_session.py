from __future__ import annotations

import secrets
from time import perf_counter

from pose.benchmarks.profiles import BenchmarkProfile
from pose.common.errors import ProtocolError, ResourceFailure
from pose.common.timing import TimingTracker
from pose.filecoin.reference import VendoredFilecoinReference, summarize_cpu_fallbacks
from pose.protocol.messages import CleanupPolicy, SessionPlan
from pose.protocol.region_payloads import (
    RegionManifest,
    SessionManifest,
    region_manifest_matches_payload,
)
from pose.protocol.result_schema import SessionResult, bootstrap_result
from pose.verifier.challenge import challenge_count_for_policy, sample_leaf_indices
from pose.verifier.deadlines import response_within_deadline
from pose.verifier.grpc_client import (
    challenge_outer,
    cleanup_session,
    commit_regions,
    finalize_session,
    generate_inner_porep,
    materialize_region_payloads,
    plan_session,
    start_ephemeral_prover_server,
    verify_inner_proofs,
    verify_outer,
    lease_regions,
)
from pose.verifier.leasing import create_host_lease, release_host_lease
from pose.verifier.outer import decode_opening_payload, verify_outer_challenge_response
from pose.verifier.host_planning import (
    build_minimal_host_session_plan,
    detect_host_memory_bytes,
    validate_minimal_host_session_plan,
)
from pose.verifier.session_store import ResidentSessionRecord, sessions_root, write_resident_session


def run_host_session_via_grpc(
    profile: BenchmarkProfile,
    *,
    requested_unit_count: int | None = None,
    requested_host_bytes: int | None = None,
    session_plan: SessionPlan | None = None,
    retain_session: bool = False,
    run_class: str | None = None,
) -> SessionResult:
    tracker = TimingTracker()
    total_started = perf_counter()
    result = bootstrap_result(profile_name=profile.name)
    result.run_class = run_class or str(profile.benchmark_class)

    if not profile.target_devices.get("host", False) or profile.target_devices.get("gpus"):
        result.verdict = "PROTOCOL_ERROR"
        result.notes.append("Only host-only profiles are supported by the current host session runner.")
        result.timings_ms["total"] = int((perf_counter() - total_started) * 1000)
        return result
    if profile.porep_unit_profile != "minimal":
        result.verdict = "RESOURCE_FAILURE"
        result.notes.append(
            "Profile requires replica/full-cache blobs, but the current Phase 1 bridge "
            "only materializes the minimal PoRep unit profile."
        )
        result.timings_ms["total"] = int((perf_counter() - total_started) * 1000)
        return result

    if session_plan is None:
        result.challenge_leaf_size = profile.leaf_size
        result.deadline_ms = int(profile.deadline_policy["response_deadline_ms"])
        result.session_nonce = secrets.token_hex(16)
        session_plan, host_plan = build_minimal_host_session_plan(
            profile,
            session_id=result.session_id,
            session_nonce=result.session_nonce,
            requested_unit_count=requested_unit_count,
            requested_host_bytes=requested_host_bytes,
            detected_host_bytes=detect_host_memory_bytes(),
        )
        validate_minimal_host_session_plan(session_plan)
        unit_count = session_plan.unit_count
        region_id = session_plan.regions[0].region_id
        planned_region_bytes = session_plan.regions[0].usable_bytes
        host_total_bytes = host_plan.total_bytes
    else:
        try:
            validate_minimal_host_session_plan(session_plan)
        except (ProtocolError, ResourceFailure) as error:
            result.verdict = "PROTOCOL_ERROR" if isinstance(error, ProtocolError) else "RESOURCE_FAILURE"
            result.notes.append(str(error))
            result.timings_ms["total"] = int((perf_counter() - total_started) * 1000)
            return result
        region_plan = session_plan.regions[0]
        result.session_id = session_plan.session_id
        result.profile_name = session_plan.profile_name
        result.challenge_leaf_size = session_plan.challenge_leaf_size
        result.deadline_ms = session_plan.deadline_policy.response_deadline_ms
        result.session_nonce = session_plan.nonce
        unit_count = session_plan.unit_count
        region_id = region_plan.region_id
        planned_region_bytes = region_plan.usable_bytes
        host_total_bytes = detect_host_memory_bytes()

    challenge_policy = session_plan.challenge_policy.to_cbor_object()
    cleanup_policy = session_plan.cleanup_policy.to_cbor_object()
    deadline_policy = session_plan.deadline_policy.to_cbor_object()
    result.session_plan_root = session_plan.plan_root_hex
    result.challenge_policy = dict(challenge_policy)
    result.cleanup_policy = {key: bool(value) for key, value in cleanup_policy.items()}
    result.notes.append(f"Host-only session over {unit_count} canonical PoRep unit(s).")
    if retain_session:
        result.notes.append("Session will be retained for explicit rechallenge until expiry or later cleanup.")

    tracker.start("region_leasing")
    lease = create_host_lease(
        session_id=result.session_id,
        region_id=region_id,
        usable_bytes=planned_region_bytes,
        cleanup_policy=CleanupPolicy(
            zeroize=bool(cleanup_policy["zeroize"]),
            verify_zeroization=bool(cleanup_policy["verify_zeroization"]),
        ),
        lease_duration_ms=int(deadline_policy["session_timeout_ms"]),
    )
    tracker.stop("region_leasing")
    tracker.values["allocation"] = tracker.values["region_leasing"]

    socket_path = str(sessions_root() / f"{result.session_id}.grpc.sock")
    process = None
    lease_closed = False
    remote_cleanup_ran = False
    try:
        tracker.start("discover")
        process = start_ephemeral_prover_server(socket_path=socket_path)
        tracker.stop("discover")

        plan_session(socket_path, session_plan)
        lease_regions(socket_path, result.session_id, [lease.record])

        tracker.start("data_generation")
        artifacts_by_region = generate_inner_porep(socket_path, result.session_id)
        commitments, worker_timings = materialize_region_payloads(socket_path, result.session_id)
        tracker.stop("data_generation")
        tracker.values["object_serialization"] = int(worker_timings.get("object_serialization", 0))
        tracker.values["copy_to_host"] = int(worker_timings.get("copy_to_host", 0))
        tracker.values["outer_tree_build"] = int(worker_timings.get("outer_tree_build", 0))
        commit_regions(socket_path, result.session_id)

        region_manifest, region_manifest_root = commitments[region_id]
        payload = lease.read()

        session_manifest = SessionManifest(
            session_id=result.session_id,
            nonce=result.session_nonce,
            profile_name=session_plan.profile_name,
            payload_profile=session_plan.porep_unit_profile,
            leaf_size=session_plan.challenge_leaf_size,
            deadline_policy=dict(deadline_policy),
            challenge_policy=dict(challenge_policy),
            cleanup_policy=dict(cleanup_policy),
            region_manifests=(region_manifest,),
        )

        verifier = VendoredFilecoinReference()
        tracker.start("inner_verify")
        artifacts = artifacts_by_region[region_id]
        result.cpu_fallback_detected, result.cpu_fallback_events = summarize_cpu_fallbacks(artifacts)
        if result.cpu_fallback_detected:
            result.notes.append("CPU fallback detected during inner proof generation.")
        result.inner_filecoin_verified = all(verifier.verify(artifact) for artifact in artifacts)
        tracker.stop("inner_verify")
        for artifact in artifacts:
            for key, value in artifact.inner_timings_ms.items():
                if key in tracker.values:
                    tracker.values[key] += int(value)
        verify_inner_proofs(socket_path, result.session_id)

        result.challenge_count = challenge_count_for_policy(
            total_leaves=region_manifest.payload_length_bytes // result.challenge_leaf_size,
            epsilon=float(challenge_policy["epsilon"]),
            lambda_bits=int(challenge_policy["lambda_bits"]),
            max_challenges=int(challenge_policy["max_challenges"]),
        )
        challenge_indices = sample_leaf_indices(
            region_manifest.payload_length_bytes // result.challenge_leaf_size,
            result.challenge_count,
            seed=session_manifest.manifest_root_hex,
        )
        opening_payloads, result.response_ms = challenge_outer(
            socket_path,
            session_id=result.session_id,
            region_id=region_id,
            session_manifest_root=session_manifest.manifest_root_hex,
            challenge_indices=challenge_indices,
        )
        tracker.values["challenge_response"] = result.response_ms
        openings = [decode_opening_payload(payload) for payload in opening_payloads]

        tracker.start("outer_verify")
        result.outer_pose_verified = (
            region_manifest_matches_payload(
                region_manifest,
                payload=payload,
                session_nonce=result.session_nonce,
                session_plan_root=session_plan.plan_root_hex,
            )
            and response_within_deadline(result.response_ms, result.deadline_ms)
            and verify_outer_challenge_response(
                expected_region_id=region_id,
                expected_session_manifest_root=session_manifest.manifest_root_hex,
                expected_indices=challenge_indices,
                root=bytes.fromhex(region_manifest.merkle_root_hex),
                leaf_size=profile.leaf_size,
                openings=openings,
            )
        )
        tracker.stop("outer_verify")
        verify_outer(socket_path, result.session_id)

        result.session_manifest_root = session_manifest.manifest_root_hex
        result.host_total_bytes = host_total_bytes
        result.host_usable_bytes = region_manifest.usable_bytes
        result.host_covered_bytes = region_manifest.payload_length_bytes
        result.real_porep_bytes = region_manifest.real_porep_bytes
        result.tail_filler_bytes = region_manifest.tail_filler_bytes
        result.real_porep_ratio = region_manifest.real_porep_ratio
        result.coverage_fraction = (
            result.host_covered_bytes / result.host_usable_bytes
            if result.host_usable_bytes
            else 0.0
        )
        result.region_roots = {region_id: region_manifest.merkle_root_hex}
        result.region_manifest_roots = {region_id: region_manifest_root}
        result.region_payload_bytes_by_region = {region_id: region_manifest.payload_length_bytes}
        result.challenge_indices_by_region = {region_id: challenge_indices}

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
            socket_path,
            session_id=result.session_id,
            verdict=result.verdict,
            success=result.success,
            retain_session=retain_session and result.success,
        )

        tracker.start("cleanup")
        if retain_session and result.success:
            result.cleanup_status = "RETAINED_FOR_RECHALLENGE"
            result.resident_socket_path = socket_path
            result.resident_process_id = 0 if process is None else process.pid
            result.lease_expiry = lease.record.lease_expiry
            lease.close()
            lease_closed = True
            resident_record = ResidentSessionRecord(
                session_id=result.session_id,
                profile_name=result.profile_name,
                session_nonce=result.session_nonce,
                session_plan_root=result.session_plan_root,
                session_manifest_root=result.session_manifest_root,
                region_id=region_id,
                region_root_hex=region_manifest.merkle_root_hex,
                region_manifest_root=region_manifest_root,
                challenge_leaf_size=result.challenge_leaf_size,
                challenge_policy=dict(challenge_policy),
                deadline_ms=result.deadline_ms,
                cleanup_policy={key: bool(value) for key, value in cleanup_policy.items()},
                host_total_bytes=result.host_total_bytes,
                host_usable_bytes=result.host_usable_bytes,
                host_covered_bytes=result.host_covered_bytes,
                real_porep_bytes=result.real_porep_bytes,
                tail_filler_bytes=result.tail_filler_bytes,
                real_porep_ratio=result.real_porep_ratio,
                coverage_fraction=result.coverage_fraction,
                inner_filecoin_verified=result.inner_filecoin_verified,
                socket_path=socket_path,
                process_id=result.resident_process_id,
                lease_expiry=result.lease_expiry,
            )
            session_record_path = write_resident_session(resident_record)
            result.notes.append(f"Resident session metadata written to {session_record_path}")
        else:
            result.cleanup_status = cleanup_session(socket_path, result.session_id)
            remote_cleanup_ran = True
            lease.close()
            lease_closed = True
        tracker.stop("cleanup")
        return result
    except ResourceFailure as error:
        result.verdict = "RESOURCE_FAILURE"
        result.notes.append(str(error))
        return result
    except ProtocolError as error:
        result.verdict = "PROTOCOL_ERROR"
        result.notes.append(str(error))
        return result
    finally:
        if not lease_closed:
            tracker.start("cleanup")
            try:
                if remote_cleanup_ran:
                    lease.close()
                else:
                    result.cleanup_status = release_host_lease(
                        lease,
                        zeroize=bool(cleanup_policy["zeroize"]),
                        verify_zeroization=bool(cleanup_policy["verify_zeroization"]),
                    )
                lease_closed = True
            except ResourceFailure as error:
                result.cleanup_status = "CLEANUP_FAILED"
                result.notes.append(str(error))
                if result.verdict == "SUCCESS":
                    result.verdict = "CLEANUP_FAILURE"
                    result.success = False
            tracker.stop("cleanup")
        if process is not None and not (retain_session and result.success):
            process.terminate()
            process.wait(timeout=5)
        result.timings_ms = tracker.values
        result.timings_ms["total"] = int((perf_counter() - total_started) * 1000)
