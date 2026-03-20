from __future__ import annotations

import json
import os
import secrets
import subprocess
import sys
from time import perf_counter

from pose.benchmarks.profiles import BenchmarkProfile
from pose.common.errors import ProtocolError, ResourceFailure
from pose.common.merkle import commit_payload
from pose.common.timing import TimingTracker
from pose.filecoin.porep_unit import build_porep_unit_from_seal_artifact
from pose.filecoin.reference import SealRequest, VendoredFilecoinReference
from pose.protocol.messages import ChallengePolicy, CleanupPolicy, DeadlinePolicy, RegionPlan, SessionPlan
from pose.protocol.region_payloads import (
    RegionManifest,
    SessionManifest,
    build_region_manifest,
    region_manifest_matches_payload,
)
from pose.protocol.result_schema import SessionResult, bootstrap_result
from pose.prover.object_builder import build_region_payload
from pose.verifier.outer import decode_opening_payload, verify_outer_challenge_response
from pose.verifier.challenge import challenge_count_for_policy, sample_leaf_indices
from pose.verifier.deadlines import response_within_deadline
from pose.verifier.leasing import create_host_lease, release_host_lease
from pose.prover.host_worker import WORKER_PROTOCOL_VERSION

BASE_SECTOR_ID = 4242


def _derive_hex(*parts: object) -> str:
    from pose.common.hashing import sha256_bytes

    seed = "|".join(str(part) for part in parts).encode("utf-8")
    output = bytearray()
    counter = 0
    while len(output) < 32:
        output.extend(sha256_bytes(seed + counter.to_bytes(4, "big")))
        counter += 1
    return bytes(output[:32]).hex()


def _seal_request_for_unit(session_id: str, region_id: str, unit_index: int) -> SealRequest:
    return SealRequest(
        prover_id_hex=_derive_hex("prover", session_id, region_id),
        sector_id=BASE_SECTOR_ID + unit_index,
        ticket_hex=_derive_hex("ticket", session_id, region_id, unit_index),
        seed_hex=_derive_hex("seed", session_id, region_id, unit_index),
    )


def _configured_unit_count(requested_unit_count: int | None) -> int:
    if requested_unit_count is not None:
        return requested_unit_count
    return max(1, int(os.environ.get("POSE_HOST_MAX_UNITS", "1")))


def _planned_region_bytes(
    profile: BenchmarkProfile,
    *,
    unit_count: int,
    requested_host_bytes: int | None,
) -> int:
    planned = unit_count * profile.leaf_size
    if requested_host_bytes is None:
        return planned
    if requested_host_bytes <= 0:
        raise ResourceFailure(f"Host lease size must be positive: {requested_host_bytes}")
    if requested_host_bytes % profile.leaf_size != 0:
        raise ResourceFailure(
            f"Host lease size {requested_host_bytes} must be a multiple of leaf size {profile.leaf_size}"
        )
    if requested_host_bytes < planned:
        raise ResourceFailure(
            f"Host lease size {requested_host_bytes} is too small for {unit_count} "
            f"unit(s) of size {profile.leaf_size}"
        )
    return requested_host_bytes


def _run_worker(request: dict[str, object], *, pass_fds: tuple[int, ...]) -> dict[str, object]:
    completed = subprocess.run(
        [sys.executable, "-m", "pose.cli.main", "prover", "host-session-worker"],
        input=json.dumps(request),
        text=True,
        capture_output=True,
        check=False,
        pass_fds=pass_fds,
    )
    if completed.returncode != 0:
        raise ResourceFailure(
            "Host worker failed with exit code "
            f"{completed.returncode}: {completed.stderr.strip() or completed.stdout.strip()}"
        )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise ResourceFailure(f"Host worker returned invalid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ResourceFailure("Host worker response must decode to a JSON object")
    return payload


def _materialize_in_subprocess(
    *,
    session_id: str,
    session_nonce: str,
    session_plan_root: str,
    region_id: str,
    storage_profile: str,
    leaf_size: int,
    unit_count: int,
    lease_fd: int,
    usable_bytes: int,
) -> tuple[list[dict[str, object]], dict[str, object], dict[str, object]]:
    response = _run_worker(
        {
            "protocol_version": WORKER_PROTOCOL_VERSION,
            "operation": "materialize",
            "session_id": session_id,
            "session_nonce": session_nonce,
            "session_plan_root": session_plan_root,
            "region_id": region_id,
            "storage_profile": storage_profile,
            "leaf_size": leaf_size,
            "unit_count": unit_count,
            "lease_fd": lease_fd,
            "usable_bytes": usable_bytes,
        },
        pass_fds=(lease_fd,),
    )
    return (
        list(response["unit_artifacts"]),  # type: ignore[arg-type]
        dict(response["region_manifest"]),  # type: ignore[arg-type]
        dict(response["timings_ms"]),  # type: ignore[arg-type]
    )


def _open_in_subprocess(
    *,
    region_id: str,
    session_manifest_root: str,
    challenge_indices: list[int],
    lease_fd: int,
    usable_bytes: int,
    leaf_size: int,
) -> tuple[list[dict[str, object]], int]:
    response = _run_worker(
        {
            "protocol_version": WORKER_PROTOCOL_VERSION,
            "operation": "open",
            "region_id": region_id,
            "session_manifest_root": session_manifest_root,
            "challenge_indices": challenge_indices,
            "lease_fd": lease_fd,
            "usable_bytes": usable_bytes,
            "leaf_size": leaf_size,
        },
        pass_fds=(lease_fd,),
    )
    return (
        list(response["openings"]),  # type: ignore[arg-type]
        int(response["response_ms"]),
    )


def _materialize_locally(
    *,
    reference: VendoredFilecoinReference,
    session_id: str,
    session_nonce: str,
    session_plan_root: str,
    region_id: str,
    storage_profile: str,
    leaf_size: int,
    unit_count: int,
    lease,
) -> tuple[list[object], object, bytes, int]:
    artifacts = []
    units = []
    payload_parts: list[bytes] = []
    object_serialization_ms = 0
    for unit_index in range(unit_count):
        artifact = reference.seal(_seal_request_for_unit(session_id, region_id, unit_index))
        serialization_started = perf_counter()
        unit = build_porep_unit_from_seal_artifact(
            artifact,
            storage_profile=storage_profile,
            leaf_alignment_bytes=leaf_size,
        )
        object_serialization_ms += int((perf_counter() - serialization_started) * 1000)
        if len(unit.serialized_bytes) != leaf_size:
            raise ResourceFailure(
                "Current host runner expects each minimal PoRep unit to occupy exactly "
                f"one leaf: got {len(unit.serialized_bytes)} bytes for leaf size {leaf_size}"
            )
        artifacts.append(artifact)
        units.append(unit)
        payload_parts.append(unit.serialized_bytes)

    real_payload = b"".join(payload_parts)
    tail_filler_bytes = lease.record.usable_bytes - len(real_payload)
    if tail_filler_bytes < 0:
        raise ResourceFailure(
            f"Materialized payload length {len(real_payload)} exceeds lease size "
            f"{lease.record.usable_bytes}"
        )
    if tail_filler_bytes > min(leaf_size, 1024 * 1024):
        raise ResourceFailure(
            f"Tail filler requirement {tail_filler_bytes} exceeds the allowed limit "
            f"for leaf size {leaf_size}"
        )
    payload = build_region_payload(
        [unit.serialized_bytes for unit in units],
        session_nonce=session_nonce,
        region_id=region_id,
        session_plan_root=session_plan_root,
        tail_filler_bytes=tail_filler_bytes,
    )
    lease.write(payload)
    commitment = commit_payload(payload, leaf_size)
    region_manifest = build_region_manifest(
        region_id=region_id,
        region_type="host",
        usable_bytes=lease.record.usable_bytes,
        leaf_size=leaf_size,
        payload=payload,
        merkle_root_hex=commitment.root_hex,
        units=tuple(units),
        tail_filler_bytes=tail_filler_bytes,
    )
    return artifacts, region_manifest, payload, object_serialization_ms


def _open_locally(*, region_id: str, session_manifest_root: str, challenge_indices: list[int], lease, leaf_size: int) -> tuple[list[dict[str, object]], int]:
    payload = lease.read()
    commitment = commit_payload(payload, leaf_size)
    started = perf_counter()
    openings = []
    for index in challenge_indices:
        opening = commitment.opening(index, lease.read_leaf(index, leaf_size))
        openings.append(
            {
                "region_id": region_id,
                "session_manifest_root": session_manifest_root,
                "leaf_index": index,
                "leaf_hex": opening.leaf.hex(),
                "sibling_hashes_hex": [value.hex() for value in opening.sibling_hashes],
            }
        )
    return openings, int((perf_counter() - started) * 1000)


def run_host_session(
    profile: BenchmarkProfile,
    *,
    reference: VendoredFilecoinReference | None = None,
    requested_unit_count: int | None = None,
    requested_host_bytes: int | None = None,
) -> SessionResult:
    tracker = TimingTracker()
    total_started = perf_counter()
    tracker.start("discover")
    tracker.stop("discover")

    result = bootstrap_result(profile_name=profile.name)
    result.challenge_leaf_size = profile.leaf_size
    result.deadline_ms = int(profile.deadline_policy["response_deadline_ms"])
    result.session_nonce = secrets.token_hex(16)
    unit_count = _configured_unit_count(requested_unit_count)
    result.notes.append(f"Host-only session over {unit_count} canonical PoRep unit(s).")

    if not profile.target_devices.get("host", False) or profile.target_devices.get("gpus"):
        result.verdict = "PROTOCOL_ERROR"
        result.notes.append("Only host-only profiles are supported by the current host session runner.")
        result.timings_ms["total"] = int((perf_counter() - total_started) * 1000)
        return result

    if profile.porep_unit_profile != "minimal":
        result.verdict = "RESOURCE_FAILURE"
        result.notes.append(
            "Profile requires replica/full-cache blobs, but the current bridge only "
            "materializes the minimal PoRep unit profile."
        )
        result.timings_ms["total"] = int((perf_counter() - total_started) * 1000)
        return result

    verifier = reference or VendoredFilecoinReference()
    cleanup_status = "NOT_RUN"
    lease = None
    region_id = "host-0"
    planned_region_bytes = _planned_region_bytes(
        profile,
        unit_count=unit_count,
        requested_host_bytes=requested_host_bytes,
    )
    session_plan = SessionPlan(
        session_id=result.session_id,
        nonce=result.session_nonce,
        profile_name=profile.name,
        porep_unit_profile=profile.porep_unit_profile,
        challenge_leaf_size=profile.leaf_size,
        challenge_policy=ChallengePolicy(**profile.challenge_policy),
        deadline_policy=DeadlinePolicy(**profile.deadline_policy),
        cleanup_policy=CleanupPolicy(**profile.cleanup_policy),
        regions=[RegionPlan(region_id=region_id, region_type="host", usable_bytes=planned_region_bytes)],
    )
    result.session_plan_root = session_plan.plan_root_hex
    try:
        tracker.start("region_leasing")
        lease = create_host_lease(
            session_id=result.session_id,
            region_id=region_id,
            usable_bytes=planned_region_bytes,
            cleanup_policy=profile_cleanup_policy(profile),
            lease_duration_ms=int(profile.deadline_policy["session_timeout_ms"]),
        )
        tracker.stop("region_leasing")
        tracker.values["allocation"] = tracker.values["region_leasing"]

        tracker.start("data_generation")
        tracker.start("object_serialization")
        tracker.start("copy_to_host")
        if reference is None:
            artifact_payloads, region_manifest_payload, worker_timings = _materialize_in_subprocess(
                session_id=result.session_id,
                session_nonce=result.session_nonce,
                session_plan_root=session_plan.plan_root_hex,
                region_id=region_id,
                storage_profile=profile.porep_unit_profile,
                leaf_size=profile.leaf_size,
                unit_count=unit_count,
                lease_fd=lease.fd,
                usable_bytes=lease.record.usable_bytes,
            )
            artifacts = artifact_payloads
            tracker.stop("copy_to_host")
            tracker.stop("object_serialization")
            tracker.stop("data_generation")
            tracker.values["copy_to_host"] = int(worker_timings["copy_to_host"])
            tracker.values["object_serialization"] = int(worker_timings["object_serialization"])
            tracker.values["data_generation"] = int(worker_timings["materialize_total"])
            for payload in artifacts:
                inner_timings = payload.get("inner_timings_ms", {})
                if isinstance(inner_timings, dict):
                    for key, value in inner_timings.items():
                        if key in tracker.values:
                            tracker.values[key] += int(value)
        else:
            artifacts, region_manifest, payload, local_object_serialization_ms = _materialize_locally(
                reference=verifier,
                session_id=result.session_id,
                session_nonce=result.session_nonce,
                session_plan_root=session_plan.plan_root_hex,
                region_id=region_id,
                storage_profile=profile.porep_unit_profile,
                leaf_size=profile.leaf_size,
                unit_count=unit_count,
                lease=lease,
            )
            tracker.stop("copy_to_host")
            tracker.stop("object_serialization")
            tracker.stop("data_generation")
            tracker.values["object_serialization"] = local_object_serialization_ms
            region_manifest_payload = {
                **region_manifest.to_cbor_object(),
                "manifest_root_hex": region_manifest.manifest_root_hex,
            }
            for artifact in artifacts:
                for key, value in artifact.inner_timings_ms.items():
                    if key in tracker.values:
                        tracker.values[key] += int(value)

        payload = lease.read()
        commitment = commit_payload(payload, profile.leaf_size)
        tracker.start("outer_tree_build")
        region_manifest = RegionManifest.from_cbor_object(region_manifest_payload)
        session_manifest = SessionManifest(
            session_id=result.session_id,
            nonce=result.session_nonce,
            profile_name=profile.name,
            payload_profile=profile.porep_unit_profile,
            leaf_size=profile.leaf_size,
            deadline_ms=result.deadline_ms,
            challenge_policy=dict(profile.challenge_policy),
            cleanup_policy=dict(profile.cleanup_policy),
            region_manifests=(region_manifest,),
        )
        tracker.stop("outer_tree_build")

        tracker.start("inner_verify")
        if reference is None:
            from pose.filecoin.reference import SealArtifact

            result.inner_filecoin_verified = all(
                verifier.verify(SealArtifact(**payload)) for payload in artifacts
            )
        else:
            result.inner_filecoin_verified = all(verifier.verify(artifact) for artifact in artifacts)
        tracker.stop("inner_verify")

        result.challenge_count = challenge_count_for_policy(
            total_leaves=commitment.leaf_count,
            epsilon=float(profile.challenge_policy["epsilon"]),
            lambda_bits=int(profile.challenge_policy["lambda_bits"]),
            max_challenges=int(profile.challenge_policy["max_challenges"]),
        )
        challenge_indices = sample_leaf_indices(
            commitment.leaf_count,
            result.challenge_count,
            seed=session_manifest.manifest_root_hex,
        )

        if reference is None:
            opening_payloads, result.response_ms = _open_in_subprocess(
                region_id=region_id,
                session_manifest_root=session_manifest.manifest_root_hex,
                challenge_indices=challenge_indices,
                lease_fd=lease.fd,
                usable_bytes=lease.record.usable_bytes,
                leaf_size=profile.leaf_size,
            )
        else:
            opening_payloads, result.response_ms = _open_locally(
                region_id=region_id,
                session_manifest_root=session_manifest.manifest_root_hex,
                challenge_indices=challenge_indices,
                lease=lease,
                leaf_size=profile.leaf_size,
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
            and response_within_deadline(
                result.response_ms, result.deadline_ms
            )
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

        result.session_manifest_root = session_manifest.manifest_root_hex
        result.host_total_bytes = region_manifest.usable_bytes
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
        result.region_manifest_roots = {
            region_id: str(region_manifest_payload["manifest_root_hex"])
        }
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

        tracker.start("cleanup")
        try:
            cleanup_status = release_host_lease(
                lease,
                zeroize=bool(profile.cleanup_policy["zeroize"]),
                verify_zeroization=bool(profile.cleanup_policy["verify_zeroization"]),
            )
            lease = None
        except ResourceFailure as error:
            cleanup_status = "CLEANUP_FAILED"
            lease = None
            result.verdict = "CLEANUP_FAILURE"
            result.success = False
            result.notes.append(str(error))
        tracker.stop("cleanup")

        result.cleanup_status = cleanup_status
        return result
    except ResourceFailure as error:
        result.verdict = "RESOURCE_FAILURE"
        result.notes.append(str(error))
        result.cleanup_status = cleanup_status
        return result
    except ProtocolError as error:
        result.verdict = "PROTOCOL_ERROR"
        result.notes.append(str(error))
        result.cleanup_status = cleanup_status
        return result
    finally:
        if lease is not None:
            try:
                result.cleanup_status = release_host_lease(
                    lease,
                    zeroize=bool(profile.cleanup_policy["zeroize"]),
                    verify_zeroization=bool(profile.cleanup_policy["verify_zeroization"]),
                )
            except ResourceFailure as error:
                result.cleanup_status = "CLEANUP_FAILED"
                result.notes.append(str(error))
                if result.verdict == "SUCCESS":
                    result.verdict = "CLEANUP_FAILURE"
                    result.success = False
        result.timings_ms = tracker.values
        result.timings_ms["total"] = int((perf_counter() - total_started) * 1000)


def profile_cleanup_policy(profile: BenchmarkProfile):
    return CleanupPolicy(
        zeroize=bool(profile.cleanup_policy["zeroize"]),
        verify_zeroization=bool(profile.cleanup_policy["verify_zeroization"]),
    )
