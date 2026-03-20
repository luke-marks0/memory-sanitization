from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from time import monotonic, sleep

import grpc

from pose.common.errors import ProtocolError, ResourceFailure
from pose.filecoin.reference import SealArtifact
from pose.protocol.grpc_codec import (
    GRPC_PROTOCOL_VERSION,
    artifact_from_proto,
    lease_record_to_proto,
    region_manifest_from_json,
    session_plan_to_proto,
)
from pose.protocol.messages import LeaseRecord, SessionPlan
from pose.protocol.region_payloads import RegionManifest
from pose.v1 import session_pb2, session_pb2_grpc


def _unix_target(socket_path: str) -> str:
    return f"unix:{socket_path}"


def _rpc_error(error: grpc.RpcError) -> Exception:
    if error.code() == grpc.StatusCode.FAILED_PRECONDITION:
        return ResourceFailure(error.details() or "gRPC precondition failed")
    return ProtocolError(error.details() or f"gRPC request failed: {error.code().name}")


def _call(stub_call, request):
    try:
        return stub_call(request)
    except grpc.RpcError as error:  # pragma: no cover - exercised in higher-level tests
        raise _rpc_error(error) from error


def _channel_stub(socket_path: str):
    channel = grpc.insecure_channel(_unix_target(socket_path))
    return channel, session_pb2_grpc.PoseSessionServiceStub(channel)


def discover(socket_path: str) -> dict[str, object]:
    channel, stub = _channel_stub(socket_path)
    try:
        response = _call(
            stub.Discover,
            session_pb2.DiscoverRequest(protocol_version=GRPC_PROTOCOL_VERSION),
        )
        return {
            "protocol_version": response.protocol_version,
            "capabilities": list(response.capabilities),
        }
    finally:
        channel.close()


def plan_session(socket_path: str, plan: SessionPlan) -> None:
    channel, stub = _channel_stub(socket_path)
    try:
        _call(
            stub.PlanSession,
            session_pb2.PlanSessionRequest(
                protocol_version=GRPC_PROTOCOL_VERSION,
                plan=session_plan_to_proto(plan),
            ),
        )
    finally:
        channel.close()


def lease_regions(socket_path: str, session_id: str, leases: list[LeaseRecord]) -> None:
    channel, stub = _channel_stub(socket_path)
    try:
        _call(
            stub.LeaseRegions,
            session_pb2.LeaseRegionsRequest(
                protocol_version=GRPC_PROTOCOL_VERSION,
                session_id=session_id,
                leases=[lease_record_to_proto(lease) for lease in leases],
            ),
        )
    finally:
        channel.close()


def generate_inner_porep(
    socket_path: str,
    session_id: str,
) -> dict[str, list[SealArtifact]]:
    channel, stub = _channel_stub(socket_path)
    try:
        response = _call(
            stub.GenerateInnerPoRep,
            session_pb2.GenerateInnerPoRepRequest(
                protocol_version=GRPC_PROTOCOL_VERSION,
                session_id=session_id,
            ),
        )
        by_region: dict[str, list[SealArtifact]] = {}
        for artifact in response.artifacts:
            region_id, decoded = artifact_from_proto(artifact)
            by_region.setdefault(region_id, []).append(decoded)
        return by_region
    finally:
        channel.close()


def materialize_region_payloads(
    socket_path: str,
    session_id: str,
) -> tuple[dict[str, tuple[RegionManifest, str]], dict[str, int]]:
    channel, stub = _channel_stub(socket_path)
    try:
        response = _call(
            stub.MaterializeRegionPayloads,
            session_pb2.MaterializeRegionPayloadsRequest(
                protocol_version=GRPC_PROTOCOL_VERSION,
                session_id=session_id,
            ),
        )
        commitments = {
            commitment.region_id: region_manifest_from_json(commitment.region_manifest_json)
            for commitment in response.commitments
        }
        return commitments, {key: int(value) for key, value in response.timings_ms.items()}
    finally:
        channel.close()


def commit_regions(socket_path: str, session_id: str) -> None:
    channel, stub = _channel_stub(socket_path)
    try:
        _call(
            stub.CommitRegions,
            session_pb2.CommitRegionsRequest(
                protocol_version=GRPC_PROTOCOL_VERSION,
                session_id=session_id,
            ),
        )
    finally:
        channel.close()


def verify_inner_proofs(socket_path: str, session_id: str) -> None:
    channel, stub = _channel_stub(socket_path)
    try:
        _call(
            stub.VerifyInnerProofs,
            session_pb2.VerifyInnerProofsRequest(
                protocol_version=GRPC_PROTOCOL_VERSION,
                session_id=session_id,
            ),
        )
    finally:
        channel.close()


def challenge_outer(
    socket_path: str,
    *,
    session_id: str,
    region_id: str,
    session_manifest_root: str,
    challenge_indices: list[int],
) -> tuple[list[dict[str, object]], int]:
    channel, stub = _channel_stub(socket_path)
    try:
        response = _call(
            stub.ChallengeOuter,
            session_pb2.ChallengeOuterRequest(
                protocol_version=GRPC_PROTOCOL_VERSION,
                session_id=session_id,
                challenges=[
                    session_pb2.OuterChallenge(
                        region_id=region_id,
                        leaf_indices=challenge_indices,
                        session_manifest_root=session_manifest_root,
                    )
                ],
            ),
        )
        openings = [
            {
                "region_id": opening.region_id,
                "session_manifest_root": opening.session_manifest_root,
                "leaf_index": int(opening.leaf_index),
                "leaf_hex": opening.leaf_bytes.hex(),
                "sibling_hashes_hex": [value.hex() for value in opening.auth_path],
            }
            for opening in response.openings
        ]
        return openings, int(response.response_ms)
    finally:
        channel.close()


def verify_outer(socket_path: str, session_id: str) -> None:
    channel, stub = _channel_stub(socket_path)
    try:
        _call(
            stub.VerifyOuter,
            session_pb2.VerifyOuterRequest(
                protocol_version=GRPC_PROTOCOL_VERSION,
                session_id=session_id,
            ),
        )
    finally:
        channel.close()


def finalize_session(
    socket_path: str,
    *,
    session_id: str,
    verdict: str,
    success: bool,
    retain_session: bool,
) -> None:
    channel, stub = _channel_stub(socket_path)
    try:
        _call(
            stub.Finalize,
            session_pb2.FinalizeRequest(
                protocol_version=GRPC_PROTOCOL_VERSION,
                session_id=session_id,
                verdict=verdict,
                success=success,
                retain_session=retain_session,
            ),
        )
    finally:
        channel.close()


def cleanup_session(socket_path: str, session_id: str) -> str:
    channel, stub = _channel_stub(socket_path)
    try:
        response = _call(
            stub.Cleanup,
            session_pb2.CleanupRequest(
                protocol_version=GRPC_PROTOCOL_VERSION,
                session_id=session_id,
            ),
        )
        return response.cleanup_status
    finally:
        channel.close()


def start_ephemeral_prover_server(
    *,
    socket_path: str,
    timeout_seconds: float = 30.0,
) -> subprocess.Popen[str]:
    repo_root = Path(__file__).resolve().parents[3]
    env = dict(os.environ)
    env.setdefault("PYTHONPATH", str(repo_root / "src"))
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "pose.cli.main",
            "prover",
            "grpc-serve",
            "--socket-path",
            socket_path,
        ],
        cwd=repo_root,
        env=env,
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    deadline = monotonic() + timeout_seconds
    while monotonic() < deadline:
        if process.poll() is not None:
            raise ResourceFailure(
                f"Ephemeral prover gRPC server exited early with code {process.returncode}"
            )
        try:
            discover(socket_path)
            return process
        except (ProtocolError, ResourceFailure):
            sleep(0.1)
    process.terminate()
    process.wait(timeout=5)
    raise ResourceFailure("Timed out waiting for the ephemeral prover gRPC server to start")
