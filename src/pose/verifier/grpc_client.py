from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from time import monotonic, perf_counter_ns, sleep

import grpc

from pose.common.errors import ProtocolError, ResourceFailure
from pose.common.sandbox import (
    ProverSandboxPolicy,
    sandboxed_child_environment,
    sandboxed_command,
)
from pose.protocol.grpc_codec import GRPC_PROTOCOL_VERSION, lease_record_to_proto, session_plan_to_proto
from pose.protocol.messages import LeaseRecord, SessionPlan
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


def seed_session(socket_path: str, session_id: str) -> None:
    channel, stub = _channel_stub(socket_path)
    try:
        _call(
            stub.SeedSession,
            session_pb2.SeedSessionRequest(
                protocol_version=GRPC_PROTOCOL_VERSION,
                session_id=session_id,
            ),
        )
    finally:
        channel.close()


def materialize_labels(
    socket_path: str,
    session_id: str,
) -> tuple[dict[str, object], dict[str, int]]:
    channel, stub = _channel_stub(socket_path)
    try:
        response = _call(
            stub.MaterializeLabels,
            session_pb2.MaterializeLabelsRequest(
                protocol_version=GRPC_PROTOCOL_VERSION,
                session_id=session_id,
            ),
        )
        return (
            {
                "graph_descriptor_digest": str(response.graph_descriptor_digest),
                "scratch_peak_bytes": int(response.scratch_peak_bytes),
                "regions": {
                    item.region_id: {
                        "covered_bytes": int(item.covered_bytes),
                        "slack_bytes": int(item.slack_bytes),
                        "declared_stage_copy_bytes": int(item.declared_stage_copy_bytes),
                    }
                    for item in response.regions
                },
            },
            {key: int(value) for key, value in response.timings_ms.items()},
        )
    finally:
        channel.close()


def prepare_fast_phase(socket_path: str, session_id: str) -> None:
    channel, stub = _channel_stub(socket_path)
    try:
        _call(
            stub.PrepareFastPhase,
            session_pb2.PrepareFastPhaseRequest(
                protocol_version=GRPC_PROTOCOL_VERSION,
                session_id=session_id,
            ),
        )
    finally:
        channel.close()


class FastPhaseClient:
    def __init__(self, socket_path: str, *, ready_timeout_seconds: float = 5.0) -> None:
        self._socket_path = socket_path
        self._ready_timeout_seconds = ready_timeout_seconds
        self._channel = None
        self._stub = None

    def __enter__(self) -> "FastPhaseClient":
        channel, stub = _channel_stub(self._socket_path)
        grpc.channel_ready_future(channel).result(timeout=self._ready_timeout_seconds)
        self._channel = channel
        self._stub = stub
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        del exc_type, exc, tb
        self.close()

    def close(self) -> None:
        if self._channel is not None:
            self._channel.close()
            self._channel = None
            self._stub = None

    def run_round(
        self,
        *,
        session_id: str,
        round_index: int,
        challenge_index: int,
    ) -> dict[str, object]:
        if self._stub is None:
            raise ProtocolError("FastPhaseClient must be entered before use.")
        started = perf_counter_ns()
        response = _call(
            self._stub.RunFastPhase,
            session_pb2.RunFastPhaseRequest(
                rounds=[
                    session_pb2.FastRoundChallenge(
                        protocol_version=GRPC_PROTOCOL_VERSION,
                        session_id=session_id,
                        round_index=round_index,
                        challenge_index=int(challenge_index),
                    )
                ],
            ),
        )
        round_trip_us = max(0, int((perf_counter_ns() - started) / 1000))
        if len(response.responses) != 1:
            raise ProtocolError(
                "RunFastPhase must return exactly one response for a single-round request."
            )
        item = response.responses[0]
        return {
            "region_id": item.region_id,
            "challenge_index": int(item.challenge_index),
            "label_bytes": bytes(item.label_bytes),
            "round_trip_us": round_trip_us,
            "prover_lookup_round_trip_us": int(item.round_trip_us),
        }


def run_fast_phase_round(
    socket_path: str,
    *,
    session_id: str,
    round_index: int,
    challenge_index: int,
) -> dict[str, object]:
    with FastPhaseClient(socket_path) as client:
        return client.run_round(
            session_id=session_id,
            round_index=round_index,
            challenge_index=challenge_index,
        )


def run_fast_phase(
    socket_path: str,
    *,
    session_id: str,
    challenge_indices: list[int],
) -> list[dict[str, object]]:
    with FastPhaseClient(socket_path) as client:
        return [
            client.run_round(
                session_id=session_id,
                round_index=round_index,
                challenge_index=int(challenge_index),
            )
            for round_index, challenge_index in enumerate(challenge_indices)
        ]

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
    prover_sandbox: ProverSandboxPolicy | None = None,
) -> subprocess.Popen[str]:
    repo_root = Path(__file__).resolve().parents[3]
    sandbox_policy = prover_sandbox or ProverSandboxPolicy()
    env = dict(os.environ)
    env.setdefault("PYTHONPATH", str(repo_root / "src"))
    command = [
        sys.executable,
        "-m",
        "pose.cli.main",
        "prover",
        "grpc-serve",
        "--socket-path",
        socket_path,
    ]
    if sandbox_policy.mode == "process_budget_dev":
        if sandbox_policy.process_memory_max_bytes <= 0:
            raise ProtocolError(
                "process_budget_dev prover sandbox mode requires a positive process_memory_max_bytes"
            )
        env = sandboxed_child_environment(
            env,
            require_no_visible_gpus=sandbox_policy.require_no_visible_gpus,
        )
        command = sandboxed_command(
            command,
            process_memory_max_bytes=sandbox_policy.process_memory_max_bytes,
            memlock_max_bytes=sandbox_policy.memlock_max_bytes,
            file_size_max_bytes=sandbox_policy.file_size_max_bytes,
        )
    process = subprocess.Popen(
        command,
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
        except (OSError, ProtocolError, ResourceFailure):
            sleep(0.05)
    process.terminate()
    raise ResourceFailure(f"Timed out waiting for prover gRPC server at {socket_path}")
