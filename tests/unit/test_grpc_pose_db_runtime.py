from __future__ import annotations

from concurrent import futures
from dataclasses import replace
from pathlib import Path
from time import sleep

import grpc
import pytest

from pose.common.host_lease import create_host_lease, release_host_lease
from pose.graphs import build_graph_descriptor, build_pose_db_graph, compute_challenge_labels
from pose.protocol.messages import ChallengePolicy, CleanupPolicy, DeadlinePolicy, RegionPlan, SessionPlan
from pose.prover.grpc_service import PoseSessionServicer
from pose.verifier.grpc_client import (
    FastPhaseClient,
    cleanup_session,
    discover,
    finalize_session,
    lease_regions,
    materialize_labels,
    plan_session,
    prepare_fast_phase,
    run_fast_phase,
    seed_session,
)
from pose.v1 import session_pb2, session_pb2_grpc


def _session_plan() -> SessionPlan:
    descriptor = build_graph_descriptor(
        label_count_m=8,
        graph_parameter_n=2,
        gamma=4,
        hash_backend="blake3-xof",
        label_width_bits=256,
    )
    return SessionPlan(
        session_id="grpc-runtime-session",
        session_seed_hex="cd" * 32,
        profile_name="dev-small",
        graph_family="pose-db-drg-v1",
        graph_parameter_n=2,
        label_count_m=8,
        gamma=4,
        label_width_bits=256,
        hash_backend="blake3-xof",
        graph_descriptor_digest=descriptor.digest,
        challenge_policy=ChallengePolicy(rounds_r=4, target_success_bound=1e-9),
        deadline_policy=DeadlinePolicy(response_deadline_us=500_000, session_timeout_ms=60_000),
        cleanup_policy=CleanupPolicy(zeroize=True, verify_zeroization=True),
        regions=[
            RegionPlan(
                region_id="host-0",
                region_type="host",
                usable_bytes=256,
                slot_count=8,
                covered_bytes=256,
                slack_bytes=0,
            )
        ],
        adversary_model="general",
        attacker_budget_bytes_assumed=4096,
        q_bound=3,
        claim_notes=["grpc-runtime-test"],
    )


def test_host_pose_db_grpc_roundtrip(tmp_path: Path) -> None:
    socket_path = tmp_path / "pose-runtime.sock"
    if socket_path.exists():
        socket_path.unlink()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    session_pb2_grpc.add_PoseSessionServiceServicer_to_server(PoseSessionServicer(), server)
    server.add_insecure_port(f"unix:{socket_path}")
    server.start()

    plan = _session_plan()
    region = plan.regions[0]
    lease = create_host_lease(
        session_id=plan.session_id,
        region_id=region.region_id,
        usable_bytes=region.usable_bytes,
        cleanup_policy=plan.cleanup_policy,
        lease_duration_ms=plan.deadline_policy.session_timeout_ms,
    )
    lease_record = replace(
        lease.record,
        slot_count=region.slot_count,
        slack_bytes=region.slack_bytes,
        gpu_device=region.gpu_device,
    )

    remote_cleaned = False
    try:
        capabilities = discover(str(socket_path))
        assert capabilities["protocol_version"] == "pose-grpc/v1"
        assert "pose-db-fast-phase" in capabilities["capabilities"]

        plan_session(str(socket_path), plan)
        lease_regions(str(socket_path), plan.session_id, [lease_record])
        seed_session(str(socket_path), plan.session_id)
        materialization, timings = materialize_labels(str(socket_path), plan.session_id)
        prepare_fast_phase(str(socket_path), plan.session_id)

        assert materialization == {
            "graph_descriptor_digest": plan.graph_descriptor_digest,
            "scratch_peak_bytes": materialization["scratch_peak_bytes"],
            "regions": {
                "host-0": {
                    "covered_bytes": 256,
                    "slack_bytes": 0,
                    "declared_stage_copy_bytes": 0,
                }
            },
        }
        assert materialization["scratch_peak_bytes"] > 0
        assert "label_generation" in timings

        challenge_indices = [0, 3, 7, 3]
        responses = run_fast_phase(
            str(socket_path),
            session_id=plan.session_id,
            challenge_indices=challenge_indices,
        )

        label_width_bytes = plan.label_width_bits // 8
        graph = build_pose_db_graph(
            label_count_m=plan.label_count_m,
            graph_parameter_n=plan.graph_parameter_n,
            gamma=plan.gamma,
            hash_backend=plan.hash_backend,
            label_width_bits=plan.label_width_bits,
        )
        expected_labels = compute_challenge_labels(
            graph,
            session_seed=plan.session_seed_hex,
            challenge_indices=challenge_indices,
        )
        assert len(responses) == len(challenge_indices)
        for response, challenge_index, expected_label in zip(
            responses,
            challenge_indices,
            expected_labels,
            strict=True,
        ):
            assert response["region_id"] == "host-0"
            assert response["challenge_index"] == challenge_index
            assert response["label_bytes"] == lease.read(
                length=label_width_bytes,
                offset=challenge_index * label_width_bytes,
            )
            assert response["label_bytes"] == expected_label
            assert response["round_trip_us"] >= 0
            assert response["prover_lookup_round_trip_us"] >= 0

        finalize_session(
            str(socket_path),
            session_id=plan.session_id,
            verdict="SUCCESS",
            success=True,
            retain_session=False,
        )
        cleanup_status = cleanup_session(str(socket_path), plan.session_id)
        remote_cleaned = True

        assert cleanup_status == "ZEROIZED_AND_VERIFIED"
        assert lease.read() == bytes(region.usable_bytes)
    finally:
        release_host_lease(
            lease,
            zeroize=not remote_cleaned and plan.cleanup_policy.zeroize,
            verify_zeroization=not remote_cleaned and plan.cleanup_policy.verify_zeroization,
        )
        server.stop(grace=None)


def test_grpc_run_fast_phase_rejects_batched_rounds(tmp_path: Path) -> None:
    socket_path = tmp_path / "pose-runtime-batched.sock"
    if socket_path.exists():
        socket_path.unlink()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    session_pb2_grpc.add_PoseSessionServiceServicer_to_server(PoseSessionServicer(), server)
    server.add_insecure_port(f"unix:{socket_path}")
    server.start()

    plan = _session_plan()
    region = plan.regions[0]
    lease = create_host_lease(
        session_id=plan.session_id,
        region_id=region.region_id,
        usable_bytes=region.usable_bytes,
        cleanup_policy=plan.cleanup_policy,
        lease_duration_ms=plan.deadline_policy.session_timeout_ms,
    )
    lease_record = replace(
        lease.record,
        slot_count=region.slot_count,
        slack_bytes=region.slack_bytes,
        gpu_device=region.gpu_device,
    )

    remote_cleaned = False
    try:
        plan_session(str(socket_path), plan)
        lease_regions(str(socket_path), plan.session_id, [lease_record])
        seed_session(str(socket_path), plan.session_id)
        materialize_labels(str(socket_path), plan.session_id)
        prepare_fast_phase(str(socket_path), plan.session_id)

        channel = grpc.insecure_channel(f"unix:{socket_path}")
        stub = session_pb2_grpc.PoseSessionServiceStub(channel)
        with pytest.raises(grpc.RpcError) as error:
            stub.RunFastPhase(
                session_pb2.RunFastPhaseRequest(
                    rounds=[
                        session_pb2.FastRoundChallenge(
                            protocol_version="pose-grpc/v1",
                            session_id=plan.session_id,
                            round_index=0,
                            challenge_index=0,
                        ),
                        session_pb2.FastRoundChallenge(
                            protocol_version="pose-grpc/v1",
                            session_id=plan.session_id,
                            round_index=1,
                            challenge_index=1,
                        ),
                    ]
                )
            )
        assert error.value.code() == grpc.StatusCode.INVALID_ARGUMENT
        assert "exactly one challenge per request" in (error.value.details() or "")
        channel.close()

        finalize_session(
            str(socket_path),
            session_id=plan.session_id,
            verdict="SUCCESS",
            success=True,
            retain_session=False,
        )
        cleanup_session(str(socket_path), plan.session_id)
        remote_cleaned = True
    finally:
        release_host_lease(
            lease,
            zeroize=not remote_cleaned and plan.cleanup_policy.zeroize,
            verify_zeroization=not remote_cleaned and plan.cleanup_policy.verify_zeroization,
        )
        server.stop(grace=None)


def test_fast_phase_client_uses_one_round_per_rpc_and_times_end_to_end(monkeypatch) -> None:
    captured_rounds: list[int] = []

    class _FakeChannel:
        def close(self) -> None:
            return None

    class _FakeReady:
        def result(self, timeout: float | None = None) -> None:
            del timeout
            return None

    class _FakeStub:
        def RunFastPhase(self, request: session_pb2.RunFastPhaseRequest) -> session_pb2.RunFastPhaseResponse:
            captured_rounds.append(len(request.rounds))
            assert len(request.rounds) == 1
            round_request = request.rounds[0]
            sleep(0.01)
            return session_pb2.RunFastPhaseResponse(
                responses=[
                    session_pb2.FastRoundResponse(
                        region_id="host-0",
                        challenge_index=round_request.challenge_index,
                        label_bytes=b"x" * 32,
                        round_trip_us=7,
                    )
                ]
            )

    monkeypatch.setattr(
        "pose.verifier.grpc_client._channel_stub",
        lambda _socket_path: (_FakeChannel(), _FakeStub()),
    )
    monkeypatch.setattr(
        "pose.verifier.grpc_client.grpc.channel_ready_future",
        lambda _channel: _FakeReady(),
    )

    with FastPhaseClient("/tmp/fake.sock") as client:
        first = client.run_round(session_id="session", round_index=0, challenge_index=3)
        second = client.run_round(session_id="session", round_index=1, challenge_index=7)

    assert captured_rounds == [1, 1]
    assert first["challenge_index"] == 3
    assert second["challenge_index"] == 7
    assert first["prover_lookup_round_trip_us"] == 7
    assert second["prover_lookup_round_trip_us"] == 7
    assert first["round_trip_us"] >= 10_000
    assert second["round_trip_us"] >= 10_000
