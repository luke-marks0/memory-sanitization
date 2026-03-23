from __future__ import annotations

from concurrent import futures
from dataclasses import replace
from pathlib import Path
from time import sleep

import grpc
import pytest

from pose.common.gpu_lease import create_gpu_lease, release_gpu_lease
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


class _FakeCudaRuntime:
    def __init__(self) -> None:
        self._next_pointer = 0x1000
        self._next_handle = 1
        self._buffers: dict[int, bytearray] = {}
        self._handles: dict[bytes, int] = {}

    def device_count(self) -> int:
        return 1

    def malloc(self, device: int, size: int) -> int:
        assert device == 0
        pointer = self._next_pointer
        self._next_pointer += max(size, 1) + 0x1000
        self._buffers[pointer] = bytearray(size)
        return pointer

    def free(self, device: int, pointer: int) -> None:
        assert device == 0
        self._buffers.pop(pointer, None)

    def memset(self, device: int, pointer: int, value: int, size: int, *, offset: int = 0) -> None:
        assert device == 0
        self._buffers[pointer][offset : offset + size] = bytes([value & 0xFF]) * size

    def synchronize(self, device: int) -> None:
        assert device == 0

    def copy_host_to_device(self, device: int, pointer: int, payload: bytes, *, offset: int = 0) -> None:
        assert device == 0
        self._buffers[pointer][offset : offset + len(payload)] = payload

    def copy_device_to_host(self, device: int, pointer: int, size: int, *, offset: int = 0) -> bytes:
        assert device == 0
        return bytes(self._buffers[pointer][offset : offset + size])

    def ipc_get_mem_handle(self, device: int, pointer: int) -> bytes:
        assert device == 0
        handle = self._next_handle.to_bytes(64, "big")
        self._next_handle += 1
        self._handles[handle] = pointer
        return handle

    def ipc_open_mem_handle(self, device: int, encoded_handle: bytes) -> int:
        assert device == 0
        return self._handles[encoded_handle]

    def ipc_close_mem_handle(self, device: int, pointer: int) -> None:
        assert device == 0
        assert pointer in self._buffers


def _gpu_session_plan() -> SessionPlan:
    descriptor = build_graph_descriptor(
        label_count_m=8,
        graph_parameter_n=2,
        gamma=4,
        hash_backend="blake3-xof",
        label_width_bits=256,
    )
    return SessionPlan(
        session_id="grpc-gpu-runtime-session",
        session_seed_hex="ef" * 32,
        profile_name="single-h100-hbm-max",
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
                region_id="gpu-0",
                region_type="gpu",
                usable_bytes=256,
                slot_count=8,
                covered_bytes=256,
                slack_bytes=0,
                gpu_device=0,
            )
        ],
        adversary_model="general",
        attacker_budget_bytes_assumed=4096,
        q_bound=3,
        claim_notes=["grpc-gpu-runtime-test"],
    )


def _hybrid_session_plan() -> SessionPlan:
    descriptor = build_graph_descriptor(
        label_count_m=8,
        graph_parameter_n=2,
        gamma=4,
        hash_backend="blake3-xof",
        label_width_bits=256,
    )
    return SessionPlan(
        session_id="grpc-hybrid-runtime-session",
        session_seed_hex="ab" * 32,
        profile_name="single-h100-hybrid-small",
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
                usable_bytes=128,
                slot_count=4,
                covered_bytes=128,
                slack_bytes=0,
            ),
            RegionPlan(
                region_id="gpu-0",
                region_type="gpu",
                usable_bytes=128,
                slot_count=4,
                covered_bytes=128,
                slack_bytes=0,
                gpu_device=0,
            ),
        ],
        adversary_model="general",
        attacker_budget_bytes_assumed=4096,
        q_bound=3,
        claim_notes=["grpc-hybrid-runtime-test"],
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


def test_gpu_pose_db_grpc_roundtrip(monkeypatch, tmp_path: Path) -> None:
    socket_path = tmp_path / "pose-gpu-runtime.sock"
    if socket_path.exists():
        socket_path.unlink()

    runtime = _FakeCudaRuntime()
    monkeypatch.setattr("pose.common.gpu_lease.get_cuda_runtime", lambda: runtime)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    session_pb2_grpc.add_PoseSessionServiceServicer_to_server(PoseSessionServicer(), server)
    server.add_insecure_port(f"unix:{socket_path}")
    server.start()

    plan = _gpu_session_plan()
    region = plan.regions[0]
    lease = create_gpu_lease(
        session_id=plan.session_id,
        region_id=region.region_id,
        device=0,
        usable_bytes=region.usable_bytes,
        cleanup_policy=plan.cleanup_policy,
        lease_duration_ms=plan.deadline_policy.session_timeout_ms,
        runtime=runtime,
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
        assert "gpu-hbm" in capabilities["capabilities"]
        assert "cuda-ipc" in capabilities["capabilities"]

        plan_session(str(socket_path), plan)
        lease_regions(str(socket_path), plan.session_id, [lease_record])
        seed_session(str(socket_path), plan.session_id)
        materialization, timings = materialize_labels(str(socket_path), plan.session_id)
        prepare_fast_phase(str(socket_path), plan.session_id)

        assert materialization == {
            "graph_descriptor_digest": plan.graph_descriptor_digest,
            "scratch_peak_bytes": materialization["scratch_peak_bytes"],
            "regions": {
                "gpu-0": {
                    "covered_bytes": 256,
                    "slack_bytes": 0,
                    "declared_stage_copy_bytes": 0,
                }
            },
        }
        assert materialization["scratch_peak_bytes"] > 0
        assert "label_generation" in timings
        assert "copy_to_hbm" in timings

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
            assert response["region_id"] == "gpu-0"
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
        release_gpu_lease(
            lease,
            zeroize=not remote_cleaned and plan.cleanup_policy.zeroize,
            verify_zeroization=not remote_cleaned and plan.cleanup_policy.verify_zeroization,
        )
        server.stop(grace=None)


def test_hybrid_pose_db_grpc_roundtrip(monkeypatch, tmp_path: Path) -> None:
    socket_path = tmp_path / "pose-hybrid-runtime.sock"
    if socket_path.exists():
        socket_path.unlink()

    runtime = _FakeCudaRuntime()
    monkeypatch.setattr("pose.common.gpu_lease.get_cuda_runtime", lambda: runtime)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    session_pb2_grpc.add_PoseSessionServiceServicer_to_server(PoseSessionServicer(), server)
    server.add_insecure_port(f"unix:{socket_path}")
    server.start()

    plan = _hybrid_session_plan()
    host_region, gpu_region = plan.regions
    host_lease = create_host_lease(
        session_id=plan.session_id,
        region_id=host_region.region_id,
        usable_bytes=host_region.usable_bytes,
        cleanup_policy=plan.cleanup_policy,
        lease_duration_ms=plan.deadline_policy.session_timeout_ms,
    )
    gpu_lease = create_gpu_lease(
        session_id=plan.session_id,
        region_id=gpu_region.region_id,
        device=0,
        usable_bytes=gpu_region.usable_bytes,
        cleanup_policy=plan.cleanup_policy,
        lease_duration_ms=plan.deadline_policy.session_timeout_ms,
        runtime=runtime,
    )
    lease_records = [
        replace(
            host_lease.record,
            slot_count=host_region.slot_count,
            slack_bytes=host_region.slack_bytes,
            gpu_device=host_region.gpu_device,
        ),
        replace(
            gpu_lease.record,
            slot_count=gpu_region.slot_count,
            slack_bytes=gpu_region.slack_bytes,
            gpu_device=gpu_region.gpu_device,
        ),
    ]

    remote_cleaned = False
    try:
        capabilities = discover(str(socket_path))
        assert capabilities["protocol_version"] == "pose-grpc/v1"
        assert "gpu-hbm" in capabilities["capabilities"]

        plan_session(str(socket_path), plan)
        lease_regions(str(socket_path), plan.session_id, lease_records)
        seed_session(str(socket_path), plan.session_id)
        materialization, timings = materialize_labels(str(socket_path), plan.session_id)
        prepare_fast_phase(str(socket_path), plan.session_id)

        assert materialization["regions"] == {
            "host-0": {
                "covered_bytes": 128,
                "slack_bytes": 0,
                "declared_stage_copy_bytes": 0,
            },
            "gpu-0": {
                "covered_bytes": 128,
                "slack_bytes": 0,
                "declared_stage_copy_bytes": 0,
            },
        }
        assert "copy_to_host" in timings
        assert "copy_to_hbm" in timings

        challenge_indices = [0, 3, 4, 7]
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
        expected_regions = ["host-0", "host-0", "gpu-0", "gpu-0"]
        for response, challenge_index, expected_label, expected_region in zip(
            responses,
            challenge_indices,
            expected_labels,
            expected_regions,
            strict=True,
        ):
            assert response["region_id"] == expected_region
            assert response["challenge_index"] == challenge_index
            if expected_region == "host-0":
                assert response["label_bytes"] == host_lease.read(
                    length=label_width_bytes,
                    offset=challenge_index * label_width_bytes,
                )
            else:
                assert response["label_bytes"] == gpu_lease.read(
                    length=label_width_bytes,
                    offset=(challenge_index - host_region.slot_count) * label_width_bytes,
                )
            assert response["label_bytes"] == expected_label

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
        assert host_lease.read() == bytes(host_region.usable_bytes)
        assert gpu_lease.read() == bytes(gpu_region.usable_bytes)
    finally:
        release_host_lease(
            host_lease,
            zeroize=not remote_cleaned and plan.cleanup_policy.zeroize,
            verify_zeroization=not remote_cleaned and plan.cleanup_policy.verify_zeroization,
        )
        release_gpu_lease(
            gpu_lease,
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
