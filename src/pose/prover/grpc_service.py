from __future__ import annotations

from concurrent import futures
from array import array
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from sys import getsizeof
from time import perf_counter

import grpc

from pose.common.errors import ProtocolError, ResourceFailure
from pose.common.gpu_lease import GpuLeaseAttachment, attach_gpu_lease
from pose.common.host_lease import HostLeaseAttachment, attach_host_lease
from pose.graphs import PoseDbGraph, build_graph_descriptor, build_pose_db_graph
from pose.hashing import internal_label_bytes, source_label_bytes
from pose.protocol.grpc_codec import GRPC_PROTOCOL_VERSION, lease_record_from_proto, session_plan_from_proto
from pose.protocol.messages import LeaseRecord, SessionPlan
from pose.v1 import session_pb2, session_pb2_grpc


@dataclass
class SessionState:
    plan: SessionPlan
    leases: dict[str, LeaseRecord] = field(default_factory=dict)
    attachments: dict[str, HostLeaseAttachment | GpuLeaseAttachment] = field(default_factory=dict)
    graph: PoseDbGraph | None = None
    seeded: bool = False
    materialized: bool = False
    finalized: bool = False
    retained: bool = False
    cleanup_status: str = "NOT_RUN"


class PoseSessionServicer(session_pb2_grpc.PoseSessionServiceServicer):
    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}

    def _require_protocol_version(self, protocol_version: str) -> None:
        if protocol_version != GRPC_PROTOCOL_VERSION:
            raise ProtocolError(f"Unsupported gRPC protocol version: {protocol_version!r}")

    def _session(self, session_id: str) -> SessionState:
        try:
            return self._sessions[session_id]
        except KeyError as error:
            raise ProtocolError(f"Unknown session id: {session_id}") from error

    def _require_not_expired(self, state: SessionState) -> None:
        if not state.leases:
            return
        now = datetime.now(UTC)
        expiries = [datetime.fromisoformat(lease.lease_expiry) for lease in state.leases.values()]
        if any(expiry <= now for expiry in expiries):
            self._cleanup_state(state)
            raise ProtocolError("Session lease expired before the requested operation.")

    def _require_plan_shape(self, plan: SessionPlan) -> None:
        expected_descriptor = build_graph_descriptor(
            label_count_m=plan.label_count_m,
            graph_parameter_n=plan.graph_parameter_n,
            gamma=plan.gamma,
            hash_backend=plan.hash_backend,
            label_width_bits=plan.label_width_bits,
            graph_family=plan.graph_family,
        )
        if plan.graph_descriptor_digest != expected_descriptor.digest:
            raise ProtocolError(
                "Session plan graph_descriptor_digest does not match the canonical descriptor digest for the "
                f"declared graph parameters: {plan.graph_descriptor_digest!r} != {expected_descriptor.digest!r}"
            )
        if plan.label_count_m <= 0:
            raise ProtocolError(f"Session plan label_count_m must be positive, got {plan.label_count_m}")
        if plan.gamma <= 0:
            raise ProtocolError(f"Session plan gamma must be positive, got {plan.gamma}")
        if not plan.regions:
            raise ProtocolError("PoSE-DB sessions require at least one planned region.")
        label_width_bytes = plan.label_width_bits // 8
        seen_region_ids: set[str] = set()
        for region in plan.regions:
            if region.region_id in seen_region_ids:
                raise ProtocolError(f"Duplicate region id in session plan: {region.region_id}")
            seen_region_ids.add(region.region_id)
            if region.region_type not in {"host", "gpu"}:
                raise ProtocolError(f"Unsupported region type: {region.region_type!r}")
            if region.region_type == "host" and region.gpu_device is not None:
                raise ProtocolError("Host regions must not set gpu_device.")
            if region.region_type == "gpu" and region.gpu_device is None:
                raise ProtocolError("GPU regions must include gpu_device.")
            if region.slot_count <= 0:
                raise ProtocolError(f"Region slot_count must be positive, got {region.slot_count}")
            expected_covered_bytes = region.slot_count * label_width_bytes
            if region.covered_bytes != expected_covered_bytes:
                raise ProtocolError(
                    f"Region covered_bytes must equal slot_count * label_width_bytes: "
                    f"{region.covered_bytes} != {expected_covered_bytes}"
                )
            if region.usable_bytes < region.covered_bytes:
                raise ProtocolError("Region usable_bytes must be at least covered_bytes.")
            if region.slack_bytes != region.usable_bytes - region.covered_bytes:
                raise ProtocolError(
                    "Region slack_bytes must equal usable_bytes - covered_bytes."
                )
        if sum(item.slot_count for item in plan.regions) != plan.label_count_m:
            raise ProtocolError("Sum of region slot_count values must equal label_count_m.")

    def _cleanup_state(self, state: SessionState) -> str:
        cleanup_status = "NOT_RUN"
        for attachment in state.attachments.values():
            if isinstance(attachment, HostLeaseAttachment):
                if state.plan.cleanup_policy.zeroize:
                    attachment.write(b"")
                    if (
                        state.plan.cleanup_policy.verify_zeroization
                        and attachment.read(attachment.usable_bytes) != bytes(attachment.usable_bytes)
                    ):
                        raise ResourceFailure("Host lease zeroization verification failed")
                    cleanup_status = (
                        "ZEROIZED_AND_VERIFIED"
                        if state.plan.cleanup_policy.verify_zeroization
                        else "ZEROIZED_AND_RELEASED"
                    )
                else:
                    cleanup_status = "RELEASED_WITHOUT_ZEROIZE"
                attachment.close()
            else:
                if state.plan.cleanup_policy.zeroize:
                    attachment.zeroize()
                    if state.plan.cleanup_policy.verify_zeroization and not attachment.verify_zeroized():
                        raise ResourceFailure("GPU lease zeroization verification failed")
                    cleanup_status = (
                        "ZEROIZED_AND_VERIFIED"
                        if state.plan.cleanup_policy.verify_zeroization
                        else "ZEROIZED_AND_RELEASED"
                    )
                else:
                    cleanup_status = "RELEASED_WITHOUT_ZEROIZE"
                attachment.close()
        state.attachments.clear()
        state.cleanup_status = cleanup_status
        return cleanup_status

    def _abort(self, context: grpc.ServicerContext, error: Exception) -> None:
        if isinstance(error, ResourceFailure):
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(error))
        context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))

    def _attachment_for_lease(
        self,
        lease: LeaseRecord,
    ) -> HostLeaseAttachment | GpuLeaseAttachment:
        if lease.region_type == "host":
            return attach_host_lease(lease.lease_handle, usable_bytes=lease.usable_bytes)
        if lease.region_type == "gpu":
            return attach_gpu_lease(lease.lease_handle, usable_bytes=lease.usable_bytes)
        raise ProtocolError(f"Unsupported region type: {lease.region_type!r}")

    def _region_offset(self, state: SessionState, region_id: str) -> int:
        offset = 0
        for region in state.plan.regions:
            if region.region_id == region_id:
                return offset
            offset += region.slot_count
        raise ProtocolError(f"Unknown region id in plan: {region_id}")

    def _resolve_slot(self, state: SessionState, challenge_index: int) -> tuple[str, int]:
        if challenge_index < 0 or challenge_index >= state.plan.label_count_m:
            raise ProtocolError(
                f"Challenge index {challenge_index} is outside [0, {state.plan.label_count_m})"
            )
        cursor = 0
        for region in state.plan.regions:
            next_cursor = cursor + region.slot_count
            if challenge_index < next_cursor:
                return region.region_id, challenge_index - cursor
            cursor = next_cursor
        raise ProtocolError(f"Challenge index {challenge_index} could not be mapped to a region.")

    def _estimated_static_scratch_bytes(
        self,
        *,
        successor_counts: array,
        challenge_index_by_node: dict[int, int],
    ) -> int:
        total = getsizeof(successor_counts)
        total += getsizeof(challenge_index_by_node)
        for node_index, challenge_index in challenge_index_by_node.items():
            total += getsizeof(node_index) + getsizeof(challenge_index)
        return total

    def Discover(
        self,
        request: session_pb2.DiscoverRequest,
        context: grpc.ServicerContext,
    ) -> session_pb2.DiscoverResponse:
        try:
            self._require_protocol_version(request.protocol_version)
        except Exception as error:  # pragma: no cover - grpc abort
            self._abort(context, error)
        return session_pb2.DiscoverResponse(
            protocol_version=GRPC_PROTOCOL_VERSION,
            capabilities=[
                "host-memory",
                "gpu-hbm",
                "cuda-ipc",
                "pose-db-control-plane",
                "pose-db-fast-phase",
                "unix-domain-sockets",
            ],
        )

    def PlanSession(
        self,
        request: session_pb2.PlanSessionRequest,
        context: grpc.ServicerContext,
    ) -> session_pb2.PlanSessionResponse:
        try:
            self._require_protocol_version(request.protocol_version)
            plan = session_plan_from_proto(request.plan)
            self._require_plan_shape(plan)
            if plan.session_id in self._sessions:
                raise ProtocolError(f"Duplicate session id: {plan.session_id}")
            self._sessions[plan.session_id] = SessionState(plan=plan)
        except Exception as error:  # pragma: no cover - grpc abort
            self._abort(context, error)
        return session_pb2.PlanSessionResponse(session_id=request.plan.session_id, accepted=True)

    def LeaseRegions(
        self,
        request: session_pb2.LeaseRegionsRequest,
        context: grpc.ServicerContext,
    ) -> session_pb2.LeaseRegionsResponse:
        try:
            self._require_protocol_version(request.protocol_version)
            state = self._session(request.session_id)
            leases = {lease.region_id: lease_record_from_proto(lease) for lease in request.leases}
            if set(leases) != {region.region_id for region in state.plan.regions}:
                raise ProtocolError("LeaseRegions must provide exactly one lease per planned region.")
            state.leases = leases
        except Exception as error:  # pragma: no cover - grpc abort
            self._abort(context, error)
        return session_pb2.LeaseRegionsResponse(accepted=True)

    def SeedSession(
        self,
        request: session_pb2.SeedSessionRequest,
        context: grpc.ServicerContext,
    ) -> session_pb2.SeedSessionResponse:
        try:
            self._require_protocol_version(request.protocol_version)
            state = self._session(request.session_id)
            self._require_not_expired(state)
            if not state.leases:
                raise ProtocolError("LeaseRegions must run before SeedSession.")
            state.graph = build_pose_db_graph(
                label_count_m=state.plan.label_count_m,
                graph_parameter_n=state.plan.graph_parameter_n,
                gamma=state.plan.gamma,
                hash_backend=state.plan.hash_backend,
                label_width_bits=state.plan.label_width_bits,
            )
            state.seeded = True
        except Exception as error:  # pragma: no cover - grpc abort
            self._abort(context, error)
        return session_pb2.SeedSessionResponse(accepted=True)

    def MaterializeLabels(
        self,
        request: session_pb2.MaterializeLabelsRequest,
        context: grpc.ServicerContext,
    ) -> session_pb2.MaterializeLabelsResponse:
        try:
            self._require_protocol_version(request.protocol_version)
            state = self._session(request.session_id)
            self._require_not_expired(state)
            if not state.seeded:
                raise ProtocolError("SeedSession must run before MaterializeLabels.")
            if state.graph is None:
                raise ProtocolError("SeedSession must initialize the session graph before MaterializeLabels.")

            label_generation_started = perf_counter()
            copy_to_host_ms = 0
            copy_to_hbm_ms = 0
            stage_buffer_cleanup_ms = 0
            scratch_peak_bytes = 0
            regions: list[session_pb2.RegionMaterialization] = []
            label_width_bytes = state.plan.label_width_bits // 8
            session_seed = bytes.fromhex(state.plan.session_seed_hex)
            challenge_index_by_node = {
                node: challenge_index
                for challenge_index, node in enumerate(state.graph.challenge_set)
            }
            successor_counts = array("I", [0]) * state.graph.node_count
            for predecessors in state.graph.predecessors:
                for predecessor in predecessors:
                    successor_counts[predecessor] += 1
            static_scratch_bytes = self._estimated_static_scratch_bytes(
                successor_counts=successor_counts,
                challenge_index_by_node=challenge_index_by_node,
            )
            region_type_by_id = {
                region.region_id: region.region_type
                for region in state.plan.regions
            }
            for region in state.plan.regions:
                lease = state.leases.get(region.region_id)
                if lease is None:
                    raise ProtocolError("LeaseRegions must run before MaterializeLabels.")
                attachment = self._attachment_for_lease(lease)
                if region.region_id in state.attachments:
                    state.attachments[region.region_id].close()
                state.attachments[region.region_id] = attachment
                regions.append(
                    session_pb2.RegionMaterialization(
                        region_id=region.region_id,
                        covered_bytes=region.covered_bytes,
                        slack_bytes=region.slack_bytes,
                        declared_stage_copy_bytes=0,
                    )
                )
            scratch_labels: dict[int, bytearray] = {}
            scratch_entry_bytes_total = 0
            scratch_entry_bytes_by_node: dict[int, int] = {}
            for node_index, predecessors in enumerate(state.graph.predecessors):
                predecessor_labels: list[bytes] = []
                for predecessor in predecessors:
                    scratch_label = scratch_labels.get(predecessor)
                    if scratch_label is not None:
                        predecessor_labels.append(bytes(scratch_label))
                        continue
                    challenge_index = challenge_index_by_node.get(predecessor, -1)
                    if challenge_index < 0:
                        raise ProtocolError(
                            f"Transient predecessor label for node {predecessor} was not retained during materialization."
                        )
                    region_id, local_slot_index = self._resolve_slot(state, challenge_index)
                    predecessor_labels.append(
                        state.attachments[region_id].read(
                            length=label_width_bytes,
                            offset=local_slot_index * label_width_bytes,
                        )
                    )
                scratch_peak_bytes = max(
                    scratch_peak_bytes,
                    static_scratch_bytes
                    + getsizeof(scratch_labels)
                    + scratch_entry_bytes_total
                    + getsizeof(predecessor_labels)
                    + sum(getsizeof(label_bytes) for label_bytes in predecessor_labels),
                )

                if not predecessors:
                    label_bytes = source_label_bytes(
                        session_seed=session_seed,
                        graph_descriptor_digest=state.graph.graph_descriptor_digest,
                        node_index=node_index,
                        hash_backend=state.plan.hash_backend,
                        output_bytes=label_width_bytes,
                    )
                else:
                    label_bytes = internal_label_bytes(
                        session_seed=session_seed,
                        graph_descriptor_digest=state.graph.graph_descriptor_digest,
                        node_index=node_index,
                        predecessor_labels=predecessor_labels,
                        hash_backend=state.plan.hash_backend,
                        output_bytes=label_width_bytes,
                    )

                challenge_index = challenge_index_by_node.get(node_index, -1)
                if challenge_index >= 0:
                    region_id, local_slot_index = self._resolve_slot(state, challenge_index)
                    copy_started = perf_counter()
                    state.attachments[region_id].write_at(
                        label_bytes,
                        offset=local_slot_index * label_width_bytes,
                    )
                    copy_elapsed_ms = int((perf_counter() - copy_started) * 1000)
                    if region_type_by_id[region_id] == "host":
                        copy_to_host_ms += copy_elapsed_ms
                    else:
                        copy_to_hbm_ms += copy_elapsed_ms
                elif successor_counts[node_index] > 0:
                    scratch_labels[node_index] = bytearray(label_bytes)
                    entry_bytes = getsizeof(node_index) + getsizeof(scratch_labels[node_index])
                    scratch_entry_bytes_by_node[node_index] = entry_bytes
                    scratch_entry_bytes_total += entry_bytes
                    scratch_peak_bytes = max(
                        scratch_peak_bytes,
                        static_scratch_bytes
                        + getsizeof(scratch_labels)
                        + scratch_entry_bytes_total
                        + getsizeof(label_bytes),
                    )

                for predecessor in predecessors:
                    successor_counts[predecessor] -= 1
                    if successor_counts[predecessor] == 0:
                        scratch_label = scratch_labels.pop(predecessor, None)
                        if scratch_label is not None:
                            scratch_entry_bytes_total -= scratch_entry_bytes_by_node.pop(predecessor, 0)
                            cleanup_started = perf_counter()
                            scratch_label[:] = b"\x00" * len(scratch_label)
                            stage_buffer_cleanup_ms += int((perf_counter() - cleanup_started) * 1000)
            label_generation_ms = int((perf_counter() - label_generation_started) * 1000)
            state.materialized = True
        except Exception as error:  # pragma: no cover - grpc abort
            self._abort(context, error)
        return session_pb2.MaterializeLabelsResponse(
            regions=regions,
            timings_ms={
                "label_generation": label_generation_ms,
                "copy_to_host": copy_to_host_ms,
                "copy_to_hbm": copy_to_hbm_ms,
                "stage_buffer_cleanup": stage_buffer_cleanup_ms,
            },
            graph_descriptor_digest=state.graph.graph_descriptor_digest,
            scratch_peak_bytes=scratch_peak_bytes,
        )

    def PrepareFastPhase(
        self,
        request: session_pb2.PrepareFastPhaseRequest,
        context: grpc.ServicerContext,
    ) -> session_pb2.PrepareFastPhaseResponse:
        try:
            self._require_protocol_version(request.protocol_version)
            state = self._session(request.session_id)
            self._require_not_expired(state)
            if not state.materialized:
                raise ProtocolError("MaterializeLabels must run before PrepareFastPhase.")
        except Exception as error:  # pragma: no cover - grpc abort
            self._abort(context, error)
        return session_pb2.PrepareFastPhaseResponse(accepted=True)

    def RunFastPhase(
        self,
        request: session_pb2.RunFastPhaseRequest,
        context: grpc.ServicerContext,
    ) -> session_pb2.RunFastPhaseResponse:
        try:
            if len(request.rounds) != 1:
                raise ProtocolError("RunFastPhase must carry exactly one challenge per request.")
            responses: list[session_pb2.FastRoundResponse] = []
            for round_request in request.rounds:
                self._require_protocol_version(round_request.protocol_version)
                state = self._session(round_request.session_id)
                self._require_not_expired(state)
                if not state.materialized:
                    raise ProtocolError("MaterializeLabels must run before RunFastPhase.")
                region_id, local_slot_index = self._resolve_slot(state, int(round_request.challenge_index))
                attachment = state.attachments.get(region_id)
                if attachment is None:
                    raise ProtocolError("MaterializeLabels must run before RunFastPhase.")
                label_width_bytes = state.plan.label_width_bits // 8
                round_started = perf_counter()
                label_bytes = attachment.read(
                    length=label_width_bytes,
                    offset=local_slot_index * label_width_bytes,
                )
                round_trip_us = int((perf_counter() - round_started) * 1_000_000)
                responses.append(
                    session_pb2.FastRoundResponse(
                        region_id=region_id,
                        challenge_index=int(round_request.challenge_index),
                        label_bytes=label_bytes,
                        round_trip_us=round_trip_us,
                    )
                )
        except Exception as error:  # pragma: no cover - grpc abort
            self._abort(context, error)
        return session_pb2.RunFastPhaseResponse(responses=responses)

    def Finalize(
        self,
        request: session_pb2.FinalizeRequest,
        context: grpc.ServicerContext,
    ) -> session_pb2.FinalizeResponse:
        try:
            self._require_protocol_version(request.protocol_version)
            state = self._session(request.session_id)
            state.finalized = True
            state.retained = bool(request.retain_session)
        except Exception as error:  # pragma: no cover - grpc abort
            self._abort(context, error)
        return session_pb2.FinalizeResponse(accepted=True)

    def Cleanup(
        self,
        request: session_pb2.CleanupRequest,
        context: grpc.ServicerContext,
    ) -> session_pb2.CleanupResponse:
        try:
            self._require_protocol_version(request.protocol_version)
            state = self._session(request.session_id)
            cleanup_status = self._cleanup_state(state)
            self._sessions.pop(request.session_id, None)
        except Exception as error:  # pragma: no cover - grpc abort
            self._abort(context, error)
        return session_pb2.CleanupResponse(cleaned=True, cleanup_status=cleanup_status)


def serve_unix(socket_path: str) -> None:
    socket_file = Path(socket_path)
    socket_file.parent.mkdir(parents=True, exist_ok=True)
    if socket_file.exists():
        socket_file.unlink()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    session_pb2_grpc.add_PoseSessionServiceServicer_to_server(
        PoseSessionServicer(),
        server,
    )
    server.add_insecure_port(f"unix:{socket_path}")
    server.start()
    try:
        server.wait_for_termination()
    finally:
        server.stop(grace=None)
        if socket_file.exists():
            socket_file.unlink()
