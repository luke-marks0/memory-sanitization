from __future__ import annotations

import mmap
from concurrent import futures
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import grpc

from pose.common.errors import ProtocolError, ResourceFailure
from pose.common.gpu_lease import GpuLeaseAttachment, attach_gpu_lease
from pose.common.host_lease import HostLeaseAttachment, attach_host_lease
from pose.common.merkle import commit_payload
from pose.filecoin.porep_unit import build_porep_unit_from_seal_artifact
from pose.filecoin.reference import SealArtifact, VendoredFilecoinReference
from pose.protocol.grpc_codec import (
    GRPC_PROTOCOL_VERSION,
    artifact_to_proto,
    lease_record_from_proto,
    region_manifest_to_json,
    session_plan_from_proto,
)
from pose.protocol.messages import LeaseRecord, SessionPlan
from pose.protocol.region_payloads import RegionManifest, build_region_manifest, build_region_payload
from pose.prover.host_worker import _cleanup_mapping, _materialize_locally
from pose.v1 import session_pb2, session_pb2_grpc


@dataclass
class SessionState:
    plan: SessionPlan
    leases: dict[str, LeaseRecord] = field(default_factory=dict)
    attachments: dict[str, HostLeaseAttachment | GpuLeaseAttachment] = field(default_factory=dict)
    artifacts_by_region: dict[str, list[SealArtifact]] = field(default_factory=dict)
    region_manifests: dict[str, RegionManifest] = field(default_factory=dict)
    commitments: dict[str, object] = field(default_factory=dict)
    finalized: bool = False
    retained: bool = False
    cleanup_status: str = "NOT_RUN"


class PoseSessionServicer(session_pb2_grpc.PoseSessionServiceServicer):
    def __init__(self) -> None:
        self._reference = VendoredFilecoinReference()
        self._sessions: dict[str, SessionState] = {}

    def _require_protocol_version(self, protocol_version: str) -> None:
        if protocol_version != GRPC_PROTOCOL_VERSION:
            raise ProtocolError(f"Unsupported gRPC protocol version: {protocol_version!r}")

    def _session(self, session_id: str) -> SessionState:
        try:
            return self._sessions[session_id]
        except KeyError as error:
            raise ProtocolError(f"Unknown session id: {session_id}") from error

    def _require_single_region_plan(self, plan: SessionPlan) -> None:
        if plan.unit_count <= 0:
            raise ProtocolError(f"Session plan unit_count must be positive, got {plan.unit_count}")
        if len(plan.regions) != 1:
            raise ProtocolError("Current gRPC prover requires exactly one planned region.")
        region = plan.regions[0]
        if region.region_type not in {"host", "gpu"}:
            raise ProtocolError(f"Unsupported region type: {region.region_type!r}")
        if region.region_type == "host" and region.gpu_device is not None:
            raise ProtocolError("Host regions must not set gpu_device.")
        if region.region_type == "gpu" and region.gpu_device is None:
            raise ProtocolError("GPU regions must include gpu_device.")
        if plan.challenge_leaf_size <= 0:
            raise ProtocolError(
                f"Session plan challenge leaf size must be positive, got {plan.challenge_leaf_size}"
            )
        if region.usable_bytes <= 0 or region.usable_bytes % plan.challenge_leaf_size != 0:
            raise ProtocolError(
                "Region usable_bytes must be a positive multiple of the challenge leaf size."
            )
        if len(plan.sector_plan) != plan.unit_count:
            raise ProtocolError(
                "Current gRPC prover requires one explicit sector-plan entry per planned PoRep unit."
            )

        seen_indices: set[int] = set()
        for item in plan.sector_plan:
            if item.region_id != region.region_id:
                raise ProtocolError(
                    f"Sector-plan entry targets unexpected region {item.region_id!r}; "
                    f"expected {region.region_id!r}"
                )
            if item.unit_index in seen_indices:
                raise ProtocolError(f"Duplicate sector-plan unit index: {item.unit_index}")
            seen_indices.add(item.unit_index)
            if item.sector_id <= 0:
                raise ProtocolError(f"Sector-plan sector_id must be positive: {item.sector_id}")
            if not item.prover_id_hex or not item.ticket_hex or not item.seed_hex:
                raise ProtocolError(
                    "Sector-plan entries must include prover_id_hex, ticket_hex, and seed_hex."
                )

        if seen_indices != set(range(plan.unit_count)):
            raise ProtocolError(
                f"Sector-plan indices must match 0..{plan.unit_count - 1}, got {sorted(seen_indices)}"
            )

    def _require_not_expired(self, state: SessionState) -> None:
        if not state.leases:
            return
        expiry = datetime.fromisoformat(next(iter(state.leases.values())).lease_expiry)
        if expiry <= datetime.now(UTC):
            self._cleanup_state(state)
            raise ProtocolError("Session lease expired before the requested operation.")

    def _cleanup_state(self, state: SessionState) -> str:
        cleanup_status = "NOT_RUN"
        for attachment in state.attachments.values():
            if isinstance(attachment, HostLeaseAttachment):
                cleanup_status = _cleanup_mapping(
                    mapping=attachment.mapping,
                    usable_bytes=attachment.usable_bytes,
                    zeroize=state.plan.cleanup_policy.zeroize,
                    verify_zeroization=state.plan.cleanup_policy.verify_zeroization,
                )
            else:
                if state.plan.cleanup_policy.zeroize:
                    attachment.zeroize()
                    if (
                        state.plan.cleanup_policy.verify_zeroization
                        and not attachment.verify_zeroized()
                    ):
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
        state.artifacts_by_region.clear()
        state.region_manifests.clear()
        state.commitments.clear()
        state.cleanup_status = cleanup_status
        return cleanup_status

    def _abort(self, context: grpc.ServicerContext, error: Exception) -> None:
        if isinstance(error, ResourceFailure):
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(error))
        context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))

    def _serialize_artifacts(
        self,
        *,
        state: SessionState,
        region_id: str,
    ) -> tuple[list[object], bytes, int]:
        artifacts = state.artifacts_by_region.get(region_id)
        if not artifacts:
            raise ProtocolError("GenerateInnerPoRep must run before MaterializeRegionPayloads.")
        units = []
        payload_parts: list[bytes] = []
        object_serialization_started = perf_counter()
        for artifact in artifacts:
            unit = build_porep_unit_from_seal_artifact(
                artifact,
                storage_profile=state.plan.porep_unit_profile,
                leaf_alignment_bytes=state.plan.challenge_leaf_size,
            )
            units.append(unit)
            payload_parts.append(unit.serialized_bytes)
        return units, b"".join(payload_parts), int(
            (perf_counter() - object_serialization_started) * 1000
        )

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
                "real-filecoin-reference",
                "rechallenge",
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
            self._require_single_region_plan(plan)
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
            leases = {
                lease.region_id: lease_record_from_proto(lease)
                for lease in request.leases
            }
            if set(leases) != {region.region_id for region in state.plan.regions}:
                raise ProtocolError("LeaseRegions must provide exactly one lease per planned region.")
            state.leases = leases
        except Exception as error:  # pragma: no cover - grpc abort
            self._abort(context, error)
        return session_pb2.LeaseRegionsResponse(accepted=True)

    def GenerateInnerPoRep(
        self,
        request: session_pb2.GenerateInnerPoRepRequest,
        context: grpc.ServicerContext,
    ) -> session_pb2.GenerateInnerPoRepResponse:
        try:
            self._require_protocol_version(request.protocol_version)
            state = self._session(request.session_id)
            self._require_not_expired(state)
            region = state.plan.regions[0]
            artifacts, _units, _payload, _object_serialization_ms = _materialize_locally(
                reference=self._reference,
                storage_profile=state.plan.porep_unit_profile,
                leaf_size=state.plan.challenge_leaf_size,
                unit_count=state.plan.unit_count,
                sector_plan=state.plan.sector_plan,
            )
            state.artifacts_by_region[region.region_id] = list(artifacts)
        except Exception as error:  # pragma: no cover - grpc abort
            self._abort(context, error)
        return session_pb2.GenerateInnerPoRepResponse(
            artifacts=[
                artifact_to_proto(region.region_id, artifact)
                for artifact in state.artifacts_by_region[region.region_id]
            ]
        )

    def MaterializeRegionPayloads(
        self,
        request: session_pb2.MaterializeRegionPayloadsRequest,
        context: grpc.ServicerContext,
    ) -> session_pb2.MaterializeRegionPayloadsResponse:
        try:
            self._require_protocol_version(request.protocol_version)
            state = self._session(request.session_id)
            self._require_not_expired(state)
            region = state.plan.regions[0]
            lease = state.leases.get(region.region_id)
            if lease is None:
                raise ProtocolError("LeaseRegions must run before MaterializeRegionPayloads.")
            units, real_payload, object_serialization_ms = self._serialize_artifacts(
                state=state,
                region_id=region.region_id,
            )
            tail_filler_bytes = region.usable_bytes - len(real_payload)
            if tail_filler_bytes < 0:
                raise ResourceFailure(
                    f"Materialized payload length {len(real_payload)} exceeds lease size {region.usable_bytes}"
                )
            if tail_filler_bytes > min(state.plan.challenge_leaf_size, 1024 * 1024):
                raise ResourceFailure(
                    f"Tail filler requirement {tail_filler_bytes} exceeds the allowed limit "
                    f"for leaf size {state.plan.challenge_leaf_size}"
                )
            payload = build_region_payload(
                [unit.serialized_bytes for unit in units],
                session_nonce=state.plan.nonce,
                region_id=region.region_id,
                session_plan_root=state.plan.plan_root_hex,
                tail_filler_bytes=tail_filler_bytes,
            )
            copy_to_host_ms = 0
            copy_to_hbm_ms = 0
            copy_started = perf_counter()
            if region.region_type == "host":
                attachment = attach_host_lease(lease.lease_handle, usable_bytes=lease.usable_bytes)
                attachment.write(payload)
                copy_to_host_ms = int((perf_counter() - copy_started) * 1000)
            else:
                attachment = attach_gpu_lease(lease.lease_handle, usable_bytes=lease.usable_bytes)
                attachment.write(payload)
                copy_to_hbm_ms = int((perf_counter() - copy_started) * 1000)
            outer_tree_started = perf_counter()
            commitment = commit_payload(payload, state.plan.challenge_leaf_size)
            outer_tree_build_ms = int((perf_counter() - outer_tree_started) * 1000)
            manifest = build_region_manifest(
                region_id=region.region_id,
                region_type=region.region_type,
                usable_bytes=lease.usable_bytes,
                leaf_size=state.plan.challenge_leaf_size,
                payload=payload,
                merkle_root_hex=commitment.root_hex,
                units=tuple(units),
                tail_filler_bytes=tail_filler_bytes,
            )
            if region.region_id in state.attachments:
                state.attachments[region.region_id].close()
            state.attachments[region.region_id] = attachment
            state.region_manifests[region.region_id] = manifest
            state.commitments[region.region_id] = commitment
        except Exception as error:  # pragma: no cover - grpc abort
            self._abort(context, error)
        return session_pb2.MaterializeRegionPayloadsResponse(
            commitments=[
                session_pb2.RegionCommitment(
                    region_id=region.region_id,
                    region_root_hex=manifest.merkle_root_hex,
                    covered_bytes=manifest.payload_length_bytes,
                    real_porep_bytes=manifest.real_porep_bytes,
                    tail_filler_bytes=manifest.tail_filler_bytes,
                    region_manifest_json=region_manifest_to_json(manifest),
                )
            ],
            timings_ms={
                "copy_to_host": copy_to_host_ms,
                "copy_to_hbm": copy_to_hbm_ms,
                "object_serialization": object_serialization_ms,
                "outer_tree_build": outer_tree_build_ms,
            },
        )

    def CommitRegions(
        self,
        request: session_pb2.CommitRegionsRequest,
        context: grpc.ServicerContext,
    ) -> session_pb2.CommitRegionsResponse:
        try:
            self._require_protocol_version(request.protocol_version)
            state = self._session(request.session_id)
            if not state.region_manifests:
                raise ProtocolError("MaterializeRegionPayloads must run before CommitRegions.")
        except Exception as error:  # pragma: no cover - grpc abort
            self._abort(context, error)
        return session_pb2.CommitRegionsResponse(accepted=True)

    def VerifyInnerProofs(
        self,
        request: session_pb2.VerifyInnerProofsRequest,
        context: grpc.ServicerContext,
    ) -> session_pb2.VerifyInnerProofsResponse:
        try:
            self._require_protocol_version(request.protocol_version)
            self._session(request.session_id)
        except Exception as error:  # pragma: no cover - grpc abort
            self._abort(context, error)
        return session_pb2.VerifyInnerProofsResponse(accepted=True)

    def ChallengeOuter(
        self,
        request: session_pb2.ChallengeOuterRequest,
        context: grpc.ServicerContext,
    ) -> session_pb2.ChallengeOuterResponse:
        try:
            self._require_protocol_version(request.protocol_version)
            state = self._session(request.session_id)
            self._require_not_expired(state)
            challenge = request.challenges[0]
            attachment = state.attachments.get(challenge.region_id)
            commitment = state.commitments.get(challenge.region_id)
            manifest = state.region_manifests.get(challenge.region_id)
            if attachment is None or manifest is None or commitment is None:
                raise ProtocolError(
                    "MaterializeRegionPayloads must run before ChallengeOuter."
                )
            response_started = perf_counter()
            openings = []
            for index in challenge.leaf_indices:
                leaf = attachment.read_leaf(int(index), manifest.leaf_size)
                opening = commitment.opening(int(index), leaf)
                openings.append(
                    session_pb2.OuterOpening(
                        region_id=challenge.region_id,
                        session_manifest_root=challenge.session_manifest_root,
                        leaf_index=int(index),
                        leaf_bytes=leaf,
                        auth_path=list(opening.sibling_hashes),
                    )
                )
            response_ms = int((perf_counter() - response_started) * 1000)
        except Exception as error:  # pragma: no cover - grpc abort
            self._abort(context, error)
        return session_pb2.ChallengeOuterResponse(openings=openings, response_ms=response_ms)

    def VerifyOuter(
        self,
        request: session_pb2.VerifyOuterRequest,
        context: grpc.ServicerContext,
    ) -> session_pb2.VerifyOuterResponse:
        try:
            self._require_protocol_version(request.protocol_version)
            self._session(request.session_id)
        except Exception as error:  # pragma: no cover - grpc abort
            self._abort(context, error)
        return session_pb2.VerifyOuterResponse(accepted=True)

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
