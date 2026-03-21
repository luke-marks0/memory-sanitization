from __future__ import annotations

from pathlib import Path

from pose.benchmarks.profiles import BenchmarkProfile
from pose.common.errors import ProtocolError
from pose.protocol.codec import load_json_file
from pose.protocol.messages import SessionPlan
from pose.protocol.result_schema import SessionResult, bootstrap_result
from pose.verifier.grpc_gpu_session import run_gpu_session_via_grpc
from pose.verifier.host_session import run_host_session
from pose.verifier.rechallenge import run_host_rechallenge
from pose.verifier.session_store import (
    ResidentSessionRecord,
    load_cold_result_artifact,
    load_plan_file,
    load_resident_session,
)


class VerifierService:
    def describe(self) -> dict[str, object]:
        return {
            "status": "phase2-single-gpu-hbm-complete",
            "supports_host_memory": True,
            "supports_gpu_hbm": True,
            "supports_rechallenge": True,
        }

    def run_session(
        self,
        profile: BenchmarkProfile,
        *,
        retain_session: bool = False,
        session_plan: SessionPlan | None = None,
    ) -> SessionResult:
        if profile.target_devices.get("host", False) and not profile.target_devices.get("gpus"):
            return run_host_session(
                profile,
                retain_session=retain_session,
                session_plan=session_plan,
            )
        gpus = profile.target_devices.get("gpus")
        if not profile.target_devices.get("host", False) and isinstance(gpus, list) and len(gpus) == 1:
            return run_gpu_session_via_grpc(
                profile,
                retain_session=retain_session,
                session_plan=session_plan,
            )
        result = bootstrap_result(
            profile_name=profile.name,
            note="Only host-only and single-gpu HBM sessions are implemented at this stage.",
        )
        result.verdict = "PROTOCOL_ERROR"
        return result

    def run_plan_file(self, path: Path) -> SessionResult:
        plan_file = load_plan_file(path)
        session_plan = plan_file.session_plan
        if len(session_plan.regions) != 1:
            result = bootstrap_result(profile_name=session_plan.profile_name)
            result.session_id = session_plan.session_id
            result.session_nonce = session_plan.nonce
            result.verdict = "PROTOCOL_ERROR"
            result.notes.append("Current Phase 1 host runner requires exactly one region in the plan file.")
            return result

        region = session_plan.regions[0]
        if region.region_type == "host" and region.gpu_device is None:
            profile = BenchmarkProfile(
                name=session_plan.profile_name,
                benchmark_class="cold",
                target_devices={"host": True, "gpus": []},
                reserve_policy={"host_bytes": region.usable_bytes, "per_gpu_bytes": 0},
                host_target_fraction=1.0,
                per_gpu_target_fraction=0.0,
                porep_unit_profile=session_plan.porep_unit_profile,
                leaf_size=session_plan.challenge_leaf_size,
                challenge_policy=session_plan.challenge_policy.to_cbor_object(),
                deadline_policy=session_plan.deadline_policy.to_cbor_object(),
                cleanup_policy=session_plan.cleanup_policy.to_cbor_object(),
                repetition_count=1,
            )
        elif region.region_type == "gpu" and region.gpu_device is not None:
            profile = BenchmarkProfile(
                name=session_plan.profile_name,
                benchmark_class="cold",
                target_devices={"host": False, "gpus": [region.gpu_device]},
                reserve_policy={"host_bytes": 0, "per_gpu_bytes": region.usable_bytes},
                host_target_fraction=0.0,
                per_gpu_target_fraction=1.0,
                porep_unit_profile=session_plan.porep_unit_profile,
                leaf_size=session_plan.challenge_leaf_size,
                challenge_policy=session_plan.challenge_policy.to_cbor_object(),
                deadline_policy=session_plan.deadline_policy.to_cbor_object(),
                cleanup_policy=session_plan.cleanup_policy.to_cbor_object(),
                repetition_count=1,
            )
        else:
            result = bootstrap_result(profile_name=session_plan.profile_name)
            result.session_id = session_plan.session_id
            result.session_nonce = session_plan.nonce
            result.verdict = "PROTOCOL_ERROR"
            result.notes.append("Current runner only supports one host or one gpu region in plan files.")
            return result
        return self.run_session(
            profile,
            retain_session=plan_file.retain_session,
            session_plan=session_plan,
        )

    def run_placeholder(self, profile: BenchmarkProfile, note: str) -> SessionResult:
        result = bootstrap_result(profile_name=profile.name, note=note)
        result.challenge_leaf_size = profile.leaf_size
        result.challenge_count = profile.challenge_policy["max_challenges"]
        result.deadline_ms = profile.deadline_policy["response_deadline_ms"]
        return result

    def rechallenge(self, session_id: str, *, release: bool = False) -> SessionResult:
        try:
            record = load_resident_session(session_id)
        except FileNotFoundError as error:
            try:
                cold_result = load_cold_result_artifact(session_id)
            except FileNotFoundError:
                raise ProtocolError(f"Unknown or expired resident session: {session_id}") from error
            if not cold_result.resident_socket_path:
                raise ProtocolError(f"Unknown or expired resident session: {session_id}") from error
            record = ResidentSessionRecord(
                session_id=cold_result.session_id,
                profile_name=cold_result.profile_name,
                session_nonce=cold_result.session_nonce,
                session_plan_root=cold_result.session_plan_root,
                session_manifest_root=cold_result.session_manifest_root,
                region_id=next(iter(cold_result.region_roots)),
                region_root_hex=next(iter(cold_result.region_roots.values())),
                region_manifest_root=next(iter(cold_result.region_manifest_roots.values())),
                challenge_leaf_size=cold_result.challenge_leaf_size,
                challenge_policy=dict(cold_result.challenge_policy),
                deadline_ms=cold_result.deadline_ms,
                cleanup_policy=dict(cold_result.cleanup_policy),
                host_total_bytes=cold_result.host_total_bytes,
                host_usable_bytes=cold_result.host_usable_bytes,
                host_covered_bytes=cold_result.host_covered_bytes,
                real_porep_bytes=cold_result.real_porep_bytes,
                tail_filler_bytes=cold_result.tail_filler_bytes,
                real_porep_ratio=cold_result.real_porep_ratio,
                coverage_fraction=cold_result.coverage_fraction,
                inner_filecoin_verified=cold_result.inner_filecoin_verified,
                socket_path=cold_result.resident_socket_path,
                process_id=cold_result.resident_process_id,
                lease_expiry=cold_result.lease_expiry,
            )
        return run_host_rechallenge(record, release=release)

    def verify_record(self, path: Path) -> SessionResult:
        payload = load_json_file(path)
        return SessionResult.from_dict(payload)
