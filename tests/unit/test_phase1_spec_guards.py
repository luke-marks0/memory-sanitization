from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path

import grpc
import pytest

from pose.protocol.grpc_codec import GRPC_PROTOCOL_VERSION, session_plan_to_proto
from pose.protocol.messages import ChallengePolicy, CleanupPolicy, DeadlinePolicy, RegionPlan, SessionPlan
from pose.protocol.region_payloads import RegionManifest, SessionManifest
from pose.prover.grpc_service import PoseSessionServicer
from pose.verifier.host_planning import build_host_sector_plan
from pose.v1 import session_pb2


def _package_imports(package_name: str) -> list[tuple[Path, str]]:
    root = Path(__file__).resolve().parents[2] / "src" / "pose" / package_name
    imports: list[tuple[Path, str]] = []
    for path in sorted(root.rglob("*.py")):
        module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(module):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append((path, alias.name))
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imports.append((path, node.module))
    return imports


def test_verifier_package_has_no_prover_imports() -> None:
    offending = [
        f"{path.relative_to(path.parents[3])}: {module}"
        for path, module in _package_imports("verifier")
        if module.startswith("pose.prover")
    ]
    assert offending == []


def test_prover_package_has_no_verifier_imports() -> None:
    offending = [
        f"{path.relative_to(path.parents[3])}: {module}"
        for path, module in _package_imports("prover")
        if module.startswith("pose.verifier")
    ]
    assert offending == []


def test_session_manifest_root_changes_when_session_timeout_changes() -> None:
    region_manifest = RegionManifest(
        region_id="host-0",
        region_type="host",
        usable_bytes=4096,
        leaf_size=4096,
        payload_length_bytes=4096,
        real_porep_bytes=4096,
        tail_filler_bytes=0,
        unit_count=1,
        unit_digests_hex=("00" * 32,),
        payload_sha256_hex="11" * 32,
        merkle_root_hex="22" * 32,
    )
    baseline = SessionManifest(
        session_id="session-id",
        nonce="session-nonce",
        profile_name="dev-small",
        payload_profile="minimal",
        leaf_size=4096,
        deadline_policy={"response_deadline_ms": 5000, "session_timeout_ms": 60000},
        challenge_policy={"epsilon": 0.01, "lambda_bits": 32, "max_challenges": 64},
        cleanup_policy={"zeroize": True, "verify_zeroization": False},
        region_manifests=(region_manifest,),
    )
    updated = replace(
        baseline,
        deadline_policy={"response_deadline_ms": 5000, "session_timeout_ms": 61000},
    )

    assert "deadline_policy" in baseline.to_cbor_object()
    assert baseline.manifest_root_hex != updated.manifest_root_hex


def test_session_plan_root_changes_when_sector_plan_changes() -> None:
    baseline = SessionPlan(
        session_id="session-id",
        nonce="session-nonce",
        profile_name="dev-small",
        porep_unit_profile="minimal",
        challenge_leaf_size=4096,
        challenge_policy=ChallengePolicy(epsilon=0.01, lambda_bits=32, max_challenges=64),
        deadline_policy=DeadlinePolicy(response_deadline_ms=5000, session_timeout_ms=60000),
        cleanup_policy=CleanupPolicy(zeroize=True, verify_zeroization=False),
        unit_count=1,
        regions=[RegionPlan(region_id="host-0", region_type="host", usable_bytes=4096)],
        sector_plan=build_host_sector_plan("session-id", "host-0", 1),
    )
    updated = replace(
        baseline,
        sector_plan=build_host_sector_plan("session-id-2", "host-0", 1),
    )

    assert "sector_plan" in baseline.to_cbor_object()
    assert baseline.plan_root_hex != updated.plan_root_hex


class _AbortCalled(Exception):
    def __init__(self, code: grpc.StatusCode, details: str) -> None:
        super().__init__(details)
        self.code = code
        self.details = details


class _FakeContext:
    def abort(self, code: grpc.StatusCode, details: str) -> None:
        raise _AbortCalled(code, details)


def test_grpc_plan_session_rejects_duplicate_session_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "pose.prover.grpc_service.VendoredFilecoinReference",
        lambda: object(),
    )
    servicer = PoseSessionServicer()
    plan = SessionPlan(
        session_id="duplicate-session",
        nonce="nonce",
        profile_name="dev-small",
        porep_unit_profile="minimal",
        challenge_leaf_size=4096,
        challenge_policy=ChallengePolicy(epsilon=0.01, lambda_bits=32, max_challenges=64),
        deadline_policy=DeadlinePolicy(response_deadline_ms=5000, session_timeout_ms=60000),
        cleanup_policy=CleanupPolicy(zeroize=True, verify_zeroization=False),
        unit_count=1,
        regions=[RegionPlan(region_id="host-0", region_type="host", usable_bytes=4096)],
        sector_plan=build_host_sector_plan("duplicate-session", "host-0", 1),
    )
    request = session_pb2.PlanSessionRequest(
        protocol_version=GRPC_PROTOCOL_VERSION,
        plan=session_plan_to_proto(plan),
    )

    first = servicer.PlanSession(request, _FakeContext())
    assert first.accepted is True

    with pytest.raises(_AbortCalled) as error:
        servicer.PlanSession(request, _FakeContext())

    assert error.value.code == grpc.StatusCode.INVALID_ARGUMENT
    assert "Duplicate session id" in error.value.details


def test_grpc_plan_session_rejects_missing_sector_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "pose.prover.grpc_service.VendoredFilecoinReference",
        lambda: object(),
    )
    servicer = PoseSessionServicer()
    plan = SessionPlan(
        session_id="missing-sector-plan",
        nonce="nonce",
        profile_name="dev-small",
        porep_unit_profile="minimal",
        challenge_leaf_size=4096,
        challenge_policy=ChallengePolicy(epsilon=0.01, lambda_bits=32, max_challenges=64),
        deadline_policy=DeadlinePolicy(response_deadline_ms=5000, session_timeout_ms=60000),
        cleanup_policy=CleanupPolicy(zeroize=True, verify_zeroization=False),
        unit_count=1,
        regions=[RegionPlan(region_id="host-0", region_type="host", usable_bytes=4096)],
    )
    request = session_pb2.PlanSessionRequest(
        protocol_version=GRPC_PROTOCOL_VERSION,
        plan=session_plan_to_proto(plan),
    )

    with pytest.raises(_AbortCalled) as error:
        servicer.PlanSession(request, _FakeContext())

    assert error.value.code == grpc.StatusCode.INVALID_ARGUMENT
    assert "sector-plan entry" in error.value.details
