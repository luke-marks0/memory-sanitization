from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path

import grpc
import pytest

from pose.graphs import build_graph_descriptor
from pose.protocol.grpc_codec import GRPC_PROTOCOL_VERSION, session_plan_to_proto
from pose.protocol.messages import ChallengePolicy, CleanupPolicy, DeadlinePolicy, RegionPlan, SessionPlan
from pose.prover.grpc_service import PoseSessionServicer
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


def _session_plan(session_id: str = "session-id") -> SessionPlan:
    descriptor = build_graph_descriptor(
        label_count_m=8,
        graph_parameter_n=2,
        gamma=4,
        hash_backend="blake3-xof",
        label_width_bits=256,
    )
    return SessionPlan(
        session_id=session_id,
        session_seed_hex="ab" * 32,
        profile_name="dev-small",
        graph_family="pose-db-drg-v1",
        graph_parameter_n=2,
        label_count_m=8,
        gamma=4,
        label_width_bits=256,
        hash_backend="blake3-xof",
        graph_descriptor_digest=descriptor.digest,
        challenge_policy=ChallengePolicy(
            rounds_r=4,
            target_success_bound=1e-9,
            sample_with_replacement=True,
        ),
        deadline_policy=DeadlinePolicy(response_deadline_us=50_000, session_timeout_ms=60_000),
        cleanup_policy=CleanupPolicy(zeroize=True, verify_zeroization=False),
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
        claim_notes=["unit-test"],
    )


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


def test_session_plan_root_changes_when_deadline_policy_changes() -> None:
    baseline = _session_plan()
    updated = replace(
        baseline,
        deadline_policy=DeadlinePolicy(response_deadline_us=60_000, session_timeout_ms=60_000),
    )

    assert "deadline_policy" in baseline.to_cbor_object()
    assert baseline.plan_root_hex != updated.plan_root_hex


class _AbortCalled(Exception):
    def __init__(self, code: grpc.StatusCode, details: str) -> None:
        super().__init__(details)
        self.code = code
        self.details = details


class _FakeContext:
    def abort(self, code: grpc.StatusCode, details: str) -> None:
        raise _AbortCalled(code, details)


def test_grpc_plan_session_rejects_duplicate_session_ids() -> None:
    servicer = PoseSessionServicer()
    plan = _session_plan(session_id="duplicate-session")
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


def test_grpc_plan_session_rejects_inconsistent_region_geometry() -> None:
    servicer = PoseSessionServicer()
    invalid_region = replace(_session_plan().regions[0], covered_bytes=224, slack_bytes=32)
    request = session_pb2.PlanSessionRequest(
        protocol_version=GRPC_PROTOCOL_VERSION,
        plan=session_plan_to_proto(replace(_session_plan(), regions=[invalid_region])),
    )

    with pytest.raises(_AbortCalled) as error:
        servicer.PlanSession(request, _FakeContext())

    assert error.value.code == grpc.StatusCode.INVALID_ARGUMENT
    assert "covered_bytes must equal slot_count * label_width_bytes" in error.value.details
