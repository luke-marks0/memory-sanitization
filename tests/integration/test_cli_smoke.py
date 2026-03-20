from __future__ import annotations

import json
import os
import subprocess
import sys
from concurrent import futures
from datetime import UTC, datetime, timedelta
from pathlib import Path

import grpc

from pose.common.merkle import commit_payload
from pose.protocol.grpc_codec import GRPC_PROTOCOL_VERSION
from pose.protocol.result_schema import bootstrap_result
from pose.verifier.session_store import ResidentSessionRecord, delete_resident_session, write_resident_session
from pose.v1 import session_pb2, session_pb2_grpc


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root / "src")
    return subprocess.run(
        [sys.executable, "-m", "pose.cli.main", *args],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_prover_inspect_succeeds() -> None:
    result = run_cli("prover", "inspect")
    assert result.returncode == 0
    assert "supports_real_filecoin_reference" in result.stdout


def test_bench_matrix_succeeds() -> None:
    result = run_cli("bench", "matrix", "--profiles", "bench_profiles/")
    assert result.returncode == 0
    assert "dev-small" in result.stdout


def test_verifier_verify_record_succeeds_for_valid_bootstrap_artifact(tmp_path: Path) -> None:
    payload = bootstrap_result("dev-small").to_dict()
    payload["session_id"] = "test-session"
    record_path = tmp_path / "result.json"
    record_path.write_text(json.dumps(payload), encoding="utf-8")

    result = run_cli("verifier", "verify-record", str(record_path))
    assert result.returncode == 0
    assert '"status": "valid"' in result.stdout


def test_verifier_verify_record_rejects_missing_required_fields(tmp_path: Path) -> None:
    payload = bootstrap_result("dev-small").to_dict()
    payload.pop("host_total_bytes")
    record_path = tmp_path / "invalid-result.json"
    record_path.write_text(json.dumps(payload), encoding="utf-8")

    result = run_cli("verifier", "verify-record", str(record_path))
    assert result.returncode == 2
    assert "host_total_bytes" in result.stderr


def test_verifier_rechallenge_succeeds_for_resident_session(tmp_path: Path) -> None:
    session_id = "cli-rechallenge-session"
    socket_path = tmp_path / "resident.sock"
    payload = (b"a" * 16) + (b"b" * 16)
    commitment = commit_payload(payload, 16)

    class FakeServicer(session_pb2_grpc.PoseSessionServiceServicer):
        def ChallengeOuter(self, request, _context):
            challenge = request.challenges[0]
            openings = []
            for index in challenge.leaf_indices:
                start = index * 16
                leaf = payload[start : start + 16]
                opening = commitment.opening(index, leaf)
                openings.append(
                    session_pb2.OuterOpening(
                        region_id="host-0",
                        session_manifest_root=challenge.session_manifest_root,
                        leaf_index=index,
                        leaf_bytes=leaf,
                        auth_path=list(opening.sibling_hashes),
                    )
                )
            return session_pb2.ChallengeOuterResponse(openings=openings, response_ms=0)

        def Finalize(self, request, _context):
            assert request.protocol_version == GRPC_PROTOCOL_VERSION
            return session_pb2.FinalizeResponse(accepted=True)

        def Cleanup(self, request, _context):
            assert request.protocol_version == GRPC_PROTOCOL_VERSION
            return session_pb2.CleanupResponse(cleaned=True, cleanup_status="ZEROIZED_AND_RELEASED")

    if socket_path.exists():
        socket_path.unlink()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    session_pb2_grpc.add_PoseSessionServiceServicer_to_server(FakeServicer(), server)
    server.add_insecure_port(f"unix:{socket_path}")
    server.start()
    record = ResidentSessionRecord(
        session_id=session_id,
        profile_name="dev-small",
        session_nonce="session-nonce",
        session_plan_root="plan-root",
        session_manifest_root="manifest-root",
        region_id="host-0",
        region_root_hex=commitment.root_hex,
        region_manifest_root="region-manifest-root",
        challenge_leaf_size=16,
        challenge_policy={"epsilon": 0.5, "lambda_bits": 1, "max_challenges": 1},
        deadline_ms=5000,
        cleanup_policy={"zeroize": True, "verify_zeroization": False},
        host_total_bytes=len(payload),
        host_usable_bytes=len(payload),
        host_covered_bytes=len(payload),
        real_porep_bytes=len(payload),
        tail_filler_bytes=0,
        real_porep_ratio=1.0,
        coverage_fraction=1.0,
        inner_filecoin_verified=True,
        socket_path=str(socket_path),
        process_id=1234,
        lease_expiry=(datetime.now(UTC) + timedelta(minutes=5)).isoformat(),
    )
    write_resident_session(record)

    try:
        result = run_cli("verifier", "rechallenge", "--session-id", session_id, "--release")
        assert result.returncode == 0
        assert '"run_class": "rechallenge"' in result.stdout
        assert '"verdict": "SUCCESS"' in result.stdout
    finally:
        delete_resident_session(session_id)
        server.stop(grace=None)
