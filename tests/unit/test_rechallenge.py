from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from pose.graphs import build_graph_descriptor, build_pose_db_graph, compute_challenge_labels
from pose.verifier.rechallenge import run_host_rechallenge
from pose.verifier.session_store import ResidentSessionRecord


def _record() -> ResidentSessionRecord:
    descriptor = build_graph_descriptor(
        label_count_m=8,
        graph_parameter_n=2,
        gamma=4,
        hash_backend="blake3-xof",
        label_width_bits=256,
    )
    return ResidentSessionRecord(
        session_id="resident-session",
        profile_name="dev-small",
        session_seed_hex="aa" * 32,
        session_plan_root=descriptor.digest,
        graph_family="pose-db-drg-v1",
        graph_parameter_n=2,
        graph_descriptor_digest=descriptor.digest,
        label_width_bits=256,
        label_count_m=8,
        gamma=4,
        hash_backend="blake3-xof",
        region_id="host-0",
        region_slot_count=8,
        challenge_policy={"rounds_r": 2, "sample_with_replacement": True, "target_success_bound": 1e-9},
        deadline_us=500_000,
        cleanup_policy={"zeroize": True, "verify_zeroization": False},
        adversary_model="general",
        attacker_budget_bytes_assumed=16,
        q_bound=3,
        host_total_bytes=1024,
        host_usable_bytes=256,
        host_covered_bytes=256,
        covered_bytes=256,
        slack_bytes=0,
        coverage_fraction=1.0,
        scratch_peak_bytes=1024,
        declared_stage_copy_bytes=0,
        formal_claim_notes=["formal"],
        operational_claim_notes=["operational"],
        claim_notes=["resident"],
        socket_path="/tmp/resident-pose.sock",
        process_id=9999,
        lease_expiry=(datetime.now(UTC) + timedelta(minutes=5)).isoformat(),
    )


def test_rechallenge_succeeds_and_releases(monkeypatch: pytest.MonkeyPatch) -> None:
    record = _record()
    challenge_indices = [5, 1]
    graph = build_pose_db_graph(
        label_count_m=record.label_count_m,
        graph_parameter_n=record.graph_parameter_n,
        gamma=record.gamma,
        hash_backend=record.hash_backend,
        label_width_bits=record.label_width_bits,
    )
    expected_labels = compute_challenge_labels(
        graph,
        session_seed=record.session_seed_hex,
        challenge_indices=challenge_indices,
        label_engine="reference",
    )

    cleanup_calls: list[str] = []
    terminate_calls: list[int] = []
    delete_calls: list[str] = []

    class _FakeFastPhaseClient:
        def __init__(self, socket_path: str) -> None:
            assert socket_path == record.socket_path
            self._responses = iter(
                [
                    {
                        "challenge_index": challenge_indices[0],
                        "label_bytes": expected_labels[0],
                        "round_trip_us": 10,
                    },
                    {
                        "challenge_index": challenge_indices[1],
                        "label_bytes": expected_labels[1],
                        "round_trip_us": 12,
                    },
                ]
            )

        def __enter__(self) -> "_FakeFastPhaseClient":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            del exc_type, exc, tb

        def run_round(self, *, session_id: str, round_index: int, challenge_index: int):
            assert session_id == record.session_id
            assert challenge_index == challenge_indices[round_index]
            return next(self._responses)

    sampled_values = iter(challenge_indices)
    monkeypatch.setattr(
        "pose.verifier.challenges.secrets.randbelow",
        lambda upper_bound: next(sampled_values),
    )
    monkeypatch.setattr("pose.verifier.rechallenge.discover", lambda _socket: {"capabilities": ["pose-db-fast-phase"]})
    monkeypatch.setattr("pose.verifier.rechallenge.FastPhaseClient", _FakeFastPhaseClient)
    monkeypatch.setattr("pose.verifier.rechallenge.finalize_session", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "pose.verifier.rechallenge.cleanup_session",
        lambda _socket, session_id: cleanup_calls.append(session_id) or "ZEROIZED_AND_RELEASED",
    )
    monkeypatch.setattr(
        "pose.verifier.rechallenge.delete_resident_session",
        lambda session_id: delete_calls.append(session_id),
    )
    monkeypatch.setattr(
        "pose.verifier.rechallenge._terminate_resident_process",
        lambda process_id: terminate_calls.append(process_id),
    )

    result = run_host_rechallenge(record, release=True)

    assert result.success is True
    assert result.run_class == "rechallenge"
    assert result.verdict == "SUCCESS"
    assert result.cleanup_status == "ZEROIZED_AND_RELEASED"
    assert result.accepted_rounds == 2
    assert cleanup_calls == [record.session_id]
    assert delete_calls == [record.session_id]
    assert terminate_calls == [record.process_id]


def test_rechallenge_rejects_expired_session(monkeypatch: pytest.MonkeyPatch) -> None:
    record = _record()
    expired = ResidentSessionRecord(
        **{
            **record.to_dict(),
            "lease_expiry": (datetime.now(UTC) - timedelta(minutes=1)).isoformat(),
        }
    )

    cleanup_calls: list[str] = []
    terminate_calls: list[int] = []
    delete_calls: list[str] = []
    monkeypatch.setattr(
        "pose.verifier.rechallenge.cleanup_session",
        lambda _socket, session_id: cleanup_calls.append(session_id) or "ZEROIZED_AND_RELEASED",
    )
    monkeypatch.setattr(
        "pose.verifier.rechallenge.delete_resident_session",
        lambda session_id: delete_calls.append(session_id),
    )
    monkeypatch.setattr(
        "pose.verifier.rechallenge._terminate_resident_process",
        lambda process_id: terminate_calls.append(process_id),
    )

    result = run_host_rechallenge(expired)

    assert result.success is False
    assert result.verdict == "RESOURCE_FAILURE"
    assert result.cleanup_status == "ZEROIZED_AND_RELEASED"
    assert cleanup_calls == [record.session_id]
    assert delete_calls == [record.session_id]
    assert terminate_calls == [record.process_id]
