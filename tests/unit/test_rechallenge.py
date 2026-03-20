from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from pose.common.merkle import commit_payload
from pose.verifier.challenge import sample_leaf_indices
from pose.verifier.rechallenge import run_host_rechallenge
from pose.verifier.session_store import ResidentSessionRecord


def _record() -> tuple[ResidentSessionRecord, bytes]:
    payload = (b"a" * 16) + (b"b" * 16)
    commitment = commit_payload(payload, 16)
    record = ResidentSessionRecord(
        session_id="resident-session",
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
        socket_path="/tmp/fake.sock",
        process_id=9999,
        lease_expiry=(datetime.now(UTC) + timedelta(minutes=5)).isoformat(),
    )
    return record, payload


def test_rechallenge_succeeds_and_can_release(monkeypatch: pytest.MonkeyPatch) -> None:
    record, payload = _record()
    commitment = commit_payload(payload, 16)
    challenge_indices = sample_leaf_indices(
        2,
        1,
        seed=f"{record.session_manifest_root}:feedfacefeedface",
    )

    monkeypatch.setattr("pose.verifier.rechallenge.secrets.token_hex", lambda _n: "feedfacefeedface")
    monkeypatch.setattr(
        "pose.verifier.rechallenge.challenge_outer",
        lambda *_args, **_kwargs: (
            [
                {
                    "region_id": record.region_id,
                    "session_manifest_root": record.session_manifest_root,
                    "leaf_index": challenge_indices[0],
                    "leaf_hex": payload[challenge_indices[0] * 16 : (challenge_indices[0] + 1) * 16].hex(),
                    "sibling_hashes_hex": [
                        value.hex()
                        for value in commitment.opening(
                            challenge_indices[0],
                            payload[challenge_indices[0] * 16 : (challenge_indices[0] + 1) * 16],
                        ).sibling_hashes
                    ],
                }
            ],
            0,
        ),
    )
    monkeypatch.setattr("pose.verifier.rechallenge.finalize_session", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("pose.verifier.rechallenge.cleanup_session", lambda *_args, **_kwargs: "ZEROIZED_AND_RELEASED")

    result = run_host_rechallenge(record, release=True)

    assert result.success is True
    assert result.run_class == "rechallenge"
    assert result.verdict == "SUCCESS"
    assert result.cleanup_status == "ZEROIZED_AND_RELEASED"
    assert result.challenge_indices_by_region[record.region_id] == challenge_indices


def test_rechallenge_rejects_expired_session() -> None:
    record, _payload = _record()
    expired = ResidentSessionRecord(
        **{
            **record.to_dict(),
            "lease_expiry": (datetime.now(UTC) - timedelta(minutes=1)).isoformat(),
        }
    )

    result = run_host_rechallenge(expired)

    assert result.success is False
    assert result.verdict == "RESOURCE_FAILURE"
    assert result.cleanup_status == "LEASE_EXPIRED"
