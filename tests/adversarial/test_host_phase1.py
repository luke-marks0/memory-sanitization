from __future__ import annotations

from dataclasses import replace

import pytest

from pose.benchmarks.profiles import BenchmarkProfile, load_profile
from pose.common.errors import ResourceFailure
from pose.filecoin.reference import SealArtifact
from pose.verifier import host_session as host_session_mod
from pose.verifier.host_session import run_host_session
from pose.verifier.leasing import HostLease


class FakeReference:
    def __init__(self) -> None:
        self._artifact = SealArtifact(
            status="phase0-real-filecoin-bridge",
            verified_after_seal=True,
            sector_size=2048,
            api_version="V1_2_0",
            registered_seal_proof=5,
            porep_id_hex="05" + ("00" * 31),
            prover_id_hex="07" * 32,
            sector_id=4242,
            ticket_hex="01" * 32,
            seed_hex="02" * 32,
            piece_size=2032,
            piece_commitment_hex="11" * 32,
            comm_d_hex="22" * 32,
            comm_r_hex="33" * 32,
            proof_hex="aabbccddeeff",
            inner_timings_ms={
                "seal_pre_commit_phase1": 17,
                "seal_pre_commit_phase2": 19,
                "seal_commit_phase1": 23,
                "seal_commit_phase2": 29,
                "verify_seal": 31,
            },
        )

    def seal(self, _request=None) -> SealArtifact:
        return self._artifact

    def verify(self, artifact: SealArtifact) -> bool:
        return artifact.comm_r_hex == self._artifact.comm_r_hex


def _profile_with_updates(
    profile: BenchmarkProfile,
    *,
    deadline_ms: int | None = None,
    verify_zeroization: bool | None = None,
) -> BenchmarkProfile:
    payload = profile.to_dict()
    if deadline_ms is not None:
        payload["deadline_policy"] = {
            **profile.deadline_policy,
            "response_deadline_ms": deadline_ms,
        }
    if verify_zeroization is not None:
        payload["cleanup_policy"] = {
            **profile.cleanup_policy,
            "verify_zeroization": verify_zeroization,
        }
    return BenchmarkProfile.from_dict(payload)


def test_replayed_old_openings_under_new_session_nonce_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = load_profile("dev-small")
    captured_openings: list[dict[str, object]] = []
    original_open = host_session_mod._open_locally

    def capture_open(**kwargs):
        openings, response_ms = original_open(**kwargs)
        captured_openings[:] = openings
        return openings, response_ms

    monkeypatch.setattr(host_session_mod, "_open_locally", capture_open)
    first = run_host_session(profile, reference=FakeReference())
    assert first.verdict == "SUCCESS"

    monkeypatch.setattr(
        host_session_mod,
        "_open_locally",
        lambda **kwargs: (captured_openings, 0),
    )
    replay = run_host_session(profile, reference=FakeReference())

    assert replay.success is False
    assert replay.verdict == "OUTER_PROOF_INVALID"
    assert replay.inner_filecoin_verified is True


def test_correct_inner_proof_with_wrong_outer_bytes_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = load_profile("dev-small")
    original_materialize = host_session_mod._materialize_locally

    def tamper_materialized_region(**kwargs):
        artifacts, region_manifest, payload, object_serialization_ms = original_materialize(**kwargs)
        lease = kwargs["lease"]
        corrupted = bytearray(lease.read())
        corrupted[0] ^= 0xFF
        lease.write(bytes(corrupted))
        return artifacts, region_manifest, payload, object_serialization_ms

    monkeypatch.setattr(host_session_mod, "_materialize_locally", tamper_materialized_region)
    result = run_host_session(profile, reference=FakeReference())

    assert result.success is False
    assert result.verdict == "OUTER_PROOF_INVALID"
    assert result.inner_filecoin_verified is True


def test_partial_overwrite_of_a_leased_region_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = load_profile("dev-small")
    original_materialize = host_session_mod._materialize_locally

    def partially_overwrite_region(**kwargs):
        artifacts, region_manifest, payload, object_serialization_ms = original_materialize(**kwargs)
        lease = kwargs["lease"]
        offset = profile.leaf_size + 128
        lease.mapping[offset : offset + 64] = b"\x99" * 64
        lease.mapping.flush()
        return artifacts, region_manifest, payload, object_serialization_ms

    monkeypatch.setattr(host_session_mod, "_materialize_locally", partially_overwrite_region)
    result = run_host_session(profile, reference=FakeReference(), requested_unit_count=2)

    assert result.success is False
    assert result.verdict == "OUTER_PROOF_INVALID"
    assert result.challenge_indices_by_region["host-0"] == [0, 1]


def test_sparse_host_region_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    profile = load_profile("dev-small")
    original_materialize = host_session_mod._materialize_locally

    def sparsify_region(**kwargs):
        artifacts, region_manifest, payload, object_serialization_ms = original_materialize(**kwargs)
        lease = kwargs["lease"]
        lease.mapping[profile.leaf_size : 2 * profile.leaf_size] = b"\x00" * profile.leaf_size
        lease.mapping.flush()
        return artifacts, region_manifest, payload, object_serialization_ms

    monkeypatch.setattr(host_session_mod, "_materialize_locally", sparsify_region)
    result = run_host_session(profile, reference=FakeReference(), requested_unit_count=2)

    assert result.success is False
    assert result.verdict == "OUTER_PROOF_INVALID"


def test_post_challenge_copy_in_timeout_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = _profile_with_updates(load_profile("dev-small"), deadline_ms=1)
    original_open = host_session_mod._open_locally

    def delayed_open(**kwargs):
        openings, _ = original_open(**kwargs)
        return openings, 2

    monkeypatch.setattr(host_session_mod, "_open_locally", delayed_open)
    result = run_host_session(profile, reference=FakeReference())

    assert result.success is False
    assert result.verdict == "TIMEOUT"
    assert result.inner_filecoin_verified is True


def test_insufficient_coverage_with_tail_filler_is_rejected() -> None:
    profile = load_profile("dev-small")
    result = run_host_session(
        profile,
        reference=FakeReference(),
        requested_unit_count=1,
        requested_host_bytes=8192,
    )

    assert result.success is False
    assert result.verdict == "COVERAGE_BELOW_THRESHOLD"
    assert result.outer_pose_verified is True
    assert result.inner_filecoin_verified is True
    assert result.real_porep_bytes == 4096
    assert result.tail_filler_bytes == 4096
    assert result.real_porep_ratio == 0.5


def test_mismatch_between_declared_and_actual_payload_length_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = load_profile("dev-small")
    original_materialize = host_session_mod._materialize_locally

    def mismatch_payload_length(**kwargs):
        artifacts, region_manifest, payload, object_serialization_ms = original_materialize(**kwargs)
        bad_manifest = replace(
            region_manifest,
            payload_length_bytes=region_manifest.payload_length_bytes - profile.leaf_size,
        )
        return artifacts, bad_manifest, payload, object_serialization_ms

    monkeypatch.setattr(host_session_mod, "_materialize_locally", mismatch_payload_length)
    result = run_host_session(profile, reference=FakeReference())

    assert result.success is False
    assert result.verdict == "OUTER_PROOF_INVALID"


def test_cleanup_failure_is_reported_explicitly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = _profile_with_updates(load_profile("dev-small"), verify_zeroization=True)
    monkeypatch.setattr(HostLease, "verify_zeroized", lambda self: False)

    result = run_host_session(profile, reference=FakeReference())

    assert result.success is False
    assert result.verdict == "CLEANUP_FAILURE"
    assert result.cleanup_status == "CLEANUP_FAILED"
    assert any("zeroization verification failed" in note for note in result.notes)
