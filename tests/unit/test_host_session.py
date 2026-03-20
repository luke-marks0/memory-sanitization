from __future__ import annotations

from pose.benchmarks.profiles import load_profile
from pose.filecoin.reference import SealArtifact
from pose.protocol.messages import ChallengePolicy, CleanupPolicy, DeadlinePolicy, RegionPlan, SessionPlan
from pose.verifier.host_session import run_host_session
from pose.verifier.host_planning import build_host_sector_plan


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


def test_host_session_succeeds_for_minimal_host_profile() -> None:
    profile = load_profile("dev-small")
    result = run_host_session(profile, reference=FakeReference())

    assert result.success is True
    assert result.verdict == "SUCCESS"
    assert result.inner_filecoin_verified is True
    assert result.outer_pose_verified is True
    assert result.host_total_bytes >= result.host_usable_bytes == result.host_covered_bytes
    assert result.real_porep_ratio == 1.0
    assert result.session_manifest_root
    assert result.region_roots["host-0"]
    assert result.region_manifest_roots["host-0"]
    assert result.challenge_indices_by_region["host-0"] == [0]


def test_host_session_supports_multiple_units_in_exact_fit_mode() -> None:
    profile = load_profile("dev-small")
    result = run_host_session(profile, reference=FakeReference(), requested_unit_count=2)

    assert result.success is True
    assert result.verdict == "SUCCESS"
    assert result.host_total_bytes >= 8192
    assert result.host_usable_bytes == 8192
    assert result.host_covered_bytes == 8192
    assert result.real_porep_bytes == 8192
    assert result.challenge_count == 2
    assert result.challenge_indices_by_region["host-0"] == [0, 1]


def test_host_session_rejects_profiles_that_need_full_cache() -> None:
    profile = load_profile("single-h100-host-max")
    result = run_host_session(profile, reference=FakeReference())

    assert result.success is False
    assert result.verdict == "RESOURCE_FAILURE"
    assert any("minimal PoRep unit profile" in note for note in result.notes)


def test_host_session_accepts_explicit_session_plan() -> None:
    profile = load_profile("dev-small")
    session_plan = SessionPlan(
        session_id="planned-session",
        nonce="planned-nonce",
        profile_name=profile.name,
        porep_unit_profile=profile.porep_unit_profile,
        challenge_leaf_size=profile.leaf_size,
        challenge_policy=ChallengePolicy(**profile.challenge_policy),
        deadline_policy=DeadlinePolicy(**profile.deadline_policy),
        cleanup_policy=CleanupPolicy(**profile.cleanup_policy),
        unit_count=1,
        regions=[RegionPlan(region_id="planned-host", region_type="host", usable_bytes=4096)],
        sector_plan=build_host_sector_plan("planned-session", "planned-host", 1),
    )

    result = run_host_session(profile, reference=FakeReference(), session_plan=session_plan)

    assert result.success is True
    assert result.session_id == "planned-session"
    assert result.session_nonce == "planned-nonce"
    assert result.region_roots["planned-host"]
    assert result.challenge_indices_by_region["planned-host"] == [0]


def test_host_session_uses_profile_host_budget_for_default_planning(monkeypatch) -> None:
    profile = load_profile("dev-small")
    monkeypatch.setattr(
        "pose.verifier.host_session.detect_host_memory_bytes",
        lambda: 1024 * 1024 * 1024,
    )

    result = run_host_session(profile, reference=FakeReference())

    assert result.success is True
    assert result.host_total_bytes == 1024 * 1024 * 1024
    assert result.host_usable_bytes == 4096
    assert result.host_covered_bytes == 4096
