from __future__ import annotations

from types import SimpleNamespace

from pose.graphs import build_graph_descriptor
from pose.protocol.messages import ChallengePolicy, CleanupPolicy, DeadlinePolicy, LeaseRecord, RegionPlan, SessionPlan
from pose.verifier.service import VerifierService


def _gpu_session_plan() -> SessionPlan:
    descriptor = build_graph_descriptor(
        label_count_m=8,
        graph_parameter_n=2,
        gamma=4,
        hash_backend="blake3-xof",
        label_width_bits=256,
    )
    return SessionPlan(
        session_id="adversarial-gpu-plan",
        session_seed_hex="99" * 32,
        profile_name="single-h100-hbm-max",
        graph_family="pose-db-drg-v1",
        graph_parameter_n=2,
        label_count_m=8,
        gamma=4,
        label_width_bits=256,
        hash_backend="blake3-xof",
        graph_descriptor_digest=descriptor.digest,
        challenge_policy=ChallengePolicy(
            rounds_r=4,
            target_success_bound=1.0e-9,
            sample_with_replacement=True,
        ),
        deadline_policy=DeadlinePolicy(response_deadline_us=2_500, session_timeout_ms=60_000),
        cleanup_policy=CleanupPolicy(zeroize=True, verify_zeroization=False),
        adversary_model="general",
        attacker_budget_bytes_assumed=16,
        q_bound=3,
        regions=[
            RegionPlan(
                region_id="gpu-0",
                region_type="gpu",
                usable_bytes=256,
                slot_count=8,
                covered_bytes=256,
                slack_bytes=0,
                gpu_device=0,
            )
        ],
        claim_notes=["adversarial-gpu"],
    )


class _FakeProcess:
    pid = 1234

    def poll(self) -> int:
        return 0

    def terminate(self) -> None:
        return None

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        return 0

    def kill(self) -> None:
        return None


def _fake_lease(plan: SessionPlan) -> SimpleNamespace:
    return SimpleNamespace(
        record=LeaseRecord(
            region_id="gpu-0",
            region_type="gpu",
            usable_bytes=plan.regions[0].usable_bytes,
            slot_count=plan.regions[0].slot_count,
            slack_bytes=plan.regions[0].slack_bytes,
            lease_handle="cuda-ipc:0:ZmFrZQ==",
            lease_expiry="2099-01-01T00:00:00+00:00",
            cleanup_policy=plan.cleanup_policy,
            gpu_device=0,
        ),
        close=lambda: None,
    )


def _materialization_report(
    plan: SessionPlan,
    *,
    declared_stage_copy_bytes: int,
) -> tuple[dict[str, object], dict[str, int]]:
    return (
        {
            "graph_descriptor_digest": plan.graph_descriptor_digest,
            "scratch_peak_bytes": 0,
            "regions": {
                "gpu-0": {
                    "covered_bytes": plan.regions[0].covered_bytes,
                    "slack_bytes": plan.regions[0].slack_bytes,
                    "declared_stage_copy_bytes": declared_stage_copy_bytes,
                }
            },
        },
        {"label_generation": 1, "copy_to_hbm": 1},
    )


def test_hidden_hbm_shadow_is_reported_and_rejected(monkeypatch) -> None:
    plan = _gpu_session_plan()

    monkeypatch.setattr(
        "pose.verifier.service.start_ephemeral_prover_server",
        lambda **_kwargs: _FakeProcess(),
    )
    monkeypatch.setattr(
        "pose.verifier.service.discover",
        lambda _socket_path: {"protocol_version": "pose-grpc/v1", "capabilities": ["pose-db-fast-phase"]},
    )
    monkeypatch.setattr(
        "pose.verifier.service._create_runtime_lease",
        lambda _plan, _region: _fake_lease(plan),
    )
    monkeypatch.setattr("pose.verifier.service.plan_session", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("pose.verifier.service.lease_regions", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("pose.verifier.service.seed_session", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "pose.verifier.service.materialize_labels",
        lambda *_args, **_kwargs: _materialization_report(
            plan,
            declared_stage_copy_bytes=64,
        ),
    )
    monkeypatch.setattr(
        "pose.verifier.service.cleanup_session",
        lambda *_args, **_kwargs: "ZEROIZED_AND_RELEASED",
    )
    monkeypatch.setattr(
        "pose.verifier.service._release_runtime_leases",
        lambda *_args, **_kwargs: None,
    )

    result = VerifierService()._run_session_plan(plan, retain_session=False, extra_notes=[])

    assert result.verdict == "PROTOCOL_ERROR"
    assert result.declared_stage_copy_bytes == 64
    assert any("GPU HBM regions" in note for note in result.operational_claim_notes)
    assert any("surviving stage copies into the fast phase totaling 64 bytes" in note for note in result.operational_claim_notes)
    assert any("Declared stage copies survive into the fast phase" in note for note in result.notes)
