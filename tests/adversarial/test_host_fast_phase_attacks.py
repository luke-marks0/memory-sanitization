from __future__ import annotations

from types import SimpleNamespace

from pose.graphs import build_graph_descriptor, build_pose_db_graph, compute_challenge_labels
from pose.protocol.messages import ChallengePolicy, CleanupPolicy, DeadlinePolicy, LeaseRecord, RegionPlan, SessionPlan
from pose.verifier.service import VerifierService


def _session_plan() -> SessionPlan:
    descriptor = build_graph_descriptor(
        label_count_m=8,
        graph_parameter_n=2,
        gamma=4,
        hash_backend="blake3-xof",
        label_width_bits=256,
    )
    return SessionPlan(
        session_id="adversarial-host-plan",
        session_seed_hex="77" * 32,
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
                region_id="host-0",
                region_type="host",
                usable_bytes=256,
                slot_count=8,
                covered_bytes=256,
                slack_bytes=0,
            )
        ],
        claim_notes=["adversarial-host"],
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
            region_id="host-0",
            region_type="host",
            usable_bytes=plan.regions[0].usable_bytes,
            slot_count=plan.regions[0].slot_count,
            slack_bytes=plan.regions[0].slack_bytes,
            lease_handle="fake-lease",
            lease_expiry="2099-01-01T00:00:00+00:00",
            cleanup_policy=plan.cleanup_policy,
        ),
        close=lambda: None,
    )


def _materialization_report(
    plan: SessionPlan,
    *,
    scratch_peak_bytes: int = 0,
    declared_stage_copy_bytes: int = 0,
) -> tuple[dict[str, object], dict[str, int]]:
    return (
        {
            "graph_descriptor_digest": plan.graph_descriptor_digest,
            "scratch_peak_bytes": scratch_peak_bytes,
            "regions": {
                "host-0": {
                    "covered_bytes": plan.regions[0].covered_bytes,
                    "slack_bytes": plan.regions[0].slack_bytes,
                    "declared_stage_copy_bytes": declared_stage_copy_bytes,
                }
            },
        },
        {"label_generation": 1},
    )


def _configure_fake_runtime(
    monkeypatch,
    plan: SessionPlan,
    *,
    challenge_indices: list[int],
    round_responses: list[dict[str, object]] | None = None,
    declared_stage_copy_bytes: int = 0,
) -> None:
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
            declared_stage_copy_bytes=declared_stage_copy_bytes,
        ),
    )
    monkeypatch.setattr("pose.verifier.service.prepare_fast_phase", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("pose.verifier.service.finalize_session", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "pose.verifier.service.cleanup_session",
        lambda *_args, **_kwargs: "ZEROIZED_AND_RELEASED",
    )
    monkeypatch.setattr(
        "pose.verifier.service._release_runtime_leases",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "pose.verifier.service._sample_challenge_indices",
        lambda _plan: list(challenge_indices),
    )
    if round_responses is not None:
        class _FakeFastPhaseClient:
            def __init__(self, _socket_path: str) -> None:
                self._responses = iter(round_responses)

            def __enter__(self) -> "_FakeFastPhaseClient":
                return self

            def __exit__(self, exc_type, exc, tb) -> None:
                del exc_type, exc, tb

            def run_round(
                self,
                *,
                session_id: str,
                round_index: int,
                challenge_index: int,
            ) -> dict[str, object]:
                del session_id, round_index, challenge_index
                return next(self._responses)

        monkeypatch.setattr("pose.verifier.service.FastPhaseClient", _FakeFastPhaseClient)


def test_recomputation_on_demand_is_detected_as_deadline_miss(monkeypatch) -> None:
    plan = _session_plan()
    challenge_indices = [0, 3, 7, 3]
    graph = build_pose_db_graph(
        label_count_m=plan.label_count_m,
        graph_parameter_n=plan.graph_parameter_n,
        gamma=plan.gamma,
        hash_backend=plan.hash_backend,
        label_width_bits=plan.label_width_bits,
    )
    expected_labels = compute_challenge_labels(
        graph,
        session_seed=plan.session_seed_hex,
        challenge_indices=challenge_indices,
        label_engine="reference",
    )
    _configure_fake_runtime(
        monkeypatch,
        plan,
        challenge_indices=challenge_indices,
        round_responses=[
            {
                "region_id": "host-0",
                "challenge_index": challenge_indices[0],
                "label_bytes": expected_labels[0],
                "round_trip_us": plan.deadline_policy.response_deadline_us + 1,
            }
        ],
    )

    result = VerifierService()._run_session_plan(plan, retain_session=False, extra_notes=[])

    assert result.verdict == "DEADLINE_MISS"
    assert result.accepted_rounds == 0
    assert result.max_round_trip_us > plan.deadline_policy.response_deadline_us


def test_hidden_host_shadow_is_reported_and_rejected(monkeypatch) -> None:
    plan = _session_plan()
    _configure_fake_runtime(
        monkeypatch,
        plan,
        challenge_indices=[0, 3, 7, 3],
        declared_stage_copy_bytes=64,
    )

    result = VerifierService()._run_session_plan(plan, retain_session=False, extra_notes=[])

    assert result.verdict == "PROTOCOL_ERROR"
    assert result.declared_stage_copy_bytes == 64
    assert any("surviving stage copies into the fast phase totaling 64 bytes" in note for note in result.operational_claim_notes)
    assert any("Declared stage copies survive into the fast phase" in note for note in result.notes)


def test_sparse_write_attack_is_detected_as_wrong_response(monkeypatch) -> None:
    plan = _session_plan()
    challenge_indices = [0, 3, 7, 3]
    graph = build_pose_db_graph(
        label_count_m=plan.label_count_m,
        graph_parameter_n=plan.graph_parameter_n,
        gamma=plan.gamma,
        hash_backend=plan.hash_backend,
        label_width_bits=plan.label_width_bits,
    )
    expected_labels = compute_challenge_labels(
        graph,
        session_seed=plan.session_seed_hex,
        challenge_indices=challenge_indices,
        label_engine="reference",
    )
    _configure_fake_runtime(
        monkeypatch,
        plan,
        challenge_indices=challenge_indices,
        round_responses=[
            {
                "region_id": "host-0",
                "challenge_index": challenge_indices[0],
                "label_bytes": expected_labels[0],
                "round_trip_us": 100,
            },
            {
                "region_id": "host-0",
                "challenge_index": challenge_indices[1],
                "label_bytes": bytes(len(expected_labels[1])),
                "round_trip_us": 100,
            },
        ],
    )

    result = VerifierService()._run_session_plan(plan, retain_session=False, extra_notes=[])

    assert result.verdict == "WRONG_RESPONSE"
    assert result.accepted_rounds == 1


def test_copy_in_after_challenge_attack_is_detected_as_wrong_response(monkeypatch) -> None:
    plan = _session_plan()
    challenge_indices = [0, 3, 7, 3]
    graph = build_pose_db_graph(
        label_count_m=plan.label_count_m,
        graph_parameter_n=plan.graph_parameter_n,
        gamma=plan.gamma,
        hash_backend=plan.hash_backend,
        label_width_bits=plan.label_width_bits,
    )
    expected_labels = compute_challenge_labels(
        graph,
        session_seed=plan.session_seed_hex,
        challenge_indices=challenge_indices,
        label_engine="reference",
    )
    _configure_fake_runtime(
        monkeypatch,
        plan,
        challenge_indices=challenge_indices,
        round_responses=[
            {
                "region_id": "host-0",
                "challenge_index": challenge_indices[0],
                "label_bytes": expected_labels[0],
                "round_trip_us": 100,
            },
            {
                "region_id": "host-0",
                "challenge_index": challenge_indices[1],
                "label_bytes": expected_labels[0],
                "round_trip_us": 100,
            },
        ],
    )

    result = VerifierService()._run_session_plan(plan, retain_session=False, extra_notes=[])

    assert result.verdict == "WRONG_RESPONSE"
    assert result.accepted_rounds == 1
