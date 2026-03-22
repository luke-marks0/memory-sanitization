from __future__ import annotations

from dataclasses import replace

import pytest

from pose.common.errors import ProtocolError
from pose.graphs import build_graph_descriptor
from pose.protocol.grpc_codec import session_plan_from_proto, session_plan_to_proto
from pose.protocol.messages import ChallengePolicy, CleanupPolicy, DeadlinePolicy, RegionPlan, SessionPlan


def _session_plan() -> SessionPlan:
    descriptor = build_graph_descriptor(
        label_count_m=4096,
        graph_parameter_n=11,
        gamma=2048,
        hash_backend="blake3-xof",
        label_width_bits=256,
    )
    return SessionPlan(
        session_id="session-id",
        session_seed_hex="ab" * 32,
        profile_name="dev-small",
        graph_family="pose-db-drg-v1",
        graph_parameter_n=11,
        label_count_m=4096,
        gamma=2048,
        label_width_bits=256,
        hash_backend="blake3-xof",
        graph_descriptor_digest=descriptor.digest,
        challenge_policy=ChallengePolicy(rounds_r=64, target_success_bound=1e-9),
        deadline_policy=DeadlinePolicy(response_deadline_us=2500, session_timeout_ms=60000),
        cleanup_policy=CleanupPolicy(zeroize=True, verify_zeroization=False),
        adversary_model="general",
        attacker_budget_bytes_assumed=33554432,
        q_bound=1024,
        regions=[
            RegionPlan(
                region_id="host-0",
                region_type="host",
                usable_bytes=131072,
                slot_count=4096,
                covered_bytes=131072,
                slack_bytes=0,
            )
        ],
        claim_notes=["host-only development profile"],
    )


def test_session_plan_root_changes_when_graph_parameter_changes() -> None:
    baseline = _session_plan()
    updated_descriptor = build_graph_descriptor(
        label_count_m=4097,
        graph_parameter_n=12,
        gamma=4096,
        hash_backend="blake3-xof",
        label_width_bits=256,
    )
    updated = replace(
        baseline,
        label_count_m=4097,
        graph_parameter_n=12,
        gamma=4096,
        graph_descriptor_digest=updated_descriptor.digest,
        regions=[
            replace(
                baseline.regions[0],
                usable_bytes=131_104,
                slot_count=4097,
                covered_bytes=131_104,
            )
        ],
    )

    assert baseline.plan_root_hex != updated.plan_root_hex


def test_session_plan_proto_round_trip_preserves_pose_db_fields() -> None:
    baseline = _session_plan()

    round_tripped = session_plan_from_proto(session_plan_to_proto(baseline))

    assert round_tripped.session_id == baseline.session_id
    assert round_tripped.session_seed_hex == baseline.session_seed_hex
    assert round_tripped.graph_family == "pose-db-drg-v1"
    assert round_tripped.challenge_policy.rounds_r == 64
    assert round_tripped.deadline_policy.response_deadline_us == 2500
    assert round_tripped.regions[0].slot_count == 4096


def test_session_plan_from_dict_rejects_unsupported_hash_backend() -> None:
    payload = _session_plan().to_cbor_object()
    payload["hash_backend"] = "sha256"

    with pytest.raises(ProtocolError, match="Unsupported hash backend"):
        SessionPlan.from_dict(payload)
