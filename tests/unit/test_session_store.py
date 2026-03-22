from __future__ import annotations

from pathlib import Path

from pose.graphs import build_graph_descriptor
from pose.verifier.session_store import ResidentSessionRecord, load_plan_file


def test_load_plan_file_supports_session_plan_and_retain_session(tmp_path: Path) -> None:
    descriptor = build_graph_descriptor(
        label_count_m=256,
        graph_parameter_n=7,
        gamma=128,
        hash_backend="blake3-xof",
        label_width_bits=256,
    )
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text(
        f"""
session_plan:
  session_id: planned-session
  session_seed_hex: "abababababababababababababababababababababababababababababababab"
  profile_name: dev-small
  graph_family: pose-db-drg-v1
  graph_parameter_n: 7
  label_count_m: 256
  gamma: 128
  label_width_bits: 256
  hash_backend: blake3-xof
  graph_descriptor_digest: {descriptor.digest}
  challenge_policy:
    rounds_r: 64
    target_success_bound: 1e-9
    sample_with_replacement: true
  deadline_policy:
    response_deadline_us: 2500
    session_timeout_ms: 60000
  cleanup_policy:
    zeroize: true
    verify_zeroization: false
  adversary_model: general
  attacker_budget_bytes_assumed: 33554432
  q_bound: 1024
  regions:
    - region_id: host-0
      region_type: host
      usable_bytes: 8192
      slot_count: 256
      covered_bytes: 8192
      slack_bytes: 0
retain_session: true
""".strip(),
        encoding="utf-8",
    )

    loaded = load_plan_file(plan_path)

    assert loaded.retain_session is True
    assert loaded.session_plan.session_id == "planned-session"
    assert loaded.session_plan.label_count_m == 256
    assert loaded.session_plan.regions[0].usable_bytes == 8192
    assert loaded.session_plan.challenge_policy.rounds_r == 64


def test_resident_session_record_round_trips_pose_db_fields() -> None:
    record = ResidentSessionRecord(
        session_id="resident-session",
        profile_name="dev-small",
        session_seed_hex="aa" * 32,
        session_plan_root="plan-root",
        graph_family="pose-db-drg-v1",
        graph_parameter_n=11,
        graph_descriptor_digest="sha256:abcd",
        label_width_bits=256,
        label_count_m=4096,
        gamma=2048,
        hash_backend="blake3-xof",
        region_id="host-0",
        region_slot_count=4096,
        challenge_policy={"rounds_r": 64, "sample_with_replacement": True, "target_success_bound": 1e-9},
        deadline_us=2500,
        cleanup_policy={"zeroize": True, "verify_zeroization": False},
        adversary_model="general",
        attacker_budget_bytes_assumed=33554432,
        q_bound=1024,
        host_total_bytes=8192,
        host_usable_bytes=8192,
        host_covered_bytes=8192,
        covered_bytes=8192,
        slack_bytes=0,
        coverage_fraction=1.0,
        scratch_peak_bytes=512,
        declared_stage_copy_bytes=0,
        formal_claim_notes=["formal"],
        operational_claim_notes=["operational"],
        claim_notes=["host-only development profile"],
        socket_path="/tmp/pose.sock",
        process_id=1234,
        lease_expiry="2026-03-22T00:00:00+00:00",
    )

    decoded = ResidentSessionRecord.from_dict(record.to_dict())

    assert decoded.graph_family == "pose-db-drg-v1"
    assert decoded.graph_parameter_n == 11
    assert decoded.region_slot_count == 4096
    assert decoded.deadline_us == 2500
    assert decoded.scratch_peak_bytes == 512
    assert decoded.declared_stage_copy_bytes == 0
    assert decoded.formal_claim_notes == ["formal"]
    assert decoded.operational_claim_notes == ["operational"]
    assert decoded.claim_notes == ["host-only development profile"]
