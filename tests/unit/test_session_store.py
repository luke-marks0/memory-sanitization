from __future__ import annotations

from pathlib import Path

from pose.verifier.session_store import load_plan_file


def test_load_plan_file_supports_session_plan_and_retain_session(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text(
        """
session_plan:
  session_id: planned-session
  nonce: planned-nonce
  profile_name: dev-small
  porep_unit_profile: minimal
  challenge_leaf_size: 4096
  challenge_policy:
    epsilon: 0.01
    lambda_bits: 32
    max_challenges: 64
  deadline_policy:
    response_deadline_ms: 5000
    session_timeout_ms: 60000
  cleanup_policy:
    zeroize: true
    verify_zeroization: false
  unit_count: 2
  regions:
    - region_id: host-0
      region_type: host
      usable_bytes: 8192
  sector_plan:
    - region_id: host-0
      unit_index: 0
      prover_id_hex: "0101"
      sector_id: 4242
      ticket_hex: "0202"
      seed_hex: "0303"
    - region_id: host-0
      unit_index: 1
      prover_id_hex: "0404"
      sector_id: 4243
      ticket_hex: "0505"
      seed_hex: "0606"
retain_session: true
""".strip(),
        encoding="utf-8",
    )

    loaded = load_plan_file(plan_path)

    assert loaded.retain_session is True
    assert loaded.session_plan.session_id == "planned-session"
    assert loaded.session_plan.unit_count == 2
    assert loaded.session_plan.regions[0].usable_bytes == 8192
    assert len(loaded.session_plan.sector_plan) == 2
