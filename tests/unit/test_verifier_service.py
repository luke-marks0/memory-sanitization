from __future__ import annotations

from pathlib import Path

from pose.protocol.result_schema import bootstrap_result
from pose.verifier.service import VerifierService
def test_run_plan_file_passes_loaded_session_plan(monkeypatch, tmp_path: Path) -> None:
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
  unit_count: 1
  regions:
    - region_id: host-0
      region_type: host
      usable_bytes: 4096
  sector_plan:
    - region_id: host-0
      unit_index: 0
      prover_id_hex: "0101"
      sector_id: 4242
      ticket_hex: "0202"
      seed_hex: "0303"
retain_session: true
""".strip(),
        encoding="utf-8",
    )

    captured = {}

    def fake_run_host_session(profile, **kwargs):
        captured["profile"] = profile
        captured["kwargs"] = kwargs
        result = bootstrap_result(profile_name=profile.name)
        result.session_id = kwargs["session_plan"].session_id
        result.session_nonce = kwargs["session_plan"].nonce
        result.verdict = "SUCCESS"
        result.success = True
        return result

    monkeypatch.setattr("pose.verifier.service.run_host_session", fake_run_host_session)

    result = VerifierService().run_plan_file(plan_path)

    assert result.success is True
    assert result.session_id == "planned-session"
    assert captured["kwargs"]["retain_session"] is True
    assert captured["kwargs"]["session_plan"].session_id == "planned-session"
    assert captured["profile"].name == "dev-small"
    assert len(captured["kwargs"]["session_plan"].sector_plan) == 1
    assert captured["kwargs"]["session_plan"].sector_plan[0].sector_id == 4242
