from __future__ import annotations

from pose.benchmarks.profiles import BenchmarkProfile
from pose.verifier.slot_planning import build_session_plan_from_profile, plan_slot_layout


def _profile() -> BenchmarkProfile:
    return BenchmarkProfile.from_dict(
        {
            "name": "hybrid-test",
            "benchmark_class": "cold",
            "target_devices": {"host": True, "gpus": [0]},
            "reserve_policy": {"host_bytes": 272, "per_gpu_bytes": 160},
            "host_target_fraction": 0.5,
            "per_gpu_target_fraction": 0.75,
            "w_bits": 256,
            "graph_family": "pose-db-drg-v1",
            "hash_backend": "blake3-xof",
            "adversary_model": "general",
            "attacker_budget_bytes_assumed": 16,
            "challenge_policy": {
                "rounds_r": 4,
                "target_success_bound": 1.0e-9,
                "sample_with_replacement": True,
            },
            "deadline_policy": {"response_deadline_us": 500000, "session_timeout_ms": 60000},
            "calibration_policy": {
                "lookup_samples": 32,
                "hash_measurement_rounds": 1,
                "hashes_per_round": 64,
                "transport_overhead_us": 100,
                "serialization_overhead_us": 50,
                "safety_margin_fraction": 0.25,
            },
            "cleanup_policy": {"zeroize": True, "verify_zeroization": False},
            "repetition_count": 1,
        }
    )


def test_plan_slot_layout_computes_region_slots_and_slack() -> None:
    layout = plan_slot_layout(
        _profile(),
        detected_host_bytes=272,
        detected_gpu_bytes_by_device={0: (160, 320)},
    )

    assert layout.label_count_m == 7
    assert layout.graph_parameter_n == 2
    assert layout.gamma == 4
    assert layout.covered_bytes == 224
    assert layout.slack_bytes == 32
    assert [region.region_id for region in layout.regions] == ["host-0", "gpu-0"]
    assert layout.regions[0].slot_count == 4
    assert layout.regions[0].slack_bytes == 8
    assert layout.regions[1].slot_count == 3
    assert layout.regions[1].slack_bytes == 24


def test_build_session_plan_from_profile_uses_calibrated_rounds_and_regions() -> None:
    profile = _profile()
    layout = plan_slot_layout(
        profile,
        detected_host_bytes=272,
        detected_gpu_bytes_by_device={0: (160, 320)},
    )
    calibration_payload = {
        "status": "calibrated",
        "artifact_path": "/tmp/hybrid-calibration.json",
        "planning": {
            **layout.to_dict(),
            "graph_family": profile.graph_family,
            "hash_backend": profile.hash_backend,
            "profile_name": profile.name,
            "effective_attacker_budget_bytes_assumed": 80,
            "claim_notes": ["untargeted_local_gpu_tier=device:1,available_bytes:64,total_bytes:128"],
        },
        "q_bound": 3,
        "rounds_r": 4,
        "notes": [],
    }

    session_plan = build_session_plan_from_profile(profile, calibration_payload)

    assert session_plan.profile_name == profile.name
    assert session_plan.label_count_m == 7
    assert session_plan.gamma == 4
    assert session_plan.q_bound == 3
    assert session_plan.challenge_policy.rounds_r == 4
    assert session_plan.attacker_budget_bytes_assumed == 80
    assert [region.region_id for region in session_plan.regions] == ["host-0", "gpu-0"]
    assert session_plan.regions[0].slot_count == 4
    assert session_plan.regions[1].slot_count == 3
    assert any("calibration_artifact=" in note for note in session_plan.claim_notes)
    assert any("untargeted_local_gpu_tier=" in note for note in session_plan.claim_notes)
