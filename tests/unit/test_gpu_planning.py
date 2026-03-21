from __future__ import annotations

from pose.benchmarks.profiles import load_profile
from pose.verifier.gpu_planning import build_gpu_session_plan, validate_single_gpu_session_plan


def test_gpu_session_plan_uses_measured_unit_size_and_available_bytes() -> None:
    profile = load_profile("single-h100-hbm-max")
    session_plan, gpu_plan = build_gpu_session_plan(
        profile,
        session_id="gpu-session",
        session_nonce="gpu-nonce",
        device=0,
        unit_size_bytes=profile.leaf_size,
        detected_gpu_bytes=(4 * profile.leaf_size, 8 * profile.leaf_size),
    )

    validate_single_gpu_session_plan(session_plan)

    assert gpu_plan.unit_count == 3
    assert gpu_plan.usable_bytes == 3 * profile.leaf_size
    assert session_plan.regions[0].region_type == "gpu"
    assert session_plan.regions[0].gpu_device == 0
    assert len(session_plan.sector_plan) == 3
