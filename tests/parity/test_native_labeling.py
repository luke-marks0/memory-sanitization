from __future__ import annotations

import pytest

from pose.common.gpu_lease import create_gpu_lease, get_cuda_runtime, release_gpu_lease
from pose.protocol.messages import CleanupPolicy
from pose.graphs import (
    build_pose_db_graph,
    compute_challenge_labels,
    compute_label_array,
    compute_node_labels,
    native_label_engine_available,
    preferred_runtime_label_engine,
)
from pose.graphs.native_engine import (
    fill_native_gpu_challenge_labels_in_place,
    fill_native_host_challenge_labels_in_place,
    native_cuda_hbm_in_place_available,
    profile_native_gpu_challenge_labels_in_place,
)


pytestmark = pytest.mark.skipif(
    not native_label_engine_available(),
    reason="native label engine extension is not installed",
)


@pytest.mark.parametrize("hash_backend", ["blake3-xof", "shake256"])
def test_native_label_engine_matches_reference_graph_outputs(hash_backend: str) -> None:
    graph = build_pose_db_graph(
        label_count_m=17,
        hash_backend=hash_backend,
        label_width_bits=256,
    )
    session_seed = "77" * 32
    challenge_indices = [0, 3, 7, 16, 3]

    assert compute_node_labels(
        graph,
        session_seed=session_seed,
        label_engine="native",
    ) == compute_node_labels(
        graph,
        session_seed=session_seed,
        label_engine="reference",
    )
    assert compute_challenge_labels(
        graph,
        session_seed=session_seed,
        challenge_indices=challenge_indices,
        label_engine="native",
    ) == compute_challenge_labels(
        graph,
        session_seed=session_seed,
        challenge_indices=challenge_indices,
        label_engine="reference",
    )
    assert compute_label_array(
        graph,
        session_seed=session_seed,
        label_engine="native",
    ) == compute_label_array(
        graph,
        session_seed=session_seed,
        label_engine="reference",
    )


def test_runtime_prefers_native_when_extension_is_available() -> None:
    assert preferred_runtime_label_engine() == "native"


@pytest.mark.parametrize("hash_backend", ["blake3-xof", "shake256"])
def test_native_in_place_host_fill_matches_reference_label_array(hash_backend: str) -> None:
    graph = build_pose_db_graph(
        label_count_m=17,
        hash_backend=hash_backend,
        label_width_bits=256,
    )
    session_seed = "55" * 32
    expected = compute_label_array(
        graph,
        session_seed=session_seed,
        label_engine="reference",
    )

    target = bytearray(len(expected))
    metrics = fill_native_host_challenge_labels_in_place(
        graph,
        session_seed=session_seed,
        target=target,
    )
    actual = bytes(target)

    assert actual == expected
    assert metrics.scratch_peak_bytes > 0


@pytest.mark.skipif(
    not native_cuda_hbm_in_place_available(),
    reason="CUDA HBM in-place native engine is not available",
)
def test_native_in_place_gpu_profile_reports_arithmetic_schedule_scratch() -> None:
    try:
        runtime = get_cuda_runtime()
    except Exception as error:  # pragma: no cover - environment-specific skip
        pytest.skip(f"CUDA runtime unavailable: {error}")
    if runtime.device_count() <= 0:
        pytest.skip("No CUDA device is available for HBM in-place parity.")

    graph = build_pose_db_graph(
        label_count_m=17,
        hash_backend="blake3-xof",
        label_width_bits=256,
    )
    target_len = graph.label_count_m * (graph.label_width_bits // 8)
    lease = create_gpu_lease(
        session_id="native-gpu-in-place-profile-test",
        region_id="gpu-0",
        device=0,
        usable_bytes=target_len,
        cleanup_policy=CleanupPolicy(zeroize=True, verify_zeroization=False),
        lease_duration_ms=60_000,
        runtime=runtime,
    )
    try:
        metrics = profile_native_gpu_challenge_labels_in_place(
            graph,
            session_seed="99" * 32,
            device=lease.device,
            target_pointer=lease.pointer,
            target_len=target_len,
        )
    finally:
        release_gpu_lease(
            lease,
            zeroize=True,
            verify_zeroization=False,
        )

    assert metrics.scratch_peak_bytes < 1024
    assert metrics.profiling_counters["host_merged_plan_builds"] == 0
    assert metrics.profiling_counters["device_merged_plan_builds"] == 0


@pytest.mark.skipif(
    not native_cuda_hbm_in_place_available(),
    reason="CUDA HBM in-place native engine is not available",
)
def test_native_in_place_gpu_fill_matches_reference_label_array() -> None:
    try:
        runtime = get_cuda_runtime()
    except Exception as error:  # pragma: no cover - environment-specific skip
        pytest.skip(f"CUDA runtime unavailable: {error}")
    if runtime.device_count() <= 0:
        pytest.skip("No CUDA device is available for HBM in-place parity.")

    graph = build_pose_db_graph(
        label_count_m=17,
        hash_backend="blake3-xof",
        label_width_bits=256,
    )
    session_seed = "66" * 32
    expected = compute_label_array(
        graph,
        session_seed=session_seed,
        label_engine="reference",
    )
    lease = create_gpu_lease(
        session_id="native-gpu-in-place-test",
        region_id="gpu-0",
        device=0,
        usable_bytes=len(expected),
        cleanup_policy=CleanupPolicy(zeroize=True, verify_zeroization=False),
        lease_duration_ms=60_000,
        runtime=runtime,
    )
    try:
        metrics = fill_native_gpu_challenge_labels_in_place(
            graph,
            session_seed=session_seed,
            device=lease.device,
            target_pointer=lease.pointer,
            target_len=len(expected),
        )
        actual = lease.read(length=len(expected))
    finally:
        release_gpu_lease(
            lease,
            zeroize=True,
            verify_zeroization=False,
        )

    assert actual == expected
    assert metrics.scratch_peak_bytes > 0
