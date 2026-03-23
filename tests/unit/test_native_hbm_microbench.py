from __future__ import annotations

from types import SimpleNamespace

from pose.benchmarks import native_hbm_microbench
from pose.graphs import build_pose_db_graph
from pose.graphs.native_engine import (
    NativeMaterializationMetrics,
    profile_native_gpu_challenge_labels_in_place,
)


def test_profile_native_gpu_challenge_labels_in_place_parses_profile_payload(monkeypatch) -> None:
    class FakeNativeModule:
        def profile_challenge_label_array_on_gpu(self, *args):
            return {
                "scratch_peak_bytes": 1234,
                "profiling_counters": {
                    "total_kernel_launches": 55,
                    "launch_internal1_copy": 8,
                },
            }

    monkeypatch.setattr("pose.graphs.native_engine._native_module", FakeNativeModule())

    graph = build_pose_db_graph(
        label_count_m=8,
        graph_parameter_n=2,
        hash_backend="blake3-xof",
        label_width_bits=256,
    )

    metrics = profile_native_gpu_challenge_labels_in_place(
        graph,
        session_seed="11" * 32,
        device=0,
        target_pointer=123,
        target_len=8 * 32,
    )

    assert metrics.scratch_peak_bytes == 1234
    assert metrics.profiling_counters == {
        "total_kernel_launches": 55,
        "launch_internal1_copy": 8,
    }


def test_run_native_hbm_microbenchmark_summarizes_variants(monkeypatch) -> None:
    fake_graph = SimpleNamespace(
        label_count_m=4,
        graph_parameter_n=2,
        node_count=30,
        label_width_bits=256,
        hash_backend="blake3-xof",
        graph_descriptor_digest="sha256:test",
    )

    class FakeRuntime:
        def __init__(self) -> None:
            self._next_pointer = 1000

        def malloc(self, device: int, size: int) -> int:
            pointer = self._next_pointer
            self._next_pointer += size
            return pointer

        def free(self, device: int, pointer: int) -> None:
            return None

        def copy_host_to_device(self, device: int, pointer: int, payload: bytes, *, offset: int = 0) -> None:
            return None

    perf_values = iter([0.0, 1.0, 2.0, 4.5, 5.0, 8.0, 9.0, 13.0])

    monkeypatch.setattr(native_hbm_microbench, "build_pose_db_graph", lambda **kwargs: fake_graph)
    monkeypatch.setattr(native_hbm_microbench, "get_cuda_runtime", lambda: FakeRuntime())
    monkeypatch.setattr(
        native_hbm_microbench,
        "fill_native_host_challenge_labels_in_place",
        lambda *args, **kwargs: NativeMaterializationMetrics(scratch_peak_bytes=11),
    )
    monkeypatch.setattr(
        native_hbm_microbench,
        "profile_native_gpu_challenge_labels_in_place",
        lambda *args, **kwargs: NativeMaterializationMetrics(
            scratch_peak_bytes=22,
            profiling_counters={"total_kernel_launches": 33},
        ),
    )

    def fake_stream_native_materialization(graph, *, session_seed, writer):
        label_bytes = b"x" * (graph.label_width_bits // 8)
        for _ in range(graph.label_count_m):
            writer(label_bytes)
        return NativeMaterializationMetrics(scratch_peak_bytes=44)

    monkeypatch.setattr(
        native_hbm_microbench,
        "stream_native_materialization",
        fake_stream_native_materialization,
    )
    monkeypatch.setattr(native_hbm_microbench, "perf_counter", lambda: next(perf_values))

    payload = native_hbm_microbench.run_native_hbm_microbenchmark(
        label_count_m=4,
        graph_parameter_n=2,
        repetitions=1,
    )

    assert payload["graph"]["label_count_m"] == 4
    assert payload["variants"]["host_in_place"]["wall_ms"]["mean"] == 1000.0
    assert payload["variants"]["gpu_in_place"]["wall_ms"]["mean"] == 2500.0
    assert payload["variants"]["stream_noop_writer"]["wall_ms"]["mean"] == 3000.0
    assert payload["variants"]["stream_gpu_writer"]["wall_ms"]["mean"] == 4000.0
    assert payload["variants"]["gpu_in_place"]["profiling_counters"]["total_kernel_launches"]["mean"] == 33.0
    assert payload["variants"]["stream_gpu_writer"]["labels_written"]["mean"] == 4.0
    assert payload["variants"]["stream_gpu_writer"]["bytes_written"]["mean"] == 128.0
    assert payload["comparisons"]["gpu_in_place_minus_stream_gpu_writer_ms"] == -1500.0
