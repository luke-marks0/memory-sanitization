from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from pose.common.gpu_lease import get_cuda_runtime
from pose.graphs import build_pose_db_graph
from pose.graphs.native_engine import (
    fill_native_host_challenge_labels_in_place,
    profile_native_gpu_challenge_labels_in_place,
    stream_native_materialization,
)
from pose.protocol.codec import dump_json_file


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] + ((ordered[upper] - ordered[lower]) * weight)


def _series_summary(values: Sequence[int | float]) -> dict[str, float]:
    if not values:
        return {
            "mean": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "p99": 0.0,
        }
    numeric = [float(value) for value in values]
    return {
        "mean": sum(numeric) / len(numeric),
        "p50": _percentile(numeric, 0.50),
        "p95": _percentile(numeric, 0.95),
        "p99": _percentile(numeric, 0.99),
    }


@dataclass(frozen=True)
class _VariantRun:
    wall_ms: float
    scratch_peak_bytes: int
    labels_written: int
    bytes_written: int
    profiling_counters: dict[str, int]


def _summarize_variant_runs(runs: Sequence[_VariantRun]) -> dict[str, object]:
    counter_keys = {
        key
        for run in runs
        for key in run.profiling_counters
    }
    return {
        "run_count": len(runs),
        "wall_ms": _series_summary([run.wall_ms for run in runs]),
        "scratch_peak_bytes": _series_summary([run.scratch_peak_bytes for run in runs]),
        "labels_written": _series_summary([run.labels_written for run in runs]),
        "bytes_written": _series_summary([run.bytes_written for run in runs]),
        "profiling_counters": {
            key: _series_summary([run.profiling_counters.get(key, 0) for run in runs])
            for key in sorted(counter_keys)
        },
    }


def run_native_hbm_microbenchmark(
    *,
    label_count_m: int,
    graph_parameter_n: int,
    hash_backend: str = "blake3-xof",
    label_width_bits: int = 256,
    device: int = 0,
    repetitions: int = 1,
    session_seed_hex: str = "11" * 32,
) -> dict[str, object]:
    graph = build_pose_db_graph(
        label_count_m=label_count_m,
        graph_parameter_n=graph_parameter_n,
        hash_backend=hash_backend,
        label_width_bits=label_width_bits,
    )
    runtime = get_cuda_runtime()
    label_width_bytes = label_width_bits // 8
    target_len = label_count_m * label_width_bytes

    host_runs: list[_VariantRun] = []
    gpu_runs: list[_VariantRun] = []
    stream_noop_runs: list[_VariantRun] = []
    stream_gpu_runs: list[_VariantRun] = []

    for _ in range(max(1, int(repetitions))):
        host_target = bytearray(target_len)
        started = perf_counter()
        host_metrics = fill_native_host_challenge_labels_in_place(
            graph,
            session_seed=session_seed_hex,
            target=host_target,
        )
        host_runs.append(
            _VariantRun(
                wall_ms=(perf_counter() - started) * 1000.0,
                scratch_peak_bytes=host_metrics.scratch_peak_bytes,
                labels_written=label_count_m,
                bytes_written=target_len,
                profiling_counters=host_metrics.profiling_counters,
            )
        )

        pointer = runtime.malloc(device, target_len)
        try:
            started = perf_counter()
            gpu_metrics = profile_native_gpu_challenge_labels_in_place(
                graph,
                session_seed=session_seed_hex,
                device=device,
                target_pointer=pointer,
                target_len=target_len,
            )
            gpu_runs.append(
                _VariantRun(
                    wall_ms=(perf_counter() - started) * 1000.0,
                    scratch_peak_bytes=gpu_metrics.scratch_peak_bytes,
                    labels_written=label_count_m,
                    bytes_written=target_len,
                    profiling_counters=gpu_metrics.profiling_counters,
                )
            )
        finally:
            runtime.free(device, pointer)

        noop_state = {"count": 0}

        def noop_writer(label_bytes: bytes) -> None:
            noop_state["count"] += 1

        started = perf_counter()
        stream_noop_metrics = stream_native_materialization(
            graph,
            session_seed=session_seed_hex,
            writer=noop_writer,
        )
        stream_noop_runs.append(
            _VariantRun(
                wall_ms=(perf_counter() - started) * 1000.0,
                scratch_peak_bytes=stream_noop_metrics.scratch_peak_bytes,
                labels_written=noop_state["count"],
                bytes_written=noop_state["count"] * label_width_bytes,
                profiling_counters=stream_noop_metrics.profiling_counters,
            )
        )

        pointer = runtime.malloc(device, target_len)
        write_state = {"count": 0, "offset": 0}

        def gpu_writer(label_bytes: bytes) -> None:
            runtime.copy_host_to_device(device, pointer, label_bytes, offset=write_state["offset"])
            write_state["offset"] += len(label_bytes)
            write_state["count"] += 1

        try:
            started = perf_counter()
            stream_gpu_metrics = stream_native_materialization(
                graph,
                session_seed=session_seed_hex,
                writer=gpu_writer,
            )
            stream_gpu_runs.append(
                _VariantRun(
                    wall_ms=(perf_counter() - started) * 1000.0,
                    scratch_peak_bytes=stream_gpu_metrics.scratch_peak_bytes,
                    labels_written=write_state["count"],
                    bytes_written=write_state["offset"],
                    profiling_counters=stream_gpu_metrics.profiling_counters,
                )
            )
        finally:
            runtime.free(device, pointer)

    host_summary = _summarize_variant_runs(host_runs)
    gpu_summary = _summarize_variant_runs(gpu_runs)
    stream_noop_summary = _summarize_variant_runs(stream_noop_runs)
    stream_gpu_summary = _summarize_variant_runs(stream_gpu_runs)

    return {
        "benchmark": "native-hbm-microbench",
        "graph": {
            "label_count_m": graph.label_count_m,
            "graph_parameter_n": graph.graph_parameter_n,
            "node_count": graph.node_count,
            "label_width_bits": graph.label_width_bits,
            "hash_backend": graph.hash_backend,
            "graph_descriptor_digest": graph.graph_descriptor_digest,
        },
        "target": {
            "device": device,
            "target_bytes": target_len,
            "repetitions": max(1, int(repetitions)),
        },
        "variants": {
            "host_in_place": host_summary,
            "gpu_in_place": gpu_summary,
            "stream_noop_writer": stream_noop_summary,
            "stream_gpu_writer": stream_gpu_summary,
        },
        "comparisons": {
            "gpu_in_place_minus_host_in_place_ms": (
                gpu_summary["wall_ms"]["mean"] - host_summary["wall_ms"]["mean"]
            ),
            "gpu_in_place_minus_stream_gpu_writer_ms": (
                gpu_summary["wall_ms"]["mean"] - stream_gpu_summary["wall_ms"]["mean"]
            ),
            "stream_gpu_writer_minus_stream_noop_writer_ms": (
                stream_gpu_summary["wall_ms"]["mean"] - stream_noop_summary["wall_ms"]["mean"]
            ),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run direct native host/HBM materialization microbenchmarks.")
    parser.add_argument("--label-count-m", type=int, required=True)
    parser.add_argument("--graph-parameter-n", type=int, required=True)
    parser.add_argument("--hash-backend", default="blake3-xof")
    parser.add_argument("--label-width-bits", type=int, default=256)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--session-seed-hex", default="11" * 32)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args(argv)

    payload = run_native_hbm_microbenchmark(
        label_count_m=args.label_count_m,
        graph_parameter_n=args.graph_parameter_n,
        hash_backend=args.hash_backend,
        label_width_bits=args.label_width_bits,
        device=args.device,
        repetitions=args.repetitions,
        session_seed_hex=args.session_seed_hex,
    )
    if args.output_json is not None:
        dump_json_file(args.output_json, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
