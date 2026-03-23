#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path


DOMAIN_NAMESPACE_LEN = 7
ENCODING_VERSION_LEN = 4
SOURCE_LABEL_DOMAIN_LEN = 23
INTERNAL_LABEL_DOMAIN_LEN = 25
CURRENT_SESSION_SEED_BYTES = 32
MAX_SESSION_SEED_BYTES = 64
MAX_DIGEST_BYTES = 128


class PoseOracleConfig(ctypes.Structure):
    _fields_ = [
        ("output_bytes", ctypes.c_uint32),
        ("session_seed_len", ctypes.c_uint32),
        ("digest_len", ctypes.c_uint32),
        ("session_seed", ctypes.c_uint8 * MAX_SESSION_SEED_BYTES),
        ("graph_digest", ctypes.c_uint8 * MAX_DIGEST_BYTES),
    ]


def _domain_prefix_len(domain_len: int) -> int:
    return DOMAIN_NAMESPACE_LEN + ENCODING_VERSION_LEN + 4 + domain_len + 4


def _source_payload_bytes(session_seed_len: int, digest_len: int) -> int:
    return _domain_prefix_len(SOURCE_LABEL_DOMAIN_LEN) + 4 + session_seed_len + 4 + digest_len + 4 + 8


def _internal_payload_1_bytes(session_seed_len: int, digest_len: int, output_bytes: int) -> int:
    return _domain_prefix_len(INTERNAL_LABEL_DOMAIN_LEN) + 4 + session_seed_len + 4 + digest_len + 4 + 8 + 8 + 4 + output_bytes


def _internal_payload_2_bytes(session_seed_len: int, digest_len: int, output_bytes: int) -> int:
    return _internal_payload_1_bytes(session_seed_len, digest_len, output_bytes) + 4 + output_bytes


def _merged_plan_bytes_for_dimension(dimension: int) -> int:
    width = 1 << dimension
    return 2 * (dimension + 1) * width * 4


def _top_dimension_contributors(dimensions: list[int], count: int = 3) -> list[dict[str, int]]:
    entries = [
        {
            "dimension": dimension,
            "bytes": _merged_plan_bytes_for_dimension(dimension),
        }
        for dimension in dimensions
    ]
    entries.sort(key=lambda entry: entry["bytes"], reverse=True)
    return entries[:count]


def _hbm_plan_dimensions_used(graph_parameter_n: int, label_count_m: int) -> list[int]:
    used: set[int] = set()

    def connected_full(level: int) -> None:
        if level == 0:
            return
        connected_full(level - 1)
        used.add(level - 1)
        connected_full(level - 1)

    def connected_prefix(level: int, retain: int) -> None:
        if retain == 0 or level == 0:
            return
        half_slots = 1 << (level - 1)
        if retain <= half_slots:
            connected_prefix(level - 1, retain)
            return
        connected_full(level - 1)
        used.add(level - 1)
        connected_prefix(level - 1, retain - half_slots)

    def standalone_base(level: int) -> None:
        if level == 0:
            return
        standalone_base(level - 1)
        connected_full(level - 1)

    def standalone_right_prefix(level: int, retain: int) -> None:
        if retain == 0:
            return
        standalone_base(level - 1)
        connected_prefix(level - 1, retain)

    level = graph_parameter_n + 1
    left_width = 1 << graph_parameter_n
    retained_from_right = label_count_m - left_width
    if retained_from_right > 0:
        standalone_right_prefix(level, retained_from_right)
    standalone_right_prefix(level, left_width)
    return sorted(used)


def _load_result(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_host_tabled_breakdown(result: dict[str, object], *, session_seed_len: int) -> dict[str, object]:
    graph_parameter_n = int(result["graph_parameter_n"])
    output_bytes = int(result["label_width_bits"]) // 8
    digest_len = len(str(result["graph_descriptor_digest"]).encode("utf-8"))
    dimensions = list(range(graph_parameter_n + 1))

    source_payload_bytes = _source_payload_bytes(session_seed_len, digest_len)
    internal1_payload_bytes = _internal_payload_1_bytes(session_seed_len, digest_len, output_bytes)
    internal2_payload_bytes = _internal_payload_2_bytes(session_seed_len, digest_len, output_bytes)
    label_oracle_payload_bytes = source_payload_bytes + internal1_payload_bytes + internal2_payload_bytes
    temp_buffer_bytes = output_bytes * 2
    merged_plan_bytes = sum(_merged_plan_bytes_for_dimension(dimension) for dimension in dimensions)
    expected_scratch = merged_plan_bytes + label_oracle_payload_bytes + temp_buffer_bytes

    return {
        "mode": "host-in-place",
        "artifact_path": str(result["artifact_path"]),
        "covered_bytes": int(result["covered_bytes"]),
        "scratch_peak_bytes": int(result["scratch_peak_bytes"]),
        "declared_stage_copy_bytes": int(result["declared_stage_copy_bytes"]),
        "scratch_over_covered_bytes": int(result["scratch_peak_bytes"]) - int(result["covered_bytes"]),
        "scratch_to_covered_ratio": int(result["scratch_peak_bytes"]) / int(result["covered_bytes"]),
        "plan_dimensions_used": dimensions,
        "components": {
            "merged_plan_index_tables": merged_plan_bytes,
            "label_oracle_payloads": label_oracle_payload_bytes,
            "temp_label_buffers": temp_buffer_bytes,
        },
        "component_details": {
            "merged_plan_top_dimensions": _top_dimension_contributors(dimensions),
            "label_oracle_payloads": {
                "source_payload": source_payload_bytes,
                "internal1_payload": internal1_payload_bytes,
                "internal2_payload": internal2_payload_bytes,
            },
            "temp_label_buffers": {
                "temp0": output_bytes,
                "temp1": output_bytes,
            },
        },
        "scratch_matches_reported": expected_scratch == int(result["scratch_peak_bytes"]),
    }


def _build_host_arithmetic_breakdown(result: dict[str, object], *, session_seed_len: int) -> dict[str, object]:
    output_bytes = int(result["label_width_bits"]) // 8
    digest_len = len(str(result["graph_descriptor_digest"]).encode("utf-8"))

    source_payload_bytes = _source_payload_bytes(session_seed_len, digest_len)
    internal1_payload_bytes = _internal_payload_1_bytes(session_seed_len, digest_len, output_bytes)
    internal2_payload_bytes = _internal_payload_2_bytes(session_seed_len, digest_len, output_bytes)
    label_oracle_payload_bytes = source_payload_bytes + internal1_payload_bytes + internal2_payload_bytes
    temp_buffer_bytes = output_bytes * 2
    expected_scratch = label_oracle_payload_bytes + temp_buffer_bytes

    return {
        "mode": "host-in-place-arithmetic",
        "artifact_path": str(result["artifact_path"]),
        "covered_bytes": int(result["covered_bytes"]),
        "scratch_peak_bytes": int(result["scratch_peak_bytes"]),
        "declared_stage_copy_bytes": int(result["declared_stage_copy_bytes"]),
        "scratch_over_covered_bytes": int(result["scratch_peak_bytes"]) - int(result["covered_bytes"]),
        "scratch_to_covered_ratio": int(result["scratch_peak_bytes"]) / int(result["covered_bytes"]),
        "components": {
            "label_oracle_payloads": label_oracle_payload_bytes,
            "temp_label_buffers": temp_buffer_bytes,
        },
        "component_details": {
            "label_oracle_payloads": {
                "source_payload": source_payload_bytes,
                "internal1_payload": internal1_payload_bytes,
                "internal2_payload": internal2_payload_bytes,
            },
            "temp_label_buffers": {
                "temp0": output_bytes,
                "temp1": output_bytes,
            },
        },
        "scratch_matches_reported": expected_scratch == int(result["scratch_peak_bytes"]),
    }


def _build_cuda_hbm_breakdown(result: dict[str, object]) -> dict[str, object]:
    graph_parameter_n = int(result["graph_parameter_n"])
    label_count_m = int(result["label_count_m"])
    dimensions = _hbm_plan_dimensions_used(graph_parameter_n, label_count_m)

    host_plan_bytes = sum(_merged_plan_bytes_for_dimension(dimension) for dimension in dimensions)
    device_plan_bytes = host_plan_bytes
    pose_oracle_config_bytes = ctypes.sizeof(PoseOracleConfig)
    expected_scratch = host_plan_bytes + device_plan_bytes + pose_oracle_config_bytes

    return {
        "mode": "cuda-hbm-in-place",
        "artifact_path": str(result["artifact_path"]),
        "covered_bytes": int(result["covered_bytes"]),
        "scratch_peak_bytes": int(result["scratch_peak_bytes"]),
        "declared_stage_copy_bytes": int(result["declared_stage_copy_bytes"]),
        "scratch_over_covered_bytes": int(result["scratch_peak_bytes"]) - int(result["covered_bytes"]),
        "scratch_to_covered_ratio": int(result["scratch_peak_bytes"]) / int(result["covered_bytes"]),
        "plan_dimensions_used": dimensions,
        "components": {
            "host_merged_plan_cache": host_plan_bytes,
            "device_merged_plan_cache": device_plan_bytes,
            "pose_oracle_config": pose_oracle_config_bytes,
        },
        "component_details": {
            "host_merged_plan_top_dimensions": _top_dimension_contributors(dimensions),
            "device_merged_plan_top_dimensions": _top_dimension_contributors(dimensions),
        },
        "scratch_matches_reported": expected_scratch == int(result["scratch_peak_bytes"]),
    }


def _build_cuda_hbm_arithmetic_breakdown(result: dict[str, object]) -> dict[str, object]:
    pose_oracle_config_bytes = ctypes.sizeof(PoseOracleConfig)
    expected_scratch = pose_oracle_config_bytes

    return {
        "mode": "cuda-hbm-in-place-arithmetic",
        "artifact_path": str(result["artifact_path"]),
        "covered_bytes": int(result["covered_bytes"]),
        "scratch_peak_bytes": int(result["scratch_peak_bytes"]),
        "declared_stage_copy_bytes": int(result["declared_stage_copy_bytes"]),
        "scratch_over_covered_bytes": int(result["scratch_peak_bytes"]) - int(result["covered_bytes"]),
        "scratch_to_covered_ratio": int(result["scratch_peak_bytes"]) / int(result["covered_bytes"]),
        "components": {
            "pose_oracle_config": pose_oracle_config_bytes,
        },
        "component_details": {},
        "scratch_matches_reported": expected_scratch == int(result["scratch_peak_bytes"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile the current native scratch accounting breakdown from a benchmark result artifact.")
    parser.add_argument(
        "--mode",
        required=True,
        choices=(
            "host-in-place",
            "host-in-place-arithmetic",
            "cuda-hbm-in-place",
            "cuda-hbm-in-place-arithmetic",
        ),
        help="Select the native path whose scratch accounting should be decomposed.",
    )
    parser.add_argument(
        "--result-artifact",
        required=True,
        type=Path,
        help="Path to a benchmark result JSON artifact.",
    )
    parser.add_argument(
        "--session-seed-bytes",
        type=int,
        default=CURRENT_SESSION_SEED_BYTES,
        help="Current verifier session-seed length in bytes. Defaults to the repository runtime value.",
    )
    args = parser.parse_args()

    result = _load_result(args.result_artifact)
    if args.mode == "host-in-place":
        payload = _build_host_tabled_breakdown(result, session_seed_len=args.session_seed_bytes)
    elif args.mode == "host-in-place-arithmetic":
        payload = _build_host_arithmetic_breakdown(result, session_seed_len=args.session_seed_bytes)
    elif args.mode == "cuda-hbm-in-place-arithmetic":
        payload = _build_cuda_hbm_arithmetic_breakdown(result)
    else:
        payload = _build_cuda_hbm_breakdown(result)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
