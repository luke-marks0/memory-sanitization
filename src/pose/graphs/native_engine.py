from __future__ import annotations

import ctypes
from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module

from pose.common.errors import ProtocolError
from pose.graphs.construction import PoseDbGraph

_native_module = None
_native_import_error: Exception | None = None

try:
    _native_module = import_module("pose_native_label_engine")
except Exception as error:  # pragma: no cover - exercised only when the native wheel is absent
    _native_import_error = error


@dataclass(frozen=True)
class NativeMaterializationMetrics:
    scratch_peak_bytes: int


def native_label_engine_available() -> bool:
    return _native_module is not None


def native_label_engine_unavailable_reason() -> str | None:
    if _native_import_error is None:
        return None
    return f"{type(_native_import_error).__name__}: {_native_import_error}"


def native_cuda_hbm_in_place_available() -> bool:
    if _native_module is None:
        return False
    return bool(getattr(_native_module, "cuda_hbm_in_place_available", lambda: False)())


def _require_native_module():
    if _native_module is None:
        reason = native_label_engine_unavailable_reason() or "native module import failed"
        raise ProtocolError(f"Native label engine is unavailable: {reason}")
    return _native_module


def _session_seed_bytes(session_seed: bytes | str) -> bytes:
    if isinstance(session_seed, bytes):
        return session_seed
    return bytes.fromhex(session_seed)


def compute_native_node_labels_buffer(
    graph: PoseDbGraph,
    *,
    session_seed: bytes | str,
) -> bytes:
    native = _require_native_module()
    return bytes(
        native.compute_node_label_buffer(
            graph.label_count_m,
            graph.graph_parameter_n,
            graph.hash_backend,
            graph.label_width_bits,
            _session_seed_bytes(session_seed),
            graph.graph_descriptor_digest,
        )
    )


def compute_native_challenge_label_array(
    graph: PoseDbGraph,
    *,
    session_seed: bytes | str,
) -> bytes:
    native = _require_native_module()
    return bytes(
        native.compute_challenge_label_array(
            graph.label_count_m,
            graph.graph_parameter_n,
            graph.hash_backend,
            graph.label_width_bits,
            _session_seed_bytes(session_seed),
            graph.graph_descriptor_digest,
        )
    )


def stream_native_materialization(
    graph: PoseDbGraph,
    *,
    session_seed: bytes | str,
    writer: Callable[[bytes], None],
) -> NativeMaterializationMetrics:
    native = _require_native_module()
    scratch_peak_bytes = int(
        native.stream_materialize_challenge_labels(
            graph.label_count_m,
            graph.graph_parameter_n,
            graph.hash_backend,
            graph.label_width_bits,
            _session_seed_bytes(session_seed),
            graph.graph_descriptor_digest,
            writer,
        )
    )
    return NativeMaterializationMetrics(scratch_peak_bytes=scratch_peak_bytes)


def fill_native_host_challenge_labels_in_place(
    graph: PoseDbGraph,
    *,
    session_seed: bytes | str,
    target: bytearray | memoryview,
) -> NativeMaterializationMetrics:
    native = _require_native_module()
    view = memoryview(target)
    if view.readonly:
        raise ProtocolError("In-place native host materialization requires a writable buffer.")
    if not view.contiguous:
        raise ProtocolError("In-place native host materialization requires a contiguous buffer.")
    if view.nbytes != graph.label_count_m * (graph.label_width_bits // 8):
        raise ProtocolError(
            "In-place native host materialization buffer size must equal label_count_m * label_width_bytes."
        )
    scratch_peak_bytes = int(
        native.fill_challenge_label_array_at_address(
            graph.label_count_m,
            graph.graph_parameter_n,
            graph.hash_backend,
            graph.label_width_bits,
            _session_seed_bytes(session_seed),
            graph.graph_descriptor_digest,
            ctypes.addressof(ctypes.c_char.from_buffer(view)),
            view.nbytes,
        )
    )
    return NativeMaterializationMetrics(scratch_peak_bytes=scratch_peak_bytes)


def fill_native_gpu_challenge_labels_in_place(
    graph: PoseDbGraph,
    *,
    session_seed: bytes | str,
    device: int,
    target_pointer: int,
    target_len: int,
) -> NativeMaterializationMetrics:
    native = _require_native_module()
    scratch_peak_bytes = int(
        native.fill_challenge_label_array_on_gpu(
            graph.label_count_m,
            graph.graph_parameter_n,
            graph.hash_backend,
            graph.label_width_bits,
            _session_seed_bytes(session_seed),
            graph.graph_descriptor_digest,
            int(device),
            int(target_pointer),
            int(target_len),
        )
    )
    return NativeMaterializationMetrics(scratch_peak_bytes=scratch_peak_bytes)
