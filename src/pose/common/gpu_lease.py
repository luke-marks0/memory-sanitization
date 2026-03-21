from __future__ import annotations

import base64
import ctypes
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from pose.common.errors import ResourceFailure
from pose.protocol.messages import CleanupPolicy, LeaseRecord

ZERO_CHUNK_BYTES = 1024 * 1024
CUDA_MEMCPY_HOST_TO_DEVICE = 1
CUDA_MEMCPY_DEVICE_TO_HOST = 2
CUDA_IPC_MEM_LAZY_ENABLE_PEER_ACCESS = 1


class _CudaIpcMemHandle(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_ubyte * 64)]


class CudaRuntime:
    def __init__(self) -> None:
        try:
            self._lib = ctypes.CDLL("libcudart.so")
        except OSError as error:
            raise ResourceFailure(
                "CUDA runtime library libcudart.so is not available on this host."
            ) from error

        self._lib.cudaGetErrorString.argtypes = [ctypes.c_int]
        self._lib.cudaGetErrorString.restype = ctypes.c_char_p
        self._lib.cudaGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
        self._lib.cudaGetDeviceCount.restype = ctypes.c_int
        self._lib.cudaSetDevice.argtypes = [ctypes.c_int]
        self._lib.cudaSetDevice.restype = ctypes.c_int
        self._lib.cudaMemGetInfo.argtypes = [
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_size_t),
        ]
        self._lib.cudaMemGetInfo.restype = ctypes.c_int
        self._lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self._lib.cudaMalloc.restype = ctypes.c_int
        self._lib.cudaFree.argtypes = [ctypes.c_void_p]
        self._lib.cudaFree.restype = ctypes.c_int
        self._lib.cudaMemset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
        self._lib.cudaMemset.restype = ctypes.c_int
        self._lib.cudaMemcpy.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        self._lib.cudaMemcpy.restype = ctypes.c_int
        self._lib.cudaDeviceSynchronize.argtypes = []
        self._lib.cudaDeviceSynchronize.restype = ctypes.c_int
        self._lib.cudaRuntimeGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
        self._lib.cudaRuntimeGetVersion.restype = ctypes.c_int
        self._lib.cudaDriverGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
        self._lib.cudaDriverGetVersion.restype = ctypes.c_int
        self._lib.cudaIpcGetMemHandle.argtypes = [
            ctypes.POINTER(_CudaIpcMemHandle),
            ctypes.c_void_p,
        ]
        self._lib.cudaIpcGetMemHandle.restype = ctypes.c_int
        self._lib.cudaIpcOpenMemHandle.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            _CudaIpcMemHandle,
            ctypes.c_uint,
        ]
        self._lib.cudaIpcOpenMemHandle.restype = ctypes.c_int
        self._lib.cudaIpcCloseMemHandle.argtypes = [ctypes.c_void_p]
        self._lib.cudaIpcCloseMemHandle.restype = ctypes.c_int

    def _error(self, code: int, action: str) -> ResourceFailure:
        detail = self._lib.cudaGetErrorString(code)
        message = detail.decode("utf-8", errors="replace") if detail else "unknown CUDA error"
        return ResourceFailure(f"{action} failed: {message} (cudaError={code})")

    def _check(self, code: int, action: str) -> None:
        if code != 0:
            raise self._error(code, action)

    def _select_device(self, device: int) -> None:
        self._check(self._lib.cudaSetDevice(int(device)), f"cudaSetDevice({device})")

    def device_count(self) -> int:
        count = ctypes.c_int()
        self._check(self._lib.cudaGetDeviceCount(ctypes.byref(count)), "cudaGetDeviceCount")
        return int(count.value)

    def mem_get_info(self, device: int) -> tuple[int, int]:
        self._select_device(device)
        free = ctypes.c_size_t()
        total = ctypes.c_size_t()
        self._check(
            self._lib.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total)),
            f"cudaMemGetInfo(device={device})",
        )
        return int(free.value), int(total.value)

    def runtime_version(self) -> str:
        version = ctypes.c_int()
        self._check(self._lib.cudaRuntimeGetVersion(ctypes.byref(version)), "cudaRuntimeGetVersion")
        value = int(version.value)
        major = value // 1000
        minor = (value % 1000) // 10
        return f"{major}.{minor}"

    def driver_version(self) -> str:
        version = ctypes.c_int()
        self._check(self._lib.cudaDriverGetVersion(ctypes.byref(version)), "cudaDriverGetVersion")
        value = int(version.value)
        major = value // 1000
        minor = (value % 1000) // 10
        return f"{major}.{minor}"

    def malloc(self, device: int, size: int) -> int:
        self._select_device(device)
        pointer = ctypes.c_void_p()
        self._check(
            self._lib.cudaMalloc(ctypes.byref(pointer), ctypes.c_size_t(size)),
            f"cudaMalloc(device={device}, size={size})",
        )
        return int(pointer.value)

    def free(self, device: int, pointer: int) -> None:
        self._select_device(device)
        self._check(
            self._lib.cudaFree(ctypes.c_void_p(pointer)),
            f"cudaFree(device={device})",
        )

    def memset(self, device: int, pointer: int, value: int, size: int, *, offset: int = 0) -> None:
        self._select_device(device)
        self._check(
            self._lib.cudaMemset(
                ctypes.c_void_p(pointer + offset),
                int(value),
                ctypes.c_size_t(size),
            ),
            f"cudaMemset(device={device}, size={size})",
        )
        self.synchronize(device)

    def synchronize(self, device: int) -> None:
        self._select_device(device)
        self._check(
            self._lib.cudaDeviceSynchronize(),
            f"cudaDeviceSynchronize(device={device})",
        )

    def copy_host_to_device(self, device: int, pointer: int, payload: bytes, *, offset: int = 0) -> None:
        self._select_device(device)
        if not payload:
            return
        buffer = ctypes.create_string_buffer(payload, len(payload))
        self._check(
            self._lib.cudaMemcpy(
                ctypes.c_void_p(pointer + offset),
                ctypes.cast(buffer, ctypes.c_void_p),
                ctypes.c_size_t(len(payload)),
                CUDA_MEMCPY_HOST_TO_DEVICE,
            ),
            f"cudaMemcpy(host_to_device, device={device}, size={len(payload)})",
        )
        self.synchronize(device)

    def copy_device_to_host(
        self,
        device: int,
        pointer: int,
        size: int,
        *,
        offset: int = 0,
    ) -> bytes:
        self._select_device(device)
        if size <= 0:
            return b""
        buffer = ctypes.create_string_buffer(size)
        self._check(
            self._lib.cudaMemcpy(
                ctypes.cast(buffer, ctypes.c_void_p),
                ctypes.c_void_p(pointer + offset),
                ctypes.c_size_t(size),
                CUDA_MEMCPY_DEVICE_TO_HOST,
            ),
            f"cudaMemcpy(device_to_host, device={device}, size={size})",
        )
        self.synchronize(device)
        return buffer.raw

    def ipc_get_mem_handle(self, device: int, pointer: int) -> bytes:
        self._select_device(device)
        handle = _CudaIpcMemHandle()
        self._check(
            self._lib.cudaIpcGetMemHandle(ctypes.byref(handle), ctypes.c_void_p(pointer)),
            f"cudaIpcGetMemHandle(device={device})",
        )
        return bytes(handle.reserved)

    def ipc_open_mem_handle(self, device: int, encoded_handle: bytes) -> int:
        self._select_device(device)
        if len(encoded_handle) != 64:
            raise ResourceFailure(
                f"CUDA IPC memory handle must be exactly 64 bytes, got {len(encoded_handle)}"
            )
        handle = _CudaIpcMemHandle()
        ctypes.memmove(ctypes.byref(handle), encoded_handle, len(encoded_handle))
        pointer = ctypes.c_void_p()
        self._check(
            self._lib.cudaIpcOpenMemHandle(
                ctypes.byref(pointer),
                handle,
                CUDA_IPC_MEM_LAZY_ENABLE_PEER_ACCESS,
            ),
            f"cudaIpcOpenMemHandle(device={device})",
        )
        return int(pointer.value)

    def ipc_close_mem_handle(self, device: int, pointer: int) -> None:
        self._select_device(device)
        self._check(
            self._lib.cudaIpcCloseMemHandle(ctypes.c_void_p(pointer)),
            f"cudaIpcCloseMemHandle(device={device})",
        )


_DEFAULT_RUNTIME: CudaRuntime | None = None


def get_cuda_runtime() -> CudaRuntime:
    global _DEFAULT_RUNTIME
    if _DEFAULT_RUNTIME is None:
        _DEFAULT_RUNTIME = CudaRuntime()
    return _DEFAULT_RUNTIME


def _lease_handle(device: int, handle_bytes: bytes) -> str:
    encoded = base64.b64encode(handle_bytes).decode("ascii")
    return f"cuda-ipc:{device}:{encoded}"


def _parse_lease_handle(lease_handle: str) -> tuple[int, bytes]:
    try:
        prefix, device_text, encoded = lease_handle.split(":", 2)
    except ValueError as error:
        raise ResourceFailure(f"Malformed CUDA IPC lease handle: {lease_handle!r}") from error
    if prefix != "cuda-ipc":
        raise ResourceFailure(f"Unsupported CUDA IPC lease handle prefix: {lease_handle!r}")
    try:
        device = int(device_text)
    except ValueError as error:
        raise ResourceFailure(f"Malformed CUDA IPC device id in handle: {lease_handle!r}") from error
    try:
        handle_bytes = base64.b64decode(encoded.encode("ascii"), validate=True)
    except Exception as error:
        raise ResourceFailure(f"Malformed CUDA IPC payload in handle: {lease_handle!r}") from error
    return device, handle_bytes


@dataclass
class GpuLease:
    record: LeaseRecord
    device: int
    pointer: int
    runtime: CudaRuntime
    _closed: bool = False

    def write(self, payload: bytes) -> None:
        if len(payload) > self.record.usable_bytes:
            raise ResourceFailure(
                f"Payload length {len(payload)} exceeds gpu lease size {self.record.usable_bytes}"
            )
        self.runtime.copy_host_to_device(self.device, self.pointer, payload)
        remaining = self.record.usable_bytes - len(payload)
        if remaining:
            self._zero_range(len(payload), remaining)

    def read(self, length: int | None = None, offset: int = 0) -> bytes:
        requested = self.record.usable_bytes if length is None else length
        if offset < 0 or requested < 0 or offset + requested > self.record.usable_bytes:
            raise ResourceFailure(
                f"Invalid gpu lease read offset={offset} length={requested} "
                f"for size {self.record.usable_bytes}"
            )
        return self.runtime.copy_device_to_host(
            self.device,
            self.pointer,
            requested,
            offset=offset,
        )

    def read_leaf(self, leaf_index: int, leaf_size: int) -> bytes:
        start = leaf_index * leaf_size
        return self.read(length=leaf_size, offset=start)

    def zeroize(self) -> None:
        self._zero_range(0, self.record.usable_bytes)

    def verify_zeroized(self) -> bool:
        remaining = self.record.usable_bytes
        offset = 0
        while remaining:
            chunk_size = min(ZERO_CHUNK_BYTES, remaining)
            if self.read(length=chunk_size, offset=offset) != bytes(chunk_size):
                return False
            remaining -= chunk_size
            offset += chunk_size
        return True

    def close(self) -> None:
        if self._closed:
            return
        try:
            self.runtime.free(self.device, self.pointer)
        finally:
            self._closed = True

    def _zero_range(self, offset: int, length: int) -> None:
        if length <= 0:
            return
        remaining = length
        cursor = offset
        while remaining:
            chunk = min(ZERO_CHUNK_BYTES, remaining)
            self.runtime.memset(self.device, self.pointer, 0, chunk, offset=cursor)
            cursor += chunk
            remaining -= chunk


@dataclass
class GpuLeaseAttachment:
    device: int
    usable_bytes: int
    pointer: int
    runtime: CudaRuntime
    _closed: bool = False

    def write(self, payload: bytes) -> None:
        if len(payload) > self.usable_bytes:
            raise ResourceFailure(
                f"Payload length {len(payload)} exceeds gpu attachment size {self.usable_bytes}"
            )
        self.runtime.copy_host_to_device(self.device, self.pointer, payload)
        remaining = self.usable_bytes - len(payload)
        if remaining:
            self.zeroize(offset=len(payload), length=remaining)

    def read(self, length: int | None = None, offset: int = 0) -> bytes:
        requested = self.usable_bytes if length is None else length
        if offset < 0 or requested < 0 or offset + requested > self.usable_bytes:
            raise ResourceFailure(
                f"Invalid gpu attachment read offset={offset} length={requested} "
                f"for size {self.usable_bytes}"
            )
        return self.runtime.copy_device_to_host(
            self.device,
            self.pointer,
            requested,
            offset=offset,
        )

    def read_leaf(self, leaf_index: int, leaf_size: int) -> bytes:
        start = leaf_index * leaf_size
        return self.read(length=leaf_size, offset=start)

    def zeroize(self, *, offset: int = 0, length: int | None = None) -> None:
        requested = self.usable_bytes - offset if length is None else length
        if requested <= 0:
            return
        remaining = requested
        cursor = offset
        while remaining:
            chunk = min(ZERO_CHUNK_BYTES, remaining)
            self.runtime.memset(self.device, self.pointer, 0, chunk, offset=cursor)
            cursor += chunk
            remaining -= chunk

    def verify_zeroized(self) -> bool:
        remaining = self.usable_bytes
        offset = 0
        while remaining:
            chunk_size = min(ZERO_CHUNK_BYTES, remaining)
            if self.read(length=chunk_size, offset=offset) != bytes(chunk_size):
                return False
            remaining -= chunk_size
            offset += chunk_size
        return True

    def close(self) -> None:
        if self._closed:
            return
        try:
            self.runtime.ipc_close_mem_handle(self.device, self.pointer)
        finally:
            self._closed = True


def create_gpu_lease(
    *,
    session_id: str,
    region_id: str,
    device: int,
    usable_bytes: int,
    cleanup_policy: CleanupPolicy,
    lease_duration_ms: int,
    runtime: CudaRuntime | None = None,
) -> GpuLease:
    if usable_bytes <= 0:
        raise ResourceFailure(f"GPU lease size must be positive: {usable_bytes}")
    active_runtime = runtime or get_cuda_runtime()
    if device < 0 or device >= active_runtime.device_count():
        raise ResourceFailure(f"Requested CUDA device is unavailable: {device}")

    pointer = active_runtime.malloc(device, usable_bytes)
    lease_expiry = (
        datetime.now(UTC) + timedelta(milliseconds=max(1, lease_duration_ms))
    ).isoformat()
    record = LeaseRecord(
        region_id=region_id,
        region_type="gpu",
        usable_bytes=usable_bytes,
        lease_handle=_lease_handle(device, active_runtime.ipc_get_mem_handle(device, pointer)),
        lease_expiry=lease_expiry,
        cleanup_policy=cleanup_policy,
    )
    return GpuLease(
        record=record,
        device=device,
        pointer=pointer,
        runtime=active_runtime,
    )


def attach_gpu_lease(
    lease_handle: str,
    *,
    usable_bytes: int,
    runtime: CudaRuntime | None = None,
) -> GpuLeaseAttachment:
    device, handle_bytes = _parse_lease_handle(lease_handle)
    active_runtime = runtime or get_cuda_runtime()
    pointer = active_runtime.ipc_open_mem_handle(device, handle_bytes)
    return GpuLeaseAttachment(
        device=device,
        usable_bytes=usable_bytes,
        pointer=pointer,
        runtime=active_runtime,
    )


def release_gpu_lease(
    lease: GpuLease,
    *,
    zeroize: bool,
    verify_zeroization: bool,
) -> str:
    try:
        if zeroize:
            lease.zeroize()
            if verify_zeroization and not lease.verify_zeroized():
                raise ResourceFailure("GPU lease zeroization verification failed")
            if verify_zeroization:
                return "ZEROIZED_AND_VERIFIED"
            return "ZEROIZED_AND_RELEASED"
        return "RELEASED_WITHOUT_ZEROIZE"
    finally:
        lease.close()
