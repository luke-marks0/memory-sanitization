from __future__ import annotations

from pose.common.gpu_lease import attach_gpu_lease, create_gpu_lease, release_gpu_lease
from pose.protocol.messages import CleanupPolicy


class FakeCudaRuntime:
    def __init__(self) -> None:
        self._next_pointer = 0x1000
        self._next_handle = 1
        self._buffers: dict[int, bytearray] = {}
        self._handles: dict[bytes, int] = {}

    def device_count(self) -> int:
        return 1

    def malloc(self, device: int, size: int) -> int:
        assert device == 0
        pointer = self._next_pointer
        self._next_pointer += max(size, 1) + 0x1000
        self._buffers[pointer] = bytearray(size)
        return pointer

    def free(self, device: int, pointer: int) -> None:
        assert device == 0
        self._buffers.pop(pointer, None)

    def memset(self, device: int, pointer: int, value: int, size: int, *, offset: int = 0) -> None:
        assert device == 0
        self._buffers[pointer][offset : offset + size] = bytes([value & 0xFF]) * size

    def synchronize(self, device: int) -> None:
        assert device == 0

    def copy_host_to_device(self, device: int, pointer: int, payload: bytes, *, offset: int = 0) -> None:
        assert device == 0
        self._buffers[pointer][offset : offset + len(payload)] = payload

    def copy_device_to_host(self, device: int, pointer: int, size: int, *, offset: int = 0) -> bytes:
        assert device == 0
        return bytes(self._buffers[pointer][offset : offset + size])

    def ipc_get_mem_handle(self, device: int, pointer: int) -> bytes:
        assert device == 0
        handle = self._next_handle.to_bytes(64, "big")
        self._next_handle += 1
        self._handles[handle] = pointer
        return handle

    def ipc_open_mem_handle(self, device: int, encoded_handle: bytes) -> int:
        assert device == 0
        return self._handles[encoded_handle]

    def ipc_close_mem_handle(self, device: int, pointer: int) -> None:
        assert device == 0
        assert pointer in self._buffers


def test_gpu_lease_round_trip_and_zeroize() -> None:
    runtime = FakeCudaRuntime()
    cleanup_policy = CleanupPolicy(zeroize=True, verify_zeroization=True)
    lease = create_gpu_lease(
        session_id="session-id",
        region_id="gpu-0",
        device=0,
        usable_bytes=4096,
        cleanup_policy=cleanup_policy,
        lease_duration_ms=60000,
        runtime=runtime,
    )
    attachment = attach_gpu_lease(
        lease.record.lease_handle,
        usable_bytes=lease.record.usable_bytes,
        runtime=runtime,
    )

    payload = bytes(range(256)) * 16
    attachment.write(payload)

    assert lease.read(length=len(payload)) == payload
    assert attachment.read_leaf(0, 256) == bytes(range(256))

    attachment.zeroize()
    assert lease.verify_zeroized() is True

    attachment.close()
    cleanup_status = release_gpu_lease(lease, zeroize=True, verify_zeroization=True)
    assert cleanup_status == "ZEROIZED_AND_VERIFIED"
