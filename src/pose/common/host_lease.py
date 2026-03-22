from __future__ import annotations

import mmap
import os
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from multiprocessing import shared_memory
from typing import BinaryIO

from pose.common.errors import ResourceFailure
from pose.protocol.messages import CleanupPolicy, LeaseRecord

ZERO_CHUNK = b"\x00" * (1024 * 1024)


def _shared_memory_name(session_id: str, region_id: str) -> str:
    raw = f"pose_{session_id}_{region_id}"
    return "".join(character if character.isalnum() else "_" for character in raw)


@dataclass
class HostLease:
    record: LeaseRecord
    fd: int
    mapping: mmap.mmap
    _backing_file: BinaryIO | None = None
    _shared_memory: shared_memory.SharedMemory | None = None

    def write(self, payload: bytes) -> None:
        if len(payload) > self.record.usable_bytes:
            raise ResourceFailure(
                f"Payload length {len(payload)} exceeds host lease size {self.record.usable_bytes}"
            )
        self.write_at(payload, offset=0)
        remaining = self.record.usable_bytes - len(payload)
        if remaining:
            self._zero_range(len(payload), remaining)
        self.mapping.flush()

    def write_at(self, payload: bytes, *, offset: int) -> None:
        if offset < 0 or offset + len(payload) > self.record.usable_bytes:
            raise ResourceFailure(
                f"Invalid host lease write offset={offset} length={len(payload)} "
                f"for size {self.record.usable_bytes}"
            )
        self.mapping[offset : offset + len(payload)] = payload

    def read(self, length: int | None = None, offset: int = 0) -> bytes:
        requested = self.record.usable_bytes if length is None else length
        if offset < 0 or requested < 0 or offset + requested > self.record.usable_bytes:
            raise ResourceFailure(
                f"Invalid host lease read offset={offset} length={requested} "
                f"for size {self.record.usable_bytes}"
            )
        self.mapping.seek(offset)
        return self.mapping.read(requested)

    def read_leaf(self, leaf_index: int, leaf_size: int) -> bytes:
        start = leaf_index * leaf_size
        return self.read(length=leaf_size, offset=start)

    def zeroize(self) -> None:
        self._zero_range(0, self.record.usable_bytes)
        self.mapping.flush()

    def verify_zeroized(self) -> bool:
        self.mapping.seek(0)
        return self.mapping.read(self.record.usable_bytes) == bytes(self.record.usable_bytes)

    def close(self) -> None:
        try:
            self.mapping.close()
        finally:
            if self._shared_memory is not None:
                try:
                    self._shared_memory.close()
                finally:
                    try:
                        self._shared_memory.unlink()
                    except FileNotFoundError:
                        pass
            elif self._backing_file is not None:
                self._backing_file.close()
            else:
                os.close(self.fd)

    def _zero_range(self, offset: int, length: int) -> None:
        if length <= 0:
            return
        remaining = length
        cursor = offset
        while remaining:
            chunk = ZERO_CHUNK[: min(len(ZERO_CHUNK), remaining)]
            self.mapping[cursor : cursor + len(chunk)] = chunk
            cursor += len(chunk)
            remaining -= len(chunk)


def build_placeholder_lease(
    region_id: str,
    region_type: str,
    usable_bytes: int,
    cleanup_policy: CleanupPolicy,
) -> LeaseRecord:
    return LeaseRecord(
        region_id=region_id,
        region_type=region_type,
        usable_bytes=usable_bytes,
        lease_handle=f"{region_type}:{region_id}:placeholder",
        lease_expiry="unbounded-foundation-placeholder",
        cleanup_policy=cleanup_policy,
    )


def create_host_lease(
    *,
    session_id: str,
    region_id: str,
    usable_bytes: int,
    cleanup_policy: CleanupPolicy,
    lease_duration_ms: int,
) -> HostLease:
    if usable_bytes <= 0:
        raise ResourceFailure(f"Host lease size must be positive: {usable_bytes}")

    lease_expiry = (
        datetime.now(UTC) + timedelta(milliseconds=max(1, lease_duration_ms))
    ).isoformat()

    if hasattr(os, "memfd_create"):
        fd = os.memfd_create(f"pose-{session_id}-{region_id}", 0)
        os.ftruncate(fd, usable_bytes)
        mapping = mmap.mmap(fd, usable_bytes)
        backing_file = None
        shared = None
        lease_handle = f"/proc/{os.getpid()}/fd/{fd}"
    else:
        shared = shared_memory.SharedMemory(
            name=_shared_memory_name(session_id, region_id),
            create=True,
            size=usable_bytes,
        )
        fd = shared._fd
        mapping = mmap.mmap(fd, usable_bytes)
        backing_file = None
        lease_handle = f"shm:{shared.name}"

    record = LeaseRecord(
        region_id=region_id,
        region_type="host",
        usable_bytes=usable_bytes,
        lease_handle=lease_handle,
        lease_expiry=lease_expiry,
        cleanup_policy=cleanup_policy,
    )
    return HostLease(
        record=record,
        fd=fd,
        mapping=mapping,
        _backing_file=backing_file,
        _shared_memory=shared,
    )


@dataclass
class HostLeaseAttachment:
    fd: int
    usable_bytes: int
    mapping: mmap.mmap
    _backing_file: BinaryIO | None = None
    _shared_memory: shared_memory.SharedMemory | None = None

    def write(self, payload: bytes) -> None:
        if len(payload) > self.usable_bytes:
            raise ResourceFailure(
                f"Payload length {len(payload)} exceeds host attachment size {self.usable_bytes}"
            )
        self.write_at(payload, offset=0)
        remaining = self.usable_bytes - len(payload)
        if remaining:
            self.mapping[len(payload) : self.usable_bytes] = bytes(remaining)
        self.mapping.flush()

    def write_at(self, payload: bytes, *, offset: int) -> None:
        if offset < 0 or offset + len(payload) > self.usable_bytes:
            raise ResourceFailure(
                f"Invalid host attachment write offset={offset} length={len(payload)} "
                f"for size {self.usable_bytes}"
            )
        self.mapping[offset : offset + len(payload)] = payload
        self.mapping.flush()

    def read(self, length: int, offset: int = 0) -> bytes:
        if offset < 0 or length < 0 or offset + length > self.usable_bytes:
            raise ResourceFailure(
                f"Invalid host attachment read offset={offset} length={length} "
                f"for size {self.usable_bytes}"
            )
        self.mapping.seek(offset)
        return self.mapping.read(length)

    def read_leaf(self, leaf_index: int, leaf_size: int) -> bytes:
        start = leaf_index * leaf_size
        return self.read(leaf_size, offset=start)

    def close(self) -> None:
        try:
            self.mapping.close()
        finally:
            if self._shared_memory is not None:
                self._shared_memory.close()
            elif self._backing_file is not None:
                self._backing_file.close()
            else:
                os.close(self.fd)


def attach_host_lease(
    lease_handle: str,
    *,
    usable_bytes: int,
    read_only: bool = False,
) -> HostLeaseAttachment:
    if lease_handle.startswith("shm:"):
        shared = shared_memory.SharedMemory(name=lease_handle.split(":", 1)[1])
        access = mmap.ACCESS_READ if read_only else mmap.ACCESS_WRITE
        return HostLeaseAttachment(
            fd=shared._fd,
            usable_bytes=usable_bytes,
            mapping=mmap.mmap(shared._fd, usable_bytes, access=access),
            _shared_memory=shared,
        )

    mode = "rb" if read_only else "r+b"
    backing_file = open(lease_handle, mode, buffering=0)
    access = mmap.ACCESS_READ if read_only else mmap.ACCESS_WRITE
    return HostLeaseAttachment(
        fd=backing_file.fileno(),
        usable_bytes=usable_bytes,
        mapping=mmap.mmap(backing_file.fileno(), usable_bytes, access=access),
        _backing_file=backing_file,
    )


def release_host_lease(
    lease: HostLease,
    *,
    zeroize: bool,
    verify_zeroization: bool,
) -> str:
    try:
        if zeroize:
            lease.zeroize()
            if verify_zeroization and not lease.verify_zeroized():
                raise ResourceFailure("Host lease zeroization verification failed")
            if verify_zeroization:
                return "ZEROIZED_AND_VERIFIED"
            return "ZEROIZED_AND_RELEASED"
        return "RELEASED_WITHOUT_ZEROIZE"
    finally:
        lease.close()
