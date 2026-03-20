from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pose.common.errors import ResourceFailure

if TYPE_CHECKING:
    from pose.verifier.leasing import HostLease


@dataclass(frozen=True)
class HostRegionConfig:
    usable_bytes: int
    mlock: bool = False
    huge_pages: bool = False
    numa_node: int | None = None


def materialize_payload(lease: HostLease, payload: bytes) -> None:
    if len(payload) > lease.record.usable_bytes:
        raise ResourceFailure(
            f"Host materialization payload exceeds lease size: "
            f"payload={len(payload)} lease={lease.record.usable_bytes}"
        )
    lease.write(payload)
