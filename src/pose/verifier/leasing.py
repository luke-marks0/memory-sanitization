from __future__ import annotations

from pose.protocol.messages import CleanupPolicy, LeaseRecord


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

