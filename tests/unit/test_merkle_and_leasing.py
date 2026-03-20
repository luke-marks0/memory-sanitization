from __future__ import annotations

from pose.common.merkle import commit_payload, verify_opening
from pose.protocol.messages import CleanupPolicy
from pose.verifier.leasing import create_host_lease, release_host_lease


def test_merkle_commitment_and_opening_verify() -> None:
    payload = (b"a" * 16) + (b"b" * 16)
    commitment = commit_payload(payload, leaf_size=16)
    opening = commitment.opening(1, payload[16:32])

    assert commitment.leaf_count == 2
    assert verify_opening(commitment.root, opening) is True


def test_host_lease_zeroizes_before_release() -> None:
    lease = create_host_lease(
        session_id="session",
        region_id="host-0",
        usable_bytes=32,
        cleanup_policy=CleanupPolicy(zeroize=True, verify_zeroization=True),
        lease_duration_ms=1000,
    )
    lease.write(b"x" * 32)

    status = release_host_lease(
        lease,
        zeroize=True,
        verify_zeroization=True,
    )

    assert status == "ZEROIZED_AND_VERIFIED"
