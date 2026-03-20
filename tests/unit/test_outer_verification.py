from __future__ import annotations

from pose.common.merkle import commit_payload
from pose.verifier.outer import OuterChallengeOpening, verify_outer_challenge_response


def _valid_opening() -> tuple[bytes, list[OuterChallengeOpening]]:
    payload = (b"a" * 16) + (b"b" * 16)
    commitment = commit_payload(payload, leaf_size=16)
    opening = commitment.opening(1, payload[16:32])
    wrapped = OuterChallengeOpening(
        region_id="host-0",
        session_manifest_root="session-root",
        leaf_index=1,
        leaf=opening.leaf,
        sibling_hashes=opening.sibling_hashes,
    )
    return commitment.root, [wrapped]


def test_outer_verification_accepts_valid_opening() -> None:
    root, openings = _valid_opening()
    assert (
        verify_outer_challenge_response(
            expected_region_id="host-0",
            expected_session_manifest_root="session-root",
            expected_indices=[1],
            root=root,
            leaf_size=16,
            openings=openings,
        )
        is True
    )


def test_outer_verification_rejects_wrong_region_id() -> None:
    root, openings = _valid_opening()
    assert (
        verify_outer_challenge_response(
            expected_region_id="host-1",
            expected_session_manifest_root="session-root",
            expected_indices=[1],
            root=root,
            leaf_size=16,
            openings=openings,
        )
        is False
    )


def test_outer_verification_rejects_stale_manifest_root() -> None:
    root, openings = _valid_opening()
    assert (
        verify_outer_challenge_response(
            expected_region_id="host-0",
            expected_session_manifest_root="different-root",
            expected_indices=[1],
            root=root,
            leaf_size=16,
            openings=openings,
        )
        is False
    )


def test_outer_verification_rejects_tampered_leaf() -> None:
    root, openings = _valid_opening()
    tampered = [
        OuterChallengeOpening(
            region_id=opening.region_id,
            session_manifest_root=opening.session_manifest_root,
            leaf_index=opening.leaf_index,
            leaf=b"c" * len(opening.leaf),
            sibling_hashes=opening.sibling_hashes,
        )
        for opening in openings
    ]
    assert (
        verify_outer_challenge_response(
            expected_region_id="host-0",
            expected_session_manifest_root="session-root",
            expected_indices=[1],
            root=root,
            leaf_size=16,
            openings=tampered,
        )
        is False
    )


def test_outer_verification_rejects_duplicate_challenge_indices() -> None:
    root, openings = _valid_opening()
    assert (
        verify_outer_challenge_response(
            expected_region_id="host-0",
            expected_session_manifest_root="session-root",
            expected_indices=[1, 1],
            root=root,
            leaf_size=16,
            openings=openings * 2,
        )
        is False
    )
