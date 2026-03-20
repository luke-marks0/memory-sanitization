from __future__ import annotations

from pathlib import Path

from pose.common.upstream import load_upstream_lock, validate_upstream_snapshot


def test_upstream_lock_matches_vendored_tree() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    validate_upstream_snapshot(repo_root)


def test_upstream_lock_records_official_source() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    metadata = load_upstream_lock(repo_root / "vendor" / "UPSTREAM.lock")
    assert metadata["upstream_url"] == "https://github.com/filecoin-project/rust-fil-proofs.git"
    assert metadata["local_patch_status"] == "clean"
    assert metadata["component_tags"]

