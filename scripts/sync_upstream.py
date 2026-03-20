#!/usr/bin/env python
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pose.common.upstream import (
    OFFICIAL_UPSTREAM_URL,
    compute_tree_sha256,
    load_upstream_lock,
    vendor_root,
)


def run(*args: str, cwd: Path | None = None) -> None:
    subprocess.run(args, cwd=cwd, check=True)


def main() -> int:
    repo_root = REPO_ROOT
    metadata = load_upstream_lock(repo_root / "vendor" / "UPSTREAM.lock")
    upstream_url = metadata["upstream_url"]
    upstream_commit = metadata["upstream_commit"]

    if upstream_url != OFFICIAL_UPSTREAM_URL:
        raise SystemExit(f"Refusing to sync non-official upstream URL: {upstream_url}")

    with tempfile.TemporaryDirectory(prefix="pose-sync-upstream-") as temp_dir:
        temp_path = Path(temp_dir)
        clone_dir = temp_path / "rust-fil-proofs"
        run("git", "init", str(clone_dir))
        run("git", "remote", "add", "origin", upstream_url, cwd=clone_dir)
        run("git", "fetch", "--depth", "1", "origin", upstream_commit, cwd=clone_dir)
        run("git", "checkout", "FETCH_HEAD", cwd=clone_dir)

        destination = vendor_root(repo_root)
        subprocess.run(
            (
                "rsync",
                "-a",
                "--delete",
                "--exclude",
                ".git",
                f"{clone_dir}/",
                f"{destination}/",
            ),
            check=True,
        )

    actual_hash = compute_tree_sha256(vendor_root(repo_root))
    expected_hash = metadata["tree_sha256"]
    if actual_hash != expected_hash:
        raise SystemExit(
            "Vendored tree hash mismatch after sync: "
            f"expected {expected_hash}, got {actual_hash}"
        )

    print(destination)
    print(f"tree_sha256={actual_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
