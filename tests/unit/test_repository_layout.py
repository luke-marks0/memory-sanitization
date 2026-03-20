from __future__ import annotations

from pathlib import Path


def test_repository_layout_exists() -> None:
    root = Path(__file__).resolve().parents[2]
    required_paths = [
        root / "docs" / "repository-spec.md",
        root / "docs" / "architecture.md",
        root / "vendor" / "UPSTREAM.lock",
        root / "rust" / "pose_filecoin_bridge" / "Cargo.toml",
        root / "proto" / "pose" / "v1" / "session.proto",
        root / "src" / "pose" / "cli" / "main.py",
        root / "bench_profiles" / "dev-small.yaml",
        root / "scripts" / "sync_upstream.sh",
        root / "tests" / "parity",
    ]
    for path in required_paths:
        assert path.exists(), path

