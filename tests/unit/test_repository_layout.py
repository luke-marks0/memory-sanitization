from __future__ import annotations

from pathlib import Path


def test_repository_layout_exists() -> None:
    root = Path(__file__).resolve().parents[2]
    required_paths = [
        root / "docs" / "repository-spec.md",
        root / "docs" / "architecture.md",
        root / "docs" / "graph-construction.md",
        root / "docs" / "security-model.md",
        root / "docs" / "references" / "software-based-memory-erasure-relaxed-isolation.pdf",
        root / "proto" / "pose" / "v1" / "session.proto",
        root / "src" / "pose" / "cli" / "main.py",
        root / "src" / "pose" / "cli" / "calibrate.py",
        root / "src" / "pose" / "graphs",
        root / "src" / "pose" / "hashing",
        root / "src" / "pose" / "benchmarks" / "calibration.py",
        root / "bench_profiles" / "dev-small.yaml",
        root / "tests" / "parity",
    ]
    for path in required_paths:
        assert path.exists(), path
