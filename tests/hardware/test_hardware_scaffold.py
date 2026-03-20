from __future__ import annotations

from pathlib import Path


def test_hardware_profiles_and_docs_exist() -> None:
    root = Path(__file__).resolve().parents[2]
    assert (root / "docs" / "hardware" / "single-h100.md").exists()
    assert (root / "docs" / "hardware" / "eight-h100.md").exists()
    assert (root / "bench_profiles" / "single-h100-hbm-max.yaml").exists()
    assert (root / "bench_profiles" / "eight-h100-hybrid-max.yaml").exists()

