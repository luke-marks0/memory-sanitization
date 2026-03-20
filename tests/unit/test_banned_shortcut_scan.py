from __future__ import annotations

from pathlib import Path

from pose.common.integrity import scan_production_tree_for_banned_shortcuts


def test_banned_shortcut_scan_passes() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    assert scan_production_tree_for_banned_shortcuts(repo_root) == []
