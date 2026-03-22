from __future__ import annotations

from pathlib import Path

from pose.common.integrity import scan_production_tree_for_banned_shortcuts


def test_banned_shortcut_scan_passes() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    assert scan_production_tree_for_banned_shortcuts(repo_root) == []


def test_banned_shortcut_scan_detects_pose_db_shortcut_flags(tmp_path: Path) -> None:
    production_file = tmp_path / "src" / "pose" / "prover" / "session.py"
    production_file.parent.mkdir(parents=True, exist_ok=True)
    production_file.write_text(
        "external_response_table = {}\nmanaged_memory_fallback = True\n",
        encoding="utf-8",
    )

    matches = scan_production_tree_for_banned_shortcuts(tmp_path)

    assert [(match.pattern, match.line_number) for match in matches] == [
        ("external_response_table", 1),
        ("managed_memory_fallback", 2),
    ]
