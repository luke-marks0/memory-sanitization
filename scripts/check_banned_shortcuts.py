#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

from pose.common.integrity import format_matches, scan_production_tree_for_banned_shortcuts


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    matches = scan_production_tree_for_banned_shortcuts(repo_root)
    if matches:
        raise SystemExit(format_matches(matches))
    print("banned shortcut scan: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
