#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

from pose.common.upstream import validate_upstream_snapshot


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    validate_upstream_snapshot(repo_root)
    print("upstream integrity: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

