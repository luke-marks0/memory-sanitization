from __future__ import annotations

import json
from pathlib import Path


def summarize_results(paths: list[str]) -> dict[str, object]:
    verdicts: dict[str, int] = {}
    total = 0
    for path_str in paths:
        payload = json.loads(Path(path_str).read_text(encoding="utf-8"))
        verdict = payload.get("verdict", "UNKNOWN")
        verdicts[verdict] = verdicts.get(verdict, 0) + 1
        total += 1
    return {
        "result_count": total,
        "verdict_counts": verdicts,
    }

