from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def _default_encoder(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")


def to_json(data: Any) -> str:
    return json.dumps(data, default=_default_encoder, indent=2, sort_keys=True)


def dump_json_file(path: Path, data: Any) -> None:
    path.write_text(to_json(data) + "\n", encoding="utf-8")


def load_json_file(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))

