from __future__ import annotations

from pathlib import Path

from pose.protocol.codec import dump_json_file
from pose.protocol.result_schema import SessionResult


def write_result(path: Path, result: SessionResult) -> None:
    dump_json_file(path, result.to_dict())

