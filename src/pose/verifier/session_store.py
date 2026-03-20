from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from pose.protocol.codec import dump_json_file, load_json_file
from pose.protocol.messages import SessionPlan
from pose.protocol.result_schema import SessionResult


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def state_root() -> Path:
    return repo_root() / ".pose"


def results_root() -> Path:
    root = state_root() / "results"
    root.mkdir(parents=True, exist_ok=True)
    return root


def sessions_root() -> Path:
    root = state_root() / "sessions"
    root.mkdir(parents=True, exist_ok=True)
    return root


def result_artifact_path(session_id: str, *, run_class: str) -> Path:
    if run_class == "cold":
        return results_root() / f"{session_id}.json"
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return results_root() / f"{session_id}-{run_class}-{timestamp}.json"


def write_result_artifact(result: SessionResult) -> Path:
    path = result_artifact_path(result.session_id, run_class=result.run_class)
    dump_json_file(path, result.to_dict())
    return path


def cold_result_artifact_path(session_id: str) -> Path:
    return results_root() / f"{session_id}.json"


def load_cold_result_artifact(session_id: str) -> SessionResult:
    return SessionResult.from_dict(load_json_file(cold_result_artifact_path(session_id)))


@dataclass(frozen=True)
class PlanFile:
    session_plan: SessionPlan
    retain_session: bool = False


def load_plan_file(path: Path) -> PlanFile:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Plan file must decode to a mapping")
    session_plan_payload = payload.get("session_plan", payload)
    if not isinstance(session_plan_payload, dict):
        raise TypeError("Plan file session_plan must decode to a mapping")
    return PlanFile(
        session_plan=SessionPlan.from_dict(dict(session_plan_payload)),
        retain_session=bool(payload.get("retain_session", False)),
    )


@dataclass(frozen=True)
class ResidentSessionRecord:
    session_id: str
    profile_name: str
    session_nonce: str
    session_plan_root: str
    session_manifest_root: str
    region_id: str
    region_root_hex: str
    region_manifest_root: str
    challenge_leaf_size: int
    challenge_policy: dict[str, int | float]
    deadline_ms: int
    cleanup_policy: dict[str, bool]
    host_total_bytes: int
    host_usable_bytes: int
    host_covered_bytes: int
    real_porep_bytes: int
    tail_filler_bytes: int
    real_porep_ratio: float
    coverage_fraction: float
    inner_filecoin_verified: bool
    socket_path: str
    process_id: int
    lease_expiry: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ResidentSessionRecord":
        return cls(
            session_id=str(payload["session_id"]),
            profile_name=str(payload["profile_name"]),
            session_nonce=str(payload["session_nonce"]),
            session_plan_root=str(payload["session_plan_root"]),
            session_manifest_root=str(payload["session_manifest_root"]),
            region_id=str(payload["region_id"]),
            region_root_hex=str(payload["region_root_hex"]),
            region_manifest_root=str(payload["region_manifest_root"]),
            challenge_leaf_size=int(payload["challenge_leaf_size"]),
            challenge_policy=dict(payload["challenge_policy"]),
            deadline_ms=int(payload["deadline_ms"]),
            cleanup_policy={key: bool(value) for key, value in dict(payload["cleanup_policy"]).items()},
            host_total_bytes=int(payload["host_total_bytes"]),
            host_usable_bytes=int(payload["host_usable_bytes"]),
            host_covered_bytes=int(payload["host_covered_bytes"]),
            real_porep_bytes=int(payload["real_porep_bytes"]),
            tail_filler_bytes=int(payload["tail_filler_bytes"]),
            real_porep_ratio=float(payload["real_porep_ratio"]),
            coverage_fraction=float(payload["coverage_fraction"]),
            inner_filecoin_verified=bool(payload["inner_filecoin_verified"]),
            socket_path=str(payload["socket_path"]),
            process_id=int(payload["process_id"]),
            lease_expiry=str(payload["lease_expiry"]),
        )


def resident_session_path(session_id: str) -> Path:
    return sessions_root() / f"{session_id}.json"


def write_resident_session(record: ResidentSessionRecord) -> Path:
    path = resident_session_path(record.session_id)
    dump_json_file(path, record.to_dict())
    return path


def load_resident_session(session_id: str) -> ResidentSessionRecord:
    return ResidentSessionRecord.from_dict(load_json_file(resident_session_path(session_id)))


def delete_resident_session(session_id: str) -> None:
    path = resident_session_path(session_id)
    if path.exists():
        path.unlink()
