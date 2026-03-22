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


def benchmarks_root() -> Path:
    root = state_root() / "benchmarks"
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
    session_seed_hex: str
    session_plan_root: str
    graph_family: str
    graph_parameter_n: int
    graph_descriptor_digest: str
    label_width_bits: int
    label_count_m: int
    gamma: int
    hash_backend: str
    region_id: str
    region_slot_count: int
    challenge_policy: dict[str, int | float | bool]
    deadline_us: int
    cleanup_policy: dict[str, bool]
    adversary_model: str
    attacker_budget_bytes_assumed: int
    q_bound: int
    host_total_bytes: int
    host_usable_bytes: int
    host_covered_bytes: int
    covered_bytes: int
    slack_bytes: int
    coverage_fraction: float
    scratch_peak_bytes: int
    declared_stage_copy_bytes: int
    formal_claim_notes: list[str]
    operational_claim_notes: list[str]
    claim_notes: list[str]
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
            session_seed_hex=str(payload["session_seed_hex"]),
            session_plan_root=str(payload["session_plan_root"]),
            graph_family=str(payload["graph_family"]),
            graph_parameter_n=int(payload["graph_parameter_n"]),
            graph_descriptor_digest=str(payload["graph_descriptor_digest"]),
            label_width_bits=int(payload["label_width_bits"]),
            label_count_m=int(payload["label_count_m"]),
            gamma=int(payload["gamma"]),
            hash_backend=str(payload["hash_backend"]),
            region_id=str(payload["region_id"]),
            region_slot_count=int(payload["region_slot_count"]),
            challenge_policy=dict(payload["challenge_policy"]),
            deadline_us=int(payload["deadline_us"]),
            cleanup_policy={key: bool(value) for key, value in dict(payload["cleanup_policy"]).items()},
            adversary_model=str(payload["adversary_model"]),
            attacker_budget_bytes_assumed=int(payload["attacker_budget_bytes_assumed"]),
            q_bound=int(payload["q_bound"]),
            host_total_bytes=int(payload["host_total_bytes"]),
            host_usable_bytes=int(payload["host_usable_bytes"]),
            host_covered_bytes=int(payload["host_covered_bytes"]),
            covered_bytes=int(payload["covered_bytes"]),
            slack_bytes=int(payload["slack_bytes"]),
            coverage_fraction=float(payload["coverage_fraction"]),
            scratch_peak_bytes=int(payload.get("scratch_peak_bytes", 0)),
            declared_stage_copy_bytes=int(payload.get("declared_stage_copy_bytes", 0)),
            formal_claim_notes=[str(item) for item in payload.get("formal_claim_notes", [])],
            operational_claim_notes=[str(item) for item in payload.get("operational_claim_notes", [])],
            claim_notes=[str(item) for item in payload.get("claim_notes", [])],
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
