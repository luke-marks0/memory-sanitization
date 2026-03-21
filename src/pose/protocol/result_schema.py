from __future__ import annotations

from dataclasses import MISSING, asdict, dataclass, field, fields
from typing import Any

from pose.common.env import capture_environment
from pose.common.errors import ProtocolError
from pose.common.timing import REQUIRED_TIMING_KEYS, empty_timings
from pose.protocol.session_ids import generate_session_id

VERDICTS = (
    "SUCCESS",
    "INNER_PROOF_INVALID",
    "OUTER_PROOF_INVALID",
    "TIMEOUT",
    "COVERAGE_BELOW_THRESHOLD",
    "RESOURCE_FAILURE",
    "CLEANUP_FAILURE",
    "PROTOCOL_ERROR",
)


@dataclass
class SessionResult:
    success: bool
    verdict: str
    session_id: str
    profile_name: str
    run_class: str = "cold"
    session_nonce: str = ""
    session_plan_root: str = ""
    session_manifest_root: str = ""
    artifact_path: str = ""
    resident_socket_path: str = ""
    resident_process_id: int = 0
    lease_expiry: str = ""
    host_total_bytes: int = 0
    host_usable_bytes: int = 0
    host_covered_bytes: int = 0
    gpu_devices: list[int] = field(default_factory=list)
    gpu_usable_bytes_by_device: dict[str, int] = field(default_factory=dict)
    gpu_covered_bytes_by_device: dict[str, int] = field(default_factory=dict)
    region_roots: dict[str, str] = field(default_factory=dict)
    region_manifest_roots: dict[str, str] = field(default_factory=dict)
    region_payload_bytes_by_region: dict[str, int] = field(default_factory=dict)
    challenge_indices_by_region: dict[str, list[int]] = field(default_factory=dict)
    real_porep_bytes: int = 0
    tail_filler_bytes: int = 0
    real_porep_ratio: float = 0.0
    coverage_fraction: float = 0.0
    inner_filecoin_verified: bool = False
    cpu_fallback_detected: bool = False
    cpu_fallback_events: list[str] = field(default_factory=list)
    outer_pose_verified: bool = False
    challenge_leaf_size: int = 4096
    challenge_policy: dict[str, int | float] = field(default_factory=dict)
    challenge_count: int = 0
    deadline_ms: int = 0
    response_ms: int = 0
    cleanup_status: str = "NOT_RUN"
    cleanup_policy: dict[str, bool] = field(default_factory=dict)
    timings_ms: dict[str, int] = field(default_factory=empty_timings)
    environment: dict[str, str] = field(default_factory=capture_environment)
    notes: list[str] = field(default_factory=list)

    def validate(self) -> None:
        if self.verdict not in VERDICTS:
            raise ProtocolError(f"Unsupported verdict: {self.verdict}")
        for key in REQUIRED_TIMING_KEYS:
            if key not in self.timings_ms:
                raise ProtocolError(f"Missing timing key: {key}")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SessionResult":
        if not isinstance(payload, dict):
            raise ProtocolError("Session result payload must decode to a mapping")

        missing = [name for name in REQUIRED_RESULT_FIELDS if name not in payload]
        if missing:
            formatted = ", ".join(sorted(missing))
            raise ProtocolError(f"Missing required result field(s): {formatted}")

        try:
            result = cls(**payload)
        except TypeError as error:
            raise ProtocolError(f"Malformed session result payload: {error}") from error
        result.validate()
        return result


REQUIRED_RESULT_FIELDS = tuple(
    item.name
    for item in fields(SessionResult)
    if item.default is MISSING and item.default_factory is MISSING
)


def bootstrap_result(profile_name: str, note: str | None = None) -> SessionResult:
    result = SessionResult(
        success=False,
        verdict="PROTOCOL_ERROR",
        session_id=generate_session_id(),
        profile_name=profile_name,
    )
    if note:
        result.notes.append(note)
    return result
