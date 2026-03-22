from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any

from pose.common.env import capture_environment
from pose.common.errors import ProtocolError
from pose.common.timing import REQUIRED_TIMING_KEYS, empty_timings
from pose.hashing import DEFAULT_HASH_BACKEND
from pose.protocol.session_ids import generate_session_id

VERDICTS = (
    "SUCCESS",
    "WRONG_RESPONSE",
    "DEADLINE_MISS",
    "CALIBRATION_INVALID",
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
    graph_family: str
    graph_parameter_n: int
    graph_descriptor_digest: str
    label_width_bits: int
    label_count_m: int
    gamma: int
    hash_backend: str
    run_class: str = "cold"
    session_seed_commitment: str = ""
    artifact_path: str = ""
    resident_socket_path: str = ""
    resident_process_id: int = 0
    lease_expiry: str = ""
    adversary_model: str = "general"
    attacker_budget_bytes_assumed: int = 0
    target_success_bound: float = 0.0
    reported_success_bound: float = 0.0
    soundness_model: str = ""
    deadline_us: int = 0
    q_bound: int = 0
    rounds_r: int = 0
    accepted_rounds: int = 0
    host_total_bytes: int = 0
    host_usable_bytes: int = 0
    host_covered_bytes: int = 0
    gpu_devices: list[int] = field(default_factory=list)
    gpu_usable_bytes_by_device: dict[str, int] = field(default_factory=dict)
    gpu_covered_bytes_by_device: dict[str, int] = field(default_factory=dict)
    covered_bytes: int = 0
    slack_bytes: int = 0
    coverage_fraction: float = 0.0
    scratch_peak_bytes: int = 0
    declared_stage_copy_bytes: int = 0
    round_trip_p50_us: int = 0
    round_trip_p95_us: int = 0
    round_trip_p99_us: int = 0
    max_round_trip_us: int = 0
    cleanup_status: str = "NOT_RUN"
    formal_claim_notes: list[str] = field(default_factory=list)
    operational_claim_notes: list[str] = field(default_factory=list)
    claim_notes: list[str] = field(default_factory=list)
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
    item.name for item in fields(SessionResult)
)


def bootstrap_result(profile_name: str, note: str | None = None) -> SessionResult:
    result = SessionResult(
        success=False,
        verdict="PROTOCOL_ERROR",
        session_id=generate_session_id(),
        profile_name=profile_name,
        graph_family="pose-db-drg-v1",
        graph_parameter_n=0,
        graph_descriptor_digest="",
        label_width_bits=256,
        label_count_m=0,
        gamma=0,
        hash_backend=DEFAULT_HASH_BACKEND,
        soundness_model="random-oracle + distant-attacker + calibrated q<gamma",
    )
    if note:
        result.notes.append(note)
    return result
