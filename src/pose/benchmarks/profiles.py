from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from pose.common.errors import ProtocolError
from pose.common.sandbox import ProverSandboxPolicy
from pose.graphs import GRAPH_FAMILY, validate_label_width_bits
from pose.hashing import DEFAULT_HASH_BACKEND, normalize_hash_backend
from pose.verifier.soundness import normalize_adversary_model


@dataclass(frozen=True)
class ProfileChallengePolicy:
    rounds_r: int = 0
    target_success_bound: float = 0.0
    sample_with_replacement: bool = True

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "ProfileChallengePolicy":
        return cls(
            rounds_r=int(payload.get("rounds_r", 0)),
            target_success_bound=float(payload.get("target_success_bound", 0.0)),
            sample_with_replacement=bool(payload.get("sample_with_replacement", True)),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "rounds_r": self.rounds_r,
            "sample_with_replacement": self.sample_with_replacement,
            "target_success_bound": self.target_success_bound,
        }


@dataclass(frozen=True)
class ProfileDeadlinePolicy:
    response_deadline_us: int
    session_timeout_ms: int

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "ProfileDeadlinePolicy":
        return cls(
            response_deadline_us=int(payload["response_deadline_us"]),
            session_timeout_ms=int(payload["session_timeout_ms"]),
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "response_deadline_us": self.response_deadline_us,
            "session_timeout_ms": self.session_timeout_ms,
        }


@dataclass(frozen=True)
class ProfileCalibrationPolicy:
    lookup_samples: int = 512
    hash_measurement_rounds: int = 3
    hashes_per_round: int = 4096
    transport_overhead_us: int = 0
    serialization_overhead_us: int = 0
    safety_margin_fraction: float = 0.25

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "ProfileCalibrationPolicy":
        return cls(
            lookup_samples=int(payload.get("lookup_samples", 512)),
            hash_measurement_rounds=int(payload.get("hash_measurement_rounds", 3)),
            hashes_per_round=int(payload.get("hashes_per_round", 4096)),
            transport_overhead_us=int(payload.get("transport_overhead_us", 0)),
            serialization_overhead_us=int(payload.get("serialization_overhead_us", 0)),
            safety_margin_fraction=float(payload.get("safety_margin_fraction", 0.25)),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "hash_measurement_rounds": self.hash_measurement_rounds,
            "hashes_per_round": self.hashes_per_round,
            "lookup_samples": self.lookup_samples,
            "safety_margin_fraction": self.safety_margin_fraction,
            "serialization_overhead_us": self.serialization_overhead_us,
            "transport_overhead_us": self.transport_overhead_us,
        }


@dataclass(frozen=True)
class BenchmarkProfile:
    name: str
    benchmark_class: str
    target_devices: dict[str, object]
    reserve_policy: dict[str, int]
    host_target_fraction: float
    per_gpu_target_fraction: float
    w_bits: int
    graph_family: str
    hash_backend: str
    adversary_model: str
    attacker_budget_bytes_assumed: int
    challenge_policy: ProfileChallengePolicy
    deadline_policy: ProfileDeadlinePolicy
    calibration_policy: ProfileCalibrationPolicy
    cleanup_policy: dict[str, bool]
    repetition_count: int
    transport_mode: str = "grpc"
    coverage_threshold: float = 0.0
    prover_sandbox: ProverSandboxPolicy = field(default_factory=ProverSandboxPolicy)

    @property
    def w_bytes(self) -> int:
        return self.w_bits // 8

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "BenchmarkProfile":
        w_bits = validate_label_width_bits(int(payload["w_bits"]))
        graph_family = str(payload.get("graph_family", GRAPH_FAMILY))
        if graph_family != GRAPH_FAMILY:
            raise ProtocolError(f"Unsupported graph family in benchmark profile: {graph_family!r}")
        return cls(
            name=str(payload["name"]),
            benchmark_class=str(payload.get("benchmark_class", "cold")),
            target_devices=dict(payload["target_devices"]),
            reserve_policy={key: int(value) for key, value in dict(payload["reserve_policy"]).items()},
            host_target_fraction=float(payload["host_target_fraction"]),
            per_gpu_target_fraction=float(payload["per_gpu_target_fraction"]),
            w_bits=w_bits,
            graph_family=graph_family,
            hash_backend=normalize_hash_backend(str(payload.get("hash_backend", DEFAULT_HASH_BACKEND))),
            adversary_model=normalize_adversary_model(str(payload.get("adversary_model", "general"))),
            attacker_budget_bytes_assumed=int(payload["attacker_budget_bytes_assumed"]),
            challenge_policy=ProfileChallengePolicy.from_dict(dict(payload["challenge_policy"])),
            deadline_policy=ProfileDeadlinePolicy.from_dict(dict(payload["deadline_policy"])),
            calibration_policy=ProfileCalibrationPolicy.from_dict(dict(payload["calibration_policy"])),
            cleanup_policy={key: bool(value) for key, value in dict(payload["cleanup_policy"]).items()},
            repetition_count=max(1, int(payload.get("repetition_count", 1))),
            transport_mode=str(payload.get("transport_mode", "grpc")),
            coverage_threshold=float(payload.get("coverage_threshold", 0.0)),
            prover_sandbox=ProverSandboxPolicy.from_dict(payload.get("prover_sandbox")),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "adversary_model": self.adversary_model,
            "attacker_budget_bytes_assumed": self.attacker_budget_bytes_assumed,
            "benchmark_class": self.benchmark_class,
            "calibration_policy": self.calibration_policy.to_dict(),
            "challenge_policy": self.challenge_policy.to_dict(),
            "cleanup_policy": dict(self.cleanup_policy),
            "coverage_threshold": self.coverage_threshold,
            "deadline_policy": self.deadline_policy.to_dict(),
            "graph_family": self.graph_family,
            "hash_backend": self.hash_backend,
            "host_target_fraction": self.host_target_fraction,
            "name": self.name,
            "per_gpu_target_fraction": self.per_gpu_target_fraction,
            "repetition_count": self.repetition_count,
            "reserve_policy": dict(self.reserve_policy),
            "target_devices": dict(self.target_devices),
            "transport_mode": self.transport_mode,
            "w_bits": self.w_bits,
            "prover_sandbox": self.prover_sandbox.to_dict(),
        }


def profiles_root() -> Path:
    return Path(__file__).resolve().parents[3] / "bench_profiles"


def required_profile_names() -> tuple[str, ...]:
    return (
        "dev-small",
        "single-h100-host-max",
        "single-h100-hbm-max",
        "single-h100-hybrid-max",
        "eight-h100-hbm-max",
        "eight-h100-hybrid-max",
    )


def resolve_profile_path(identifier: str) -> Path:
    candidate = Path(identifier)
    if candidate.exists():
        return candidate
    root = profiles_root()
    named = root / f"{identifier}.yaml"
    if named.exists():
        return named
    raise FileNotFoundError(f"Unknown benchmark profile: {identifier}")


def load_profile(identifier: str) -> BenchmarkProfile:
    path = resolve_profile_path(identifier)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return BenchmarkProfile.from_dict(payload)


def load_profiles(directory: str | Path | None = None) -> list[BenchmarkProfile]:
    root = Path(directory) if directory is not None else profiles_root()
    profiles = [
        BenchmarkProfile.from_dict(
            yaml.safe_load(path.read_text(encoding="utf-8"))
        )
        for path in sorted(root.glob("*.yaml"))
    ]
    required = set(required_profile_names())
    return [profile for profile in profiles if profile.name in required]
