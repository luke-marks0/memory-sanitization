from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class BenchmarkProfile:
    name: str
    benchmark_class: str
    target_devices: dict[str, object]
    reserve_policy: dict[str, int]
    host_target_fraction: float
    per_gpu_target_fraction: float
    porep_unit_profile: str
    leaf_size: int
    challenge_policy: dict[str, int | float]
    deadline_policy: dict[str, int]
    cleanup_policy: dict[str, bool]
    repetition_count: int

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "BenchmarkProfile":
        return cls(**payload)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "benchmark_class": self.benchmark_class,
            "target_devices": self.target_devices,
            "reserve_policy": self.reserve_policy,
            "host_target_fraction": self.host_target_fraction,
            "per_gpu_target_fraction": self.per_gpu_target_fraction,
            "porep_unit_profile": self.porep_unit_profile,
            "leaf_size": self.leaf_size,
            "challenge_policy": self.challenge_policy,
            "deadline_policy": self.deadline_policy,
            "cleanup_policy": self.cleanup_policy,
            "repetition_count": self.repetition_count,
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
    return [
        BenchmarkProfile.from_dict(
            yaml.safe_load(path.read_text(encoding="utf-8"))
        )
        for path in sorted(root.glob("*.yaml"))
    ]

