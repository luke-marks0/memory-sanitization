from __future__ import annotations

from pathlib import Path

from pose.benchmarks.profiles import BenchmarkProfile
from pose.protocol.codec import load_json_file
from pose.protocol.result_schema import SessionResult, bootstrap_result
from pose.verifier.host_session import run_host_session


class VerifierService:
    def describe(self) -> dict[str, object]:
        return {
            "status": "phase1-host-complete",
            "supports_host_memory": True,
            "supports_gpu_hbm": False,
            "supports_rechallenge": False,
        }

    def run_session(self, profile: BenchmarkProfile) -> SessionResult:
        if profile.target_devices.get("host", False) and not profile.target_devices.get("gpus"):
            return run_host_session(profile)
        result = bootstrap_result(
            profile_name=profile.name,
            note="Only host-only sessions are implemented at this stage.",
        )
        result.verdict = "PROTOCOL_ERROR"
        return result

    def run_placeholder(self, profile: BenchmarkProfile, note: str) -> SessionResult:
        result = bootstrap_result(profile_name=profile.name, note=note)
        result.challenge_leaf_size = profile.leaf_size
        result.challenge_count = profile.challenge_policy["max_challenges"]
        result.deadline_ms = profile.deadline_policy["response_deadline_ms"]
        return result

    def verify_record(self, path: Path) -> SessionResult:
        payload = load_json_file(path)
        return SessionResult.from_dict(payload)
