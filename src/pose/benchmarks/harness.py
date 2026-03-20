from __future__ import annotations

from pathlib import Path

from pose.benchmarks.profiles import load_profile, load_profiles
from pose.verifier.service import VerifierService


def prepare_run(profile_identifier: str) -> dict[str, object]:
    profile = load_profile(profile_identifier)
    return {
        "status": "foundation-scaffold",
        "profile": profile.to_dict(),
        "note": "Benchmark execution is not implemented yet.",
    }


def prepare_matrix(profiles_directory: str | Path) -> dict[str, object]:
    return {
        "status": "foundation-scaffold",
        "profiles": [profile.to_dict() for profile in load_profiles(profiles_directory)],
    }


def placeholder_result(profile_identifier: str) -> dict[str, object]:
    profile = load_profile(profile_identifier)
    verifier = VerifierService()
    result = verifier.run_placeholder(
        profile,
        note="Verifier benchmark execution is not implemented in the foundation phase.",
    )
    return result.to_dict()

