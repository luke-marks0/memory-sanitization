from __future__ import annotations

from pathlib import Path

from pose.benchmarks.profiles import load_profile, load_profiles
from pose.verifier.service import VerifierService


def prepare_run(profile_identifier: str) -> dict[str, object]:
    profile = load_profile(profile_identifier)
    executable = bool(profile.target_devices.get("host", False)) and not bool(
        profile.target_devices.get("gpus")
    )
    return {
        "status": "host-session-ready" if executable else "profile-not-yet-executable",
        "profile": profile.to_dict(),
        "note": (
            "Host-only profiles execute the current local host-memory session path."
            if executable
            else "This profile requires later HBM/hybrid work."
        ),
    }


def prepare_matrix(profiles_directory: str | Path) -> dict[str, object]:
    return {
        "status": "foundation-scaffold",
        "profiles": [profile.to_dict() for profile in load_profiles(profiles_directory)],
    }


def placeholder_result(profile_identifier: str) -> dict[str, object]:
    profile = load_profile(profile_identifier)
    verifier = VerifierService()
    result = verifier.run_session(profile)
    return result.to_dict()
