from __future__ import annotations

from pose.benchmarks.profiles import load_profiles, required_profile_names


def test_required_profiles_load() -> None:
    profiles = load_profiles()
    assert {profile.name for profile in profiles} == set(required_profile_names())

