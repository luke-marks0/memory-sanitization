from __future__ import annotations

import json

from pose.benchmarks.calibration import prepare_calibration


def render_calibration_payload(profile_identifier: str) -> str:
    return json.dumps(prepare_calibration(profile_identifier), indent=2, sort_keys=True)
