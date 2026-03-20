from __future__ import annotations

import pytest

from pose.common.errors import ProtocolError
from pose.protocol.result_schema import bootstrap_result


def test_invalid_verdict_is_rejected() -> None:
    result = bootstrap_result("dev-small")
    result.verdict = "INVALID"
    with pytest.raises(ProtocolError):
        result.validate()


def test_missing_timing_key_is_rejected() -> None:
    result = bootstrap_result("dev-small")
    result.timings_ms.pop("total")
    with pytest.raises(ProtocolError):
        result.validate()

