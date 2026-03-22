from __future__ import annotations

import pytest

from pose.common.errors import ProtocolError
from pose.protocol.result_schema import SessionResult, bootstrap_result


def test_bootstrap_result_contains_required_fields() -> None:
    result = bootstrap_result("dev-small", note="unit-test")
    payload = result.to_dict()
    assert payload["profile_name"] == "dev-small"
    assert payload["verdict"] == "PROTOCOL_ERROR"
    assert "total" in payload["timings_ms"]
    assert payload["notes"] == ["unit-test"]


def test_from_dict_rejects_missing_required_fields() -> None:
    payload = bootstrap_result("dev-small").to_dict()
    payload.pop("profile_name")

    with pytest.raises(ProtocolError, match="profile_name"):
        SessionResult.from_dict(payload)


def test_from_dict_rejects_missing_defaulted_fields_from_canonical_artifact() -> None:
    payload = bootstrap_result("dev-small").to_dict()
    payload.pop("host_total_bytes")
    payload.pop("claim_notes")
    payload.pop("notes")

    with pytest.raises(ProtocolError, match="claim_notes|host_total_bytes|notes"):
        SessionResult.from_dict(payload)
