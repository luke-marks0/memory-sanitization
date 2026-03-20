from __future__ import annotations

from pose.protocol.result_schema import bootstrap_result


def test_bootstrap_result_contains_required_fields() -> None:
    result = bootstrap_result("dev-small", note="unit-test")
    payload = result.to_dict()
    assert payload["profile_name"] == "dev-small"
    assert payload["verdict"] == "PROTOCOL_ERROR"
    assert "total" in payload["timings_ms"]
    assert payload["notes"] == ["unit-test"]

