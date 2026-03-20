from __future__ import annotations

import json

import pytest

from pose.common.errors import ProtocolError
from pose.prover.host_worker import handle_request_json


def test_host_worker_rejects_unknown_operation() -> None:
    payload = {
        "protocol_version": "pose-host-worker/v1",
        "operation": "unknown",
    }

    with pytest.raises(ProtocolError, match="Unsupported host worker operation"):
        handle_request_json(json.dumps(payload))


def test_host_worker_rejects_malformed_json() -> None:
    with pytest.raises(ProtocolError, match="Malformed host worker request JSON"):
        handle_request_json("{")


def test_host_worker_rejects_protocol_version_mismatch() -> None:
    payload = {
        "protocol_version": "pose-host-worker/v0",
        "operation": "open",
        "lease_fd": 1,
        "usable_bytes": 4096,
        "leaf_size": 4096,
        "region_id": "host-0",
        "session_manifest_root": "root",
        "challenge_indices": [0],
    }

    with pytest.raises(ProtocolError, match="Unsupported host worker protocol version"):
        handle_request_json(json.dumps(payload))
