from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

from pose.common.errors import ProtocolError, ResourceFailure


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _send_socket_request(socket_path: str, payload: dict[str, object]) -> dict[str, object]:
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        client.connect(socket_path)
        client.sendall(json.dumps(payload, sort_keys=True).encode("utf-8"))
        client.shutdown(socket.SHUT_WR)
        response = bytearray()
        while True:
            chunk = client.recv(65536)
            if not chunk:
                break
            response.extend(chunk)
    finally:
        client.close()

    try:
        decoded = json.loads(response.decode("utf-8"))
    except json.JSONDecodeError as error:
        raise ProtocolError(f"Resident worker returned invalid JSON: {error}") from error
    if not isinstance(decoded, dict):
        raise ProtocolError("Resident worker response must decode to a JSON object")
    return decoded


def start_resident_worker(
    *,
    startup_request: dict[str, object],
    lease_fd: int,
) -> tuple[subprocess.Popen[str], dict[str, object]]:
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        [sys.executable, "-u", "-m", "pose.cli.main", "prover", "host-session-resident"],
        cwd=_repo_root(),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        pass_fds=(lease_fd,),
        start_new_session=True,
    )
    assert process.stdin is not None
    process.stdin.write(json.dumps(startup_request, sort_keys=True))
    process.stdin.close()

    assert process.stdout is not None
    response_line = process.stdout.readline()
    if not response_line:
        stderr = ""
        if process.stderr is not None:
            stderr = process.stderr.read().strip()
        raise ResourceFailure(
            "Resident host worker did not return a startup response"
            + (f": {stderr}" if stderr else "")
        )
    try:
        response = json.loads(response_line)
    except json.JSONDecodeError as error:
        raise ProtocolError(f"Resident host worker startup response was invalid JSON: {error}") from error
    if not isinstance(response, dict):
        raise ProtocolError("Resident host worker startup response must decode to a JSON object")
    return process, response


def open_resident_session(
    *,
    socket_path: str,
    region_id: str,
    session_manifest_root: str,
    challenge_indices: list[int],
) -> tuple[list[dict[str, object]], int]:
    response = _send_socket_request(
        socket_path,
        {
            "operation": "open",
            "region_id": region_id,
            "session_manifest_root": session_manifest_root,
            "challenge_indices": challenge_indices,
        },
    )
    return (
        list(response["openings"]),  # type: ignore[arg-type]
        int(response["response_ms"]),
    )


def cleanup_resident_session(
    *,
    socket_path: str,
) -> str:
    response = _send_socket_request(socket_path, {"operation": "cleanup"})
    return str(response["cleanup_status"])
