from __future__ import annotations

import json
import mmap
import os
import socket
from datetime import UTC, datetime
from time import perf_counter
from typing import Any

from pose.common.errors import ProtocolError, ResourceFailure
from pose.common.merkle import commit_payload
from pose.filecoin.porep_unit import build_porep_unit_from_seal_artifact
from pose.filecoin.reference import VendoredFilecoinReference
from pose.protocol.host_worker_protocol import WORKER_PROTOCOL_VERSION
from pose.protocol.messages import SectorPlanEntry
from pose.protocol.region_payloads import build_region_manifest, build_region_payload

ZERO_CHUNK = b"\x00" * (1024 * 1024)


def _materialize_locally(
    *,
    reference: VendoredFilecoinReference,
    storage_profile: str,
    leaf_size: int,
    unit_count: int,
    sector_plan: list[SectorPlanEntry],
) -> tuple[list[Any], list[Any], bytes, int]:
    artifacts = []
    units = []
    payload_parts: list[bytes] = []
    object_serialization_ms = 0
    ordered_sector_plan = sorted(sector_plan, key=lambda item: item.unit_index)
    if len(ordered_sector_plan) != unit_count:
        raise ProtocolError(
            f"Expected {unit_count} sector-plan entries, got {len(ordered_sector_plan)}"
        )
    for entry in ordered_sector_plan:
        artifact = reference.seal(entry.to_seal_request())
        serialization_started = perf_counter()
        unit = build_porep_unit_from_seal_artifact(
            artifact,
            storage_profile=storage_profile,
            leaf_alignment_bytes=leaf_size,
        )
        object_serialization_ms += int((perf_counter() - serialization_started) * 1000)
        if len(unit.serialized_bytes) != leaf_size:
            raise ResourceFailure(
                "Current host worker expects each minimal PoRep unit to occupy exactly "
                f"one leaf: got {len(unit.serialized_bytes)} bytes for leaf size {leaf_size}"
            )
        artifacts.append(artifact)
        units.append(unit)
        payload_parts.append(unit.serialized_bytes)
    return artifacts, units, b"".join(payload_parts), object_serialization_ms


def run_materialize_operation(request: dict[str, object]) -> dict[str, object]:
    _validate_protocol_version(request)
    lease_fd = int(request["lease_fd"])
    usable_bytes = int(request["usable_bytes"])
    leaf_size = int(request["leaf_size"])
    unit_count = int(request["unit_count"])
    session_nonce = str(request["session_nonce"])
    session_plan_root = str(request["session_plan_root"])
    region_id = str(request["region_id"])
    storage_profile = str(request["storage_profile"])
    sector_plan = [
        SectorPlanEntry.from_dict(dict(item))
        for item in request.get("sector_plan", [])
    ]

    if unit_count <= 0:
        raise ProtocolError(f"unit_count must be positive, got {unit_count}")
    if len(sector_plan) != unit_count:
        raise ProtocolError(
            f"materialize expects {unit_count} sector-plan entries, got {len(sector_plan)}"
        )
    if any(item.region_id != region_id for item in sector_plan):
        raise ProtocolError("materialize sector-plan entries must all target the requested region")

    reference = VendoredFilecoinReference()
    materialize_started = perf_counter()
    artifacts, units, real_payload, object_serialization_ms = _materialize_locally(
        reference=reference,
        storage_profile=storage_profile,
        leaf_size=leaf_size,
        unit_count=unit_count,
        sector_plan=sector_plan,
    )
    tail_filler_bytes = usable_bytes - len(real_payload)
    if tail_filler_bytes < 0:
        raise ResourceFailure(
            f"Materialized payload length {len(real_payload)} exceeds lease size {usable_bytes}"
        )
    if tail_filler_bytes > min(leaf_size, 1024 * 1024):
        raise ResourceFailure(
            f"Tail filler requirement {tail_filler_bytes} exceeds the allowed limit "
            f"for leaf size {leaf_size}"
        )
    payload = build_region_payload(
        [unit.serialized_bytes for unit in units],
        session_nonce=session_nonce,
        region_id=region_id,
        session_plan_root=session_plan_root,
        tail_filler_bytes=tail_filler_bytes,
    )

    copy_started = perf_counter()
    mapping = mmap.mmap(lease_fd, usable_bytes)
    try:
        mapping.seek(0)
        mapping.write(payload)
        mapping.flush()
    finally:
        mapping.close()
    copy_to_host_ms = int((perf_counter() - copy_started) * 1000)

    outer_tree_started = perf_counter()
    commitment = commit_payload(payload, leaf_size)
    outer_tree_build_ms = int((perf_counter() - outer_tree_started) * 1000)
    region_manifest = build_region_manifest(
        region_id=region_id,
        region_type="host",
        usable_bytes=usable_bytes,
        leaf_size=leaf_size,
        payload=payload,
        merkle_root_hex=commitment.root_hex,
        units=tuple(units),
        tail_filler_bytes=tail_filler_bytes,
    )
    return {
        "protocol_version": WORKER_PROTOCOL_VERSION,
        "operation": "materialize",
        "region_id": region_id,
        "unit_count": unit_count,
        "region_root_hex": region_manifest.merkle_root_hex,
        "region_manifest": {
            **region_manifest.to_cbor_object(),
            "manifest_root_hex": region_manifest.manifest_root_hex,
        },
        "timings_ms": {
            "copy_to_host": copy_to_host_ms,
            "materialize_total": int((perf_counter() - materialize_started) * 1000),
            "object_serialization": object_serialization_ms,
            "outer_tree_build": outer_tree_build_ms,
        },
        "unit_artifacts": [artifact.to_bridge_payload() for artifact in artifacts],
    }


def run_open_operation(request: dict[str, object]) -> dict[str, object]:
    _validate_protocol_version(request)
    lease_fd = int(request["lease_fd"])
    usable_bytes = int(request["usable_bytes"])
    leaf_size = int(request["leaf_size"])
    region_id = str(request["region_id"])
    session_manifest_root = str(request["session_manifest_root"])
    challenge_indices = [int(index) for index in request["challenge_indices"]]  # type: ignore[index]

    mapping = mmap.mmap(lease_fd, usable_bytes, access=mmap.ACCESS_READ)
    try:
        mapping.seek(0)
        payload = mapping.read(usable_bytes)
    finally:
        mapping.close()

    commitment = commit_payload(payload, leaf_size)
    started = perf_counter()
    openings = []
    for index in challenge_indices:
        start = index * leaf_size
        leaf = payload[start : start + leaf_size]
        opening = commitment.opening(index, leaf)
        openings.append(
            {
                "region_id": region_id,
                "session_manifest_root": session_manifest_root,
                "leaf_index": index,
                "leaf_hex": leaf.hex(),
                "sibling_hashes_hex": [value.hex() for value in opening.sibling_hashes],
            }
        )
    response_ms = int((perf_counter() - started) * 1000)
    return {
        "protocol_version": WORKER_PROTOCOL_VERSION,
        "operation": "open",
        "region_id": region_id,
        "region_root_hex": commitment.root_hex,
        "response_ms": response_ms,
        "openings": openings,
    }


def _zeroize_mapping(mapping: mmap.mmap, usable_bytes: int) -> None:
    remaining = usable_bytes
    cursor = 0
    while remaining:
        chunk = ZERO_CHUNK[: min(len(ZERO_CHUNK), remaining)]
        mapping[cursor : cursor + len(chunk)] = chunk
        cursor += len(chunk)
        remaining -= len(chunk)
    mapping.flush()


def _verify_zeroized(mapping: mmap.mmap, usable_bytes: int) -> bool:
    mapping.seek(0)
    return mapping.read(usable_bytes) == bytes(usable_bytes)


def _cleanup_mapping(
    *,
    mapping: mmap.mmap,
    usable_bytes: int,
    zeroize: bool,
    verify_zeroization: bool,
) -> str:
    if zeroize:
        _zeroize_mapping(mapping, usable_bytes)
        if verify_zeroization and not _verify_zeroized(mapping, usable_bytes):
            raise ResourceFailure("Host lease zeroization verification failed")
        if verify_zeroization:
            return "ZEROIZED_AND_VERIFIED"
        return "ZEROIZED_AND_RELEASED"
    return "RELEASED_WITHOUT_ZEROIZE"


def _serve_resident_requests(
    *,
    mapping: mmap.mmap,
    lease_fd: int,
    socket_path: str,
    usable_bytes: int,
    leaf_size: int,
    region_id: str,
    lease_expiry: str,
    cleanup_policy: dict[str, bool],
) -> None:
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    if os.path.exists(socket_path):
        os.unlink(socket_path)
    server.bind(socket_path)
    server.listen(1)
    server.settimeout(0.5)

    try:
        while True:
            expiry = datetime.fromisoformat(lease_expiry)
            if expiry <= datetime.now(UTC):
                _cleanup_mapping(
                    mapping=mapping,
                    usable_bytes=usable_bytes,
                    zeroize=bool(cleanup_policy["zeroize"]),
                    verify_zeroization=bool(cleanup_policy["verify_zeroization"]),
                )
                return

            try:
                connection, _ = server.accept()
            except TimeoutError:
                continue

            with connection:
                request_bytes = bytearray()
                while True:
                    chunk = connection.recv(65536)
                    if not chunk:
                        break
                    request_bytes.extend(chunk)
                try:
                    request = json.loads(request_bytes.decode("utf-8"))
                except json.JSONDecodeError as error:
                    response = {"error": f"Malformed resident request JSON: {error}"}
                else:
                    operation = str(request.get("operation", ""))
                    if operation == "open":
                        response = run_open_operation(
                            {
                                "protocol_version": WORKER_PROTOCOL_VERSION,
                                "operation": "open",
                                "lease_fd": lease_fd,
                                "usable_bytes": usable_bytes,
                                "leaf_size": leaf_size,
                                "region_id": request.get("region_id", region_id),
                                "session_manifest_root": request["session_manifest_root"],
                                "challenge_indices": request["challenge_indices"],
                            }
                        )
                    elif operation == "cleanup":
                        cleanup_status = _cleanup_mapping(
                            mapping=mapping,
                            usable_bytes=usable_bytes,
                            zeroize=bool(cleanup_policy["zeroize"]),
                            verify_zeroization=bool(cleanup_policy["verify_zeroization"]),
                        )
                        response = {"cleanup_status": cleanup_status, "operation": "cleanup"}
                        connection.sendall(json.dumps(response, sort_keys=True).encode("utf-8"))
                        return
                    else:
                        response = {"error": f"Unsupported resident worker operation: {operation!r}"}
                connection.sendall(json.dumps(response, sort_keys=True).encode("utf-8"))
    finally:
        server.close()
        if os.path.exists(socket_path):
            os.unlink(socket_path)
        mapping.close()
        os.close(lease_fd)


def run_resident_materialize(request: dict[str, object]) -> dict[str, object]:
    _validate_protocol_version(request)
    lease_fd = int(request["lease_fd"])
    usable_bytes = int(request["usable_bytes"])
    leaf_size = int(request["leaf_size"])
    unit_count = int(request["unit_count"])
    session_nonce = str(request["session_nonce"])
    session_plan_root = str(request["session_plan_root"])
    region_id = str(request["region_id"])
    storage_profile = str(request["storage_profile"])
    socket_path = str(request["socket_path"])
    lease_expiry = str(request["lease_expiry"])
    sector_plan = [
        SectorPlanEntry.from_dict(dict(item))
        for item in request.get("sector_plan", [])
    ]
    cleanup_policy = {
        key: bool(value) for key, value in dict(request["cleanup_policy"]).items()
    }

    if unit_count <= 0:
        raise ProtocolError(f"unit_count must be positive, got {unit_count}")
    if len(sector_plan) != unit_count:
        raise ProtocolError(
            f"resident materialize expects {unit_count} sector-plan entries, got {len(sector_plan)}"
        )
    if any(item.region_id != region_id for item in sector_plan):
        raise ProtocolError(
            "resident materialize sector-plan entries must all target the requested region"
        )

    reference = VendoredFilecoinReference()
    materialize_started = perf_counter()
    artifacts, units, real_payload, object_serialization_ms = _materialize_locally(
        reference=reference,
        storage_profile=storage_profile,
        leaf_size=leaf_size,
        unit_count=unit_count,
        sector_plan=sector_plan,
    )
    tail_filler_bytes = usable_bytes - len(real_payload)
    if tail_filler_bytes < 0:
        raise ResourceFailure(
            f"Materialized payload length {len(real_payload)} exceeds lease size {usable_bytes}"
        )
    if tail_filler_bytes > min(leaf_size, 1024 * 1024):
        raise ResourceFailure(
            f"Tail filler requirement {tail_filler_bytes} exceeds the allowed limit "
            f"for leaf size {leaf_size}"
        )
    payload = build_region_payload(
        [unit.serialized_bytes for unit in units],
        session_nonce=session_nonce,
        region_id=region_id,
        session_plan_root=session_plan_root,
        tail_filler_bytes=tail_filler_bytes,
    )

    copy_started = perf_counter()
    mapping = mmap.mmap(lease_fd, usable_bytes)
    mapping.seek(0)
    mapping.write(payload)
    mapping.flush()
    copy_to_host_ms = int((perf_counter() - copy_started) * 1000)

    outer_tree_started = perf_counter()
    commitment = commit_payload(payload, leaf_size)
    outer_tree_build_ms = int((perf_counter() - outer_tree_started) * 1000)
    region_manifest = build_region_manifest(
        region_id=region_id,
        region_type="host",
        usable_bytes=usable_bytes,
        leaf_size=leaf_size,
        payload=payload,
        merkle_root_hex=commitment.root_hex,
        units=tuple(units),
        tail_filler_bytes=tail_filler_bytes,
    )
    startup_response = {
        "protocol_version": WORKER_PROTOCOL_VERSION,
        "operation": "resident-materialize",
        "region_id": region_id,
        "region_root_hex": region_manifest.merkle_root_hex,
        "region_manifest": {
            **region_manifest.to_cbor_object(),
            "manifest_root_hex": region_manifest.manifest_root_hex,
        },
        "socket_path": socket_path,
        "lease_expiry": lease_expiry,
        "timings_ms": {
            "copy_to_host": copy_to_host_ms,
            "materialize_total": int((perf_counter() - materialize_started) * 1000),
            "object_serialization": object_serialization_ms,
            "outer_tree_build": outer_tree_build_ms,
        },
        "unit_artifacts": [artifact.to_bridge_payload() for artifact in artifacts],
    }
    print(json.dumps(startup_response, sort_keys=True), flush=True)
    _serve_resident_requests(
        mapping=mapping,
        lease_fd=lease_fd,
        socket_path=socket_path,
        usable_bytes=usable_bytes,
        leaf_size=leaf_size,
        region_id=region_id,
        lease_expiry=lease_expiry,
        cleanup_policy=cleanup_policy,
    )
    return startup_response


def handle_request_json(request_json: str) -> str:
    try:
        request = json.loads(request_json)
    except json.JSONDecodeError as error:
        raise ProtocolError(f"Malformed host worker request JSON: {error}") from error
    if not isinstance(request, dict):
        raise ProtocolError("Host worker request must decode to a JSON object")

    operation = str(request.get("operation", ""))
    if operation == "materialize":
        response = run_materialize_operation(request)
    elif operation == "open":
        response = run_open_operation(request)
    else:
        raise ProtocolError(f"Unsupported host worker operation: {operation!r}")
    return json.dumps(response, sort_keys=True)


def handle_resident_session_json(request_json: str) -> None:
    try:
        request = json.loads(request_json)
    except json.JSONDecodeError as error:
        raise ProtocolError(f"Malformed host worker request JSON: {error}") from error
    if not isinstance(request, dict):
        raise ProtocolError("Host worker request must decode to a JSON object")
    run_resident_materialize(request)


def _validate_protocol_version(request: dict[str, object]) -> None:
    protocol_version = str(request.get("protocol_version", ""))
    if protocol_version != WORKER_PROTOCOL_VERSION:
        raise ProtocolError(
            f"Unsupported host worker protocol version: {protocol_version!r}"
        )
