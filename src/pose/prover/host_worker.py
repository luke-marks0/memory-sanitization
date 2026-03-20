from __future__ import annotations

import json
import mmap
import os
from time import perf_counter
from typing import Any

from pose.common.errors import ProtocolError, ResourceFailure
from pose.common.hashing import sha256_bytes
from pose.common.merkle import commit_payload
from pose.filecoin.porep_unit import build_porep_unit_from_seal_artifact
from pose.filecoin.reference import SealRequest, VendoredFilecoinReference
from pose.protocol.region_payloads import build_region_manifest
from pose.prover.object_builder import build_region_payload

WORKER_PROTOCOL_VERSION = "pose-host-worker/v1"
BASE_SECTOR_ID = 4242


def _derive_hex(*parts: object) -> str:
    seed = "|".join(str(part) for part in parts).encode("utf-8")
    output = bytearray()
    counter = 0
    while len(output) < 32:
        output.extend(sha256_bytes(seed + counter.to_bytes(4, "big")))
        counter += 1
    return bytes(output[:32]).hex()


def _seal_request_for_unit(session_id: str, region_id: str, unit_index: int) -> SealRequest:
    return SealRequest(
        prover_id_hex=_derive_hex("prover", session_id, region_id),
        sector_id=BASE_SECTOR_ID + unit_index,
        ticket_hex=_derive_hex("ticket", session_id, region_id, unit_index),
        seed_hex=_derive_hex("seed", session_id, region_id, unit_index),
    )


def _materialize_locally(
    *,
    reference: VendoredFilecoinReference,
    session_id: str,
    region_id: str,
    storage_profile: str,
    leaf_size: int,
    unit_count: int,
) -> tuple[list[Any], list[Any], bytes, int]:
    artifacts = []
    units = []
    payload_parts: list[bytes] = []
    object_serialization_ms = 0
    for unit_index in range(unit_count):
        artifact = reference.seal(_seal_request_for_unit(session_id, region_id, unit_index))
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
    session_id = str(request["session_id"])
    session_nonce = str(request["session_nonce"])
    session_plan_root = str(request["session_plan_root"])
    region_id = str(request["region_id"])
    storage_profile = str(request["storage_profile"])

    if unit_count <= 0:
        raise ProtocolError(f"unit_count must be positive, got {unit_count}")

    reference = VendoredFilecoinReference()
    materialize_started = perf_counter()
    artifacts, units, real_payload, object_serialization_ms = _materialize_locally(
        reference=reference,
        session_id=session_id,
        region_id=region_id,
        storage_profile=storage_profile,
        leaf_size=leaf_size,
        unit_count=unit_count,
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

    commitment = commit_payload(payload, leaf_size)
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


def _validate_protocol_version(request: dict[str, object]) -> None:
    protocol_version = str(request.get("protocol_version", ""))
    if protocol_version != WORKER_PROTOCOL_VERSION:
        raise ProtocolError(
            f"Unsupported host worker protocol version: {protocol_version!r}"
        )
