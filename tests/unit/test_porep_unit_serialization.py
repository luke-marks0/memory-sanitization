from __future__ import annotations

import json
from pathlib import Path

import pytest

from pose.common.errors import ConfigurationError, ProtocolError
from pose.filecoin.porep_unit import (
    build_porep_unit_from_seal_artifact,
    parse_serialized_porep_unit,
)
from pose.filecoin.reference import SealArtifact


FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "porep_unit_minimal_fixture.json"


def _fixture_payload() -> dict[str, object]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _sample_artifact() -> SealArtifact:
    return SealArtifact(
        status="phase0-real-filecoin-bridge",
        verified_after_seal=True,
        sector_size=2048,
        api_version="V1_2_0",
        registered_seal_proof=5,
        porep_id_hex="05" + ("00" * 31),
        prover_id_hex="07" * 32,
        sector_id=4242,
        ticket_hex="01" * 32,
        seed_hex="02" * 32,
        piece_size=2032,
        piece_commitment_hex="11" * 32,
        comm_d_hex="22" * 32,
        comm_r_hex="33" * 32,
        proof_hex="aabbccddeeff",
        inner_timings_ms={
            "seal_pre_commit_phase1": 17,
            "seal_pre_commit_phase2": 19,
            "seal_commit_phase1": 23,
            "seal_commit_phase2": 29,
            "verify_seal": 31,
        },
    )


def test_minimal_porep_unit_fixture_is_stable() -> None:
    fixture = _fixture_payload()
    artifact = SealArtifact(**fixture["artifact"])
    unit = build_porep_unit_from_seal_artifact(
        artifact,
        storage_profile=fixture["storage_profile"],
        leaf_alignment_bytes=fixture["leaf_alignment_bytes"],
        upstream_snapshot_id=fixture["upstream_snapshot_id"],
    )
    parsed = parse_serialized_porep_unit(unit.serialized_bytes)

    assert unit.serialized_bytes.hex() == fixture["serialized_hex"]
    assert parsed.manifest.upstream_snapshot_id == fixture["upstream_snapshot_id"]
    assert parsed.blob_kinds == tuple(fixture["expected_blob_kinds"])
    assert parsed.alignment_padding_bytes == fixture["alignment_padding_bytes"]
    assert parsed.manifest.payload_length_bytes == fixture["payload_length_bytes"]


def test_full_cache_profile_requires_auxiliary_cache_blob() -> None:
    with pytest.raises(ConfigurationError, match="auxiliary cache blob"):
        build_porep_unit_from_seal_artifact(
            _sample_artifact(),
            storage_profile="full-cache",
            upstream_snapshot_id="rust-fil-proofs:test",
            extra_blobs={"sealed_replica": b"replica"},
        )


def test_replica_and_full_cache_profiles_sort_blobs_canonically() -> None:
    artifact = _sample_artifact()

    replica_unit = build_porep_unit_from_seal_artifact(
        artifact,
        storage_profile="replica",
        upstream_snapshot_id="rust-fil-proofs:test",
        extra_blobs={"sealed_replica": b"replica-bytes"},
    )
    full_cache_unit = build_porep_unit_from_seal_artifact(
        artifact,
        storage_profile="full-cache",
        upstream_snapshot_id="rust-fil-proofs:test",
        extra_blobs={
            "cache_file": b"cache-file-bytes",
            "sealed_replica": b"replica-bytes",
        },
    )

    assert replica_unit.blob_kinds == (
        "seal_proof",
        "sealed_replica",
        "public_inputs",
        "proof_metadata",
    )
    assert full_cache_unit.blob_kinds == (
        "seal_proof",
        "sealed_replica",
        "cache_file",
        "public_inputs",
        "proof_metadata",
    )


def test_porep_unit_parse_rejects_payload_tampering() -> None:
    artifact = _sample_artifact()
    unit = build_porep_unit_from_seal_artifact(
        artifact,
        upstream_snapshot_id="rust-fil-proofs:test",
    )
    mutated = bytearray(unit.serialized_bytes)
    mutated[len(unit.manifest_bytes)] ^= 0x01

    with pytest.raises(ProtocolError, match="digest mismatch"):
        parse_serialized_porep_unit(bytes(mutated))
