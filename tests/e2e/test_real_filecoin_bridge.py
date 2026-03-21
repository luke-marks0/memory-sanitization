from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from pose.filecoin.porep_unit import (
    build_porep_unit_from_seal_artifact,
    parse_serialized_porep_unit,
)
from pose.filecoin.reference import VendoredFilecoinReference


pytestmark = pytest.mark.skipif(
    os.environ.get("POSE_RUN_REAL_BRIDGE_TESTS") != "1",
    reason="real vendored Filecoin bridge test runs only in the dedicated bridge target",
)


def test_real_vendored_filecoin_bridge_seals_and_verifies() -> None:
    root = Path(__file__).resolve().parents[2]
    subprocess.run([sys.executable, "scripts/build_bridge.py"], cwd=root, check=True)

    reference = VendoredFilecoinReference()
    status = reference.bridge_status()
    artifact = reference.seal()
    porep_unit = build_porep_unit_from_seal_artifact(artifact)
    full_cache_unit = build_porep_unit_from_seal_artifact(
        artifact,
        storage_profile="full-cache",
        leaf_alignment_bytes=1024 * 1024,
    )
    parsed = parse_serialized_porep_unit(porep_unit.serialized_bytes)
    full_cache_parsed = parse_serialized_porep_unit(full_cache_unit.serialized_bytes)

    assert status["supports_real_filecoin_reference"] is True
    assert status["status"] == "phase0-real-filecoin-bridge"
    assert artifact.status == "phase0-real-filecoin-bridge"
    assert artifact.verified_after_seal is True
    assert artifact.sector_size == 2048
    assert artifact.piece_size > 0
    assert "seal_pre_commit_phase1" in artifact.inner_timings_ms
    assert "sealed_replica" in (artifact.extra_blobs_hex or {})
    assert parsed.manifest.storage_profile == "minimal"
    assert parsed.manifest.comm_r_hex == artifact.comm_r_hex
    assert parsed.blob_kinds == ("seal_proof", "public_inputs", "proof_metadata")
    assert "sealed_replica" in full_cache_parsed.blob_kinds
    assert any(
        kind in full_cache_parsed.blob_kinds
        for kind in ("tree_c", "tree_r_last", "persistent_aux", "temporary_aux", "labels", "cache_file")
    )
    assert reference.verify(artifact) is True


def _flip_hex_byte(value: str) -> str:
    first_byte = int(value[:2], 16) ^ 0xFF
    return f"{first_byte:02x}{value[2:]}"


def _verification_fails(reference: VendoredFilecoinReference, artifact) -> bool:
    try:
        return reference.verify(artifact) is False
    except RuntimeError:
        return True


def test_real_vendored_filecoin_bridge_rejects_tampered_inner_proof_inputs() -> None:
    root = Path(__file__).resolve().parents[2]
    subprocess.run([sys.executable, "scripts/build_bridge.py"], cwd=root, check=True)

    reference = VendoredFilecoinReference()
    artifact = reference.seal()
    tampered_cases = {
        "comm_d": replace(artifact, comm_d_hex=_flip_hex_byte(artifact.comm_d_hex)),
        "comm_r": replace(artifact, comm_r_hex=_flip_hex_byte(artifact.comm_r_hex)),
        "prover_id": replace(artifact, prover_id_hex=_flip_hex_byte(artifact.prover_id_hex)),
        "sector_id": replace(artifact, sector_id=artifact.sector_id + 1),
        "ticket": replace(artifact, ticket_hex=_flip_hex_byte(artifact.ticket_hex)),
        "proof": replace(artifact, proof_hex=_flip_hex_byte(artifact.proof_hex)),
    }

    for label, tampered in tampered_cases.items():
        assert _verification_fails(reference, tampered), label
