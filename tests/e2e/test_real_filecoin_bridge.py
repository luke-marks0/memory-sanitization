from __future__ import annotations

import os
import subprocess
import sys
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
    parsed = parse_serialized_porep_unit(porep_unit.serialized_bytes)

    assert status["supports_real_filecoin_reference"] is True
    assert status["status"] == "phase0-real-filecoin-bridge"
    assert artifact.status == "phase0-real-filecoin-bridge"
    assert artifact.verified_after_seal is True
    assert artifact.sector_size == 2048
    assert artifact.piece_size > 0
    assert "seal_pre_commit_phase1" in artifact.inner_timings_ms
    assert parsed.manifest.storage_profile == "minimal"
    assert parsed.manifest.comm_r_hex == artifact.comm_r_hex
    assert parsed.blob_kinds == ("seal_proof", "public_inputs", "proof_metadata")
    assert reference.verify(artifact) is True
