from __future__ import annotations

import hashlib
from pathlib import Path

from pose.common.upstream import (
    compute_file_digest,
    extract_missing_parameter_filenames,
    load_upstream_artifacts,
    minimal_upstream_test_artifacts,
)


def test_minimal_upstream_artifacts_cover_2kib_suite_and_srs() -> None:
    repo_root = Path(__file__).resolve().parents[2]

    artifact_names = {artifact.name for artifact in minimal_upstream_test_artifacts(repo_root)}

    assert len(artifact_names) == 9
    assert "v28-fil-inner-product-v1.srs" in artifact_names
    assert all(
        "-fb9e095bebdd77511c0269b967b4d87ba8b8a525edaa0e165de23ba454510194." in name
        or "-0170db1f394b35d995252228ee359194b13199d259380541dc529fb0099096b0." in name
        or "-3ea05428c9d11689f23529cde32fd30aabd50f7d2c93657c1d3650bca3e8ea9e." in name
        or "-032d3138d22506ec0082ed72b2dcba18df18477904e35bafee82b3793b06832f." in name
        or name == "v28-fil-inner-product-v1.srs"
        for name in artifact_names
    )


def test_extract_missing_parameter_filenames_preserves_order_and_deduplicates() -> None:
    output = """
thread 'bench aggregation' panicked at 'open /var/tmp/filecoin-proof-parameters/v28-fil-inner-product-v1.srs: No such file or directory'
thread 'api test' panicked at 'open /var/tmp/filecoin-proof-parameters/v28-proof-of-spacetime-fallback-merkletree-poseidon_hasher-8-0-0-0170db1f394b35d995252228ee359194b13199d259380541dc529fb0099096b0.params: No such file or directory'
thread 'api test' panicked at 'open /var/tmp/filecoin-proof-parameters/v28-fil-inner-product-v1.srs: No such file or directory'
"""

    assert extract_missing_parameter_filenames(output) == (
        "v28-fil-inner-product-v1.srs",
        "v28-proof-of-spacetime-fallback-merkletree-poseidon_hasher-8-0-0-0170db1f394b35d995252228ee359194b13199d259380541dc529fb0099096b0.params",
    )


def test_upstream_artifact_catalog_contains_minimal_artifacts() -> None:
    repo_root = Path(__file__).resolve().parents[2]

    catalog = load_upstream_artifacts(repo_root)
    minimal_names = {artifact.name for artifact in minimal_upstream_test_artifacts(repo_root)}

    assert minimal_names.issubset(catalog)


def test_compute_file_digest_matches_upstream_blake2b_truncation(tmp_path: Path) -> None:
    payload = b"real-filecoin-proof-artifact"
    path = tmp_path / "artifact.bin"
    path.write_bytes(payload)

    expected = hashlib.blake2b(payload).hexdigest()[:32]

    assert compute_file_digest(path) == expected
