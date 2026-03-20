from __future__ import annotations

import hashlib
import json
import os
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

from pose.common.errors import ProtocolError

OFFICIAL_UPSTREAM_URL = "https://github.com/filecoin-project/rust-fil-proofs.git"
PROOFS_FILECOIN_IO_BASE_URL = "https://proofs.filecoin.io"
FIL_PROOFS_PARAMETER_CACHE_ENV = "FIL_PROOFS_PARAMETER_CACHE"
DEFAULT_PARAMETER_CACHE_DIR = Path("/var/tmp/filecoin-proof-parameters")
UPSTREAM_TEST_SECTOR_SIZE = 2048


@dataclass(frozen=True)
class ProofArtifact:
    name: str
    cid: str
    digest: str
    sector_size: int


def vendor_root(repo_root: Path) -> Path:
    return repo_root / "vendor" / "rust-fil-proofs"


def upstream_lock_path(repo_root: Path) -> Path:
    return repo_root / "vendor" / "UPSTREAM.lock"


def parameter_manifest_path(repo_root: Path) -> Path:
    return vendor_root(repo_root) / "filecoin-proofs" / "parameters.json"


def srs_manifest_path(repo_root: Path) -> Path:
    return vendor_root(repo_root) / "srs-inner-product.json"


def upstream_toolchain_path(repo_root: Path) -> Path:
    return vendor_root(repo_root) / "rust-toolchain"


def load_upstream_lock(path: Path) -> dict[str, object]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def read_upstream_toolchain(repo_root: Path) -> str:
    return upstream_toolchain_path(repo_root).read_text(encoding="utf-8").strip()


def parameter_cache_dir() -> Path:
    override = os.environ.get(FIL_PROOFS_PARAMETER_CACHE_ENV)
    return Path(override) if override else DEFAULT_PARAMETER_CACHE_DIR


def load_artifact_manifest(path: Path) -> dict[str, ProofArtifact]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {
        name: ProofArtifact(
            name=name,
            cid=str(metadata["cid"]),
            digest=str(metadata["digest"]),
            sector_size=int(metadata["sector_size"]),
        )
        for name, metadata in raw.items()
    }


def load_upstream_artifacts(repo_root: Path) -> dict[str, ProofArtifact]:
    artifacts = load_artifact_manifest(parameter_manifest_path(repo_root))
    artifacts.update(load_artifact_manifest(srs_manifest_path(repo_root)))
    return artifacts


def minimal_upstream_test_artifacts(repo_root: Path) -> tuple[ProofArtifact, ...]:
    parameter_artifacts = load_artifact_manifest(parameter_manifest_path(repo_root))
    srs_artifacts = load_artifact_manifest(srs_manifest_path(repo_root))
    selected_parameters = sorted(
        (
            artifact
            for artifact in parameter_artifacts.values()
            if artifact.sector_size == UPSTREAM_TEST_SECTOR_SIZE
        ),
        key=lambda artifact: artifact.name,
    )
    return tuple(
        [*selected_parameters, *sorted(srs_artifacts.values(), key=lambda artifact: artifact.name)]
    )


def iter_vendor_files(path: Path) -> list[Path]:
    excluded_roots = {".git", "target", "__pycache__"}
    return sorted(
        file_path
        for file_path in path.rglob("*")
        if file_path.is_file() and excluded_roots.isdisjoint(file_path.parts)
    )


def compute_tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for file_path in iter_vendor_files(path):
        relative_path = file_path.relative_to(path).as_posix().encode("utf-8")
        file_digest = hashlib.sha256(file_path.read_bytes()).hexdigest().encode("ascii")
        digest.update(relative_path)
        digest.update(b"\0")
        digest.update(file_digest)
        digest.update(b"\n")
    return digest.hexdigest()


def compute_file_digest(path: Path) -> str:
    digest = hashlib.blake2b()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()[:32]


def artifact_download_url(name: str) -> str:
    return f"{PROOFS_FILECOIN_IO_BASE_URL}/{name}"


def extract_missing_parameter_filenames(output: str) -> tuple[str, ...]:
    seen: list[str] = []
    for filename in re.findall(r"filecoin-proof-parameters/([A-Za-z0-9._-]+)", output):
        if filename not in seen:
            seen.append(filename)
    return tuple(seen)


def required_vendor_files(path: Path) -> tuple[Path, ...]:
    return (
        path / "Cargo.toml",
        path / "Cargo.lock",
        path / "README.md",
        path / "LICENSE-APACHE",
        path / "LICENSE-MIT",
        path / "COPYRIGHT",
        path / "filecoin-proofs" / "Cargo.toml",
        path / "storage-proofs-porep" / "Cargo.toml",
    )


def validate_upstream_snapshot(repo_root: Path) -> None:
    metadata = load_upstream_lock(upstream_lock_path(repo_root))
    required_keys = {
        "upstream_url",
        "upstream_default_branch",
        "upstream_commit",
        "upstream_tag",
        "component_tags",
        "source_commit_date",
        "sync_date",
        "local_patch_status",
        "tree_sha256",
    }
    missing = required_keys.difference(metadata)
    if missing:
        raise ProtocolError(f"Missing UPSTREAM.lock keys: {sorted(missing)}")

    upstream_url = str(metadata["upstream_url"])
    if upstream_url != OFFICIAL_UPSTREAM_URL:
        raise ProtocolError(f"Unexpected upstream URL: {upstream_url}")

    upstream_commit = str(metadata["upstream_commit"])
    if not re.fullmatch(r"[0-9a-f]{40}", upstream_commit):
        raise ProtocolError(f"Invalid upstream commit SHA: {upstream_commit}")

    local_patch_status = str(metadata["local_patch_status"])
    if local_patch_status != "clean":
        raise ProtocolError(
            "Vendored snapshot is not marked clean. Explicit patch acknowledgment "
            "has not been implemented yet."
        )

    component_tags = metadata["component_tags"]
    if not isinstance(component_tags, list) or not component_tags:
        raise ProtocolError("UPSTREAM.lock must record the component release tags.")

    vendored_tree = vendor_root(repo_root)
    if not vendored_tree.exists():
        raise ProtocolError(f"Vendored tree is missing: {vendored_tree}")

    for required_file in required_vendor_files(vendored_tree):
        if not required_file.exists():
            raise ProtocolError(f"Vendored tree is missing required file: {required_file}")

    expected_hash = str(metadata["tree_sha256"])
    actual_hash = compute_tree_sha256(vendored_tree)
    if actual_hash != expected_hash:
        raise ProtocolError(
            "Vendored tree hash mismatch: "
            f"expected {expected_hash}, got {actual_hash}"
        )

    for manifest_path in (
        parameter_manifest_path(repo_root),
        srs_manifest_path(repo_root),
        upstream_toolchain_path(repo_root),
    ):
        if not manifest_path.exists():
            raise ProtocolError(f"Vendored tree is missing required upstream metadata: {manifest_path}")
