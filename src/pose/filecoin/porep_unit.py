from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Mapping

import cbor2

from pose.common.errors import ConfigurationError, ProtocolError
from pose.common.hashing import sha256_hex
from pose.common.upstream import load_upstream_lock, upstream_lock_path

if TYPE_CHECKING:
    from pose.filecoin.reference import SealArtifact


POREP_UNIT_FORMAT_NAME = "pose-filecoin-porep-unit"
POREP_UNIT_FORMAT_VERSION = 1
POREP_UNIT_PROTOCOL_VERSION = "pose-filecoin-porep-unit/v1"
POREP_UNIT_BLOB_ORDER_VERSION = 1
POREP_UNIT_ALIGNMENT_SCHEME = "zero-pad/v1"
DEFAULT_LEAF_ALIGNMENT_BYTES = 4096

StorageProfile = Literal["minimal", "replica", "full-cache"]

ALLOWED_BLOB_KINDS = (
    "seal_proof",
    "sealed_replica",
    "tree_c",
    "tree_r_last",
    "persistent_aux",
    "temporary_aux",
    "labels",
    "cache_file",
    "public_inputs",
    "proof_metadata",
)

BLOB_KIND_ORDER = {kind: index for index, kind in enumerate(ALLOWED_BLOB_KINDS)}
BLOB_KIND_ENCODINGS = {
    "seal_proof": "raw",
    "sealed_replica": "raw",
    "tree_c": "raw",
    "tree_r_last": "raw",
    "persistent_aux": "raw",
    "temporary_aux": "raw",
    "labels": "raw",
    "cache_file": "raw",
    "public_inputs": "cbor",
    "proof_metadata": "cbor",
}
PROFILE_REQUIRED_BLOB_KINDS: dict[str, frozenset[str]] = {
    "minimal": frozenset(("seal_proof", "public_inputs", "proof_metadata")),
    "replica": frozenset(("seal_proof", "public_inputs", "proof_metadata", "sealed_replica")),
    "full-cache": frozenset(("seal_proof", "public_inputs", "proof_metadata", "sealed_replica")),
}
FULL_CACHE_AUXILIARY_BLOB_KINDS = frozenset(
    ("tree_c", "tree_r_last", "persistent_aux", "temporary_aux", "labels", "cache_file")
)
INNER_PROOF_TIMING_KEYS = (
    "seal_pre_commit_phase1",
    "seal_pre_commit_phase2",
    "seal_commit_phase1",
    "seal_commit_phase2",
    "verify_seal",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def canonical_cbor_dumps(value: Any) -> bytes:
    return cbor2.dumps(value, canonical=True)


def default_upstream_snapshot_id(repo_root: Path | None = None) -> str:
    root = repo_root if repo_root is not None else _repo_root()
    metadata = load_upstream_lock(upstream_lock_path(root))
    upstream_commit = str(metadata["upstream_commit"])
    return f"rust-fil-proofs:{upstream_commit}"


def proof_config_identifier(
    registered_seal_proof: int,
    api_version: str,
    sector_size: int,
) -> str:
    api_token = api_version.lower().replace(".", "_")
    return (
        f"registered-seal-proof:{registered_seal_proof}"
        f"/api:{api_token}"
        f"/sector-size:{sector_size}"
    )


def _sorted_unique_blob_kinds(kinds: set[str]) -> tuple[str, ...]:
    return tuple(sorted(kinds, key=lambda kind: BLOB_KIND_ORDER[kind]))


def _ensure_storage_profile(storage_profile: str) -> None:
    if storage_profile not in PROFILE_REQUIRED_BLOB_KINDS:
        raise ConfigurationError(f"Unsupported PoRep unit storage profile: {storage_profile}")


def _ensure_blob_kind(kind: str) -> None:
    if kind not in BLOB_KIND_ORDER:
        raise ProtocolError(f"Unsupported PoRep blob kind: {kind}")


def _normalize_inner_timings(inner_timings_ms: Mapping[str, int] | None) -> dict[str, int]:
    source = dict(inner_timings_ms or {})
    normalized: dict[str, int] = {}

    for key in INNER_PROOF_TIMING_KEYS:
        value = int(source.pop(key, 0))
        if value < 0:
            raise ProtocolError(f"Inner proof timing must be non-negative: {key}={value}")
        normalized[key] = value

    for key in sorted(source):
        value = int(source[key])
        if value < 0:
            raise ProtocolError(f"Inner proof timing must be non-negative: {key}={value}")
        normalized[key] = value

    return normalized


def _build_public_inputs_payload(artifact: SealArtifact) -> bytes:
    return canonical_cbor_dumps(
        {
            "api_version": artifact.api_version,
            "comm_d_hex": artifact.comm_d_hex,
            "comm_r_hex": artifact.comm_r_hex,
            "piece_infos": [
                {
                    "commitment_hex": artifact.piece_commitment_hex,
                    "size": artifact.piece_size,
                }
            ],
            "porep_id_hex": artifact.porep_id_hex,
            "prover_id_hex": artifact.prover_id_hex,
            "registered_seal_proof": artifact.registered_seal_proof,
            "sector_id": artifact.sector_id,
            "sector_size": artifact.sector_size,
            "seed_hex": artifact.seed_hex,
            "ticket_hex": artifact.ticket_hex,
        }
    )


def _build_proof_metadata_payload(
    artifact: SealArtifact,
    proof_bytes: bytes,
    inner_timings_ms: Mapping[str, int],
) -> bytes:
    return canonical_cbor_dumps(
        {
            "bridge_status": artifact.status,
            "inner_timings_ms": dict(inner_timings_ms),
            "proof_sha256_hex": sha256_hex(proof_bytes),
            "proof_size_bytes": len(proof_bytes),
            "verified_after_seal": artifact.verified_after_seal,
        }
    )


def _normalize_extra_blobs(extra_blobs: Mapping[str, bytes] | None) -> tuple["PoRepUnitBlob", ...]:
    if not extra_blobs:
        return ()

    normalized: list[PoRepUnitBlob] = []
    for kind, payload in extra_blobs.items():
        _ensure_blob_kind(kind)
        if kind in ("seal_proof", "public_inputs", "proof_metadata"):
            raise ConfigurationError(
                f"Blob kind {kind!r} is reserved for the canonical serializer and "
                "must not be overridden through extra_blobs."
            )
        normalized.append(PoRepUnitBlob(kind=kind, payload=bytes(payload)))
    return tuple(normalized)


def _validate_blob_profile(storage_profile: str, blobs: tuple["PoRepUnitBlob", ...]) -> None:
    _ensure_storage_profile(storage_profile)
    blob_kinds = {blob.kind for blob in blobs}
    required = PROFILE_REQUIRED_BLOB_KINDS[storage_profile]
    missing = required.difference(blob_kinds)
    if missing:
        formatted = ", ".join(_sorted_unique_blob_kinds(set(missing)))
        raise ConfigurationError(
            f"PoRep unit profile {storage_profile!r} is missing required blob kinds: {formatted}"
        )

    if storage_profile == "full-cache" and blob_kinds.isdisjoint(FULL_CACHE_AUXILIARY_BLOB_KINDS):
        raise ConfigurationError(
            "PoRep unit profile 'full-cache' requires at least one auxiliary cache blob."
        )


@dataclass(frozen=True)
class PieceInfoRecord:
    size: int
    commitment_hex: str

    def to_cbor_object(self) -> dict[str, Any]:
        return {
            "commitment_hex": self.commitment_hex,
            "size": self.size,
        }

    @classmethod
    def from_cbor_object(cls, payload: Mapping[str, Any]) -> "PieceInfoRecord":
        return cls(
            size=int(payload["size"]),
            commitment_hex=str(payload["commitment_hex"]),
        )


@dataclass(frozen=True)
class PoRepUnitBlob:
    kind: str
    payload: bytes

    def __post_init__(self) -> None:
        _ensure_blob_kind(self.kind)

    @property
    def encoding(self) -> str:
        return BLOB_KIND_ENCODINGS[self.kind]

    @property
    def length(self) -> int:
        return len(self.payload)

    @property
    def digest_hex(self) -> str:
        return sha256_hex(self.payload)


@dataclass(frozen=True)
class BlobManifestEntry:
    kind: str
    offset: int
    length: int
    sha256_hex: str
    encoding: str

    def __post_init__(self) -> None:
        _ensure_blob_kind(self.kind)
        if self.encoding != BLOB_KIND_ENCODINGS[self.kind]:
            raise ProtocolError(
                f"Unexpected encoding for blob kind {self.kind!r}: {self.encoding!r}"
            )

    def to_cbor_object(self) -> dict[str, Any]:
        return {
            "encoding": self.encoding,
            "kind": self.kind,
            "length": self.length,
            "offset": self.offset,
            "sha256_hex": self.sha256_hex,
        }

    @classmethod
    def from_cbor_object(cls, payload: Mapping[str, Any]) -> "BlobManifestEntry":
        return cls(
            kind=str(payload["kind"]),
            offset=int(payload["offset"]),
            length=int(payload["length"]),
            sha256_hex=str(payload["sha256_hex"]),
            encoding=str(payload["encoding"]),
        )


@dataclass(frozen=True)
class PoRepUnitManifest:
    upstream_snapshot_id: str
    proof_config_id: str
    storage_profile: str
    sector_size: int
    registered_seal_proof: int
    api_version: str
    porep_id_hex: str
    prover_id_hex: str
    sector_id: int
    ticket_hex: str
    seed_hex: str
    piece_infos: tuple[PieceInfoRecord, ...]
    comm_d_hex: str
    comm_r_hex: str
    inner_timings_ms: dict[str, int]
    leaf_alignment_bytes: int
    payload_length_bytes: int
    blobs: tuple[BlobManifestEntry, ...]

    def to_cbor_object(self) -> dict[str, Any]:
        return {
            "api_version": self.api_version,
            "blob_order_version": POREP_UNIT_BLOB_ORDER_VERSION,
            "blobs": [blob.to_cbor_object() for blob in self.blobs],
            "comm_d_hex": self.comm_d_hex,
            "comm_r_hex": self.comm_r_hex,
            "format_name": POREP_UNIT_FORMAT_NAME,
            "format_version": POREP_UNIT_FORMAT_VERSION,
            "inner_timings_ms": dict(self.inner_timings_ms),
            "leaf_alignment_bytes": self.leaf_alignment_bytes,
            "padding_scheme": POREP_UNIT_ALIGNMENT_SCHEME,
            "payload_length_bytes": self.payload_length_bytes,
            "piece_infos": [piece.to_cbor_object() for piece in self.piece_infos],
            "porep_id_hex": self.porep_id_hex,
            "proof_config_id": self.proof_config_id,
            "protocol_version": POREP_UNIT_PROTOCOL_VERSION,
            "prover_id_hex": self.prover_id_hex,
            "registered_seal_proof": self.registered_seal_proof,
            "sector_id": self.sector_id,
            "sector_size": self.sector_size,
            "seed_hex": self.seed_hex,
            "storage_profile": self.storage_profile,
            "ticket_hex": self.ticket_hex,
            "upstream_snapshot_id": self.upstream_snapshot_id,
        }

    @classmethod
    def from_cbor_object(cls, payload: Mapping[str, Any]) -> "PoRepUnitManifest":
        if str(payload["format_name"]) != POREP_UNIT_FORMAT_NAME:
            raise ProtocolError(f"Unsupported PoRep unit format name: {payload['format_name']!r}")
        if int(payload["format_version"]) != POREP_UNIT_FORMAT_VERSION:
            raise ProtocolError(
                f"Unsupported PoRep unit format version: {payload['format_version']!r}"
            )
        if str(payload["protocol_version"]) != POREP_UNIT_PROTOCOL_VERSION:
            raise ProtocolError(
                f"Unsupported PoRep unit protocol version: {payload['protocol_version']!r}"
            )
        if int(payload["blob_order_version"]) != POREP_UNIT_BLOB_ORDER_VERSION:
            raise ProtocolError(
                f"Unsupported PoRep unit blob order version: {payload['blob_order_version']!r}"
            )
        if str(payload["padding_scheme"]) != POREP_UNIT_ALIGNMENT_SCHEME:
            raise ProtocolError(f"Unsupported padding scheme: {payload['padding_scheme']!r}")

        manifest = cls(
            upstream_snapshot_id=str(payload["upstream_snapshot_id"]),
            proof_config_id=str(payload["proof_config_id"]),
            storage_profile=str(payload["storage_profile"]),
            sector_size=int(payload["sector_size"]),
            registered_seal_proof=int(payload["registered_seal_proof"]),
            api_version=str(payload["api_version"]),
            porep_id_hex=str(payload["porep_id_hex"]),
            prover_id_hex=str(payload["prover_id_hex"]),
            sector_id=int(payload["sector_id"]),
            ticket_hex=str(payload["ticket_hex"]),
            seed_hex=str(payload["seed_hex"]),
            piece_infos=tuple(
                PieceInfoRecord.from_cbor_object(item) for item in payload["piece_infos"]
            ),
            comm_d_hex=str(payload["comm_d_hex"]),
            comm_r_hex=str(payload["comm_r_hex"]),
            inner_timings_ms=_normalize_inner_timings(payload["inner_timings_ms"]),
            leaf_alignment_bytes=int(payload["leaf_alignment_bytes"]),
            payload_length_bytes=int(payload["payload_length_bytes"]),
            blobs=tuple(
                BlobManifestEntry.from_cbor_object(item) for item in payload["blobs"]
            ),
        )
        _validate_blob_profile(manifest.storage_profile, _manifest_entries_as_blobs(manifest.blobs))
        if manifest.leaf_alignment_bytes <= 0:
            raise ProtocolError("leaf_alignment_bytes must be positive")
        return manifest


@dataclass(frozen=True)
class SerializedPoRepUnit:
    manifest: PoRepUnitManifest
    blobs: tuple[PoRepUnitBlob, ...]
    manifest_bytes: bytes
    serialized_bytes: bytes
    alignment_padding_bytes: int
    consumed_bytes: int

    @property
    def payload_length_bytes(self) -> int:
        return self.manifest.payload_length_bytes

    @property
    def blob_kinds(self) -> tuple[str, ...]:
        return tuple(blob.kind for blob in self.blobs)


def _manifest_entries_as_blobs(entries: tuple[BlobManifestEntry, ...]) -> tuple[PoRepUnitBlob, ...]:
    return tuple(PoRepUnitBlob(kind=entry.kind, payload=b"") for entry in entries)


def _sort_blobs(blobs: tuple[PoRepUnitBlob, ...]) -> tuple[PoRepUnitBlob, ...]:
    return tuple(sorted(blobs, key=lambda blob: BLOB_KIND_ORDER[blob.kind]))


def _blob_manifest_entries(blobs: tuple[PoRepUnitBlob, ...]) -> tuple[BlobManifestEntry, ...]:
    entries: list[BlobManifestEntry] = []
    offset = 0
    seen: set[str] = set()
    for blob in blobs:
        if blob.kind in seen:
            raise ProtocolError(f"Duplicate PoRep blob kind in unit serialization: {blob.kind}")
        seen.add(blob.kind)
        entries.append(
            BlobManifestEntry(
                kind=blob.kind,
                offset=offset,
                length=blob.length,
                sha256_hex=blob.digest_hex,
                encoding=blob.encoding,
            )
        )
        offset += blob.length
    return tuple(entries)


def build_porep_unit_from_seal_artifact(
    artifact: SealArtifact,
    *,
    storage_profile: StorageProfile = "minimal",
    leaf_alignment_bytes: int = DEFAULT_LEAF_ALIGNMENT_BYTES,
    upstream_snapshot_id: str | None = None,
    inner_timings_ms: Mapping[str, int] | None = None,
    extra_blobs: Mapping[str, bytes] | None = None,
) -> SerializedPoRepUnit:
    if leaf_alignment_bytes <= 0:
        raise ConfigurationError("leaf_alignment_bytes must be positive")

    proof_bytes = bytes.fromhex(artifact.proof_hex)
    normalized_timings = _normalize_inner_timings(
        inner_timings_ms
        if inner_timings_ms is not None
        else getattr(artifact, "inner_timings_ms", None)
    )

    blobs = _sort_blobs(
        (
            PoRepUnitBlob(kind="seal_proof", payload=proof_bytes),
            PoRepUnitBlob(
                kind="public_inputs",
                payload=_build_public_inputs_payload(artifact),
            ),
            PoRepUnitBlob(
                kind="proof_metadata",
                payload=_build_proof_metadata_payload(
                    artifact=artifact,
                    proof_bytes=proof_bytes,
                    inner_timings_ms=normalized_timings,
                ),
            ),
            *_normalize_extra_blobs(extra_blobs),
        )
    )
    _validate_blob_profile(storage_profile, blobs)

    manifest = PoRepUnitManifest(
        upstream_snapshot_id=upstream_snapshot_id or default_upstream_snapshot_id(),
        proof_config_id=proof_config_identifier(
            registered_seal_proof=artifact.registered_seal_proof,
            api_version=artifact.api_version,
            sector_size=artifact.sector_size,
        ),
        storage_profile=storage_profile,
        sector_size=artifact.sector_size,
        registered_seal_proof=artifact.registered_seal_proof,
        api_version=artifact.api_version,
        porep_id_hex=artifact.porep_id_hex,
        prover_id_hex=artifact.prover_id_hex,
        sector_id=artifact.sector_id,
        ticket_hex=artifact.ticket_hex,
        seed_hex=artifact.seed_hex,
        piece_infos=(
            PieceInfoRecord(
                size=artifact.piece_size,
                commitment_hex=artifact.piece_commitment_hex,
            ),
        ),
        comm_d_hex=artifact.comm_d_hex,
        comm_r_hex=artifact.comm_r_hex,
        inner_timings_ms=normalized_timings,
        leaf_alignment_bytes=leaf_alignment_bytes,
        payload_length_bytes=sum(blob.length for blob in blobs),
        blobs=_blob_manifest_entries(blobs),
    )
    manifest_bytes = canonical_cbor_dumps(manifest.to_cbor_object())
    payload_bytes = b"".join(blob.payload for blob in blobs)
    alignment_padding_bytes = (
        -(len(manifest_bytes) + manifest.payload_length_bytes)
    ) % leaf_alignment_bytes
    serialized_bytes = manifest_bytes + payload_bytes + (b"\x00" * alignment_padding_bytes)
    return SerializedPoRepUnit(
        manifest=manifest,
        blobs=blobs,
        manifest_bytes=manifest_bytes,
        serialized_bytes=serialized_bytes,
        alignment_padding_bytes=alignment_padding_bytes,
        consumed_bytes=len(serialized_bytes),
    )


def parse_serialized_porep_unit(
    serialized: bytes,
    *,
    require_exact: bool = True,
) -> SerializedPoRepUnit:
    stream = BytesIO(serialized)
    payload = cbor2.load(stream)
    if not isinstance(payload, dict):
        raise ProtocolError("PoRep unit manifest must decode to a CBOR map")

    manifest = PoRepUnitManifest.from_cbor_object(payload)
    manifest_bytes = serialized[: stream.tell()]
    expected_padding_bytes = (
        -(len(manifest_bytes) + manifest.payload_length_bytes)
    ) % manifest.leaf_alignment_bytes
    total_length = len(manifest_bytes) + manifest.payload_length_bytes + expected_padding_bytes

    if len(serialized) < total_length:
        raise ProtocolError(
            "Serialized PoRep unit is truncated: "
            f"expected at least {total_length} bytes, got {len(serialized)}"
        )
    if require_exact and len(serialized) != total_length:
        raise ProtocolError(
            "Serialized PoRep unit length mismatch: "
            f"expected {total_length} bytes, got {len(serialized)}"
        )

    payload_bytes = serialized[len(manifest_bytes) : len(manifest_bytes) + manifest.payload_length_bytes]
    padding_bytes = serialized[
        len(manifest_bytes) + manifest.payload_length_bytes : total_length
    ]
    if any(byte != 0 for byte in padding_bytes):
        raise ProtocolError("PoRep unit alignment padding must contain only zero bytes")

    blobs: list[PoRepUnitBlob] = []
    expected_offset = 0
    previous_order = -1
    for entry in manifest.blobs:
        order = BLOB_KIND_ORDER[entry.kind]
        if order < previous_order:
            raise ProtocolError("PoRep unit blob manifest is not in canonical order")
        previous_order = order
        if entry.offset != expected_offset:
            raise ProtocolError(
                f"PoRep unit blob offsets are not contiguous at {entry.kind!r}: "
                f"expected {expected_offset}, got {entry.offset}"
            )
        start = entry.offset
        stop = entry.offset + entry.length
        blob_payload = payload_bytes[start:stop]
        if len(blob_payload) != entry.length:
            raise ProtocolError(f"PoRep unit blob {entry.kind!r} is truncated")
        if sha256_hex(blob_payload) != entry.sha256_hex:
            raise ProtocolError(f"PoRep unit blob {entry.kind!r} digest mismatch")
        blobs.append(PoRepUnitBlob(kind=entry.kind, payload=blob_payload))
        expected_offset = stop

    if expected_offset != manifest.payload_length_bytes:
        raise ProtocolError(
            "PoRep unit blob table does not cover the full payload length: "
            f"{expected_offset} != {manifest.payload_length_bytes}"
        )

    return SerializedPoRepUnit(
        manifest=manifest,
        blobs=tuple(blobs),
        manifest_bytes=manifest_bytes,
        serialized_bytes=serialized[:total_length],
        alignment_padding_bytes=expected_padding_bytes,
        consumed_bytes=total_length,
    )
