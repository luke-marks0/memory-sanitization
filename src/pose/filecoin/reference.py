from __future__ import annotations

import importlib
import json
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Protocol

from pose.common.errors import ResourceFailure


class FilecoinReference(Protocol):
    def bridge_status(self) -> dict[str, Any]:
        ...


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def build_bridge() -> None:
    repo_root = _repo_root()
    subprocess.run(
        [sys.executable, "scripts/build_bridge.py"],
        cwd=repo_root,
        check=True,
    )


def _import_bridge_module(build_if_missing: bool = False) -> Any:
    try:
        return importlib.import_module("pose_filecoin_bridge")
    except ModuleNotFoundError as error:
        if build_if_missing:
            build_bridge()
            importlib.invalidate_caches()
            return importlib.import_module("pose_filecoin_bridge")
        raise ResourceFailure(
            "The real Filecoin bridge is not installed. Run `make build-bridge` or "
            "`python scripts/build_bridge.py` first."
        ) from error


@dataclass(slots=True)
class SealRequest:
    piece_bytes: bytes | None = None
    prover_id_hex: str | None = None
    sector_id: int | None = None
    ticket_hex: str | None = None
    seed_hex: str | None = None
    porep_id_hex: str | None = None
    verify_after_seal: bool = True

    def to_bridge_payload(self) -> dict[str, object]:
        return {
            "piece_bytes_hex": self.piece_bytes.hex() if self.piece_bytes is not None else None,
            "prover_id_hex": self.prover_id_hex,
            "sector_id": self.sector_id,
            "ticket_hex": self.ticket_hex,
            "seed_hex": self.seed_hex,
            "porep_id_hex": self.porep_id_hex,
            "verify_after_seal": self.verify_after_seal,
        }


@dataclass(slots=True)
class SealArtifact:
    status: str
    verified_after_seal: bool
    sector_size: int
    api_version: str
    registered_seal_proof: int
    porep_id_hex: str
    prover_id_hex: str
    sector_id: int
    ticket_hex: str
    seed_hex: str
    piece_size: int
    piece_commitment_hex: str
    comm_d_hex: str
    comm_r_hex: str
    proof_hex: str
    inner_timings_ms: dict[str, int]
    cpu_fallback_detected: bool = False
    cpu_fallback_events: list[str] = field(default_factory=list)
    extra_blobs_hex: dict[str, str] | None = None

    def to_bridge_payload(self) -> dict[str, object]:
        return asdict(self)


class VendoredFilecoinReference:
    def __init__(self, *, build_if_missing: bool = False) -> None:
        self._module = _import_bridge_module(build_if_missing=build_if_missing)

    def bridge_status(self) -> dict[str, Any]:
        return json.loads(self._module.bridge_status_json())

    def seal(self, request: SealRequest | None = None) -> SealArtifact:
        payload = request.to_bridge_payload() if request is not None else {}
        return SealArtifact(**json.loads(self._module.seal_json(json.dumps(payload))))

    def verify(self, artifact: SealArtifact) -> bool:
        payload = artifact.to_bridge_payload()
        result = json.loads(self._module.verify_json(json.dumps(payload)))
        return bool(result["verified"])

    def seal_porep_unit(
        self,
        request: SealRequest | None = None,
        *,
        storage_profile: str = "minimal",
        leaf_alignment_bytes: int = 4096,
        extra_blobs: dict[str, bytes] | None = None,
    ) -> Any:
        from pose.filecoin.porep_unit import build_porep_unit_from_seal_artifact

        artifact = self.seal(request=request)
        return build_porep_unit_from_seal_artifact(
            artifact,
            storage_profile=storage_profile,
            leaf_alignment_bytes=leaf_alignment_bytes,
            extra_blobs=extra_blobs,
        )


def summarize_cpu_fallbacks(artifacts: Iterable[SealArtifact]) -> tuple[bool, list[str]]:
    detected = False
    events: list[str] = []
    seen: set[str] = set()
    for artifact in artifacts:
        artifact_detected = bool(
            getattr(artifact, "cpu_fallback_detected", False)
            or getattr(artifact, "cpu_fallback_events", ())
        )
        detected = detected or artifact_detected
        for event in getattr(artifact, "cpu_fallback_events", ()):
            normalized = str(event)
            if normalized in seen:
                continue
            seen.add(normalized)
            events.append(normalized)
    if detected and not events:
        events.append("CPU fallback detected during inner proof generation.")
    return detected, events
