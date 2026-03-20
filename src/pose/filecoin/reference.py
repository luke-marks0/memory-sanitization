from __future__ import annotations

from typing import Protocol

from pose.common.errors import UnsupportedPhaseError


class FilecoinReference(Protocol):
    def bridge_status(self) -> str:
        ...


class UnavailableFilecoinReference:
    def bridge_status(self) -> str:
        return "unavailable"

    def seal_and_verify(self) -> None:
        raise UnsupportedPhaseError(
            "The real Filecoin reference bridge is not implemented in the foundation phase."
        )

