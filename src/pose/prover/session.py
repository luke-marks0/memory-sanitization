from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProverSessionState:
    session_id: str
    profile_name: str
    status: str = "planned"
    region_count: int = 0

