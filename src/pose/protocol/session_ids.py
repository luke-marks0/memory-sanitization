from __future__ import annotations

import secrets
from datetime import UTC, datetime


def generate_session_id(now: datetime | None = None) -> str:
    timestamp = (now or datetime.now(UTC)).strftime("%Y-%m-%dT%H:%M:%SZ")
    return f"{timestamp}-{secrets.token_hex(4)}"

