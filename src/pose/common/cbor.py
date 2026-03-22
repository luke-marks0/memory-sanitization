from __future__ import annotations

from typing import Any

import cbor2


def canonical_cbor_dumps(value: Any) -> bytes:
    return cbor2.dumps(value, canonical=True)
