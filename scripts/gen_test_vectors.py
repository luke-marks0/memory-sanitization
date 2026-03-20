#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

from pose.filecoin.mirror.comms import assemble_commitment
from pose.filecoin.mirror.labels import derive_label
from pose.filecoin.mirror.parents import drg_parents, expander_parents
from pose.filecoin.mirror.replica_id import derive_replica_id


def build_vectors() -> dict[str, object]:
    replica_id = derive_replica_id(
        prover_id=b"foundation-prover",
        sector_id=42,
        ticket=b"ticket-bytes",
        porep_id=b"porep-id",
    )
    drg = drg_parents(node=7, degree=6, nodes=256)
    expander = expander_parents(node=7, degree=8, nodes=256)
    label = derive_label(replica_id, layer=2, node=7, parents=drg + expander)
    commitment = assemble_commitment(
        [replica_id.encode("ascii"), label.encode("ascii")]
    )
    return {
        "replica_id": replica_id,
        "drg_parents": drg,
        "expander_parents": expander,
        "label": label,
        "commitment": commitment,
    }


def main() -> int:
    destination = Path("tests/parity/generated_vectors.json")
    destination.write_text(json.dumps(build_vectors(), indent=2) + "\n", encoding="utf-8")
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

