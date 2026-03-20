from __future__ import annotations

from pose.filecoin.mirror.comms import assemble_commitment
from pose.filecoin.mirror.labels import derive_label
from pose.filecoin.mirror.parents import drg_parents, expander_parents
from pose.filecoin.mirror.replica_id import derive_replica_id


def test_deterministic_mirror_helpers_are_stable() -> None:
    replica_id = derive_replica_id(b"prover", 9, b"ticket", b"porep")
    parents = drg_parents(9, 6, 256) + expander_parents(9, 8, 256)
    label = derive_label(replica_id, 2, 9, parents)
    commitment = assemble_commitment([replica_id.encode("ascii"), label.encode("ascii")])

    assert replica_id == derive_replica_id(b"prover", 9, b"ticket", b"porep")
    assert parents == drg_parents(9, 6, 256) + expander_parents(9, 8, 256)
    assert len(label) == 64
    assert len(commitment) == 64

