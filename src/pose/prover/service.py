from __future__ import annotations

from pose.common.errors import UnsupportedPhaseError
from pose.filecoin.mirror.comms import assemble_commitment
from pose.filecoin.mirror.labels import derive_label
from pose.filecoin.mirror.parents import drg_parents, expander_parents
from pose.filecoin.mirror.replica_id import derive_replica_id


class ProverService:
    def describe(self) -> dict[str, object]:
        return {
            "status": "foundation-scaffold",
            "supports_host_memory": False,
            "supports_gpu_hbm": False,
            "supports_real_filecoin_reference": False,
        }

    def self_test(self) -> dict[str, object]:
        replica_id = derive_replica_id(
            prover_id=b"foundation-prover",
            sector_id=1,
            ticket=b"ticket",
            porep_id=b"porep",
        )
        parents = drg_parents(7, 6, 256) + expander_parents(7, 8, 256)
        label = derive_label(replica_id, layer=1, node=7, parents=parents)
        return {
            "status": "ok",
            "replica_id": replica_id,
            "label": label,
            "commitment": assemble_commitment(
                [replica_id.encode("ascii"), label.encode("ascii")]
            ),
        }

    def serve(self, _config_path: str) -> None:
        raise UnsupportedPhaseError(
            "Prover serving is not implemented in the foundation phase."
        )

