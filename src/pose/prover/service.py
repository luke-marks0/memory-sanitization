from __future__ import annotations

from pose.common.errors import ResourceFailure, UnsupportedPhaseError
from pose.filecoin.reference import VendoredFilecoinReference
from pose.filecoin.mirror.comms import assemble_commitment
from pose.filecoin.mirror.labels import derive_label
from pose.filecoin.mirror.parents import drg_parents, expander_parents
from pose.filecoin.mirror.replica_id import derive_replica_id


class ProverService:
    def describe(self) -> dict[str, object]:
        try:
            bridge_status = VendoredFilecoinReference().bridge_status()
        except ResourceFailure as error:
            bridge_status = {
                "status": "bridge-unavailable",
                "supports_real_filecoin_reference": False,
                "note": str(error),
            }
        return {
            "status": "phase1-host-complete",
            "supports_host_memory": True,
            "supports_gpu_hbm": False,
            "supports_real_filecoin_reference": bool(
                bridge_status.get("supports_real_filecoin_reference", False)
            ),
            "filecoin_bridge": bridge_status,
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
            "real_filecoin_bridge_available": self.describe()[
                "supports_real_filecoin_reference"
            ],
        }

    def serve(self, _config_path: str) -> None:
        raise UnsupportedPhaseError(
            "Prover serving is not implemented in the foundation phase."
        )
