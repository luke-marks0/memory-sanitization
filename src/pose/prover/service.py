from __future__ import annotations

import tomllib
from pathlib import Path

from pose.common.errors import UnsupportedPhaseError
from pose.graphs import build_pose_db_graph, compute_challenge_labels
from pose.prover.grpc_service import serve_unix


class ProverService:
    def describe(self) -> dict[str, object]:
        return {
            "status": "pose-db-fast-phase-ready",
            "protocol": "graph-based PoSE-DB",
            "supports_host_memory": True,
            "supports_gpu_hbm": True,
            "transport": "gRPC over Unix domain sockets",
            "fast_phase": "timed single-label challenge rounds",
        }

    def self_test(self) -> dict[str, object]:
        graph = build_pose_db_graph(
            label_count_m=8,
            hash_backend="blake3-xof",
            label_width_bits=256,
        )
        labels = compute_challenge_labels(
            graph,
            session_seed="11" * 32,
            challenge_indices=[0, 1],
        )
        return {
            "status": "ok",
            "graph_descriptor_digest": graph.graph_descriptor_digest,
            "label_count_m": graph.label_count_m,
            "gamma": graph.gamma,
            "challenge_label_hex": [label.hex() for label in labels],
        }

    def serve(self, config_path: str) -> None:
        payload = tomllib.loads(Path(config_path).read_text(encoding="utf-8"))
        transport = payload.get("transport", {})
        if not isinstance(transport, dict):
            raise UnsupportedPhaseError("Prover config [transport] section must be a table.")
        socket_path = transport.get("uds_path") or payload.get("socket_path")
        if not socket_path:
            raise UnsupportedPhaseError(
                "Prover config must define transport.uds_path for the gRPC Unix socket."
            )
        serve_unix(str(socket_path))
