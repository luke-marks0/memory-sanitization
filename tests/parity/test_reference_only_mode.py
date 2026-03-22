from __future__ import annotations

import tomllib
from pathlib import Path

from pose.graphs import build_pose_db_graph, compute_challenge_labels, compute_label_array


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_reference_only_repository_has_no_enabled_native_accelerators() -> None:
    root = _repo_root()
    payload = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    dev_dependencies = payload["project"]["optional-dependencies"]["dev"]

    assert not (root / "Cargo.toml").exists()
    assert not (root / "rust-toolchain.toml").exists()
    assert all(not str(item).startswith("maturin") for item in dev_dependencies)


def test_python_reference_paths_agree_on_challenge_label_layout() -> None:
    graph = build_pose_db_graph(
        label_count_m=8,
        graph_parameter_n=2,
        gamma=4,
        hash_backend="blake3-xof",
        label_width_bits=256,
    )

    challenge_labels = compute_challenge_labels(graph, session_seed="33" * 32)
    label_array = compute_label_array(graph, session_seed="33" * 32)

    assert label_array == b"".join(challenge_labels)
