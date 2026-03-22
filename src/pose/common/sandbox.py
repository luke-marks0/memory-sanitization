from __future__ import annotations

import shutil
from dataclasses import dataclass

from pose.common.errors import ProtocolError, ResourceFailure


@dataclass(frozen=True)
class ProverSandboxPolicy:
    mode: str = "none"
    process_memory_max_bytes: int = 0
    require_no_visible_gpus: bool = False
    memlock_max_bytes: int = 0
    file_size_max_bytes: int = 0

    @classmethod
    def from_dict(cls, payload: dict[str, object] | None) -> "ProverSandboxPolicy":
        if payload is None:
            return cls()
        mode = str(payload.get("mode", "none"))
        if mode not in {"none", "process_budget_dev"}:
            raise ProtocolError(f"Unsupported prover sandbox mode: {mode!r}")
        process_memory_max_bytes = int(payload.get("process_memory_max_bytes", 0))
        memlock_max_bytes = int(payload.get("memlock_max_bytes", 0))
        file_size_max_bytes = int(payload.get("file_size_max_bytes", 0))
        if process_memory_max_bytes < 0:
            raise ProtocolError(
                "prover sandbox process_memory_max_bytes must be non-negative, got "
                f"{process_memory_max_bytes}"
            )
        if memlock_max_bytes < 0:
            raise ProtocolError(
                f"prover sandbox memlock_max_bytes must be non-negative, got {memlock_max_bytes}"
            )
        if file_size_max_bytes < 0:
            raise ProtocolError(
                f"prover sandbox file_size_max_bytes must be non-negative, got {file_size_max_bytes}"
            )
        return cls(
            mode=mode,
            process_memory_max_bytes=process_memory_max_bytes,
            require_no_visible_gpus=bool(
                payload.get("require_no_visible_gpus", mode == "process_budget_dev")
            ),
            memlock_max_bytes=memlock_max_bytes,
            file_size_max_bytes=file_size_max_bytes,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "process_memory_max_bytes": self.process_memory_max_bytes,
            "require_no_visible_gpus": self.require_no_visible_gpus,
            "memlock_max_bytes": self.memlock_max_bytes,
            "file_size_max_bytes": self.file_size_max_bytes,
        }


def sandboxed_child_environment(
    base_env: dict[str, str],
    *,
    require_no_visible_gpus: bool,
) -> dict[str, str]:
    env = dict(base_env)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if require_no_visible_gpus:
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["NVIDIA_VISIBLE_DEVICES"] = "void"
    return env


def sandboxed_command(
    argv: list[str],
    *,
    process_memory_max_bytes: int,
    memlock_max_bytes: int,
    file_size_max_bytes: int,
) -> list[str]:
    prlimit_path = shutil.which("prlimit")
    if prlimit_path is None:
        raise ResourceFailure("Sandboxed prover launch requires `prlimit` to be installed")
    command = [
        prlimit_path,
    ]
    if int(process_memory_max_bytes) > 0:
        command.append(f"--as={int(process_memory_max_bytes)}")
    command.extend(
        [
            f"--memlock={max(0, int(memlock_max_bytes))}",
            f"--fsize={max(0, int(file_size_max_bytes))}",
            "--",
            *argv,
        ]
    )
    return command


def sandbox_claim_notes(policy: ProverSandboxPolicy) -> list[str]:
    if policy.mode == "none":
        return []
    notes = [f"prover_sandbox_mode={policy.mode}"]
    if policy.mode == "process_budget_dev":
        notes.extend(
            (
                "development_only_attacker_budget_override=true",
                "development_only_not_for_production=true",
                "development_only_full_local_memory_claim=false",
                "development_only_process_budget_enforced_via=prlimit_rlimit_as",
                f"prover_sandbox_process_memory_max_bytes={policy.process_memory_max_bytes}",
            )
        )
    if policy.require_no_visible_gpus:
        notes.append("prover_sandbox_hidden_gpu_tiers=all")
    notes.append(f"prover_sandbox_memlock_max_bytes={policy.memlock_max_bytes}")
    notes.append(f"prover_sandbox_file_size_max_bytes={policy.file_size_max_bytes}")
    return notes
