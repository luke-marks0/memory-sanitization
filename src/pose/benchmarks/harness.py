from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path
from shutil import copy2
from time import process_time
from typing import Any

from pose.benchmarks.profiles import load_profile, load_profiles
from pose.benchmarks.summarize import summarize_session_results
from pose.common.env import capture_environment
from pose.common.errors import ResourceFailure
from pose.common.gpu_lease import get_cuda_runtime
from pose.protocol.codec import dump_json_file
from pose.protocol.result_schema import SessionResult
from pose.verifier.service import VerifierService
from pose.verifier.session_store import benchmarks_root, repo_root


def _run_directory(profile_name: str, *, output_dir: str | Path | None = None) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    base = Path(output_dir) if output_dir is not None else benchmarks_root()
    root = base / profile_name / timestamp
    root.mkdir(parents=True, exist_ok=False)
    return root


def _command_output(*argv: str) -> str:
    try:
        completed = subprocess.run(
            argv,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unavailable"
    return completed.stdout.strip() or "unavailable"


def _capture_git_metadata() -> dict[str, str]:
    root = repo_root()
    return {
        "head_commit": _command_output("git", "-C", str(root), "rev-parse", "HEAD"),
        "head_status": _command_output("git", "-C", str(root), "status", "--short"),
    }


def _capture_toolchains() -> dict[str, str]:
    root = repo_root()
    return {
        "cargo_version": _command_output("cargo", "--version"),
        "rustc_version": _command_output("rustc", "--version"),
        "rust_toolchain": (root / "rust-toolchain.toml").read_text(encoding="utf-8"),
    }


def _capture_gpu_inventory() -> dict[str, object]:
    try:
        runtime = get_cuda_runtime()
        device_count = runtime.device_count()
        devices = []
        for device in range(device_count):
            free_bytes, total_bytes = runtime.mem_get_info(device)
            devices.append(
                {
                    "device": device,
                    "available_bytes": free_bytes,
                    "total_bytes": total_bytes,
                }
            )
        payload: dict[str, object] = {
            "cuda_runtime_available": True,
            "cuda_runtime_version": runtime.runtime_version(),
            "cuda_driver_version": runtime.driver_version(),
            "device_count": device_count,
            "devices": devices,
            "nvidia_smi": _command_output("nvidia-smi", "--query-gpu=index,name,memory.total,driver_version", "--format=csv,noheader"),
        }
        return payload
    except ResourceFailure as error:
        return {
            "cuda_runtime_available": False,
            "error": str(error),
            "devices": [],
            "nvidia_smi": _command_output("nvidia-smi", "--query-gpu=index,name,memory.total,driver_version", "--format=csv,noheader"),
        }


def _extract_note_value(notes: list[str], prefix: str) -> str:
    needle = f"{prefix}="
    for note in notes:
        if note.startswith(needle):
            return note[len(needle) :].strip()
    return ""


def _write_run_artifact(
    run_root: Path,
    *,
    result: SessionResult,
    run_index: int,
) -> Path:
    path = run_root / f"run-{run_index:03d}.result.json"
    result.artifact_path = str(path)
    dump_json_file(path, result.to_dict())
    return path


def _write_archive_metadata(
    run_root: Path,
    *,
    plan: dict[str, object],
) -> None:
    dump_json_file(run_root / "plan.json", plan)
    dump_json_file(run_root / "environment.json", {**capture_environment(), **_capture_git_metadata()})
    dump_json_file(run_root / "toolchains.json", _capture_toolchains())
    dump_json_file(run_root / "gpu_inventory.json", _capture_gpu_inventory())


def _run_cold_or_warm_benchmark(profile_identifier: str) -> tuple[list[SessionResult], list[int]]:
    profile = load_profile(profile_identifier)
    verifier = VerifierService()
    results: list[SessionResult] = []
    verifier_cpu_times_ms: list[int] = []
    for _ in range(max(1, int(profile.repetition_count))):
        cpu_started = process_time()
        result = verifier.run_session(profile)
        verifier_cpu_times_ms.append(int((process_time() - cpu_started) * 1000))
        results.append(result)
    return results, verifier_cpu_times_ms


def _run_rechallenge_benchmark(profile_identifier: str) -> tuple[list[SessionResult], list[int]]:
    profile = load_profile(profile_identifier)
    if not profile.target_devices.get("host", False) or profile.target_devices.get("gpus"):
        raise ResourceFailure("Rechallenge benchmarks currently require a host-only profile.")

    verifier = VerifierService()
    warmup = verifier.run_session(profile, retain_session=True)
    results: list[SessionResult] = [warmup]
    verifier_cpu_times_ms: list[int] = [0]
    if not warmup.success:
        return results, verifier_cpu_times_ms

    for run_index in range(max(1, int(profile.repetition_count))):
        cpu_started = process_time()
        result = verifier.rechallenge(
            warmup.session_id,
            release=bool(run_index == (max(1, int(profile.repetition_count)) - 1)),
        )
        verifier_cpu_times_ms.append(int((process_time() - cpu_started) * 1000))
        results.append(result)
    return results, verifier_cpu_times_ms


def prepare_run(profile_identifier: str) -> dict[str, object]:
    profile = load_profile(profile_identifier)
    return {
        "status": "session-ready",
        "profile": profile.to_dict(),
        "note": "Slot-planned PoSE-DB profile execution is available; GPU and hybrid profiles still depend on matching local hardware.",
    }


def prepare_matrix(profiles_directory: str | Path) -> dict[str, object]:
    profiles = load_profiles(profiles_directory)
    return {
        "status": "profile-matrix",
        "profiles": [
            {
                **profile.to_dict(),
                "execution_status": prepare_run(profile.name)["status"],
            }
            for profile in profiles
        ],
    }


def run_benchmark(
    profile_identifier: str,
    *,
    output_dir: str | Path | None = None,
) -> dict[str, object]:
    profile = load_profile(profile_identifier)
    plan = prepare_run(profile_identifier)
    if plan["status"] != "session-ready":
        return {
            "status": "profile-not-yet-executable",
            "plan": plan,
        }

    run_root = _run_directory(profile.name, output_dir=output_dir)
    _write_archive_metadata(run_root, plan=plan)

    if profile.benchmark_class == "rechallenge":
        results, verifier_cpu_times_ms = _run_rechallenge_benchmark(profile_identifier)
    else:
        results, verifier_cpu_times_ms = _run_cold_or_warm_benchmark(profile_identifier)

    log_lines: list[str] = []
    result_paths: list[str] = []
    calibration_artifact_paths: list[str] = []
    for index, (result, cpu_ms) in enumerate(
        zip(results, verifier_cpu_times_ms, strict=True),
        start=1,
    ):
        path = _write_run_artifact(run_root, result=result, run_index=index)
        result_paths.append(str(path))
        calibration_artifact = _extract_note_value(result.notes, "calibration_artifact")
        if calibration_artifact:
            source_path = Path(calibration_artifact)
            if source_path.exists():
                copied_path = run_root / "calibration" / f"run-{index:03d}.calibration.json"
                copied_path.parent.mkdir(parents=True, exist_ok=True)
                copy2(source_path, copied_path)
                calibration_artifact_paths.append(str(copied_path))
        log_lines.append(
            " ".join(
                (
                    f"run={index:03d}",
                    f"session_id={result.session_id}",
                    f"verdict={result.verdict}",
                    f"success={str(result.success).lower()}",
                    f"coverage_fraction={result.coverage_fraction:.6f}",
                    f"q_bound={result.q_bound}",
                    f"gamma={result.gamma}",
                    f"reported_success_bound={result.reported_success_bound}",
                    f"scratch_peak_bytes={result.scratch_peak_bytes}",
                    f"q_over_gamma={(result.q_bound / float(result.gamma)) if result.gamma else 0.0:.6f}",
                    f"round_trip_p95_us={result.round_trip_p95_us}",
                    f"max_round_trip_us={result.max_round_trip_us}",
                    f"fast_phase_ms={result.timings_ms['fast_phase_total']}",
                    f"total_ms={result.timings_ms['total']}",
                    f"verifier_cpu_ms={cpu_ms}",
                )
            )
        )

    summary = summarize_session_results(results, verifier_cpu_times_ms=verifier_cpu_times_ms)
    dump_json_file(run_root / "summary.json", summary)
    (run_root / "benchmark.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    dump_json_file(
        run_root / "manifest.json",
        {
            "profile_name": profile.name,
            "benchmark_class": profile.benchmark_class,
            "run_directory": str(run_root),
            "result_paths": result_paths,
            "calibration_artifact_paths": calibration_artifact_paths,
            "summary_path": str(run_root / "summary.json"),
            "log_path": str(run_root / "benchmark.log"),
            "environment_path": str(run_root / "environment.json"),
            "toolchains_path": str(run_root / "toolchains.json"),
            "gpu_inventory_path": str(run_root / "gpu_inventory.json"),
        },
    )
    return {
        "status": "benchmark-archived",
        "plan": plan,
        "archive": {
            "run_directory": str(run_root),
            "result_paths": result_paths,
            "calibration_artifact_paths": calibration_artifact_paths,
            "summary_path": str(run_root / "summary.json"),
            "log_path": str(run_root / "benchmark.log"),
            "environment_path": str(run_root / "environment.json"),
            "toolchains_path": str(run_root / "toolchains.json"),
            "gpu_inventory_path": str(run_root / "gpu_inventory.json"),
        },
        "summary": summary,
    }
