#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pose.common.errors import ResourceFailure
from pose.common.upstream import (
    FIL_PROOFS_PARAMETER_CACHE_ENV,
    ProofArtifact,
    artifact_download_url,
    compute_file_digest,
    extract_missing_parameter_filenames,
    load_upstream_artifacts,
    minimal_upstream_test_artifacts,
    parameter_cache_dir,
    read_upstream_toolchain,
    validate_upstream_snapshot,
    vendor_root,
)

UBUNTU_BUILD_PACKAGES = (
    "build-essential",
    "clang",
    "cmake",
    "pkg-config",
    "libhwloc-dev",
    "ocl-icd-opencl-dev",
    "ca-certificates",
)
DEFAULT_CARGO_TEST_ARGS = ("test", "--workspace", "--all-targets")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Bootstrap the vendored Filecoin Rust test environment, hydrate the "
            "required proof artifacts, and run the upstream workspace test suite."
        )
    )
    parser.add_argument(
        "--bootstrap-only",
        action="store_true",
        help="Install prerequisites but do not hydrate artifacts or run cargo tests.",
    )
    parser.add_argument(
        "--hydrate-only",
        action="store_true",
        help="Install prerequisites and hydrate artifacts without running cargo tests.",
    )
    parser.add_argument(
        "--skip-system-deps",
        action="store_true",
        help="Do not install Ubuntu build packages automatically.",
    )
    parser.add_argument(
        "--skip-rust-bootstrap",
        action="store_true",
        help="Do not install rustup or the upstream-pinned toolchain automatically.",
    )
    parser.add_argument(
        "--skip-artifact-hydration",
        action="store_true",
        help="Do not prefetch the minimal proof artifact set before running tests.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Number of cargo reruns allowed after missing-artifact recovery.",
    )
    parser.add_argument(
        "cargo_args",
        nargs=argparse.REMAINDER,
        help=(
            "Optional cargo arguments. Use '--' before them. Defaults to "
            "'cargo test --workspace --all-targets'."
        ),
    )
    args = parser.parse_args()
    if args.bootstrap_only and args.hydrate_only:
        parser.error("--bootstrap-only and --hydrate-only cannot be combined")
    return args


def prepend_path(env: dict[str, str], value: Path) -> dict[str, str]:
    updated = env.copy()
    current = updated.get("PATH", "")
    updated["PATH"] = f"{value}{os.pathsep}{current}" if current else str(value)
    return updated


def cargo_environment() -> dict[str, str]:
    env = prepend_path(os.environ.copy(), Path.home() / ".cargo" / "bin")
    env.setdefault("CARGO_REGISTRIES_CRATES_IO_PROTOCOL", "sparse")
    env.setdefault(FIL_PROOFS_PARAMETER_CACHE_ENV, str(parameter_cache_dir()))
    if "CARGO_BUILD_JOBS" not in env:
        env["CARGO_BUILD_JOBS"] = str(max(1, (os.cpu_count() or 1)))
    return env


def installed_debian_packages(packages: tuple[str, ...]) -> list[str]:
    if shutil.which("dpkg-query") is None:
        return list(packages)

    missing: list[str] = []
    for package in packages:
        completed = subprocess.run(
            ("dpkg-query", "-W", "-f=${Status}", package),
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0 or "install ok installed" not in completed.stdout:
            missing.append(package)
    return missing


def ensure_system_packages() -> None:
    if sys.platform != "linux" or shutil.which("apt-get") is None:
        print("Skipping automatic system package bootstrap on this platform.")
        return

    missing = installed_debian_packages(UBUNTU_BUILD_PACKAGES)
    if not missing:
        print("System build packages already installed.")
        return

    install_command = (
        "apt-get update && apt-get install --no-install-recommends --yes " + " ".join(missing)
    )
    if os.geteuid() != 0:
        raise ResourceFailure(
            "Missing Ubuntu build packages: "
            f"{', '.join(missing)}. Install them with: sudo {install_command}"
        )

    print(f"Installing Ubuntu build packages: {', '.join(missing)}")
    subprocess.run(("apt-get", "update"), check=True)
    subprocess.run(
        ("apt-get", "install", "--no-install-recommends", "--yes", *missing),
        check=True,
    )


def ensure_rustup_installed(env: dict[str, str]) -> dict[str, str]:
    if shutil.which("rustup", path=env["PATH"]) and shutil.which("cargo", path=env["PATH"]):
        return env

    print("Installing rustup and cargo into ~/.cargo")
    with tempfile.TemporaryDirectory(prefix="pose-rustup-") as temp_dir:
        installer_path = Path(temp_dir) / "rustup-init.sh"
        urllib.request.urlretrieve("https://sh.rustup.rs", installer_path)
        subprocess.run(
            ("sh", str(installer_path), "-y", "--profile", "minimal", "--default-toolchain", "none"),
            check=True,
            env=env,
        )
    return cargo_environment()


def ensure_upstream_toolchain(env: dict[str, str], toolchain: str) -> None:
    print(f"Ensuring upstream Rust toolchain {toolchain} is installed")
    subprocess.run(
        ("rustup", "toolchain", "install", toolchain, "--profile", "minimal"),
        check=True,
        env=env,
    )


def ensure_artifact(artifact: ProofArtifact, cache_dir: Path) -> bool:
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / artifact.name
    if destination.exists() and compute_file_digest(destination) == artifact.digest:
        return False

    if destination.exists():
        destination.unlink()

    temporary_path = destination.with_name(destination.name + ".part")
    if temporary_path.exists():
        temporary_path.unlink()

    url = artifact_download_url(artifact.name)
    print(f"Downloading {artifact.name}")
    try:
        with urllib.request.urlopen(url) as response, temporary_path.open("wb") as handle:
            while True:
                chunk = response.read(8 * 1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
    except urllib.error.URLError as error:
        raise ResourceFailure(f"Failed to download {url}: {error}") from error

    actual_digest = compute_file_digest(temporary_path)
    if actual_digest != artifact.digest:
        temporary_path.unlink(missing_ok=True)
        raise ResourceFailure(
            f"Downloaded digest mismatch for {artifact.name}: "
            f"expected {artifact.digest}, got {actual_digest}"
        )

    temporary_path.replace(destination)
    return True


def hydrate_minimal_artifacts(repo_root: Path) -> list[str]:
    cache_dir = parameter_cache_dir()
    downloaded: list[str] = []
    for artifact in minimal_upstream_test_artifacts(repo_root):
        if ensure_artifact(artifact, cache_dir):
            downloaded.append(artifact.name)
    return downloaded


def recover_missing_artifacts(repo_root: Path, output: str) -> list[str]:
    known_artifacts = load_upstream_artifacts(repo_root)
    recovered: list[str] = []
    cache_dir = parameter_cache_dir()
    for filename in extract_missing_parameter_filenames(output):
        artifact = known_artifacts.get(filename)
        if artifact is None:
            continue
        if ensure_artifact(artifact, cache_dir):
            recovered.append(filename)
    return recovered


def run_and_stream(command: tuple[str, ...], cwd: Path, env: dict[str, str]) -> tuple[int, str]:
    collected: list[str] = []
    with subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    ) as process:
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            collected.append(line)
        return_code = process.wait()
    return return_code, "".join(collected)


def run_upstream_cargo_tests(
    repo_root: Path,
    env: dict[str, str],
    max_retries: int,
    cargo_args: tuple[str, ...],
) -> int:
    workspace = vendor_root(repo_root)
    for attempt in range(max_retries + 1):
        print(f"Running vendored upstream cargo command (attempt {attempt + 1}/{max_retries + 1})")
        return_code, output = run_and_stream(("cargo", *cargo_args), cwd=workspace, env=env)
        if return_code == 0:
            return 0

        recovered = recover_missing_artifacts(repo_root, output)
        if not recovered or attempt == max_retries:
            return return_code

        print("Recovered missing proof artifacts:")
        for filename in recovered:
            print(f"- {filename}")
    return 1


def main() -> int:
    args = parse_args()
    repo_root = REPO_ROOT
    validate_upstream_snapshot(repo_root)

    if not args.skip_system_deps:
        ensure_system_packages()

    env = cargo_environment()
    if not args.skip_rust_bootstrap:
        env = ensure_rustup_installed(env)
        ensure_upstream_toolchain(env, read_upstream_toolchain(repo_root))

    cargo_args = tuple(args.cargo_args[1:]) if args.cargo_args[:1] == ["--"] else tuple(args.cargo_args)
    if not cargo_args:
        cargo_args = DEFAULT_CARGO_TEST_ARGS

    if args.bootstrap_only:
        print("Upstream Rust bootstrap complete.")
        return 0

    if not args.skip_artifact_hydration:
        downloaded = hydrate_minimal_artifacts(repo_root)
        cache_dir = parameter_cache_dir()
        if downloaded:
            print(f"Hydrated {len(downloaded)} upstream proof artifact(s) into {cache_dir}")
        else:
            print(f"Upstream proof artifacts already present in {cache_dir}")

    if args.hydrate_only:
        return 0

    return run_upstream_cargo_tests(
        repo_root=repo_root,
        env=env,
        max_retries=max(0, args.max_retries),
        cargo_args=cargo_args,
    )


if __name__ == "__main__":
    raise SystemExit(main())
