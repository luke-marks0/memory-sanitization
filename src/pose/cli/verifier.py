from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pose.benchmarks.calibration import prepare_calibration
from pose.benchmarks.profiles import load_profile
from pose.common.errors import ProtocolError
from pose.protocol.result_schema import SessionResult
from pose.verifier.service import VerifierService
from pose.verifier.session_store import write_result_artifact


def _print_json(payload: dict[str, object]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _print_summary(result: SessionResult) -> None:
    formal_notes = "; ".join(result.formal_claim_notes) if result.formal_claim_notes else "none"
    operational_notes = "; ".join(result.operational_claim_notes) if result.operational_claim_notes else "none"
    print(
        "\n".join(
            [
                f"verdict: {result.verdict}",
                f"session_id: {result.session_id}",
                f"graph_family: {result.graph_family}",
                f"graph_parameter_n: {result.graph_parameter_n}",
                f"label_count_m: {result.label_count_m}",
                f"gamma: {result.gamma}",
                f"hash_backend: {result.hash_backend}",
                f"host_covered_bytes: {result.host_covered_bytes}",
                f"gpu_covered_bytes: {sum(result.gpu_covered_bytes_by_device.values())}",
                f"covered_bytes: {result.covered_bytes}",
                f"slack_bytes: {result.slack_bytes}",
                f"coverage_fraction: {result.coverage_fraction:.6f}",
                f"rounds_r: {result.rounds_r}",
                f"deadline_us: {result.deadline_us}",
                f"q_bound: {result.q_bound}",
                f"adversary_model: {result.adversary_model}",
                f"attacker_budget_bytes_assumed: {result.attacker_budget_bytes_assumed}",
                f"reported_success_bound: {result.reported_success_bound}",
                f"formal_claim_notes: {formal_notes}",
                f"operational_claim_notes: {operational_notes}",
                f"scratch_peak_bytes: {result.scratch_peak_bytes}",
                f"round_trip_p50_us: {result.round_trip_p50_us}",
                f"round_trip_p95_us: {result.round_trip_p95_us}",
                f"round_trip_p99_us: {result.round_trip_p99_us}",
                f"total_runtime_ms: {result.timings_ms['total']}",
                f"max_round_trip_us: {result.max_round_trip_us}",
                f"cleanup_status: {result.cleanup_status}",
            ]
        ),
        file=sys.stderr,
    )


def handle_run(args: argparse.Namespace) -> int:
    if not args.profile and not args.plan:
        print("Either --profile or --plan is required.", file=sys.stderr)
        return 2
    if args.plan:
        result = VerifierService().run_plan_file(Path(args.plan))
    else:
        profile = load_profile(args.profile)
        result = VerifierService().run_session(profile, retain_session=bool(args.retain_session))
    artifact_path = write_result_artifact(result)
    result.artifact_path = str(artifact_path)
    write_result_artifact(result)
    _print_summary(result)
    _print_json(result.to_dict())
    if result.verdict == "SUCCESS":
        return 0
    if result.verdict == "PROTOCOL_ERROR":
        return 2
    return 1


def handle_rechallenge(args: argparse.Namespace) -> int:
    try:
        result = VerifierService().rechallenge(args.session_id, release=bool(args.release))
    except ProtocolError as error:
        print(str(error), file=sys.stderr)
        return 2
    artifact_path = write_result_artifact(result)
    result.artifact_path = str(artifact_path)
    write_result_artifact(result)
    _print_summary(result)
    _print_json(result.to_dict())
    if result.verdict == "SUCCESS":
        return 0
    if result.verdict == "PROTOCOL_ERROR":
        return 2
    return 1


def handle_verify_record(args: argparse.Namespace) -> int:
    try:
        result = VerifierService().verify_record(Path(args.path))
    except ProtocolError as error:
        print(str(error), file=sys.stderr)
        return 2
    _print_json({"status": "valid", "session_id": result.session_id, "verdict": result.verdict})
    return 0


def handle_calibrate(args: argparse.Namespace) -> int:
    payload = prepare_calibration(args.profile)
    _print_json(payload)
    return 0 if payload.get("status") == "calibrated" else 1


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("verifier", help="Verifier persona commands.")
    verifier_subparsers = parser.add_subparsers(dest="verifier_command", required=True)

    run_parser = verifier_subparsers.add_parser("run", help="Run a verifier session.")
    run_parser.add_argument("--profile")
    run_parser.add_argument("--plan")
    run_parser.add_argument(
        "--retain-session",
        action="store_true",
        help="Keep a successful host session resident for later rechallenge until expiry.",
    )
    run_parser.add_argument("--json", action="store_true", help="Retained for spec compatibility.")
    run_parser.set_defaults(func=handle_run)

    rechallenge_parser = verifier_subparsers.add_parser("rechallenge", help="Rechallenge a session.")
    rechallenge_parser.add_argument("--session-id", required=True)
    rechallenge_parser.add_argument(
        "--release",
        action="store_true",
        help="Release the resident session after the rechallenge completes.",
    )
    rechallenge_parser.set_defaults(func=handle_rechallenge)

    verify_record_parser = verifier_subparsers.add_parser(
        "verify-record",
        help="Validate a result artifact against the current result schema.",
    )
    verify_record_parser.add_argument("path")
    verify_record_parser.set_defaults(func=handle_verify_record)

    calibrate_parser = verifier_subparsers.add_parser(
        "calibrate",
        help="Prepare or run profile calibration.",
    )
    calibrate_parser.add_argument("--profile", required=True)
    calibrate_parser.set_defaults(func=handle_calibrate)
