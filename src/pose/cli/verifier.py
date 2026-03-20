from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pose.benchmarks.profiles import load_profile
from pose.verifier.service import VerifierService


def _print_json(payload: dict[str, object]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def handle_run(args: argparse.Namespace) -> int:
    if not args.profile and not args.plan:
        print("Either --profile or --plan is required.", file=sys.stderr)
        return 2
    if args.plan:
        print("Plan-file execution is not implemented in the foundation phase.", file=sys.stderr)
        return 2
    profile = load_profile(args.profile)
    result = VerifierService().run_placeholder(
        profile,
        note="Verifier run flow is scaffolded but not implemented.",
    )
    _print_json(result.to_dict())
    return 2


def handle_rechallenge(args: argparse.Namespace) -> int:
    _print_json(
        {
            "status": "foundation-scaffold",
            "session_id": args.session_id,
            "note": "Rechallenge support is not implemented yet.",
        }
    )
    return 2


def handle_verify_record(args: argparse.Namespace) -> int:
    result = VerifierService().verify_record(Path(args.path))
    _print_json({"status": "valid", "session_id": result.session_id, "verdict": result.verdict})
    return 0


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("verifier", help="Verifier persona commands.")
    verifier_subparsers = parser.add_subparsers(dest="verifier_command", required=True)

    run_parser = verifier_subparsers.add_parser("run", help="Run a verifier session.")
    run_parser.add_argument("--profile")
    run_parser.add_argument("--plan")
    run_parser.add_argument("--json", action="store_true", help="Retained for spec compatibility.")
    run_parser.set_defaults(func=handle_run)

    rechallenge_parser = verifier_subparsers.add_parser("rechallenge", help="Rechallenge a session.")
    rechallenge_parser.add_argument("--session-id", required=True)
    rechallenge_parser.set_defaults(func=handle_rechallenge)

    verify_record_parser = verifier_subparsers.add_parser(
        "verify-record",
        help="Validate a result artifact against the bootstrap schema.",
    )
    verify_record_parser.add_argument("path")
    verify_record_parser.set_defaults(func=handle_verify_record)

