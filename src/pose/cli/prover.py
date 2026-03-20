from __future__ import annotations

import argparse
import json
import sys

from pose.prover.service import ProverService
from pose.prover.grpc_service import serve_unix
from pose.prover.host_worker import handle_request_json, handle_resident_session_json


def _print_json(payload: dict[str, object]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def handle_inspect(_args: argparse.Namespace) -> int:
    _print_json(ProverService().describe())
    return 0


def handle_self_test(_args: argparse.Namespace) -> int:
    _print_json(ProverService().self_test())
    return 0


def handle_serve(args: argparse.Namespace) -> int:
    ProverService().serve(args.config)
    return 0


def handle_grpc_serve(args: argparse.Namespace) -> int:
    serve_unix(args.socket_path)
    return 0


def handle_host_session_worker(_args: argparse.Namespace) -> int:
    response = handle_request_json(sys.stdin.read())
    print(response)
    return 0


def handle_host_session_resident(_args: argparse.Namespace) -> int:
    handle_resident_session_json(sys.stdin.read())
    return 0


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("prover", help="Prover persona commands.")
    prover_subparsers = parser.add_subparsers(dest="prover_command", required=True)

    serve_parser = prover_subparsers.add_parser("serve", help="Run the prover service.")
    serve_parser.add_argument("--config", required=True)
    serve_parser.set_defaults(func=handle_serve)

    inspect_parser = prover_subparsers.add_parser("inspect", help="Describe prover status.")
    inspect_parser.set_defaults(func=handle_inspect)

    self_test_parser = prover_subparsers.add_parser("self-test", help="Run prover scaffold self-checks.")
    self_test_parser.set_defaults(func=handle_self_test)

    worker_parser = prover_subparsers.add_parser(
        "host-session-worker",
        help=argparse.SUPPRESS,
    )
    worker_parser.set_defaults(func=handle_host_session_worker)

    resident_parser = prover_subparsers.add_parser(
        "host-session-resident",
        help=argparse.SUPPRESS,
    )
    resident_parser.set_defaults(func=handle_host_session_resident)

    grpc_parser = prover_subparsers.add_parser(
        "grpc-serve",
        help=argparse.SUPPRESS,
    )
    grpc_parser.add_argument("--socket-path", required=True)
    grpc_parser.set_defaults(func=handle_grpc_serve)
