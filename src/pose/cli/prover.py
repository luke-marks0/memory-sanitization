from __future__ import annotations

import argparse
import json

from pose.prover.service import ProverService
from pose.prover.grpc_service import serve_unix


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


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("prover", help="Prover persona commands.")
    prover_subparsers = parser.add_subparsers(dest="prover_command", required=True)

    serve_parser = prover_subparsers.add_parser("serve", help="Run the prover service.")
    serve_parser.add_argument("--config", required=True)
    serve_parser.set_defaults(func=handle_serve)

    inspect_parser = prover_subparsers.add_parser("inspect", help="Describe prover status.")
    inspect_parser.set_defaults(func=handle_inspect)

    self_test_parser = prover_subparsers.add_parser("self-test", help="Run a PoSE-DB self-check.")
    self_test_parser.set_defaults(func=handle_self_test)

    grpc_parser = prover_subparsers.add_parser(
        "grpc-serve",
        help=argparse.SUPPRESS,
    )
    grpc_parser.add_argument("--socket-path", required=True)
    grpc_parser.set_defaults(func=handle_grpc_serve)
