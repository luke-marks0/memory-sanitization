from __future__ import annotations

import argparse
import json

from pose.benchmarks.harness import placeholder_result, prepare_matrix, prepare_run
from pose.benchmarks.summarize import summarize_results


def _print_json(payload: dict[str, object]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def handle_run(args: argparse.Namespace) -> int:
    payload = {
        "plan": prepare_run(args.profile),
        "result": placeholder_result(args.profile),
    }
    _print_json(
        {
            "plan": payload["plan"],
            "result": payload["result"],
        }
    )
    return 0 if payload["result"].get("verdict") == "SUCCESS" else 1


def handle_matrix(args: argparse.Namespace) -> int:
    _print_json(prepare_matrix(args.profiles))
    return 0


def handle_summarize(args: argparse.Namespace) -> int:
    _print_json(summarize_results(args.results))
    return 0


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("bench", help="Benchmark persona commands.")
    bench_subparsers = parser.add_subparsers(dest="bench_command", required=True)

    run_parser = bench_subparsers.add_parser("run", help="Prepare a benchmark run.")
    run_parser.add_argument("--profile", required=True)
    run_parser.set_defaults(func=handle_run)

    matrix_parser = bench_subparsers.add_parser("matrix", help="Inspect a profile directory.")
    matrix_parser.add_argument("--profiles", required=True)
    matrix_parser.set_defaults(func=handle_matrix)

    summarize_parser = bench_subparsers.add_parser("summarize", help="Summarize result artifacts.")
    summarize_parser.add_argument("results", nargs="+")
    summarize_parser.set_defaults(func=handle_summarize)
