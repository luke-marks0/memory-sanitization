from __future__ import annotations

import argparse

from pose.cli import bench, prover, verifier


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pose",
        description="Python-first PoSE repository scaffold.",
    )
    subparsers = parser.add_subparsers(dest="persona", required=True)
    prover.register(subparsers)
    verifier.register(subparsers)
    bench.register(subparsers)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

