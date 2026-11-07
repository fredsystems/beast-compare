"""Command-line entry point."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from beast_compare import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="beast-compare",
        description="Record and compare ADS-B receivers from their Beast outputs.",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    record = commands.add_parser(
        "record", help="record the Beast streams of the configured receivers"
    )
    record.add_argument("--config", type=Path, required=True, help="recorder YAML")

    compare = commands.add_parser("compare", help="analyse a recorded run")
    compare.add_argument("run_dir", type=Path, help="run directory to analyse")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(f"beast-compare: {args.command} is not implemented yet", file=sys.stderr)
    return 2
