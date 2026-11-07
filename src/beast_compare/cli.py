"""Command-line entry point."""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
from collections.abc import Sequence
from pathlib import Path

from beast_compare import __version__
from beast_compare.config import ConfigError, RecorderConfig, load_config
from beast_compare.manifest import ManifestError
from beast_compare.recorder import record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="beast-compare",
        description="Record and compare ADS-B receivers from their Beast outputs.",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    record_parser = commands.add_parser(
        "record", help="record the Beast streams of the configured receivers"
    )
    record_parser.add_argument(
        "--config", type=Path, required=True, help="recorder YAML"
    )

    compare = commands.add_parser("compare", help="analyse a recorded run")
    compare.add_argument("run_dir", type=Path, help="run directory to analyse")

    return parser


async def _record_until_signalled(config: RecorderConfig) -> None:
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(signum, stop.set)
    await record(config, stop)


def _record(config_path: Path) -> int:
    try:
        config = load_config(config_path)
    except (OSError, ConfigError) as exc:
        print(f"beast-compare: {exc}", file=sys.stderr)
        return 2
    try:
        asyncio.run(_record_until_signalled(config))
    except ManifestError as exc:
        print(f"beast-compare: {exc}", file=sys.stderr)
        return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.command == "record":
        return _record(args.config)
    print(f"beast-compare: {args.command} is not implemented yet", file=sys.stderr)
    return 2
