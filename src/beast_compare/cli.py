"""Command-line entry point."""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
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
    site = compare.add_argument_group(
        "receiver site", "the antenna position (kept out of the recording)"
    )
    site.add_argument("--lat", type=float, required=True, help="degrees")
    site.add_argument("--lon", type=float, required=True, help="degrees")
    site.add_argument(
        "--alt-m", type=float, required=True, help="metres above the ellipsoid"
    )
    compare.add_argument(
        "--reference", help="receiver to align on (default: the first configured)"
    )
    compare.add_argument(
        "--start", type=_utc, help="window start, ISO 8601 (default: run start)"
    )
    compare.add_argument(
        "--end", type=_utc, help="window end, ISO 8601 (default: run end)"
    )
    compare.add_argument(
        "--out", type=Path, help="report directory (default: <run_dir>/reports/...)"
    )
    compare.add_argument(
        "--window-minutes",
        type=float,
        default=60.0,
        help="process the run this many minutes at a time (default 60)",
    )
    compare.add_argument(
        "--resamples",
        type=int,
        default=200,
        help="bootstrap resamples for the intervals (default 200)",
    )

    return parser


def _utc(text: str) -> int:
    """An ISO 8601 time as Unix nanoseconds; naive times are taken as UTC."""
    when = dt.datetime.fromisoformat(text)
    if when.tzinfo is None:
        when = when.replace(tzinfo=dt.UTC)
    return int(when.timestamp() * 1e9)


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


def _compare(args: argparse.Namespace) -> int:
    # Imported here so that the recorder never loads the analysis stack.
    from beast_compare.analysis import AnalysisSettings, analyse
    from beast_compare.geo import Site
    from beast_compare.metrics import Resampler
    from beast_compare.report import write_report

    def progress(done: int, total: int) -> None:
        print(f"analysed window {done}/{total}", file=sys.stderr)

    try:
        analysis = analyse(
            args.run_dir,
            Site(args.lat, args.lon, args.alt_m),
            start_ns=args.start,
            end_ns=args.end,
            reference=args.reference,
            settings=AnalysisSettings(window_s=args.window_minutes * 60),
            progress=progress,
        )
    except (OSError, ValueError, KeyError) as exc:
        print(f"beast-compare: {exc}", file=sys.stderr)
        return 1

    stamp = "{}_{}".format(
        *(
            dt.datetime.fromtimestamp(ns / 1e9, tz=dt.UTC).strftime("%Y%m%dT%H%M")
            for ns in (analysis.start_ns, analysis.end_ns)
        )
    )
    out = args.out or args.run_dir / "reports" / stamp
    path = write_report(analysis, out, Resampler(resamples=args.resamples))
    print(path)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.command == "record":
        return _record(args.config)
    return _compare(args)
