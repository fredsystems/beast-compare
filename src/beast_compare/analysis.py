"""The analysis pipeline: from a recorded run to the analysis table.

A long run does not fit in memory frame by frame (48 hours of four
receivers is hundreds of millions of frames), so the frame-level stages
(ingest, alignment, clustering, positions) run one window at a time. Each
window is loaded with ``overlap_s`` extra on both sides, so that clustering,
CPR pairing and track interpolation near its edges see their neighbours,
and keeps only the transmissions whose earliest arrival falls inside it.
The per-transmission tables are small and are concatenated, and visits are
then numbered over the whole run.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from beast_compare.align import align
from beast_compare.cluster import cluster
from beast_compare.geo import Site
from beast_compare.ingest import FileReport, load_run
from beast_compare.manifest import read_manifest
from beast_compare.positions import attribute, decode_positions
from beast_compare.table import add_visits, build_table
from beast_compare.validity import validity

HOUR_S = 3600.0


@dataclass(frozen=True, slots=True)
class AnalysisSettings:
    window_s: float = HOUR_S
    overlap_s: float = 60.0


@dataclass(slots=True)
class Analysis:
    run_dir: Path
    receivers: list[str]
    reference: str
    site: Site
    start_ns: int
    end_ns: int
    table: pl.DataFrame
    validity: pl.DataFrame
    anchors: pl.DataFrame
    frames: pl.DataFrame
    files: list[FileReport] = field(default_factory=list)


def run_span(run_dir: Path) -> tuple[int, int]:
    """First session start to last session stop (or now, if recording)."""
    sessions = read_manifest(run_dir)["sessions"]

    def ns(text: str | None) -> int:
        when = dt.datetime.fromisoformat(text) if text else dt.datetime.now(dt.UTC)
        return int(when.timestamp() * 1e9)

    return ns(sessions[0]["start"]), ns(sessions[-1]["stop"])


def windows(start_ns: int, end_ns: int, window_s: float) -> list[tuple[int, int]]:
    step = int(window_s * 1e9)
    return [(s, min(s + step, end_ns)) for s in range(start_ns, end_ns, step)]


def analyse(
    run_dir: Path,
    site: Site,
    start_ns: int | None = None,
    end_ns: int | None = None,
    reference: str | None = None,
    settings: AnalysisSettings = AnalysisSettings(),
    progress: Callable[[int, int], None] | None = None,
) -> Analysis:
    receivers = [r["name"] for r in read_manifest(run_dir)["config"]["receivers"]]
    reference = reference or receivers[0]
    if reference not in receivers:
        raise ValueError(f"reference {reference!r} is not one of {receivers}")
    span_start, span_end = run_span(run_dir)
    start_ns = span_start if start_ns is None else start_ns
    end_ns = span_end if end_ns is None else end_ns

    overlap = int(settings.overlap_s * 1e9)
    tables: list[pl.DataFrame] = []
    minutes: list[pl.DataFrame] = []
    anchors: list[pl.DataFrame] = []
    frame_counts: list[pl.DataFrame] = []
    files: dict[Path, FileReport] = {}
    parts = windows(start_ns, end_ns, settings.window_s)
    for index, (window_start, window_end) in enumerate(parts):
        ingested = load_run(run_dir, window_start - overlap, window_end + overlap)
        files.update({f.path: f for f in ingested.files})
        aligned = align(ingested.frames, reference)
        clustered = cluster(aligned.frames)
        positions = decode_positions(clustered.transmissions, site)
        placed = attribute(clustered.transmissions, positions, site)
        inside = aligned.frames.filter(
            pl.col("arrival_ns").is_between(window_start, window_end, closed="left")
        )
        window_validity = validity(run_dir, inside, receivers, window_start, window_end)
        table = build_table(clustered, placed, positions, window_validity, receivers)
        tables.append(
            table.filter(
                pl.col("wall_s").is_between(
                    window_start / 1e9, window_end / 1e9, closed="left"
                )
            )
        )
        minutes.append(window_validity)
        anchors.append(aligned.anchors)
        frame_counts.append(
            inside.group_by("receiver").agg(
                frames=pl.len(), unaligned=pl.col("t_ref").is_null().sum()
            )
        )
        if progress:
            progress(index + 1, len(parts))

    table = add_visits(
        pl.concat(tables)
        .drop("tx")
        .with_row_index("tx")
        .with_columns(pl.col("tx").cast(pl.UInt64))
    )
    frames = (
        pl.concat(frame_counts)
        .group_by("receiver")
        .agg(pl.col("frames", "unaligned").sum())
        .sort("receiver")
    )
    return Analysis(
        run_dir=run_dir,
        receivers=receivers,
        reference=reference,
        site=site,
        start_ns=start_ns,
        end_ns=end_ns,
        table=table,
        validity=pl.concat(minutes).unique(["receiver", "minute_ns"], keep="first"),
        anchors=pl.concat(anchors),
        frames=frames,
        files=sorted(files.values(), key=lambda f: str(f.path)),
    )
