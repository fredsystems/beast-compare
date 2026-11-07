"""Per aircraft visit: who acquired it first, who held it longest, and how
continuously each receiver tracked it.

A visit is scored only where every receiver was valid at the moment that
matters (its start for acquisition, its end for fade-out, throughout for
continuity), and visits that begin or end within ``edge_s`` of the
analysed window are left out: they were in view before recording started,
or still in view when it stopped.

Acquisition and fade-out are measured on any message and on positions
separately, on each transmission's wall-clock time (``wall_s``), which is
shared by every receiver that heard it. Times within ``tie_s`` of each
other are a tie.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import polars as pl

from beast_compare.table import heard, valid


@dataclass(frozen=True, slots=True)
class VisitSettings:
    edge_s: float = 300.0
    tie_s: float = 1.0
    bin_s: float = 10.0
    min_duration_s: float = 60.0


def _interior_visits(table: pl.DataFrame, settings: VisitSettings) -> pl.DataFrame:
    window_start = table["wall_s"].min()
    window_end = table["wall_s"].max()
    return (
        table.filter(pl.col("visit").is_not_null())
        .group_by("visit")
        .agg(
            pl.col("address").first(),
            start=pl.col("wall_s").min(),
            end=pl.col("wall_s").max(),
        )
        .filter(
            (pl.col("start") - pl.lit(window_start) >= settings.edge_s)
            & (pl.lit(window_end) - pl.col("end") >= settings.edge_s)
        )
        .select("visit", "address", "start", "end")
    )


def _events(
    table: pl.DataFrame, receivers: list[str], settings: VisitSettings
) -> pl.DataFrame:
    """Per (visit, receiver): first and last message and position times and
    ranges, and whether every receiver was valid at the visit's start/end."""
    visits = _interior_visits(table, settings)
    rows = table.join(visits.select("visit"), on="visit").sort("wall_s")
    every = pl.all_horizontal([pl.col(valid(r)) for r in receivers])
    edges = rows.group_by("visit").agg(
        valid_start=every.first(), valid_end=every.last(), valid_all=every.all()
    )
    parts = []
    for receiver in receivers:
        own = rows.filter(pl.col(heard(receiver)))
        positions = own.filter(pl.col("position"))
        first_last = own.group_by("visit").agg(
            first=pl.col("wall_s").first(),
            last=pl.col("wall_s").last(),
            first_range=pl.col("range_nm").first(),
            last_range=pl.col("range_nm").last(),
        )
        first_last_pos = positions.group_by("visit").agg(
            first_position=pl.col("wall_s").first(),
            last_position=pl.col("wall_s").last(),
        )
        parts.append(
            visits.select("visit")
            .join(first_last, on="visit", how="left")
            .join(first_last_pos, on="visit", how="left")
            .with_columns(receiver=pl.lit(receiver))
        )
    return pl.concat(parts).join(edges, on="visit")


def by_receiver(
    events: pl.DataFrame, receivers: list[str], column: str, valid_column: str
) -> pl.DataFrame:
    """One row per visit valid at ``valid_column``, one ``column`` value per
    receiver (null where it never had one); every receiver has a column even
    when there are no visits."""
    wide = events.filter(pl.col(valid_column)).pivot(
        on="receiver", index="visit", values=column
    )
    missing = [r for r in receivers if r not in wide.columns]
    return wide.with_columns(
        [pl.lit(None, dtype=pl.Float64).alias(r) for r in missing]
    ).select("visit", *receivers)


def _lead_summary(
    events: pl.DataFrame,
    receivers: list[str],
    column: str,
    later_is_better: bool,
    valid_column: str,
    tie_s: float,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    wide = by_receiver(events, receivers, column, valid_column)
    # Only visits some receiver has a value for (e.g. a position) count.
    wide = wide.filter(pl.any_horizontal([pl.col(r).is_not_null() for r in receivers]))
    sign = -1.0 if later_is_better else 1.0

    summary = []
    for receiver in receivers:
        values = wide[receiver].to_numpy().astype(np.float64)
        others = np.column_stack(
            [wide[r].to_numpy().astype(np.float64) for r in receivers]
        )
        seen = ~np.isnan(values)
        best = (
            np.nanmax(others, axis=1) if later_is_better else np.nanmin(others, axis=1)
        )
        lag = sign * (values - best)
        summary.append(
            {
                "receiver": receiver,
                "visits": int(len(values)),
                "detected": int(seen.sum()),
                "best": int((lag[seen] <= tie_s).sum()),
                "median_lag_s": float(np.median(lag[seen])) if seen.any() else None,
                "p90_lag_s": (
                    float(np.percentile(lag[seen], 90)) if seen.any() else None
                ),
            }
        )

    pairs = []
    for a, b in combinations(receivers, 2):
        va = wide[a].to_numpy().astype(np.float64)
        vb = wide[b].to_numpy().astype(np.float64)
        both = ~np.isnan(va) & ~np.isnan(vb)
        # Positive: a did better than b.
        advantage = (sign * (vb - va))[both]
        pairs.append(
            {
                "a": a,
                "b": b,
                "visits": int(both.sum()),
                "a_better": int((advantage > tie_s).sum()),
                "b_better": int((advantage < -tie_s).sum()),
                "ties": int((np.abs(advantage) <= tie_s).sum()),
                "median_advantage_s": (
                    float(np.median(advantage)) if both.any() else None
                ),
            }
        )
    summary_frame = pl.DataFrame(
        summary,
        schema={
            "receiver": pl.String,
            "visits": pl.Int64,
            "detected": pl.Int64,
            "best": pl.Int64,
            "median_lag_s": pl.Float64,
            "p90_lag_s": pl.Float64,
        },
    )
    pair_frame = pl.DataFrame(
        pairs,
        schema={
            "a": pl.String,
            "b": pl.String,
            "visits": pl.Int64,
            "a_better": pl.Int64,
            "b_better": pl.Int64,
            "ties": pl.Int64,
            "median_advantage_s": pl.Float64,
        },
    )
    return summary_frame, pair_frame


@dataclass(frozen=True, slots=True)
class VisitMetrics:
    """``events``: per (visit, receiver) first/last times and ranges.
    ``acquisition`` / ``fade``: per receiver, on any message; the
    ``*_position`` variants on positions; ``*_pairs``: head to head.
    ``continuity``: per receiver, share of 10 s bins of each visit with a
    position, against the share any receiver achieved."""

    events: pl.DataFrame
    acquisition: pl.DataFrame
    acquisition_pairs: pl.DataFrame
    acquisition_position: pl.DataFrame
    fade: pl.DataFrame
    fade_pairs: pl.DataFrame
    fade_position: pl.DataFrame
    continuity: pl.DataFrame


def continuity(
    table: pl.DataFrame, receivers: list[str], settings: VisitSettings
) -> pl.DataFrame:
    visits = _interior_visits(table, settings).filter(
        pl.col("end") - pl.col("start") >= settings.min_duration_s
    )
    rows = table.join(visits.select("visit", "start"), on="visit")
    # Validity over every transmission of the visit, not only its positions:
    # an outage during which only short replies were heard still counts.
    every = pl.all_horizontal([pl.col(valid(r)) for r in receivers])
    usable = rows.group_by("visit").agg(ok=every.all()).filter(pl.col("ok"))
    rows = rows.filter(pl.col("position"))
    rows = rows.join(usable.select("visit"), on="visit").with_columns(
        bin=((pl.col("wall_s") - pl.col("start")) // settings.bin_s).cast(pl.Int64)
    )
    bins = rows.group_by("visit", "bin").agg(
        *[pl.col(heard(r)).any().alias(r) for r in receivers]
    )
    per_visit = bins.group_by("visit").agg(
        any_bins=pl.len(), *[pl.col(r).sum() for r in receivers]
    )
    span = visits.select(
        "visit",
        total_bins=((pl.col("end") - pl.col("start")) // settings.bin_s + 1).cast(
            pl.Int64
        ),
    )
    per_visit = per_visit.join(span, on="visit")
    out = []
    for receiver in receivers:
        share = per_visit[receiver] / per_visit["total_bins"]
        relative = per_visit[receiver] / per_visit["any_bins"]
        out.append(
            {
                "receiver": receiver,
                "visits": per_visit.height,
                "median_coverage": share.median() if per_visit.height else None,
                "median_of_reference": relative.median() if per_visit.height else None,
            }
        )
    return pl.DataFrame(
        out,
        schema={
            "receiver": pl.String,
            "visits": pl.Int64,
            "median_coverage": pl.Float64,
            "median_of_reference": pl.Float64,
        },
    )


def visit_metrics(
    table: pl.DataFrame,
    receivers: list[str],
    settings: VisitSettings = VisitSettings(),
) -> VisitMetrics:
    events = _events(table, receivers, settings)
    tie = settings.tie_s
    acquisition, acquisition_pairs = _lead_summary(
        events, receivers, "first", False, "valid_start", tie
    )
    acquisition_position, _ = _lead_summary(
        events, receivers, "first_position", False, "valid_start", tie
    )
    fade, fade_pairs = _lead_summary(events, receivers, "last", True, "valid_end", tie)
    fade_position, _ = _lead_summary(
        events, receivers, "last_position", True, "valid_end", tie
    )
    return VisitMetrics(
        events,
        acquisition,
        acquisition_pairs,
        acquisition_position,
        fade,
        fade_pairs,
        fade_position,
        continuity(table, receivers, settings),
    )
