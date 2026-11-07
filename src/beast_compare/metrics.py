"""Detection metrics over the analysis table (see beast_compare.table).

Every rate is a receiver's share of the reference transmissions in a band,
counted only over minutes in which that receiver was valid, with a 95 %
interval from resampling aircraft visits (see beast_compare.bootstrap).
Comparisons between receivers (wins, exclusives) use only minutes in which
every receiver was valid, so a receiver being down never counts against
the others.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

import numpy as np
import numpy.typing as npt
import polars as pl

from beast_compare.bootstrap import (
    DEFAULT_RESAMPLES,
    DEFAULT_SEED,
    Interval,
    ratio,
    ratio_difference,
    visit_weights,
)
from beast_compare.table import DF_GROUPS, heard, signal, valid

Floats = npt.NDArray[np.float64]

# A receiver valid for less than this share of the run is not comparable.
COMPARABLE_SHARE = 0.9
# Bins with fewer transmissions are left out of sensitivity thresholds.
MIN_BIN_TRANSMISSIONS = 50


@dataclass(frozen=True, slots=True)
class Dimension:
    """Bands of a column: band i holds values in [edges[i-1], edges[i])."""

    name: str
    column: str
    edges: tuple[float, ...]
    labels: tuple[str, ...]

    def band(self) -> pl.Expr:
        index = pl.sum_horizontal(
            [(pl.col(self.column) >= e).cast(pl.UInt8) for e in self.edges]
        )
        return pl.when(pl.col(self.column).is_not_null()).then(index)


def _steps(name: str, column: str, step: int, top: int, unit: str) -> Dimension:
    edges = tuple(float(e) for e in range(step, top + 1, step))
    lows = (0, *range(step, top + 1, step))
    labels = tuple(
        f"{lo}-{lo + step}{unit}" if lo < top else f"{top}+{unit}" for lo in lows
    )
    return Dimension(name, column, edges, labels)


RANGE = _steps("range", "range_nm", 25, 250, " nm")
ELEVATION = Dimension(
    "elevation",
    "elevation",
    (0.0, 1.0, 2.0, 5.0, 10.0, 20.0),
    ("<0", "0-1", "1-2", "2-5", "5-10", "10-20", "20+"),
)
BEARING = Dimension(
    "bearing",
    "bearing",
    tuple(float(e) for e in range(30, 331, 30)),
    tuple(f"{lo}-{lo + 30}" for lo in range(0, 331, 30)),
)
ALTITUDE = Dimension(
    "altitude",
    "alt_ft",
    tuple(float(e) for e in range(5000, 45001, 5000)),
    tuple(f"{lo // 1000}-{lo // 1000 + 5}k ft" for lo in range(0, 45000, 5000))
    + ("45k+ ft",),
)
MESSAGE_TYPE = Dimension(
    "message type",
    "df_group",
    (1.0, 2.0, 3.0, 4.0, 5.0),
    DF_GROUPS,
)
DENSITY = _steps("traffic density", "density", 250, 1000, " tx/s")
DIMENSIONS = (RANGE, ELEVATION, BEARING, ALTITUDE, MESSAGE_TYPE, DENSITY)


@dataclass(slots=True)
class Resampler:
    """Draws visit weights reproducibly."""

    resamples: int = DEFAULT_RESAMPLES
    rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(DEFAULT_SEED)
    )

    def weights(self, visits: int) -> npt.NDArray[np.int64]:
        return visit_weights(visits, self.resamples, self.rng)


def _visit_counts(rows: pl.DataFrame, receivers: list[str]) -> pl.DataFrame:
    """Per visit: n transmissions and, per receiver, k decoded."""
    return rows.group_by("visit").agg(
        pl.len().alias("n"), *[pl.col(heard(r)).sum().alias(r) for r in receivers]
    )


def _arrays(counts: pl.DataFrame, receiver: str) -> tuple[Floats, Floats]:
    return (
        counts["n"].to_numpy().astype(np.float64),
        counts[receiver].to_numpy().astype(np.float64),
    )


def _scored(table: pl.DataFrame) -> pl.DataFrame:
    return table.filter(pl.col("visit").is_not_null())


def rates_by(
    table: pl.DataFrame,
    dimension: Dimension | None,
    receivers: list[str],
    resampler: Resampler,
) -> pl.DataFrame:
    """Each receiver's detection rate per band (``dimension=None``: overall)."""
    rows: list[dict[str, object]] = []
    base = _scored(table)
    band = dimension.band() if dimension else pl.lit(0, dtype=pl.UInt8)
    base = base.with_columns(band=band).filter(pl.col("band").is_not_null())
    for receiver in receivers:
        own = base.filter(pl.col(valid(receiver)))
        for (index,), group in own.group_by("band"):
            counts = _visit_counts(group, [receiver])
            n, k = _arrays(counts, receiver)
            interval = ratio(n, k, resampler.weights(len(counts)))
            rows.append(
                {
                    "receiver": receiver,
                    "order": cast(int, index),
                    "band": dimension.labels[cast(int, index)] if dimension else "all",
                    "n": int(n.sum()),
                    "visits": len(counts),
                    "rate": interval.estimate,
                    "low": interval.low,
                    "high": interval.high,
                }
            )
    return _frame(
        rows,
        {
            "receiver": pl.Enum(receivers),
            "order": pl.Int64(),
            "band": pl.String(),
            "n": pl.Int64(),
            "visits": pl.Int64(),
            "rate": pl.Float64(),
            "low": pl.Float64(),
            "high": pl.Float64(),
        },
    ).sort("order", "receiver")


def _frame(
    rows: list[dict[str, object]], schema: dict[str, pl.DataType]
) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=pl.Schema(schema))


def all_valid(table: pl.DataFrame, receivers: list[str]) -> pl.Expr:
    return pl.all_horizontal([pl.col(valid(r)) for r in receivers])


def wins_by(
    table: pl.DataFrame,
    dimension: Dimension,
    receivers: list[str],
    resampler: Resampler,
) -> pl.DataFrame:
    """Per band, over minutes where every receiver was valid: the best
    receiver, its margin over the runner-up, and whether the margin's
    interval excludes zero (``decided``)."""
    rows: list[dict[str, object]] = []
    base = (
        _scored(table)
        .filter(all_valid(table, receivers))
        .with_columns(band=dimension.band())
        .filter(pl.col("band").is_not_null())
    )
    for (index,), group in base.group_by("band"):
        counts = _visit_counts(group, receivers)
        weights = resampler.weights(len(counts))
        n = counts["n"].to_numpy().astype(np.float64)
        rates = {r: counts[r].sum() / n.sum() for r in receivers}
        ranked = sorted(receivers, key=lambda r: rates[r], reverse=True)
        margin = Interval(np.nan, np.nan, np.nan)
        if len(ranked) > 1:
            best, second = ranked[0], ranked[1]
            margin = ratio_difference(
                n,
                counts[best].to_numpy().astype(np.float64),
                n,
                counts[second].to_numpy().astype(np.float64),
                weights,
            )
        rows.append(
            {
                "order": cast(int, index),
                "band": dimension.labels[cast(int, index)],
                "n": int(n.sum()),
                "visits": len(counts),
                "best": ranked[0],
                "runner_up": ranked[1] if len(ranked) > 1 else None,
                "margin": margin.estimate,
                "low": margin.low,
                "high": margin.high,
                "decided": margin.excludes_zero,
                **{f"rate:{r}": rates[r] for r in receivers},
            }
        )
    schema: dict[str, pl.DataType] = {
        "order": pl.Int64(),
        "band": pl.String(),
        "n": pl.Int64(),
        "visits": pl.Int64(),
        "best": pl.String(),
        "runner_up": pl.String(),
        "margin": pl.Float64(),
        "low": pl.Float64(),
        "high": pl.Float64(),
        "decided": pl.Boolean(),
    }
    schema.update({f"rate:{r}": pl.Float64() for r in receivers})
    return _frame(rows, schema).sort("order")


def dbfs(column: str) -> pl.Expr:
    """A Beast signal byte (amplitude, 255 = full scale) in dBFS."""
    return 20 * (pl.col(column).cast(pl.Float64).clip(lower_bound=1) / 255).log10()


def sensitivity(
    table: pl.DataFrame, receivers: list[str], resampler: Resampler
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Detection rate of each receiver against each other receiver's signal
    level, in 1 dB bins; and the level where each curve first reaches 50 %.

    Signal scales differ between decoders, so a curve is read against the
    reference receiver named in ``by``, never across ``by`` values.
    """
    curves: list[dict[str, object]] = []
    thresholds: list[dict[str, object]] = []
    base = _scored(table)
    for by in receivers:
        for receiver in receivers:
            if receiver == by:
                continue
            rows = base.filter(
                pl.col(valid(receiver)) & pl.col(valid(by)) & pl.col(heard(by))
            ).with_columns(level=dbfs(signal(by)).floor())
            curve: list[tuple[float, float, int]] = []
            for (level,), group in rows.group_by("level"):
                counts = _visit_counts(group, [receiver])
                n, k = _arrays(counts, receiver)
                interval = ratio(n, k, resampler.weights(len(counts)))
                lvl = cast(float, level)
                curves.append(
                    {
                        "receiver": receiver,
                        "by": by,
                        "level_dbfs": lvl,
                        "n": int(n.sum()),
                        "rate": interval.estimate,
                        "low": interval.low,
                        "high": interval.high,
                    }
                )
                curve.append((lvl, interval.estimate, int(n.sum())))
            thresholds.append(
                {
                    "receiver": receiver,
                    "by": by,
                    "level_at_50_dbfs": _crossing(curve, 0.5),
                }
            )
    curve_frame = _frame(
        curves,
        {
            "receiver": pl.String(),
            "by": pl.String(),
            "level_dbfs": pl.Float64(),
            "n": pl.Int64(),
            "rate": pl.Float64(),
            "low": pl.Float64(),
            "high": pl.Float64(),
        },
    ).sort("by", "receiver", "level_dbfs")
    threshold_frame = _frame(
        thresholds,
        {"receiver": pl.String(), "by": pl.String(), "level_at_50_dbfs": pl.Float64()},
    )
    return curve_frame, threshold_frame


def _crossing(curve: list[tuple[float, float, int]], target: float) -> float | None:
    """First level, weak to strong, where the rate reaches ``target``,
    interpolated between bins; bins with too few transmissions skipped."""
    points = sorted((lvl, rate) for lvl, rate, n in curve if n >= MIN_BIN_TRANSMISSIONS)
    for (l0, r0), (l1, r1) in zip(points, points[1:], strict=False):
        if r0 < target <= r1:
            return l0 + (target - r0) / (r1 - r0) * (l1 - l0)
    if points and points[0][1] >= target:
        return points[0][0]
    return None


def exclusives(table: pl.DataFrame, receivers: list[str]) -> pl.DataFrame:
    """Over minutes where every receiver was valid: what each receiver heard
    that no other did, and what it alone missed."""
    base = table.filter(all_valid(table, receivers))
    rows: list[dict[str, object]] = []
    for receiver in receivers:
        others = [heard(r) for r in receivers if r != receiver]
        any_other = pl.any_horizontal(others) if others else pl.lit(False)
        all_others = pl.all_horizontal(others) if others else pl.lit(False)
        only = pl.col(heard(receiver)) & ~any_other
        missed_only = ~pl.col(heard(receiver)) & all_others
        by_visit = (
            base.filter(pl.col("visit").is_not_null())
            .group_by("visit")
            .agg(
                own=pl.col(heard(receiver)).any(),
                other=any_other.any(),
            )
        )
        summary = base.select(
            heard=pl.col(heard(receiver)).sum(),
            only=only.sum(),
            missed_only=missed_only.sum(),
            positions_only=(only & pl.col("position")).sum(),
        ).row(0, named=True)
        rows.append(
            {
                "receiver": receiver,
                "transmissions": len(base),
                **summary,
                "visits_only": by_visit.filter(pl.col("own") & ~pl.col("other")).height,
            }
        )
    return _frame(
        rows,
        {
            "receiver": pl.String(),
            "transmissions": pl.Int64(),
            "heard": pl.Int64(),
            "only": pl.Int64(),
            "missed_only": pl.Int64(),
            "positions_only": pl.Int64(),
            "visits_only": pl.Int64(),
        },
    )


def timing(anchors: pl.DataFrame) -> pl.DataFrame:
    """Per receiver: how tightly its timestamps follow the reference's."""
    return (
        anchors.group_by("receiver")
        .agg(
            anchors=pl.len(),
            mad_us=(pl.col("residual_us") - pl.col("residual_us").median())
            .abs()
            .median(),
            p99_us=pl.col("residual_us").abs().quantile(0.99),
        )
        .with_columns(pl.col("receiver").cast(pl.String))
        .sort("receiver")
    )


def comparability(validity_table: pl.DataFrame) -> pl.DataFrame:
    return (
        validity_table.group_by("receiver")
        .agg(minutes=pl.len(), valid_share=pl.col("valid").mean())
        .with_columns(
            pl.col("receiver").cast(pl.String),
            comparable=pl.col("valid_share") >= COMPARABLE_SHARE,
        )
        .sort("receiver")
    )
