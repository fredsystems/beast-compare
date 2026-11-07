"""Where each receiver hears: polar detection cells, range outlines and
altitude-by-range coverage, from the analysis table.

These are plain detection rates per cell, without intervals: they are for
seeing the shape of coverage. Cells with fewer than ``min_n`` reference
transmissions are dropped, since a rate over a handful of messages is noise.
"""

from __future__ import annotations

import polars as pl

from beast_compare.table import heard, valid

RANGE_STEP_NM = 10
BEARING_STEP = 10
ALTITUDE_STEP_FT = 2000


def polar_cells(
    table: pl.DataFrame, receivers: list[str], min_n: int = 30
) -> pl.DataFrame:
    """receiver, range_lo (nm), bearing_lo (deg), n, rate."""
    placed = table.filter(pl.col("range_nm").is_not_null()).with_columns(
        range_lo=(pl.col("range_nm") // RANGE_STEP_NM * RANGE_STEP_NM).cast(pl.Int32),
        bearing_lo=(pl.col("bearing") // BEARING_STEP * BEARING_STEP).cast(pl.Int32),
    )
    parts = [
        placed.filter(pl.col(valid(r)))
        .group_by("range_lo", "bearing_lo")
        .agg(n=pl.len(), rate=pl.col(heard(r)).mean())
        .with_columns(receiver=pl.lit(r))
        for r in receivers
    ]
    return (
        pl.concat(parts)
        .filter(pl.col("n") >= min_n)
        .select("receiver", "range_lo", "bearing_lo", "n", "rate")
        .sort("receiver", "range_lo", "bearing_lo")
    )


def polar_difference(
    table: pl.DataFrame, receivers: list[str], reference: str, min_n: int = 30
) -> pl.DataFrame:
    """Per cell, each receiver's rate minus the reference's, over rows where
    both were valid: receiver, range_lo, bearing_lo, n, difference."""
    placed = table.filter(pl.col("range_nm").is_not_null()).with_columns(
        range_lo=(pl.col("range_nm") // RANGE_STEP_NM * RANGE_STEP_NM).cast(pl.Int32),
        bearing_lo=(pl.col("bearing") // BEARING_STEP * BEARING_STEP).cast(pl.Int32),
    )
    parts = [
        placed.filter(pl.col(valid(r)) & pl.col(valid(reference)))
        .group_by("range_lo", "bearing_lo")
        .agg(
            n=pl.len(),
            difference=pl.col(heard(r)).mean() - pl.col(heard(reference)).mean(),
        )
        .with_columns(receiver=pl.lit(r))
        for r in receivers
        if r != reference
    ]
    if not parts:
        return pl.DataFrame(
            schema={
                "receiver": pl.String,
                "range_lo": pl.Int32,
                "bearing_lo": pl.Int32,
                "n": pl.UInt32,
                "difference": pl.Float64,
            }
        )
    return (
        pl.concat(parts)
        .filter(pl.col("n") >= min_n)
        .select("receiver", "range_lo", "bearing_lo", "n", "difference")
        .sort("receiver", "range_lo", "bearing_lo")
    )


def range_outline(
    table: pl.DataFrame, receivers: list[str], quantile: float = 0.99
) -> pl.DataFrame:
    """Per receiver and 10 degree sector: the ``quantile`` range of the
    transmissions it decoded (a high quantile, not the maximum, so that one
    stray far decode does not draw the outline)."""
    placed = table.filter(pl.col("range_nm").is_not_null()).with_columns(
        bearing_lo=(pl.col("bearing") // BEARING_STEP * BEARING_STEP).cast(pl.Int32)
    )
    parts = [
        placed.filter(pl.col(valid(r)) & pl.col(heard(r)))
        .group_by("bearing_lo")
        .agg(range_nm=pl.col("range_nm").quantile(quantile))
        .with_columns(receiver=pl.lit(r))
        for r in receivers
    ]
    return (
        pl.concat(parts)
        .select("receiver", "bearing_lo", "range_nm")
        .sort("receiver", "bearing_lo")
    )


def altitude_by_range(
    table: pl.DataFrame, receivers: list[str], min_n: int = 30
) -> pl.DataFrame:
    """receiver, range_lo (nm), alt_lo (ft), n, rate."""
    placed = table.filter(
        pl.col("range_nm").is_not_null() & pl.col("alt_ft").is_not_null()
    ).with_columns(
        range_lo=(pl.col("range_nm") // RANGE_STEP_NM * RANGE_STEP_NM).cast(pl.Int32),
        alt_lo=(pl.col("alt_ft") // ALTITUDE_STEP_FT * ALTITUDE_STEP_FT).cast(pl.Int32),
    )
    parts = [
        placed.filter(pl.col(valid(r)))
        .group_by("range_lo", "alt_lo")
        .agg(n=pl.len(), rate=pl.col(heard(r)).mean())
        .with_columns(receiver=pl.lit(r))
        for r in receivers
    ]
    return (
        pl.concat(parts)
        .filter(pl.col("n") >= min_n)
        .select("receiver", "range_lo", "alt_lo", "n", "rate")
        .sort("receiver", "range_lo", "alt_lo")
    )
