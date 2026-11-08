"""The analysis table: one row per transmission, everything needed to score it.

Columns, besides the transmission's own (``tx``, ``ref_segment``, ``t_ref``,
``df``, ``address``, ``msg_len``, ...):

- ``t``: seconds on the reference clock (precise, but only comparable
  within one reference segment of one analysis window); ``wall_s``: the
  recorder's earliest arrival time for it, in Unix seconds (coarse, but
  comparable across the whole run); ``minute_ns``: the wall-clock minute.
- ``heard:<r>``, ``signal:<r>``, ``valid:<r>`` for every receiver: whether
  it decoded the transmission, its signal byte, and whether the receiver
  was valid that minute.
- ``lat``, ``lon``, ``alt_ft``, ``range_nm``, ``bearing``, ``elevation``:
  where the aircraft was (null where unknown).
- ``density``: reference transmissions in the same second.
- ``visit`` (added by :func:`add_visits` over the whole run): the aircraft
  visit, a continuous presence of one address in the reference set with no
  gap longer than ``visit_gap_s``. Visits are the unit the confidence
  intervals resample. A transmission without an address (rare: formats
  with no address field) is a visit of its own, so it is still scored.
- ``position``: whether the transmission is a decoded airborne position.
- ``df_group``: index into ``DF_GROUPS``.
"""

from __future__ import annotations

import polars as pl

from beast_compare.align import NOMINAL_HZ
from beast_compare.cluster import Transmissions
from beast_compare.validity import MINUTE_NS

VISIT_GAP_S = 300.0

# Message types, in the order of ``df_group``.
DF_GROUPS = ("DF0/16", "DF4/5", "DF11", "DF17/18", "DF20/21", "Comm-D")
_DF_GROUP = {0: 0, 16: 0, 4: 1, 5: 1, 11: 2, 17: 3, 18: 3, 20: 4, 21: 4}


def heard(receiver: str) -> str:
    return f"heard:{receiver}"


def signal(receiver: str) -> str:
    return f"signal:{receiver}"


def valid(receiver: str) -> str:
    return f"valid:{receiver}"


def build_table(
    clustered: Transmissions,
    attributed: pl.DataFrame,
    positions: pl.DataFrame,
    validity_table: pl.DataFrame,
    receivers: list[str],
) -> pl.DataFrame:
    detections = clustered.detections
    table = clustered.transmissions.with_columns(t=pl.col("t_ref") / NOMINAL_HZ)

    minute = detections.group_by("tx").agg(
        wall_s=pl.col("arrival_ns").min() / 1e9,
        minute_ns=pl.col("arrival_ns").min() // MINUTE_NS * MINUTE_NS,
    )
    table = table.join(minute, on="tx", how="left")

    for receiver in receivers:
        own = detections.filter(pl.col("receiver") == receiver).select(
            "tx", pl.col("signal").alias(signal(receiver))
        )
        table = table.join(own, on="tx", how="left").with_columns(
            pl.col(signal(receiver)).is_not_null().alias(heard(receiver))
        )
        minutes = validity_table.filter(pl.col("receiver") == receiver).select(
            "minute_ns", pl.col("valid").alias(valid(receiver))
        )
        table = table.join(minutes, on="minute_ns", how="left").with_columns(
            pl.col(valid(receiver)).fill_null(False)
        )

    table = table.join(attributed, on="tx", how="left")
    table = table.with_columns(
        df_group=pl.col("df").replace_strict(
            _DF_GROUP, default=5, return_dtype=pl.UInt8
        ),
        position=pl.col("tx").is_in(positions["tx"].implode()),
        density=pl.len().over("ref_segment", pl.col("t").floor()).cast(pl.UInt32),
    )
    return table.sort("tx")


def add_visits(table: pl.DataFrame, visit_gap_s: float = VISIT_GAP_S) -> pl.DataFrame:
    """Number aircraft visits over the whole run, on wall-clock time."""
    ordered = table.with_row_index("_row")
    addressed = ordered.filter(pl.col("address").is_not_null()).sort(
        "address", "wall_s"
    )
    gap = pl.col("wall_s").diff().over("address")
    starts = (gap > visit_gap_s).fill_null(True)
    visits = addressed.select(
        "_row", visit=starts.cast(pl.UInt32).cum_sum().cast(pl.UInt32) - 1
    )
    joined = ordered.join(visits, on="_row", how="left").sort("_row")
    first_alone = len(visits)
    alone = pl.col("visit").is_null().cast(pl.UInt32).cum_sum() - 1 + first_alone
    return joined.with_columns(
        visit=pl.coalesce(pl.col("visit"), alone).cast(pl.UInt32)
    ).drop("_row")
