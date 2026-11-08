"""Group aligned frames into transmissions.

Frames from different receivers with identical message bytes, on the same
reference clock segment and within ``tolerance_us`` of each other on it, are
one transmission. Aligned timestamps agree to well under a microsecond (see
:mod:`beast_compare.align`), while the same content is never transmitted
twice within microseconds, so the default 2 us separates genuine repeats
(DF11 squitters, DF17 identification) with a wide margin either side.

The union of all transmissions is the reference set every receiver is
scored against: a transmission no receiver decoded is invisible.

A decoder occasionally outputs one transmission twice, a fraction of a
microsecond apart (readsb did, once in 19,000 frames of a capture). Such a
receiver counts once; the extra copies are kept in ``duplicates``.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from beast_compare.align import NOMINAL_HZ

_KEY = ("ref_segment", "msg_len", "msg_hi", "msg_lo")


@dataclass(frozen=True, slots=True)
class Transmissions:
    """``transmissions``: one row per transmission (``tx``), with its time
    on the reference clock, its decoded fields and how many receivers heard
    it. ``detections``: one row per (transmission, receiver) that decoded
    it. ``unaligned``: frames left out because they have no reference time.
    ``duplicates``: a receiver's second and later copies of a transmission.
    """

    transmissions: pl.DataFrame
    detections: pl.DataFrame
    unaligned: pl.DataFrame
    duplicates: pl.DataFrame

    def heard(self, receivers: list[str]) -> pl.DataFrame:
        """``tx`` plus one boolean column per receiver."""
        heard = self.transmissions.select("tx")
        for receiver in receivers:
            ids = self.detections.filter(pl.col("receiver") == receiver)["tx"]
            heard = heard.with_columns(
                pl.col("tx").is_in(ids.implode()).alias(receiver)
            )
        return heard


def cluster(aligned: pl.DataFrame, tolerance_us: float = 2.0) -> Transmissions:
    tolerance = tolerance_us * NOMINAL_HZ / 1e6
    has_time = pl.col("t_ref").is_not_null()
    frames = aligned.filter(has_time).sort(*_KEY, "t_ref")

    same_key = pl.all_horizontal(pl.col(k) == pl.col(k).shift(1) for k in _KEY)
    close = (pl.col("t_ref") - pl.col("t_ref").shift(1)) <= tolerance
    starts = ~(same_key & close).fill_null(False)
    frames = frames.with_columns(tx=starts.cast(pl.UInt64).cum_sum() - 1)

    transmissions = (
        frames.group_by("tx")
        .agg(
            pl.col("ref_segment").first(),
            pl.col("t_ref").mean(),
            pl.col("df").first(),
            pl.col("address").first(),
            pl.col("typecode").first(),
            pl.col("msg_len").first(),
            pl.col("msg_hi").first(),
            pl.col("msg_lo").first(),
            pl.col("receiver").n_unique().cast(pl.UInt32).alias("receivers"),
        )
        .sort("tx")
    )
    columns = ("tx", "receiver", "signal", "t_ref", "arrival_ns", "counter")
    ranked = frames.select(*columns).with_columns(
        copy=pl.int_range(pl.len()).over("tx", "receiver")
    )
    detections = ranked.filter(pl.col("copy") == 0).drop("copy").sort("tx", "receiver")
    duplicates = ranked.filter(pl.col("copy") > 0).drop("copy")
    return Transmissions(
        transmissions, detections, aligned.filter(~has_time), duplicates
    )
