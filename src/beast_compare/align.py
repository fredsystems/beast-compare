"""Put every receiver's frames on the reference receiver's clock.

Each decoder stamps frames with its own counter (normally 12 MHz), and the
counters drift against each other by up to a few hundred ppm. Matching the
same transmission across receivers needs them on one timeline to within a
few microseconds, so every other receiver is mapped onto the reference's
counter.

Anchors are long messages that occur exactly once in a chunk on both
receivers: almost always the same transmission. Not always, though. Two
transmissions of identical content, each heard by only one receiver, pair
up wrongly and are off by milliseconds to seconds, which is why the fit is
robust: per window, a median gate first, then a tight gate around a first
fit. The centroids of the surviving anchors in each window are the knots of
a piecewise-linear map from the receiver's counter to the reference's.

A decoder restart resets its counter, so each receiver's frames are split
into segments wherever the counter goes backwards or jumps far beyond the
wall-clock time between frames. A segment is aligned only where it overlaps
a reference segment, and a frame more than ``max_anchor_distance_s`` from
the nearest knot is left unaligned rather than extrapolated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import polars as pl

NOMINAL_HZ = 12_000_000

Floats = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class AlignmentSettings:
    window_s: float = 1.0
    min_pairs: int = 5
    coarse_tolerance_us: float = 1000.0
    fine_tolerance_us: float = 5.0
    # On real captures, knots 20 s apart still interpolated held-out anchors
    # to 0.05 us (median) and 0.19 us (p99); the counters drift linearly.
    max_anchor_distance_s: float = 10.0
    chunk_s: float = 600.0
    max_arrival_skew_s: float = 2.0
    restart_slack_s: float = 10.0


@dataclass(frozen=True, slots=True)
class Alignment:
    """``frames`` gains ``segment``, ``ref_segment`` and ``t_ref``.

    ``t_ref`` is the frame's time in reference counter ticks, null where the
    frame could not be aligned. ``anchors`` holds every anchor that survived
    both gates, with its residual against the final map; ``knots`` holds the
    map itself.
    """

    reference: str
    frames: pl.DataFrame
    anchors: pl.DataFrame
    knots: pl.DataFrame


def add_segments(
    frames: pl.DataFrame, settings: AlignmentSettings = AlignmentSettings()
) -> pl.DataFrame:
    """Number each receiver's clock segments from 0, in arrival order."""
    slack_ticks = settings.restart_slack_s * NOMINAL_HZ
    ordered = frames.with_row_index("_row").sort("receiver", "arrival_ns", "_row")
    step = pl.col("counter").diff().over("receiver")
    elapsed = pl.col("arrival_ns").diff().over("receiver") * (NOMINAL_HZ / 1e9)
    # A restarted decoder's counter starts again near zero. Small backward
    # steps are tolerated so that slightly reordered output never splits.
    backwards = step < -NOMINAL_HZ
    restart = (backwards | (step - elapsed > slack_ticks)).fill_null(False)
    return ordered.with_columns(
        segment=restart.cast(pl.UInt32).cum_sum().over("receiver")
    ).drop("_row")


def _unique_long(frames: pl.DataFrame, chunk_ns: int) -> pl.DataFrame:
    return (
        frames.filter(pl.col("msg_len") == 14)
        .with_columns(chunk=pl.col("arrival_ns") // chunk_ns)
        .group_by("chunk", "msg_hi", "msg_lo")
        .agg(
            pl.len().alias("n"),
            pl.col("counter").first(),
            pl.col("arrival_ns").first(),
        )
        .filter(pl.col("n") == 1)
        .drop("n")
    )


def anchor_pairs(
    reference: pl.DataFrame,
    other: pl.DataFrame,
    settings: AlignmentSettings = AlignmentSettings(),
) -> pl.DataFrame:
    """Candidate anchors: ``x`` reference ticks, ``y`` the other's ticks."""
    chunk_ns = int(settings.chunk_s * 1e9)
    skew_ns = int(settings.max_arrival_skew_s * 1e9)
    return (
        _unique_long(reference, chunk_ns)
        .join(
            _unique_long(other, chunk_ns),
            on=["chunk", "msg_hi", "msg_lo"],
            suffix="_other",
        )
        .filter((pl.col("arrival_ns") - pl.col("arrival_ns_other")).abs() < skew_ns)
        .select(
            x=pl.col("counter").cast(pl.Float64),
            y=pl.col("counter_other").cast(pl.Float64),
        )
        .sort("x")
    )


def _centroids(
    x: Floats, y: Floats, window: npt.NDArray[np.int64], keep: npt.NDArray[np.bool_]
) -> tuple[Floats, Floats, npt.NDArray[np.int64]]:
    knots = (
        pl.DataFrame({"w": window[keep], "x": x[keep], "y": y[keep]})
        .group_by("w")
        .agg(pl.col("x").mean(), pl.col("y").mean(), pl.len().alias("n"))
        .sort("w")
    )
    return (
        knots["x"].to_numpy().astype(np.float64),
        knots["y"].to_numpy().astype(np.float64),
        knots["n"].to_numpy().astype(np.int64),
    )


def _drop_thin(
    cx: Floats, cy: Floats, n: npt.NDArray[np.int64], min_pairs: int
) -> tuple[Floats, Floats, npt.NDArray[np.int64]]:
    enough = n >= min_pairs
    return cx[enough], cy[enough], n[enough]


def piecewise_map(values: Floats, src: Floats, dst: Floats, max_gap: float) -> Floats:
    """Map ``values`` through the knots ``src -> dst``.

    Linear between knots, extended linearly past the ends, and NaN for any
    value further than ``max_gap`` from its nearest knot.
    """
    out = np.full(values.shape, np.nan)
    if len(src) < 2:
        return out
    out = np.interp(values, src, dst)
    below = values < src[0]
    above = values > src[-1]
    out[below] = dst[0] + (values[below] - src[0]) * (dst[1] - dst[0]) / (
        src[1] - src[0]
    )
    out[above] = dst[-1] + (values[above] - src[-1]) * (dst[-1] - dst[-2]) / (
        src[-1] - src[-2]
    )
    right = np.clip(np.searchsorted(src, values), 0, len(src) - 1)
    left = np.clip(right - 1, 0, len(src) - 1)
    distance = np.minimum(np.abs(values - src[left]), np.abs(values - src[right]))
    out[distance > max_gap] = np.nan
    return out


@dataclass(frozen=True, slots=True)
class SegmentFit:
    """Knots mapping one receiver segment onto one reference segment."""

    ref_knots: Floats
    knots: Floats
    pairs: npt.NDArray[np.int64]
    anchor_x: Floats
    anchor_y: Floats
    residual_us: Floats


def fit_segment(
    pairs: pl.DataFrame, settings: AlignmentSettings = AlignmentSettings()
) -> SegmentFit:
    x = pairs["x"].to_numpy().astype(np.float64)
    y = pairs["y"].to_numpy().astype(np.float64)
    empty = np.zeros(0)
    if len(x) == 0:
        return SegmentFit(empty, empty, np.zeros(0, np.int64), empty, empty, empty)

    window = np.floor((x - x[0]) / (settings.window_s * NOMINAL_HZ)).astype(np.int64)
    ticks_per_us = NOMINAL_HZ / 1e6
    max_gap = settings.max_anchor_distance_s * NOMINAL_HZ

    # Pass 1: per window, drop anchors far from the window's median offset.
    offset = y - x
    median = (
        pl.DataFrame({"w": window, "r": offset})
        .with_columns(m=pl.col("r").median().over("w"))["m"]
        .to_numpy()
    )
    coarse = np.abs(offset - median) < settings.coarse_tolerance_us * ticks_per_us
    cx, cy, n = _drop_thin(*_centroids(x, y, window, coarse), settings.min_pairs)

    # Pass 2: drop anchors off the first map by more than the fine tolerance.
    first = piecewise_map(x, cx, cy, max_gap)
    fine = np.abs(y - first) < settings.fine_tolerance_us * ticks_per_us
    cx, cy, n = _drop_thin(*_centroids(x, y, window, fine), settings.min_pairs)

    final = piecewise_map(x, cx, cy, max_gap)
    kept = fine & ~np.isnan(final)
    residual = (y[kept] - final[kept]) / ticks_per_us
    return SegmentFit(cx, cy, n, x[kept], y[kept], residual)


def align(
    frames: pl.DataFrame,
    reference: str,
    settings: AlignmentSettings = AlignmentSettings(),
) -> Alignment:
    segmented = add_segments(frames, settings)
    max_gap = settings.max_anchor_distance_s * NOMINAL_HZ

    is_reference = pl.col("receiver") == reference
    ref_frames = segmented.filter(is_reference)
    ref_spans = (
        ref_frames.group_by("segment")
        .agg(
            pl.col("arrival_ns").min().alias("start"),
            pl.col("arrival_ns").max().alias("end"),
        )
        .sort("segment")
    )

    pieces = [
        ref_frames.with_columns(
            ref_segment=pl.col("segment"),
            t_ref=pl.col("counter").cast(pl.Float64),
        )
    ]
    anchors: list[pl.DataFrame] = []
    knots: list[pl.DataFrame] = []

    others = segmented.filter(~is_reference)
    for (receiver, segment), group in others.group_by(
        "receiver", "segment", maintain_order=True
    ):
        t_ref = np.full(len(group), np.nan)
        ref_segment = np.full(len(group), -1, dtype=np.int64)
        arrival = group["arrival_ns"].to_numpy()
        counter = group["counter"].to_numpy().astype(np.float64)
        for ref_seg, start, end in ref_spans.iter_rows():
            inside = (arrival >= start) & (arrival <= end)
            if not inside.any():
                continue
            ref_part = ref_frames.filter(pl.col("segment") == ref_seg)
            fit = fit_segment(anchor_pairs(ref_part, group, settings), settings)
            mapped = piecewise_map(counter[inside], fit.knots, fit.ref_knots, max_gap)
            t_ref[inside] = mapped
            ref_segment[inside] = np.where(np.isnan(mapped), -1, ref_seg)
            ids = {"receiver": receiver, "segment": segment, "ref_segment": ref_seg}
            anchors.append(
                pl.DataFrame(
                    {
                        "ref_tick": fit.anchor_x,
                        "tick": fit.anchor_y,
                        "residual_us": fit.residual_us,
                    }
                ).with_columns(**{k: pl.lit(v) for k, v in ids.items()})
            )
            knots.append(
                pl.DataFrame(
                    {"ref_tick": fit.ref_knots, "tick": fit.knots, "pairs": fit.pairs}
                ).with_columns(**{k: pl.lit(v) for k, v in ids.items()})
            )
        pieces.append(
            group.with_columns(
                pl.Series("ref_segment", ref_segment), pl.Series("t_ref", t_ref)
            ).with_columns(
                ref_segment=pl.when(pl.col("ref_segment") >= 0)
                .then(pl.col("ref_segment"))
                .cast(pl.UInt32),
                t_ref=pl.col("t_ref").fill_nan(None),
            )
        )

    id_schema = pl.Schema(
        {
            "receiver": frames.schema["receiver"],
            "segment": pl.UInt32(),
            "ref_segment": pl.UInt32(),
        }
    )
    anchor_schema = pl.Schema(
        {"ref_tick": pl.Float64(), "tick": pl.Float64(), "residual_us": pl.Float64()}
    )
    knot_schema = pl.Schema(
        {"ref_tick": pl.Float64(), "tick": pl.Float64(), "pairs": pl.Int64()}
    )
    anchor_schema.update(id_schema)
    knot_schema.update(id_schema)
    anchor_frame = pl.concat(
        [pl.DataFrame(schema=anchor_schema), *[a.cast(id_schema) for a in anchors]]
    )
    knot_frame = pl.concat(
        [pl.DataFrame(schema=knot_schema), *[k.cast(id_schema) for k in knots]]
    )
    return Alignment(
        reference=reference,
        frames=pl.concat(pieces).sort("receiver", "arrival_ns", maintain_order=True),
        anchors=anchor_frame,
        knots=knot_frame,
    )
