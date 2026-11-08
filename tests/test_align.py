from pathlib import Path

import numpy as np
import polars as pl
from frames import HZ, Row, arrival, frames
from runs import fixture_records, make_run

from beast_compare.align import add_segments, align, piecewise_map
from beast_compare.ingest import load_run

RECEIVERS = ["ref", "other"]


def synthetic(seed: int = 7) -> tuple[pl.DataFrame, dict[int, float]]:
    """Two receivers hearing the same transmissions with different clocks.

    ``other`` runs 3 ppm fast with a slight curve, and its decoder restarts
    at t = 60 s. Each receiver misses 20 % of transmissions. Some content is
    transmitted twice, with each receiver hearing a different copy: those
    pair up as anchors and must be rejected.
    """
    rng = np.random.default_rng(seed)
    rows: list[Row] = []
    truth: dict[int, float] = {}

    def other_counter(t: float) -> int:
        if t < 60:
            ticks = 5_000_000_000 + t * HZ * (1 + 3e-6) + 2e-9 * t * t * HZ
        else:
            ticks = 1_000 + (t - 60) * HZ * (1 + 3e-6)
        return int(round(ticks + rng.normal(0, 0.03 * 12)))

    def ref_counter(t: float) -> int:
        return int(round(1_000_000 + t * HZ + rng.normal(0, 0.03 * 12)))

    times = np.sort(rng.uniform(0, 120, 6000))
    for i, t in enumerate(times):
        truth[i] = 1_000_000 + t * HZ
        if rng.random() > 0.2:
            rows.append(Row("ref", arrival(t + 0.01), ref_counter(t), i))
        if rng.random() > 0.2:
            rows.append(Row("other", arrival(t + 0.02), other_counter(t), i))

    for k in range(300):
        first = rng.uniform(0, 115)
        second = first + rng.uniform(0.5, 3)
        content = 10**6 + k
        rows.append(Row("ref", arrival(first + 0.01), ref_counter(first), content))
        rows.append(
            Row("other", arrival(second + 0.02), other_counter(second), content)
        )
        truth[content] = 1_000_000 + second * HZ

    return frames(rows, RECEIVERS), truth


def test_segments() -> None:
    data, _ = synthetic()
    segmented = add_segments(data)
    counts = dict(
        segmented.group_by("receiver").agg(pl.col("segment").n_unique()).iter_rows()
    )
    assert counts == {"ref": 1, "other": 2}


def test_recorder_gap_is_not_a_restart() -> None:
    rows = [
        Row("ref", arrival(0), 0, 1),
        # 30 s with nothing recorded, counter consistent with it.
        Row("ref", arrival(30), 30 * HZ, 2),
        # Counter 20 s ahead of the wall clock: a different decoder run.
        Row("ref", arrival(31), 51 * HZ, 3),
    ]
    segmented = add_segments(frames(rows, RECEIVERS))
    assert segmented["segment"].to_list() == [0, 0, 1]


def test_align_recovers_the_reference_clock() -> None:
    data, truth = synthetic()
    result = align(data, "ref")
    other = result.frames.filter(pl.col("receiver") == "other")

    aligned = other.filter(pl.col("t_ref").is_not_null())
    assert len(aligned) / len(other) > 0.97

    expected = np.array([truth[m] for m in aligned["msg_hi"].to_list()])
    error_us = (aligned["t_ref"].to_numpy() - expected) / 12
    assert np.abs(error_us).max() < 0.5

    assert np.abs(result.anchors["residual_us"].to_numpy()).max() < 1
    # Both of other's segments were fitted against ref's only segment.
    assert set(result.knots["segment"].to_list()) == {0, 1}


def test_reference_frames_keep_their_counter() -> None:
    data, _ = synthetic()
    ref = align(data, "ref").frames.filter(pl.col("receiver") == "ref")
    assert (ref["t_ref"] == ref["counter"].cast(pl.Float64)).all()
    assert (ref["ref_segment"] == 0).all()


def test_piecewise_map() -> None:
    src = np.array([0.0, 10.0, 20.0])
    dst = np.array([100.0, 110.0, 130.0])
    values = np.array([-2.0, 5.0, 15.0, 22.0, 50.0])
    out = piecewise_map(values, src, dst, max_gap=5.0)
    assert out[:4].tolist() == [98.0, 105.0, 120.0, 134.0]
    assert np.isnan(out[4])
    assert np.isnan(piecewise_map(values, src[:1], dst[:1], max_gap=5.0)).all()


def test_real_captures_align(tmp_path: Path) -> None:
    config = make_run(
        tmp_path,
        {
            "airspy": fixture_records("airspy-60s"),
            "fa-proplus": fixture_records("faproplus-60s"),
        },
    )
    result = align(load_run(config.run_dir).frames, "airspy")
    other = result.frames.filter(pl.col("receiver") == "fa-proplus")
    assert other["t_ref"].is_not_null().sum() / len(other) > 0.95
    residual = result.anchors["residual_us"].to_numpy()
    assert len(residual) > 3000
    assert np.median(np.abs(residual)) < 0.1
