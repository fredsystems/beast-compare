from pathlib import Path

import polars as pl
import pytest
from tables import RECEIVERS, Tx, table

from beast_compare.metrics import (
    RANGE,
    Resampler,
    comparability,
    dbfs,
    exclusives,
    rates_by,
    sensitivity,
    timing,
    wins_by,
)


def two_bands() -> pl.DataFrame:
    """40 visits of 10 transmissions each. Within 25 nm a hears everything
    and b every other one; beyond, a hears everything and b nothing. One
    minute where b is invalid holds transmissions b would have missed."""
    rows = []
    for visit in range(40):
        near = visit < 20
        for i in range(10):
            b_hears = near and i % 2 == 0
            rows.append(
                Tx(
                    t=visit * 100 + i,
                    visit=visit,
                    heard={"a": 100, "b": 50 if b_hears else None},
                    range_nm=10.0 if near else 30.0,
                )
            )
    rows += [
        Tx(t=9000 + i, visit=99, heard={"a": 100, "b": None}, valid={"b": False})
        for i in range(10)
    ]
    return table(rows)


def test_rates_by_band() -> None:
    rates = rates_by(two_bands(), RANGE, RECEIVERS, Resampler())
    got = {
        (r, band): rate
        for r, band, rate in rates.select("receiver", "band", "rate").iter_rows()
    }
    assert got[("a", "0-25 nm")] == 1.0
    assert got[("b", "0-25 nm")] == 0.5
    assert got[("b", "25-50 nm")] == 0.0
    # b's invalid minute is not counted against it; a's is counted for a.
    n = {
        (r, band): n for r, band, n in rates.select("receiver", "band", "n").iter_rows()
    }
    assert n[("a", "0-25 nm")] == 210
    assert n[("b", "0-25 nm")] == 200
    row = rates.filter((pl.col("receiver") == "b") & (pl.col("band") == "0-25 nm"))
    assert row["low"][0] <= 0.5 <= row["high"][0]


def test_overall_rate() -> None:
    overall = rates_by(two_bands(), None, RECEIVERS, Resampler())
    assert overall.filter(pl.col("receiver") == "b")["rate"][0] == pytest.approx(
        100 / 400
    )


def test_wins_are_decided_only_with_a_clear_margin() -> None:
    wins = wins_by(two_bands(), RANGE, RECEIVERS, Resampler())
    near = wins.filter(pl.col("band") == "0-25 nm").row(0, named=True)
    assert near["best"] == "a"
    assert near["margin"] == pytest.approx(0.5)
    assert near["decided"]
    # b's invalid minute is excluded from the comparison entirely.
    assert near["n"] == 200

    tie = table([Tx(t=i, visit=i, heard={"a": 1, "b": 1}) for i in range(10)])
    row = wins_by(tie, RANGE, RECEIVERS, Resampler()).row(0, named=True)
    assert row["margin"] == 0
    assert not row["decided"]


def test_sensitivity_threshold() -> None:
    # b decodes exactly the transmissions a received at -20 dBFS or more.
    rows = []
    for visit in range(200):
        level = -40 + visit // 5  # -40 .. -1 dBFS at a
        byte = round(255 * 10 ** (level / 20))
        rows.append(
            Tx(
                t=visit,
                visit=visit,
                heard={"a": byte, "b": 9 if level >= -20 else None},
            )
        )
    # Repeat each transmission so every 1 dB bin passes the minimum count.
    many = table(rows * 12).with_columns(
        tx=pl.int_range(pl.len()), visit=pl.int_range(pl.len()).cast(pl.UInt32)
    )
    curves, thresholds = sensitivity(many, RECEIVERS, Resampler(resamples=20))
    b_by_a = thresholds.filter((pl.col("receiver") == "b") & (pl.col("by") == "a"))
    assert b_by_a["level_at_50_dbfs"][0] == pytest.approx(-21, abs=1)
    assert set(curves["by"].to_list()) == {"a", "b"}


def test_dbfs() -> None:
    frame = pl.DataFrame({"s": [255, 26, 0]})
    values = frame.select(dbfs("s").alias("dbfs"))["dbfs"].to_list()
    assert values[0] == pytest.approx(0.0)
    assert values[1] == pytest.approx(-19.83, abs=0.01)
    assert values[2] == pytest.approx(-48.13, abs=0.01)  # clipped to 1


def test_exclusives() -> None:
    rows = [
        Tx(t=0, visit=0, heard={"a": 1, "b": 1}),
        Tx(t=1, visit=0, heard={"a": 1, "b": None}, position=True),
        Tx(t=2, visit=1, heard={"a": None, "b": 1}),
        # Excluded: b invalid.
        Tx(t=120, visit=2, heard={"a": 1, "b": None}, valid={"b": False}),
    ]
    result = {
        r["receiver"]: r
        for r in exclusives(table(rows), RECEIVERS).iter_rows(named=True)
    }
    assert result["a"]["transmissions"] == 3
    assert (result["a"]["only"], result["a"]["missed_only"]) == (1, 1)
    assert result["a"]["positions_only"] == 1
    assert (result["b"]["only"], result["b"]["visits_only"]) == (1, 1)
    assert result["a"]["visits_only"] == 0


def test_timing_and_comparability() -> None:
    anchors = pl.DataFrame(
        {"receiver": ["b"] * 4, "residual_us": [0.0, 0.1, -0.1, 0.0]}
    )
    row = timing(anchors).row(0, named=True)
    assert row["anchors"] == 4
    assert row["mad_us"] == pytest.approx(0.05)

    validity = pl.DataFrame(
        {
            "receiver": ["a"] * 10 + ["b"] * 10,
            "valid": [True] * 10 + [True] * 8 + [False] * 2,
        }
    )
    shares = {
        r: c
        for r, c in comparability(validity).select("receiver", "comparable").iter_rows()
    }
    assert shares == {"a": True, "b": False}


def test_end_to_end_on_captures(tmp_path: Path) -> None:
    from runs import fixture_records, make_run

    from beast_compare.align import align
    from beast_compare.cluster import cluster
    from beast_compare.geo import Site
    from beast_compare.ingest import load_run
    from beast_compare.positions import attribute, decode_positions
    from beast_compare.table import add_visits, build_table
    from beast_compare.validity import validity

    config = make_run(
        tmp_path,
        {
            "airspy": fixture_records("airspy-60s"),
            "fa-proplus": fixture_records("faproplus-60s"),
        },
    )
    receivers = ["airspy", "fa-proplus"]
    frames = load_run(config.run_dir).frames
    aligned = align(frames, "airspy")
    clustered = cluster(aligned.frames)
    site = Site(35.0, -106.6, 1600.0)
    positions = decode_positions(clustered.transmissions, site)
    placed = attribute(clustered.transmissions, positions, site)
    start, end = frames["arrival_ns"].min(), frames["arrival_ns"].max()
    assert isinstance(start, int) and isinstance(end, int)
    minutes = validity(config.run_dir, aligned.frames, receivers, start, end + 1)
    # No events were recorded, so mark every minute valid for this smoke test.
    minutes = minutes.with_columns(valid=pl.lit(True))
    analysis = add_visits(build_table(clustered, placed, positions, minutes, receivers))
    overall = rates_by(analysis, None, receivers, Resampler(resamples=20))
    rate = dict(overall.select(pl.col("receiver").cast(pl.String), "rate").iter_rows())
    assert rate["airspy"] > rate["fa-proplus"] > 0.5
    assert analysis["visit"].null_count() < len(analysis) * 0.01
