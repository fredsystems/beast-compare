"""Regression tests for the review of the analysis pipeline (PR #2)."""

from pathlib import Path

import numpy as np
import polars as pl
import pytest
from frames import HZ
from runs import fixture_records, make_run
from tables import RECEIVERS, Tx, table
from test_validity import minute, write_jsonl

from beast_compare.analysis import windows
from beast_compare.geo import Site
from beast_compare.ingest import load_run
from beast_compare.metrics import RANGE, Resampler, _crossing, rates_by
from beast_compare.positions import attribute
from beast_compare.spatial import range_outline
from beast_compare.table import add_visits
from beast_compare.validity import MINUTE_NS, validity
from beast_compare.visits import VisitSettings, visit_metrics


def test_windows_are_whole_minutes_and_validated() -> None:
    start = 90 * 10**9  # 1.5 minutes in
    parts = windows(start, start + 5 * MINUTE_NS, 120)
    assert parts[0][0] == start
    assert all(edge % MINUTE_NS == 0 for _, edge in parts[:-1])
    assert parts[-1][1] == start + 5 * MINUTE_NS
    with pytest.raises(ValueError, match="minute"):
        windows(0, MINUTE_NS, 10)
    with pytest.raises(ValueError, match="before"):
        windows(MINUTE_NS, 0, 60)


def test_crossing_checks_the_first_bin_first() -> None:
    curve = [(-40.0, 0.8, 100), (-30.0, 0.2, 100), (-20.0, 0.7, 100)]
    assert _crossing(curve, 0.5) == -40.0


def test_two_disconnects_in_a_minute_do_not_duplicate_rows(tmp_path: Path) -> None:
    write_jsonl(
        tmp_path / "events.jsonl",
        [
            {"time_ns": minute(0), "receiver": "a", "event": "connected"},
            {"time_ns": minute(0), "receiver": "b", "event": "connected"},
            {"time_ns": minute(1.1), "receiver": "a", "event": "disconnected"},
            {"time_ns": minute(1.2), "receiver": "a", "event": "connected"},
            {"time_ns": minute(1.5), "receiver": "a", "event": "disconnected"},
            {"time_ns": minute(1.6), "receiver": "a", "event": "connected"},
        ],
    )
    empty = pl.DataFrame(
        schema={
            "receiver": pl.Enum(RECEIVERS),
            "arrival_ns": pl.Int64,
            "t_ref": pl.Float64,
        }
    )
    result = validity(tmp_path, empty, RECEIVERS, minute(0), minute(3))
    assert len(result) == 6
    assert result.filter(pl.col("receiver") == "a")["valid"].to_list() == [
        True,
        False,
        True,
    ]


def test_addressless_transmissions_get_their_own_visits() -> None:
    frame = pl.DataFrame(
        {"address": [1, None, 1, None], "wall_s": [0.0, 1.0, 2.0, 3.0]},
        schema_overrides={"address": pl.UInt32},
    )
    visits = add_visits(frame)["visit"].to_list()
    assert visits[0] == visits[2]
    assert len({visits[0], visits[1], visits[3]}) == 3
    assert None not in visits


def test_comm_d_is_addressed(tmp_path: Path) -> None:
    config = make_run(tmp_path, {"airspy": fixture_records("airspy-60s")})
    frames = load_run(config.run_dir).frames
    comm_d = frames.filter(pl.col("df") == 24)
    known = frames.filter(pl.col("df").is_in((11, 17)))["address"].implode()
    assert comm_d["address"].null_count() == 0
    # As for DF5/21: a few are aircraft that sent no DF11/17 that minute.
    assert comm_d["address"].is_in(known).sum() / len(comm_d) > 0.95


def test_interpolation_across_the_antimeridian() -> None:
    track = pl.DataFrame(
        {
            "ref_segment": [0, 0],
            "address": [1, 1],
            "t": [0.0, 10.0],
            "alt_ft": [30_000.0, 30_000.0],
            "lat": [0.0, 0.0],
            "lon": [179.9, -179.9],
        },
        schema_overrides={"ref_segment": pl.UInt32, "address": pl.UInt32},
    )
    transmissions = pl.DataFrame(
        {"tx": [1], "ref_segment": [0], "address": [1], "t_ref": [5.0 * HZ]},
        schema_overrides={"ref_segment": pl.UInt32, "address": pl.UInt32},
    )
    placed = attribute(transmissions, track, Site(0.0, 179.0, 0.0))
    assert abs(placed["lon"][0]) == pytest.approx(180.0, abs=1e-6)


def test_acquisition_range_is_the_first_rows() -> None:
    rows = [
        Tx(t=0, visit=9, heard={"a": 1, "b": 1}),
        Tx(t=2000, visit=9, heard={"a": 1, "b": 1}),
        # First detection could not be placed; the next one could.
        Tx(t=500, visit=0, heard={"a": 1, "b": None}, range_nm=None),
        Tx(t=510, visit=0, heard={"a": 1, "b": 1}, range_nm=120.0),
    ]
    metrics = visit_metrics(table(rows), RECEIVERS, VisitSettings(edge_s=100))
    a = metrics.events.filter((pl.col("visit") == 0) & (pl.col("receiver") == "a"))
    assert a["first_range"][0] is None


def test_continuity_excludes_visits_with_an_outage_in_short_replies() -> None:
    rows = [
        Tx(t=0, visit=9, heard={"a": 1, "b": 1}),
        Tx(t=2000, visit=9, heard={"a": 1, "b": 1}),
    ]
    rows += [
        Tx(t=t, visit=0, heard={"a": 1, "b": 1}, position=True)
        for t in (500.0, 510.0, 620.0, 630.0)
    ]
    # b is invalid at 570 s, when only a DF11 was heard.
    rows.append(
        Tx(t=570, visit=0, heard={"a": 1, "b": None}, valid={"a": True, "b": False})
    )
    metrics = visit_metrics(table(rows), RECEIVERS, VisitSettings(edge_s=100))
    assert metrics.continuity["visits"].to_list() == [0, 0]


def test_range_outline_needs_enough_decodes() -> None:
    rows = [Tx(t=i, visit=i, heard={"a": 1, "b": None}) for i in range(5)]
    assert range_outline(table(rows), RECEIVERS, min_n=30).is_empty()
    assert len(range_outline(table(rows), RECEIVERS, min_n=5)) == 1


def test_intervals_are_reproducible() -> None:
    rows = [
        Tx(
            t=i,
            visit=i // 3,
            heard={"a": 1, "b": 1 if i % 4 else None},
            range_nm=float(i % 60),
        )
        for i in range(600)
    ]
    data = table(rows)
    first = rates_by(data, RANGE, RECEIVERS, Resampler(resamples=50))
    second = rates_by(data, RANGE, RECEIVERS, Resampler(resamples=50))
    assert first.equals(second)
    assert not np.isnan(first["low"].to_numpy()).any()
