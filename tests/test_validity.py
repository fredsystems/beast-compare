import json
from pathlib import Path
from typing import Any

import polars as pl
from frames import Row, frames
from runs import BASE_NS

from beast_compare.validity import MINUTE_NS, disconnected_spans, validity

RECEIVERS = ["airspy", "rtl"]


def minute(n: float) -> int:
    return BASE_NS + int(n * MINUTE_NS)


def write_jsonl(path: Path, lines: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))


def readsb(gain_db: float, dropped: int = 0) -> dict[str, Any]:
    return {"gain_db": gain_db, "last1min": {"local": {"samples_dropped": dropped}}}


def airspy(gain: int, lost: int = 0) -> dict[str, Any]:
    return {"gain": gain, "lost_buffers": lost}


def test_disconnected_spans() -> None:
    events: list[dict[str, Any]] = [
        {"time_ns": 10, "receiver": None, "event": "session_start"},
        {"time_ns": 20, "receiver": "airspy", "event": "connected"},
        {"time_ns": 25, "receiver": "rtl", "event": "connect_failed"},
        {"time_ns": 30, "receiver": "rtl", "event": "connected"},
        {"time_ns": 50, "receiver": "airspy", "event": "disconnected"},
        {"time_ns": 60, "receiver": "airspy", "event": "connected"},
        {"time_ns": 90, "receiver": None, "event": "session_stop"},
    ]
    assert sorted(disconnected_spans(events, RECEIVERS, 100)) == [
        ("airspy", 0, 20),
        ("airspy", 50, 60),
        ("airspy", 90, 100),
        ("rtl", 0, 30),
        ("rtl", 90, 100),
    ]


def test_validity(tmp_path: Path) -> None:
    write_jsonl(
        tmp_path / "events.jsonl",
        [
            {"time_ns": minute(0), "receiver": "airspy", "event": "connected"},
            {"time_ns": minute(0), "receiver": "rtl", "event": "connected"},
            # rtl drops out for part of minute 3.
            {"time_ns": minute(3.2), "receiver": "rtl", "event": "disconnected"},
            {"time_ns": minute(3.5), "receiver": "rtl", "event": "connected"},
        ],
    )
    # Each poll describes the minute before it.
    write_jsonl(
        tmp_path / "airspy" / "stats.jsonl",
        [
            {"time_ns": minute(n + 1), "stats": airspy(9, lost=int(n == 1))}
            for n in range(6)
        ],
    )
    write_jsonl(
        tmp_path / "rtl" / "stats.jsonl",
        [
            {"time_ns": minute(n + 1), "stats": readsb(0.0 if n == 4 else 12.5)}
            for n in range(5)
        ]
        + [{"time_ns": minute(6), "stats": None, "error": "refused"}],
    )
    # rtl's frames in minute 2 are mostly unaligned.
    rows = [Row("rtl", minute(2.1 + i / 100), 0, i) for i in range(10)]
    aligned = frames(rows, RECEIVERS).with_columns(
        t_ref=pl.Series([1.0] + [None] * 9, dtype=pl.Float64)
    )

    table = validity(tmp_path, aligned, RECEIVERS, minute(0), minute(6))
    invalid = {
        (r, (m - BASE_NS) // MINUTE_NS)
        for r, m, valid in table.select("receiver", "minute_ns", "valid").iter_rows()
        if not valid
    }
    assert invalid == {("airspy", 1), ("rtl", 2), ("rtl", 3), ("rtl", 4)}

    rtl = table.filter(pl.col("receiver") == "rtl")
    reasons = rtl.select("disconnected", "unaligned", "gain_floor").rows()
    assert reasons[2] == (False, True, False)
    assert reasons[3] == (True, False, False)
    assert reasons[4] == (False, False, True)
    assert rtl["stats_missing"].to_list() == [False] * 5 + [True]
