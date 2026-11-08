from collections.abc import Sequence
from pathlib import Path

import polars as pl
from frames import Row, arrival, frames
from runs import fixture_records, make_run

from beast_compare.align import align
from beast_compare.cluster import cluster
from beast_compare.ingest import load_run

RECEIVERS = ["a", "b", "c"]
US = 12  # reference ticks per microsecond


def aligned(rows: Sequence[tuple[str, float, int, int | None]]) -> pl.DataFrame:
    """Rows of (receiver, t_ref in us, content, ref_segment)."""
    data = frames(
        [Row(r, arrival(t / 1e6), int(t * US), m) for r, t, m, _ in rows], RECEIVERS
    )
    return data.with_columns(
        t_ref=pl.Series([t * US for _, t, _, _ in rows], dtype=pl.Float64),
        ref_segment=pl.Series([s for *_, s in rows], dtype=pl.UInt32),
    ).with_columns(
        t_ref=pl.when(pl.col("ref_segment").is_not_null()).then(pl.col("t_ref"))
    )


def test_groups_by_content_and_time() -> None:
    result = cluster(
        aligned(
            [
                # One transmission heard by all three, within 0.4 us.
                ("a", 1000.0, 1, 0),
                ("b", 1000.3, 1, 0),
                ("c", 999.9, 1, 0),
                # The same content again 300 us later, heard by two.
                ("a", 1300.0, 1, 0),
                ("c", 1300.2, 1, 0),
                # Different content at the same moment.
                ("b", 1000.1, 2, 0),
                # Same content and time, but another reference segment.
                ("a", 1000.0, 1, 1),
                # Unaligned.
                ("b", 5000.0, 3, None),
            ]
        )
    )
    tx = result.transmissions
    assert len(tx) == 4
    assert sorted(tx["receivers"].to_list()) == [1, 1, 2, 3]
    assert len(result.unaligned) == 1

    heard = result.heard(RECEIVERS).join(tx, on="tx")
    first = heard.filter((pl.col("msg_hi") == 1) & (pl.col("receivers") == 3))
    assert first.select(RECEIVERS).row(0) == (True, True, True)
    repeat = heard.filter((pl.col("msg_hi") == 1) & (pl.col("receivers") == 2))
    assert repeat.select(RECEIVERS).row(0) == (True, False, True)


def test_decoder_duplicate_counts_once() -> None:
    result = cluster(
        aligned([("a", 100.0, 1, 0), ("b", 100.1, 1, 0), ("b", 100.27, 1, 0)])
    )
    assert result.transmissions["receivers"].to_list() == [2]
    assert len(result.detections) == 2
    assert result.duplicates["receiver"].to_list() == ["b"]


def test_tolerance() -> None:
    rows = [("a", 0.0, 1, 0), ("b", 3.0, 1, 0)]
    assert len(cluster(aligned(rows)).transmissions) == 2
    assert len(cluster(aligned(rows), tolerance_us=5).transmissions) == 1


def test_real_captures(tmp_path: Path) -> None:
    config = make_run(
        tmp_path,
        {
            "airspy": fixture_records("airspy-60s"),
            "fa-proplus": fixture_records("faproplus-60s"),
        },
    )
    result = cluster(align(load_run(config.run_dir).frames, "airspy").frames)
    # A receiver is detected at most once per transmission; this capture
    # holds one readsb duplicate, 0.17 us after the original.
    per_receiver = result.detections.group_by("tx", "receiver").len()
    assert per_receiver["len"].max() == 1
    assert result.duplicates["receiver"].to_list() == ["fa-proplus"]
    heard = result.heard(["airspy", "fa-proplus"])
    fa = heard.filter(pl.col("fa-proplus"))
    # The Airspy hears nearly everything the Pro Stick Plus does.
    assert fa["airspy"].sum() / len(fa) > 0.9
