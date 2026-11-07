import polars as pl
import pytest
from tables import RECEIVERS, Tx, table

from beast_compare.visits import VisitSettings, visit_metrics

SETTINGS = VisitSettings(edge_s=100, tie_s=1, bin_s=10, min_duration_s=60)


def run() -> pl.DataFrame:
    rows = [
        # Window edges: transmissions at t=0 and t=2000 from another visit
        # so that the visits below are interior.
        Tx(t=0, visit=9, heard={"a": 1, "b": 1}),
        Tx(t=2000, visit=9, heard={"a": 1, "b": 1}),
    ]
    # Visit 0, t=500..800: a first at 500, b at 530; b last at 800, a at 760.
    # Positions every 5 s; a hears all of them, b only after 600.
    for t in range(500, 801, 5):
        rows.append(
            Tx(
                t=t,
                visit=0,
                heard={
                    "a": 1 if t <= 760 else None,
                    "b": 1 if t >= 530 and (t >= 600 or t % 10 == 0) else None,
                },
                position=True,
                range_nm=100.0 + (t - 500) / 10,
            )
        )
    # Visit 1, t=1000..1050 (too short for continuity): a tie throughout.
    for second in (1000.0, 1000.5, 1030.0, 1050.0):
        rows.append(Tx(t=second, visit=1, heard={"a": 1, "b": 1}, position=True))
    return table(rows)


def test_acquisition_and_fade() -> None:
    metrics = visit_metrics(run(), RECEIVERS, SETTINGS)
    acquisition = {r["receiver"]: r for r in metrics.acquisition.iter_rows(named=True)}
    assert acquisition["a"]["visits"] == 2
    assert acquisition["a"]["best"] == 2
    assert acquisition["b"]["best"] == 1  # the tie at 1000/1000.5 counts for b
    assert acquisition["b"]["median_lag_s"] == pytest.approx((30 + 0) / 2)

    pair = metrics.acquisition_pairs.row(0, named=True)
    assert (pair["a_better"], pair["b_better"], pair["ties"]) == (1, 0, 1)

    fade = {r["receiver"]: r for r in metrics.fade.iter_rows(named=True)}
    assert fade["b"]["best"] == 2
    assert fade["a"]["median_lag_s"] == pytest.approx((40 + 0) / 2)

    events = metrics.events.filter((pl.col("visit") == 0) & (pl.col("receiver") == "a"))
    assert events["first_range"][0] == pytest.approx(100.0)


def test_edge_visits_are_excluded() -> None:
    metrics = visit_metrics(run(), RECEIVERS, SETTINGS)
    assert 9 not in metrics.events["visit"].to_list()


def test_continuity() -> None:
    metrics = visit_metrics(run(), RECEIVERS, SETTINGS)
    rows = {r["receiver"]: r for r in metrics.continuity.iter_rows(named=True)}
    # Only visit 0 lasts 60 s. Its 31 bins of 10 s: a covers 500-760 (27),
    # b covers 530-800 (28).
    assert rows["a"]["visits"] == 1
    assert rows["a"]["median_coverage"] == pytest.approx(27 / 31)
    assert rows["b"]["median_coverage"] == pytest.approx(28 / 31)
    assert rows["a"]["median_of_reference"] == pytest.approx(27 / 31)
