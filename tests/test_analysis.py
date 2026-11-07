from pathlib import Path

import polars as pl
import pytest
from runs import BASE_NS, connect_all, fixture_records, make_run

from beast_compare.analysis import AnalysisSettings, analyse, windows
from beast_compare.cli import main
from beast_compare.geo import Site
from beast_compare.metrics import Resampler
from beast_compare.report import write_report
from beast_compare.table import heard

SITE = Site(35.0, -106.6, 1600.0)
RECEIVERS = ["airspy", "fa-proplus"]
START = BASE_NS - 10**9
END = BASE_NS + 120 * 10**9


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    config = make_run(
        tmp_path,
        {
            "airspy": fixture_records("airspy-60s"),
            "fa-proplus": fixture_records("faproplus-60s"),
        },
    )
    connect_all(config, START)
    return config.run_dir


def test_windows() -> None:
    assert windows(0, 25, 10e-9) == [(0, 10), (10, 20), (20, 25)]


def heard_counts(table: pl.DataFrame) -> dict[str, int]:
    return {r: int(table[heard(r)].sum()) for r in RECEIVERS}


def test_windowing_does_not_change_the_result(run_dir: Path) -> None:
    whole = analyse(run_dir, SITE, START, END)
    pieces = analyse(
        run_dir, SITE, START, END, settings=AnalysisSettings(window_s=20, overlap_s=5)
    )
    assert len(pieces.table) == pytest.approx(len(whole.table), rel=1e-3)
    for receiver, count in heard_counts(whole.table).items():
        assert heard_counts(pieces.table)[receiver] == pytest.approx(count, rel=1e-3)
    assert pieces.table["tx"].n_unique() == len(pieces.table)
    assert whole.frames.sort("receiver")["frames"].to_list() == [23340, 19128]


def test_report(run_dir: Path, tmp_path: Path) -> None:
    analysis = analyse(run_dir, SITE, START, END)
    path = write_report(analysis, tmp_path / "report", Resampler(resamples=20))
    page = path.read_text()
    for text in ("airspy", "fa-proplus", "Where each receiver wins", "Plotly"):
        assert text in page
    assert (tmp_path / "report" / "transmissions.parquet").exists()
    wins = pl.read_parquet(tmp_path / "report" / "wins_range.parquet")
    assert set(wins["best"].to_list()) <= set(RECEIVERS)


def test_unknown_reference(run_dir: Path) -> None:
    with pytest.raises(ValueError, match="reference"):
        analyse(run_dir, SITE, START, END, reference="nope")


def test_cli_compare(run_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "cli-report"
    code = main(
        [
            "compare",
            str(run_dir),
            "--lat",
            "35",
            "--lon",
            "-106.6",
            "--alt-m",
            "1600",
            "--start",
            "2026-11-07T15:59:59",
            "--end",
            "2026-11-07T16:02:00+00:00",
            "--resamples",
            "10",
            "--out",
            str(out),
        ]
    )
    assert code == 0
    assert (out / "report.html").exists()
