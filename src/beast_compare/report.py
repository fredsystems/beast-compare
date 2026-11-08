"""The report: every metric for an analysed window, as one HTML page and the
parquet tables it was built from."""

from __future__ import annotations

import datetime as dt
import html
from dataclasses import dataclass
from pathlib import Path

import plotly.offline
import polars as pl

from beast_compare import __version__, figures
from beast_compare.analysis import Analysis
from beast_compare.manifest import read_manifest
from beast_compare.metrics import (
    ALTITUDE,
    BEARING,
    DENSITY,
    DIMENSIONS,
    ELEVATION,
    MESSAGE_TYPE,
    RANGE,
    Resampler,
    comparability,
    exclusives,
    rates_by,
    sensitivity,
    timing,
    wins_by,
)
from beast_compare.spatial import (
    altitude_by_range,
    polar_cells,
    polar_difference,
    range_outline,
)
from beast_compare.validity import gain_series
from beast_compare.visits import visit_metrics

WIN_DIMENSIONS = (RANGE, ELEVATION, ALTITUDE, BEARING, MESSAGE_TYPE, DENSITY)


@dataclass(frozen=True, slots=True)
class Section:
    title: str
    body: str


def _when(ns: int) -> str:
    return dt.datetime.fromtimestamp(ns / 1e9, tz=dt.UTC).strftime("%Y-%m-%d %H:%M UTC")


def _cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4g}"
    return html.escape(str(value))


def table_html(frame: pl.DataFrame, percent: tuple[str, ...] = ()) -> str:
    head = "".join(f"<th>{html.escape(c)}</th>" for c in frame.columns)
    rows = []
    for row in frame.iter_rows(named=True):
        cells = []
        for column, value in row.items():
            is_pct = isinstance(value, float) and (
                column in percent or column.startswith("rate:")
            )
            text = f"{value:.1%}" if is_pct else _cell(value)
            cells.append(f"<td>{text}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return (
        f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"
    )


def _receivers_html(run_dir: Path) -> str:
    config = read_manifest(run_dir)["config"]
    rows = [
        {
            "receiver": r["name"],
            "beast": r["beast"],
            **{k: v for k, v in r.get("metadata", {}).items()},
        }
        for r in config["receivers"]
    ]
    return table_html(pl.DataFrame(rows))


CSS = """
body { font-family: system-ui, sans-serif; margin: 2rem auto; max-width: 1200px;
       color: #222; }
h1 { margin-bottom: 0.2rem; }
h2 { border-bottom: 1px solid #ccc; padding-bottom: 0.2rem; margin-top: 2.5rem; }
table { border-collapse: collapse; margin: 0.8rem 0; font-size: 0.9rem; }
th, td { border: 1px solid #ddd; padding: 0.25rem 0.6rem; text-align: right; }
th { background: #f4f4f4; }
td:first-child, th:first-child { text-align: left; }
.note { color: #555; font-size: 0.9rem; max-width: 60rem; }
.grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
"""


def write_report(
    analysis: Analysis, out_dir: Path, resampler: Resampler | None = None
) -> Path:
    resampler = resampler or Resampler()
    out_dir.mkdir(parents=True, exist_ok=True)
    table = analysis.table
    receivers = analysis.receivers
    reference = analysis.reference
    several = len(receivers) > 1

    outputs: dict[str, pl.DataFrame] = {
        "comparability": comparability(analysis.validity),
        "overall": rates_by(table, None, receivers, resampler),
        "exclusives": exclusives(table, receivers),
        "timing": timing(analysis.anchors),
        "frames": analysis.frames,
        "polar_cells": polar_cells(table, receivers),
        "polar_difference": polar_difference(table, receivers, reference),
        "range_outline": range_outline(table, receivers),
        "altitude_by_range": altitude_by_range(table, receivers),
        "gain": gain_series(analysis.run_dir, receivers).filter(
            pl.col("time_ns").is_between(analysis.start_ns, analysis.end_ns)
        ),
        "validity": analysis.validity,
    }
    for dimension in DIMENSIONS:
        key = dimension.name.replace(" ", "_")
        outputs[f"rates_{key}"] = rates_by(table, dimension, receivers, resampler)
    if several:
        for dimension in WIN_DIMENSIONS:
            key = dimension.name.replace(" ", "_")
            outputs[f"wins_{key}"] = wins_by(table, dimension, receivers, resampler)
    curves, thresholds = sensitivity(table, receivers, resampler)
    outputs["sensitivity"] = curves
    outputs["sensitivity_thresholds"] = thresholds
    visits = visit_metrics(table, receivers)
    outputs.update(
        {
            "visit_events": visits.events,
            "acquisition": visits.acquisition,
            "acquisition_pairs": visits.acquisition_pairs,
            "acquisition_position": visits.acquisition_position,
            "fade": visits.fade,
            "fade_pairs": visits.fade_pairs,
            "fade_position": visits.fade_position,
            "continuity": visits.continuity,
        }
    )
    for name, frame in outputs.items():
        frame.write_parquet(out_dir / f"{name}.parquet")
    table.write_parquet(out_dir / "transmissions.parquet")

    sections = _sections(analysis, outputs, several)
    page = _page(analysis, sections)
    path = out_dir / "report.html"
    path.write_text(page)
    return path


def _sections(
    analysis: Analysis, outputs: dict[str, pl.DataFrame], several: bool
) -> list[Section]:
    receivers = analysis.receivers
    reference = analysis.reference
    truncated = [f for f in analysis.files if f.truncated]
    comparable = outputs["comparability"]
    not_comparable = comparable.filter(~pl.col("comparable"))["receiver"].to_list()
    warning = (
        "<p class='note'><b>Not comparable:</b> "
        + ", ".join(html.escape(r) for r in not_comparable)
        + " (valid for under 90 % of the window). Their numbers are shown but"
        " should not be compared with the others.</p>"
        if not_comparable
        else ""
    )

    sections = [
        Section(
            "Receivers",
            _receivers_html(analysis.run_dir)
            + table_html(comparable, percent=("valid_share",))
            + warning
            + table_html(analysis.frames)
            + (
                f"<p class='note'>{len(truncated)} recording file(s) were cut short"
                " and read up to the damage.</p>"
                if truncated
                else ""
            ),
        ),
        Section(
            "Health",
            figures.health(outputs["validity"], outputs["gain"], receivers)
            + "<p class='note'>A minute is invalid for a receiver when it was"
            " disconnected, could not be aligned, dropped samples, or was"
            " overloaded (readsb: noise floor above -20 dBFS). A receiver is"
            " scored only over its"
            " valid minutes; comparisons use minutes where all were valid.</p>",
        ),
        Section(
            "Overall",
            table_html(
                outputs["overall"].drop("order", "band"),
                percent=("rate", "low", "high"),
            )
            + "<p class='note'>Share of all transmissions any receiver decoded,"
            " with a 95 % interval from resampling aircraft visits.</p>",
        ),
    ]
    if several:
        wins = "".join(
            f"<h3>By {d.name}</h3>"
            + table_html(
                outputs[f"wins_{d.name.replace(' ', '_')}"].drop("order"),
                percent=("margin", "low", "high"),
            )
            for d in WIN_DIMENSIONS
        )
        sections.append(
            Section(
                "Where each receiver wins",
                "<p class='note'>Best receiver per band, its margin over the"
                " runner-up and the margin's 95 % interval. <i>decided</i> is"
                " false where the interval includes zero: no difference.</p>" + wins,
            )
        )
    sections.append(
        Section(
            "Detection rate",
            "".join(
                figures.rates(
                    outputs[f"rates_{d.name.replace(' ', '_')}"],
                    receivers,
                    f"Detection rate by {d.name}",
                )
                for d in DIMENSIONS
            ),
        )
    )
    maps = "".join(
        figures.polar_rates(outputs["polar_cells"], r, f"{r}: detection rate")
        for r in receivers
    )
    differences = "".join(
        figures.polar_difference(outputs["polar_difference"], r, reference)
        for r in receivers
        if r != reference
    )
    sections.append(
        Section(
            "Coverage",
            figures.outlines(outputs["range_outline"], receivers)
            + f"<div class='grid'>{maps}{differences}</div>"
            + "".join(
                figures.coverage(outputs["altitude_by_range"], r) for r in receivers
            ),
        )
    )
    if several:
        sections.append(
            Section(
                "Sensitivity",
                "".join(
                    figures.sensitivity(outputs["sensitivity"], receivers, by)
                    for by in receivers
                )
                + table_html(outputs["sensitivity_thresholds"])
                + "<p class='note'>Signal scales differ between decoders, so each"
                " chart is read against one receiver's own scale.</p>",
            )
        )
        sections.append(
            Section(
                "Heard by one receiver alone",
                table_html(outputs["exclusives"])
                + "<p class='note'>Over minutes where every receiver was valid."
                " <i>only</i>: decoded by this receiver and no other;"
                " <i>missed_only</i>: decoded by every other receiver but not"
                " this one.</p>",
            )
        )
    sections.append(
        Section(
            "Acquisition and fade-out",
            "<p class='note'>Per aircraft visit, how many seconds after the"
            " first receiver each one decoded it (acquisition), and how long"
            " before the last one it lost it (fade-out). Ties within 1 s count"
            " for both.</p>"
            + "<h3>Acquisition, any message</h3>"
            + table_html(outputs["acquisition"])
            + table_html(outputs["acquisition_pairs"])
            + "<h3>Acquisition, first position</h3>"
            + table_html(outputs["acquisition_position"])
            + figures.lags(
                outputs["visit_events"], receivers, "first", "Acquisition lag"
            )
            + "<h3>Fade-out, any message</h3>"
            + table_html(outputs["fade"])
            + table_html(outputs["fade_pairs"])
            + "<h3>Fade-out, last position</h3>"
            + table_html(outputs["fade_position"])
            + figures.lags(outputs["visit_events"], receivers, "last", "Fade-out lag")
            + "<h3>Continuity</h3>"
            + table_html(
                outputs["continuity"],
                percent=("median_coverage", "median_of_reference"),
            )
            + "<p class='note'>Share of each visit's 10 s bins in which the"
            " receiver decoded a position, against the bins in which any receiver"
            " did.</p>",
        )
    )
    if several:
        sections.append(
            Section(
                "Timestamps",
                table_html(outputs["timing"])
                + f"<p class='note'>Residuals of each receiver's timestamps"
                f" against {html.escape(reference)}'s after alignment: how"
                " suitable it is for MLAT.</p>",
            )
        )
    return sections


def _page(analysis: Analysis, sections: list[Section]) -> str:
    manifest = read_manifest(analysis.run_dir)
    body = "".join(f"<h2>{html.escape(s.title)}</h2>{s.body}" for s in sections)
    title = f"beast-compare: {html.escape(manifest['run_id'])}"
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<title>{title}</title><style>{CSS}</style>"
        f"<script>{plotly.offline.get_plotlyjs()}</script></head><body>"
        f"<h1>{title}</h1>"
        f"<p class='note'>{_when(analysis.start_ns)} to {_when(analysis.end_ns)}."
        f" Reference receiver: {html.escape(analysis.reference)}."
        f" beast-compare {__version__}.</p>"
        f"{body}</body></html>"
    )
