"""Plotly figures for the report, each returned as an HTML fragment.

The page loads plotly.js once; fragments do not include it.
"""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go
import polars as pl

from beast_compare.visits import by_receiver

PALETTE = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b")
LAYOUT: dict[str, Any] = {
    "template": "plotly_white",
    "margin": {"l": 50, "r": 20, "t": 50, "b": 50},
    "height": 420,
}


def colour(receivers: list[str], receiver: str) -> str:
    return PALETTE[receivers.index(receiver) % len(PALETTE)]


def _html(figure: Any) -> str:
    return str(figure.to_html(full_html=False, include_plotlyjs=False))


def rates(rates_frame: pl.DataFrame, receivers: list[str], title: str) -> str:
    """Detection rate per band with its interval, one trace per receiver."""
    figure = go.Figure()
    for receiver in receivers:
        rows = rates_frame.filter(pl.col("receiver") == receiver).sort("order")
        rate = rows["rate"].to_list()
        figure.add_trace(
            go.Scatter(
                x=rows["band"].to_list(),
                y=rate,
                name=receiver,
                mode="lines+markers",
                line={"color": colour(receivers, receiver)},
                error_y={
                    "type": "data",
                    "symmetric": False,
                    "array": [h - r for h, r in zip(rows["high"], rate, strict=True)],
                    "arrayminus": [
                        r - lo for lo, r in zip(rows["low"], rate, strict=True)
                    ],
                },
                customdata=rows["n"].to_list(),
                hovertemplate="%{x}: %{y:.1%} (n=%{customdata})",
            )
        )
    figure.update_layout(
        title=title, yaxis={"tickformat": ".0%", "rangemode": "tozero"}, **LAYOUT
    )
    return _html(figure)


def polar_rates(cells: pl.DataFrame, receiver: str, title: str) -> str:
    rows = cells.filter(pl.col("receiver") == receiver)
    figure = go.Figure(
        go.Barpolar(
            r=[10] * len(rows),
            base=rows["range_lo"].to_list(),
            theta=[b + 5 for b in rows["bearing_lo"].to_list()],
            width=[10] * len(rows),
            marker={
                "color": rows["rate"].to_list(),
                "colorscale": "Viridis",
                "cmin": 0,
                "cmax": 1,
                "colorbar": {"title": "rate", "tickformat": ".0%"},
            },
            customdata=rows["n"].to_list(),
            hovertemplate="%{base}-%{r} nm, %{theta} deg: %{marker.color:.1%}"
            " (n=%{customdata})<extra></extra>",
        )
    )
    figure.update_layout(
        title=title,
        polar={"angularaxis": {"direction": "clockwise", "rotation": 90}},
        **{**LAYOUT, "height": 520},
    )
    return _html(figure)


def polar_difference(cells: pl.DataFrame, receiver: str, reference: str) -> str:
    rows = cells.filter(pl.col("receiver") == receiver)
    figure = go.Figure(
        go.Barpolar(
            r=[10] * len(rows),
            base=rows["range_lo"].to_list(),
            theta=[b + 5 for b in rows["bearing_lo"].to_list()],
            width=[10] * len(rows),
            marker={
                "color": rows["difference"].to_list(),
                "colorscale": "RdBu",
                "cmid": 0,
                "colorbar": {"title": "difference", "tickformat": "+.0%"},
            },
            customdata=rows["n"].to_list(),
            hovertemplate="%{base}-%{r} nm, %{theta} deg: %{marker.color:+.1%}"
            " (n=%{customdata})<extra></extra>",
        )
    )
    figure.update_layout(
        title=f"{receiver} minus {reference}",
        polar={"angularaxis": {"direction": "clockwise", "rotation": 90}},
        **{**LAYOUT, "height": 520},
    )
    return _html(figure)


def outlines(outline: pl.DataFrame, receivers: list[str]) -> str:
    figure = go.Figure()
    for receiver in receivers:
        rows = outline.filter(pl.col("receiver") == receiver).sort("bearing_lo")
        if rows.is_empty():
            continue
        theta = [b + 5 for b in rows["bearing_lo"].to_list()]
        r = rows["range_nm"].to_list()
        figure.add_trace(
            go.Scatterpolar(
                r=r + r[:1],
                theta=theta + theta[:1],
                name=receiver,
                mode="lines",
                line={"color": colour(receivers, receiver)},
            )
        )
    figure.update_layout(
        title="Range outline (99th percentile range per 10 degree sector, nm)",
        polar={"angularaxis": {"direction": "clockwise", "rotation": 90}},
        **{**LAYOUT, "height": 560},
    )
    return _html(figure)


def sensitivity(curves: pl.DataFrame, receivers: list[str], by: str) -> str:
    figure = go.Figure()
    for receiver in receivers:
        rows = curves.filter(
            (pl.col("by") == by) & (pl.col("receiver") == receiver)
        ).sort("level_dbfs")
        if rows.is_empty():
            continue
        figure.add_trace(
            go.Scatter(
                x=rows["level_dbfs"].to_list(),
                y=rows["rate"].to_list(),
                name=receiver,
                mode="lines+markers",
                line={"color": colour(receivers, receiver)},
                customdata=rows["n"].to_list(),
                hovertemplate="%{x} dBFS: %{y:.1%} (n=%{customdata})",
            )
        )
    figure.update_layout(
        title=f"Detection rate against {by}'s signal level",
        xaxis={"title": f"signal at {by} (dBFS)"},
        yaxis={"tickformat": ".0%", "range": [0, 1.02]},
        **LAYOUT,
    )
    return _html(figure)


def coverage(cells: pl.DataFrame, receiver: str) -> str:
    rows = cells.filter(pl.col("receiver") == receiver)
    figure = go.Figure(
        go.Heatmap(
            x=rows["range_lo"].to_list(),
            y=rows["alt_lo"].to_list(),
            z=rows["rate"].to_list(),
            zmin=0,
            zmax=1,
            colorscale="Viridis",
            colorbar={"title": "rate", "tickformat": ".0%"},
        )
    )
    figure.update_layout(
        title=f"{receiver}: detection rate by range and altitude",
        xaxis={"title": "range (nm)"},
        yaxis={"title": "altitude (ft)"},
        **LAYOUT,
    )
    return _html(figure)


def lags(events: pl.DataFrame, receivers: list[str], column: str, title: str) -> str:
    """How far behind the best receiver each one was, per visit."""
    later_is_better = column.startswith("last")
    valid_column = "valid_end" if later_is_better else "valid_start"
    wide = by_receiver(events, receivers, column, valid_column)
    best = (
        pl.max_horizontal(receivers)
        if later_is_better
        else pl.min_horizontal(receivers)
    )
    figure = go.Figure()
    for receiver in receivers:
        lag = wide.select(
            lag=(
                (best - pl.col(receiver))
                if later_is_better
                else (pl.col(receiver) - best)
            )
        )["lag"].drop_nulls()
        figure.add_trace(
            go.Histogram(
                x=lag.to_list(),
                name=receiver,
                opacity=0.6,
                marker={"color": colour(receivers, receiver)},
                xbins={"size": 5},
            )
        )
    figure.update_layout(
        title=title,
        barmode="overlay",
        xaxis={"title": "seconds behind the best receiver"},
        yaxis={"title": "visits"},
        **LAYOUT,
    )
    return _html(figure)


REASONS = ("disconnected", "unaligned", "dropped", "overloaded")
REASON_COLOURS = {
    "valid": "#2ca02c",
    "disconnected": "#7f7f7f",
    "unaligned": "#9467bd",
    "dropped": "#ff7f0e",
    "overloaded": "#d62728",
}


def health(validity: pl.DataFrame, gains: pl.DataFrame, receivers: list[str]) -> str:
    """Valid minutes per receiver, coloured by the first failing reason,
    above each receiver's gain over time."""
    reason = pl.lit("valid")
    for name in reversed(REASONS):
        reason = pl.when(pl.col(name)).then(pl.lit(name)).otherwise(reason)
    minutes = validity.with_columns(
        reason=reason,
        time=pl.from_epoch(pl.col("minute_ns"), time_unit="ns"),
        receiver=pl.col("receiver").cast(pl.String),
    )
    figure = go.Figure()
    for name, shade in REASON_COLOURS.items():
        rows = minutes.filter(pl.col("reason") == name)
        if rows.is_empty():
            continue
        figure.add_trace(
            go.Scatter(
                x=rows["time"].to_list(),
                y=rows["receiver"].to_list(),
                mode="markers",
                marker={"symbol": "square", "size": 9, "color": shade},
                name=name,
                yaxis="y",
            )
        )
    for receiver in receivers:
        rows = gains.filter(pl.col("receiver") == receiver).with_columns(
            time=pl.from_epoch(pl.col("time_ns"), time_unit="ns")
        )
        if rows.is_empty():
            continue
        figure.add_trace(
            go.Scatter(
                x=rows["time"].to_list(),
                y=rows["gain"].to_list(),
                mode="lines",
                name=f"{receiver} gain",
                line={"color": colour(receivers, receiver)},
                yaxis="y2",
            )
        )
    figure.update_layout(
        title="Health: valid minutes, and gain over time (UTC)",
        yaxis={
            "domain": [0.6, 1],
            "categoryorder": "array",
            "categoryarray": receivers,
        },
        yaxis2={"domain": [0, 0.5], "title": "gain"},
        **{**LAYOUT, "height": 520},
    )
    return _html(figure)
