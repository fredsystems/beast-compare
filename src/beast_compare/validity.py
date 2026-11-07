"""When each receiver's results can be trusted, minute by minute.

A receiver is scored only over its valid minutes; the reference set still
includes everything every receiver heard. A minute is invalid for a
receiver when it was:

- disconnected from the recorder for any part of it;
- unaligned: more than ``unaligned_share`` of its frames that minute have
  no reference time (a decoder restart, or no anchors);
- dropping samples (readsb) or losing buffers (airspy_adsb);
- overloaded (readsb only): a noise floor above ``max_noise_dbfs``, or
  more than ``max_strong_share`` of its messages above -3 dBFS.

Overload is judged from these direct signs of clipping, not from the gain.
A receiver at the bottom of its gain range is not necessarily overloaded:
on 2026-11-07 the Pro Stick Plus sat at 0 dB with a -31 dBFS noise floor
and 2.7 % strong messages, and a 10 dB attenuator, which let its gain rise
to 3.7 dB, cost it 6.5 % of its messages. The same stick at 36.4 dB
earlier that day, clipping, had a -2 dBFS noise floor. airspy_adsb reports
neither a noise floor in dBFS nor a strong-message count, so the Airspy is
judged on its connection, alignment and lost buffers alone.

Stats are polled once a minute and describe the minute before the poll, so
each poll is assigned to the minute holding its midpoint. A minute with no
stats is flagged ``stats_missing`` but not made invalid by that alone: a
receiver whose decoder is down also shows up as disconnected.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from beast_compare.recorder import EVENTS_NAME, STATS_NAME

MINUTE_NS = 60 * 10**9


@dataclass(frozen=True, slots=True)
class ValiditySettings:
    unaligned_share: float = 0.1
    max_noise_dbfs: float = -20.0
    max_strong_share: float = 0.1


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def disconnected_spans(
    events: list[dict[str, Any]], receivers: list[str], end_ns: int
) -> list[tuple[str, int, int]]:
    """(receiver, start, end) for every span a receiver was not connected.

    Receivers start disconnected; ``connected`` and ``disconnected`` events
    switch state, and a session stop disconnects everyone.
    """
    down_since: dict[str, int | None] = dict.fromkeys(receivers, 0)
    spans: list[tuple[str, int, int]] = []

    def connect(receiver: str, at: int) -> None:
        since = down_since.get(receiver)
        if since is not None:
            spans.append((receiver, since, at))
        down_since[receiver] = None

    def disconnect(receiver: str, at: int) -> None:
        if down_since.get(receiver) is None:
            down_since[receiver] = at

    for event in sorted(events, key=lambda e: int(e["time_ns"])):
        at = int(event["time_ns"])
        kind = event["event"]
        receiver = event.get("receiver")
        if kind == "connected" and receiver in down_since:
            connect(receiver, at)
        elif kind == "disconnected" and receiver in down_since:
            disconnect(receiver, at)
        elif kind == "session_stop":
            for name in receivers:
                disconnect(name, at)
    for receiver, since in down_since.items():
        if since is not None:
            spans.append((receiver, since, end_ns))
    return spans


def _stats_flags(stats: dict[str, Any], settings: ValiditySettings) -> dict[str, bool]:
    if "gain_db" in stats:  # readsb
        local = stats.get("last1min", {}).get("local", {})
        dropped = (local.get("samples_dropped", 0) or 0) + (
            local.get("samples_lost", 0) or 0
        ) > 0
        noise = local.get("noise")
        accepted = sum(local.get("accepted", []) or [])
        strong = local.get("strong_signals", 0) or 0
        noisy = noise is not None and float(noise) > settings.max_noise_dbfs
        hot = accepted > 0 and strong / accepted > settings.max_strong_share
        return {"dropped": dropped, "overloaded": noisy or hot}
    if "lost_buffers" in stats:  # airspy_adsb
        return {"dropped": int(stats["lost_buffers"]) > 0, "overloaded": False}
    return {"dropped": False, "overloaded": False}


def stats_by_minute(
    run_dir: Path, receivers: list[str], settings: ValiditySettings
) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for receiver in receivers:
        for line in _jsonl(run_dir / receiver / STATS_NAME):
            if line.get("stats") is None:
                continue
            midpoint = int(line["time_ns"]) - MINUTE_NS // 2
            rows.append(
                {
                    "receiver": receiver,
                    "minute_ns": midpoint // MINUTE_NS * MINUTE_NS,
                    **_stats_flags(line["stats"], settings),
                }
            )
    schema = pl.Schema(
        {
            "receiver": pl.Enum(receivers),
            "minute_ns": pl.Int64(),
            "dropped": pl.Boolean(),
            "overloaded": pl.Boolean(),
        }
    )
    return (
        pl.DataFrame(rows, schema=schema)
        .group_by("receiver", "minute_ns")
        .agg(pl.col("dropped", "overloaded").any())
    )


def gain_series(run_dir: Path, receivers: list[str]) -> pl.DataFrame:
    """Every stats poll's gain: receiver, time_ns, gain (dB for readsb,
    the 0-21 linearity step for airspy_adsb)."""
    rows: list[dict[str, Any]] = []
    for receiver in receivers:
        for line in _jsonl(run_dir / receiver / STATS_NAME):
            stats = line.get("stats")
            if stats is None:
                continue
            gain = stats.get("gain_db", stats.get("gain"))
            if gain is not None:
                rows.append(
                    {
                        "receiver": receiver,
                        "time_ns": int(line["time_ns"]),
                        "gain": float(gain),
                    }
                )
    return pl.DataFrame(
        rows,
        schema={"receiver": pl.String, "time_ns": pl.Int64, "gain": pl.Float64},
    ).sort("receiver", "time_ns")


def validity(
    run_dir: Path,
    aligned: pl.DataFrame,
    receivers: list[str],
    start_ns: int,
    end_ns: int,
    settings: ValiditySettings = ValiditySettings(),
) -> pl.DataFrame:
    """One row per receiver and minute in [start_ns, end_ns), with each
    reason as a boolean column and ``valid`` combining them."""
    first = start_ns // MINUTE_NS * MINUTE_NS
    minutes = list(range(first, end_ns, MINUTE_NS))
    grid = pl.DataFrame(
        {
            "receiver": [r for r in receivers for _ in minutes],
            "minute_ns": minutes * len(receivers),
        },
        schema={"receiver": pl.Enum(receivers), "minute_ns": pl.Int64},
    )

    spans = disconnected_spans(_jsonl(run_dir / EVENTS_NAME), receivers, end_ns)
    down = [
        (receiver, minute)
        for receiver, start, end in spans
        for minute in minutes
        if start < minute + MINUTE_NS and end > minute
    ]
    disconnected = pl.DataFrame(
        {"receiver": [d[0] for d in down], "minute_ns": [d[1] for d in down]},
        schema={"receiver": pl.Enum(receivers), "minute_ns": pl.Int64},
    ).with_columns(disconnected=pl.lit(True))

    unaligned = (
        aligned.with_columns(minute_ns=pl.col("arrival_ns") // MINUTE_NS * MINUTE_NS)
        .group_by("receiver", "minute_ns")
        .agg(share=pl.col("t_ref").is_null().mean())
        .select(
            "receiver",
            "minute_ns",
            unaligned=pl.col("share") > settings.unaligned_share,
        )
    )
    stats = stats_by_minute(run_dir, receivers, settings)
    with_stats = {
        r for r in receivers if (run_dir / r / STATS_NAME).exists()
    }  # receivers configured with a stats source

    table = (
        grid.join(disconnected, on=["receiver", "minute_ns"], how="left")
        .join(unaligned, on=["receiver", "minute_ns"], how="left")
        .join(stats, on=["receiver", "minute_ns"], how="left")
        .with_columns(
            stats_missing=pl.col("dropped").is_null()
            & pl.col("receiver").is_in(sorted(with_stats)),
        )
        .with_columns(
            pl.col("disconnected", "unaligned", "dropped", "overloaded").fill_null(
                False
            )
        )
    )
    return table.with_columns(
        valid=~pl.any_horizontal("disconnected", "unaligned", "dropped", "overloaded")
    ).sort("receiver", "minute_ns")
