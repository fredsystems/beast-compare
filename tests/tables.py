"""Hand-built analysis tables (see beast_compare.table) for metric tests."""

from dataclasses import dataclass, field

import polars as pl

from beast_compare.table import heard, signal, valid

RECEIVERS = ["a", "b"]


@dataclass
class Tx:
    t: float
    visit: int
    heard: dict[str, int | None]  # receiver -> signal byte, None if missed
    valid: dict[str, bool] = field(default_factory=lambda: {"a": True, "b": True})
    range_nm: float | None = 10.0
    position: bool = False
    df_group: int = 3
    address: int = 0xABCDEF


def table(rows: list[Tx], receivers: list[str] = RECEIVERS) -> pl.DataFrame:
    columns: dict[str, list[object]] = {
        "tx": list(range(len(rows))),
        "ref_segment": [0] * len(rows),
        "t": [float(r.t) for r in rows],
        "wall_s": [float(r.t) for r in rows],
        "minute_ns": [int(r.t // 60) * 60 * 10**9 for r in rows],
        "visit": [r.visit for r in rows],
        "address": [r.address for r in rows],
        "position": [r.position for r in rows],
        "range_nm": [r.range_nm for r in rows],
        "elevation": [1.5] * len(rows),
        "bearing": [45.0] * len(rows),
        "alt_ft": [10_000.0] * len(rows),
        "df_group": [r.df_group for r in rows],
        "density": [100] * len(rows),
    }
    for receiver in receivers:
        columns[heard(receiver)] = [r.heard.get(receiver) is not None for r in rows]
        columns[signal(receiver)] = [r.heard.get(receiver) for r in rows]
        columns[valid(receiver)] = [r.valid.get(receiver, True) for r in rows]
    return pl.DataFrame(
        columns,
        schema_overrides={
            "ref_segment": pl.UInt32,
            "visit": pl.UInt32,
            "address": pl.UInt32,
            "df_group": pl.UInt8,
            "density": pl.UInt32,
            **{signal(r): pl.UInt8 for r in receivers},
        },
    )
