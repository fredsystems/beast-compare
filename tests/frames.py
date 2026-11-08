"""Build ingest-shaped frames directly, for analysis tests."""

from dataclasses import dataclass

import polars as pl

from runs import BASE_NS

HZ = 12_000_000


@dataclass
class Row:
    receiver: str
    arrival_ns: int
    counter: int
    msg_hi: int
    msg_lo: int = 0
    msg_len: int = 14
    signal: int = 64
    df: int = 17
    address: int | None = 0xABCDEF
    typecode: int | None = 11


def frames(rows: list[Row], receivers: list[str]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "receiver": [r.receiver for r in rows],
            "arrival_ns": [r.arrival_ns for r in rows],
            "counter": [r.counter for r in rows],
            "signal": [r.signal for r in rows],
            "msg_len": [r.msg_len for r in rows],
            "msg_hi": [r.msg_hi for r in rows],
            "msg_lo": [r.msg_lo for r in rows],
            "df": [r.df for r in rows],
            "address": [r.address for r in rows],
            "typecode": [r.typecode for r in rows],
        },
        schema={
            "receiver": pl.Enum(receivers),
            "arrival_ns": pl.Int64,
            "counter": pl.Int64,
            "signal": pl.UInt8,
            "msg_len": pl.UInt8,
            "msg_hi": pl.UInt64,
            "msg_lo": pl.UInt64,
            "df": pl.UInt8,
            "address": pl.UInt32,
            "typecode": pl.UInt8,
        },
    )


def arrival(seconds: float) -> int:
    return BASE_NS + int(seconds * 1e9)
