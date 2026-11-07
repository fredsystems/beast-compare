"""Load a recorded run into one polars frame of Mode S messages.

One row per recorded Mode S frame, from every receiver:

=========== ======= ==========================================================
column      dtype   meaning
=========== ======= ==========================================================
receiver    Enum    receiver name, in the run's configured order
arrival_ns  Int64   when the recorder read it (coarse, recorder clock)
counter     Int64   the decoder's own timestamp counter (normally 12 MHz)
signal      UInt8   the decoder's signal byte
msg_len     UInt8   7 or 14 bytes
msg_hi      UInt64  message bytes 0-7, big-endian
msg_lo      UInt64  message bytes 8-13, big-endian, zero for short messages
df          UInt8   downlink format; 24 stands for every Comm-D format
address     UInt32  ICAO address, from the AA field or the parity, else null
typecode    UInt8   ADS-B type code for DF17/18, else null
=========== ======= ==========================================================

The message is held as two integers rather than bytes so that matching the
same transmission across receivers is an integer join.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import polars as pl

from beast_compare.beast import FrameType
from beast_compare.manifest import read_manifest
from beast_compare.recordfile import RECORD_HEADER, SUFFIX, decompress_file

Bytes = npt.NDArray[np.uint8]
U64 = npt.NDArray[np.uint64]

# Mode S CRC-24 generator polynomial.
CRC_GENERATOR = 0xFFF409
HOUR_NS = 3600 * 10**9

# Record length (type byte + body) of each Mode S frame type.
_RECORD_LENGTHS = {FrameType.MODE_S_SHORT: 15, FrameType.MODE_S_LONG: 22}
# Within a record: header, type byte, 6-byte counter, signal, message.
_TYPE_AT = RECORD_HEADER.size
_COUNTER_AT = _TYPE_AT + 1
_SIGNAL_AT = _COUNTER_AT + 6
_MESSAGE_AT = _SIGNAL_AT + 1

# Formats whose address is overlaid on the parity (address/parity field).
_PARITY_ADDRESS_DFS = (0, 4, 5, 16, 20, 21)
# Formats carrying the address in clear in bytes 1-3.
_CLEAR_ADDRESS_DFS = (11, 17, 18)


def _crc_table() -> npt.NDArray[np.uint32]:
    table = np.zeros(256, dtype=np.uint32)
    for byte in range(256):
        crc = byte << 16
        for _ in range(8):
            crc = (crc << 1) ^ CRC_GENERATOR if crc & 0x800000 else crc << 1
            crc &= 0xFFFFFF
        table[byte] = crc
    return table


_CRC_TABLE = _crc_table()


def crc24(messages: Bytes) -> npt.NDArray[np.uint32]:
    """Mode S CRC-24 of each row of an (n, k) byte matrix."""
    crc = np.zeros(messages.shape[0], dtype=np.uint32)
    for column in range(messages.shape[1]):
        index = ((crc >> 16) ^ messages[:, column]) & 0xFF
        crc = ((crc << 8) & 0xFFFFFF) ^ _CRC_TABLE[index]
    return crc


def _big_endian(columns: Bytes) -> U64:
    value = np.zeros(columns.shape[0], dtype=np.uint64)
    for column in range(columns.shape[1]):
        value = (value << np.uint64(8)) | columns[:, column].astype(np.uint64)
    return value


@dataclass(frozen=True, slots=True)
class FileReport:
    receiver: str
    path: Path
    records: int
    mode_s: int
    truncated: bool


@dataclass(frozen=True, slots=True)
class Ingested:
    frames: pl.DataFrame
    files: tuple[FileReport, ...]

    @property
    def truncated_files(self) -> tuple[FileReport, ...]:
        return tuple(f for f in self.files if f.truncated)


def _scan(data: bytes) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], bool]:
    """Offsets and lengths of every complete record in a decompressed file."""
    offsets: list[int] = []
    lengths: list[int] = []
    unpack = RECORD_HEADER.unpack_from
    header = RECORD_HEADER.size
    offset = 0
    end = len(data)
    while offset + header <= end:
        _, length = unpack(data, offset)
        if offset + header + length > end:
            break
        offsets.append(offset)
        lengths.append(length)
        offset += header + length
    cut = offset != end
    return (
        np.asarray(offsets, dtype=np.int64),
        np.asarray(lengths, dtype=np.int64),
        cut,
    )


def _empty(receivers: list[str]) -> pl.DataFrame:
    return pl.DataFrame(
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
        }
    )


def read_file(
    path: Path,
    receiver: str,
    receivers: list[str],
    start_ns: int | None = None,
    end_ns: int | None = None,
) -> tuple[pl.DataFrame, FileReport]:
    data, truncated = decompress_file(path)
    offsets, lengths, cut = _scan(data)
    truncated = truncated or cut
    buffer = np.frombuffer(data, dtype=np.uint8)

    type_bytes = buffer[offsets + _TYPE_AT] if len(offsets) else offsets
    is_short = (type_bytes == FrameType.MODE_S_SHORT) & (
        lengths == _RECORD_LENGTHS[FrameType.MODE_S_SHORT]
    )
    is_long = (type_bytes == FrameType.MODE_S_LONG) & (
        lengths == _RECORD_LENGTHS[FrameType.MODE_S_LONG]
    )
    mode_s = is_short | is_long
    report = FileReport(receiver, path, len(offsets), int(mode_s.sum()), truncated)

    offsets = offsets[mode_s]
    is_long = is_long[mode_s]
    arrival = _big_endian(
        buffer[offsets[:, None] + np.arange(8)].reshape(-1, 8)
    ).astype(np.int64)
    window = np.ones(len(offsets), dtype=bool)
    if start_ns is not None:
        window &= arrival >= start_ns
    if end_ns is not None:
        window &= arrival < end_ns
    offsets, is_long, arrival = offsets[window], is_long[window], arrival[window]

    counter = _big_endian(
        buffer[offsets[:, None] + _COUNTER_AT + np.arange(6)].reshape(-1, 6)
    ).astype(np.int64)
    signal = buffer[offsets + _SIGNAL_AT]
    msg_len = np.where(is_long, 14, 7).astype(np.uint8)

    # Every message as 14 bytes, short ones zero-padded. Indices past a short
    # message's end are clipped into the buffer and then masked out.
    columns = np.arange(14)
    index = np.minimum(offsets[:, None] + _MESSAGE_AT + columns, len(buffer) - 1)
    message = np.where(
        columns < msg_len[:, None].astype(np.int64),
        buffer[index].reshape(-1, 14),
        0,
    ).astype(np.uint8)

    df = np.minimum(message[:, 0] >> 3, 24).astype(np.uint8)

    clear_address = _big_endian(message[:, 1:4]).astype(np.uint32)
    parity_short = crc24(message[:, :4]) ^ _big_endian(message[:, 4:7]).astype(
        np.uint32
    )
    parity_long = crc24(message[:, :11]) ^ _big_endian(message[:, 11:14]).astype(
        np.uint32
    )
    parity_address = np.where(is_long, parity_long, parity_short)

    frame = pl.DataFrame(
        {
            "receiver": pl.Series([receiver] * len(offsets), dtype=pl.Enum(receivers)),
            "arrival_ns": arrival,
            "counter": counter,
            "signal": signal,
            "msg_len": msg_len,
            "msg_hi": _big_endian(message[:, :8]),
            "msg_lo": _big_endian(message[:, 8:14]),
            "df": df,
            "_clear": clear_address,
            "_parity": parity_address.astype(np.uint32),
            "_typecode": (message[:, 4] >> 3).astype(np.uint8),
        }
    )
    frame = frame.with_columns(
        address=pl.when(pl.col("df").is_in(_CLEAR_ADDRESS_DFS))
        .then(pl.col("_clear"))
        .when(pl.col("df").is_in(_PARITY_ADDRESS_DFS))
        .then(pl.col("_parity"))
        .otherwise(None),
        typecode=pl.when(pl.col("df").is_in((17, 18)))
        .then(pl.col("_typecode"))
        .otherwise(None),
    ).drop("_clear", "_parity", "_typecode")
    return frame, report


def message_bytes(frames: pl.DataFrame) -> Bytes:
    """The (n, 14) message byte matrix of a frame, short messages zero-padded."""
    hi = frames["msg_hi"].to_numpy().astype(np.uint64)
    lo = frames["msg_lo"].to_numpy().astype(np.uint64)
    out = np.zeros((len(frames), 14), dtype=np.uint8)
    for i in range(8):
        out[:, i] = (hi >> np.uint64(56 - 8 * i)) & np.uint64(0xFF)
    for i in range(6):
        out[:, 8 + i] = (lo >> np.uint64(40 - 8 * i)) & np.uint64(0xFF)
    return out


def _file_hour_ns(path: Path) -> int | None:
    stem = path.name.removesuffix(SUFFIX)
    try:
        hour = dt.datetime.strptime(stem, "%Y%m%dT%H").replace(tzinfo=dt.UTC)
    except ValueError:
        return None
    return int(hour.timestamp()) * 10**9


def load_run(
    run_dir: Path, start_ns: int | None = None, end_ns: int | None = None
) -> Ingested:
    """Every receiver's Mode S frames, optionally limited to a time window."""
    manifest = read_manifest(run_dir)
    receivers = [r["name"] for r in manifest["config"]["receivers"]]
    frames = [_empty(receivers)]
    reports: list[FileReport] = []
    for receiver in receivers:
        for path in sorted((run_dir / receiver).glob("*" + SUFFIX)):
            hour = _file_hour_ns(path)
            if hour is not None:
                if end_ns is not None and hour >= end_ns:
                    continue
                if start_ns is not None and hour + HOUR_NS <= start_ns:
                    continue
            frame, report = read_file(path, receiver, receivers, start_ns, end_ns)
            frames.append(frame)
            reports.append(report)
    return Ingested(pl.concat(frames, how="vertical"), tuple(reports))
