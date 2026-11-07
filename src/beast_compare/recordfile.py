"""Recording files: hourly, zstd-compressed, append-only.

Each record is ``[u64 arrival_unix_ns][u8 len][frame]``, big-endian, where
``frame`` is a Beast frame's type byte plus unescaped body (see
:meth:`beast_compare.beast.Frame.encode`). ``arrival_unix_ns`` is when the
recorder read the chunk the frame arrived in; it is only a coarse time.

Files are opened for append and the compressor ends a zstd frame at every
flush, so a file is a sequence of independent zstd frames. A crash loses
at most the data since the last flush, and a restarted recorder appends to
the same hour's file without rewriting it.
"""

from __future__ import annotations

import datetime as dt
import struct
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import zstandard

from beast_compare.beast import MAX_FRAME_LEN, Frame

RECORD_HEADER = struct.Struct(">QB")
SUFFIX = ".bcr.zst"


@dataclass(frozen=True, slots=True)
class Record:
    arrival_ns: int
    frame: Frame


def hour_file_name(arrival_ns: int) -> str:
    # Integer seconds: a float division rounds the last nanoseconds of an
    # hour up into the next one.
    hour = dt.datetime.fromtimestamp(arrival_ns // 10**9, tz=dt.UTC)
    return hour.strftime("%Y%m%dT%H") + SUFFIX


def encode_record(record: Record) -> bytes:
    raw = record.frame.encode()
    if len(raw) > MAX_FRAME_LEN:
        raise ValueError(f"frame of {len(raw)} bytes is too long to record")
    return RECORD_HEADER.pack(record.arrival_ns, len(raw)) + raw


class RecordWriter:
    """Writes one receiver's records into hourly files under ``directory``."""

    def __init__(self, directory: Path) -> None:
        self._directory = directory
        self._directory.mkdir(parents=True, exist_ok=True)
        self._compressor = zstandard.ZstdCompressor(level=3)
        self._file_name: str | None = None
        self._handle: BinaryIO | None = None
        # One compression object per zstd frame; None between frames.
        self._frame: zstandard.ZstdCompressionObj | None = None

    @property
    def current_path(self) -> Path | None:
        if self._file_name is None:
            return None
        return self._directory / self._file_name

    def write(self, record: Record) -> None:
        name = hour_file_name(record.arrival_ns)
        if name != self._file_name:
            self.close()
            self._handle = (self._directory / name).open("ab")
            self._file_name = name
        assert self._handle is not None
        if self._frame is None:
            self._frame = self._compressor.compressobj()
        self._handle.write(self._frame.compress(encode_record(record)))

    def flush(self) -> None:
        """End the current zstd frame so everything so far is readable."""
        if self._frame is None:
            return
        assert self._handle is not None
        self._handle.write(self._frame.flush(zstandard.COMPRESSOBJ_FLUSH_FINISH))
        self._handle.flush()
        self._frame = None

    def close(self) -> None:
        self.flush()
        if self._handle is not None:
            self._handle.close()
        self._handle = None
        self._file_name = None


class RecordReader:
    """Iterates a recording file's records.

    A file cut short by a crash still yields every complete record before
    the damage; :attr:`truncated` is then set once iteration finishes.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.truncated = False

    def __iter__(self) -> Iterator[Record]:
        data, self.truncated = decompress_file(self.path)
        offset = 0
        header_len = RECORD_HEADER.size
        while offset < len(data):
            if offset + header_len > len(data):
                self.truncated = True
                return
            arrival_ns, length = RECORD_HEADER.unpack_from(data, offset)
            start = offset + header_len
            end = start + length
            if end > len(data):
                self.truncated = True
                return
            yield Record(arrival_ns, Frame.decode(data[start:end]))
            offset = end


def decompress_file(path: Path) -> tuple[bytes, bool]:
    """A recording file's records as one buffer, and whether it was cut short.

    Decompressed zstd frame by zstd frame, so that a frame cut short is
    detected: a streaming reader returns its partial content without error.
    """
    chunks: list[bytes] = []
    decompressor = zstandard.ZstdDecompressor()
    remaining = path.read_bytes()
    truncated = False
    while remaining:
        frame = decompressor.decompressobj()
        try:
            chunks.append(frame.decompress(remaining))
        except zstandard.ZstdError:
            truncated = True
            break
        if not frame.eof:
            truncated = True
            break
        remaining = frame.unused_data
    return b"".join(chunks), truncated


def recording_files(directory: Path) -> list[Path]:
    """A receiver's recording files, oldest first."""
    return sorted(directory.glob("*" + SUFFIX))
