from pathlib import Path

import pytest

from beast_compare.beast import Frame, FrameType
from beast_compare.recordfile import (
    Record,
    RecordReader,
    RecordWriter,
    encode_record,
    hour_file_name,
    recording_files,
)

HOUR_NS = 3600 * 10**9
# 2026-11-07T16:00:00Z
BASE_NS = 1_794_067_200 * 10**9


def record(arrival_ns: int, n: int) -> Record:
    body = (
        n.to_bytes(6, "big") + b"\x40" + bytes.fromhex("8d4ca2d158c901a0c0b4a0f1a2b3")
    )
    return Record(arrival_ns, Frame(FrameType.MODE_S_LONG, body))


def read_all(directory: Path) -> list[Record]:
    return [r for path in recording_files(directory) for r in RecordReader(path)]


def test_hour_file_name() -> None:
    assert hour_file_name(BASE_NS) == "20261107T16.bcr.zst"
    assert hour_file_name(BASE_NS + HOUR_NS - 1) == "20261107T16.bcr.zst"
    assert hour_file_name(BASE_NS + HOUR_NS) == "20261107T17.bcr.zst"


def test_round_trip_across_hours(tmp_path: Path) -> None:
    records = [record(BASE_NS + i * HOUR_NS // 4, i) for i in range(8)]
    writer = RecordWriter(tmp_path)
    for r in records:
        writer.write(r)
    writer.close()
    assert [p.name for p in recording_files(tmp_path)] == [
        "20261107T16.bcr.zst",
        "20261107T17.bcr.zst",
    ]
    assert read_all(tmp_path) == records


def test_restart_appends_to_the_same_hour(tmp_path: Path) -> None:
    first = [record(BASE_NS + i, i) for i in range(3)]
    second = [record(BASE_NS + 100 + i, 100 + i) for i in range(3)]
    for batch in (first, second):
        writer = RecordWriter(tmp_path)
        for r in batch:
            writer.write(r)
        writer.close()
    assert read_all(tmp_path) == first + second


def test_flushed_data_survives_a_crash(tmp_path: Path) -> None:
    writer = RecordWriter(tmp_path)
    kept = [record(BASE_NS + i, i) for i in range(5)]
    for r in kept:
        writer.write(r)
    writer.flush()
    for i in range(5, 10):
        writer.write(record(BASE_NS + i, i))
    writer.flush()
    path = writer.current_path
    assert path is not None
    # Simulate a crash part way through writing the second zstd frame.
    data = path.read_bytes()
    path.write_bytes(data[:-7])

    reader = RecordReader(path)
    recovered = list(reader)
    assert recovered[: len(kept)] == kept
    assert reader.truncated


def test_complete_file_is_not_truncated(tmp_path: Path) -> None:
    writer = RecordWriter(tmp_path)
    writer.write(record(BASE_NS, 1))
    writer.close()
    reader = RecordReader(recording_files(tmp_path)[0])
    assert len(list(reader)) == 1
    assert not reader.truncated


def test_oversized_frame_is_rejected() -> None:
    with pytest.raises(ValueError):
        encode_record(Record(BASE_NS, Frame(0xE3, bytes(255))))
