"""Build recorded runs on disk for tests."""

import json
from pathlib import Path

from conftest import fixture_stream

from beast_compare.beast import BeastParser, Frame
from beast_compare.config import RecorderConfig, parse_config
from beast_compare.manifest import begin_session
from beast_compare.recordfile import Record, RecordWriter

# 2026-11-07T16:00:00Z
BASE_NS = 1_794_067_200 * 10**9


def make_run(data_dir: Path, receivers: dict[str, list[Record]]) -> RecorderConfig:
    config = parse_config(
        {
            "run_id": "test",
            "data_dir": str(data_dir),
            "receivers": [
                {"name": name, "beast": f"127.0.0.1:{30000 + i}"}
                for i, name in enumerate(receivers)
            ],
        }
    )
    begin_session(config)
    for name, records in receivers.items():
        writer = RecordWriter(config.run_dir / name)
        for record in records:
            writer.write(record)
        writer.close()
    return config


def fixture_records(name: str, step_ns: int | None = None) -> list[Record]:
    """A capture's frames as records starting at BASE_NS.

    By default each frame arrives when its own counter says it was decoded,
    rounded up to the next 20 ms (a recorder read), as in a real recording.
    With ``step_ns``, frames are instead spaced evenly by index.
    """
    frames: list[Frame] = BeastParser().feed(fixture_stream(name))
    mode_s = [f for f in frames if f.type_byte != 0x31]
    first = mode_s[0].timestamp
    records = []
    for i, frame in enumerate(frames):
        if step_ns is not None:
            arrival = BASE_NS + i * step_ns
        elif frame.type_byte == 0x31:
            arrival = BASE_NS
        else:
            decoded_ns = (frame.timestamp - first) * 1000 // 12
            arrival = BASE_NS + (decoded_ns // 20_000_000 + 1) * 20_000_000
        records.append(Record(arrival, frame))
    return records


def connect_all(config: RecorderConfig, at_ns: int) -> None:
    """Record every receiver as connected from ``at_ns``, as the recorder
    would have."""
    lines = [
        {"time_ns": at_ns, "receiver": r.name, "event": "connected"}
        for r in config.receivers
    ]
    (config.run_dir / "events.jsonl").write_text(
        "".join(json.dumps(line) + "\n" for line in lines)
    )
