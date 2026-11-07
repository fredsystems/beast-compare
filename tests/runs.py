"""Build recorded runs on disk for tests."""

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


def fixture_records(name: str, step_ns: int = 1_000_000) -> list[Record]:
    """A capture's frames as records, ``step_ns`` apart from BASE_NS."""
    frames: list[Frame] = BeastParser().feed(fixture_stream(name))
    return [Record(BASE_NS + i * step_ns, f) for i, f in enumerate(frames)]
