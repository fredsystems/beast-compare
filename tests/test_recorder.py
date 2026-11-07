import asyncio
import json
import socket
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from beast_compare.beast import Frame, FrameType, escape
from beast_compare.cli import main
from beast_compare.config import RecorderConfig, StatsSource, parse_config
from beast_compare.manifest import read_manifest
from beast_compare.recorder import EVENTS_NAME, STATS_NAME, fetch_stats, record
from beast_compare.recordfile import RecordReader, recording_files

HEARTBEAT = Frame(FrameType.MODE_AC, bytes(9))


def frame(n: int) -> Frame:
    body = (
        n.to_bytes(6, "big") + b"\x40" + bytes.fromhex("8d4ca2d158c901a0c0b4a0f1a2b3")
    )
    return Frame(FrameType.MODE_S_LONG, body)


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port: int = sock.getsockname()[1]
    return port


def make_config(tmp_path: Path, receivers: list[dict[str, Any]]) -> RecorderConfig:
    return parse_config(
        {
            "run_id": "test-run",
            "data_dir": str(tmp_path),
            "stats_interval_s": 0.05,
            "flush_interval_s": 0.05,
            "idle_timeout_s": 30,
            "receivers": receivers,
        }
    )


def recorded_frames(directory: Path) -> list[Frame]:
    return [r.frame for p in recording_files(directory) for r in RecordReader(p)]


def events(run_dir: Path) -> list[dict[str, Any]]:
    lines = (run_dir / EVENTS_NAME).read_text().splitlines()
    return [json.loads(line) for line in lines]


class FakeDecoder:
    """A Beast server: the first connection sends a batch and hangs up, the
    second sends another batch and stays open."""

    def __init__(self) -> None:
        self.batches = [
            [HEARTBEAT, frame(1), frame(2)],
            [frame(3), frame(4)],
        ]
        self.connections = 0
        self.client_half_closed: bool | None = None
        self.done = asyncio.Event()

    async def handle(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        index = self.connections
        self.connections += 1
        batch = self.batches[index]
        # Split mid-frame to make the recorder reassemble across reads.
        data = b"".join(escape(f) for f in batch)
        writer.write(data[:5])
        await writer.drain()
        await asyncio.sleep(0.02)
        writer.write(data[5:])
        await writer.drain()
        if index == 0:
            writer.close()
            return
        try:
            await asyncio.wait_for(reader.read(1), timeout=0.3)
            self.client_half_closed = True
        except TimeoutError:
            self.client_half_closed = False
        self.done.set()
        await asyncio.sleep(3600)


def test_records_reconnects_and_never_half_closes(tmp_path: Path) -> None:
    stats_file = tmp_path / "decoder-stats.json"
    stats_file.write_text(json.dumps({"gain_db": 7.7}))

    async def scenario() -> tuple[FakeDecoder, RecorderConfig]:
        decoder = FakeDecoder()
        server = await asyncio.start_server(decoder.handle, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        config = make_config(
            tmp_path,
            [
                {
                    "name": "decoder",
                    "beast": f"127.0.0.1:{port}",
                    "stats": {"file": str(stats_file)},
                },
                {"name": "unreachable", "beast": f"127.0.0.1:{free_port()}"},
            ],
        )
        stop = asyncio.Event()
        task = asyncio.create_task(record(config, stop, max_backoff_s=0.05))
        await asyncio.wait_for(decoder.done.wait(), timeout=10)
        await asyncio.sleep(0.2)
        stop.set()
        await asyncio.wait_for(task, timeout=10)
        server.close()
        return decoder, config

    decoder, config = asyncio.run(scenario())
    run_dir = config.run_dir

    assert decoder.client_half_closed is False
    assert recorded_frames(run_dir / "decoder") == [frame(n) for n in (1, 2, 3, 4)]

    log = events(run_dir)
    by_kind = [(e["receiver"], e["event"]) for e in log]
    assert by_kind[0] == (None, "session_start")
    assert by_kind[-1] == (None, "session_stop")
    assert by_kind.count(("decoder", "connected")) == 2
    disconnects = [e for e in log if e["event"] == "disconnected"]
    assert [d["reason"] for d in disconnects] == ["eof", "stopped"]
    assert disconnects[0]["heartbeats"] == 1
    # Retried many times, but logged once.
    assert by_kind.count(("unreachable", "connect_failed")) == 1

    stats_lines = (run_dir / "decoder" / STATS_NAME).read_text().splitlines()
    assert stats_lines
    assert json.loads(stats_lines[0])["stats"] == {"gain_db": 7.7}

    session = read_manifest(run_dir)["sessions"][0]
    assert session["stop"] is not None


def test_idle_connection_is_dropped(tmp_path: Path) -> None:
    async def silent(_r: asyncio.StreamReader, _w: asyncio.StreamWriter) -> None:
        await asyncio.sleep(3600)

    async def scenario() -> RecorderConfig:
        server = await asyncio.start_server(silent, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        config = parse_config(
            {
                "run_id": "idle",
                "data_dir": str(tmp_path),
                "idle_timeout_s": 0.1,
                "receivers": [{"name": "quiet", "beast": f"127.0.0.1:{port}"}],
            }
        )
        stop = asyncio.Event()
        task = asyncio.create_task(record(config, stop))
        await asyncio.sleep(0.4)
        stop.set()
        await asyncio.wait_for(task, timeout=10)
        server.close()
        return config

    config = asyncio.run(scenario())
    reasons = [e["reason"] for e in events(config.run_dir) if "reason" in e]
    assert reasons[0] == "idle timeout"


@pytest.fixture
def stats_server() -> Iterator[str]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            body = json.dumps({"gain_db": 0.0}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}/data/stats.json"
    server.shutdown()


def test_fetch_stats_over_http(stats_server: str) -> None:
    assert json.loads(fetch_stats(StatsSource(url=stats_server))) == {"gain_db": 0.0}


def test_cli_rejects_a_bad_config(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "recorder.yaml"
    path.write_text("run_id: r1\n")
    assert main(["record", "--config", str(path)]) == 2
    assert "receivers" in capsys.readouterr().err
