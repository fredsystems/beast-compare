"""The ``record`` command: capture every receiver's Beast stream.

Run directory layout::

    <data_dir>/<run_id>/
        manifest.json          configuration and sessions
        events.jsonl           connects, disconnects, session start and stop
        <receiver>/
            YYYYMMDDTHH.bcr.zst   frames, see beast_compare.recordfile
            stats.jsonl           the decoder's stats, polled
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
import urllib.request
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from beast_compare.beast import BeastParser
from beast_compare.config import ReceiverConfig, RecorderConfig, StatsSource
from beast_compare.manifest import begin_session, end_session
from beast_compare.recordfile import Record, RecordWriter

log = logging.getLogger(__name__)

Clock = Callable[[], int]

EVENTS_NAME = "events.jsonl"
STATS_NAME = "stats.jsonl"
CONNECT_TIMEOUT_S = 10.0
MIN_BACKOFF_S = 1.0
MAX_BACKOFF_S = 60.0
READ_SIZE = 1 << 16


def _append_json(path: Path, line: dict[str, Any]) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(line, separators=(",", ":")) + "\n")


class EventLog:
    def __init__(self, path: Path, clock: Clock) -> None:
        self._path = path
        self._clock = clock

    def write(self, receiver: str | None, event: str, **detail: Any) -> None:
        line: dict[str, Any] = {
            "time_ns": self._clock(),
            "receiver": receiver,
            "event": event,
        }
        line.update(detail)
        _append_json(self._path, line)


class ReceiverRecorder:
    """Holds one receiver's connection open and records what it sends."""

    def __init__(
        self,
        receiver: ReceiverConfig,
        run_dir: Path,
        events: EventLog,
        idle_timeout_s: float,
        clock: Clock,
        max_backoff_s: float = MAX_BACKOFF_S,
    ) -> None:
        self.receiver = receiver
        self.writer = RecordWriter(run_dir / receiver.name)
        self._events = events
        self._idle_timeout_s = idle_timeout_s
        self._clock = clock
        self._max_backoff_s = max_backoff_s

    async def run(self) -> None:
        backoff = MIN_BACKOFF_S
        failing = False
        while True:
            try:
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection(self.receiver.host, self.receiver.port),
                    timeout=CONNECT_TIMEOUT_S,
                )
            except (OSError, TimeoutError) as exc:
                if not failing:
                    # Logged once per outage, not on every retry.
                    self._events.write(
                        self.receiver.name, "connect_failed", error=str(exc)
                    )
                    log.warning("%s: connect failed: %s", self.receiver.name, exc)
                    failing = True
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, self._max_backoff_s)
                continue

            failing = False
            backoff = MIN_BACKOFF_S
            await self._session(reader, writer)
            await asyncio.sleep(MIN_BACKOFF_S)

    async def _session(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        name = self.receiver.name
        self._events.write(name, "connected")
        log.info("%s: connected", name)
        parser = BeastParser()
        heartbeats = 0
        reason = "eof"
        # The sending side is never closed: readsb drops a client that
        # half-closes its connection.
        try:
            while True:
                data = await asyncio.wait_for(
                    reader.read(READ_SIZE), timeout=self._idle_timeout_s
                )
                if not data:
                    break
                arrival_ns = self._clock()
                for frame in parser.feed(data):
                    if frame.is_heartbeat:
                        heartbeats += 1
                        continue
                    self.writer.write(Record(arrival_ns, frame))
        except TimeoutError:
            reason = "idle timeout"
        except OSError as exc:
            reason = f"error: {exc}"
        except asyncio.CancelledError:
            reason = "stopped"
            raise
        finally:
            writer.close()
            with contextlib.suppress(OSError):
                await writer.wait_closed()
            self._events.write(
                name,
                "disconnected",
                reason=reason,
                heartbeats=heartbeats,
                **asdict(parser.stats),
            )
            log.info("%s: disconnected (%s)", name, reason)

    def close(self) -> None:
        self.writer.close()


def fetch_stats(source: StatsSource) -> str:
    if source.url is not None:
        with urllib.request.urlopen(source.url, timeout=10) as response:
            body: bytes = response.read()
            return body.decode()
    assert source.file is not None
    return source.file.read_text()


async def poll_stats(
    receiver: ReceiverConfig, run_dir: Path, interval_s: float, clock: Clock
) -> None:
    assert receiver.stats is not None
    path = run_dir / receiver.name / STATS_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    while True:
        line: dict[str, Any] = {"time_ns": clock(), "receiver": receiver.name}
        try:
            text = await asyncio.to_thread(fetch_stats, receiver.stats)
            line["stats"] = json.loads(text)
            line["error"] = None
        except (OSError, ValueError) as exc:
            line["stats"] = None
            line["error"] = str(exc)
        _append_json(path, line)
        await asyncio.sleep(interval_s)


async def flush_periodically(
    recorders: list[ReceiverRecorder], interval_s: float
) -> None:
    while True:
        await asyncio.sleep(interval_s)
        for recorder in recorders:
            recorder.writer.flush()


async def record(
    config: RecorderConfig,
    stop: asyncio.Event,
    clock: Clock = time.time_ns,
    max_backoff_s: float = MAX_BACKOFF_S,
) -> None:
    """Record until ``stop`` is set.

    Each receiver is handled independently: one being unreachable never
    stops the others. A bug that ends any task, though, ends the run, so
    that the service manager restarts it instead of a receiver silently
    going unrecorded.
    """
    session = begin_session(config)
    run_dir = config.run_dir
    events = EventLog(run_dir / EVENTS_NAME, clock)
    events.write(None, "session_start", session=session)

    recorders = [
        ReceiverRecorder(
            receiver, run_dir, events, config.idle_timeout_s, clock, max_backoff_s
        )
        for receiver in config.receivers
    ]
    tasks = [asyncio.create_task(r.run(), name=r.receiver.name) for r in recorders]
    tasks += [
        asyncio.create_task(
            poll_stats(r, run_dir, config.stats_interval_s, clock),
            name=f"{r.name}-stats",
        )
        for r in config.receivers
        if r.stats is not None
    ]
    tasks.append(
        asyncio.create_task(flush_periodically(recorders, config.flush_interval_s))
    )
    stopper = asyncio.create_task(stop.wait())

    try:
        done, _ = await asyncio.wait(
            [stopper, *tasks], return_when=asyncio.FIRST_COMPLETED
        )
        for task in done:
            if task is not stopper:
                task.result()
                raise RuntimeError(f"task {task.get_name()} ended unexpectedly")
    finally:
        for task in [stopper, *tasks]:
            task.cancel()
        await asyncio.gather(stopper, *tasks, return_exceptions=True)
        for recorder in recorders:
            recorder.close()
        events.write(None, "session_stop", session=session)
        end_session(config, session)
