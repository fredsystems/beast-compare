"""Streaming parser for the Beast binary protocol.

A Beast frame is ``0x1a``, a type byte, then a body. Inside the body every
``0x1a`` is doubled, so an undoubled ``0x1a`` always starts a new frame.
For the Mode S and Mode A/C types the body is a 6-byte timestamp, a 1-byte
signal level and the message itself. Other types (receiver IDs, status) are
kept as opaque bodies running up to the next frame start.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

SYNC = 0x1A
TIMESTAMP_LEN = 6
SIGNAL_LEN = 1
MESSAGE_OFFSET = TIMESTAMP_LEN + SIGNAL_LEN

# The recorder stores a frame as type byte plus body behind a one-byte
# length, so an opaque body longer than this cannot be recorded.
MAX_FRAME_LEN = 255


class FrameType(enum.IntEnum):
    MODE_AC = 0x31
    MODE_S_SHORT = 0x32
    MODE_S_LONG = 0x33


MESSAGE_LENGTHS: dict[FrameType, int] = {
    FrameType.MODE_AC: 2,
    FrameType.MODE_S_SHORT: 7,
    FrameType.MODE_S_LONG: 14,
}


def _known_type(type_byte: int) -> FrameType | None:
    try:
        return FrameType(type_byte)
    except ValueError:
        return None


def _body_length(frame_type: FrameType) -> int:
    return MESSAGE_OFFSET + MESSAGE_LENGTHS[frame_type]


@dataclass(frozen=True, slots=True)
class Frame:
    """One unescaped Beast frame."""

    type_byte: int
    body: bytes

    @property
    def frame_type(self) -> FrameType | None:
        return _known_type(self.type_byte)

    def _require_known(self) -> FrameType:
        frame_type = self.frame_type
        if frame_type is None:
            raise ValueError(f"frame type 0x{self.type_byte:02x} has no fields")
        return frame_type

    @property
    def timestamp(self) -> int:
        """The receiver's counter, normally 12 MHz."""
        self._require_known()
        return int.from_bytes(self.body[:TIMESTAMP_LEN], "big")

    @property
    def signal(self) -> int:
        self._require_known()
        return self.body[TIMESTAMP_LEN]

    @property
    def message(self) -> bytes:
        self._require_known()
        return self.body[MESSAGE_OFFSET:]

    @property
    def is_heartbeat(self) -> bool:
        """readsb sends all-zero Mode A/C frames as keepalives."""
        return self.type_byte == FrameType.MODE_AC and not any(self.body)

    def encode(self) -> bytes:
        """Type byte plus body, unescaped: the form the recorder stores."""
        return bytes((self.type_byte,)) + self.body

    @classmethod
    def decode(cls, raw: bytes) -> Frame:
        if not raw:
            raise ValueError("empty frame")
        frame = cls(raw[0], bytes(raw[1:]))
        frame_type = frame.frame_type
        if frame_type is not None and len(frame.body) != _body_length(frame_type):
            raise ValueError(
                f"frame type 0x{frame.type_byte:02x} needs a "
                f"{_body_length(frame_type)}-byte body, got {len(frame.body)}"
            )
        return frame


@dataclass(slots=True)
class ParserStats:
    frames: int = 0
    # A known frame cut short by the start of the next one.
    truncated: int = 0
    # Opaque frames too long to record.
    oversized: int = 0
    # Bytes outside any frame, skipped while looking for a frame start.
    skipped_bytes: int = 0


class _State(enum.Enum):
    SEEK = enum.auto()
    TYPE = enum.auto()
    BODY = enum.auto()


@dataclass(slots=True)
class BeastParser:
    """Incremental parser: feed it bytes as they arrive, in any chunking."""

    stats: ParserStats = field(default_factory=ParserStats)
    _state: _State = _State.SEEK
    _type_byte: int = 0
    _expected: int | None = None
    _body: bytearray = field(default_factory=bytearray)
    _escape_pending: bool = False

    def feed(self, data: bytes) -> list[Frame]:
        frames: list[Frame] = []
        for byte in data:
            if self._state is _State.SEEK:
                if byte == SYNC:
                    self._state = _State.TYPE
                else:
                    self.stats.skipped_bytes += 1
            elif self._state is _State.TYPE:
                self._start_frame(byte)
            else:
                self._body_byte(byte, frames)
        return frames

    def _start_frame(self, type_byte: int) -> None:
        if type_byte == SYNC:
            # An escaped 0x1a with no frame around it: still out of sync.
            self.stats.skipped_bytes += 2
            self._state = _State.SEEK
            return
        frame_type = _known_type(type_byte)
        self._type_byte = type_byte
        self._expected = None if frame_type is None else _body_length(frame_type)
        self._body.clear()
        self._escape_pending = False
        self._state = _State.BODY

    def _body_byte(self, byte: int, frames: list[Frame]) -> None:
        if self._escape_pending:
            self._escape_pending = False
            if byte == SYNC:
                self._append(SYNC, frames)
                return
            # An undoubled 0x1a: the current frame has ended and this byte
            # is the type of the next one.
            self._end_early(frames)
            self._start_frame(byte)
            return
        if byte == SYNC:
            self._escape_pending = True
            return
        self._append(byte, frames)

    def _append(self, byte: int, frames: list[Frame]) -> None:
        self._body.append(byte)
        if self._expected is not None and len(self._body) == self._expected:
            frames.append(Frame(self._type_byte, bytes(self._body)))
            self.stats.frames += 1
            self._state = _State.SEEK

    def _end_early(self, frames: list[Frame]) -> None:
        if self._expected is not None:
            self.stats.truncated += 1
        elif len(self._body) + 1 > MAX_FRAME_LEN:
            self.stats.oversized += 1
        else:
            frames.append(Frame(self._type_byte, bytes(self._body)))
            self.stats.frames += 1


def escape(frame: Frame) -> bytes:
    """Encode a frame for the wire, doubling every 0x1a in the body."""
    body = frame.body.replace(bytes((SYNC,)), bytes((SYNC, SYNC)))
    return bytes((SYNC, frame.type_byte)) + body
