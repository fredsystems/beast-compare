from collections import Counter

import pytest
from conftest import fixture_stream

from beast_compare.beast import BeastParser, Frame, FrameType, escape


def mode_s(timestamp: int, signal: int, message: bytes) -> Frame:
    frame_type = FrameType.MODE_S_LONG if len(message) == 14 else FrameType.MODE_S_SHORT
    return Frame(frame_type, timestamp.to_bytes(6, "big") + bytes((signal,)) + message)


LONG = bytes.fromhex("8d4ca2d158c901a0c0b4a0f1a2b3")
SHORT = bytes.fromhex("5d4ca2d1a2b3c4")

# 0x1a inside the timestamp, the signal and the message, so escaping is
# exercised in every field.
FRAMES = [
    mode_s(0x1A1A0000001A, 0x1A, LONG),
    mode_s(0x000102030405, 0x80, SHORT),
    mode_s(0x0000001A0001, 0x10, bytes.fromhex("1a1a1a1a1a1a1a1a1a1a1a1a1a1a")),
    mode_s(0x0000000000FF, 0x7F, bytes.fromhex("8d1a")[:1] + SHORT[1:]),
]
STREAM = b"".join(escape(f) for f in FRAMES)


def test_fields() -> None:
    frame = FRAMES[0]
    assert frame.frame_type is FrameType.MODE_S_LONG
    assert frame.timestamp == 0x1A1A0000001A
    assert frame.signal == 0x1A
    assert frame.message == LONG


def test_round_trip_in_one_chunk() -> None:
    parser = BeastParser()
    assert parser.feed(STREAM) == FRAMES
    assert parser.stats.frames == len(FRAMES)
    assert parser.stats.truncated == 0
    assert parser.stats.skipped_bytes == 0


@pytest.mark.parametrize("split", range(1, len(STREAM)))
def test_any_split_point(split: int) -> None:
    parser = BeastParser()
    frames = parser.feed(STREAM[:split]) + parser.feed(STREAM[split:])
    assert frames == FRAMES


def test_one_byte_at_a_time() -> None:
    parser = BeastParser()
    frames = [f for b in STREAM for f in parser.feed(bytes((b,)))]
    assert frames == FRAMES


def test_skips_garbage_before_first_frame() -> None:
    parser = BeastParser()
    assert parser.feed(b"\x00\x01\x02" + STREAM) == FRAMES
    assert parser.stats.skipped_bytes == 3


def test_truncated_frame_is_dropped_and_parsing_resumes() -> None:
    cut = escape(FRAMES[0])[:9]
    parser = BeastParser()
    assert parser.feed(cut + escape(FRAMES[1])) == [FRAMES[1]]
    assert parser.stats.truncated == 1


def test_heartbeat() -> None:
    heartbeat = Frame(FrameType.MODE_AC, bytes(9))
    assert heartbeat.is_heartbeat
    assert not Frame(FrameType.MODE_AC, bytes(8) + b"\x01").is_heartbeat
    assert not FRAMES[0].is_heartbeat
    assert BeastParser().feed(escape(heartbeat)) == [heartbeat]


def test_unknown_type_ends_at_next_frame() -> None:
    receiver_id = Frame(0xE3, bytes.fromhex("0011223344556677"))
    parser = BeastParser()
    # Not emitted until the next frame start shows where it ends.
    assert parser.feed(escape(receiver_id)) == []
    assert parser.feed(escape(FRAMES[0])) == [receiver_id, FRAMES[0]]
    with pytest.raises(ValueError):
        _ = receiver_id.timestamp


def test_oversized_unknown_frame_is_counted_not_emitted() -> None:
    parser = BeastParser()
    frames = parser.feed(escape(Frame(0xE4, bytes(300))) + escape(FRAMES[0]))
    assert frames == [FRAMES[0]]
    assert parser.stats.oversized == 1


def test_decode_round_trip_and_validation() -> None:
    for frame in FRAMES:
        assert Frame.decode(frame.encode()) == frame
    with pytest.raises(ValueError):
        Frame.decode(b"")
    with pytest.raises(ValueError):
        Frame.decode(FRAMES[0].encode()[:-1])


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        # Counts from an independent parser run over the same captures.
        ("airspy-60s", {0x32: 11707, 0x33: 11633}),
        ("faproplus-60s", {0x31: 5, 0x32: 9478, 0x33: 9650}),
    ],
)
def test_real_captures(name: str, expected: dict[int, int]) -> None:
    parser = BeastParser()
    frames = parser.feed(fixture_stream(name))
    assert dict(Counter(f.type_byte for f in frames)) == expected
    assert parser.stats.truncated == 0
    # readsb opens a connection with a burst of heartbeats.
    assert all(f.is_heartbeat for f in frames if f.type_byte == FrameType.MODE_AC)
