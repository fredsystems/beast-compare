from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl
from runs import BASE_NS, fixture_records, make_run

from beast_compare.ingest import crc24, load_run, message_bytes

# The classic DF17 example: ICAO 4840D6, parity 576098.
DF17 = bytes.fromhex("8D4840D6202CC371C32CE0576098")


def test_crc24_known_message() -> None:
    message = np.frombuffer(DF17, dtype=np.uint8)[None, :]
    assert crc24(message[:, :11])[0] == 0x576098
    assert crc24(message)[0] == 0


def load_fixtures(tmp_path: Path) -> pl.DataFrame:
    config = make_run(
        tmp_path,
        {
            "airspy": fixture_records("airspy-60s"),
            "fa-proplus": fixture_records("faproplus-60s"),
        },
    )
    ingested = load_run(config.run_dir)
    assert not ingested.truncated_files
    return ingested.frames


def test_counts_match_the_captures(tmp_path: Path) -> None:
    frames = load_fixtures(tmp_path)
    by_receiver = dict(frames.group_by("receiver").len().iter_rows())
    # Mode S frames only: the Pro Stick's five heartbeats are Mode A/C.
    assert by_receiver == {"airspy": 23340, "fa-proplus": 19128}
    airspy = frames.filter(pl.col("receiver") == "airspy")
    df_counts = Counter(airspy["df"].to_list())
    # From an independent parser over the same capture.
    assert df_counts[17] == 11115
    assert df_counts[11] == 5204
    assert df_counts[0] == 4584
    assert df_counts[4] == 1856
    assert df_counts[24] == 7 + 16 + 19 + 15 + 11 + 15 + 23 + 15


def test_every_df17_passes_crc(tmp_path: Path) -> None:
    frames = load_fixtures(tmp_path).filter(pl.col("df") == 17)
    assert len(frames) > 0
    assert (frames["msg_len"] == 14).all()
    assert not crc24(message_bytes(frames)).any()
    assert frames["address"].null_count() == 0
    assert frames["typecode"].null_count() == 0


def test_parity_addresses_are_known_aircraft(tmp_path: Path) -> None:
    frames = load_fixtures(tmp_path)
    for receiver in ("airspy", "fa-proplus"):
        own = frames.filter(pl.col("receiver") == receiver)
        known = set(own.filter(pl.col("df").is_in((11, 17)))["address"].to_list())
        recovered = own.filter(pl.col("df").is_in((0, 4, 5, 20, 21)))["address"]
        # Decoders only accept address/parity replies from aircraft they
        # already know, so a correct CRC recovers a known address.
        assert len(recovered) > 0
        assert recovered.is_in(list(known)).sum() / len(recovered) > 0.99


def test_message_bytes_round_trip(tmp_path: Path) -> None:
    frames = load_fixtures(tmp_path)
    short = frames.filter(pl.col("msg_len") == 7)
    assert not message_bytes(short)[:, 7:].any()
    assert (message_bytes(frames)[:, 0] >> 3 >= frames["df"].to_numpy()).all()


def test_time_window(tmp_path: Path) -> None:
    records = fixture_records("airspy-60s", step_ns=1_000_000)
    config = make_run(tmp_path, {"airspy": records})
    start = BASE_NS + 1000 * 1_000_000
    end = BASE_NS + 2000 * 1_000_000
    frames = load_run(config.run_dir, start, end).frames
    assert len(frames) == 1000
    assert frames["arrival_ns"].min() == start
    assert load_run(config.run_dir, end_ns=BASE_NS - 1).frames.is_empty()


def test_truncated_file_is_reported(tmp_path: Path) -> None:
    config = make_run(tmp_path, {"airspy": fixture_records("airspy-60s")[:100]})
    path = next((config.run_dir / "airspy").glob("*.bcr.zst"))
    path.write_bytes(path.read_bytes()[:-5])
    ingested = load_run(config.run_dir)
    assert [f.path for f in ingested.truncated_files] == [path]


def test_empty_run(tmp_path: Path) -> None:
    config = make_run(tmp_path, {"airspy": [], "fa-proplus": []})
    frames = load_run(config.run_dir).frames
    assert frames.is_empty()
    assert frames.schema["receiver"] == pl.Enum(["airspy", "fa-proplus"])
