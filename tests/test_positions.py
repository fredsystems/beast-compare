from pathlib import Path

import numpy as np
import polars as pl
import pyModeS
import pytest
from frames import HZ
from runs import fixture_records, make_run

from beast_compare.align import align
from beast_compare.cluster import cluster
from beast_compare.geo import Site, look_angles
from beast_compare.ingest import load_run, message_bytes
from beast_compare.positions import (
    airborne_messages,
    altitude_ft,
    attribute,
    cpr_global,
    decode_positions,
    field,
    nl,
)

# pyModeS's documented example pair: even at t=1457996402, odd at t=...400.
EVEN = "8D40621D58C382D690C8AC2863A7"
ODD = "8D40621D58C386435CC412692AD6"
# Somewhere with real traffic for the fixture captures, not the site.
SITE = Site(lat=35.0, lon=-106.6, alt_m=1600.0)


def as_bytes(*messages: str) -> np.ndarray:
    return np.array([list(bytes.fromhex(m)) for m in messages], dtype=np.uint8)


def test_fields_and_altitude() -> None:
    m = as_bytes(EVEN, ODD)
    assert field(m, 1, 5).tolist() == [17, 17]
    assert field(m, 54, 54).tolist() == [0, 1]
    typecode = field(m, 33, 37).astype(np.uint8)
    assert altitude_ft(typecode, field(m, 41, 52)).tolist() == [38000.0, 38000.0]


def test_cpr_matches_pymodes_example() -> None:
    m = as_bytes(EVEN, ODD)
    lat_cpr = field(m, 55, 71).astype(float) / (1 << 17)
    lon_cpr = field(m, 72, 88).astype(float) / (1 << 17)
    lat, lon = cpr_global(
        lat_cpr[:1], lon_cpr[:1], lat_cpr[1:], lon_cpr[1:], np.array([False])
    )
    expected = pyModeS.adsb.position(EVEN, ODD, 1457996402, 1457996400)
    assert expected is not None
    assert lat[0] == pytest.approx(expected[0], abs=1e-6)
    assert lon[0] == pytest.approx(expected[1], abs=1e-6)


def test_nl_matches_pymodes() -> None:
    lats = np.linspace(-90, 90, 3601)
    ours = nl(lats)
    theirs = [pyModeS.common.cprNL(float(x)) for x in lats]
    assert ours.tolist() == theirs


def transmissions_from_fixtures(tmp_path: Path) -> pl.DataFrame:
    config = make_run(
        tmp_path,
        {
            "airspy": fixture_records("airspy-60s"),
            "fa-proplus": fixture_records("faproplus-60s"),
        },
    )
    aligned = align(load_run(config.run_dir).frames, "airspy").frames
    return cluster(aligned).transmissions


def test_decoded_positions_match_pymodes(tmp_path: Path) -> None:
    transmissions = transmissions_from_fixtures(tmp_path)
    positions = decode_positions(transmissions, SITE)
    assert positions["address"].n_unique() > 20

    # Check every decoded position against pyModeS decoding the message with
    # the aircraft's previous position as reference.
    by_tx = transmissions.filter(pl.col("tx").is_in(positions["tx"].implode()))
    hex_of = {
        tx: bytes(row).hex()
        for tx, row in zip(by_tx["tx"].to_list(), message_bytes(by_tx), strict=True)
    }
    checked = 0
    for address, track in positions.sort("t").group_by("address"):
        rows = track.rows(named=True)
        for previous, current in zip(rows, rows[1:], strict=False):
            lat, lon = pyModeS.adsb.airborne_position_with_ref(
                hex_of[current["tx"]], previous["lat"], previous["lon"]
            )
            assert current["lat"] == pytest.approx(lat, abs=1e-4), address
            assert current["lon"] == pytest.approx(lon, abs=1e-4), address
            checked += 1
    assert checked > 500


def test_airborne_altitudes_match_pymodes(tmp_path: Path) -> None:
    transmissions = transmissions_from_fixtures(tmp_path)
    airborne = airborne_messages(transmissions).filter(pl.col("alt_ft").is_not_null())
    sample = transmissions.join(airborne.select("tx", "alt_ft"), on="tx").head(300)
    for message, alt in zip(
        message_bytes(sample), sample["alt_ft"].to_list(), strict=True
    ):
        assert alt == pyModeS.adsb.altitude(bytes(message).hex())


def test_look_angles() -> None:
    site = Site(lat=35.0, lon=-106.0, alt_m=0.0)
    r, b, e = look_angles(
        site,
        np.array([36.0, 35.0, 35.0]),
        np.array([-106.0, -106.0, -105.0]),
        np.array([0.0, 10_000.0, 0.0]),
    )
    assert r[0] == pytest.approx(60.0, rel=0.01)
    assert b[0] == pytest.approx(0.0, abs=0.01)
    assert e[0] < 0  # a target at sea level 60 nm away is below the horizon
    assert e[1] == pytest.approx(90.0, abs=0.01)
    assert b[2] == pytest.approx(90.0, abs=0.5)


def test_attribute() -> None:
    track = pl.DataFrame(
        {
            "tx": [100, 101],
            "ref_segment": [0, 0],
            "address": [0xABCDEF, 0xABCDEF],
            "t": [0.0, 20.0],
            "alt_ft": [10_000.0, 12_000.0],
            "lat": [35.0, 35.2],
            "lon": [-106.0, -106.0],
        },
        schema_overrides={"ref_segment": pl.UInt32, "address": pl.UInt32},
    )
    # Transmissions at 10 s (bracketed), 25 s (5 s after the last position),
    # 45 s (25 s after), and one from an aircraft with no track.
    times = [10.0, 25.0, 45.0, 10.0]
    transmissions = pl.DataFrame(
        {
            "tx": [1, 2, 3, 4],
            "ref_segment": [0, 0, 0, 0],
            "address": [0xABCDEF, 0xABCDEF, 0xABCDEF, 0x123456],
            "t_ref": [t * HZ for t in times],
        },
        schema_overrides={"ref_segment": pl.UInt32, "address": pl.UInt32},
    )
    site = Site(lat=35.0, lon=-106.0, alt_m=0.0)
    placed = attribute(transmissions, track, site)
    assert placed["lat"].to_list()[:2] == pytest.approx([35.1, 35.2])
    assert placed["alt_ft"].to_list()[:2] == pytest.approx([11_000.0, 12_000.0])
    assert placed["lat"].to_list()[2:] == [None, None]
    assert placed["range_nm"][0] == pytest.approx(6.0, rel=0.01)
