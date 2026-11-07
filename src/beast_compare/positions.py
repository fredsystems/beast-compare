"""Aircraft positions, and a position for every transmission.

Positions come from DF17/18 airborne position messages (type codes 9-18
with barometric altitude, 20-22 with GNSS height), decoded globally from
an even/odd CPR pair of the same aircraft no more than ``pair_window_s``
apart. Decoding is vectorised; surface positions (type codes 5-8) are not
decoded, so ground traffic has no position.

Every transmission with an address, from any receiver and of any downlink
format, is then placed on its aircraft's track: interpolated between the
positions either side of it when both are within ``max_bracket_s``, else
taken from the nearest position within ``max_nearest_s``, else left without
a position. A receiver that only caught a DF11 from an aircraft at 180 nm is
thereby credited at 180 nm.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import polars as pl

from beast_compare.align import NOMINAL_HZ
from beast_compare.geo import FEET, NAUTICAL_MILE, Site, ground_range_nm, look_angles
from beast_compare.ingest import Bytes, message_bytes

Floats = npt.NDArray[np.float64]
U64 = npt.NDArray[np.uint64]

CPR_SCALE = float(1 << 17)
METRES_TO_FEET = 1 / FEET


@dataclass(frozen=True, slots=True)
class PositionSettings:
    pair_window_s: float = 10.0
    max_range_nm: float = 450.0
    max_speed_kt: float = 1500.0
    max_bracket_s: float = 30.0
    max_nearest_s: float = 10.0


def field(messages: Bytes, first: int, last: int) -> U64:
    """Bits ``first``-``last`` (1-based, inclusive) of each message."""
    first_byte = (first - 1) // 8
    last_byte = (last - 1) // 8
    value = np.zeros(messages.shape[0], dtype=np.uint64)
    for byte in range(first_byte, last_byte + 1):
        value = (value << np.uint64(8)) | messages[:, byte].astype(np.uint64)
    shift = np.uint64((last_byte + 1) * 8 - last)
    mask = np.uint64((1 << (last - first + 1)) - 1)
    result: U64 = (value >> shift) & mask
    return result


def altitude_ft(typecode: npt.NDArray[np.uint8], code: U64) -> Floats:
    """Altitude from an airborne position's 12-bit altitude field.

    Barometric altitude with the Q bit set is in 25 ft steps. Gillham-coded
    altitudes (Q bit clear) are rare in ADS-B and come out NaN. GNSS height
    (type codes 20-22) is in metres.
    """
    q_bit = (code & np.uint64(0x010)) != 0
    n = ((code & np.uint64(0xFE0)) >> np.uint64(1)) | (code & np.uint64(0x00F))
    baro = np.where(q_bit, n.astype(np.float64) * 25 - 1000, np.nan)
    gnss = code.astype(np.float64) * METRES_TO_FEET
    result: Floats = np.where((typecode >= 20) & (typecode <= 22), gnss, baro)
    return result


def nl(lat: Floats) -> Floats:
    """Number of longitude zones at a latitude (CPR NL function)."""
    a = 1 - np.cos(np.pi / 30)
    b = np.cos(np.radians(np.abs(lat))) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        zones = np.floor(2 * np.pi / np.arccos(1 - a / b))
    zones = np.where(np.abs(lat) > 87, 1.0, zones)
    zones = np.where(np.abs(lat) == 87, 2.0, zones)
    result: Floats = np.where(lat == 0, 59.0, zones)
    return result


def cpr_global(
    lat_even: Floats,
    lon_even: Floats,
    lat_odd: Floats,
    lon_odd: Floats,
    use_odd: npt.NDArray[np.bool_],
) -> tuple[Floats, Floats]:
    """Global airborne CPR decode of even/odd pairs (CPR fields as fractions).

    The position returned is the odd message's where ``use_odd``, else the
    even one's. NaN where the pair straddles a longitude-zone boundary.
    """
    j = np.floor(59 * lat_even - 60 * lat_odd + 0.5)
    rlat_even = (360 / 60) * (np.mod(j, 60) + lat_even)
    rlat_odd = (360 / 59) * (np.mod(j, 59) + lat_odd)
    rlat_even = np.where(rlat_even >= 270, rlat_even - 360, rlat_even)
    rlat_odd = np.where(rlat_odd >= 270, rlat_odd - 360, rlat_odd)

    lat = np.where(use_odd, rlat_odd, rlat_even)
    zones = nl(lat)
    consistent = nl(rlat_even) == nl(rlat_odd)
    ni = np.maximum(np.where(use_odd, zones - 1, zones), 1)
    m = np.floor(lon_even * (zones - 1) - lon_odd * zones + 0.5)
    own_lon = np.where(use_odd, lon_odd, lon_even)
    lon = (360 / ni) * (np.mod(m, ni) + own_lon)
    lon = np.where(lon >= 180, lon - 360, lon)
    return np.where(consistent, lat, np.nan), np.where(consistent, lon, np.nan)


def airborne_messages(transmissions: pl.DataFrame) -> pl.DataFrame:
    """Airborne position transmissions with their CPR fields decoded."""
    candidates = transmissions.filter(
        pl.col("df").is_in((17, 18))
        & (pl.col("typecode").is_between(9, 18) | pl.col("typecode").is_between(20, 22))
    )
    messages = message_bytes(candidates)
    typecode = candidates["typecode"].to_numpy().astype(np.uint8)
    return candidates.select(
        "tx", "ref_segment", "address", t=pl.col("t_ref") / NOMINAL_HZ
    ).with_columns(
        odd=pl.Series(field(messages, 54, 54) == 1),
        lat_cpr=pl.Series(field(messages, 55, 71).astype(np.float64) / CPR_SCALE),
        lon_cpr=pl.Series(field(messages, 72, 88).astype(np.float64) / CPR_SCALE),
        alt_ft=pl.Series(altitude_ft(typecode, field(messages, 41, 52))).fill_nan(None),
    )


def _pair(own: pl.DataFrame, opposite: pl.DataFrame, window_s: float) -> pl.DataFrame:
    """Each message with the nearest opposite-parity message, preferring the
    latest one before it and falling back to the first one after."""
    other = opposite.select(
        "ref_segment",
        "address",
        "t",
        t_other=pl.col("t"),
        lat_other=pl.col("lat_cpr"),
        lon_other=pl.col("lon_cpr"),
    ).sort("t")
    own = own.sort("t")
    keys = ["ref_segment", "address"]
    # Both sides are sorted by t overall, hence within every group too.
    before = own.join_asof(
        other,
        on="t",
        by=keys,
        strategy="backward",
        tolerance=window_s,
        check_sortedness=False,
    )
    after = own.join_asof(
        other,
        on="t",
        by=keys,
        strategy="forward",
        tolerance=window_s,
        check_sortedness=False,
    )
    return before.with_columns(
        t_other=pl.coalesce(before["t_other"], after["t_other"]),
        lat_other=pl.coalesce(before["lat_other"], after["lat_other"]),
        lon_other=pl.coalesce(before["lon_other"], after["lon_other"]),
    ).filter(pl.col("t_other").is_not_null())


def floats(series: pl.Series) -> Floats:
    """A series as float64, nulls as NaN."""
    result: Floats = series.cast(pl.Float64).fill_null(np.nan).to_numpy()
    return result


def distance_nm(lat1: Floats, lon1: Floats, lat2: Floats, lon2: Floats) -> Floats:
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dlam = np.radians(lon2 - lon1)
    h = (
        np.sin((phi2 - phi1) / 2) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    )
    result: Floats = 2 * 6_371_008.8 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))
    return result / NAUTICAL_MILE


def decode_positions(
    transmissions: pl.DataFrame,
    site: Site,
    settings: PositionSettings = PositionSettings(),
) -> pl.DataFrame:
    """Valid airborne positions: tx, ref_segment, address, t (s), lat, lon,
    alt_ft (null when Gillham-coded)."""
    airborne = airborne_messages(transmissions)
    even = airborne.filter(~pl.col("odd"))
    odd = airborne.filter(pl.col("odd"))
    paired = pl.concat(
        [
            _pair(even, odd, settings.pair_window_s),
            _pair(odd, even, settings.pair_window_s),
        ]
    )
    is_odd = paired["odd"].to_numpy()
    own_lat = paired["lat_cpr"].to_numpy()
    own_lon = paired["lon_cpr"].to_numpy()
    other_lat = paired["lat_other"].to_numpy()
    other_lon = paired["lon_other"].to_numpy()
    lat, lon = cpr_global(
        np.where(is_odd, other_lat, own_lat),
        np.where(is_odd, other_lon, own_lon),
        np.where(is_odd, own_lat, other_lat),
        np.where(is_odd, own_lon, other_lon),
        is_odd,
    )
    positions = (
        paired.select("tx", "ref_segment", "address", "t", "alt_ft")
        .with_columns(lat=pl.Series(lat), lon=pl.Series(lon))
        .filter(pl.col("lat").is_not_nan())
    )
    in_range = (
        ground_range_nm(site, positions["lat"].to_numpy(), positions["lon"].to_numpy())
        <= settings.max_range_nm
    )
    positions = positions.filter(pl.Series(in_range))
    return _drop_jumps(positions, settings.max_speed_kt).sort("ref_segment", "t")


def _drop_jumps(positions: pl.DataFrame, max_speed_kt: float) -> pl.DataFrame:
    """Drop positions implausibly far from both neighbours on the track
    (or from the only neighbour): a wrong CPR decode."""
    keys = ["ref_segment", "address"]
    ordered = positions.sort(*keys, "t")
    neighbours = ordered.with_columns(
        [
            pl.col(c).shift(n).over(keys).alias(f"{c}_{name}")
            for c in ("t", "lat", "lon")
            for n, name in ((1, "prev"), (-1, "next"))
        ]
    )

    def speed(side: str) -> Floats:
        dt = np.abs(neighbours["t"].to_numpy() - floats(neighbours[f"t_{side}"]))
        d = distance_nm(
            neighbours["lat"].to_numpy(),
            neighbours["lon"].to_numpy(),
            floats(neighbours[f"lat_{side}"]),
            floats(neighbours[f"lon_{side}"]),
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            # Within 1 s, CPR quantisation dominates: allow 1 nm of slack.
            kt: Floats = np.maximum(d - 1, 0) / np.maximum(dt, 1e-9) * 3600
        return kt

    fast_prev = speed("prev") > max_speed_kt
    fast_next = speed("next") > max_speed_kt
    has_prev = neighbours["t_prev"].is_not_null().to_numpy()
    has_next = neighbours["t_next"].is_not_null().to_numpy()
    jump = np.where(
        has_prev & has_next,
        fast_prev & fast_next,
        np.where(has_prev, fast_prev, np.where(has_next, fast_next, False)),
    )
    return ordered.filter(pl.Series(~jump))


def attribute(
    transmissions: pl.DataFrame,
    positions: pl.DataFrame,
    site: Site,
    settings: PositionSettings = PositionSettings(),
) -> pl.DataFrame:
    """Each transmission's tx with lat, lon, alt_ft, range_nm, bearing and
    elevation, all null where it cannot be placed."""
    keys = ["ref_segment", "address"]
    targets = (
        transmissions.filter(pl.col("address").is_not_null())
        .select("tx", "ref_segment", "address", t=pl.col("t_ref") / NOMINAL_HZ)
        .sort("t")
    )
    track = positions.select(
        *keys, "t", t_pos=pl.col("t"), lat_pos="lat", lon_pos="lon", alt_pos="alt_ft"
    ).sort("t")
    # Both sides are sorted by t overall, hence within every group too.
    before = targets.join_asof(
        track, on="t", by=keys, strategy="backward", check_sortedness=False
    )
    after = targets.join_asof(
        track, on="t", by=keys, strategy="forward", check_sortedness=False
    )

    t = before["t"].to_numpy()

    def side(frame: pl.DataFrame, name: str) -> Floats:
        return floats(frame[name])

    t0, t1 = side(before, "t_pos"), side(after, "t_pos")
    gap0, gap1 = t - t0, t1 - t
    bracketed = (gap0 <= settings.max_bracket_s) & (gap1 <= settings.max_bracket_s)
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(t1 > t0, gap0 / (t1 - t0), 0.0)

    def blend(name: str) -> Floats:
        v0, v1 = side(before, name), side(after, name)
        interpolated = v0 + w * (v1 - v0)
        nearest = np.where(
            np.nan_to_num(gap0, nan=np.inf) <= np.nan_to_num(gap1, nan=np.inf), v0, v1
        )
        near_enough = np.fmin(gap0, gap1) <= settings.max_nearest_s
        result: Floats = np.where(
            bracketed, interpolated, np.where(near_enough, nearest, np.nan)
        )
        return result

    lat, lon, alt = blend("lat_pos"), blend("lon_pos"), blend("alt_pos")
    placed = ~np.isnan(lat)
    range_nm = np.full(len(t), np.nan)
    bearing = np.full(len(t), np.nan)
    elevation = np.full(len(t), np.nan)
    if placed.any():
        r, b, e = look_angles(
            site, lat[placed], lon[placed], np.nan_to_num(alt[placed]) * FEET
        )
        range_nm[placed], bearing[placed] = r, b
        elevation[placed] = np.where(np.isnan(alt[placed]), np.nan, e)

    attributed = before.select("tx").with_columns(
        lat=pl.Series(lat),
        lon=pl.Series(lon),
        alt_ft=pl.Series(alt),
        range_nm=pl.Series(range_nm),
        bearing=pl.Series(bearing),
        elevation=pl.Series(elevation),
    )
    return (
        transmissions.select("tx")
        .join(attributed, on="tx", how="left")
        .with_columns(pl.exclude("tx").fill_nan(None))
        .sort("tx")
    )
