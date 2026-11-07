"""Range, bearing and elevation of aircraft as seen from the receiver site."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

Floats = npt.NDArray[np.float64]

# WGS-84.
_A = 6_378_137.0
_E2 = 6.694_379_990_14e-3
FEET = 0.3048
NAUTICAL_MILE = 1852.0


@dataclass(frozen=True, slots=True)
class Site:
    """The antenna: degrees, and metres above the WGS-84 ellipsoid."""

    lat: float
    lon: float
    alt_m: float


def ecef(lat: Floats, lon: Floats, alt_m: Floats) -> tuple[Floats, Floats, Floats]:
    phi = np.radians(lat)
    lam = np.radians(lon)
    n = _A / np.sqrt(1 - _E2 * np.sin(phi) ** 2)
    x = (n + alt_m) * np.cos(phi) * np.cos(lam)
    y = (n + alt_m) * np.cos(phi) * np.sin(lam)
    z = (n * (1 - _E2) + alt_m) * np.sin(phi)
    return x, y, z


def look_angles(
    site: Site, lat: Floats, lon: Floats, alt_m: Floats
) -> tuple[Floats, Floats, Floats]:
    """Ground range (nm), bearing (degrees true) and elevation (degrees)."""
    sx, sy, sz = ecef(
        np.array([site.lat]), np.array([site.lon]), np.array([site.alt_m])
    )
    x, y, z = ecef(lat, lon, alt_m)
    dx, dy, dz = x - sx, y - sy, z - sz
    phi = np.radians(site.lat)
    lam = np.radians(site.lon)
    east = -np.sin(lam) * dx + np.cos(lam) * dy
    north = (
        -np.sin(phi) * np.cos(lam) * dx
        - np.sin(phi) * np.sin(lam) * dy
        + np.cos(phi) * dz
    )
    up = (
        np.cos(phi) * np.cos(lam) * dx
        + np.cos(phi) * np.sin(lam) * dy
        + np.sin(phi) * dz
    )
    bearing = np.degrees(np.arctan2(east, north)) % 360
    # A tiny negative angle wraps to exactly 360.
    bearing = np.where(bearing >= 360, bearing - 360, bearing)
    elevation = np.degrees(np.arctan2(up, np.hypot(east, north)))
    return ground_range_nm(site, lat, lon), bearing, elevation


def ground_range_nm(site: Site, lat: Floats, lon: Floats) -> Floats:
    """Great-circle distance from the site, on a sphere of mean radius."""
    phi1, phi2 = np.radians(site.lat), np.radians(lat)
    dphi = phi2 - phi1
    dlam = np.radians(lon - site.lon)
    h = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    metres = 2 * 6_371_008.8 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))
    distance: Floats = metres / NAUTICAL_MILE
    return distance
