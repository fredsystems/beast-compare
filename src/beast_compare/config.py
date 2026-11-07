"""Recorder configuration.

Example::

    run_id: 2026-11-baseline
    data_dir: /opt/adsb/data/beast-compare
    receivers:
      - name: airspy
        beast: 127.0.0.1:30105
        stats:
          file: /opt/adsb/data/airspy_adsb/stats.json
        metadata:
          sdr: Airspy R2
      - name: fa-proplus
        beast: 127.0.0.1:30106
        stats:
          url: http://127.0.0.1:8089/data/stats.json
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]*$")
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class ConfigError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class StatsSource:
    """Exactly one of ``url`` and ``file`` is set."""

    url: str | None = None
    file: Path | None = None

    def to_dict(self) -> dict[str, str]:
        if self.url is not None:
            return {"url": self.url}
        return {"file": str(self.file)}


@dataclass(frozen=True, slots=True)
class ReceiverConfig:
    name: str
    host: str
    port: int
    stats: StatsSource | None = None
    metadata: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "name": self.name,
            "beast": f"{self.host}:{self.port}",
            "metadata": dict(sorted(self.metadata.items())),
        }
        if self.stats is not None:
            result["stats"] = self.stats.to_dict()
        return result


@dataclass(frozen=True, slots=True)
class RecorderConfig:
    run_id: str
    data_dir: Path
    receivers: tuple[ReceiverConfig, ...]
    stats_interval_s: float = 60.0
    flush_interval_s: float = 10.0
    idle_timeout_s: float = 300.0

    @property
    def run_dir(self) -> Path:
        return self.data_dir / self.run_id

    def to_dict(self) -> dict[str, Any]:
        """The configuration as recorded in the run manifest."""
        return {
            "run_id": self.run_id,
            "data_dir": str(self.data_dir),
            "stats_interval_s": self.stats_interval_s,
            "flush_interval_s": self.flush_interval_s,
            "idle_timeout_s": self.idle_timeout_s,
            "receivers": [r.to_dict() for r in self.receivers],
        }


def load_config(path: Path) -> RecorderConfig:
    with path.open() as handle:
        try:
            raw = yaml.safe_load(handle)
        except yaml.YAMLError as exc:
            raise ConfigError(f"{path}: {exc}") from exc
    return parse_config(raw)


def parse_config(raw: object) -> RecorderConfig:
    data = _mapping(raw, "config")
    _reject_unknown(
        data,
        {
            "run_id",
            "data_dir",
            "receivers",
            "stats_interval_s",
            "flush_interval_s",
            "idle_timeout_s",
        },
        "config",
    )

    run_id = _string(data.get("run_id"), "run_id")
    if not RUN_ID_PATTERN.match(run_id):
        raise ConfigError(f"run_id {run_id!r} must match {RUN_ID_PATTERN.pattern}")

    receivers_raw = data.get("receivers")
    if not isinstance(receivers_raw, list) or not receivers_raw:
        raise ConfigError("receivers must be a non-empty list")
    receivers = tuple(
        _receiver(item, f"receivers[{i}]") for i, item in enumerate(receivers_raw)
    )
    names = [r.name for r in receivers]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ConfigError(f"duplicate receiver names: {', '.join(duplicates)}")

    return RecorderConfig(
        run_id=run_id,
        data_dir=Path(_string(data.get("data_dir"), "data_dir")),
        receivers=receivers,
        stats_interval_s=_positive(data, "stats_interval_s", 60.0),
        flush_interval_s=_positive(data, "flush_interval_s", 10.0),
        idle_timeout_s=_positive(data, "idle_timeout_s", 300.0),
    )


def _receiver(raw: object, where: str) -> ReceiverConfig:
    data = _mapping(raw, where)
    _reject_unknown(data, {"name", "beast", "stats", "metadata"}, where)

    name = _string(data.get("name"), f"{where}.name")
    if not NAME_PATTERN.match(name):
        raise ConfigError(f"{where}.name {name!r} must match {NAME_PATTERN.pattern}")

    host, port = _endpoint(_string(data.get("beast"), f"{where}.beast"), where)

    stats = None
    if data.get("stats") is not None:
        stats = _stats(data["stats"], f"{where}.stats")

    metadata_raw = data.get("metadata") or {}
    metadata = _mapping(metadata_raw, f"{where}.metadata")
    for key, value in metadata.items():
        if not isinstance(value, (str, int, float)) or isinstance(value, bool):
            raise ConfigError(f"{where}.metadata.{key} must be a string or number")

    return ReceiverConfig(
        name=name,
        host=host,
        port=port,
        stats=stats,
        metadata={str(k): str(v) for k, v in metadata.items()},
    )


def _endpoint(value: str, where: str) -> tuple[str, int]:
    host, sep, port_text = value.rpartition(":")
    if not sep or not host or not port_text.isdigit():
        raise ConfigError(f"{where}.beast {value!r} must be host:port")
    port = int(port_text)
    if not 1 <= port <= 65535:
        raise ConfigError(f"{where}.beast port {port} is out of range")
    return host.strip("[]"), port


def _stats(raw: object, where: str) -> StatsSource:
    data = _mapping(raw, where)
    _reject_unknown(data, {"url", "file"}, where)
    if ("url" in data) == ("file" in data):
        raise ConfigError(f"{where} needs exactly one of url and file")
    if "url" in data:
        url = _string(data["url"], f"{where}.url")
        if not url.startswith(("http://", "https://")):
            raise ConfigError(f"{where}.url must be http or https")
        return StatsSource(url=url)
    return StatsSource(file=Path(_string(data["file"], f"{where}.file")))


def _mapping(raw: object, where: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ConfigError(f"{where} must be a mapping")
    return {str(k): v for k, v in raw.items()}


def _string(raw: object, where: str) -> str:
    if not isinstance(raw, str) or not raw:
        raise ConfigError(f"{where} must be a non-empty string")
    return raw


def _positive(data: dict[str, Any], key: str, default: float) -> float:
    raw = data.get(key, default)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)) or raw <= 0:
        raise ConfigError(f"{key} must be a positive number")
    return float(raw)


def _reject_unknown(data: dict[str, Any], allowed: set[str], where: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ConfigError(f"{where} has unknown keys: {', '.join(unknown)}")
