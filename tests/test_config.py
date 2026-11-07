from pathlib import Path
from typing import Any

import pytest

from beast_compare.config import ConfigError, StatsSource, load_config, parse_config


def valid() -> dict[str, Any]:
    return {
        "run_id": "2026-11-baseline",
        "data_dir": "/data",
        "receivers": [
            {
                "name": "airspy",
                "beast": "127.0.0.1:30105",
                "stats": {"file": "/opt/airspy/stats.json"},
                "metadata": {"sdr": "Airspy R2", "attenuation_db": 0},
            },
            {
                "name": "fa-proplus",
                "beast": "sdrhub.local:30106",
                "stats": {"url": "http://127.0.0.1:8089/data/stats.json"},
            },
            {"name": "rtl-v4", "beast": "[::1]:30005"},
        ],
    }


def test_valid() -> None:
    config = parse_config(valid())
    assert config.run_dir == Path("/data/2026-11-baseline")
    airspy, fa, v4 = config.receivers
    assert (airspy.host, airspy.port) == ("127.0.0.1", 30105)
    assert airspy.stats == StatsSource(file=Path("/opt/airspy/stats.json"))
    assert airspy.metadata == {"sdr": "Airspy R2", "attenuation_db": "0"}
    assert fa.stats == StatsSource(url="http://127.0.0.1:8089/data/stats.json")
    assert (v4.host, v4.port, v4.stats) == ("::1", 30005, None)
    assert config.stats_interval_s == 60.0


def test_to_dict_is_stable() -> None:
    assert parse_config(valid()).to_dict() == parse_config(valid()).to_dict()


def mutate(path: list[Any], value: Any) -> dict[str, Any]:
    data = valid()
    target: Any = data
    for key in path[:-1]:
        target = target[key]
    if value is DELETE:
        del target[path[-1]]
    else:
        target[path[-1]] = value
    return data


DELETE = object()


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (["run_id"], DELETE, "run_id"),
        (["run_id"], "has space", "run_id"),
        (["data_dir"], "", "data_dir"),
        (["receivers"], [], "receivers"),
        (["receivers", 0, "name"], "Airspy", "name"),
        (["receivers", 1, "name"], "airspy", "duplicate"),
        (["receivers", 0, "beast"], "localhost", "host:port"),
        (["receivers", 0, "beast"], "localhost:70000", "out of range"),
        (["receivers", 0, "stats"], {}, "exactly one"),
        (["receivers", 0, "stats"], {"url": "http://x", "file": "/x"}, "exactly one"),
        (["receivers", 0, "stats"], {"url": "ftp://x"}, "http"),
        (["receivers", 0, "metadata"], {"x": True}, "metadata.x"),
        (["receivers", 0, "colour"], "blue", "unknown keys"),
        (["stats_interval_s"], 0, "stats_interval_s"),
        (["surprise"], 1, "unknown keys"),
    ],
)
def test_invalid(path: list[Any], value: Any, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        parse_config(mutate(path, value))


def test_not_a_mapping() -> None:
    with pytest.raises(ConfigError):
        parse_config(["not", "a", "mapping"])


def test_load_config(tmp_path: Path) -> None:
    path = tmp_path / "recorder.yaml"
    path.write_text(
        "run_id: r1\ndata_dir: /data\nreceivers:\n"
        "  - name: airspy\n    beast: 127.0.0.1:30105\n"
    )
    assert load_config(path).receivers[0].name == "airspy"
    path.write_text("run_id: [unclosed\n")
    with pytest.raises(ConfigError):
        load_config(path)
