from pathlib import Path

import pytest

from beast_compare.config import RecorderConfig, parse_config
from beast_compare.manifest import (
    ManifestError,
    begin_session,
    end_session,
    read_manifest,
)


def config(data_dir: Path, port: int = 30105) -> RecorderConfig:
    return parse_config(
        {
            "run_id": "r1",
            "data_dir": str(data_dir),
            "receivers": [{"name": "airspy", "beast": f"127.0.0.1:{port}"}],
        }
    )


def test_new_run(tmp_path: Path) -> None:
    cfg = config(tmp_path)
    assert begin_session(cfg) == 0
    manifest = read_manifest(cfg.run_dir)
    assert manifest["run_id"] == "r1"
    assert manifest["config"] == cfg.to_dict()
    assert manifest["sessions"][0]["stop"] is None
    end_session(cfg, 0)
    assert read_manifest(cfg.run_dir)["sessions"][0]["stop"] is not None


def test_resume_with_same_config(tmp_path: Path) -> None:
    cfg = config(tmp_path)
    end_session(cfg, begin_session(cfg))
    assert begin_session(cfg) == 1
    assert len(read_manifest(cfg.run_dir)["sessions"]) == 2


def test_refuses_a_changed_config(tmp_path: Path) -> None:
    begin_session(config(tmp_path))
    with pytest.raises(ManifestError, match="different configuration"):
        begin_session(config(tmp_path, port=30106))
