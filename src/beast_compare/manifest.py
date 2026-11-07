"""The run manifest: what was recorded, and when.

A run is defined by its configuration. Restarting the recorder with the
same configuration resumes the run and adds a session; restarting with a
different one is refused, because the run's data would then no longer
describe one setup. Change the ``run_id`` to start a new run instead.
"""

from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
from typing import Any

from beast_compare import __version__
from beast_compare.config import RecorderConfig

MANIFEST_NAME = "manifest.json"
FORMAT_VERSION = 1


class ManifestError(RuntimeError):
    pass


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def _write(path: Path, manifest: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(temporary, path)


def read_manifest(run_dir: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((run_dir / MANIFEST_NAME).read_text())
    return data


def begin_session(config: RecorderConfig) -> int:
    """Create or resume the run; returns the new session's index."""
    run_dir = config.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / MANIFEST_NAME
    recorded = config.to_dict()

    if path.exists():
        manifest = read_manifest(run_dir)
        if manifest.get("format") != FORMAT_VERSION:
            raise ManifestError(
                f"{path} has format {manifest.get('format')}, expected {FORMAT_VERSION}"
            )
        if manifest.get("config") != recorded:
            raise ManifestError(
                f"{path} was recorded with a different configuration; "
                "use a new run_id for a new setup"
            )
    else:
        manifest = {
            "format": FORMAT_VERSION,
            "run_id": config.run_id,
            "config": recorded,
            "sessions": [],
        }

    sessions: list[dict[str, Any]] = manifest["sessions"]
    sessions.append({"start": _now(), "stop": None, "tool_version": __version__})
    _write(path, manifest)
    return len(sessions) - 1


def end_session(config: RecorderConfig, session: int) -> None:
    run_dir = config.run_dir
    manifest = read_manifest(run_dir)
    manifest["sessions"][session]["stop"] = _now()
    _write(run_dir / MANIFEST_NAME, manifest)
