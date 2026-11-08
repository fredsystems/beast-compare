from pathlib import Path

import pytest

from beast_compare import __version__
from beast_compare.cli import main


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_command_required() -> None:
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2


def test_compare_needs_the_site(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["compare", "some-run"])
    assert exc.value.code == 2
    assert "--lat" in capsys.readouterr().err


def test_compare_missing_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    site = ["--lat", "35", "--lon", "-106", "--alt-m", "1600"]
    assert main(["compare", str(tmp_path / "nope"), *site]) == 1
    assert "nope" in capsys.readouterr().err
