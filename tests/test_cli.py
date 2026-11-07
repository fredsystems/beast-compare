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


def test_compare_not_implemented(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["compare", "some-run"]) == 2
    assert "not implemented" in capsys.readouterr().err
