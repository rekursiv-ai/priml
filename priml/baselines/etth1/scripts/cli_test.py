"""Check help and argument errors through the public command entry points."""

from pathlib import Path
from typing import Final

import runpy
import subprocess
import sys

import pytest


_CWD: Final = Path(__file__).resolve().parent


@pytest.mark.parametrize("script", ["prepare_data", "evaluate", "verify_reference"])
@pytest.mark.parametrize(("argument", "status"), [("--help", 0), ("--unknown", 2)])
def test_module_entry_point(
    script: str,
    argument: str,
    status: int,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", [script, argument])
    with pytest.raises(SystemExit) as result:
        runpy.run_module(
            f"priml.baselines.etth1.scripts.{script}",
            run_name="__main__",
        )
    assert result.value.code == status
    captured = capsys.readouterr()
    assert "usage:" in captured.out + captured.err
    if script == "verify_reference" and status == 0:
        assert "pytest" in captured.out


@pytest.mark.cli_uv
@pytest.mark.parametrize("script", ["prepare_data", "evaluate", "verify_reference"])
def test_executable_entry_point_from_another_directory(
    script: str,
    tmp_path: Path,
) -> None:
    executable = _CWD / f"{script}.py"
    result = subprocess.run(  # noqa: S603 -- Execute the known CLI scripts.
        [str(executable), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
