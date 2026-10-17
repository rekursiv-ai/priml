"""Tests for the synthetic-clone builder."""

from __future__ import annotations

from pathlib import Path

import pytest

from priml.baselines.mlsbench.scripts import prepare_data
from priml.baselines.mlsbench.testing import SYNTH_TASK


def test_builds_the_synthetic_clone_idempotently(tmp_path: Path) -> None:
    dest = tmp_path / "smoke"
    assert prepare_data.main(["--directory", str(dest)]) == 0
    (dest / "tasks" / SYNTH_TASK / "stale.txt").write_text("x")
    assert prepare_data.main(["--directory", str(dest)]) == 0
    assert (dest / "tasks" / SYNTH_TASK / "config.json").is_file()
    assert not (dest / "tasks" / SYNTH_TASK / "stale.txt").exists()


def test_default_directory_is_where_exp_smoke_reads() -> None:
    assert prepare_data.default_directory() == Path("/opt/scratch/mlsbench_smoke")


def test_help_shows_the_docstring(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit, match="0"):
        prepare_data.main(["--help"])
    assert "Build the synthetic MLS-Bench clone" in capsys.readouterr().out


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
