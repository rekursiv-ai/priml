"""Tests that the prepare_data executable exits with the data entry point's code."""

import runpy

import pytest

from priml.baselines.etth1.scripts import data


def test_executable_exits_with_data_main(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(data, "main", lambda: 7)
    # Run, never import: an imported copy makes cli_test's runpy warn and fail.
    with pytest.raises(SystemExit) as result:
        runpy.run_module(
            "priml.baselines.etth1.scripts.prepare_data",
            run_name="__main__",
        )
    assert result.value.code == 7


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
