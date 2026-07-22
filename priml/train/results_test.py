"""Tests for the shared run-metrics reader."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import json

from priml.train.results import format_summary, read_metrics, summarize


if TYPE_CHECKING:
    import pytest


def _write(tmp_path: Path, metrics: dict[str, float]) -> Path:
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps(metrics))
    return path


def test_read_metrics_round_trip(tmp_path: Path):
    path = _write(tmp_path, {"eval/total_loss": 3.045})
    assert read_metrics(path) == {"eval/total_loss": 3.045}


def test_format_summary_orders_by_keys_and_strips_prefix():
    metrics = {"eval/total_loss": 3.0, "eval/time": 4.2, "eval/extra": 1.0}
    summary = format_summary(metrics, ["eval/total_loss", "eval/time"])
    assert summary == "total_loss=3.0 time=4.2"


def test_format_summary_skips_missing_keys():
    summary = format_summary({"eval/a": 1.0}, ["eval/a", "eval/absent"])
    assert summary == "a=1.0"


def test_format_summary_none_keys_shows_all_sorted():
    summary = format_summary({"eval/b": 2.0, "eval/a": 1.0})
    assert summary == "a=1.0 b=2.0"


def test_format_summary_empty():
    assert format_summary({}) == "no eval metrics found"


def test_summarize_prints_summary(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    path = _write(tmp_path, {"eval/total_loss": 3.045, "eval/time": 4.2})
    code = summarize(["--path", str(path)], keys=["eval/total_loss"])
    assert code == 0
    assert capsys.readouterr().out.strip() == "total_loss=3.045"


def test_summarize_missing_file_returns_1(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    code = summarize(["--path", str(tmp_path / "absent.json")])
    assert code == 1
    assert "No metrics file" in capsys.readouterr().err


def test_summarize_appends_note(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    path = _write(tmp_path, {"eval/steps": 15000.0, "eval/budget": 15000.0})
    code = summarize(
        ["--path", str(path)],
        keys=["eval/steps"],
        note=lambda m: " (sentinel)" if m["eval/steps"] == m["eval/budget"] else "",
    )
    assert code == 0
    assert capsys.readouterr().out.strip() == "steps=15000.0 (sentinel)"


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
