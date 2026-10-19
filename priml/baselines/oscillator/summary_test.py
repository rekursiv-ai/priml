"""Check system-level uncertainty, paired comparisons, and failure accounting."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import pytest

from priml.baselines.oscillator.suite_test import tiny_suite


if TYPE_CHECKING:
    from pathlib import Path


def test_summary_averages_seeds_before_equal_weighting_families(tmp_path: Path) -> None:
    cfg = tiny_suite()
    suite = cfg.make()
    records, _ = suite.run(tmp_path / "run", phase="audit")
    for record in records:
        assert record["result"] is not None
        value = 2.0 if record["family"] == "cubic" else 8.0
        if record["method"] == "zero":
            value -= 1
        for scores in record["result"]["scores"].values():
            scores["method"]["scaled_mse"] = value
    summary = suite.summarize(records, phase="audit")
    base = summary[0]
    other = summary[1]
    assert base["scaled_mse"] == 5.0
    assert base["independent_systems"] == 3
    assert base["attempted_runs"] == 6
    assert base["scaled_mse_interval"] == [5.0, 5.0]
    assert other["scaled_mse"] == 4.0
    assert other["paired_difference_from_dlinear"] == -1.0
    assert other["paired_difference_interval"] == [-1.0, -1.0]
    assert base["worst_system_scaled_mse"] == 8.0
    assert suite.summarize(records, phase="audit") == summary


@pytest.mark.parametrize(
    "bad",
    [
        "missing",
        "duplicate",
        "foreign",
        "identity",
        "budget",
        "nonfinite",
        "empty",
        "failed_without_error",
    ],
)
def test_incomplete_or_malformed_results_are_rejected(tmp_path: Path, bad: str) -> None:
    cfg = tiny_suite()
    cfg.audit[0].instances = 1
    cfg.audit = cfg.audit[:1]
    cfg.recipes = cfg.recipes[:1]
    cfg.recipes[0].seeds = [11]
    cfg.budgets = cfg.budgets[:1]
    suite = cfg.make()
    records, _ = suite.run(tmp_path / "run", phase="audit")
    row = records[0]
    if bad == "missing":
        records.clear()
    elif bad == "duplicate":
        records.append(deepcopy(row))
    elif bad == "foreign":
        row["run_id"] = "foreign"
    elif bad == "identity":
        row["method"] = "foreign"
    elif bad == "budget":
        assert row["result"] is not None
        row["result"]["reference_fit_trajectories"] += 1
    elif bad == "nonfinite":
        assert row["result"] is not None
        row["result"]["scores"]["familiar"]["method"]["mse"] = float("inf")
    elif bad == "empty":
        row["result"] = None
    else:
        row["status"] = "failed"
        row["result"] = None
    with pytest.raises(
        ValueError,
        match=r"planned run|identity|budget|finite|complete run|failed run",
    ):
        suite.summarize(records, phase="audit")


def test_a_failed_model_gets_no_finite_average(tmp_path: Path) -> None:
    cfg = tiny_suite()
    suite = cfg.make()
    records, _ = suite.run(tmp_path / "run", phase="audit")
    first = records[0]
    first["status"] = "failed"
    first["result"] = None
    first["error"] = "Nonfinite forecast"
    summary = suite.summarize(records, phase="audit")
    assert summary[0]["failed_runs"] == 1
    assert summary[0]["scaled_mse"] is None
    assert summary[1]["paired_difference_from_dlinear"] is None


@pytest.mark.parametrize("bad", ["controls", "split", "budget", "geometry", "elapsed"])
def test_summary_rejects_unpaired_controls_and_inconsistent_accounting(
    tmp_path: Path,
    bad: str,
) -> None:
    cfg = tiny_suite()
    suite = cfg.make()
    records, _ = suite.run(tmp_path / "run", phase="audit")
    row = records[1]
    assert row["result"] is not None
    if bad == "controls":
        row["result"]["scores"]["familiar"]["cheap_simulator"]["mse"] += 1
    elif bad == "split":
        row["result"]["scores"].pop("shifted")
    elif bad == "budget":
        row["result"]["reference_budget"] += 1
    elif bad == "geometry":
        row["result"]["timesteps_per_query"] += 1
    else:
        row["elapsed_seconds"] = float("nan")
    with pytest.raises(ValueError, match=r"controls|splits|budget|geometry|Elapsed"):
        suite.summarize(records, phase="audit")
