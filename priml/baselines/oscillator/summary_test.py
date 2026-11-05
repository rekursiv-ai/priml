"""Check system-level uncertainty, paired comparisons, and failure accounting."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import pytest

from priml.baselines.oscillator.summary import BenchmarkSummary


if TYPE_CHECKING:
    from priml.baselines.oscillator.benchmark import BenchmarkRun
    from priml.baselines.oscillator.summary import RunSpec


def summary_inputs() -> tuple[list[BenchmarkRun], dict[str, RunSpec]]:
    """Keep unequal family sizes, paired methods, seeds and nested budgets."""
    records: list[BenchmarkRun] = []
    expected: dict[str, RunSpec] = {}
    for budget, train, validation in (("small", 4, 5), ("large", 8, 10)):
        for system in ("cubic-000", "cubic-001", "sine-000"):
            for method, seeds in (("dlinear", (11, 12)), ("zero", (11,))):
                for seed in seeds:
                    run_id = f"{system}--{budget}--{method}--{seed}"
                    family = system.split("-")[0]
                    expected[run_id] = {
                        "method": method,
                        "system_id": system,
                        "family": family,
                        "model_seed": seed,
                        "access": "adapted",
                        "budget": budget,
                        "experiment": "test",
                        "train": train,
                        "validation": validation,
                        "cheap": 11,
                        "timesteps": 9,
                    }
                    records.append(
                        {
                            "run_id": run_id,
                            "family": family,
                            "system_id": system,
                            "budget": budget,
                            "method": method,
                            "access": "adapted",
                            "model_seed": seed,
                            "elapsed_seconds": 0.25,
                            "status": "complete",
                            "error": None,
                            "result": {
                                "experiment": "test",
                                "seed": seed,
                                "reference_fit_trajectories": train,
                                "reference_validation_trajectories": validation,
                                "reference_budget": train + validation,
                                "cheap_training_trajectories": 11,
                                "timesteps_per_query": 9,
                                "training": {
                                    "updates": 0,
                                    "best_update": 0,
                                    "pretrain_updates": 0,
                                },
                                "scores": {
                                    split: {
                                        name: {
                                            "mse": 2.0,
                                            "mae": 1.0,
                                            "scaled_mse": 0.5,
                                        }
                                        for name in (
                                            "method",
                                            "cheap_simulator",
                                            "last_value",
                                        )
                                    }
                                    for split in ("familiar", "shifted")
                                },
                            },
                        },
                    )
    return records, expected


@pytest.mark.parametrize("invalid", ["resamples", "confidence"])
def test_summary_rejects_invalid_bootstrap_settings(invalid: str) -> None:
    cfg = BenchmarkSummary.Config()
    if invalid == "resamples":
        cfg.resamples = 0
    else:
        cfg.confidence = 1
    with pytest.raises(ValueError, match="Bootstrap"):
        cfg.make()


@pytest.mark.parametrize("baseline", ["dlinear", "zero"])
def test_summary_averages_seeds_before_equal_weighting_families(
    baseline: str,
) -> None:
    summary_model = BenchmarkSummary.Config(resamples=20).make()
    records, expected = summary_inputs()
    for record in records:
        assert record["result"] is not None
        value = 2.0 if record["family"] == "cubic" else 8.0
        if record["method"] == "zero":
            value -= 1
        for scores in record["result"]["scores"].values():
            scores["method"]["scaled_mse"] = value
    summary = summary_model.summarize(records, expected=expected, baseline=baseline)
    base = summary[0]
    other = summary[1]
    assert base["scaled_mse"] == 5.0
    assert base["independent_systems"] == 3
    assert base["attempted_runs"] == 6
    assert base["scaled_mse_interval"] == [5.0, 5.0]
    assert other["scaled_mse"] == 4.0
    assert all(row["baseline"] == baseline for row in summary)
    difference = -1.0 if baseline == "dlinear" else 0.0
    assert other["paired_difference"] == difference
    assert other["paired_difference_interval"] == [difference, difference]
    assert base["paired_difference"] == (0.0 if baseline == "dlinear" else 1.0)
    assert base["worst_system_scaled_mse"] == 8.0
    assert (
        summary_model.summarize(records, expected=expected, baseline=baseline)
        == summary
    )


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
def test_incomplete_or_malformed_results_are_rejected(bad: str) -> None:
    records, expected = summary_inputs()
    summary_model = BenchmarkSummary.Config(resamples=20).make()
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
        summary_model.summarize(records, expected=expected, baseline="dlinear")


def test_a_failed_model_gets_no_finite_average() -> None:
    records, expected = summary_inputs()
    summary_model = BenchmarkSummary.Config(resamples=20).make()
    first = records[0]
    first["status"] = "failed"
    first["result"] = None
    first["error"] = "Nonfinite forecast"
    summary = summary_model.summarize(records, expected=expected, baseline="dlinear")
    assert summary[0]["failed_runs"] == 1
    assert summary[0]["scaled_mse"] is None
    assert summary[1]["paired_difference"] is None


@pytest.mark.parametrize(
    "bad",
    ["controls", "missing_control", "split", "budget", "geometry", "elapsed"],
)
def test_summary_rejects_unpaired_controls_and_inconsistent_accounting(
    bad: str,
) -> None:
    records, expected = summary_inputs()
    summary_model = BenchmarkSummary.Config(resamples=20).make()
    row = records[1]
    assert row["result"] is not None
    if bad == "controls":
        row["result"]["scores"]["familiar"]["cheap_simulator"]["mse"] += 1
    elif bad == "missing_control":
        row["result"]["scores"]["familiar"].pop("last_value")
    elif bad == "split":
        row["result"]["scores"].pop("shifted")
    elif bad == "budget":
        row["result"]["reference_budget"] += 1
    elif bad == "geometry":
        row["result"]["timesteps_per_query"] += 1
    else:
        row["elapsed_seconds"] = float("nan")
    with pytest.raises(ValueError, match=r"controls|splits|budget|geometry|Elapsed"):
        summary_model.summarize(records, expected=expected, baseline="dlinear")


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
