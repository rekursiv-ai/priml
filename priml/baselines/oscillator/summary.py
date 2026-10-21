"""Summarize complete benchmark runs without counting model seeds as systems."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, TypedDict

import math

from configgle import Fig


if TYPE_CHECKING:
    import torch

    from priml.baselines.oscillator.benchmark import BenchmarkRun, ForecastScores
else:
    from wrapt import lazy_import

    torch = lazy_import("torch")


class RunSpec(TypedDict):
    """Expected identity and budget of one planned fit."""

    method: str
    system_id: str
    family: str
    model_seed: int
    access: str
    budget: str
    experiment: str
    train: int
    validation: int
    cheap: int
    timesteps: int


class BenchmarkSummary:
    """Bootstrap independent systems within each declared mechanism family."""

    class Config(Fig["BenchmarkSummary"]):
        resamples: int = 2000
        """Stratified paired bootstrap draws."""

        seed: int = 90_210
        """Independent reporting RNG, never used to generate or fit data."""

        confidence: float = 0.95
        """Central bootstrap interval mass."""

    def __init__(self, config: Config) -> None:
        if (
            config.resamples <= 0
            or math.isnan(config.confidence)
            or config.confidence <= 0
            or config.confidence >= 1
        ):
            raise ValueError("Bootstrap count and confidence must be valid.")
        self.config = config

    def summarize(
        self,
        records: list[BenchmarkRun],
        *,
        expected: dict[str, RunSpec],
        baseline: str,
    ) -> list[dict[str, object]]:
        """Require complete unique runs, then reduce seeds before bootstrapping.

        Args:
          records: Every attempted run, including failures.
          expected: Frozen identities, seeds, and budgets for the selected phase.
          baseline: Method used for paired system comparisons.

        Returns:
          rows: Family-balanced errors, paired differences, and system counts.

        Raises:
          ValueError: Runs are missing, duplicated, or not in the protocol.

        """
        actual = [row["run_id"] for row in records]
        if len(actual) != len(set(actual)) or set(actual) != set(expected):
            raise ValueError(
                "Benchmark results must contain every planned run exactly once.",
            )
        controls: dict[tuple[str, str, str], ForecastScores] = {}
        for row in records:
            spec = expected[row["run_id"]]
            if (
                row["method"],
                row["system_id"],
                row["family"],
                row["model_seed"],
                row["access"],
                row["budget"],
            ) != (
                spec["method"],
                spec["system_id"],
                spec["family"],
                spec["model_seed"],
                spec["access"],
                spec["budget"],
            ):
                raise ValueError("A result's identity does not match its planned run.")
            result = row["result"]
            if not math.isfinite(row["elapsed_seconds"]) or row["elapsed_seconds"] < 0:
                raise ValueError("Elapsed time must be finite and nonnegative.")
            if row["status"] == "complete":
                if result is None or row["error"] is not None:
                    raise ValueError(
                        "A complete run must include valid results and no error.",
                    )
                if (
                    result["seed"],
                    result["experiment"],
                    result["reference_fit_trajectories"],
                    result["reference_validation_trajectories"],
                ) != (
                    spec["model_seed"],
                    spec["experiment"],
                    spec["train"],
                    spec["validation"],
                ):
                    raise ValueError(
                        "A fit's seed or reference budget differs from the protocol.",
                    )
                if any(
                    not math.isfinite(value) or value < 0
                    for split in result["scores"].values()
                    for scores in split.values()
                    for value in (scores["mse"], scores["mae"], scores["scaled_mse"])
                ):
                    raise ValueError(
                        "Completed runs must contain finite nonnegative errors.",
                    )
                if (
                    result["reference_budget"],
                    result["cheap_training_trajectories"],
                    result["timesteps_per_query"],
                ) != (
                    spec["train"] + spec["validation"],
                    spec["cheap"],
                    spec["timesteps"],
                ):
                    raise ValueError(
                        "A reported data budget or geometry differs from the protocol.",
                    )
                if set(result["scores"]) != {"familiar", "shifted"}:
                    raise ValueError("A complete run must report both test splits.")
                for split, scores in result["scores"].items():
                    if set(scores) != {"method", "cheap_simulator", "last_value"}:
                        raise ValueError(
                            "A complete run must report the method and both controls.",
                        )
                    for control in ("cheap_simulator", "last_value"):
                        key = row["system_id"], split, control
                        if key in controls and controls[key] != scores[control]:
                            raise ValueError(
                                "Paired methods and budgets must use identical test controls.",
                            )
                        controls[key] = scores[control]
            elif (
                row["status"] != "failed" or result is not None or row["error"] is None
            ):
                raise ValueError(
                    "A failed run must include an error and no finite result.",
                )
        rows: list[dict[str, object]] = []
        budgets = {
            spec["budget"]: spec["train"] + spec["validation"]
            for spec in expected.values()
        }
        methods = {spec["method"]: spec["access"] for spec in expected.values()}
        for budget, reference_budget in budgets.items():
            selected = [row for row in records if row["budget"] == budget]
            for split in ("familiar", "shifted"):
                base_records = [row for row in selected if row["method"] == baseline]
                base_errors = self._systems(base_records, split=split)
                for method, access in methods.items():
                    group = [row for row in selected if row["method"] == method]
                    systems = self._systems(group, split=split)
                    failures = sum(row["status"] != "complete" for row in group)
                    aggregate: dict[str, object] = {
                        "budget": budget,
                        "reference_budget": reference_budget,
                        "split": split,
                        "method": method,
                        "baseline": baseline,
                        "access": access,
                        "attempted_runs": len(group),
                        "failed_runs": failures,
                        "independent_systems": len(
                            {item["system_id"] for item in group},
                        ),
                        "scaled_mse": None,
                        "scaled_mse_interval": None,
                        "paired_difference": None,
                        "paired_difference_interval": None,
                        "family_scaled_mse": None,
                        "worst_system_scaled_mse": None,
                        "elapsed_seconds": sum(
                            item["elapsed_seconds"] for item in group
                        ),
                    }
                    if failures == 0:
                        values, interval, families = self._bootstrap(systems)
                        aggregate.update(
                            scaled_mse=values,
                            scaled_mse_interval=interval,
                            family_scaled_mse=families,
                            worst_system_scaled_mse=max(systems.values()),
                        )
                        if base_errors.keys() == systems.keys() and all(
                            math.isfinite(value) for value in base_errors.values()
                        ):
                            differences = {
                                key: value - base_errors[key]
                                for key, value in systems.items()
                            }
                            paired, paired_interval, _ = self._bootstrap(differences)
                            aggregate.update(
                                paired_difference=paired,
                                paired_difference_interval=paired_interval,
                            )
                    rows.append(aggregate)
        return rows

    def _systems(
        self,
        records: list[BenchmarkRun],
        *,
        split: str,
    ) -> dict[tuple[str, str], float]:
        grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
        for row in records:
            result = row["result"]
            value = (
                math.inf
                if row["status"] != "complete" or result is None
                else result["scores"][split]["method"]["scaled_mse"]
            )
            grouped[row["family"], row["system_id"]].append(value)
        return {key: sum(values) / len(values) for key, values in grouped.items()}

    def _bootstrap(
        self,
        systems: dict[tuple[str, str], float],
    ) -> tuple[float, list[float], dict[str, float]]:
        grouped: dict[str, list[float]] = defaultdict(list)
        for (family, _), value in sorted(systems.items()):
            grouped[family].append(value)
        generator = torch.Generator().manual_seed(self.config.seed)
        samples = torch.zeros(self.config.resamples, dtype=torch.float64)
        means: dict[str, float] = {}
        for family, values in sorted(grouped.items()):
            data = torch.tensor(values, dtype=torch.float64)
            means[family] = float(data.mean())
            indices = torch.randint(
                len(data),
                (self.config.resamples, len(data)),
                generator=generator,
            )
            samples += data[indices].mean(dim=1) / len(grouped)
        tail = (1 - self.config.confidence) / 2
        bounds = torch.quantile(
            samples,
            torch.tensor([tail, 1 - tail], dtype=torch.float64),
        )
        interval = [float(bounds[0]), float(bounds[1])]
        return sum(means.values()) / len(means), interval, means
