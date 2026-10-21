"""Run a fixed adaptation benchmark on independent held-out force laws."""

from __future__ import annotations

from dataclasses import field
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Self, TypedDict, cast, override

import hashlib
import json
import logging
import math
import re
import time

from configgle import Fig, Maker, Makes, traverse
from torch import Tensor
from treekle import from_plain, json_loads, to_plain

import torch

from priml.baselines.oscillator.benchmark import (
    BenchmarkRun,
    ForecastStudy,
    StudyRecord,
    forecast_scores,
)
from priml.baselines.oscillator.data import ForecastData, Windows
from priml.baselines.oscillator.mechanisms import SystemFamily
from priml.baselines.oscillator.summary import BenchmarkSummary, RunSpec
from priml.paths import validated_output_path


if TYPE_CHECKING:
    from collections.abc import Iterator


class BudgetStudy(ForecastStudy):
    """Keep evaluation scales fixed when the adaptation budget changes."""

    class Config(Makes["BudgetStudy"], ForecastStudy.Config):
        scoring_trajectories: int = 64
        """Common training prefix used only to set the reporting scale."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        if (
            config.scoring_trajectories <= 0
            or config.scoring_trajectories > config.data.train.count
        ):
            raise ValueError(
                "The scoring prefix must fit inside the reference training budget.",
            )
        self.scoring_trajectories = config.scoring_trajectories

    @override
    def score_scale(self, train: Windows) -> Tensor:
        """Use the same training prefix at every reference budget."""
        prefix = Windows(
            media=train.media[: self.scoring_trajectories],
            label=train.label[: self.scoring_trajectories],
        )
        return prefix.channel_stats()[1]


class ReferenceBudget:
    """Count all accurate trajectories available for fitting and selection."""

    class Config(Fig["ReferenceBudget"]):
        name: str = "small"
        """Budget label used in result identities."""

        train: int = 64
        """Reference training trajectories."""

        validation: int = 32
        """Reference validation trajectories; these also cost queries."""

    def __init__(self, config: Config) -> None:
        if not config.name or min(config.train, config.validation) <= 0:
            raise ValueError(
                "Reference budgets need a label and positive split counts.",
            )
        self.config = config


class BenchmarkRecipe:
    """Keep a named forecasting recipe and its model seeds together."""

    class Config(Fig["BenchmarkRecipe"]):
        name: str = "dlinear"
        """Method label; it does not choose an implementation."""

        study: BudgetStudy.Config = field(default_factory=BudgetStudy.Config)
        """Complete forecasting recipe; the suite supplies the task's data."""

        seeds: list[int] = field(default_factory=lambda: [2021, 2022, 2023])
        """Repeated model seeds; deterministic methods need only one."""

        access: Literal["adapted", "zero-shot"] = "adapted"
        """Reporting label describing use of reference targets."""

        @override
        def finalize(self) -> Self:
            """Finalize this owner while its task-specific study remains pending."""
            self._finalized = True
            return self

    def __init__(self, config: Config) -> None:
        for match in traverse(config.study, Maker, recurse=True):
            if getattr(match.config, "_finalized", False):
                raise ValueError("Benchmark recipes require unfinalized templates.")
        if (
            not config.name
            or not config.seeds
            or len(set(config.seeds)) != len(config.seeds)
        ):
            raise ValueError("A recipe needs a name and distinct model seeds.")
        if any(seed < 0 or seed >= 2**32 for seed in config.seeds):
            raise ValueError("Model seeds must lie in [0, 2**32).")
        self.config = config


class TaskDiagnostics(TypedDict):
    """Checks on a task's data generation, not a competing learner."""

    system_id: str
    budget: str
    oracle_scaled_mse: float
    finer_grid_scaled_mse: float
    unique_initial_states: int


class BenchmarkSuite:
    """Keep development and audit mechanisms separate under one printable config."""

    class Config(Fig["BenchmarkSuite"]):
        version: str = "0.1.0"
        """Protocol identity; a revised audit claim needs a new version."""

        working_dir: Path = Path("/opt/scratch/artifacts/oscillator/exp040")
        """New output directory for the native experiment launch."""

        phase: Literal["development", "audit"] = "audit"
        """Family collection evaluated by the native experiment launch."""

        data: ForecastData.Config = field(default_factory=ForecastData.Config)
        """Common geometry, initial-condition ranges, and split counts."""

        development: list[SystemFamily.Config] = field(
            default_factory=list[SystemFamily.Config],
        )
        """Mechanisms available before the audit is frozen."""

        audit: list[SystemFamily.Config] = field(
            default_factory=list[SystemFamily.Config],
        )
        """Mechanisms excluded from development."""

        budgets: list[ReferenceBudget.Config] = field(
            default_factory=lambda: [ReferenceBudget.Config()],
        )
        """Nested reference-data budgets, including validation."""

        recipes: list[BenchmarkRecipe.Config] = field(
            default_factory=list[BenchmarkRecipe.Config],
        )
        """Fixed competitors, each with its own declared model seeds."""

        baseline: str = "dlinear"
        """Method used for paired error comparisons."""

        summary: BenchmarkSummary.Config = field(
            default_factory=BenchmarkSummary.Config,
        )
        """System-level uncertainty calculation, separate from model RNG."""

        integration_refinement: int = 4
        """Substep multiplier for the finer-grid diagnostic."""

        integration_error_limit: float = 1e-3
        """Largest allowed training-scaled discrepancy from the finer grid."""

        @override
        def finalize(self) -> Self:
            """Finalize concrete owners while their task templates remain pending."""
            self.budgets = [budget.finalized() for budget in self.budgets]
            self.summary = self.summary.finalized()
            self.recipes = [recipe.finalized() for recipe in self.recipes]
            self.development = [family.finalized() for family in self.development]
            self.audit = [family.finalized() for family in self.audit]
            self._finalized = True
            return self

    def __init__(self, config: Config) -> None:
        for match in traverse(config.data, Maker, recurse=True):
            if getattr(match.config, "_finalized", False):
                raise ValueError("Benchmark suites require unfinalized templates.")
        names = [recipe.name for recipe in config.recipes]
        budget_names = [budget.name for budget in config.budgets]
        families = [*config.development, *config.audit]
        family_names = [family.name for family in families]
        if any(
            re.fullmatch(r"[a-z][a-z0-9_]*", name) is None
            for name in [*names, *budget_names, *family_names]
        ):
            raise ValueError(
                "Benchmark labels must be lowercase names without path separators.",
            )
        if not names or config.baseline not in names or len(set(names)) != len(names):
            raise ValueError(
                "Recipes need distinct names and must include the baseline.",
            )
        if not budget_names or len(set(budget_names)) != len(budget_names):
            raise ValueError("Budget labels must be distinct.")
        budgets = sorted(
            config.budgets,
            key=lambda budget: budget.train + budget.validation,
        )
        if any(
            small.train > large.train or small.validation > large.validation
            for small, large in pairwise(budgets)
        ):
            raise ValueError("Reference budgets must be nested in both splits.")
        if (
            not config.development
            or not config.audit
            or len(set(family_names)) != len(family_names)
        ):
            raise ValueError(
                "Development and audit must contain disjoint named families.",
            )
        development_laws = {
            family.reference.pformat(hide_default_values=False)
            for family in config.development
        }
        if development_laws.intersection(
            family.reference.pformat(hide_default_values=False)
            for family in config.audit
        ):
            raise ValueError(
                "An audit equation prototype was already used for development.",
            )
        system_seeds = [
            family.seed + index
            for family in families
            for index in range(family.instances)
        ]
        if len(set(system_seeds)) != len(system_seeds):
            raise ValueError("Every system needs an independent seed.")
        if any(seed < 0 or 10 * seed + 5 >= 2**32 for seed in system_seeds):
            raise ValueError(
                "System seeds must leave room for five distinct data streams.",
            )
        if (
            config.integration_refinement < 2
            or math.isnan(config.integration_error_limit)
            or config.integration_error_limit <= 0
            or math.isinf(config.integration_error_limit)
        ):
            raise ValueError(
                "Integration checks need a finer grid and a finite positive tolerance.",
            )
        for child in [*config.budgets, *config.recipes, *families]:
            child.make()
        self.config = config

    def tasks(
        self,
        phase: str,
    ) -> Iterator[tuple[str, ForecastData.Config]]:
        """Yield task data configs without constructing a forecasting model.

        Args:
          phase: The development or held-out family collection.

        Yields:
          task: System identity and its complete data config.

        """
        if phase not in ("development", "audit"):
            raise ValueError("Unknown benchmark phase.")
        families = (
            self.config.development if phase == "development" else self.config.audit
        )
        for family in families:
            sampler = family.make()
            for index in range(family.instances):
                cfg = self.config.data.copy_tree()
                cfg.reference = sampler.draw(index)
                cfg.reference.step_size = cfg.cheap.step_size
                cfg.reference.substeps = cfg.cheap.substeps
                seed = 10 * (family.seed + index)
                for offset, stream in enumerate(
                    (cfg.train, cfg.validation, cfg.synthetic, cfg.test, cfg.shifted),
                    start=1,
                ):
                    stream.seed = seed + offset
                yield f"{family.name}-{index:03d}", cfg

    def plan(
        self,
        phase: Literal["development", "audit"],
    ) -> Iterator[tuple[BenchmarkRecipe.Config, str, str, BudgetStudy.Config]]:
        """Yield each recipe on exactly the same task and reference budget.

        Args:
          phase: The development or held-out family collection.

        Yields:
          recipe: Named forecasting recipe and model seeds.
          system_id: Sampled system identity.
          run_id: Identity combining the system, budget, recipe, and seed.
          config: Pending study with its task, budget, and seed assigned.

        """
        for system_id, data in self.tasks(phase):
            for budget in self.config.budgets:
                for recipe in self.config.recipes:
                    for seed in recipe.seeds:
                        cfg = recipe.study.copy_tree()
                        cfg.data = data.copy_tree()
                        cfg.data.train.count = budget.train
                        cfg.data.validation.count = budget.validation
                        cfg.scoring_trajectories = min(
                            item.train for item in self.config.budgets
                        )
                        cfg.seed = seed
                        run_id = f"{system_id}--{budget.name}--{recipe.name}--{seed}"
                        yield recipe, system_id, run_id, cfg

    def diagnose(
        self,
        system_id: str,
        data: ForecastData.Config,
        *,
        budget: str,
    ) -> TaskDiagnostics:
        """Check split isolation, continuous rollout targets, and integration error.

        Args:
          system_id: Sampled system identity recorded with the diagnostics.
          data: Pending task data config to construct for these checks.
          budget: Reference-budget label recorded with the diagnostics.

        Returns:
          diagnostics: Unique-state count and scaled oracle and refinement errors.

        """
        source = data.make()
        cfg = source.config
        splits = [
            source.windows(stream, simulator=source.reference)
            for stream in (cfg.train, cfg.validation, cfg.test, cfg.shifted)
        ]
        states = torch.cat([split.media[:, 0] for split in splits])
        if torch.unique(states, dim=0).shape != states.shape:
            raise ValueError(
                "A trajectory's initial state appears in more than one split.",
            )
        count = min(item.train for item in self.config.budgets)
        prefix = Windows(media=splits[0].media[:count], label=splits[0].label[:count])
        _, scale = prefix.channel_stats()
        oracle_error = fine_error = 0.0
        for split in splits[2:]:
            oracle = source.reference.forecast_history(split.media, steps=cfg.pred_len)
            if not torch.equal(oracle, split.label):
                raise ValueError(
                    "The target does not continue the observed reference trajectory.",
                )
            score = forecast_scores(oracle, target=split.label, scale=scale)[
                "scaled_mse"
            ]
            if not math.isfinite(score):
                raise ValueError("Reference diagnostic scores must be finite.")
            oracle_error = max(oracle_error, score)
            refined = source.reference.forecast_history(
                split.media,
                steps=cfg.pred_len,
                substeps=source.reference.config.substeps
                * self.config.integration_refinement,
            )
            score = forecast_scores(refined, target=split.label, scale=scale)[
                "scaled_mse"
            ]
            if not math.isfinite(score):
                raise ValueError("Reference diagnostic scores must be finite.")
            fine_error = max(fine_error, score)
        if fine_error > self.config.integration_error_limit:
            raise ValueError(
                "The reference integration discrepancy exceeds the declared tolerance.",
            )
        return {
            "system_id": system_id,
            "budget": budget,
            "oracle_scaled_mse": oracle_error,
            "finer_grid_scaled_mse": fine_error,
            "unique_initial_states": len(states),
        }

    def summarize(
        self,
        records: list[BenchmarkRun],
        *,
        phase: Literal["development", "audit"],
    ) -> list[dict[str, object]]:
        """Build the expected run identities before reducing completed results.

        Args:
          records: Attempt records, including failed methods.
          phase: Family collection defining the expected run identities.

        Returns:
          summary: Paired system-level comparisons and completion counts.

        """
        expected: dict[str, RunSpec] = {}
        for recipe, system_id, run_id, cfg in self.plan(phase):
            expected[run_id] = {
                "method": recipe.name,
                "system_id": system_id,
                "family": system_id.rsplit("-", 1)[0],
                "model_seed": cast(int, cfg.seed),
                "access": recipe.access,
                "budget": run_id.split("--")[1],
                "experiment": cfg.experiment_name,
                "train": cfg.data.train.count,
                "validation": cfg.data.validation.count,
                "cheap": cfg.data.synthetic.count,
                "timesteps": cfg.data.seq_len + cfg.data.pred_len,
            }
        return self.config.summary.make().summarize(
            records,
            expected=expected,
            baseline=self.config.baseline,
        )

    def run(
        self,
        directory: Path | None = None,
        *,
        phase: Literal["development", "audit"] | None = None,
        resume: bool = False,
        provenance: dict[str, object] | None = None,
    ) -> tuple[list[BenchmarkRun], list[TaskDiagnostics]]:
        """Retain every attempt and write the scientific report.

        With no directory argument, use the configured output location and
        raise after writing the report if any method failed. Explicit directory
        calls retain failed records in their return value without raising.
        Resuming requires an identical declared run.

        Args:
          directory: New output directory; None uses the configured working_dir.
          phase: Family collection; None uses the configured phase.
          resume: Retain stored attempts after checking the run manifest.
          provenance: Optional caller-supplied identity checked when resuming.

        Returns:
          records: Every planned attempt, including failed methods.
          diagnostics: Integration and split-isolation checks by system and budget.

        """
        if directory is not None and not isinstance(directory, Path):
            raise TypeError("The output directory must be a Path or None.")
        native = directory is None
        directory = validated_output_path(
            self.config.working_dir if directory is None else directory,
        )
        phase = self.config.phase if phase is None else phase
        manifest = {
            "config_sha256": suite_digest(self.config),
            "torch_default_dtype": str(torch.get_default_dtype()),
            "phase": phase,
            "provenance": provenance,
        }
        if resume:
            if (
                json_loads(
                    (directory / "run-manifest.json").read_text(),
                    dict[str, object],
                )
                != manifest
            ):
                raise ValueError(
                    "Cannot resume a different protocol, source, or environment.",
                )
        else:
            directory.mkdir(parents=True, exist_ok=False)
            (directory / "run-manifest.json").write_text(
                json.dumps(to_plain(manifest), indent=2, allow_nan=False) + "\n",
            )
        records: list[BenchmarkRun] = []
        diagnostics: dict[tuple[str, str], TaskDiagnostics] = {}
        expected = {run_id for _, _, run_id, _ in self.plan(phase)}
        if any(path.stem not in expected for path in directory.glob("*--*.json")):
            raise ValueError("The output contains an unplanned run.")
        for recipe, system_id, run_id, cfg in self.plan(phase):
            budget = run_id.split("--")[1]
            key = system_id, budget
            if key not in diagnostics:
                diagnostics[key] = self.diagnose(system_id, cfg.data, budget=budget)
            destination = directory / f"{run_id}.json"
            if resume and destination.exists():
                raw = json_loads(destination.read_text(), dict[str, object])
                from_plain(raw, BenchmarkRun)
                # Keep opaque fit extensions after validating the common run schema.
                records.append(cast(BenchmarkRun, raw))
                logging.getLogger(__name__).info("%s retained %s", len(records), run_id)
                continue
            (directory / f"{run_id}.txt").write_text(
                cfg.pformat(hide_default_values=False) + "\n",
            )
            started = time.perf_counter()
            result: StudyRecord | None = None
            error: str | None = None
            try:
                result = _run_study(cfg)
            except Exception as exc:
                logging.getLogger(__name__).exception("Method failed: %s", run_id)
                error = f"{type(exc).__name__}: {exc}"
                result = None
            record: BenchmarkRun = {
                "run_id": run_id,
                "family": system_id.rsplit("-", 1)[0],
                "system_id": system_id,
                "budget": budget,
                "method": recipe.name,
                "access": recipe.access,
                "model_seed": cast(int, cfg.seed),
                "elapsed_seconds": time.perf_counter() - started,
                "status": "complete" if error is None else "failed",
                "error": error,
                "result": result,
            }
            temporary = destination.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(to_plain(record), indent=2, allow_nan=False) + "\n",
            )
            temporary.replace(destination)
            records.append(record)
            logging.getLogger(__name__).info(
                "%s %s %s",
                len(records),
                record["status"],
                run_id,
            )
        report = {
            "version": self.config.version,
            "phase": phase,
            "diagnostics": list(diagnostics.values()),
            "summary": self.summarize(records, phase=phase),
            "runs": records,
        }
        (directory / "protocol.txt").write_text(
            self.config.pformat(
                hide_default_values=False,
                mask_memory_addresses=True,
            )
            + "\n",
        )
        destination = directory / "results.json"
        destination.write_text(
            json.dumps(to_plain(report), indent=2, allow_nan=False) + "\n",
        )
        logging.getLogger(__name__).info("Results: %s", destination)
        if native and any(record["status"] != "complete" for record in records):
            raise RuntimeError(
                f"Benchmark methods failed; records saved to {destination}",
            )
        return records, list(diagnostics.values())


def suite_digest(config: BenchmarkSuite.Config) -> str:
    """Hash the complete declared protocol, including all recipes and seeds."""
    return hashlib.sha256(
        config.pformat(hide_default_values=False, mask_memory_addresses=True).encode(),
    ).hexdigest()


def _run_study(config: ForecastStudy.Config) -> StudyRecord:
    result = config.make().run()
    if any(
        not math.isfinite(value)
        for split in result["scores"].values()
        for scores in split.values()
        for value in (scores["mse"], scores["mae"], scores["scaled_mse"])
    ):
        raise ValueError("A forecast produced a nonfinite metric.")
    return result
