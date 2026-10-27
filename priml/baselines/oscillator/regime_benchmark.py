"""Evaluate hidden velocity and observed regime changes without future leakage."""

from __future__ import annotations

from dataclasses import field
from pathlib import Path
from typing import Literal, Protocol, Self, TypedDict, override, runtime_checkable

import hashlib
import json
import logging
import math
import platform
import re
import time

from configgle import Fig, Makeable
from torch import Tensor
from treekle import json_loads

import torch

from priml.baselines.oscillator.adaptation import (
    ObservedForecaster,
    ObservedGeometry,
    ObservedNeuralForecast,
)
from priml.baselines.oscillator.hidden import HiddenData, HiddenDomain
from priml.baselines.oscillator.regimes import DynamicsDomain, RegimeData
from priml.model.custom_types import propagate_attr
from priml.paths import validated_output_path


@runtime_checkable
class ObservationChannels(Protocol):
    """Visible width declared without building the observation transform."""

    @property
    def channels(self) -> int:
        """Number of observed coordinates."""
        ...


class EpisodeScores(TypedDict):
    """Errors retaining independent episodes and forecast horizons separately."""

    scaled_mse: list[float]
    horizon_scaled_mse: list[float]


class RegimeRun(TypedDict):
    """One declared model-seed attempt, including failures."""

    run_id: str
    method: str
    seed: int
    status: str
    error: str | None
    fit: dict[str, int | float]
    families: dict[str, EpisodeScores]
    diagnostics: dict[str, float]
    elapsed_seconds: float


class RegimeFamily:
    """Keep an evaluation domain and its independent episode stream together."""

    class Config(Fig["RegimeFamily"]):
        name: str = "in_domain"
        """Result label, not an implementation selector."""

        domain: DynamicsDomain.Config | HiddenDomain.Config = field(
            default_factory=DynamicsDomain.Config,
        )
        """Coefficients sampled after the observed change."""

        seed: int = 3301
        """Evaluation-only random stream."""

        episodes: int = 32
        """Independent coefficient and initial-state draws."""

    def __init__(self, config: Config) -> None:
        if (
            not re.fullmatch(r"[a-z][a-z0-9_]*", config.name)
            or config.episodes < 1
            or not 0 <= config.seed < 2**32
        ):
            raise ValueError(
                "Evaluation families need a label, episodes, and a valid seed.",
            )
        self.config = config


class ObservedRecipe:
    """Declare one forecaster and its repeated initialization seeds."""

    class Config(Fig["ObservedRecipe"]):
        name: str = "dlinear"
        """Method label recorded with its actual configuration."""

        method: Makeable[ObservedForecaster] = field(
            default_factory=ObservedNeuralForecast.Config,
        )
        """Forecaster fitted without evaluator-only state."""

        seeds: list[int] = field(default_factory=lambda: [2021, 2022, 2023])
        """Repeated model seeds, averaged before episode uncertainty."""

    def __init__(self, config: Config) -> None:
        if (
            not re.fullmatch(r"[a-z][a-z0-9_]*", config.name)
            or not config.seeds
            or len(config.seeds) != len(set(config.seeds))
            or any(not 0 <= v < 2**32 for v in config.seeds)
        ):
            raise ValueError("Recipes need a label and distinct valid seeds.")
        self.config = config


class RegimeBenchmark:
    """Keep one observation ablation fixed while evaluating every declared recipe."""

    class Config(Fig["RegimeBenchmark"]):
        experiment_name: str = "exp041"
        """Experiment identity used in reports."""

        working_dir: Path = Path("/opt/scratch/artifacts/oscillator/exp041")
        """New output location; generated records remain outside Git."""

        phase: Literal["development", "audit"] = "development"
        """Evaluation collection; audit uses independently reserved streams."""

        data: RegimeData.Config | HiddenData.Config = field(
            default_factory=RegimeData.Config,
        )
        """Motion, schedule, and variables exposed to a method."""

        train_episodes: int = 256
        """Accurate trajectories available for fitting."""

        validation_episodes: int = 64
        """Accurate trajectories reserved for checkpoint selection."""

        scoring_episodes: int = 256
        """Shared training prefix setting the position-error denominator."""

        train_seed: int = 1101
        """Independent fitting-data stream."""

        validation_seed: int = 2201
        """Independent model-selection stream."""

        recipes: list[ObservedRecipe.Config] = field(
            default_factory=lambda: [ObservedRecipe.Config()],
        )
        """Every competitor and its model seeds."""

        comparisons: list[tuple[str, str]] = field(
            default_factory=list[tuple[str, str]],
        )
        """Paired method-minus-reference differences to report."""

        development: list[RegimeFamily.Config] = field(
            default_factory=lambda: [RegimeFamily.Config()],
        )
        """Families available while designing the experiment."""

        audit: list[RegimeFamily.Config] = field(
            default_factory=list[RegimeFamily.Config],
        )
        """Fresh parameter domains excluded from development evaluation."""

        resamples: int = 2000
        """Paired episode-bootstrap draws within each fixed family."""

        bootstrap_seed: int = 8041
        """Uncertainty stream, separate from data and models."""

        threads: int = 1
        """Torch CPU threads for a reproducible local execution budget."""

        @override
        def finalize(self) -> Self:
            self.data = self.data.finalized()
            observation = self.data.observation
            if not isinstance(observation, ObservationChannels):
                raise TypeError("The observation config must declare its channels.")
            for recipe in self.recipes:
                if isinstance(recipe.method, ObservedGeometry):
                    propagate_attr(
                        recipe.method,
                        "seq_len",
                        self.data.seq_len,
                        protocol=ObservedGeometry,
                    )
                    propagate_attr(
                        recipe.method,
                        "pred_len",
                        self.data.pred_len,
                        protocol=ObservedGeometry,
                    )
                    propagate_attr(
                        recipe.method,
                        "channels",
                        observation.channels,
                        protocol=ObservedGeometry,
                    )
            return super().finalize()

    def __init__(self, config: Config) -> None:
        if (
            min(
                config.train_episodes,
                config.validation_episodes,
                config.scoring_episodes,
                config.resamples,
                config.threads,
            )
            <= 0
            or config.scoring_episodes > config.train_episodes
        ):
            raise ValueError(
                "Training, uncertainty, and execution budgets must be valid.",
            )
        names = [r.name for r in config.recipes]
        families = [*config.development, *config.audit]
        if (
            not names
            or len(names) != len(set(names))
            or len({f.name for f in families}) != len(families)
        ):
            raise ValueError("Recipe and family names must be nonempty and distinct.")
        if any(
            a not in names or b not in names or a == b for a, b in config.comparisons
        ):
            raise ValueError("Paired comparisons must name two distinct recipes.")
        if config.phase not in ("development", "audit") or not config.development:
            raise ValueError("A study needs a valid phase and development families.")
        seeds = [config.train_seed, config.validation_seed, *[f.seed for f in families]]
        if len(seeds) != len(set(seeds)) or any(not 0 <= s < 2**32 for s in seeds):
            raise ValueError(
                "Training, validation, and evaluation streams must be distinct.",
            )
        if config.phase == "audit" and not config.audit:
            raise ValueError("An audit needs reserved evaluation families.")
        for recipe in config.recipes:
            recipe.make()
        for family in families:
            family.make()
        self.config = config

    def run(
        self,
        directory: Path | None = None,
        *,
        resume: bool = False,
    ) -> dict[str, object]:
        """Fit before constructing evaluation pools and retain every attempt.

        Position error is reported for every observation ablation. The oracle
        is an evaluator-only continuation check, never a competing learner.
        A failed seed invalidates its method's finite aggregate score.

        """
        cfg = self.config
        output = validated_output_path(
            cfg.working_dir if directory is None else directory,
        )
        config_text = cfg.pformat(hide_default_values=False)
        manifest = {
            "config_sha256": hashlib.sha256(config_text.encode()).hexdigest(),
            "source_sha256": {
                str(p.relative_to(Path(__file__).resolve().parents[2])): hashlib.sha256(
                    p.read_bytes(),
                ).hexdigest()
                for p in sorted(Path(__file__).resolve().parents[2].rglob("*.py"))
                if "__pycache__" not in p.parts
            },
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "threads": cfg.threads,
        }
        if resume:
            if json.loads((output / "manifest.json").read_text()) != manifest:
                raise ValueError(
                    "Cannot resume changed source, configuration, or environment.",
                )
        else:
            output.mkdir(parents=True, exist_ok=False)
            _write_json(output / "manifest.json", manifest)
            (output / "protocol.txt").write_text(config_text + "\n")
        data = cfg.data.make()
        train = data.sample(count=cfg.train_episodes, seed=cfg.train_seed).windows
        validation = data.sample(
            count=cfg.validation_episodes,
            seed=cfg.validation_seed,
        ).windows
        scale = (
            train.media[: cfg.scoring_episodes, 0, 0].std(correction=0).clamp_min(1e-6)
        )
        records: list[RegimeRun] = []
        expected = {f"{r.name}--{seed}" for r in cfg.recipes for seed in r.seeds}
        if any(p.stem not in expected for p in output.glob("*--*.json")):
            raise ValueError("The output contains an unplanned attempt.")
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(cfg.threads)
        try:
            for recipe in cfg.recipes:
                for seed in recipe.seeds:
                    run_id = f"{recipe.name}--{seed}"
                    destination = output / f"{run_id}.json"
                    if resume and destination.exists():
                        record = json_loads(destination.read_text(), RegimeRun)
                        if record.get("run_id") != run_id:
                            raise ValueError(
                                "A retained result has the wrong identity.",
                            )
                        records.append(record)
                        continue
                    started = time.perf_counter()
                    record = RegimeRun(
                        run_id=run_id,
                        method=recipe.name,
                        seed=seed,
                        status="failed",
                        error=None,
                        fit={},
                        families={},
                        diagnostics={},
                        elapsed_seconds=0.0,
                    )
                    try:
                        with torch.random.fork_rng(devices=[]):
                            torch.default_generator.manual_seed(seed)
                            method = recipe.method.make()
                            fit = method.fit(
                                train,
                                validation=validation,
                                dynamics=data.dynamics,
                            )
                            _validate_fit(fit)
                            scores, diagnostics = self.evaluate(
                                method,
                                data=data,
                                scale=scale,
                            )
                        record.update(
                            status="complete",
                            fit=fit,
                            families=scores,
                            diagnostics=diagnostics,
                        )
                    except Exception as exc:
                        logging.getLogger(__name__).exception("Failed %s", run_id)
                        record["error"] = f"{type(exc).__name__}: {exc}"
                    record["elapsed_seconds"] = time.perf_counter() - started
                    _write_json(destination, record)
                    records.append(record)
                    logging.getLogger(__name__).info("%s %s", record["status"], run_id)
        finally:
            torch.set_num_threads(previous_threads)
        report: dict[str, object] = {
            "version": "0.2.0",
            "experiment": cfg.experiment_name,
            "phase": cfg.phase,
            "train_episodes": cfg.train_episodes,
            "validation_episodes": cfg.validation_episodes,
            "scoring_variance": float(scale.square()),
            "manifest": manifest,
            "runs": records,
            "summary": self.summarize(records),
            "comparisons": self.compare(records),
        }
        _write_json(output / "results.json", report)
        if any(r["status"] != "complete" for r in records):
            raise RuntimeError(
                "A forecasting method failed; attempts are retained in results.json.",
            )
        return report

    def evaluate(
        self,
        method: ObservedForecaster,
        *,
        data: RegimeData | HiddenData,
        scale: Tensor,
    ) -> tuple[dict[str, EpisodeScores], dict[str, float]]:
        """Keep evaluator-only parameters outside the forecaster's input."""
        scores: dict[str, EpisodeScores] = {}
        diagnostics: dict[str, float] = {}
        cfg = self.config
        families = cfg.development if cfg.phase == "development" else cfg.audit
        for family in families:
            episode = data.sample(
                count=family.episodes,
                seed=family.seed,
                domain=family.domain,  # pyright: ignore[reportArgumentType]
            )
            # Clean position: observations may carry sensor noise, scores never do.
            target = episode.states[:, cfg.data.seq_len :, 0]
            prediction = method.predict(episode.windows.media)
            if (
                prediction.shape != episode.windows.label.shape
                or not torch.isfinite(prediction).all()
            ):
                raise ValueError("A method returned an invalid observed forecast.")
            errors = ((prediction[..., 0] - target) / scale).square()
            if not torch.isfinite(errors).all():
                raise ValueError("A method produced nonfinite scored errors.")
            scores[family.name] = {
                "scaled_mse": [float(v) for v in errors.mean(1)],
                "horizon_scaled_mse": [float(v) for v in errors.mean(0)],
            }
            oracle = data.dynamics.forecast(
                episode.states[:, cfg.data.seq_len - 1],
                coefficients=episode.after,
                steps=cfg.data.pred_len,
            )
            oracle_error = float(
                (oracle - episode.states[:, cfg.data.seq_len :]).abs().max(),
            )
            if oracle_error != 0:
                raise ValueError("Reference continuation does not match its targets.")
            diagnostics[family.name] = oracle_error
        return scores, diagnostics

    def summarize(self, records: list[RegimeRun]) -> list[dict[str, object]]:
        """Average model seeds before resampling independent paired episodes."""
        cfg = self.config
        families = cfg.development if cfg.phase == "development" else cfg.audit
        expected = {f"{r.name}--{s}" for r in cfg.recipes for s in r.seeds}
        actual = [r["run_id"] for r in records]
        if len(actual) != len(set(actual)) or set(actual) != expected:
            raise ValueError("Every planned attempt must appear exactly once.")
        for record in records:
            if (
                record["run_id"] != f"{record['method']}--{record['seed']}"
                or record["status"] not in ("complete", "failed")
                or not math.isfinite(record["elapsed_seconds"])
                or record["elapsed_seconds"] < 0
            ):
                raise ValueError("A retained attempt has invalid identity or status.")
            if record["status"] == "complete":
                if record["error"] is not None or set(record["families"]) != {
                    f.name for f in families
                }:
                    raise ValueError(
                        "A complete attempt needs exactly its planned families.",
                    )
                for family in families:
                    horizon = torch.tensor(
                        record["families"][family.name]["horizon_scaled_mse"],
                    )
                    if (
                        horizon.shape != (cfg.data.pred_len,)
                        or not torch.isfinite(horizon).all()
                        or (horizon < 0).any()
                    ):
                        raise ValueError(
                            "Retained horizon errors do not match the protocol.",
                        )
        rows: list[dict[str, object]] = []
        for recipe in cfg.recipes:
            group = [r for r in records if r["method"] == recipe.name]
            if any(r["status"] != "complete" for r in group):
                rows.append(
                    {"method": recipe.name, "status": "failed", "scaled_mse": None},
                )
                continue
            family_errors = self.episode_errors(records, name=recipe.name)
            mean, bounds = self.bootstrap(family_errors)
            rows.append(
                {
                    "method": recipe.name,
                    "status": "complete",
                    "scaled_mse": mean,
                    "interval": bounds,
                    "families": {k: float(v.mean()) for k, v in family_errors.items()},
                    "independent_episodes": sum(f.episodes for f in families),
                    "model_seeds": len(recipe.seeds),
                },
            )
        return rows

    def episode_errors(
        self,
        records: list[RegimeRun],
        *,
        name: str,
    ) -> dict[str, Tensor]:
        """Average repeated fits without counting them as independent episodes."""
        cfg = self.config
        recipe = next(r for r in cfg.recipes if r.name == name)
        families = cfg.development if cfg.phase == "development" else cfg.audit
        group = [r for r in records if r["method"] == name]
        result: dict[str, Tensor] = {}
        for family in families:
            if any(
                len(r["families"][family.name]["scaled_mse"]) != family.episodes
                for r in group
            ):
                raise ValueError("Retained episode errors do not match the protocol.")
            errors = torch.tensor(
                [r["families"][family.name]["scaled_mse"] for r in group],
                dtype=torch.float64,
            )
            if (
                errors.shape != (len(recipe.seeds), family.episodes)
                or not torch.isfinite(errors).all()
                or (errors < 0).any()
            ):
                raise ValueError("Retained episode errors do not match the protocol.")
            result[family.name] = errors.mean(0)
        return result

    def bootstrap(self, values: dict[str, Tensor]) -> tuple[float, list[float]]:
        """Resample independent episodes within each declared family."""
        cfg = self.config
        generator = torch.Generator().manual_seed(cfg.bootstrap_seed)
        samples = torch.zeros(cfg.resamples, dtype=torch.float64)
        for errors in values.values():
            indices = torch.randint(
                len(errors),
                (cfg.resamples, len(errors)),
                generator=generator,
            )
            samples += errors[indices].mean(1) / len(values)
        bounds = torch.quantile(
            samples,
            torch.tensor([0.025, 0.975], dtype=samples.dtype),
        )
        return sum(float(v.mean()) for v in values.values()) / len(values), [
            float(v) for v in bounds
        ]

    def compare(self, records: list[RegimeRun]) -> list[dict[str, object]]:
        """Compare methods on the same episodes, retaining their paired errors."""
        comparisons: list[dict[str, object]] = []
        for method, reference in self.config.comparisons:
            group = [r for r in records if r["method"] in (method, reference)]
            if any(r["status"] != "complete" for r in group):
                comparisons.append(
                    {"method": method, "reference": reference, "status": "failed"},
                )
                continue
            left = self.episode_errors(records, name=method)
            right = self.episode_errors(records, name=reference)
            difference = {key: left[key] - right[key] for key in left}
            mean, bounds = self.bootstrap(difference)
            comparisons.append(
                {
                    "method": method,
                    "reference": reference,
                    "status": "complete",
                    "scaled_mse_difference": mean,
                    "interval": bounds,
                },
            )
        return comparisons


class RegimeCampaign:
    """Run the matched observation and dynamics ablations under one declaration."""

    class Config(Fig["RegimeCampaign"]):
        working_dir: Path = Path("/opt/scratch/artifacts/oscillator/exp045")
        """Parent location for the separate condition records."""

        studies: list[RegimeBenchmark.Config] = field(
            default_factory=list[RegimeBenchmark.Config],
        )
        """Concrete experiments; every condition retains its own full protocol."""

    def __init__(self, config: Config) -> None:
        names = [s.experiment_name for s in config.studies]
        if not names or len(names) != len(set(names)):
            raise ValueError("A campaign needs distinct named conditions.")
        self.config = config

    def run(
        self,
        directory: Path | None = None,
        *,
        resume: bool = False,
    ) -> list[dict[str, object]]:
        """Run every ablation and leave its per-episode evidence available."""
        output = validated_output_path(
            self.config.working_dir if directory is None else directory,
        )
        if not resume:
            output.mkdir(parents=True, exist_ok=False)
        results = [
            study.make().run(
                output / study.experiment_name,
                resume=resume and (output / study.experiment_name).exists(),
            )
            for study in self.config.studies
        ]
        _write_json(output / "results.json", results)
        return results


def _validate_fit(fit: dict[str, int | float]) -> None:
    """Reject invalid diagnostics before persisting a successful attempt."""
    if not all(math.isfinite(v) for v in fit.values()):
        raise ValueError("Fit diagnostics must be finite.")


def _write_json(path: Path, value: object) -> None:
    """Replace a complete local record only after its temporary file is written."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
