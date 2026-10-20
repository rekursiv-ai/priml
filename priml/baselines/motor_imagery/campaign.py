"""Timed complete public-reference workflow built from Priml training loops."""

from __future__ import annotations

from dataclasses import field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, Self, cast, override, runtime_checkable

import contextlib
import math
import random
import time

from configgle import Fig, Makeable
from torch import Tensor, nn
from torch.nn import functional
from treekle import from_plain

import numpy as np
import torch

from priml.baselines.motor_imagery.custom_types import FoldPolicy
from priml.baselines.motor_imagery.data import (
    NUM_CLASSES,
    Trial,
    write_submission,
)
from priml.baselines.motor_imagery.folds import GroupedFolds
from priml.baselines.motor_imagery.metric import EEGAccuracy, pooled_accuracy
from priml.baselines.motor_imagery.train_loop import MotorImageryLoop
from priml.baselines.motor_imagery.train_step import MotorImageryStep
from priml.math.custom_types import TensorFn
from priml.math.seed import get_rng_state, set_rng_state
from priml.model.custom_types import ChannelsIn, propagate_attr
from priml.paths import resolve_working_dir, validated_output_path
from priml.runtime import RuntimeProtocol, runtime_initialized
from priml.train.budget import WallBudget, provenance


if TYPE_CHECKING:
    from collections.abc import Generator


class MotorImageryCampaign:
    """Run all validation folds, then train and average the full-cohort ensemble."""

    class Config(Fig["MotorImageryCampaign"]):
        study_name: str = ""
        """Study identity used in campaign artifacts and inherited by both loops."""

        experiment_name: str = ""
        """Recipe identity, inherited by both loops."""

        doc: str = ""
        """Factory contract stamped by the standard Priml launcher."""

        base_dir: Path | str | None = "/opt/scratch"
        """Root for the declared run-output pattern."""

        working_dir: Path | str = "/runs/{study_name}/{experiment_name}"
        """Destination for checkpoints, predictions, resolved config, and results."""

        validation: MotorImageryLoop.Config = field(
            default_factory=MotorImageryLoop.Config,
        )
        """Loop fitting one validation fold: model, raw data, preprocessing, runtime.

        The campaign reads the raw data location, preprocessing and runtime from
        this loop alone; its fold people come from ``folds``."""

        ensemble_loop: MotorImageryLoop.Config = field(
            default_factory=MotorImageryLoop.Config,
        )
        """Loop fitting each ensemble member on the whole cohort.

        Its ``step.model`` and ``dataset.preprocessing`` are never built: each
        member supplies its own model and the campaign shares one preprocessing
        result. Its runtime must equal the validation loop's, since one process
        holds one runtime."""

        ensemble: list[Makeable[nn.Module]] = field(
            default_factory=list[Makeable[nn.Module]],
        )
        """All full-cohort architectures, in source construction order."""

        folds: Makeable[FoldPolicy] = field(default_factory=GroupedFolds.Config)
        """Validation cohorts, each held out of its fold's training."""

        seed: int = 0
        """Unsalted source seed, applied once before diagnostic initialization.

        The source seeds once and draws every later initialization, dropout and
        shuffle from that one stream. Each loop's own seeding would restart it,
        so the campaign restores its stream after building every loop."""

        diagnostic_batch_size: int = 1
        """Examples in the disposable pre-validation forward pass of each model.

        The source's diagnostics consume initialization and dropout draws, so
        their batch size changes every later random value."""

        expected_submission_rows: int = 0
        """Official competition metadata and source submission count."""

        budget: WallBudget.Config = field(default_factory=WallBudget.Config)
        """Outer cold-preparation, fit, evaluation, and publication allowance."""

        @override
        def finalize(self) -> Self:
            """Propagate preprocessing dimensions into every configured model."""
            working_dir = (
                self.working_dir.format(
                    study_name=self.study_name,
                    experiment_name=self.experiment_name,
                )
                if isinstance(self.working_dir, str)
                else self.working_dir
            )
            self.working_dir = resolve_working_dir(self.base_dir, working_dir)
            if self.budget.receipt_path is None:
                self.budget.receipt_path = Path(self.working_dir) / "result.json"
            preprocessing = self.validation.dataset.preprocessing
            for loop in (self.validation, self.ensemble_loop):
                if not loop.study_name:
                    loop.study_name = self.study_name
                if not loop.experiment_name:
                    loop.experiment_name = self.experiment_name
            for config in [self.validation.step.model, *self.ensemble]:
                propagate_attr(
                    config,
                    name="channels_in",
                    value=len(preprocessing.channels),
                    protocol=ChannelsIn,
                )
                propagate_attr(config, name="channels_out", value=NUM_CLASSES)
                propagate_attr(
                    config,
                    name="num_samples",
                    value=preprocessing.window_size,
                    protocol=_NumSamples,
                )
            return super().finalize()

    def __init__(self, config: Config) -> None:
        if config.ensemble_loop.runtime != config.validation.runtime:
            raise ValueError("The validation and ensemble loops must share a runtime.")
        self.config = config
        self.folds = config.folds.make()

    def run(self, *args: str) -> None:
        """Execute the complete timed recipe through the standard Priml launcher.

        Args:
          *args: No campaign-specific passthrough arguments are accepted.

        Raises:
          ValueError: Arguments, paths, or held-out subject membership are invalid.
          FileNotFoundError: Real raw data has not been staged.

        """
        if args:
            raise ValueError(f"Unexpected campaign arguments: {args}.")
        cfg = self.config
        budget = cfg.budget.make()
        directory = Path(cfg.validation.dataset.working_dir)
        train_dir = directory / "Training set"
        test_dir = directory / "Cross participant test set"
        if not train_dir.is_dir() or not test_dir.is_dir():
            raise FileNotFoundError(
                f"Raw EEG competition data missing at {directory}. Run the documented "
                "prepare_data staging command with existing Kaggle authentication.",
            )
        output = validated_output_path(cfg.working_dir, protected=[directory])
        output.mkdir(parents=True, exist_ok=False)
        budget.metadata.update(
            {
                "study_name": cfg.study_name,
                "experiment_name": cfg.experiment_name,
                "run_id": str(output.resolve()),
                "started_unix_sec": time.time(),
                "metric_key": "eval/accuracy",
                "direction": "maximize",
                "eval/accuracy": None,
                "budget_enforced": math.isfinite(budget.seconds),
                "reference_commit": "e2d892f158c68afa92ecda8229b9ad255c7d0aaf",
                "reference_notebook_version": 353_208_444,
                "reference_public_score": 0.56,
                "parity_status": "target-backend-and-full-run-not-verified",
            },
        )
        with budget, budget.interruptible():
            (output / "config.txt").write_text(
                cfg.pformat(hide_default_values=False) + "\n",
            )
            random.seed(cfg.seed)
            torch.manual_seed(cfg.seed)
            runtime = cfg.validation.runtime.make()
            with _runtime_scope(runtime):
                budget.metadata.update(
                    self._execute(output=output, device=runtime.device),
                )

    def _execute(self, *, output: Path, device: torch.device) -> dict[str, object]:
        cfg = self.config
        directory = Path(cfg.validation.dataset.working_dir)
        preprocessing = cfg.validation.dataset.preprocessing
        _diagnostic_draws(
            cfg.ensemble,
            batch_size=cfg.diagnostic_batch_size,
            channels=len(preprocessing.channels),
            num_samples=preprocessing.window_size,
        )
        processor = preprocessing.make()
        trials = processor.read(
            sorted((directory / "Training set").glob("*.csv")),
            labelled=True,
        )
        test_trials = processor.read(
            sorted((directory / "Cross participant test set").glob("*.csv")),
            labelled=False,
        )
        if {trial.subject for trial in trials} & {
            trial.subject for trial in test_trials
        }:
            raise ValueError("Training and test people overlap.")
        fold_metrics = self._validate(trials, output=output)
        models = [_build_cpu(config).to(device) for config in cfg.ensemble]
        fitted: list[MotorImageryStep] = []
        for index, model in enumerate(models):
            loop = self._make_loop(
                model,
                config=cfg.ensemble_loop,
                trials=trials,
                validation_subjects=[],
            )
            loop.train()
            assert isinstance(loop.step, MotorImageryStep)
            fitted.append(loop.step)
            with loop.step.ema.apply_to(loop.step.model):
                torch.save(
                    loop.step.model.state_dict(),
                    f=output / f"ensemble-{index}.pt",
                )
        probabilities = predict(fitted, trials=test_trials, device=device)
        write_submission(
            output / "submission.csv",
            probabilities=probabilities,
            expected_rows=cfg.expected_submission_rows,
        )
        return {
            "eval/accuracy": fold_metrics["trial_accuracy"],
            **fold_metrics,
            "validation_subjects": sorted(
                {person for people in self.folds(trials) for person in people},
            ),
            "hardware": str(device),
            **provenance(packages=["torch"]),
            "submission_rows": len(probabilities),
        }

    def _validate(self, trials: list[Trial], *, output: Path) -> dict[str, float]:
        probabilities: list[Tensor] = []
        labels: list[Tensor] = []
        identifiers: list[Tensor] = []
        correct = 0
        for fold, people in enumerate(self.folds(trials)):
            model = _build_cpu(self.config.validation.step.model)
            loop = self._make_loop(
                model,
                config=self.config.validation,
                trials=trials,
                validation_subjects=people,
            )
            loop.train()
            metric = loop.metrics_eval[""]
            assert isinstance(metric, EEGAccuracy)
            if metric.best_probabilities.numel() == 0:
                raise ValueError("No validation epoch exceeded zero window accuracy.")
            probabilities.append(metric.best_probabilities)
            labels.append(metric.best_labels)
            identifiers.append(metric.best_trials)
            correct += metric.best_correct
            torch.save(metric.state_dict(), f=output / f"fold-{fold}.pt")
        probability = torch.cat(probabilities)
        label = torch.cat(labels)
        trial = torch.cat(identifiers)
        return {
            "trial_accuracy": pooled_accuracy(probability, labels=label, trials=trial),
            "source_window_accuracy": correct / len(label),
            "validation_trials": float(len(set(from_plain(trial.tolist(), list[int])))),
        }

    def _make_loop(
        self,
        model: nn.Module,
        *,
        config: MotorImageryLoop.Config,
        trials: list[Trial],
        validation_subjects: list[str],
    ) -> MotorImageryLoop:
        cfg = config.copy_tree()
        cfg.dataset.validation_subjects = validation_subjects
        # TrainLoop salts seeds per independent run. These models are already
        # initialized: restore the campaign's continuous source stream afterward.
        state = get_rng_state()
        # The source seeds CPU and pending CUDA generators identically. CUDA
        # may first initialize inside cfg.make(), after Priml's salted reseed;
        # an absent CUDA snapshot must restore that original pending seed too.
        pending_cuda_seed = None if "cuda" in state else torch.initial_seed()
        try:
            step = cfg.step.parent_class(cfg.step, model=model)
            dataset = cfg.dataset.parent_class(cfg.dataset, trials=trials)
            loop = cast(object, cfg.parent_class(cfg, step=step, dataset=dataset))
            assert isinstance(loop, MotorImageryLoop)
            return loop
        finally:
            set_rng_state(state)
            if pending_cuda_seed is not None:
                torch.cuda.manual_seed_all(pending_cuda_seed)


def predict(
    steps: list[MotorImageryStep],
    *,
    trials: list[Trial],
    device: torch.device,
) -> list[float]:
    """Average every model/window probability using the reference NumPy reduction.

    Args:
      steps: All fitted ensemble members in source order.
      trials: Sorted test-file cues, including the source short-cue fallback.
      device: Model execution device.

    Returns:
      probabilities: P(move) per cue, using the source's float32 mean order.

    """
    probabilities: list[float] = []
    with torch.no_grad():
        for trial in trials:
            media = trial.windows[:, None].to(device=device, dtype=torch.float32)
            window_probs = [
                functional.softmax(
                    cast(Tensor, step.call_eval(media)).to(dtype=torch.float32),
                    dim=1,
                )[:, 1]
                .cpu()
                .numpy()
                for step in steps
            ]
            probabilities.append(float(np.mean(window_probs)))
    return probabilities


def _build_cpu(config: Makeable[nn.Module]) -> nn.Module:
    if torch.get_default_device().type == "cpu":
        return config.make()
    with torch.device("cpu"):
        return config.make()


def _diagnostic_draws(
    configs: list[Makeable[nn.Module]],
    *,
    batch_size: int,
    channels: int,
    num_samples: int,
) -> None:
    models = [_build_cpu(config) for config in configs]
    # These notebook diagnostics precede CV and consume initialization/dropout draws.
    dummy = torch.randn(batch_size, 1, channels, num_samples, device="cpu")
    with torch.no_grad():
        for model in models:
            cast(TensorFn, model)(dummy)


@runtime_checkable
class _NumSamples(Protocol):
    num_samples: int


@contextlib.contextmanager
def _runtime_scope(runtime: RuntimeProtocol) -> Generator[None]:
    owns_runtime = not runtime_initialized()
    runtime.initialize()
    try:
        yield
    finally:
        if owns_runtime:
            runtime.destroy()
