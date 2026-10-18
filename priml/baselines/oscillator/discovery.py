"""Fit small force combinations and choose them using validation forecasts."""

from __future__ import annotations

from dataclasses import field
from itertools import combinations
from typing import TYPE_CHECKING, Protocol, TypedDict, cast

import math

from configgle import Fig, Makeable, PartialConfig
from scipy.optimize import OptimizeResult, least_squares
from torch import Tensor

import numpy as np
import torch

from priml.baselines.oscillator.benchmark import forecast_scores
from priml.baselines.oscillator.data import STATE_CHANNELS, Oscillator, Windows
from priml.baselines.oscillator.forces import (
    CorrectedOscillator,
    ForceTerm,
    PowerForce,
    WeightedForce,
)
from priml.baselines.oscillator.train_step import FitReport


if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray


class CoefficientSearch(Protocol):
    """Fit force strengths to a vector of training residuals."""

    def __call__(
        self,
        fun: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        x0: NDArray[np.float64],
        /,
    ) -> OptimizeResult:
        """Return fitted strengths and convergence status."""
        ...


class FittedForce(TypedDict):
    """One selected term and its fitted strength."""

    name: str
    coefficient: float


class DiscoveryReport(FitReport):
    """The chosen equation, validation score, and search cost."""

    fitted_forces: list[FittedForce]
    selection_score: float
    candidates_tried: int
    candidates_converged: int
    training_evaluations: int
    validation_evaluations: int


class SparseSimulator:
    """Select a small force correction from an explicit candidate library."""

    class Config(Fig["SparseSimulator"]):
        terms: list[Makeable[ForceTerm]] = field(
            default_factory=lambda: [
                PowerForce.Config(axis=0, power=1),
                PowerForce.Config(axis=0, power=3),
                PowerForce.Config(axis=0, power=5),
                PowerForce.Config(axis=1, power=1),
                PowerForce.Config(axis=1, power=2),
                PowerForce.Config(axis=1, power=3),
            ],
        )
        """Allowed force shapes; their presence is prior knowledge."""

        max_terms: int = 2
        """Largest combination to fit; every smaller combination is also tried."""

        complexity_penalty: float = 1e-6
        """Validation scaled-MSE penalty for each selected term."""

        initial_coefficient: float = 0.1
        """Starting strength for every candidate term."""

        search: Makeable[CoefficientSearch] = field(
            default_factory=lambda: PartialConfig(
                least_squares,
                bounds=(0.0, 2.0),
                max_nfev=40,
                ftol=1e-9,
                xtol=1e-9,
                gtol=1e-9,
            ),
        )
        """Bounded training fit, including its numerical stopping rules."""

        dtype: torch.dtype = torch.float64
        """Precision for fitting and comparing candidate trajectories."""

    def __init__(self, config: Config) -> None:
        if config.max_terms < 0 or config.max_terms > len(config.terms):
            raise ValueError("The term budget must fit the supplied library.")
        if not all(
            math.isfinite(value) and value >= 0
            for value in (config.complexity_penalty, config.initial_coefficient)
        ):
            raise ValueError("The penalty and starting strength must be nonnegative.")
        names = [term.make().name for term in config.terms]
        if len(set(names)) != len(names):
            raise ValueError("Force names must be unique within the library.")
        self.config = config
        self.search = config.search.make()
        self.simulator: Oscillator | None = None
        self.horizon = 0

    def fit(
        self,
        train: Windows,
        *,
        validation: Windows,
        synthetic: Windows,
        simulator: Oscillator,
    ) -> DiscoveryReport:
        """Fit on training transitions and select on full validation forecasts.

        Args:
          train: Reference histories and targets used to fit coefficients.
          validation: Separate trajectories used only to choose a combination.
          synthetic: Unused; no extra simulator trajectories are needed.
          simulator: The supplied dynamics, without the reference equations.

        Returns:
          report: Selected terms, score, and actual residual/validation calls.
            Finite-difference Jacobian calls count as training evaluations.
            Gradient-update counts stay zero because SciPy owns the fit.

        Raises:
          ValueError: The horizons differ, or the unchanged simulator cannot
            produce finite validation forecasts.

        """
        del synthetic
        self.simulator = None
        if train.label.shape[1:] != validation.label.shape[1:]:
            raise ValueError("Training and validation horizons must match.")
        cfg = self.config
        self.horizon = train.label.shape[1]
        _, scale = train.channel_stats()
        values = torch.cat((train.media, train.label), dim=1).to(cfg.dtype)
        states = values[:, :-1].reshape(-1, STATE_CHANNELS)
        targets = values[:, 1:].reshape(-1, STATE_CHANNELS)
        history = validation.media.to(cfg.dtype)
        best_simulator = simulator
        best_loss = self._validation_loss(
            simulator,
            history=history,
            validation=validation,
            scale=scale,
        )
        if not math.isfinite(best_loss):
            raise ValueError(
                "The supplied simulator must have a finite validation score.",
            )
        best_score = best_loss
        selected: list[FittedForce] = []
        tried, converged, evaluations, validations = 0, 0, 0, 1
        for size in range(1, cfg.max_terms + 1):
            for indices in combinations(range(len(cfg.terms)), size):
                objective = _ForceResiduals(
                    simulator.config,
                    terms=[cfg.terms[index] for index in indices],
                    states=states,
                    targets=targets,
                    scale=scale.reshape(STATE_CHANNELS).to(cfg.dtype),
                )
                tried += 1
                start = np.full(size, cfg.initial_coefficient, dtype=np.float64)
                # SciPy differences its starting residuals before rejecting
                # nonfinite ones, so an overflowing candidate is skipped here.
                if not np.isfinite(objective(start)).all():
                    evaluations += objective.evaluations
                    continue
                result = self.search(objective, start)
                evaluations += objective.evaluations
                coefficients = np.asarray(
                    cast("NDArray[np.float64]", result.x),
                    dtype=np.float64,
                )
                if (
                    not cast(bool, result.success)
                    or not np.isfinite(coefficients).all()
                    or coefficients.ndim != 1
                    or coefficients.size != size
                    or (coefficients < 0).any()
                ):
                    continue
                candidate = objective.simulator(coefficients)
                loss = self._validation_loss(
                    candidate,
                    history=history,
                    validation=validation,
                    scale=scale,
                )
                converged += 1
                validations += 1
                score = loss + cfg.complexity_penalty * size
                if math.isfinite(score) and score < best_score:
                    best_loss, best_score = loss, score
                    best_simulator = candidate
                    selected = [
                        {"name": term.term.name, "coefficient": term.coefficient}
                        for term in candidate.corrections[-size:]
                    ]
        self.simulator = best_simulator
        return {
            "updates": 0,
            "best_update": 0,
            "pretrain_updates": 0,
            "validation_scaled_mse": best_loss,
            "fitted_forces": selected,
            "selection_score": best_score,
            "candidates_tried": tried,
            "candidates_converged": converged,
            "training_evaluations": evaluations,
            "validation_evaluations": validations,
        }

    def predict(self, media: Tensor) -> Tensor:
        """Forecast with the selected force terms in the input dtype.

        Args:
          media: Observed histories shaped (trajectories, seq_len, channels).

        Returns:
          prediction: Forecasts for the training horizon.

        Raises:
          RuntimeError: No successful fit has completed.

        """
        if self.simulator is None:
            raise RuntimeError("Fit the simulator before predicting.")
        return self.simulator.forecast_history(media, steps=self.horizon)

    def _validation_loss(
        self,
        simulator: Oscillator,
        *,
        history: Tensor,
        validation: Windows,
        scale: Tensor,
    ) -> float:
        prediction = simulator.forecast_history(history, steps=self.horizon)
        return forecast_scores(
            prediction,
            target=validation.label,
            scale=scale,
        )["scaled_mse"]


class _ForceResiduals:
    """Compare observed transitions using the same integration steps as rollout."""

    def __init__(
        self,
        config: Oscillator.Config,
        *,
        terms: list[Makeable[ForceTerm]],
        states: Tensor,
        targets: Tensor,
        scale: Tensor,
    ) -> None:
        self.config = CorrectedOscillator.Config()
        self.config.update(config)
        self.existing_terms = len(self.config.corrections)
        self.config.corrections = [
            *self.config.corrections,
            *(WeightedForce.Config(term=term) for term in terms),
        ]
        self.model = self.config.make()
        self.states = states
        self.targets = targets
        self.scale = scale
        self.evaluations = 0

    def simulator(self, coefficients: NDArray[np.float64]) -> CorrectedOscillator:
        for term, coefficient in zip(
            self.config.corrections[self.existing_terms :],
            cast("list[float]", coefficients.tolist()),
            strict=True,
        ):
            term.coefficient = coefficient
        return self.config.make()

    def __call__(self, coefficients: NDArray[np.float64]) -> NDArray[np.float64]:
        for term, coefficient in zip(
            self.model.corrections[self.existing_terms :],
            cast("list[float]", coefficients.tolist()),
            strict=True,
        ):
            term.coefficient = coefficient
        prediction = self.model.forecast(self.states, steps=1)[:, 0]
        self.evaluations += 1
        return ((prediction - self.targets) / self.scale).flatten().numpy()
