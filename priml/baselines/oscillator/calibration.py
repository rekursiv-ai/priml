"""Fit the simulator's cubic force using observed training transitions."""

from __future__ import annotations

from dataclasses import field
from typing import TYPE_CHECKING, Protocol, cast

import math

from configgle import Fig, Makeable, PartialConfig
from scipy.optimize import OptimizeResult, minimize_scalar
from torch import Tensor

import torch

from priml.baselines.oscillator.data import STATE_CHANNELS, Oscillator
from priml.baselines.oscillator.train_step import FitReport


if TYPE_CHECKING:
    from collections.abc import Callable

    from priml.baselines.oscillator.data import Windows


class ScalarSearch(Protocol):
    """Search one simulator parameter using a supplied training loss."""

    def __call__(self, fun: Callable[[float], float], /) -> OptimizeResult:
        """Return the search result for the supplied loss."""
        ...


class CalibrationReport(FitReport):
    """The fitted force and the work used to choose it."""

    calibrated_cubic: float
    training_scaled_mse: float
    simulator_evaluations: int


class CalibratedSimulator:
    """Fit one known force term, then forecast with the fitted simulator."""

    class Config(Fig["CalibratedSimulator"]):
        search: Makeable[ScalarSearch] = field(
            default_factory=lambda: PartialConfig(
                minimize_scalar,
                bounds=(0.0, 1.0),
                method="bounded",
                options={"maxiter": 64, "xatol": 1e-6},
            ),
        )
        """Search bounds and stopping rules for the cubic force."""

        dtype: torch.dtype = torch.float64
        """Precision used to compare training transitions."""

    def __init__(self, config: Config) -> None:
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
    ) -> CalibrationReport:
        """Estimate the cubic force from consecutive training observations.

        Args:
          train: Reference histories and targets; both are training data.
          validation: Unused; the search uses training error only.
          synthetic: Unused; no extra simulator trajectories are needed.
          simulator: Known dynamics, with an unknown cubic coefficient.

        Returns:
          report: The fitted coefficient, one-step loss, and simulator calls.
            Each call scores all training transitions. There are no gradient
            updates, so the shared update counts are zero.

        Raises:
          ValueError: Training windows are empty, malformed, or nonfinite.
          RuntimeError: The search does not converge to a finite fit.

        """
        del validation, synthetic
        for values in (train.media, train.label):
            if (
                values.ndim != 3
                or values.shape[-1] != STATE_CHANNELS
                or values.numel() == 0
                or not torch.isfinite(values).all()
            ):
                raise ValueError("Expected nonempty finite position/velocity windows.")
        if train.media.shape[0] != train.label.shape[0]:
            raise ValueError("Training histories and targets must pair by trajectory.")
        self.simulator = None
        values = torch.cat((train.media, train.label), dim=1).to(self.config.dtype)
        _, scale = train.channel_stats()
        objective = _TransitionLoss(
            simulator.config,
            states=values[:, :-1].reshape(-1, STATE_CHANNELS),
            targets=values[:, 1:].reshape(-1, STATE_CHANNELS),
            scale=scale.reshape(STATE_CHANNELS).to(self.config.dtype),
        )
        # Keep the original simulator if fitting cannot improve its training loss.
        objective(simulator.config.cubic)
        result = self.search(objective)
        if not cast(bool, result.success) or objective.best_config is None:
            raise RuntimeError("Calibration did not find a finite converged fit.")
        self.simulator = objective.best_config.make()
        self.horizon = train.label.shape[1]
        return {
            "updates": 0,
            "best_update": 0,
            "pretrain_updates": 0,
            "calibrated_cubic": self.simulator.config.cubic,
            "training_scaled_mse": objective.best_loss,
            "simulator_evaluations": objective.evaluations,
        }

    def predict(self, media: Tensor) -> Tensor:
        """Roll out from the last observed state in the input dtype.

        Args:
          media: Histories shaped (trajectories, seq_len, channels).

        Returns:
          prediction: Fitted-simulator forecasts for the training horizon.

        Raises:
          RuntimeError: No successful fit has been completed.

        """
        if self.simulator is None:
            raise RuntimeError("Fit the simulator before predicting.")
        return self.simulator.forecast(media[:, -1], steps=self.horizon)


class _TransitionLoss:
    """Score one observation step without joining separate trajectories."""

    def __init__(
        self,
        config: Oscillator.Config,
        *,
        states: Tensor,
        targets: Tensor,
        scale: Tensor,
    ) -> None:
        self.config = config
        self.states = states
        self.targets = targets
        self.scale = scale
        self.best_config: Oscillator.Config | None = None
        self.best_loss = math.inf
        self.evaluations = 0

    def __call__(self, cubic: float) -> float:
        cfg = self.config.copy_tree()
        cfg.cubic = float(cubic)
        prediction = cfg.make().forecast(self.states, steps=1)[:, 0]
        loss = float(((prediction - self.targets) / self.scale).square().mean())
        self.evaluations += 1
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_config = cfg
        return loss if math.isfinite(loss) else math.inf
