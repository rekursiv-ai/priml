"""Generate separate motion sequences for training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import math

from configgle import Fig
from torch import Tensor

import torch


STATE_CHANNELS: Final = 2
"""Each observation is one position/velocity pair."""


class Oscillator:
    """Damped motion with an optional cubic restoring force."""

    class Config(Fig["Oscillator"]):
        step_size: float = 0.1
        """Time between observations."""

        substeps: int = 4
        """Integration steps between observations."""

        stiffness: float = 1.0
        """Linear restoring force."""

        damping: float = 0.15
        """Resistance to motion."""

        cubic: float = 0.0
        """Extra restoring force at larger displacements."""

    def __init__(self, config: Config) -> None:
        values = (config.step_size, config.stiffness, config.damping, config.cubic)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Simulator settings must be finite.")
        if min(config.step_size, config.stiffness, config.substeps) <= 0:
            raise ValueError("Time step, stiffness, and substeps must be positive.")
        if min(config.damping, config.cubic) < 0:
            raise ValueError("Damping and cubic force must be nonnegative.")
        self.config = config

    def forecast(self, state: Tensor, *, steps: int) -> Tensor:
        """Advance position and velocity, excluding the supplied initial state.

        Args:
          state: Floating-point position/velocity pairs shaped (trajectories, 2).
          steps: Observations to produce after ``state``.

        Returns:
          values: States shaped (trajectories, steps, 2).

        Raises:
          ValueError: ``state`` is not floating-point position/velocity pairs, or
            ``steps`` is not positive.

        """
        if state.ndim != 2 or state.shape[-1] != STATE_CHANNELS or steps <= 0:
            raise ValueError("Expected position/velocity pairs and positive steps.")
        if not state.is_floating_point():
            raise ValueError("Initial states must be floating-point values.")
        position, velocity = state.unbind(dim=-1)
        delta = self.config.step_size / self.config.substeps
        frames: list[Tensor] = []
        for _ in range(steps):
            for _ in range(self.config.substeps):
                acceleration = (
                    -self.config.stiffness * position
                    - self.config.damping * velocity
                    - self.config.cubic * position.pow(3)
                )
                velocity = velocity + delta * acceleration
                position = position + delta * velocity
            frames.append(torch.stack((position, velocity), dim=-1))
        return torch.stack(frames, dim=1)


class InitialStates:
    """Draw initial conditions with an independent random stream."""

    class Config(Fig["InitialStates"]):
        seed: int = 101
        """Seed used only for these initial conditions."""

        count: int = 32
        """Number of independent trajectories."""

        amplitude: tuple[float, float] = (0.2, 1.5)
        """Range of initial displacement magnitudes."""

        velocity: float = 0.5
        """Maximum absolute initial velocity."""

    def __init__(self, config: Config) -> None:
        low, high = config.amplitude
        if not all(math.isfinite(value) for value in (low, high, config.velocity)):
            raise ValueError("Initial-condition bounds must be finite.")
        if config.count <= 0 or low < 0 or low >= high or config.velocity < 0:
            raise ValueError("Invalid trajectory count or initial-condition bounds.")
        self.config = config

    def sample(self) -> Tensor:
        """Return position/velocity pairs without changing global random state.

        Returns:
          states: Initial conditions shaped (count, 2).

        """
        generator = torch.Generator().manual_seed(self.config.seed)
        draws = torch.rand(self.config.count, 3, generator=generator)
        low, high = self.config.amplitude
        position = (low + (high - low) * draws[:, 0]) * (
            2 * (draws[:, 1] >= 0.5).float() - 1
        )
        velocity = (2 * draws[:, 2] - 1) * self.config.velocity
        return torch.stack((position, velocity), dim=-1)


@dataclass(frozen=True, kw_only=True, slots=True)
class Windows:
    """One history and forecast target per independent trajectory."""

    media: Tensor
    label: Tensor

    def channel_stats(self) -> tuple[Tensor, Tensor]:
        """Return per-channel statistics over every history and target cell.

        Returns:
          mean: Channel means shaped (1, 1, channels).
          scale: Population standard deviations floored at ``1e-6``, same shape.

        """
        values = torch.cat((self.media, self.label), dim=1)
        return (
            values.mean(dim=(0, 1), keepdim=True),
            values.std(dim=(0, 1), correction=0, keepdim=True).clamp_min(1e-6),
        )


class ForecastData:
    """Keep reference training data separate from the evaluator's test data."""

    class Config(Fig["ForecastData"]):
        seq_len: int = 24
        """Observed timesteps per trajectory."""

        pred_len: int = 12
        """Forecast timesteps per trajectory."""

        cheap: Oscillator.Config = field(default_factory=Oscillator.Config)
        """Simulator available to the forecasting method."""

        reference: Oscillator.Config = field(
            default_factory=lambda: Oscillator.Config(cubic=0.4),
        )
        """Evaluator's reference dynamics."""

        train: InitialStates.Config = field(default_factory=InitialStates.Config)
        """Reference trajectories available for fitting."""

        validation: InitialStates.Config = field(
            default_factory=lambda: InitialStates.Config(seed=202, count=16),
        )
        """Reference trajectories used only for model selection."""

        synthetic: InitialStates.Config = field(
            default_factory=lambda: InitialStates.Config(seed=303, count=256),
        )
        """Cheap trajectories available for pretraining."""

        test: InitialStates.Config = field(
            default_factory=lambda: InitialStates.Config(seed=404, count=64),
        )
        """Unseen trajectories drawn from the training range."""

        shifted: InitialStates.Config = field(
            default_factory=lambda: InitialStates.Config(
                seed=505,
                count=64,
                amplitude=(1.5, 2.5),
            ),
        )
        """Unseen trajectories with larger initial displacements."""

    def __init__(self, config: Config) -> None:
        if min(config.seq_len, config.pred_len) <= 0:
            raise ValueError("History and horizon must be positive.")
        streams = (
            config.train,
            config.validation,
            config.synthetic,
            config.test,
            config.shifted,
        )
        if len({stream.seed for stream in streams}) != len(streams):
            raise ValueError("Each data split needs a separate seed.")
        if (config.cheap.step_size, config.cheap.substeps) != (
            config.reference.step_size,
            config.reference.substeps,
        ):
            raise ValueError("Both simulators must use the same observation grid.")
        self.config = config
        self.cheap = config.cheap.make()
        self.reference = config.reference.make()

    def windows(
        self,
        states: InitialStates.Config,
        *,
        simulator: Oscillator,
    ) -> Windows:
        """Build complete trajectories before splitting histories from targets.

        Args:
          states: Initial conditions for this split.
          simulator: Dynamics that generate the trajectories.

        Returns:
          windows: Histories of ``seq_len`` steps and targets of ``pred_len``.

        """
        initial = states.make().sample()
        values = torch.cat(
            (
                initial[:, None, :],
                simulator.forecast(
                    initial,
                    steps=self.config.seq_len + self.config.pred_len - 1,
                ),
            ),
            dim=1,
        )
        return Windows(
            media=values[:, : self.config.seq_len],
            label=values[:, self.config.seq_len :],
        )
