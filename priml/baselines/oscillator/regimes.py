"""Generate paired observations before and after a change in dynamics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

import math

from configgle import Fig, Makeable
from torch import Tensor

import torch

from priml.baselines.oscillator.data import STATE_CHANNELS


if TYPE_CHECKING:
    from collections.abc import Callable


class Observation(Protocol):
    """Select the variables a forecasting method can observe."""

    channels: int

    def __call__(self, states: Tensor) -> Tensor:
        """Project states without supplying hidden parameters."""
        ...


class StateObservation:
    """Expose position and velocity as the fully observed control."""

    class Config(Fig["StateObservation"]):
        @property
        def channels(self) -> int:
            """One channel for each state variable."""
            return STATE_CHANNELS

    def __init__(self, config: Config) -> None:
        self.channels = config.channels

    def __call__(self, states: Tensor) -> Tensor:
        """Return an independent copy of both state channels."""
        return states.clone()


class PositionObservation:
    """Expose position alone; velocity stays inside the evaluator."""

    class Config(Fig["PositionObservation"]):
        @property
        def channels(self) -> int:
            """Only the position coordinate is observed."""
            return 1

    def __init__(self, config: Config) -> None:
        self.channels = config.channels

    def __call__(self, states: Tensor) -> Tensor:
        """Copy positions without retaining a view of the hidden velocity."""
        return states[..., :1].clone()


class RegimeSchedule(Protocol):
    """Choose when the future dynamics become active."""

    def __call__(self, draws: Tensor, *, seq_len: int) -> Tensor:
        """Return the first observation advanced with the new parameters."""
        ...


class StationaryRegime:
    """Use the future dynamics throughout the observed history."""

    class Config(Fig["StationaryRegime"]):
        pass

    def __init__(self, config: Config) -> None:
        del config

    def __call__(self, draws: Tensor, *, seq_len: int) -> Tensor:
        """Return zero, so the matched control has no change in its history."""
        del seq_len
        return torch.zeros_like(draws, dtype=torch.long)


class ObservedChange:
    """Change the dynamics while leaving a declared amount of recent evidence."""

    class Config(Fig["ObservedChange"]):
        recent_steps: tuple[int, int] = (24, 48)
        """Inclusive range of observed states following the change."""

    def __init__(self, config: Config) -> None:
        low, high = config.recent_steps
        if low < 5 or high < low:
            raise ValueError("A change must leave at least five observed states.")
        self.config = config

    def __call__(self, draws: Tensor, *, seq_len: int) -> Tensor:
        """Keep every change before the forecast, with an earlier regime visible."""
        low, high = self.config.recent_steps
        if high >= seq_len:
            raise ValueError("The history must also contain the earlier regime.")
        recent = low + (draws * (high - low + 1)).long()
        return seq_len - recent


class DynamicsDomain:
    """Sample independent oscillator coefficients for each episode."""

    class Config(Fig["DynamicsDomain"]):
        stiffness: tuple[float, float] = (0.5, 2.0)
        """Range of positive linear restoring coefficients."""

        damping: tuple[float, float] = (0.05, 0.3)
        """Range of nonnegative drag coefficients."""

        cubic: tuple[float, float] = (0.0, 0.6)
        """Range of nonnegative cubic restoring coefficients."""

    def __init__(self, config: Config) -> None:
        for low, high in (config.stiffness, config.damping, config.cubic):
            if not math.isfinite(low + high) or low < 0 or high < low:
                raise ValueError("Coefficient ranges must be finite and ordered.")
        if config.stiffness[0] <= 0:
            raise ValueError("The stiffness must remain positive.")
        self.config = config

    def __call__(self, draws: Tensor) -> Tensor:
        """Map three shared uniform draws to physical coefficients."""
        bounds = torch.tensor(
            [self.config.stiffness, self.config.damping, self.config.cubic],
            dtype=draws.dtype,
        )
        return bounds[:, 0] + draws * (bounds[:, 1] - bounds[:, 0])


class RegimeDynamics:
    """Advance a batch of springs whose coefficients can differ by episode."""

    class Config(Fig["RegimeDynamics"]):
        step_size: float = 0.05
        """Time between observed states."""

        substeps: int = 8
        """Semi-implicit Euler steps between observations."""

    def __init__(self, config: Config) -> None:
        if (
            not math.isfinite(config.step_size)
            or min(config.step_size, config.substeps) <= 0
        ):
            raise ValueError("The time step and integration count must be positive.")
        self.config = config

    def advance(self, state: Tensor, coefficients: Tensor) -> Tensor:
        """Advance one observation interval with the supplied coefficients."""
        position, velocity = state.unbind(dim=-1)
        stiffness, damping, cubic = coefficients.unbind(dim=-1)
        delta = self.config.step_size / self.config.substeps
        for _ in range(self.config.substeps):
            acceleration = (
                -stiffness * position - damping * velocity - cubic * position.pow(3)
            )
            velocity = velocity + delta * acceleration
            position = position + delta * velocity
        return torch.stack((position, velocity), dim=-1)

    def forecast(self, state: Tensor, *, coefficients: Tensor, steps: int) -> Tensor:
        """Roll out future states without reading any future observation."""
        if steps <= 0:
            raise ValueError("The forecast length must be positive.")
        frames: list[Tensor] = []
        for _ in range(steps):
            state = self.advance(state, coefficients)
            frames.append(state)
        return torch.stack(frames, dim=1)


@dataclass(frozen=True, kw_only=True, slots=True)
class ObservedWindows:
    """Only the observed histories and targets available for fitting."""

    media: Tensor
    label: Tensor

    def __post_init__(self) -> None:
        """Reject empty, nonfinite, or mismatched observations."""
        if (
            self.media.ndim != 3
            or self.label.ndim != 3
            or min(self.media.shape) <= 0
            or min(self.label.shape) <= 0
            or self.media.shape[0] != self.label.shape[0]
            or self.media.shape[-1] != self.label.shape[-1]
            or not self.media.is_floating_point()
            or not self.label.is_floating_point()
            or not torch.isfinite(self.media).all()
            or not torch.isfinite(self.label).all()
        ):
            raise ValueError("Expected finite paired observed sequences.")

    def channel_stats(self) -> tuple[Tensor, Tensor]:
        """Fit channel scales using only the supplied fitting windows."""
        values = torch.cat((self.media, self.label), dim=1)
        return values.mean((0, 1), keepdim=True), values.std(
            (0, 1),
            correction=0,
            keepdim=True,
        ).clamp_min(1e-6)


@dataclass(frozen=True, kw_only=True, slots=True)
class EvaluatorEpisodes:
    """Keep the hidden state and parameters out of the forecaster's input."""

    windows: ObservedWindows
    states: Tensor
    before: Tensor
    after: Tensor
    change_at: Tensor


class RegimeData:
    """Draw paired episodes with independent streams and no future regime change."""

    class Config(Fig["RegimeData"]):
        seq_len: int = 96
        """Observed states before the forecast starts."""

        pred_len: int = 64
        """Future states per episode."""

        dynamics: RegimeDynamics.Config = field(default_factory=RegimeDynamics.Config)
        """Time discretization shared by the reference and forecasting controls."""

        domain: DynamicsDomain.Config = field(default_factory=DynamicsDomain.Config)
        """Coefficient ranges used for the training pool and earlier regime."""

        observation: Makeable[Observation] = field(
            default_factory=StateObservation.Config,
        )
        """Variables exposed to every method, including its training targets."""

        schedule: Makeable[RegimeSchedule] = field(
            default_factory=StationaryRegime.Config,
        )
        """When the new coefficients become active in each observed history."""

        amplitude: tuple[float, float] = (0.3, 1.5)
        """Range of initial displacement magnitudes."""

        velocity: float = 0.5
        """Largest absolute initial velocity."""

    def __init__(self, config: Config) -> None:
        low, high = config.amplitude
        if (
            min(config.seq_len, config.pred_len) <= 0
            or config.seq_len < 7
            or not all(math.isfinite(v) for v in (low, high, config.velocity))
            or low < 0
            or high <= low
            or config.velocity < 0
        ):
            raise ValueError("Episode geometry and initial-state bounds must be valid.")
        self.config = config
        self.dynamics = config.dynamics.make()
        self.domain = config.domain.make()
        self.observation = config.observation.make()
        self.schedule = config.schedule.make()

    def sample(
        self,
        *,
        count: int,
        seed: int,
        domain: Makeable[Callable[[Tensor], Tensor]] | None = None,
    ) -> EvaluatorEpisodes:
        """Generate an independent pool without exposing its latent parameters.

        The draw layout is shared across observation and schedule ablations, so
        masking velocity cannot change the underlying motion or random stream.
        Larger counts retain the smaller pool as a prefix.

        """
        if count <= 0 or not 0 <= seed < 2**32:
            raise ValueError("Episode count and random seed must be valid.")
        cfg = self.config
        draws = torch.rand(
            count,
            10,
            generator=torch.Generator().manual_seed(seed),
            dtype=torch.float64,
        )
        before = self.domain(draws[:, :3])
        after = (self.domain if domain is None else domain.make())(draws[:, 3:6])
        low, high = cfg.amplitude
        position = (low + (high - low) * draws[:, 6]) * torch.where(
            draws[:, 7] < 0.5,
            -1,
            1,
        )
        state = torch.stack((position, (2 * draws[:, 8] - 1) * cfg.velocity), dim=-1)
        change_at = self.schedule(draws[:, 9], seq_len=cfg.seq_len)
        frames = [state]
        for index in range(1, cfg.seq_len + cfg.pred_len):
            coefficients = torch.where((index >= change_at)[:, None], after, before)
            state = self.dynamics.advance(state, coefficients)
            frames.append(state)
        states = torch.stack(frames, dim=1)
        observed = self.observation(states)
        windows = ObservedWindows(
            media=observed[:, : cfg.seq_len].clone(),
            label=observed[:, cfg.seq_len :].clone(),
        )
        return EvaluatorEpisodes(
            windows=windows,
            states=states,
            before=before,
            after=after,
            change_at=change_at,
        )
