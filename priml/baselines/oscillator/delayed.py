"""Test whether forecasts use nonlinear interactions in the observed history."""

from __future__ import annotations

from dataclasses import field
from typing import TYPE_CHECKING, Protocol, cast, override

import math

from configgle import Fig, Makeable, Makes
from torch import Tensor

import torch

from priml.baselines.oscillator.benchmark import ForecastStudy, StudyRecord
from priml.baselines.oscillator.calibration import ScalarSearch
from priml.baselines.oscillator.data import (
    STATE_CHANNELS,
    ForecastData,
    InitialStates,
    Oscillator,
    Windows,
)
from priml.baselines.oscillator.search import BoundedScalarSearch
from priml.baselines.oscillator.train_step import FitReport


if TYPE_CHECKING:
    from collections.abc import Callable


class Pulse:
    """A smooth displacement pulse and its matching velocity."""

    class Config(Fig["Pulse"]):
        start: int = 0
        """First observation touched by the pulse."""

        width: int = 32
        """Observations between the two zero crossings."""

    def __init__(self, config: Config) -> None:
        if config.start < 0 or config.width < 3:
            raise ValueError(
                "Pulses need a nonnegative start and width of at least three.",
            )
        self.config = config

    def sample(self, *, steps: int, step_size: float) -> Tensor:
        """Return displacement and its derivative for a unit-height pulse.

        Args:
          steps: Observations in the supplied history.
          step_size: Time between observations.

        Returns:
          values: A (steps, 2) position/velocity history, zero outside the pulse.

        """
        phase = ((torch.arange(steps) - self.config.start) / self.config.width).clamp(
            min=0,
        )
        active = phase < 1
        position = torch.where(active, torch.sin(math.pi * phase).square(), 0.0)
        velocity = torch.where(
            active,
            math.pi / (self.config.width * step_size) * torch.sin(2 * math.pi * phase),
            0.0,
        )
        return torch.column_stack((position, velocity))


class HistorySampler(Protocol):
    """Build visible histories and expose their pulse spans to the evaluator."""

    pulses: list[Pulse.Config]

    def sample(
        self,
        states: InitialStates.Config,
        *,
        steps: int,
        step_size: float,
    ) -> Tensor:
        """Return complete sign groups shaped (count, steps, 2)."""
        ...


class PairedHistory:
    """Keep all four sign combinations of each independently sampled pulse pair."""

    class Config(Fig["PairedHistory"]):
        first: Pulse.Config = field(default_factory=Pulse.Config)
        """Displacement pulse whose height uses the split's amplitude range."""

        second: Pulse.Config = field(default_factory=lambda: Pulse.Config(start=32))
        """Pulse whose peak speed uses the split's velocity bound."""

    def __init__(self, config: Config) -> None:
        if config.first.start + config.first.width > config.second.start:
            raise ValueError("The two history pulses must not overlap.")
        self.config = config
        self.pulses = [config.first, config.second]
        self.first = config.first.make()
        self.second = config.second.make()

    def sample(
        self,
        states: InitialStates.Config,
        *,
        steps: int,
        step_size: float,
    ) -> Tensor:
        """Build balanced histories without changing the split or global RNG.

        Args:
          states: Independent split seed, total count, height and speed bounds.
          steps: History length; it must include a quiet tail after both pulses.
          step_size: Time between observations.

        Returns:
          media: Histories shaped (count, steps, 2). Each consecutive group of
            four shares magnitudes and covers ++, +-, -+, and -- signs.

        Raises:
          ValueError: The count cannot form complete quartets or the tail is absent.

        """
        if states.count % 4 or states.count <= 0:
            raise ValueError("Each split must contain complete four-history groups.")
        if steps <= self.config.second.start + self.config.second.width:
            raise ValueError("Histories need a quiet tail after both pulses.")
        source = states.copy_tree()
        source.count //= 4
        magnitudes = source.make().sample().abs()
        first = self.first.sample(steps=steps, step_size=step_size)
        second = self.second.sample(steps=steps, step_size=step_size)
        second = second / second[:, 1].abs().max()
        signs = torch.tensor([[1, 1], [1, -1], [-1, 1], [-1, -1]])
        return (
            magnitudes[:, None, 0, None, None] * signs[None, :, 0, None, None] * first
            + magnitudes[:, None, 1, None, None]
            * signs[None, :, 1, None, None]
            * second
        ).reshape(states.count, steps, STATE_CHANNELS)


class SignedHistory:
    """Vary interaction order while retaining every sign combination."""

    class Config(Fig["SignedHistory"]):
        pulses: list[Pulse.Config] = field(
            default_factory=lambda: [
                Pulse.Config(start=0),
                Pulse.Config(start=32),
                Pulse.Config(start=64),
                Pulse.Config(start=96),
            ],
        )
        """Nonoverlapping pulses; each adds one independent sign and magnitude."""

    def __init__(self, config: Config) -> None:
        if len(config.pulses) < 2:
            raise ValueError("The interaction needs at least two pulses.")
        if any(
            left.start + left.width > right.start
            for left, right in zip(config.pulses[:-1], config.pulses[1:], strict=True)
        ):
            raise ValueError("History pulses must be ordered and nonoverlapping.")
        self.config = config
        self.pulses = config.pulses
        self.samplers = [pulse.make() for pulse in config.pulses]

    def sample(
        self,
        states: InitialStates.Config,
        *,
        steps: int,
        step_size: float,
    ) -> Tensor:
        """Draw magnitudes once per group, then enumerate every sign assignment.

        Args:
          states: Split seed and total trajectory count. The first pulse uses
            displacement bounds; later pulses use the peak-speed bound.
          steps: History length, including a quiet tail.
          step_size: Time between observations.

        Returns:
          media: Histories with 2**len(pulses) consecutive rows per independent
            magnitude tuple. No sign group is split across data partitions.

        Raises:
          ValueError: Groups are incomplete or the quiet tail is absent.

        """
        order = len(self.pulses)
        group_size = 1 << order
        # The magnitudes reuse the split's bounds and seed; reject them as it would.
        states.make()
        if states.count % group_size or states.count <= 0:
            raise ValueError("The split needs complete interaction sign groups.")
        if steps <= self.pulses[-1].start + self.pulses[-1].width:
            raise ValueError("Histories need a quiet tail after every pulse.")
        generator = torch.Generator().manual_seed(states.seed)
        draws = torch.rand(states.count // group_size, order, generator=generator)
        low, high = states.amplitude
        magnitudes = draws * states.velocity
        magnitudes[:, 0] = low + (high - low) * draws[:, 0]
        indices = torch.arange(group_size)[:, None]
        bits = torch.arange(order - 1, -1, -1)[None]
        signs = 1 - 2 * ((indices >> bits) & 1)
        media = torch.zeros(len(draws), group_size, steps, STATE_CHANNELS)
        for index, sampler in enumerate(self.samplers):
            pulse = sampler.sample(steps=steps, step_size=step_size)
            if index:
                pulse = pulse / pulse[:, 1].abs().max()
            media += (
                magnitudes[:, index, None, None, None]
                * signs[None, :, index, None, None]
                * pulse
            )
        return media.reshape(states.count, steps, STATE_CHANNELS)


class DelayedOscillator(Oscillator):
    """An oscillator driven by a saturated product of past displacements."""

    class Config(Makes["DelayedOscillator"], Oscillator.Config):
        lags: list[int] = field(default_factory=lambda: [80, 48])
        """Observation lag for each factor in the delayed interaction."""

        gain: float = 2.0
        """Strength of the delayed force."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        if len(config.lags) < 2 or min(config.lags) < 1:
            raise ValueError("Feedback needs at least two positive lags.")
        if not math.isfinite(config.gain) or config.gain < 0:
            raise ValueError("Feedback gain must be finite and nonnegative.")
        self.delay = config

    @override
    def forecast(
        self,
        state: Tensor,
        *,
        steps: int,
        substeps: int | None = None,
    ) -> Tensor:
        """Forecast with constant prehistory when only a state is supplied.

        Args:
          state: Position/velocity pairs shaped (trajectories, 2).
          steps: Future observations to produce.
          substeps: Integration steps per observation; None uses the config.

        Returns:
          prediction: Forecasts assuming the supplied state filled every delay.

        """
        if state.ndim != 2 or state.shape[-1] != STATE_CHANNELS:
            raise ValueError("Expected position/velocity pairs.")
        length = max(self.delay.lags)
        return self.forecast_history(
            state[:, None].expand(-1, length, -1),
            steps=steps,
            substeps=substeps,
        )

    @override
    def forecast_history(
        self,
        media: Tensor,
        *,
        steps: int,
        substeps: int | None = None,
    ) -> Tensor:
        """Roll out using observed history, then previously predicted states.

        Args:
          media: Floating histories shaped (trajectories, history, 2).
          steps: Future observations to produce.
          substeps: Integration steps per observation; None uses the config.

        Returns:
          prediction: States shaped (trajectories, steps, 2). The delayed force
            is held fixed within each observation interval; nonfinite
            histories propagate into the forecast.

        Raises:
          ValueError: Shapes, horizon, dtype, available history, or substeps are invalid.

        """
        if (
            media.ndim != 3
            or media.shape[-1] != STATE_CHANNELS
            or media.shape[1] < max(self.delay.lags)
            or media.numel() == 0
            or not media.is_floating_point()
            or steps <= 0
        ):
            raise ValueError("Expected floating histories covering all feedback lags.")
        substeps = self.config.substeps if substeps is None else substeps
        if substeps <= 0:
            raise ValueError("Integration substeps must be positive.")
        history = list(media.unbind(dim=1))
        position, velocity = history[-1].unbind(dim=-1)
        delta = self.config.step_size / substeps
        frames: list[Tensor] = []
        for _ in range(steps):
            product = torch.ones_like(position)
            for lag in self.delay.lags:
                product = product * history[-lag][:, 0]
            force = self.delay.gain * product.tanh()
            for _ in range(substeps):
                acceleration = self.acceleration(position, velocity=velocity) + force
                velocity = velocity + delta * acceleration
                position = position + delta * velocity
            state = torch.stack((position, velocity), dim=-1)
            history.append(state)
            frames.append(state)
        return torch.stack(frames, dim=1)


class DelayedData(ForecastData):
    """Use prescribed histories as boundary conditions for a delayed response."""

    class Config(Makes["DelayedData"], ForecastData.Config):
        seq_len: int = 96
        """Observed history, including the pulses and a quiet tail."""

        pred_len: int = 48
        """Future observations before feedback needs newly predicted inputs."""

        reference: Oscillator.Config = field(
            default_factory=cast(
                "Callable[[], Oscillator.Config]",
                DelayedOscillator.Config,
            ),
        )
        """Reference system with delayed feedback."""

        history: Makeable[HistorySampler] = field(default_factory=PairedHistory.Config)
        """Visible pulses; no hidden signs or future input is needed."""

        train: InitialStates.Config = field(
            default_factory=lambda: InitialStates.Config(velocity=2.0),
        )
        """Training histories, keeping every sign combination together."""

        validation: InitialStates.Config = field(
            default_factory=lambda: InitialStates.Config(
                seed=202,
                count=16,
                velocity=2.0,
            ),
        )
        """Independent magnitude combinations for model selection."""

        synthetic: InitialStates.Config = field(
            default_factory=lambda: InitialStates.Config(
                seed=303,
                count=256,
                velocity=2.0,
            ),
        )
        """Cheap responses with their own independent histories."""

        test: InitialStates.Config = field(
            default_factory=lambda: InitialStates.Config(
                seed=404,
                count=64,
                velocity=2.0,
            ),
        )
        """Unseen magnitude combinations in the training range."""

        shifted: InitialStates.Config = field(
            default_factory=lambda: InitialStates.Config(
                seed=505,
                count=64,
                amplitude=(1.5, 2.5),
                velocity=2.0,
            ),
        )
        """Unseen combinations with larger displacement pulses."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.history = config.history.make()
        if isinstance(self.reference, DelayedOscillator):
            delay = self.reference.delay
            if config.pred_len > min(delay.lags):
                raise ValueError(
                    "The delayed-response horizon must stay within all feedback lags.",
                )
            spans = self.history.pulses
            if len(spans) != len(delay.lags):
                raise ValueError("Each history pulse needs one feedback lag.")
            overlaps = 0
            for step in range(config.pred_len):
                active = [
                    [
                        index
                        for index, pulse in enumerate(spans)
                        if pulse.start < time < pulse.start + pulse.width
                    ]
                    for time in (config.seq_len - lag + step for lag in delay.lags)
                ]
                if all(active):
                    if len({indices[0] for indices in active}) != len(spans):
                        raise ValueError(
                            "Feedback must combine different history pulses.",
                        )
                    overlaps += 1
            if not overlaps:
                raise ValueError(
                    "Feedback lags must expose a visible pulse interaction.",
                )

    @override
    def windows(
        self,
        states: InitialStates.Config,
        *,
        simulator: Oscillator,
    ) -> Windows:
        """Keep every complete sign group in its own independently seeded split.

        Args:
          states: Seed and pulse bounds for this split.
          simulator: Dynamics that generate only the future response.

        Returns:
          windows: Prescribed histories and their deterministic future response.

        """
        media = self.history.sample(
            states,
            steps=self.config.seq_len,
            step_size=self.config.cheap.step_size,
        )
        return Windows(
            media=media,
            label=simulator.forecast_history(media, steps=self.config.pred_len),
        )


class DelayedRecord(StudyRecord):
    """A response forecast with skill relative to the zero-response control."""

    forecast_skill: dict[str, float | None]


class DelayedStudy(ForecastStudy):
    """Report response skill without turning small absolute errors into accuracy."""

    class Config(Makes["DelayedStudy"], ForecastStudy.Config):
        data: ForecastData.Config = field(
            default_factory=cast(
                "Callable[[], ForecastData.Config]",
                DelayedData.Config,
            ),
        )
        """Delayed-response task and independent history splits."""

    @override
    def run(self) -> DelayedRecord:
        """Fit and score, then compare MSE with the cheap zero-response forecast.

        Returns:
          record: Original errors plus 1 - method MSE / control MSE. One is
            exact, zero matches the control, and negative is worse. Matched
            cases with zero control error report None because skill is undefined.

        """
        record = super().run()
        return {  # pyrefly: ignore[open-unpacking] -- The spread TypedDict is open; none of its extra keys collide here.
            **record,
            "forecast_skill": {
                split: 1 - scores["method"]["mse"] / scores["cheap_simulator"]["mse"]
                if scores["cheap_simulator"]["mse"] > 0
                else None
                for split, scores in record["scores"].items()
            },
        }


class DelayFitReport(FitReport):
    """Gain recovered by a method given the correct delayed equation."""

    fitted_gain: float
    training_scaled_mse: float
    simulator_evaluations: int


class DelayedCalibration:
    """Check learnability with the correct delay equation and one fitted gain."""

    class Config(Fig["DelayedCalibration"]):
        simulator: DelayedOscillator.Config = field(
            default_factory=lambda: DelayedOscillator.Config(gain=0.0),
        )
        """Privileged equation form and lags; its gain is fitted, not assumed."""

        search: Makeable[ScalarSearch] = field(
            default_factory=lambda: BoundedScalarSearch.Config(bounds=(0.0, 3.0)),
        )
        """Bounded search on reference training targets only."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.search = config.search.make()
        self.simulator: DelayedOscillator | None = None
        self.horizon = 0

    def fit(
        self,
        train: Windows,
        *,
        validation: Windows,
        synthetic: Windows,
        simulator: Oscillator,
    ) -> DelayFitReport:
        """Fit the gain on full training responses, keeping test data out.

        Args:
          train: Observed histories and reference training responses.
          validation: Unused; the known equation has one fitted coefficient.
          synthetic: Unused.
          simulator: Supplies the common linear dynamics and observation grid.

        Returns:
          report: Fitted gain, training error, and actual simulator evaluations.

        Raises:
          RuntimeError: No finite converged gain is found.

        """
        del validation, synthetic
        self.simulator = None
        cfg = self.config.simulator.copy_tree()
        cfg.step_size = simulator.config.step_size
        cfg.substeps = simulator.config.substeps
        cfg.stiffness = simulator.config.stiffness
        cfg.damping = simulator.config.damping
        cfg.cubic = simulator.config.cubic
        objective = _ResponseLoss(cfg, train=train)
        objective(0.0)
        result = self.search(objective)
        if not result.success or objective.best_config is None:
            raise RuntimeError(
                "Delayed calibration did not find a finite converged fit.",
            )
        self.simulator = objective.best_config.make()
        self.horizon = train.label.shape[1]
        return {
            "updates": 0,
            "best_update": 0,
            "pretrain_updates": 0,
            "fitted_gain": self.simulator.delay.gain,
            "training_scaled_mse": objective.best_loss,
            "simulator_evaluations": objective.evaluations,
        }

    def predict(self, media: Tensor) -> Tensor:
        """Forecast with the fitted gain and the supplied observed history."""
        if self.simulator is None:
            raise RuntimeError("Fit the delayed equation before predicting.")
        return self.simulator.forecast_history(media, steps=self.horizon)


class _ResponseLoss:
    def __init__(self, config: DelayedOscillator.Config, *, train: Windows) -> None:
        self.config = config
        self.media = train.media.double()
        self.target = train.label.double()
        _, scale = train.channel_stats()
        self.scale = scale.double()
        self.best_config: DelayedOscillator.Config | None = None
        self.best_loss = math.inf
        self.evaluations = 0

    def __call__(self, gain: float) -> float:
        cfg = self.config.copy_tree()
        cfg.gain = float(gain)
        prediction = cfg.make().forecast_history(self.media, steps=self.target.shape[1])
        loss = float(((prediction - self.target) / self.scale).square().mean())
        self.evaluations += 1
        if loss < self.best_loss:
            self.best_loss, self.best_config = loss, cfg
        return loss if math.isfinite(loss) else math.inf
