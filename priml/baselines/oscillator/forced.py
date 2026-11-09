"""Forecast a glitchy sensor on a driven, possibly chaotic, repeatedly changing system."""

from __future__ import annotations

from dataclasses import field
from typing import TYPE_CHECKING, ClassVar, override

import math

from configgle import Fig, Makeable, Makes
from torch import Tensor

import torch

from priml.baselines.oscillator.hidden import LibraryShooting
from priml.baselines.oscillator.regimes import EvaluatorEpisodes, ObservedWindows


if TYPE_CHECKING:
    from collections.abc import Callable


COEFFICIENTS = 11
"""Observed stiffness, damping, cubic, friction, quadratic, self-excitation;
coupling; partner stiffness, damping; drive amplitude, drive frequency."""

OBSERVED_TERMS = 6
"""Leading coefficients that belong to the observed mass and may change."""

LATENT = 3
"""Unobserved state besides the observed mass: partner position, velocity, drive phase."""


class ForcedDomain:
    """Sample per-episode coefficients; signed terms permit a double well."""

    class Config(Fig["ForcedDomain"]):
        stiffness: tuple[float, float] = (-1.5, 1.0)
        """Linear restoring coefficient; negative values make a double well."""

        damping: tuple[float, float] = (0.02, 0.12)
        """Viscous drag of the observed mass; negative values pump energy."""

        cubic: tuple[float, float] = (0.3, 1.2)
        """Cubic restoring coefficient; positive so double wells stay bounded."""

        friction: tuple[float, float] = (0.0, 0.05)
        """Dry (Coulomb) friction of the observed mass."""

        quadratic: tuple[float, float] = (0.0, 0.0)
        """Asymmetric restoring term; absent in training."""

        excitation: tuple[float, float] = (0.0, 0.0)
        """Van der Pol amplitude-dependent damping; absent in training."""

        coupling: tuple[float, float] = (0.05, 0.4)
        """Spring joining the observed mass to the hidden one."""

        partner_stiffness: tuple[float, float] = (0.5, 2.0)
        """Linear restoring coefficient of the hidden mass."""

        partner_damping: tuple[float, float] = (0.01, 0.1)
        """Viscous drag of the hidden mass."""

        drive: tuple[float, float] = (0.3, 0.8)
        """Amplitude of the unobserved periodic force on the observed mass."""

        frequency: tuple[float, float] = (0.6, 1.6)
        """Angular frequency of the unobserved force."""

    def __init__(self, config: Config) -> None:
        bounds = self.bounds(config)
        if (
            not torch.isfinite(bounds).all()
            or (bounds[:, 1] < bounds[:, 0]).any()
            or (bounds[[2, 3, 5, 6, 8, 9], 0] < 0).any()
            or bounds[7, 0] <= 0
            or bounds[10, 0] <= 0
        ):
            raise ValueError("Coefficient ranges must be finite, physical and ordered.")
        self.config = config

    @staticmethod
    def bounds(config: Config) -> Tensor:
        """Stack the ranges in the dynamics' coefficient order."""
        return torch.tensor(
            [
                config.stiffness,
                config.damping,
                config.cubic,
                config.friction,
                config.quadratic,
                config.excitation,
                config.coupling,
                config.partner_stiffness,
                config.partner_damping,
                config.drive,
                config.frequency,
            ],
            dtype=torch.float64,
        )

    def __call__(self, draws: Tensor) -> Tensor:
        """Map uniform draws to coefficients."""
        bounds = self.bounds(self.config).to(draws)
        return bounds[:, 0] + draws * (bounds[:, 1] - bounds[:, 0])


class ForcedDynamics:
    """Advance a driven mass coupled to a hidden one; only the first is observed."""

    class Config(Fig["ForcedDynamics"]):
        step_size: float = 0.05
        """Time between observed states."""

        substeps: int = 8
        """Semi-implicit Euler steps between observations."""

        stick_velocity: float = 0.05
        """Velocity scale smoothing the sign in dry friction."""

    def __init__(self, config: Config) -> None:
        if (
            not math.isfinite(config.step_size + config.stick_velocity)
            or min(config.step_size, config.substeps, config.stick_velocity) <= 0
        ):
            raise ValueError("Time steps and the friction scale must be positive.")
        self.config = config

    def advance(self, state: Tensor, coefficients: Tensor) -> Tensor:
        """Advance one observation interval; differentiable in both arguments."""
        x1, v1, x2, v2, phase = state.unbind(dim=-1)
        k1, c1, a1, mu, q1, e1, kc, k2, c2, drive, omega = coefficients.unbind(dim=-1)
        delta = self.config.step_size / self.config.substeps
        for _ in range(self.config.substeps):
            spring = kc * (x2 - x1)
            acc1 = (
                -k1 * x1
                - (c1 + e1 * x1.square()) * v1
                - q1 * x1.square()
                - a1 * x1.pow(3)
                - mu * torch.tanh(v1 / self.config.stick_velocity)
                + spring
                + drive * torch.cos(phase)
            )
            acc2 = -k2 * x2 - c2 * v2 - spring
            v1 = v1 + delta * acc1
            v2 = v2 + delta * acc2
            x1 = x1 + delta * v1
            x2 = x2 + delta * v2
            phase = phase + delta * omega
        return torch.stack((x1, v1, x2, v2, phase), dim=-1)

    def forecast(self, state: Tensor, *, coefficients: Tensor, steps: int) -> Tensor:
        """Roll out future states without reading any future observation."""
        if steps <= 0:
            raise ValueError("The forecast length must be positive.")
        frames: list[Tensor] = []
        for _ in range(steps):
            state = self.advance(state, coefficients)
            frames.append(state)
        return torch.stack(frames, dim=1)


class GlitchySensor:
    """Expose the first position with noise, sparse spikes and quantization."""

    class Config(Fig["GlitchySensor"]):
        noise: tuple[float, float] = (0.02, 0.06)
        """Range of per-episode Gaussian noise standard deviations."""

        spike_rate: float = 0.02
        """Probability that any one reading is a glitch."""

        spike: tuple[float, float] = (0.5, 1.5)
        """Range of glitch magnitudes; the sign is random."""

        quantum: float = 0.01
        """Converter resolution; zero disables rounding."""

        @property
        def channels(self) -> int:
            """Only the first position is observed."""
            return 1

    def __init__(self, config: Config) -> None:
        values = (*config.noise, *config.spike, config.spike_rate, config.quantum)
        if (
            not all(math.isfinite(v) and v >= 0 for v in values)
            or config.noise[1] < config.noise[0]
            or config.spike[1] < config.spike[0]
            or config.spike_rate > 1
        ):
            raise ValueError("Sensor ranges must be finite, nonnegative and ordered.")
        self.config = config
        self.channels = config.channels

    def __call__(self, states: Tensor, *, level: Tensor, draws: Tensor) -> Tensor:
        """Corrupt a copy of the position; ``draws`` holds per-reading uniforms."""
        cfg = self.config
        sigma = cfg.noise[0] + level * (cfg.noise[1] - cfg.noise[0])
        gauss = torch.special.ndtri(draws[..., 0].clamp(1e-12, 1 - 1e-12))
        size = cfg.spike[0] + (2 * draws[..., 2] - 1).abs() * (
            cfg.spike[1] - cfg.spike[0]
        )
        glitch = torch.where(draws[..., 1] < cfg.spike_rate, size, 0)
        sign = torch.where(draws[..., 2] < 0.5, -1, 1)
        reading = states[..., 0] + sigma[:, None] * gauss + sign * glitch
        if cfg.quantum > 0:
            reading = torch.round(reading / cfg.quantum) * cfg.quantum
        return reading[..., None]


class ForcedData:
    """Draw episodes with up to several unannounced changes and latent forcing."""

    coefficients: ClassVar[int] = COEFFICIENTS
    """Coefficients per regime, in the dynamics' order."""

    extra_draws: ClassVar[int] = 0
    """Initial-state draws appended after part four's layout, keeping its stream."""

    reading_draws: ClassVar[int] = 3
    """Uniforms per reading consumed by the sensor."""

    class Config(Fig["ForcedData"]):
        seq_len: int = 256
        """Observed states before the forecast starts."""

        pred_len: int = 128
        """Future states per episode."""

        dynamics: ForcedDynamics.Config = field(default_factory=ForcedDynamics.Config)
        """Reference motion shared by the targets and evaluator oracle."""

        domain: ForcedDomain.Config = field(default_factory=ForcedDomain.Config)
        """Training ranges; also every earlier regime of the observed mass."""

        observation: GlitchySensor.Config = field(default_factory=GlitchySensor.Config)
        """Sensor exposed to every method, including its training targets."""

        changes: tuple[float, ...] = (0.15, 0.35, 0.3, 0.2)
        """Probability of zero, one, two, ... changes inside the history."""

        recent_steps: tuple[int, int] = (64, 120)
        """Inclusive range of observed states after the last change."""

        segment_steps: tuple[int, int] = (24, 64)
        """Inclusive range of states between consecutive earlier changes."""

        amplitude: tuple[float, float] = (0.3, 1.5)
        """Range of initial displacement magnitudes of the observed mass."""

        velocity: float = 0.5
        """Largest absolute initial velocity of either mass."""

        partner: float = 1.0
        """Largest absolute initial displacement of the hidden mass."""

    def __init__(self, config: Config) -> None:
        low, high = config.recent_steps
        gap_low, gap_high = config.segment_steps
        earliest = config.seq_len - high - (len(config.changes) - 2) * gap_high
        if (
            min(config.pred_len, config.velocity, config.partner) <= 0
            or not 5 <= low <= high
            or not 1 <= gap_low <= gap_high
            or len(config.changes) < 1
            or min(config.changes) < 0
            or not math.isclose(sum(config.changes), 1)
            or earliest < 1
            or config.amplitude[0] < 0
            or config.amplitude[1] <= config.amplitude[0]
        ):
            raise ValueError("Episode geometry, schedule and bounds must be valid.")
        self.config = config
        self.dynamics = config.dynamics.make()
        self.domain = config.domain.make()
        self.observation = config.observation.make()

    def sample(
        self,
        *,
        count: int,
        seed: int,
        domain: Makeable[Callable[[Tensor], Tensor]] | None = None,
    ) -> EvaluatorEpisodes:
        """Generate a pool; larger counts keep the smaller pool as a prefix.

        ``before`` and ``change_at`` describe only the last change; ``label``
        holds sensor readings, and scores use the clean position in ``states``.

        """
        if count <= 0 or not 0 <= seed < 2**32:
            raise ValueError("Episode count and random seed must be valid.")
        cfg = self.config
        regimes = len(cfg.changes)
        steps = cfg.seq_len + cfg.pred_len
        draws = torch.rand(
            count,
            (regimes + 1) * self.coefficients + 8 + regimes + self.extra_draws,
            generator=torch.Generator().manual_seed(seed),
            dtype=torch.float64,
        )
        # Offset keeps the sensor stream distinct from every 32-bit episode seed.
        readings = torch.rand(
            count,
            steps,
            self.reading_draws,
            generator=torch.Generator().manual_seed(seed + 2**32),
            dtype=torch.float64,
        )
        width = self.coefficients
        table = self.domain(
            draws[:, : regimes * width].view(count, regimes, width),
        )
        final = (self.domain if domain is None else domain.make())(
            draws[:, regimes * width : (regimes + 1) * width],
        )
        u = draws[:, (regimes + 1) * width : (regimes + 1) * width + 8 + regimes]
        total = torch.bucketize(
            u[:, 0].contiguous(),
            torch.tensor(cfg.changes, dtype=torch.float64).cumsum(0)[:-1],
            right=True,
        )
        rows = torch.arange(count)
        table[rows, total] = final
        table[:, :, OBSERVED_TERMS:] = final[:, None, OBSERVED_TERMS:]
        first, last = cfg.recent_steps
        gap_first, gap_last = cfg.segment_steps
        gaps = gap_first + (u[:, 8:] * (gap_last - gap_first + 1)).long()
        gaps[:, 0] = first + (u[:, 8] * (last - first + 1)).long()
        # times[:, j] is the j-th change counted back from the newest.
        times = cfg.seq_len - gaps[:, :-1].cumsum(1)
        later = torch.arange(regimes - 1) < total[:, None]
        times = torch.where(later, times, steps)
        state = self.initial_state(u, draws[:, (regimes + 1) * width + 8 + regimes :])
        frames = [state]
        for index in range(1, steps):
            regime = (times <= index).sum(1)
            state = self.dynamics.advance(state, table[rows, regime])
            frames.append(state)
        states = torch.stack(frames, dim=1)
        if not torch.isfinite(states).all() or states[..., 0].abs().max() > 1e3:
            raise ValueError("Sampled coefficients produced an unbounded trajectory.")
        observed = self.observation(states, level=u[:, 7], draws=readings)
        return EvaluatorEpisodes(
            windows=ObservedWindows(
                media=observed[:, : cfg.seq_len].clone(),
                label=observed[:, cfg.seq_len :].clone(),
            ),
            states=states,
            before=table[rows, (total - 1).clamp_min(0)],
            after=final,
            change_at=torch.where(total > 0, times[:, :1].sum(1), 0),
        )

    def initial_state(self, u: Tensor, extra: Tensor) -> Tensor:
        """Both masses' positions and velocities, then the drive phase."""
        del extra
        cfg = self.config
        low, high = cfg.amplitude
        return torch.stack(
            (
                (low + (high - low) * u[:, 1]) * torch.where(u[:, 2] < 0.5, -1, 1),
                (2 * u[:, 3] - 1) * cfg.velocity,
                (2 * u[:, 4] - 1) * cfg.partner,
                (2 * u[:, 5] - 1) * cfg.velocity,
                2 * math.pi * u[:, 6],
            ),
            dim=-1,
        )


class ForcedShooting(LibraryShooting):
    """Shoot the driven library with a robust loss that discounts sensor glitches."""

    dynamics_type = ForcedDynamics
    latent_draws = LATENT

    class Config(Makes["ForcedShooting"], LibraryShooting.Config):
        visible_steps: int = 64
        """Newest observations fitted; may straddle a change, which is not detected."""

        robust: float = 0.1
        """Cauchy scale in position units, so glitches barely move the fit."""

        lower: tuple[float, ...] = (
            -3.0, -0.5, 0.0, 0.0, -1.5, 0.0, 0.0, 0.05, 0.0, 0.0, 0.2,
        )  # fmt: skip
        """Lower coefficient bounds shared across every tested domain."""

        upper: tuple[float, ...] = (
            4.0, 0.6, 2.0, 0.3, 1.5, 1.5, 1.5, 4.0, 0.5, 1.0, 5.0,
        )  # fmt: skip
        """Upper coefficient bounds shared across every tested domain."""

    @staticmethod
    @override
    def latent(start: Tensor) -> Tensor:
        """Partner position and velocity, then a uniformly drawn drive phase."""
        return torch.stack(
            (2 * start[:, 0] - 1, start[:, 1] - 0.5, 2 * math.pi * start[:, 2]),
            dim=-1,
        )
