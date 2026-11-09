"""Forecast one noisy coordinate of a system with an unobserved coupled partner."""

from __future__ import annotations

from dataclasses import field
from typing import TYPE_CHECKING, ClassVar, Final, Protocol

import math

from configgle import Fig, Makeable
from torch import Tensor

import torch

from priml.baselines.oscillator.regimes import (
    EvaluatorEpisodes,
    ObservedWindows,
    RegimeDynamics,
)


if TYPE_CHECKING:
    from collections.abc import Callable


COEFFICIENTS: Final = 7
"""Observed stiffness, damping, cubic, dry friction; coupling; partner stiffness, damping."""

OBSERVED_TERMS: Final = 4
"""Leading coefficients that belong to the observed mass and may change."""


class HiddenDomain:
    """Sample per-episode coefficients for the visible mass and its hidden partner."""

    class Config(Fig["HiddenDomain"]):
        stiffness: tuple[float, float] = (0.5, 2.0)
        """Linear restoring coefficient of the observed mass."""

        damping: tuple[float, float] = (0.02, 0.15)
        """Viscous drag of the observed mass."""

        cubic: tuple[float, float] = (0.0, 0.6)
        """Cubic restoring coefficient of the observed mass."""

        friction: tuple[float, float] = (0.0, 0.02)
        """Dry (Coulomb) friction of the observed mass; nearly absent in training."""

        coupling: tuple[float, float] = (0.05, 0.3)
        """Spring joining the observed mass to the hidden one."""

        partner_stiffness: tuple[float, float] = (0.5, 2.0)
        """Linear restoring coefficient of the hidden mass."""

        partner_damping: tuple[float, float] = (0.01, 0.1)
        """Viscous drag of the hidden mass."""

    def __init__(self, config: Config) -> None:
        bounds = self.bounds(config)
        if (  # house-ignore[tensor-value-guard] -- Validate CPU coefficient bounds at construction.
            not torch.isfinite(bounds).all()
            or (bounds < 0).any()
            or (bounds[:, 1] < bounds[:, 0]).any()
            or bounds[0, 0] <= 0
            or bounds[5, 0] <= 0
        ):
            raise ValueError("Coefficient ranges must be finite, physical and ordered.")
        self.config = config

    @classmethod
    def bounds(cls, config: Config) -> Tensor:
        """Stack the ranges in the dynamics' coefficient order.

        Args:
          config: Coefficient ranges in the domain configuration.

        Returns:
          bounds: Double-precision tensor with one lower and upper bound per coefficient.

        """
        return torch.tensor(
            [
                config.stiffness,
                config.damping,
                config.cubic,
                config.friction,
                config.coupling,
                config.partner_stiffness,
                config.partner_damping,
            ],
            dtype=torch.float64,
        )

    def __call__(self, draws: Tensor) -> Tensor:
        """Map uniform draws to coefficients."""
        bounds = self.bounds(self.config).to(draws)
        return bounds[:, 0] + draws * (bounds[:, 1] - bounds[:, 0])


class HiddenDynamics:
    """Advance two coupled masses; only the first is ever observed."""

    class Config(Fig["HiddenDynamics"]):
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
        """Advance one observation interval; differentiable in both arguments.

        Args:
          state: Current state for each episode.
          coefficients: Dynamics coefficients for each episode.

        Returns:
          state: State after one observation interval.

        """
        x1, v1, x2, v2 = state.unbind(dim=-1)
        k1, c1, a1, mu, kc, k2, c2 = coefficients.unbind(dim=-1)
        delta = self.config.step_size / self.config.substeps
        for _ in range(self.config.substeps):
            spring = kc * (x2 - x1)
            acc1 = (
                -k1 * x1
                - c1 * v1
                - a1 * x1.pow(3)
                - mu * torch.tanh(v1 / self.config.stick_velocity)
                + spring
            )
            acc2 = -k2 * x2 - c2 * v2 - spring
            v1 = v1 + delta * acc1
            v2 = v2 + delta * acc2
            x1 = x1 + delta * v1
            x2 = x2 + delta * v2
        return torch.stack((x1, v1, x2, v2), dim=-1)

    def forecast(self, state: Tensor, *, coefficients: Tensor, steps: int) -> Tensor:
        """Roll out future states without reading any future observation.

        Args:
          state: Initial state for each episode.
          coefficients: Fixed dynamics coefficients for each episode.
          steps: Number of future observation intervals.

        Returns:
          trajectory: Future states, excluding the initial state.

        """
        if steps <= 0:
            raise ValueError("The forecast length must be positive.")
        frames: list[Tensor] = []
        for _ in range(steps):
            state = self.advance(state, coefficients)
            frames.append(state)
        return torch.stack(frames, dim=1)


class NoisyPosition:
    """Expose the first mass's position with per-episode sensor noise."""

    class Config(Fig["NoisyPosition"]):
        noise: tuple[float, float] = (0.01, 0.04)
        """Range of per-episode noise standard deviations, in position units."""

        @property
        def channels(self) -> int:
            """Only the first position is observed."""
            return 1

    def __init__(self, config: Config) -> None:
        low, high = config.noise
        if not math.isfinite(low + high) or low < 0 or high < low:
            raise ValueError("The noise range must be finite and ordered.")
        self.config = config
        self.channels = config.channels

    def __call__(self, states: Tensor, *, level: Tensor, noise: Tensor) -> Tensor:
        """Add scaled noise to a copy of the observed position."""
        low, high = self.config.noise
        sigma = low + level * (high - low)
        return states[..., :1] + (sigma[:, None] * noise)[..., None]


class HiddenData:
    """Draw episodes whose change time, partner state and noise are all latent."""

    class Config(Fig["HiddenData"]):
        seq_len: int = 160
        """Observed states before the forecast starts."""

        pred_len: int = 64
        """Future states per episode."""

        dynamics: HiddenDynamics.Config = field(default_factory=HiddenDynamics.Config)
        """Reference motion shared by the targets and evaluator oracle."""

        domain: HiddenDomain.Config = field(default_factory=HiddenDomain.Config)
        """Training ranges; also the observed mass's regime before any change."""

        observation: NoisyPosition.Config = field(default_factory=NoisyPosition.Config)
        """Sensor exposed to every method, including its training targets."""

        stationary: float = 0.25
        """Probability that an episode has no change at all."""

        recent_steps: tuple[int, int] = (48, 112)
        """Inclusive range of observed states after a change, when there is one."""

        amplitude: tuple[float, float] = (0.3, 1.5)
        """Range of initial displacement magnitudes of the observed mass."""

        velocity: float = 0.5
        """Largest absolute initial velocity of either mass."""

        partner: float = 1.0
        """Largest absolute initial displacement of the hidden mass."""

    def __init__(self, config: Config) -> None:
        low, high = config.recent_steps
        if (
            min(config.pred_len, config.velocity, config.partner) <= 0
            or low < 5
            or low > high
            or high >= config.seq_len
            or math.isnan(config.stationary)
            or config.stationary < 0
            or config.stationary > 1
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

        ``label`` holds noisy observations for fitting. Scores use the clean
        position in ``states``, so the noise is never part of the target.

        Args:
          count: Number of episodes in the pool.
          seed: Independent random stream for this pool.
          domain: Optional coefficient domain for the final regime.

        Returns:
          episodes: Episodes with observed windows and evaluator-only state.

        """
        if count <= 0 or seed < 0 or seed >= 2**32:
            raise ValueError("Episode count and random seed must be valid.")
        cfg = self.config
        steps = cfg.seq_len + cfg.pred_len
        draws = torch.rand(
            count,
            2 * COEFFICIENTS + 8,
            generator=torch.Generator().manual_seed(seed),
            dtype=torch.float64,
        )
        # Offset keeps the noise stream distinct from every 32-bit episode seed;
        # transforming uniforms keeps rows aligned so pools stay nested.
        noise = torch.special.ndtri(
            torch.rand(
                count,
                steps,
                generator=torch.Generator().manual_seed(seed + 2**32),
                dtype=torch.float64,
            ),
        )
        before = self.domain(draws[:, :COEFFICIENTS])
        after = (self.domain if domain is None else domain.make())(
            draws[:, COEFFICIENTS : 2 * COEFFICIENTS],
        )
        before[:, OBSERVED_TERMS:] = after[:, OBSERVED_TERMS:]
        u = draws[:, 2 * COEFFICIENTS :]
        low, high = cfg.amplitude
        state = torch.stack(
            (
                (low + (high - low) * u[:, 0]) * torch.where(u[:, 1] < 0.5, -1, 1),
                (2 * u[:, 2] - 1) * cfg.velocity,
                (2 * u[:, 3] - 1) * cfg.partner,
                (2 * u[:, 4] - 1) * cfg.velocity,
            ),
            dim=-1,
        )
        first, last = cfg.recent_steps
        recent = first + (u[:, 6] * (last - first + 1)).long()
        change_at = torch.where(u[:, 5] < cfg.stationary, 0, cfg.seq_len - recent)
        frames = [state]
        for index in range(1, steps):
            coefficients = torch.where((index >= change_at)[:, None], after, before)
            state = self.dynamics.advance(state, coefficients)
            frames.append(state)
        states = torch.stack(frames, dim=1)
        observed = self.observation(states, level=u[:, 7], noise=noise)
        return EvaluatorEpisodes(
            windows=ObservedWindows(
                media=observed[:, : cfg.seq_len].clone(),
                label=observed[:, cfg.seq_len :].clone(),
            ),
            states=states,
            before=before,
            after=after,
            change_at=change_at,
        )


class _ShootingGeometry(Protocol):
    @property
    def step_size(self) -> float: ...


class _ShootingDynamics(Protocol):
    @property
    def config(self) -> _ShootingGeometry: ...

    def forecast(
        self,
        state: Tensor,
        *,
        coefficients: Tensor,
        steps: int,
    ) -> Tensor: ...


class LibraryShooting:
    """Fit the full mechanism library and hidden partner by multiple-start shooting.

    This is a learnability reference: it knows the equation form, the friction
    smoothing scale and a fixed fitting window, but not the
    change time, coefficients, hidden state or noise level of any episode.

    """

    class Config(Fig["LibraryShooting"]):
        visible_steps: int = 96
        """Newest observations fitted; may straddle a change, which is not detected."""

        restarts: int = 16
        """Independent initializations per episode; the best fit is kept."""

        iterations: int = 1200
        """Adam updates per fit."""

        lr: float = 0.02
        """Adam step size in unconstrained coordinates."""

        lower: tuple[float, ...] = (0.05, 0.0, 0.0, 0.0, 0.0, 0.05, 0.0)
        """Lower coefficient bounds shared across every tested domain."""

        upper: tuple[float, ...] = (4.0, 0.5, 2.0, 0.3, 2.0, 4.0, 0.5)
        """Upper coefficient bounds shared across every tested domain."""

        seed: int = 4242
        """Restart stream, independent of data and evaluation."""

        robust: float = 0.0
        """Cauchy loss scale in position units; zero keeps squared error."""

    dynamics_type: ClassVar[type[_ShootingDynamics]] = HiddenDynamics
    """Equation form the library declares; other physics is rejected."""

    latent_draws: ClassVar[int] = 2
    """Uniform draws initializing the unobserved state of each restart."""

    @classmethod
    def latent(cls, start: Tensor) -> Tensor:
        """Map restart draws to the hidden partner's position and velocity."""
        return torch.stack((2 * start[:, 0] - 1, start[:, 1] - 0.5), dim=-1)

    def __init__(self, config: Config) -> None:
        if (
            config.visible_steps < 8
            or min(config.restarts, config.iterations) < 1
            or math.isnan(config.lr)
            or config.lr <= 0
            or config.robust < 0
            or len(config.lower) != len(config.upper)
            or any(b <= a for a, b in zip(config.lower, config.upper, strict=True))
        ):
            raise ValueError("Shooting needs a window, budgets and ordered bounds.")
        self.config = config
        self.dynamics: _ShootingDynamics | None = None
        self.coefficients = len(config.lower)
        self.pred_len = 0

    def fit(
        self,
        train: ObservedWindows,
        *,
        validation: ObservedWindows,
        dynamics: _ShootingDynamics,
    ) -> dict[str, int | float]:
        """Take the time geometry and declared equation form; no training data.

        Args:
          train: Observed fitting windows.
          validation: Held-out windows for checkpoint selection, if used.
          dynamics: Reference dynamics for the observation step and equation form.

        Returns:
          diagnostics: Fitting diagnostics, including the number of attempted updates.

        """
        del validation
        if isinstance(dynamics, RegimeDynamics) or not isinstance(
            dynamics,
            self.dynamics_type,
        ):
            raise TypeError("Library shooting needs its declared equation form.")
        self.dynamics = dynamics
        self.pred_len = train.label.shape[1]
        return {"attempted_updates": 0}

    def predict(self, media: Tensor) -> Tensor:
        """Identify each episode from its newest observations, then roll forward.

        Args:
          media: Batch of observed histories.

        Returns:
          forecast: Forecast in the original observation units.

        """
        dynamics = self.dynamics
        if dynamics is None:
            raise RuntimeError("Fit the shooting reference before predicting.")
        cfg = self.config
        if media.ndim != 3 or media.shape[2] != 1 or media.shape[1] < cfg.visible_steps:
            raise ValueError("Expected position-only histories covering the window.")
        observed = media[:, -cfg.visible_steps :, 0].double()
        batch, restarts = len(observed), cfg.restarts
        target = observed.repeat_interleave(restarts, dim=0)
        lower = torch.tensor(cfg.lower, dtype=torch.float64)
        upper = torch.tensor(cfg.upper, dtype=torch.float64)
        generator = torch.Generator().manual_seed(cfg.seed)
        start = torch.rand(
            batch * restarts,
            self.coefficients + self.latent_draws,
            generator=generator,
        ).double()
        # First restart of each episode starts from mid-range coefficients.
        start[::restarts] = 0.5
        raw = torch.logit(
            start[:, : self.coefficients].clamp(0.02, 0.98),
        ).requires_grad_()
        step = dynamics.config.step_size
        state = torch.cat(
            (
                target[:, :1],
                (target[:, 2:3] - target[:, :1]) / (2 * step),
                self.latent(start[:, self.coefficients :]),
            ),
            dim=-1,
        ).requires_grad_()
        optimizer = torch.optim.Adam([raw, state], lr=cfg.lr)
        for _ in range(cfg.iterations):
            optimizer.zero_grad()
            self._residual(
                raw,
                state,
                target,
                lower,
                upper,
                dynamics=dynamics,
            ).sum().backward()
            optimizer.step()
        with torch.no_grad():
            loss = self._residual(raw, state, target, lower, upper, dynamics=dynamics)
            best = loss.view(batch, restarts).argmin(1) + torch.arange(batch) * restarts
            coefficients = lower + (upper - lower) * torch.sigmoid(raw[best])
            path = dynamics.forecast(
                state[best],
                coefficients=coefficients,
                steps=cfg.visible_steps - 1 + self.pred_len,
            )
        return path[:, -self.pred_len :, :1].to(media)

    def _residual(
        self,
        raw: Tensor,
        state: Tensor,
        target: Tensor,
        lower: Tensor,
        upper: Tensor,
        *,
        dynamics: _ShootingDynamics,
    ) -> Tensor:
        """Mean (robust) mismatch to the observed window for every restart."""
        coefficients = lower + (upper - lower) * torch.sigmoid(raw)
        path = dynamics.forecast(
            state,
            coefficients=coefficients,
            steps=target.shape[1] - 1,
        )
        error = torch.cat((state[:, None, 0], path[..., 0]), dim=1).sub(target)
        if self.config.robust > 0:
            return error.div(self.config.robust).square().log1p().mean(1)
        return error.square().mean(1)
