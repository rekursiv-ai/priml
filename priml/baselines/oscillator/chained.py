"""Forecast a stuck, glitchy sensor on a driven chain of three masses."""

from __future__ import annotations

from dataclasses import field, replace
from typing import TYPE_CHECKING, override

import math

from configgle import Makeable, Makes
from torch import Tensor

import torch

from priml.baselines.oscillator.forced import (
    ForcedData,
    ForcedDomain,
    ForcedDynamics,
    ForcedShooting,
    GlitchySensor,
)
from priml.baselines.oscillator.hidden import LibraryShooting
from priml.baselines.oscillator.regimes import EvaluatorEpisodes, ObservedWindows


if TYPE_CHECKING:
    from collections.abc import Callable


COEFFICIENTS = 16
"""Part four's eleven, then far coupling, far stiffness, far damping, second
drive amplitude and second drive frequency."""

LATENT = 6
"""Unobserved state: near partner position, velocity, drive phase, far partner
position, velocity, second drive phase."""


class ChainedDomain(ForcedDomain):
    """Part four's ranges plus a far hidden mass and a second drive tone."""

    class Config(Makes["ChainedDomain"], ForcedDomain.Config):
        far_coupling: tuple[float, float] = (0.05, 0.4)
        """Spring joining the near hidden mass to the far one."""

        far_stiffness: tuple[float, float] = (0.3, 3.0)
        """Linear restoring coefficient of the far hidden mass."""

        far_damping: tuple[float, float] = (0.01, 0.1)
        """Viscous drag of the far hidden mass."""

        second_drive: tuple[float, float] = (0.2, 0.6)
        """Amplitude of the second unobserved tone on the observed mass."""

        second_frequency: tuple[float, float] = (1.7, 2.9)
        """Angular frequency of the second tone, above the first's training range."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        extra = self.bounds(config)[11:]
        if (extra[[0, 2, 3], 0] < 0).any() or (extra[[1, 4], 0] <= 0).any():
            raise ValueError("Coefficient ranges must be finite, physical and ordered.")

    @staticmethod
    @override
    def bounds(config: ForcedDomain.Config) -> Tensor:
        """Part four's rows, then the far mass and second tone."""
        assert isinstance(config, ChainedDomain.Config)
        extra = torch.tensor(
            [
                config.far_coupling,
                config.far_stiffness,
                config.far_damping,
                config.second_drive,
                config.second_frequency,
            ],
            dtype=torch.float64,
        )
        return torch.cat((ForcedDomain.bounds(config), extra))


class ChainedDynamics(ForcedDynamics):
    """Observed mass, near and far hidden masses in a chain, and two drive tones."""

    class Config(Makes["ChainedDynamics"], ForcedDynamics.Config):
        pass

    @override
    def advance(self, state: Tensor, coefficients: Tensor) -> Tensor:
        """Advance one observation interval; differentiable in both arguments."""
        x1, v1, x2, v2, phase, x3, v3, tone = state.unbind(dim=-1)
        k1, c1, a1, mu, q1, e1, kc, k2, c2, drive, omega, kd, k3, c3, g, w2 = (
            coefficients.unbind(dim=-1)
        )
        delta = self.config.step_size / self.config.substeps
        for _ in range(self.config.substeps):
            near = kc * (x2 - x1)
            far = kd * (x3 - x2)
            acc1 = (
                -k1 * x1
                - (c1 + e1 * x1.square()) * v1
                - q1 * x1.square()
                - a1 * x1.pow(3)
                - mu * torch.tanh(v1 / self.config.stick_velocity)
                + near
                + drive * torch.cos(phase)
                + g * torch.cos(tone)
            )
            acc2 = -k2 * x2 - c2 * v2 - near + far
            acc3 = -k3 * x3 - c3 * v3 - far
            v1 = v1 + delta * acc1
            v2 = v2 + delta * acc2
            v3 = v3 + delta * acc3
            x1 = x1 + delta * v1
            x2 = x2 + delta * v2
            x3 = x3 + delta * v3
            phase = phase + delta * omega
            tone = tone + delta * w2
        return torch.stack((x1, v1, x2, v2, phase, x3, v3, tone), dim=-1)


class StuckSensor(GlitchySensor):
    """Part four's sensor that also freezes on its last reading in random bursts."""

    class Config(Makes["StuckSensor"], GlitchySensor.Config):
        stuck_rate: float = 0.01
        """Probability that a free reading starts a frozen burst."""

        stuck_length: float = 6.0
        """Mean burst length in readings; bursts end geometrically."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        if not 0 <= config.stuck_rate <= 1 or not config.stuck_length >= 1:
            raise ValueError("Sensor ranges must be finite, nonnegative and ordered.")
        self.stuck_rate = config.stuck_rate
        self.stay = 1 - 1 / config.stuck_length

    @override
    def __call__(self, states: Tensor, *, level: Tensor, draws: Tensor) -> Tensor:
        """Corrupt as part four, then repeat the previous reading while frozen."""
        reading = super().__call__(states, level=level, draws=draws)
        frames = [reading[:, 0]]
        stuck = torch.zeros_like(level, dtype=torch.bool)
        for index in range(1, reading.shape[1]):
            u = draws[:, index, 3]
            stuck = torch.where(stuck, u < self.stay, u < self.stuck_rate)
            frames.append(torch.where(stuck[:, None], frames[-1], reading[:, index]))
        return torch.stack(frames, dim=1)


class ChainedData(ForcedData):
    """A second hidden mass, up to four changes, and a sensor outage before the forecast."""

    coefficients = COEFFICIENTS
    extra_draws = 3
    reading_draws = 4

    class Config(Makes["ChainedData"], ForcedData.Config):
        seq_len: int = 384
        """Observed states before the forecast starts."""

        pred_len: int = 192
        """Future states per episode."""

        # Lambdas: ty types a bare Makes config class as a Maker union.
        dynamics: ForcedDynamics.Config = field(
            default_factory=lambda: ChainedDynamics.Config(),  # noqa: PLW0108
        )
        """Reference motion shared by the targets and evaluator oracle."""

        domain: ForcedDomain.Config = field(
            default_factory=lambda: ChainedDomain.Config(),  # noqa: PLW0108
        )
        """Training ranges; also every earlier regime of the observed mass."""

        observation: GlitchySensor.Config = field(
            default_factory=lambda: StuckSensor.Config(),  # noqa: PLW0108
        )
        """Sensor exposed to every method, including its training targets."""

        changes: tuple[float, ...] = (0.1, 0.2, 0.3, 0.25, 0.15)
        """Probability of zero to four changes inside the history."""

        recent_steps: tuple[int, int] = (112, 168)
        """Inclusive range of states after the last change, including the outage."""

        segment_steps: tuple[int, int] = (24, 56)
        """Inclusive range of states between consecutive earlier changes."""

        outage: tuple[int, int] = (16, 24)
        """Inclusive range of newest readings frozen at their last good value."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        low, high = config.outage
        if not 0 <= low <= high < config.recent_steps[0]:
            raise ValueError("Episode geometry, schedule and bounds must be valid.")
        self.outage = config.outage

    @override
    def sample(
        self,
        *,
        count: int,
        seed: int,
        domain: Makeable[Callable[[Tensor], Tensor]] | None = None,
    ) -> EvaluatorEpisodes:
        """Part four's draw, then freeze the newest readings; labels stay live."""
        episode = super().sample(count=count, seed=seed, domain=domain)
        low, high = self.outage
        # Offset keeps the outage stream distinct from the episode and sensor streams.
        draws = torch.rand(
            count,
            generator=torch.Generator().manual_seed(seed + 2**33),
            dtype=torch.float64,
        )
        media = episode.windows.media
        good = media.shape[1] - 1 - low - (draws * (high - low + 1)).long()
        frozen = torch.arange(media.shape[1]) > good[:, None]
        last = media[torch.arange(count), good][:, None]
        media = torch.where(frozen[..., None], last, media)
        return replace(
            episode,
            windows=ObservedWindows(media=media, label=episode.windows.label),
        )

    @override
    def initial_state(self, u: Tensor, extra: Tensor) -> Tensor:
        """Part four's state, then the far mass and the second tone's phase."""
        cfg = self.config
        far = torch.stack(
            (
                (2 * extra[:, 0] - 1) * cfg.partner,
                (2 * extra[:, 1] - 1) * cfg.velocity,
                2 * math.pi * extra[:, 2],
            ),
            dim=-1,
        )
        return torch.cat((super().initial_state(u, extra), far), dim=-1)


class ChainedShooting(LibraryShooting):
    """Shoot the chained library with part four's robust loss and window."""

    dynamics_type = ChainedDynamics
    latent_draws = LATENT

    class Config(Makes["ChainedShooting"], LibraryShooting.Config):
        visible_steps: int = 64
        """Newest observations fitted; may straddle a change, which is not detected."""

        robust: float = 0.1
        """Cauchy scale in position units, so glitches barely move the fit."""

        lower: tuple[float, ...] = (
            -3.0, -0.5, 0.0, 0.0, -1.5, 0.0, 0.0, 0.05, 0.0, 0.0, 0.2,
            0.0, 0.05, 0.0, 0.0, 0.2,
        )  # fmt: skip
        """Lower coefficient bounds shared across every tested domain."""

        upper: tuple[float, ...] = (
            4.0, 0.6, 2.0, 0.3, 1.5, 1.5, 1.5, 4.0, 0.5, 1.0, 5.0,
            1.5, 6.0, 0.5, 1.0, 5.0,
        )  # fmt: skip
        """Upper coefficient bounds shared across every tested domain."""

    @staticmethod
    @override
    def latent(start: Tensor) -> Tensor:
        """Part four's latent draws for each hidden mass and tone in turn."""
        return torch.cat(
            (ForcedShooting.latent(start[:, :3]), ForcedShooting.latent(start[:, 3:])),
            dim=-1,
        )
