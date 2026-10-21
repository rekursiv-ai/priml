"""Draw independent systems with missing force laws for the transfer benchmark."""

from __future__ import annotations

from dataclasses import field
from typing import Self, override

import math

from configgle import Fig, Maker, traverse
from torch import Tensor

import torch

from priml.baselines.oscillator.forces import CorrectedOscillator


class SaturatingForce:
    """A bounded restoring force or bounded friction."""

    class Config(Fig["SaturatingForce"]):
        axis: int = 0
        """State channel: displacement or velocity."""

        scale: float = 1.0
        """State magnitude at which the force starts to flatten."""

    def __init__(self, config: Config) -> None:
        if (
            config.axis not in (0, 1)
            or not math.isfinite(config.scale)
            or config.scale <= 0
        ):
            raise ValueError(
                "A saturating force needs a state channel and positive scale.",
            )
        self.config = config

    @property
    def name(self) -> str:
        """Return the force expression without its coefficient."""
        state = "position" if self.config.axis == 0 else "velocity"
        return f"-{self.config.scale}*tanh({state}/{self.config.scale})"

    def __call__(self, position: Tensor, *, velocity: Tensor) -> Tensor:
        """Return a bounded opposing force in the state dtype."""
        state = position if self.config.axis == 0 else velocity
        return -self.config.scale * (state / self.config.scale).tanh()


class SineForce:
    """A pendulum-like restoring term added to the known linear spring."""

    class Config(Fig["SineForce"]):
        scale: float = 1.0
        """Displacement scale of the sine response."""

    def __init__(self, config: Config) -> None:
        if not math.isfinite(config.scale) or config.scale <= 0:
            raise ValueError("The sine scale must be finite and positive.")
        self.config = config

    @property
    def name(self) -> str:
        """Return the extra spring's expression."""
        return f"-{self.config.scale}*sin(position/{self.config.scale})"

    def __call__(self, position: Tensor, *, velocity: Tensor) -> Tensor:
        """Return the extra restoring force."""
        del velocity
        return -self.config.scale * (position / self.config.scale).sin()


class CoupledDrag:
    """Let position control the strength of an opposing velocity force."""

    class Config(Fig["CoupledDrag"]):
        scale: float = 1.0
        """Displacement scale that bounds the drag multiplier."""

    def __init__(self, config: Config) -> None:
        if not math.isfinite(config.scale) or config.scale <= 0:
            raise ValueError("The coupled-drag scale must be finite and positive.")
        self.config = config

    @property
    def name(self) -> str:
        """Return the state-dependent drag expression."""
        return f"-position^2*velocity/(1+(position/{self.config.scale})^2)"

    def __call__(self, position: Tensor, *, velocity: Tensor) -> Tensor:
        """Return drag that always removes mechanical energy."""
        return (
            -position.square()
            * velocity
            / (1 + (position / self.config.scale).square())
        )


class SystemFamily:
    """Draw force coefficients without sharing a random stream with trajectories."""

    class Config(Fig["SystemFamily"]):
        name: str = "linear"
        """Family label used only for reporting and split checks."""

        reference: CorrectedOscillator.Config = field(
            default_factory=CorrectedOscillator.Config,
        )
        """Nominal force law; the forecaster never receives this config."""

        coefficient_scale: tuple[float, float] = (0.6, 1.4)
        """Independent multiplier range for each nonzero extra coefficient."""

        seed: int = 11_001
        """First system seed; successive instances add one."""

        instances: int = 4
        """Independent systems in this fixed family."""

        @override
        def finalize(self) -> Self:
            """Finalize this owner, leaving its sampled equation template pending."""
            self._finalized = True
            return self

    def __init__(self, config: Config) -> None:
        for match in traverse(config.reference, Maker, recurse=True):
            if getattr(match.config, "_finalized", False):
                raise ValueError("System families require unfinalized templates.")
        low, high = config.coefficient_scale
        if not config.name or config.instances <= 0 or config.seed < 0:
            raise ValueError(
                "Families need a name, positive count, and nonnegative seed.",
            )
        if (
            not all(math.isfinite(value) for value in (low, high))
            or low <= 0
            or low > high
        ):
            raise ValueError(
                "Coefficient multipliers must be finite, positive, and ordered.",
            )
        self.config = config

    def draw(self, index: int) -> CorrectedOscillator.Config:
        """Return one system's equation, leaving the prototype and global RNG alone.

        Args:
          index: Instance index within this family's declared count.

        Returns:
          reference: Equation with independently sampled extra-force coefficients.

        Raises:
          ValueError: The requested instance does not exist.

        """
        if index < 0 or index >= self.config.instances:
            raise ValueError("System index is outside the family.")
        generator = torch.Generator().manual_seed(self.config.seed + index)
        cfg = self.config.reference.copy_tree()
        low, high = self.config.coefficient_scale
        draws = low + (high - low) * torch.rand(
            1 + len(cfg.corrections),
            generator=generator,
        )
        cfg.cubic *= float(draws[0])
        for term, multiplier in zip(cfg.corrections, draws[1:], strict=True):
            term.coefficient *= float(multiplier)
        return cfg
