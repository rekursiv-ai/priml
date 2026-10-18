"""Add replaceable force terms to the supplied oscillator."""

from __future__ import annotations

from dataclasses import field
from typing import Protocol, override

import math

from configgle import Fig, Makeable, Makes
from torch import Tensor

from priml.baselines.oscillator.data import Oscillator


class ForceTerm(Protocol):
    """One named force evaluated at an observed state."""

    @property
    def name(self) -> str:
        """Return the expression used in fit reports."""
        ...

    def __call__(self, position: Tensor, *, velocity: Tensor) -> Tensor:
        """Return a force with the same shape as position."""
        ...


class PowerForce:
    """An odd restoring force or drag term."""

    class Config(Fig["PowerForce"]):
        axis: int = 0
        """State channel: zero for position, one for velocity."""

        power: int = 3
        """Exponent of the force magnitude; the sign follows the state."""

    def __init__(self, config: Config) -> None:
        if config.axis not in (0, 1) or config.power < 1:
            raise ValueError("Force terms need a state channel and positive power.")
        self.config = config

    @property
    def name(self) -> str:
        """Return the signed power expression, without its fitted coefficient."""
        state = "position" if self.config.axis == 0 else "velocity"
        return f"-{state}*abs({state})^{self.config.power - 1}"

    def __call__(self, position: Tensor, *, velocity: Tensor) -> Tensor:
        """Evaluate the restoring force or drag.

        Args:
          position: Displacements with any batch shape.
          velocity: Velocities with the same shape.

        Returns:
          force: An opposing force with the input shape and dtype.

        """
        state = position if self.config.axis == 0 else velocity
        return -state * state.abs().pow(self.config.power - 1)


class WeightedForce:
    """Keep each coefficient beside the force it scales."""

    class Config(Fig["WeightedForce"]):
        term: Makeable[ForceTerm] = field(default_factory=PowerForce.Config)
        """Force supplied by this correction."""

        coefficient: float = 0.0
        """Nonnegative force strength."""

    def __init__(self, config: Config) -> None:
        if not math.isfinite(config.coefficient) or config.coefficient < 0:
            raise ValueError("Force strengths must be finite and nonnegative.")
        self.term = config.term.make()
        self.coefficient = config.coefficient

    def __call__(self, position: Tensor, *, velocity: Tensor) -> Tensor:
        """Return the weighted force at the supplied state."""
        return self.coefficient * self.term(position, velocity=velocity)


class CorrectedOscillator(Oscillator):
    """Run the existing integrator with additional force terms."""

    class Config(Makes["CorrectedOscillator"], Oscillator.Config):
        corrections: list[WeightedForce.Config] = field(
            default_factory=list[WeightedForce.Config],
        )
        """Extra forces, evaluated at every integration substep."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.corrections: list[WeightedForce] = [
            term.make() for term in config.corrections
        ]

    @override
    def acceleration(self, position: Tensor, *, velocity: Tensor) -> Tensor:
        """Add the configured corrections to the original force.

        Args:
          position: Current displacements.
          velocity: Current velocities with the same shape.

        Returns:
          acceleration: Total force at the current state, with unit mass.

        """
        result = super().acceleration(position, velocity=velocity)
        for correction in self.corrections:
            result = result + correction(position, velocity=velocity)
        return result
