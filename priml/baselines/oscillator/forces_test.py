"""Check force signs, coefficients, and unchanged baseline integration."""

import pytest
import torch

from priml.baselines.oscillator.data import Oscillator
from priml.baselines.oscillator.forces import (
    CorrectedOscillator,
    PowerForce,
    WeightedForce,
)


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("power", [1, 2, 3, 5])
def test_power_force_sign_and_magnitude(axis: int, power: int) -> None:
    position = torch.tensor([-2.0, 0.0, 3.0])
    velocity = torch.tensor([4.0, -1.0, 0.0])
    force = PowerForce.Config(axis=axis, power=power).make()
    state = position if axis == 0 else velocity
    expected = -state.sign() * state.abs().pow(power)
    assert torch.equal(force(position, velocity=velocity), expected)
    assert (force(position, velocity=velocity) * state <= 0).all()
    assert ("position" if axis == 0 else "velocity") in force.name


@pytest.mark.parametrize(("axis", "power"), [(-1, 3), (2, 3), (0, 0)])
def test_invalid_force_is_rejected(axis: int, power: int) -> None:
    with pytest.raises(ValueError, match="positive power"):
        PowerForce.Config(axis=axis, power=power).make()


@pytest.mark.parametrize("coefficient", [-0.1, float("nan"), float("inf")])
def test_invalid_strength_is_rejected(coefficient: float) -> None:
    with pytest.raises(ValueError, match="finite and nonnegative"):
        WeightedForce.Config(coefficient=coefficient).make()


def test_empty_correction_matches_original_bit_for_bit() -> None:
    states = torch.tensor([[0.3, -0.2], [-0.8, 0.5], [1.1, 0.4]])
    base = Oscillator.Config(cubic=0.27)
    cfg = CorrectedOscillator.Config()
    cfg.update(base)
    assert torch.equal(
        cfg.make().forecast(states, steps=4),
        base.make().forecast(states, steps=4),
    )


def test_extra_drag_is_applied_inside_the_integrator() -> None:
    cfg = CorrectedOscillator.Config()
    cfg.substeps = 1
    cfg.corrections = [
        WeightedForce.Config(term=PowerForce.Config(axis=1, power=2), coefficient=0.3),
    ]
    model = cfg.make()
    states = torch.tensor([[0.3, -0.2], [-0.8, 0.5], [1.1, 0.4]])
    position, velocity = states.unbind(dim=-1)
    force = (
        -cfg.stiffness * position
        - cfg.damping * velocity
        - 0.3 * velocity * velocity.abs()
    )
    next_velocity = velocity + cfg.step_size * force
    next_position = position + cfg.step_size * next_velocity
    expected = torch.stack((next_position, next_velocity), dim=-1)
    # One output step is required to check this integrator update directly.
    torch.testing.assert_close(model.forecast(states, steps=1)[:, 0], expected)


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
