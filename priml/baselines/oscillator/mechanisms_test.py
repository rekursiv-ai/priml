"""Check force behavior and independent coefficient draws."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from priml.baselines.oscillator.forces import WeightedForce
from priml.baselines.oscillator.mechanisms import (
    CoupledDrag,
    SaturatingForce,
    SineForce,
    SystemFamily,
)


if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.mark.parametrize(
    "factory",
    [SaturatingForce.Config, SineForce.Config, CoupledDrag.Config],
)
def test_extra_forces_preserve_shape_and_reverse_with_the_state(
    factory: Callable[
        [],
        SaturatingForce.Config | SineForce.Config | CoupledDrag.Config,
    ],
) -> None:
    cfg = factory()
    term = cfg.make()
    position = torch.tensor([-1.3, 0.0, 0.7], dtype=torch.float64)
    velocity = torch.tensor([0.4, -0.2, -0.8], dtype=torch.float64)
    actual = term(position, velocity=velocity)
    assert actual.shape == position.shape
    assert actual.dtype == position.dtype
    assert torch.isfinite(actual).all()
    assert torch.equal(term(-position, velocity=-velocity), -actual)
    assert term.name
    cfg.scale = 2.0
    assert not torch.equal(cfg.make()(position, velocity=velocity), actual)
    cfg.scale = 0.0
    with pytest.raises(ValueError, match="scale"):
        cfg.make()


def test_drag_cannot_add_mechanical_energy_and_saturation_is_bounded() -> None:
    position = torch.tensor([-9.0, 0.0, 7.0])
    velocity = torch.tensor([0.4, -0.2, -0.8])
    assert (
        CoupledDrag.Config().make()(position, velocity=velocity) * velocity <= 0
    ).all()
    cfg = SaturatingForce.Config()
    cfg.axis = 1
    force = cfg.make()(position, velocity=velocity)
    assert (force * velocity <= 0).all()
    assert force.abs().max() <= cfg.scale
    cfg.axis = 3
    with pytest.raises(ValueError, match="channel"):
        cfg.make()


def test_system_draws_are_repeatable_and_leave_prototypes_and_rng_untouched() -> None:
    cfg = SystemFamily.Config()
    cfg.reference.cubic = 0.4
    cfg.reference.corrections = [
        WeightedForce.Config(term=CoupledDrag.Config(), coefficient=0.5),
    ]
    before = cfg.copy_tree()
    state = torch.random.get_rng_state()
    family = cfg.make()
    first, second = family.draw(0), family.draw(1)
    assert first == family.draw(0)
    assert first != second
    assert first.stiffness == second.stiffness == cfg.reference.stiffness
    assert first.cubic != second.cubic
    assert cfg == before
    assert torch.equal(torch.random.get_rng_state(), state)
    for index in (-1, cfg.instances):
        with pytest.raises(ValueError, match="index"):
            family.draw(index)


@pytest.mark.parametrize("bad", ["name", "count", "seed", "bounds", "finite"])
def test_invalid_family_settings_are_rejected(bad: str) -> None:
    cfg = SystemFamily.Config()
    if bad == "name":
        cfg.name = ""
    elif bad == "count":
        cfg.instances = 0
    elif bad == "seed":
        cfg.seed = -1
    elif bad == "bounds":
        cfg.coefficient_scale = (2.0, 1.0)
    else:
        cfg.coefficient_scale = (1.0, float("nan"))
    with pytest.raises(ValueError, match=r"Families|Coefficient"):
        cfg.make()


def test_system_families_require_raw_templates() -> None:
    cfg = SystemFamily.Config()
    cfg.reference.finalize()
    with pytest.raises(ValueError, match="unfinalized"):
        cfg.make()


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
