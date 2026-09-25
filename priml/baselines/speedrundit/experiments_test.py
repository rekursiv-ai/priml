"""Published and smoke SpeedrunDiT configurations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from configgle.testing import assert_pprint_golden

import pytest
import torch

from priml.baselines.speedrundit.experiments import (
    SpeedrunTrainLoop,
    exp000,
    exp001,
    exp_smoke,
)
from priml.optimizers.composite import CompositeOptimizer
from priml.optimizers.muon import Muon


if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.mark.parametrize(
    ("name", "recipe"),
    [("exp000", exp000), ("exp001", exp001)],
)
def test_experiment_config_goldens(
    name: str,
    recipe: Callable[[], SpeedrunTrainLoop],
) -> None:
    assert_pprint_golden(test_file=__file__, name=name, config=recipe())


def test_smoke_keeps_the_training_recipe() -> None:
    """The smoke run only reduces cost and the number of updates."""
    base, smoke = exp000(), exp_smoke()
    assert smoke.max_steps == smoke.step.train_budget_steps == 5
    assert smoke.max_steps < base.max_steps
    assert smoke.step.model.hidden_size < base.step.model.hidden_size
    assert smoke.step.model.depth < base.step.model.depth
    assert smoke.step.model.in_channels == base.step.model.in_channels
    assert smoke.step.model.patch_size == base.step.model.patch_size
    assert smoke.step.model.drop_ratio == base.step.model.drop_ratio
    assert smoke.step.model.qk_norm == base.step.model.qk_norm
    assert smoke.step.latent_scale == base.step.latent_scale
    assert smoke.step.projection_coeff == base.step.projection_coeff
    assert smoke.step.cfm_coeff == base.step.cfm_coeff


def test_exp001_changes_only_numerical_implementations() -> None:
    source, simpler = exp000(), exp001()
    assert source.step.model.position_compute_dtype == torch.float64
    assert simpler.step.model.position_compute_dtype == torch.float32
    assert source.step.model.reference_rope
    assert not simpler.step.model.reference_rope
    assert isinstance(source.step.optimizer, CompositeOptimizer.Config)
    assert isinstance(simpler.step.optimizer, CompositeOptimizer.Config)
    source_muon = source.step.optimizer.optimizers[1]
    simpler_muon = simpler.step.optimizer.optimizers[1]
    assert isinstance(source_muon, Muon.Config)
    assert isinstance(simpler_muon, Muon.Config)
    assert source_muon.reference_numerics
    assert not simpler_muon.reference_numerics
    assert source.step.model.depth == simpler.step.model.depth
    assert source.step.model.projection_depths == simpler.step.model.projection_depths
    assert source.step.projection_coeff == simpler.step.projection_coeff
    assert source.step.cls_coeff == simpler.step.cls_coeff
    assert source.step.cfm_coeff == simpler.step.cfm_coeff


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
