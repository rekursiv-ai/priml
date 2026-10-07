"""Published and smoke SpeedrunDiT configurations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import math

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
from priml.runtime import SingleProcess
from priml.testing.golden import assert_pprint_golden
from priml.train.parallelism import NoParallel


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
    assert math.isinf(base.num_steps_eval)
    assert smoke.experiment_name == "exp_smoke"
    assert isinstance(smoke.runtime, SingleProcess.Config)
    assert isinstance(smoke.step.parallelism, NoParallel.Config)
    assert smoke.max_steps == smoke.step.train_budget_steps == 5
    assert smoke.step.model.hidden_size == 32
    assert smoke.step.model.num_heads == 4
    assert smoke.step.model.depth == 6
    assert smoke.step.model.projector_hidden == 64
    assert smoke.step.model.projection_depths == (2, 3, 6)
    assert smoke.dataset.batch_size == 2
    assert smoke.checkpointer is None
    assert smoke.dataset.num_workers == 0
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
