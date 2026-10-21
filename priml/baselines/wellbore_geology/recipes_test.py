"""Require typed complete public family settings and native geometry propagation."""

from pathlib import Path
from typing import Final

import pytest
import torch

from priml.baselines.wellbore_geology.data import FoldObservations
from priml.baselines.wellbore_geology.metrics import PooledRMSE
from priml.baselines.wellbore_geology.recipes import (
    Family,
    _paths,
    _stopping as family_stopping,
    family_0719_v1,
    family_0724_v1,
    family_0729_v3,
    family_0801_v1,
    family_0801_v2,
    family_0803_v2,
)
from priml.baselines.wellbore_geology.train_step import WellboreTrainStep
from priml.optimizers.composite import CompositeOptimizer
from priml.optimizers.lr_schedule import RecursiveCosine
from priml.testing.golden import mismatches, read_tensors
from priml.train.epoch_loop import EarlyStopping
from priml.train.train_loop import TrainLoop
from priml.train.train_step import TrainStep


_THIS: Final = Path(__file__).resolve()
_CWD: Final = _THIS.parent


def _stopping(family: Family.Config) -> EarlyStopping.Config:
    stopping = family.training.early_stopping
    assert isinstance(stopping, EarlyStopping.Config)
    return stopping


def test_complete_public_training_schedules() -> None:
    families = [
        factory()
        for factory in (
            family_0719_v1,
            family_0724_v1,
            family_0729_v3,
            family_0801_v1,
            family_0801_v2,
            family_0803_v2,
        )
    ]
    actual: dict[str, torch.Tensor] = {}
    for family in families:
        values = {
            "epochs": family.training.max_epochs,
            "min_epochs": _stopping(family).min_epochs,
            "patience": _stopping(family).patience,
            "train_batch": family.training.dataset.loader.batch_size,
            "validation_batch": family.training.dataset.validation_batch_size,
            "accumulation": family.training.step.accumulate_grad_batches,
            "repeats": family.cv.repeats,
            "folds": family.cv.folds,
            "cv_seed": family.cv.split_seed,
        }
        actual.update(
            {
                f"{family.name}/{name}": torch.tensor(value)
                for name, value in values.items()
            },
        )
    expected = read_tensors(_THIS.with_name("testdata") / "recipe_settings.pt")
    assert mismatches(expected, actual) == []


def test_family_geometry_reaches_every_native_consumer() -> None:
    config = family_0803_v2()
    features = config.training.dataset.loader.dataset.transform.features
    features.prefix_rows = 12
    features.target_rows = 54
    features.downsample = 3
    features.reference_columns = 20
    features.reference_window = 24
    features.target_stats = (2, 7)
    config = config.finalized()
    assert config.training.step.model.channels_in == len(features.channels)
    assert config.training.step.model.reference_columns == 20
    assert config.training.step.model.reference_window == 24
    assert config.training.step.model.target_stats == (2, 7)
    assert config.particle is not None
    assert config.particle.prefix_rows == 12
    assert config.particle.target_rows == 54
    assert config.particle.downsample == 3
    assert config.particle.reference_columns == 20
    assert config.particle.reference_window == 24
    metric = config.training.metrics_eval[""]
    assert isinstance(metric, PooledRMSE.Config)
    for expansion in (config.prediction.expansion, metric.expansion):
        assert expansion.prefix_rows == 12
        assert expansion.target_rows == 54
        assert expansion.downsample == 3
        assert expansion.target_stats == (2, 7)


def test_family_geometry_never_overwrites_a_declared_consumer_value() -> None:
    config = family_0803_v2()
    assert config.particle is not None
    config.particle.downsample = 5
    config.training.step.model.reference_columns = 9
    config.prediction.expansion.target_stats = (3, 4)
    config = config.finalized()
    assert config.particle is not None
    assert config.particle.downsample == 5
    assert config.particle.prefix_rows == 1024
    assert config.training.step.model.reference_columns == 9
    assert config.training.step.model.reference_window == 100
    assert config.prediction.expansion.target_stats == (3, 4)
    assert config.prediction.expansion.downsample == 32


def test_native_step_uses_shared_training_lifecycle() -> None:
    assert issubclass(WellboreTrainStep, TrainStep)


def test_family_owns_printable_shared_training_slots() -> None:
    family = family_0803_v2()
    assert isinstance(family.training, TrainLoop.Config)
    assert isinstance(family.training.step, TrainStep.Config)
    assert isinstance(family.training.step.optimizer, CompositeOptimizer.Config)
    assert family.training.step.accumulate_grad_batches == 1
    assert family.training.step.gradient_clip_norm == 10
    assert family.training.step.dtype_autocast is torch.bfloat16
    assert family.training.max_epochs == 200


def test_family_declares_the_exact_schedule() -> None:
    schedule = family_0803_v2().training.step.lr_schedule
    assert isinstance(schedule, RecursiveCosine.Config)
    assert schedule.eta_min == 5e-7


def test_family_rejects_a_nonpositive_fold_count() -> None:
    config = family_0719_v1()
    config.first_n_folds = 0
    with pytest.raises(ValueError, match="first_n_folds"):
        config.make()


def test_family_particle_prior_must_agree_with_its_features() -> None:
    config = family_0803_v2()
    config.particle = None
    with pytest.raises(ValueError, match="exactly when"):
        config.make()


def test_published_family_helpers_reject_a_missing_rule() -> None:
    config = family_0803_v2()
    config.training.early_stopping = None
    with pytest.raises(TypeError, match="early stopping"):
        family_stopping(config)
    paths = family_0803_v2()
    paths.training.dataset.loader.dataset.transform.paths = None
    with pytest.raises(TypeError, match="path simulation"):
        _paths(paths)


def test_family_without_held_out_observations_still_resolves_its_inputs() -> None:
    config = family_0803_v2()
    config.dataset_directory = Path("/data")
    config.training.dataset.validation = None
    resolved = config.finalized()
    train = resolved.training.dataset.loader.dataset.observations
    assert isinstance(train, FoldObservations.Config)
    assert train.directory == Path("/data/train")


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
