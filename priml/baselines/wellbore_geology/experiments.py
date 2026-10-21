"""Native complete public-best wellbore study and separate budgeted candidates."""

from __future__ import annotations

from typing import TYPE_CHECKING

from priml.baselines.wellbore_geology.inference import (
    SingleFamilyPrediction,
)
from priml.baselines.wellbore_geology.pipeline import WellboreStudy
from priml.baselines.wellbore_geology.recipes import (
    family_0719_v1,
    family_0724_v1,
    family_0729_v3,
    family_0801_v1,
    family_0801_v2,
    family_0803_v2,
)
from priml.model.convnext import ConvNeXtBlock, ConvNeXtStage
from priml.train.budget import WallBudget


if TYPE_CHECKING:
    from priml.baselines.wellbore_geology.model import SequenceModel
    from priml.baselines.wellbore_geology.particles import ParticlePrior


def exp000() -> WellboreStudy.Config:
    """Train all six native families across all ninety public geographic folds.

    Hypothesis:
      The complete released winning recipe is a faithful agent benchmark seed.

    References:
      https://github.com/IAmAValidUsername/kaggle_ROGII_1st_place_solution_Ruby/tree/fade6112ffebd9e27c08ac7f053e8d4c3d5c319a

    Results:
      Public solution reports private RMSE 5.639. Native full-run results pending.

    Returns:
      config: Complete native model, loss, optimizer, data and publication recipe.

    """
    cfg = WellboreStudy.Config()
    cfg.families = [
        family_0719_v1(),
        family_0724_v1(),
        family_0729_v3(),
        family_0801_v1(),
        family_0801_v2(),
        family_0803_v2(),
    ]
    return cfg


def exp001() -> WellboreStudy.Config:
    """Restrict exp000 to its unchanged latest family.

    Hypothesis:
      The latest family can stand alone. Its single-family publication policy
      is inseparable from removing the other five ensemble members.

    References:
      exp000.

    Results:
      TBD.

    Returns:
      config: Complete latest family with all fifteen original folds.

    """
    cfg = exp000()
    cfg.experiment_name = "exp001"
    cfg.families = cfg.families[-1:]
    cfg.results.prediction = SingleFamilyPrediction.Config()
    return cfg


def exp002() -> WellboreStudy.Config:
    """Reduce exp001 to one complete geographic repeat.

    Hypothesis:
      One repeat retains useful geographic coverage at one third the training cost.

    References:
      exp001.

    Results:
      TBD.

    Returns:
      config: Latest family with all five folds in one repeat.

    """
    cfg = exp001()
    cfg.experiment_name = "exp002"
    cfg.families[0].cv.repeats = 1
    return cfg


def exp003() -> WellboreStudy.Config:
    """Restrict exp002 to its first complete fold.

    Hypothesis:
      One fold provides a bounded training candidate before adding wall truncation.

    References:
      exp002.

    Results:
      TBD.

    Returns:
      config: One unchanged latest-family fold.

    """
    cfg = exp002()
    cfg.experiment_name = "exp003"
    cfg.families[0].first_n_folds = 1
    return cfg


def exp004() -> WellboreStudy.Config:
    """Apply a cold outer wall budget to exp003.

    Hypothesis:
      One winning model can learn useful updates within five H100/H200 minutes.
      The evaluation reserve belongs to the same outer wall allowance. Charging
      cold particle preparation to that allowance makes the run prepare its own
      records, so the two travel together.

    References:
      exp003.

    Results:
      TBD.

    Returns:
      config: One full-recipe fold with 285 measured seconds and a local-eval reserve.

    """
    cfg = exp003()
    cfg.experiment_name = "exp004"
    cfg.budget = WallBudget.Config()
    cfg.budget.seconds = 285
    cfg.budget.reserve_evaluation_sec = 30
    cfg.families[0].prepare_particles = True
    return cfg


def exp005() -> WellboreStudy.Config:
    """Reduce exp004 to 64 particles per filtering run.

    Hypothesis:
      A smaller cloud leaves more cold wall time for model training.

    References:
      exp004.

    Results:
      TBD.

    Returns:
      config: Unchanged seed budget with 64 particles per run.

    """
    cfg = exp004()
    cfg.experiment_name = "exp005"
    _particle(cfg).particles = 64
    return cfg


def exp006() -> WellboreStudy.Config:
    """Reduce exp005 to eight particle seed runs.

    Hypothesis:
      Fewer independent runs leave more cold wall time for model training.

    References:
      exp005.

    Results:
      TBD.

    Returns:
      config: Separate candidate changing only the particle seed-run budget.

    """
    cfg = exp005()
    cfg.experiment_name = "exp006"
    _particle(cfg).seeds = 8
    return cfg


def exp_smoke() -> WellboreStudy.Config:
    """Run every native stage with narrower models and fewer complete folds.

    Hypothesis:
      Reduced model/data geometry exposes wiring failures before full training.

    References:
      exp000.

    Results:
      Pending real-data execution.

    Returns:
      config: All six model families at explicit smoke-run dimensions.

    """
    cfg = exp000()
    cfg.experiment_name = "exp_smoke"
    for family in cfg.families:
        family.training.max_epochs = 1
        family.cv.folds = 2
        family.cv.repeats = 1
        shrink_model(family.training.step.model)
        family.training.dataset.loader.dataset.transform.features.target_rows = 64
        family.training.dataset.loader.dataset.transform.features.reference_columns = 32
        if family.particle is not None:
            family.particle.particles = 3
            family.particle.seeds = 5
            family.particle.paths = 2
    return cfg


def shrink_model(model: SequenceModel.Config) -> None:
    """Narrow every stage of the posterior model to a few channels and no weights.

    Args:
      model: Complete posterior model recipe, edited in place.

    """
    model.channels_hidden = 2
    model.unet.channels_hidden = (2, 3, 4, 5)
    model.unet.backbone.network.blocks_per_stage = (1, 1, 1, 1)
    model.kernel_size = 1
    stage = model.unet.backbone.network.stage
    assert isinstance(stage, ConvNeXtStage.Config)
    assert isinstance(stage.block, ConvNeXtBlock.Config)
    stage.block.conv_dw.kernel_size = 3
    stage.block.conv_dw.padding = 1
    model.unet.backbone.checkpoint = None


def _particle(cfg: WellboreStudy.Config) -> ParticlePrior.Config:
    """Return the first family's particle prior, which every candidate declares."""
    particle = cfg.families[0].particle
    if particle is None:
        raise ValueError("Expected particle is not None.")
    return particle
