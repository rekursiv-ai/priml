"""Require the complete native public recipe and explicit independent deltas."""

from collections.abc import Callable
from pathlib import Path

import pytest

from priml.baselines.wellbore_geology.experiments import (
    _particle as candidate_particle,
    exp000,
    exp001,
    exp002,
    exp003,
    exp004,
    exp005,
    exp006,
    exp_smoke,
)
from priml.baselines.wellbore_geology.inference import (
    SingleFamilyPrediction,
)
from priml.baselines.wellbore_geology.particles import ParticlePrior
from priml.baselines.wellbore_geology.pipeline import WellboreStudy
from priml.testing.golden import assert_pprint_golden, assert_text_golden
from priml.train.budget import WallBudget


def _delta(parent: WellboreStudy.Config, child: WellboreStudy.Config) -> str:
    """Render the declared change with named, whitespace-free diff headers."""
    _, _, *body = parent.udiff(
        child,
        finalize=False,
        color=False,
        n=1,
        hide_default_values=True,
    ).split("\n")
    return "\n".join(
        [f"--- {parent.experiment_name}", f"+++ {child.experiment_name}", *body],
    )


def _exp001_parent() -> WellboreStudy.Config:
    parent = exp000()
    parent.families = parent.families[-1:]
    return parent


@pytest.mark.parametrize(
    ("name", "parent", "factory"),
    [
        ("exp001", _exp001_parent, exp001),
        ("exp002", exp001, exp002),
        ("exp003", exp002, exp003),
        ("exp004", exp003, exp004),
        ("exp005", exp004, exp005),
        ("exp006", exp005, exp006),
        ("exp_smoke", exp000, exp_smoke),
    ],
)
def test_fork_delta_golden(
    name: str,
    parent: Callable[[], WellboreStudy.Config],
    factory: Callable[[], WellboreStudy.Config],
) -> None:
    assert_text_golden(
        test_file=__file__,
        name=f"{name}_delta",
        rendered=_delta(parent(), factory()),
    )


@pytest.fixture(scope="module")
def finalized_exp000() -> WellboreStudy.Config:
    """Finalize the complete study once; every golden test reads it."""
    return exp000().finalized()


@pytest.mark.parametrize(
    ("name", "index"),
    [
        ("exp000_0719_v1", 0),
        ("exp000_0724_v1", 1),
        ("exp000_0729_v3", 2),
        ("exp000_0801_v1", 3),
        ("exp000_0801_v2", 4),
        ("exp000_0803_v2", 5),
    ],
)
def test_exp000_family_golden(
    finalized_exp000: WellboreStudy.Config,
    name: str,
    index: int,
) -> None:
    assert_pprint_golden(
        test_file=__file__,
        name=name,
        config=finalized_exp000.families[index],
    )


def test_exp000_study_golden_beyond_its_families(
    finalized_exp000: WellboreStudy.Config,
) -> None:
    study = finalized_exp000.copy_tree()
    study.families = []
    assert_pprint_golden(test_file=__file__, name="exp000_study", config=study)


def test_production_exp000_model_depth() -> None:
    assert [
        family.training.step.model.unet.backbone.network.blocks_per_stage
        for family in exp000().families
    ] == [
        (3, 3, 27, 3),
    ] * 6


def test_exp000_declares_every_native_family_and_ninety_models() -> None:
    config = exp000()
    assert [
        (family.name, family.cv.folds, family.cv.repeats) for family in config.families
    ] == [
        ("0719_V1", 5, 3),
        ("0724_V1", 5, 3),
        ("0729_V3", 5, 3),
        ("0801_V1", 5, 3),
        ("0801_V2", 5, 3),
        ("0803_V2", 5, 3),
    ]
    assert [family.training.max_epochs for family in config.families] == [
        300,
        300,
        200,
        300,
        300,
        200,
    ]
    base_channels = [
        "tw_gr",
        "gr",
        "gr_abs_diff",
        "gr_isnan_rate",
        "gr_std",
        "gr_first_last_delta",
        "gr_slope",
        "gr_quadratic_a",
        "gr_quadratic_b",
        "gr_quadratic_c",
        "gr_quadratic_rmse",
        "tw_gr_is_nan",
        "tw_tvt_rel",
        "seen_tvt_rel",
        "tw_seen_tvt_abs_diff",
        "z_diff",
    ]
    assert [
        family.training.dataset.loader.dataset.transform.features.channels
        for family in config.families
    ] == [
        base_channels,
        [*base_channels, "geo_tvt_diff"],
        [*base_channels, "pf_particle_density_prob", "pf_tvt_ffbsi"],
        base_channels,
        [*base_channels[:-1], "geo_tvt_diff"],
        [
            *base_channels[:-1],
            "geo_tvt_diff",
            "pf_particle_density_prob",
            "pf_prob_ffbsi",
            "geo_tvt_abs_diff",
            "geo_tvt_rel",
        ],
    ]
    assert [family.training.step.model.dropout for family in config.families] == [
        0.05,
        0.05,
        0.05,
        -1,
        -1,
        -1,
    ]
    assert config.budget is None
    other = exp000()
    config.families[
        0
    ].training.dataset.loader.dataset.transform.features.target_rows = 7
    assert (
        other.families[0].training.dataset.loader.dataset.transform.features.target_rows
        == 10_000
    )


def test_five_minute_candidate_preserves_the_complete_training_recipe() -> None:
    config = exp004()
    assert config.experiment_name == "exp004"
    resolved = config.finalized()
    assert Path(resolved.working_dir).name == "exp004"
    assert Path(resolved.pf_directory).name == "exp004-pf"
    assert config.budget is not None
    assert config.budget.seconds == 285
    assert config.families[0].prepare_particles
    assert len(config.families) == 1
    family = config.families[0]
    reference = exp000().families[-1]
    reference.cv.repeats = 1
    reference.first_n_folds = 1
    reference.prepare_particles = True
    assert family == reference


@pytest.mark.parametrize(
    ("factory", "parent", "change"),
    [
        (exp001, exp000, "family"),
        (exp002, exp001, "repeat"),
        (exp003, exp002, "fold"),
        (exp004, exp003, "wall"),
        (exp005, exp004, "particles"),
        (exp006, exp005, "seeds"),
    ],
)
def test_each_named_fork_has_one_declared_change(
    factory: Callable[[], WellboreStudy.Config],
    parent: Callable[[], WellboreStudy.Config],
    change: str,
) -> None:
    config, reference = factory(), parent()
    reference.experiment_name = config.experiment_name
    if change == "family":
        reference.families = reference.families[-1:]
        reference.results.prediction = SingleFamilyPrediction.Config()
    elif change == "repeat":
        reference.families[0].cv.repeats = 1
    elif change == "fold":
        reference.families[0].first_n_folds = 1
    elif change == "wall":
        reference.budget = WallBudget.Config(seconds=285, reserve_evaluation_sec=30)
        reference.families[0].prepare_particles = True
    elif change == "particles":
        _particle(reference).particles = 64
    else:
        assert change == "seeds"
        _particle(reference).seeds = 8
    assert reference == config, reference.udiff(config, finalize=False, color=False)


def _particle(study: WellboreStudy.Config) -> ParticlePrior.Config:
    particle = study.families[0].particle
    assert particle is not None
    return particle


def test_final_candidate_retains_both_explicit_monte_carlo_reductions() -> None:
    config, reference = exp006(), exp004()
    assert _particle(config).particles == 64
    assert _particle(config).seeds == 8
    reference.experiment_name = "exp006"
    _particle(reference).particles = 64
    _particle(reference).seeds = 8
    assert config == reference


def test_smoke_reduces_geometry_and_retains_every_family_stage() -> None:
    config = exp_smoke().finalized()
    assert [family.name for family in config.families] == [
        "0719_V1",
        "0724_V1",
        "0729_V3",
        "0801_V1",
        "0801_V2",
        "0803_V2",
    ]
    for family in config.families:
        assert family.training.max_epochs == family.cv.repeats == 1
        assert family.cv.folds == 2
        assert family.training.step.model.unet.channels_hidden == (2, 3, 4, 5)
        assert family.training.step.model.unet.backbone.network.blocks_per_stage == (
            1,
            1,
            1,
            1,
        )
        assert family.training.step.model.unet.backbone.checkpoint is None
        assert family.training.step.model.reference_columns == 32
        assert (
            family.training.dataset.loader.dataset.transform.features.target_rows == 64
        )


def test_particle_candidates_require_a_particle_family() -> None:
    with pytest.raises(ValueError, match="particle is not None"):
        candidate_particle(exp000())


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
