"""Require the complete native public recipe and explicit independent deltas."""

from collections.abc import Callable
from pathlib import Path

import pytest

from priml.baselines.wellbore_geology.data import (
    FixedNanPattern,
    FoldObservations,
    ParticleFeatures,
    SampledNanPattern,
)
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
    shrink_model,
)
from priml.baselines.wellbore_geology.folds import GeographicCV
from priml.baselines.wellbore_geology.inference import (
    SingleFamilyPrediction,
    SurfaceSmoothing,
)
from priml.baselines.wellbore_geology.particles import (
    ParticleCache,
    ParticlePrior,
)
from priml.baselines.wellbore_geology.pipeline import WellboreStudy
from priml.baselines.wellbore_geology.recipes import (
    Family,
    _paths,
    family_0719_v1,
    family_0724_v1,
    family_0729_v3,
    family_0801_v1,
    family_0801_v2,
    family_0803_v2,
)
from priml.testing.golden import assert_pprint_golden
from priml.train.budget import WallBudget
from priml.train.epoch_loop import EarlyStopping


@pytest.fixture(scope="module")
def finalized_exp000() -> WellboreStudy.Config:
    """Finalize the study with its nexus family alone; each family finalizes apart."""
    study = exp000()
    study.families = study.families[-1:]
    return study.finalized()


def test_exp000_nexus_family_golden(finalized_exp000: WellboreStudy.Config) -> None:
    family = finalized_exp000.families[-1]
    assert family.name == "0803_V2"
    assert_pprint_golden(test_file=__file__, name="exp000_0803_v2", config=family)


def test_exp000_study_golden_beyond_its_families(
    finalized_exp000: WellboreStudy.Config,
) -> None:
    study = finalized_exp000.copy_tree()
    study.families = []
    assert_pprint_golden(test_file=__file__, name="exp000_study", config=study)


@pytest.mark.parametrize(
    ("factory", "parent", "change"),
    [
        (family_0801_v2, family_0801_v1, "geographic_plane"),
        (family_0803_v2, family_0801_v2, "particle_planes"),
        (family_0719_v1, family_0801_v1, "earlier_simulation"),
        (family_0724_v1, family_0719_v1, "geographic_derivative"),
        (family_0729_v3, family_0724_v1, "particle_path"),
    ],
)
def test_each_family_differs_from_its_parent_by_its_declared_change(
    factory: Callable[[], Family.Config],
    parent: Callable[[], Family.Config],
    change: str,
) -> None:
    config, reference = factory(), parent()
    reference.name = config.name
    features = reference.training.dataset.loader.dataset.transform.features
    paths = _paths(reference)
    stopping = reference.training.early_stopping
    assert isinstance(stopping, EarlyStopping.Config)
    if change == "geographic_plane":
        features.channels = [*features.channels[:-1], "geo_tvt_diff"]
    elif change == "particle_planes":
        reference.training.max_epochs = 200
        features.channels += [
            "pf_particle_density_prob",
            "pf_prob_ffbsi",
            "geo_tvt_abs_diff",
            "geo_tvt_rel",
        ]
        _use_particles(reference)
    elif change == "earlier_simulation":
        reference.cv.split_seed = GeographicCV.Config().split_seed
        reference.training.step.model.dropout = 0.05
        reference.training.dataset.validation_batch_size = 32
        stopping.patience = 30
        paths.shift_probability = 0.75
        paths.shift_nan_pattern = FixedNanPattern.Config(shift=False)
        paths.gamma_noise_shift_probability = 0.1
        paths.reference_jitter.probability = 0.0
        paths.reference_transform.gain_std = 0.03
        paths.reference_drift_probability = 0.0
        reference.prediction.smoothing = SurfaceSmoothing.Config(
            window=1025,
            blend=0.6828376753426914,
        )
        reference.prediction.diagnostic_names = ()
    elif change == "geographic_derivative":
        reference.training.dataset.validation_batch_size = 16
        features.channels += ["geo_tvt_diff"]
        paths.shift_nan_pattern = SampledNanPattern.Config(weights={False: 1.0})
        paths.gamma_noise_shift_probability = 0.0
        reference.prediction.diagnostic_names = (
            "geo_nbr_distance",
            "geo_radial_extrap_score",
            "geo_nbr_path_alignment",
        )
    else:
        assert change == "particle_path"
        reference.training.max_epochs = 200
        stopping.min_epochs = 150
        stopping.patience = 50
        reference.cv.split_seed = 2233
        features.channels = [
            *features.channels[:-1],
            "pf_particle_density_prob",
            "pf_tvt_ffbsi",
        ]
        _use_particles(reference)
        paths.prefix_sample_rows = 256
        reference.prediction.diagnostic_names += ("geo_prefix_weight_ratio",)
    assert reference == config, reference.udiff(config, finalize=False, color=False)


def _use_particles(family: Family.Config) -> None:
    features = family.training.dataset.loader.dataset.transform.features
    features.particle = ParticleFeatures.Config(path="pf_tvt_rel_pred_ffbsi")
    family.particle = ParticlePrior.Config()
    for observations in (
        family.training.dataset.loader.dataset.observations,
        family.training.dataset.validation,
    ):
        assert isinstance(observations, FoldObservations.Config)
        observations.particle = ParticleCache.Config()


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
        (exp_smoke, exp000, "smoke"),
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
    elif change == "seeds":
        _particle(reference).seeds = 8
    else:
        assert change == "smoke"
        for family in reference.families:
            family.training.max_epochs = family.cv.repeats = 1
            family.cv.folds = 2
            shrink_model(family.training.step.model)
            features = family.training.dataset.loader.dataset.transform.features
            features.target_rows = 64
            features.reference_columns = 32
            if family.particle is not None:
                family.particle.particles = 3
                family.particle.seeds = 5
                family.particle.paths = 2
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
        model = family.training.step.model
        assert model.channels_hidden == 2
        assert model.kernel_size == 1
        assert model.unet.channels_hidden == (2, 3, 4, 5)
        assert model.unet.backbone.network.blocks_per_stage == (1, 1, 1, 1)
        stages = model.unet.backbone.network.stage
        assert isinstance(stages, list)
        for stage in stages:
            assert isinstance(stage.block, list)
            for block in stage.block:
                assert block.conv_dw.kernel_size == 3
                assert block.conv_dw.padding == 1
        assert model.unet.backbone.checkpoint is None
        assert model.reference_columns == 32
        assert (
            family.training.dataset.loader.dataset.transform.features.target_rows == 64
        )
        if family.particle is not None:
            assert family.particle.particles == 3
            assert family.particle.seeds == 5
            assert family.particle.paths == 2


def test_particle_candidates_require_a_particle_family() -> None:
    with pytest.raises(ValueError, match="particle is not None"):
        candidate_particle(exp000())


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
