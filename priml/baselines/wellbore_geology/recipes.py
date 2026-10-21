"""Declare the six complete winning model, loader and training trees."""

from __future__ import annotations

from dataclasses import field
from pathlib import Path
from typing import TYPE_CHECKING, Self, override

from configgle import Fig, PartialConfig

from priml.baselines.wellbore_geology.data import (
    FixedNanPattern,
    FoldObservations,
    ParticleFeatures,
    PathSimulation,
    SampledNanPattern,
)
from priml.baselines.wellbore_geology.folds import GeographicCV, GeographicPrior
from priml.baselines.wellbore_geology.inference import (
    InferenceRecipe,
    PredictionBuilder,
)
from priml.baselines.wellbore_geology.metrics import PooledRMSE
from priml.baselines.wellbore_geology.particles import ParticleCache, ParticlePrior
from priml.baselines.wellbore_geology.train_loop import WellboreFoldLoop
from priml.optimizers.composite import CompositeOptimizer
from priml.optimizers.lr_schedule import RecursiveCosine
from priml.optimizers.parameter_filter import everything
from priml.runtime import CudnnSingleProcess, SingleProcess
from priml.train.ema import EMA, average_floating_state, average_persistent_buffers
from priml.train.epoch_loop import EarlyStopping, first_periodic_and_last
from priml.train.parallelism import NoParallel


if TYPE_CHECKING:
    import torch

    from priml.baselines.wellbore_geology.custom_types import RowWindow
else:
    from wrapt import lazy_import

    torch = lazy_import("torch")


class Family:
    """Declare one complete family with shared training and reconstruction slots."""

    class Config(Fig["Family"]):
        name: str = "0803_V2"
        """Public family identity in the final ensemble."""

        training: WellboreFoldLoop.Config = field(
            default_factory=WellboreFoldLoop.Config,
        )
        """Actual loop, step, model, optimizer and reconstructible fold data."""

        cv: GeographicCV.Config = field(default_factory=GeographicCV.Config)
        """Complete repeat-major geographic partitions."""

        geography: GeographicPrior.Config = field(
            default_factory=GeographicPrior.Config,
        )
        """Fold-safe relative surfaces and pointwise diagnostics."""

        particle: ParticlePrior.Config | None = None
        """Complete filtering, mixture and backward-trajectory preparation, or None
        for a family without particle planes."""

        prediction: PredictionBuilder.Config = field(
            default_factory=PredictionBuilder.Config,
        )
        """Original-row expansion, smoothing and diagnostic publication."""

        inference: InferenceRecipe.Config = field(
            default_factory=InferenceRecipe.Config,
        )
        """How fresh inference reconstructs and evaluates every selected model."""

        dataset_directory: Path | None = None
        """Staged competition release; None leaves it to the study."""

        pf_directory: Path | None = None
        """Family-isolated particle cache root; None leaves it to the study."""

        prepare_particles: bool = False
        """Generate missing particle records inside the run, charged to any outer
        budget; otherwise a complete prepared cache is required."""

        seed: int = 7
        """Initial family seed before flattened fold offsets."""

        first_n_folds: int | None = None
        """Separate budget experiment's leading folds; None keeps all partitions."""

        @override
        def finalize(self) -> Self:
            features = self.training.dataset.loader.dataset.transform.features
            model = self.training.step.model
            if model.channels_in == -1:
                model.channels_in = len(features.channels)
            if model.reference_columns == -1:
                model.reference_columns = features.reference_columns
            if model.reference_window < 0:
                model.reference_window = features.reference_window
            if model.target_stats is None:
                model.target_stats = features.target_stats
            windows: list[RowWindow] = []
            if self.particle is not None:
                if self.particle.reference_columns == -1:
                    self.particle.reference_columns = features.reference_columns
                if self.particle.reference_window < 0:
                    self.particle.reference_window = features.reference_window
                windows.append(self.particle)
            for expansion in (
                self.prediction.expansion,
                *(
                    metric.expansion
                    for metric in self.training.metrics_eval.values()
                    if isinstance(metric, PooledRMSE.Config)
                ),
            ):
                windows.append(expansion)
                if expansion.target_stats is None:
                    expansion.target_stats = features.target_stats
            for window in windows:
                if window.prefix_rows == -1:
                    window.prefix_rows = features.prefix_rows
                if window.target_rows == -1:
                    window.target_rows = features.target_rows
                if window.downsample == -1:
                    window.downsample = features.downsample
            runtime = self.training.runtime
            if (
                isinstance(runtime, SingleProcess.Config)
                and isinstance(self.inference.parallelism, NoParallel.Config)
                and self.inference.parallelism.device is None
            ):
                self.inference.parallelism.device = runtime.device
            self._push_inputs()
            return super().finalize()

        def _push_inputs(self) -> None:
            """Fill the unset dataset, cache and geography of every fold observation."""
            backbone = self.training.step.model.unet.backbone
            if self.dataset_directory is not None and isinstance(
                backbone.checkpoint,
                str,
            ):
                backbone.checkpoint = Path(
                    backbone.checkpoint.format(
                        dataset_directory=self.dataset_directory,
                    ),
                )
            train = self.training.dataset.loader.dataset.observations
            for observations in (train, self.training.dataset.validation):
                if not isinstance(observations, FoldObservations.Config):
                    continue
                if observations.geography is None:
                    observations.geography = self.geography.copy_tree()
                if (
                    observations.directory is None
                    and self.dataset_directory is not None
                ):
                    observations.directory = self.dataset_directory / "train"
                if (
                    isinstance(observations.particle, ParticleCache.Config)
                    and observations.particle.directory is None
                    and self.pf_directory is not None
                ):
                    observations.particle.directory = self.pf_directory / "train"

    def __init__(self, config: Config) -> None:
        if config.first_n_folds is not None and config.first_n_folds < 1:
            raise ValueError("first_n_folds must be positive.")
        features = config.training.dataset.loader.dataset.transform.features
        if (config.particle is None) != (features.particle is None):
            raise ValueError(
                "A family declares particle priors exactly when its features use them.",
            )
        self.config = config


def family_0719_v1() -> Family.Config:
    """Declare the first base-image model and its complete earlier simulation.

    Returns:
      family: Complete independently native first published configuration.

    """
    cfg = _base_family()
    cfg.name = "0719_V1"
    cfg.training.step.model.dropout = 0.05
    cfg.training.dataset.validation_batch_size = 32
    _stopping(cfg).patience = 30
    paths = _paths(cfg)
    paths.shift_probability = 0.75
    paths.shift_nan_pattern = FixedNanPattern.Config(shift=False)
    paths.gamma_noise_shift_probability = 0.1
    paths.reference_jitter.probability = 0.0
    paths.reference_transform.gain_std = 0.03
    paths.reference_drift_probability = 0.0
    cfg.prediction.smoothing.window = 1025
    cfg.prediction.smoothing.blend = 0.6828376753426914
    cfg.prediction.diagnostic_names = ()
    return cfg


def family_0724_v1() -> Family.Config:
    """Declare the base-image model with its added geographic derivative plane.

    Returns:
      family: Complete independently native geographic-plane configuration.

    """
    cfg = family_0719_v1()
    cfg.name = "0724_V1"
    cfg.training.dataset.validation_batch_size = 16
    cfg.training.dataset.loader.dataset.transform.features.channels += ["geo_tvt_diff"]
    paths = _paths(cfg)
    nan_pattern = paths.shift_nan_pattern = SampledNanPattern.Config()
    nan_pattern.weights = {False: 1.0}
    paths.gamma_noise_shift_probability = 0.0
    cfg.prediction.diagnostic_names = (
        "geo_nbr_distance",
        "geo_radial_extrap_score",
        "geo_nbr_path_alignment",
    )
    return cfg


def family_0729_v3() -> Family.Config:
    """Declare the particle and backward-path model with prefix simulation.

    Returns:
      family: Complete independently native particle-path configuration.

    """
    cfg = family_0724_v1()
    cfg.name = "0729_V3"
    cfg.training.max_epochs = 200
    _stopping(cfg).min_epochs = 150
    _stopping(cfg).patience = 50
    cfg.cv.split_seed = 2233
    features = cfg.training.dataset.loader.dataset.transform.features
    features.channels = [
        *features.channels[:-1],
        "pf_particle_density_prob",
        "pf_tvt_ffbsi",
    ]
    _use_particle_priors(cfg)
    paths = _paths(cfg)
    paths.prefix_sample_rows = 256
    cfg.prediction.diagnostic_names += ("geo_prefix_weight_ratio",)
    return cfg


def family_0801_v1() -> Family.Config:
    """Declare the later base-image model with coherent reference corruption.

    Returns:
      family: Complete independently native later simulation configuration.

    """
    cfg = _base_family()
    cfg.name = "0801_V1"
    cfg.cv.split_seed = 42
    return cfg


def family_0801_v2() -> Family.Config:
    """Replace the later vertical derivative plane with the geographic derivative."""
    cfg = family_0801_v1()
    cfg.name = "0801_V2"
    features = cfg.training.dataset.loader.dataset.transform.features
    features.channels = [*features.channels[:-1], "geo_tvt_diff"]
    return cfg


def family_0803_v2() -> Family.Config:
    """Add complete particle and geographic image planes to the later model.

    Returns:
      family: Complete independently native latest published configuration.

    """
    cfg = family_0801_v2()
    cfg.name = "0803_V2"
    cfg.training.max_epochs = 200
    features = cfg.training.dataset.loader.dataset.transform.features
    features.channels += [
        "pf_particle_density_prob",
        "pf_prob_ffbsi",
        "geo_tvt_abs_diff",
        "geo_tvt_rel",
    ]
    _use_particle_priors(cfg)
    return cfg


def _base_family() -> Family.Config:
    """Declare the optimization, schedule and selection recipe every family shares."""
    cfg = Family.Config()
    step = cfg.training.step
    adamw = PartialConfig(torch.optim.AdamW)
    adamw.lr = 0.0002
    adamw.weight_decay = 0.02
    step.optimizer = CompositeOptimizer.Config()
    step.optimizer.optimizers = [adamw]
    step.lr_schedule = RecursiveCosine.Config()
    step.lr_schedule.eta_min = 5e-7
    step.ema = EMA.Config()
    step.ema.decay = 0.99
    step.ema.select = everything
    step.ema.update_fn = average_floating_state
    step.ema.buffer_update_fn = average_persistent_buffers
    step.gradient_clip_norm = 10.0
    step.dtype_autocast = torch.bfloat16
    step.autocast_cache_enabled = True
    step.compile = None
    step.model.unet.backbone.checkpoint = (
        "{dataset_directory}/initialization/convnext_small.safetensors"
    )
    runtime = CudnnSingleProcess.Config()
    runtime.float32_matmul_precision = "high"
    runtime.cudnn_benchmark = True
    runtime.cudnn_deterministic = False
    runtime.cudnn_allow_tf32 = True
    cfg.training.runtime = runtime
    cfg.training.model_torch_seed = torch.random.manual_seed
    cfg.training.max_epochs = 300
    cfg.training.eval_epochs = PartialConfig(first_periodic_and_last, every=5)
    stopping = EarlyStopping.Config()
    stopping.metric = "rmse"
    stopping.patience = 50
    stopping.min_epochs = 170
    cfg.training.early_stopping = stopping
    return cfg


def _stopping(cfg: Family.Config) -> EarlyStopping.Config:
    """Return the family's early-stopping rule, which every published family declares."""
    stopping = cfg.training.early_stopping
    if not isinstance(stopping, EarlyStopping.Config):
        raise TypeError("Every published family declares early stopping.")
    return stopping


def _paths(cfg: Family.Config) -> PathSimulation.Config:
    """Return the family's path simulation, which every published family declares."""
    paths = cfg.training.dataset.loader.dataset.transform.paths
    if not isinstance(paths, PathSimulation.Config):
        raise TypeError("Every published family declares a path simulation.")
    return paths


def _use_particle_priors(cfg: Family.Config) -> None:
    """Enable the particle planes, their cache reads and the producing prior."""
    features = cfg.training.dataset.loader.dataset.transform.features
    features.particle = ParticleFeatures.Config()
    features.particle.path = "pf_tvt_rel_pred_ffbsi"
    cfg.particle = ParticlePrior.Config()
    for observations in (
        cfg.training.dataset.loader.dataset.observations,
        cfg.training.dataset.validation,
    ):
        assert isinstance(observations, FoldObservations.Config)
        observations.particle = ParticleCache.Config()
