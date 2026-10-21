"""Native complete public-best wellbore study: family training, fresh inference and publication."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, Self, cast, override

import argparse
import gc
import json
import logging
import math
import platform
import subprocess
import sys

from configgle import Fig, Makeable
from treekle import json_loads, to_plain

import numpy as np

from priml.baselines.wellbore_geology.data import (
    FoldObservations,
    discover_wells,
    load_split,
    read_centroids,
)
from priml.baselines.wellbore_geology.inference import (
    FamilyArtifacts,
    PublishResults,
    write_table,
)
from priml.baselines.wellbore_geology.particles import (
    ParticleCache,
    check_cold_cache,
    prepare_particles,
    validate_particles,
)
from priml.baselines.wellbore_geology.recipes import Family
from priml.baselines.wellbore_geology.train_loop import (
    WellboreFoldLoop,
    require_fresh_runtime,
)
from priml.baselines.wellbore_geology.train_step import WellboreTrainStep
from priml.paths import resolve_working_dir, validated_output_path
from priml.runtime import SingleProcess
from priml.train.budget import WallBudget


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from numpy.typing import NDArray
    from torch import Tensor

    import torch

    from priml.baselines.wellbore_geology.data import RawWell
    from priml.baselines.wellbore_geology.inference import PredictionTable
else:
    from wrapt import lazy_import

    torch = lazy_import("torch")


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True, kw_only=True)
class TrainedFamily:
    """Complete selected model states and repeated out-of-fold artifact."""

    checkpoints: tuple[str, ...]
    oof: Path


def train_family(
    family: Family.Config,
    *,
    raw: Mapping[str, RawWell],
    output_directory: Path,
    budget: WallBudget | None = None,
) -> TrainedFamily:
    """Run every declared fold through its reconstructible shared loop tree.

    Args:
      family: Complete family configuration naming its dataset and cache roots.
      raw: Every complete training observation in original well order.
      output_directory: External selected-checkpoint and OOF directory.
      budget: Optional separate experiment's cold wall allowance.

    Returns:
      trained: Every selected complete CPU state and averaged original-row OOF file.

    Raises:
      ValueError: A dataset directory or a nonempty validation fold is missing.

    """
    require_fresh_runtime()
    family = family.finalized()
    family.make()
    if family.dataset_directory is None:
        raise ValueError("Training requires the family's dataset directory.")
    names = tuple(raw)
    coordinates = read_centroids(
        family.dataset_directory / "train_png_typewell_map.csv",
        well_ids=names,
    )
    folds = family.cv.make().split(coordinates)
    if family.first_n_folds is not None:
        folds = folds[: family.first_n_folds]
    torch.manual_seed(family.seed)
    torch.cuda.manual_seed_all(family.seed)
    if family.particle is not None:
        if family.pf_directory is None:
            raise ValueError("Particle priors require the family's cache directory.")
        if family.prepare_particles:
            prepare_particles(
                raw,
                config=family.particle,
                directory=family.pf_directory / "train",
                has_target=True,
                budget=budget,
            )
        else:
            validate_particles(
                names,
                config=family.particle,
                directory=family.pf_directory / "train",
            )
    builder = family.prediction.make()
    tables: list[PredictionTable] = []
    checkpoints: list[str] = []
    summaries: list[dict[str, object]] = []
    runtime = family.training.runtime
    assert isinstance(runtime, SingleProcess.Config)
    if runtime.device is None:
        raise ValueError("Training requires the family's runtime device.")
    device = str(runtime.device)
    for index, (training_rows, validation_rows) in enumerate(folds):
        training = tuple(names[int(position)] for position in training_rows.flat)
        validation = tuple(names[int(position)] for position in validation_rows.flat)
        if not validation:
            raise ValueError("Native OOF training requires a nonempty validation fold.")
        configuration = fold_settings(
            family,
            index=index,
            training=training,
            validation=validation,
        )
        configuration.evaluate_truncated = budget is not None
        configuration.max_time = min(
            configuration.max_time,
            budget.training_seconds() if budget is not None else math.inf,
        )
        fold_name = fold_identity(family, index=index)
        loop = configuration.finalized().make()
        try:
            if budget is not None:
                loop.max_time = min(loop.max_time, budget.training_seconds())
            loop.train()
        finally:
            loop.close()
        step = loop.native_step
        if budget is not None and step.global_step == 0:
            raise TimeoutError("The outer budget expired before an optimizer update.")
        selected = (
            loop.best_state
            if loop.best_state is not None
            else {
                name: value.detach().cpu().clone()
                for name, value in step.model.state_dict().items()
            }
        )
        step.model.load_state_dict(selected)
        step.model.cpu()
        logger.info(
            "%s: priml best_epoch=%s, best_rmse=%.4f",
            fold_name,
            loop.best_epoch,
            loop.best_score,
        )
        filename = f"{fold_name}.pt"
        torch.save(step.model.state_dict(), output_directory / filename)
        checkpoints.append(filename)
        if loop.fold_data.validation is None:
            raise ValueError("Expected loop.fold_data.validation is not None.")
        prediction, metadata = step.predict(
            loop.fold_data.validation,
            model=step.model,
            move_cpu=True,
        )
        tables.append(
            builder.build(
                prediction,
                metadata=metadata,
                include_target=True,
                prior=loop.fold_data.held_out_surfaces,
                curves=raw,
                summarize=False,
            ),
        )
        summaries.append(
            {
                "fold": fold_name,
                "best_score": loop.best_score,
                "best_epoch": loop.best_epoch,
                "validation": loop.val_history,
            },
        )
        del loop, step
        release_accelerator(device)
    oof = builder.average_repeats(tables, curves=raw, repeats=family.cv.repeats)
    if family.prediction.diagnostic_names:
        oof = builder.summarize_distance(oof)
    path = output_directory / "oof_df.pqt"
    write_table(path, oof)
    (output_directory / "checkpoints.json").write_text(
        json.dumps(to_plain(checkpoints)),
    )
    (output_directory / "folds.json").write_text(
        json.dumps(to_plain(summaries), indent=2),
    )
    return TrainedFamily(checkpoints=tuple(checkpoints), oof=path)


def fold_settings(
    family: Family.Config,
    *,
    index: int,
    training: Sequence[str],
    validation: Sequence[str],
) -> WellboreFoldLoop.Config:
    """Bind a fold's identity and membership to a finalized copy of the family loop tree.

    Args:
      family: Complete finalized native family tree.
      index: Flattened repeat-major fold position, selecting its random streams.
      training: Training-fold wells, which also support the geographic priors.
      validation: Held-out wells scored against the training-fold support.

    Returns:
      settings: Reconstructible loop configuration for exactly this fold.

    """
    config = family.training.copy_tree()
    config.fold = index
    if config.base_seed == -1:
        config.base_seed = family.seed
    train = config.dataset.loader.dataset.observations
    held_out = config.dataset.validation
    assert isinstance(train, FoldObservations.Config)
    assert isinstance(held_out, FoldObservations.Config)
    train.well_ids = train.support_well_ids = tuple(training)
    held_out.well_ids = tuple(validation)
    held_out.support_well_ids = tuple(training)
    return config.finalize()


def fold_identity(family: Family.Config, *, index: int) -> str:
    """Name every complete repeat-major fold in public checkpoint order."""
    return (
        f"repeat{index // family.cv.folds}_fold{index % family.cv.folds}"
        if family.cv.repeats > 1
        else f"fold{index}"
    )


def release_accelerator(device: str) -> None:
    """Finish queued GPU work and release cached fold allocations after deletion."""
    if torch.device(device).type == "cuda":
        gc.collect()
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()


_THIS: Final = Path(__file__).resolve()


class FamilyInference(Protocol):
    """Complete native inference at the external interpreter boundary."""

    def predict(self, *, configuration: Path, directory: Path) -> Path:
        """Return the complete FP32 original-row inference artifact."""
        ...


class FreshInference:
    """Launch native complete-model prediction with fresh Torch backend defaults."""

    class Config(Fig["FreshInference"]):
        python_executable: Path | None = None
        """Interpreter reconstructing the complete native model states; None uses the
        running one."""

        working_directory: Path | None = None
        """Repository root containing the native inference module; None uses the
        checkout the running code was loaded from."""

    def __init__(self, config: Config) -> None:
        self.config = config

    def predict(self, *, configuration: Path, directory: Path) -> Path:
        """Predict all test rows in a fresh FP32 interpreter.

        Args:
          configuration: Complete native family configuration serialized by training.
          directory: Family run directory with every selected CPU checkpoint.

        Returns:
          path: Complete native inference original-row Arrow artifact.

        """
        output = directory / "inference"
        subprocess.run(  # noqa: S603 -- fixed native interpreter/module argv, no shell.
            [
                str(self.config.python_executable or sys.executable),
                "-m",
                "priml",
                "priml.baselines.wellbore_geology.pipeline.family_prediction",
                "--configuration",
                str(configuration),
                "--directory",
                str(directory),
                "--output",
                str(output),
            ],
            cwd=self.config.working_directory or _THIS.parents[4],
            check=True,
        )
        return output / "submission_details.pqt"


def predict_family(family: Family.Config, *, directory: Path, output: Path) -> Path:
    """Average every selected model in float32 bins before native row expansion.

    Args:
      family: Complete native family with its dataset and cache directories.
      directory: Family run directory holding the ordered checkpoint manifest.
      output: New external fresh-inference artifact directory.

    Returns:
      path: Complete test original-row Arrow table.

    Raises:
      ValueError: A checkpoint, dataset directory or runtime device is missing.

    """
    family = family.finalized()
    dataset_directory = family.dataset_directory
    runtime = family.training.runtime
    assert isinstance(runtime, SingleProcess.Config)
    if dataset_directory is None or runtime.device is None:
        raise ValueError(
            "Inference requires the family's dataset directory and device.",
        )
    device = torch.device(runtime.device)
    checkpoints = json_loads((directory / "checkpoints.json").read_text(), list[str])
    output = validated_output_path(
        output,
        protected=(
            dataset_directory,
            *((family.pf_directory,) if family.pf_directory is not None else ()),
            directory / "checkpoints.json",
            *(directory / name for name in checkpoints),
        ),
    )
    if not checkpoints:
        raise ValueError("Fresh inference requires every selected checkpoint.")
    output.mkdir(parents=True, exist_ok=False)
    test = load_split(dataset_directory / "test")
    recipe = family.inference
    observations = FoldObservations.Config()
    observations.directory = dataset_directory / "test"
    observations.support_directory = dataset_directory / "train"
    observations.well_ids = tuple(test)
    observations.support_well_ids = discover_wells(observations.support_directory)
    observations.geography = family.geography.copy_tree()
    observations.leave_one_out = False
    if family.particle is not None:
        if family.pf_directory is None:
            raise ValueError("Particle priors require the family's cache directory.")
        if family.prepare_particles:
            prepare_particles(
                test,
                config=family.particle,
                directory=family.pf_directory / "test",
                has_target=False,
            )
        else:
            validate_particles(
                test,
                config=family.particle,
                directory=family.pf_directory / "test",
            )
        observations.particle = ParticleCache.Config()
        observations.particle.directory = family.pf_directory / "test"
    config = family.training.dataset.loader.copy_tree()
    config.pin_memory = device.type == "cuda"
    config.seed = family.seed + recipe.seed_offset
    config.batch_size = (
        family.training.dataset.validation_batch_size
        if recipe.batch_size == -1
        else recipe.batch_size
    )
    config.shuffle = config.drop_last = False
    config.dataset.training = False
    config.dataset.transform.paths = config.dataset.transform.images = None
    config.dataset.streams.seed = family.seed
    config.dataset.observations = observations
    loader = config.make()
    assert isinstance(loader.dataset.observations, FoldObservations)
    prior = loader.dataset.observations.surfaces
    model_config = family.training.step.model.copy_tree()
    model_config.unet.backbone.checkpoint = None
    step_config = family.training.step.copy_tree()
    step_config.dtype_autocast = recipe.dtype_autocast
    step_config.ema = recipe.ema.copy_tree()
    step_config.parallelism = recipe.parallelism.copy_tree()
    total: NDArray[np.float32] | None = None
    metadata: list[dict[str, object]] = []
    for filename in checkpoints:
        step_config.model = model_config
        step = step_config.make()
        assert isinstance(step, WellboreTrainStep)
        model = step.model
        state = cast(
            "dict[str, Tensor]",
            torch.load(directory / filename, map_location="cpu", weights_only=True),
        )
        model.load_state_dict(state, strict=True)
        prediction, metadata = step.predict(loader.loader, model=model, move_cpu=True)
        total = accumulate_prediction(total, prediction, count=len(checkpoints))
        del model, step
    assert isinstance(total, np.ndarray)
    table = family.prediction.make().build(
        total,
        metadata=metadata,
        include_target=False,
        prior=prior,
        curves=test,
        summarize=True,
    )
    path = output / "submission_details.pqt"
    write_table(path, table)
    return path


def accumulate_prediction(
    total: NDArray[np.float32] | None,
    prediction: NDArray[np.float32],
    *,
    count: int,
) -> NDArray[np.float32]:
    """Add one model's float32 contribution in manifest insertion order.

    Args:
      total: Owned complete running bin sum, or None before the first model.
      prediction: Complete float32 bins from one selected model.
      count: Complete number of selected family checkpoints.

    Returns:
      total: Complete float32 running sum after this model contribution.

    """
    if count < 1:
        raise ValueError("The complete model count must be positive.")
    if total is None:
        total = np.zeros_like(prediction)
    total += prediction / count
    return total


class FamilyPrediction:
    """Load a trusted native family configuration and run fresh FP32 prediction."""

    class Config(Fig["FamilyPrediction"]):
        """The command itself declares nothing; its paths are command-line flags."""

    def __init__(self, config: Config) -> None:
        self.config = config

    def run(self, *args: str) -> None:
        """Predict one trained family from its checkpoint directory.

        Args:
          *args: ``--configuration``, ``--directory`` and ``--output`` paths.

        """
        parser = argparse.ArgumentParser(description=(self.__doc__ or "").strip())
        _add_prediction_arguments(parser)
        flags = cast("Flags", parser.parse_args(args))
        family = Family.Config.deserialize(json_loads(flags.configuration.read_text()))
        predict_family(family, directory=flags.directory, output=flags.output)


def family_prediction() -> FamilyPrediction.Config:
    """Run fresh FP32 prediction of one trained family (``uv run -m priml``)."""
    return FamilyPrediction.Config()


class Flags(Protocol):
    """Explicit native checkpoint, input and inference output paths."""

    configuration: Path
    directory: Path
    output: Path


class FamilyRunner(Protocol):
    """Own one complete family's numerical training and inference lifecycle."""

    def run(
        self,
        family: Family.Config,
        *,
        output_directory: Path,
        budget: WallBudget | None,
    ) -> FamilyArtifacts:
        """Return complete native training and fresh inference artifacts."""
        ...


class NativeFamilyRunner:
    """Train every native fold and launch its final inference in a new interpreter."""

    class Config(Fig["NativeFamilyRunner"]):
        inference: Makeable[FamilyInference] = field(
            default_factory=FreshInference.Config,
        )
        """Fresh interpreter boundary for the published FP32 correction."""

    def __init__(self, config: Config) -> None:
        self.config = config

    def run(
        self,
        family: Family.Config,
        *,
        output_directory: Path,
        budget: WallBudget | None,
    ) -> FamilyArtifacts:
        """Complete one family's native training, OOF aggregation and fresh inference.

        Args:
          family: Complete native public family configuration with its directories.
          output_directory: New external family checkpoint/artifact directory.
          budget: Separate experiment's optional remaining wall allowance.

        Returns:
          artifacts: Complete averaged OOF and fresh FP32 test original-row tables.

        """
        if family.dataset_directory is None:
            raise ValueError("A native family requires its dataset directory.")
        output_directory.mkdir(parents=True, exist_ok=False)
        configuration = output_directory / "family.json"
        configuration.write_text(json.dumps(to_plain(family), indent=2))
        trained = train_family(
            family,
            raw=load_split(family.dataset_directory / "train"),
            output_directory=output_directory,
            budget=budget,
        )
        test = self.config.inference.make().predict(
            configuration=configuration,
            directory=output_directory,
        )
        return FamilyArtifacts(oof=trained.oof, test=test)


class WellboreStudy:
    """Train the complete native study and publish every official target row."""

    class Config(Fig["WellboreStudy"]):
        study_name: str = "wellbore_geology"
        """Recorded study identity."""

        experiment_name: str = "exp000"
        """Recorded experiment identity."""

        reference_revision: str = "fade6112ffebd9e27c08ac7f053e8d4c3d5c319a"
        """Public scientific recipe revision; no original runtime is loaded."""

        dataset_directory: Path | str = "/datasets/rogii-ruby-fade6112"
        """Complete staged CSVs, geographic map and official sample submission; a
        string is a logical path beneath base_dir, a Path stays literal."""

        base_dir: Path | None = Path("/opt/scratch")
        """Resource root for declared run and rebuildable-cache templates."""

        working_dir: Path | str = "/runs/{study_name}/{experiment_name}"
        """New run directory; an explicit Path is retained literally."""

        pf_directory: Path | str = "/caches/{study_name}/{experiment_name}-pf"
        """Rebuildable native particle cache root with a directory per family."""

        device: str = "cuda"
        """Training, validation and fresh inference device."""

        families: list[Family.Config] = field(default_factory=list[Family.Config])
        """Complete native public families in execution and ensemble order."""

        runner: Makeable[FamilyRunner] = field(
            default_factory=NativeFamilyRunner.Config,
        )
        """Complete per-family training and inference lifecycle."""

        results: PublishResults.Config = field(default_factory=PublishResults.Config)
        """Complete native ensemble, pooled-row metric and aligned CSV publication."""

        budget: WallBudget.Config | None = None
        """Separate experiment's cold wall budget; None leaves exp000 uncapped."""

        @override
        def finalize(self) -> Self:
            names = {
                "study_name": self.study_name,
                "experiment_name": self.experiment_name,
            }
            if isinstance(self.working_dir, str):
                self.working_dir = resolve_working_dir(
                    self.base_dir,
                    self.working_dir.format(**names),
                )
            if isinstance(self.pf_directory, str):
                self.pf_directory = resolve_working_dir(
                    self.base_dir,
                    self.pf_directory.format(**names),
                )
            if isinstance(self.dataset_directory, str):
                self.dataset_directory = resolve_working_dir(
                    self.base_dir,
                    self.dataset_directory.format(**names),
                )
            for family in self.families:
                self._push_into(family)
            return super().finalize()

        def _push_into(self, family: Family.Config) -> None:
            """Fill the family's unset directories, run identity and device."""
            assert isinstance(self.dataset_directory, Path)
            assert isinstance(self.pf_directory, Path)
            assert isinstance(self.working_dir, Path)
            if family.dataset_directory is None:
                family.dataset_directory = self.dataset_directory
            if family.pf_directory is None:
                family.pf_directory = self.pf_directory / family.name
            training = family.training
            if training.study_name == "":
                training.study_name = self.study_name
            if training.experiment_name == "":
                training.experiment_name = self.experiment_name
            training.base_dir = self.base_dir
            if isinstance(training.working_dir, str):
                training.working_dir = self.working_dir / family.name
            runtime = training.runtime
            if isinstance(runtime, SingleProcess.Config) and runtime.device is None:
                runtime.device = self.device

    def __init__(self, config: Config) -> None:
        self.config = config

    def run(self, *args: str) -> None:
        """Execute the complete study through the standard Priml launcher.

        Args:
          *args: --pprint inspects the complete configuration without data access.

        """
        parser = argparse.ArgumentParser(description=type(self).__doc__)
        _add_arguments(parser)
        flags = cast("RunFlags", parser.parse_args(args))
        if flags.pprint:
            self.config.pprint(hide_default_values=False)
            return
        require_fresh_runtime()
        cfg = self.config
        if not cfg.families:
            raise ValueError("A wellbore study requires at least one complete family.")
        wall = (cfg.budget if cfg.budget is not None else WallBudget.Config()).make()
        budget = wall if cfg.budget is not None else None
        assert isinstance(cfg.dataset_directory, Path)
        assert isinstance(cfg.pf_directory, Path)
        if budget is not None:
            for family in cfg.families:
                if family.prepare_particles and family.pf_directory is not None:
                    check_cold_cache(family.pf_directory, required=True)
        training = discover_wells(cfg.dataset_directory / "train")
        test = discover_wells(cfg.dataset_directory / "test")
        output = validated_output_path(
            cfg.working_dir,
            protected=(cfg.dataset_directory, cfg.pf_directory),
        )
        output.mkdir(parents=True)
        if wall.receipt_path is None:
            wall.receipt_path = output / "run.json"
        (output / "config.txt").write_text(cfg.pformat(hide_default_values=False))
        wall.metadata.update(
            {
                "schema_version": 1,
                "study": cfg.study_name,
                "experiment": cfg.experiment_name,
                "reference_revision": cfg.reference_revision,
                "competition": "rogii-wellbore-geology-prediction",
                "metric": "rmse",
                "metric_direction": "minimize",
                "device": cfg.device,
                "host": platform.node(),
                "train_wells": len(training),
                "test_wells": len(test),
                "outer_budget_sec": budget.seconds if budget is not None else None,
                "timing_scope": "Study.run including cold preprocessing, complete configured training, fresh inference and publication; excludes downloads and launcher imports",
            },
        )
        with wall:
            runner = cfg.runner.make()
            files = {
                family.name: runner.run(
                    family,
                    output_directory=output / family.name,
                    budget=budget,
                )
                for family in cfg.families
            }
            wall.metadata["metrics"] = cfg.results.make().publish(
                output,
                files=files,
                sample_path=cfg.dataset_directory / "sample_submission.csv",
            )


class RunFlags(Protocol):
    """Arguments forwarded by the Priml launcher."""

    pprint: bool


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--pprint",
        action="store_true",
        help="Print the complete finalized configuration without running.",
    )


def _add_prediction_arguments(parser: argparse.ArgumentParser) -> None:
    for name in ("configuration", "directory", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
