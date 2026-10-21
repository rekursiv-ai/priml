"""Shared fixtures and replay tools for the wellbore tests."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import field
from typing import TYPE_CHECKING, cast, override

import csv

from configgle import Makes, PartialConfig
from torch import Tensor, nn

import numpy as np
import torch

from priml.baselines.wellbore_geology.custom_types import (
    BatchLoader,
    InitializableEMA,
)
from priml.baselines.wellbore_geology.experiments import shrink_model
from priml.baselines.wellbore_geology.loss import WellboreLoss
from priml.baselines.wellbore_geology.metrics import PooledRMSE
from priml.baselines.wellbore_geology.model import SequenceModel
from priml.baselines.wellbore_geology.recipes import (
    Family,
    _paths,
    family_0719_v1,
    family_0803_v2,
)
from priml.baselines.wellbore_geology.train_loop import FoldData, WellboreFoldLoop
from priml.optimizers.composite import CompositeOptimizer
from priml.optimizers.lr_schedule import RecursiveCosine
from priml.runtime import SingleProcess
from priml.testing.golden import PinnedDropout, read_tensors
from priml.timer import CheckpointableStepTimer
from priml.train.ema import EMA
from priml.train.epoch_loop import EarlyStopping
from priml.train.parallelism import NoParallel


if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray
    from torch._ops import OpOverload

    from priml.loss.custom_types import LossOutput


def tiny_predictions() -> dict[str, NDArray[np.float32]]:
    """Return complete family vectors exposing float32 accumulation order.

    Returns:
      predictions: All six complete float32 family vectors.

    """
    return {
        "0719_V1": np.asarray([1e8, -0.0, 1.25], dtype=np.float32),
        "0724_V1": np.asarray([1.0, 2.5, 1e8], dtype=np.float32),
        "0729_V3": np.asarray([8.0, -1.25, -1e8], dtype=np.float32),
        "0801_V1": np.asarray([-1e8, 5.0, 6.25], dtype=np.float32),
        "0801_V2": np.asarray([2.0, 4.5, -1e8], dtype=np.float32),
        "0803_V2": np.asarray([4.0, 6.5, 9.5], dtype=np.float32),
    }


def write_tiny_wells(directory: Path) -> None:
    """Write two complete deterministic coordinate CSV pairs for map tests.

    Args:
      directory: Test-owned split directory receiving both complete pairs.

    """
    directory.mkdir(parents=True, exist_ok=True)
    for well_id, values in (
        ("bbbbbbbb", "5,4\n8,6\n9,13\n"),
        ("aaaaaaaa", "1.0000004,1\n3.0000004,2.5\n7.0000004,7\n"),
    ):
        (directory / f"{well_id}__horizontal_well.csv").write_text("X,Y\n" + values)
        (directory / f"{well_id}__typewell.csv").write_text("TVT,GR\n1,2\n3,4\n")


def write_training_wells(directory: Path, *, count: int = 4, rows: int = 13) -> None:
    """Write complete tiny training pairs with finite anchored suffixes.

    Args:
      directory: Test-owned complete split directory.
      count: Number of distinct complete wells.
      rows: Full horizontal path length, including four visible prefix rows.

    """
    directory.mkdir(parents=True, exist_ok=True)
    for number in range(count):
        name = f"well{number}"
        with (directory / f"{name}__horizontal_well.csv").open(
            "w",
            newline="",
        ) as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(("MD", "X", "Y", "Z", "GR", "TVT_input", "TVT"))
            for row in range(rows):
                depth = (
                    100 + number + 8 * row if row < 4 else 124 + number + (row - 3) / 32
                )
                writer.writerow(
                    (
                        row,
                        number * 20 + row / 4,
                        number * 7 + row / 8,
                        -depth + row / 64,
                        50 + row / 8,
                        depth if row < 4 else "",
                        depth,
                    ),
                )
        with (directory / f"{name}__typewell.csv").open("w", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(("TVT", "GR"))
            writer.writerows((row, 40 + row / 8) for row in range(600))


def tiny_family(*, particle: bool) -> Family.Config:
    """Retain all native stages and draw policies at a small test geometry.

    Args:
      particle: Select the latest particle family or the earlier plain family.

    Returns:
      family: Complete one-epoch two-fold CPU model and data configuration.

    """
    family = family_0803_v2() if particle else family_0719_v1()
    family.training.max_epochs = family.cv.repeats = 1
    family.cv.clusters = family.cv.folds = 2
    assert isinstance(family.training.runtime, SingleProcess.Config)
    family.training.runtime.device = "cpu"
    family.training.step.dtype_autocast = None
    shrink_model(family.training.step.model)
    family.training.dataset.loader.num_workers = 0
    family.training.dataset.loader.batch_size = (
        family.training.dataset.validation_batch_size
    ) = 2
    features = family.training.dataset.loader.dataset.transform.features
    features.prefix_rows = 4
    features.target_rows = 28
    features.downsample = 2
    features.reference_columns = 32
    paths = _paths(family)
    paths.trend.minimum_suffix_rows = 8
    if family.particle is not None:
        family.particle.particles = 3
        family.particle.seeds = 5
        family.particle.paths = 2
    return family


def write_study_inputs(directory: Path, *, rows: int = 33) -> None:
    """Write complete small study inputs with real simulation donor suffixes.

    Args:
      directory: Test-owned complete competition release directory.
      rows: Complete path length; the default meets the public donor minimum.

    """
    for split in ("train", "test"):
        write_training_wells(directory / split, rows=rows)
    (directory / "train_png_typewell_map.csv").write_text(
        "well_id,horizontal_avg_X,horizontal_avg_Y\n"
        "well0,0,0\nwell1,1,1\nwell2,40,14\nwell3,41,15\n",
    )
    (directory / "sample_submission.csv").write_text(
        "id,tvt\n"
        + "".join(
            f"well{number}_{row},0\n"
            for number in range(3, -1, -1)
            for row in range(rows - 1, 3, -1)
        ),
    )


type Record = dict[str, Tensor | str | int | float | bool | None]


def freeze_record(value: object) -> Record:
    """Copy every numerical leaf into a flat, weights-only-loadable record."""
    record: Record = {}
    _freeze(value, path="", record=record)
    return record


def numerical_tensors(value: object) -> dict[str, Tensor]:
    """Encode every numerical and structural leaf without narrowing precision.

    Args:
      value: Complete nested model, optimizer and training observation.

    Returns:
      record: Every whole tensor plus lossless scalar and structural markers.

    """
    record: dict[str, Tensor] = {}
    for key, child in freeze_record(value).items():
        if isinstance(child, Tensor):
            record[key] = child
        elif isinstance(child, str):
            record[f"{key}/@str"] = torch.tensor(
                list(child.encode()),
                dtype=torch.uint8,
            )
        elif child is None:
            record[f"{key}/@none"] = torch.empty(0, dtype=torch.uint8)
        elif isinstance(child, bool):
            record[key] = torch.tensor(child, dtype=torch.bool)
        elif isinstance(child, int):
            record[key] = torch.tensor(child, dtype=torch.int64)
        else:
            record[key] = torch.tensor(child, dtype=torch.float64)
    return record


def historical_cosine_state(
    *,
    horizon: int,
    eta_min: float,
    initial_lrs: list[float],
    last_lrs: list[float],
    completed_steps: int,
) -> dict[str, object]:
    """Describe the original schedule state without owning an optimizer.

    Args:
      horizon: Complete recurrence horizon.
      eta_min: Absolute final rate.
      initial_lrs: Every parameter group's remembered initial rate.
      last_lrs: Every group's current absolute rate.
      completed_steps: Successful updates already applied to the recurrence.

    Returns:
      state: Every field retained by the frozen original Torch scheduler records.

    """
    return {
        "T_max": horizon,
        "eta_min": eta_min,
        "base_lrs": initial_lrs,
        "last_epoch": completed_steps,
        "_step_count": completed_steps + 1,
        "_is_initial": False,
        "_get_lr_called_within_step": False,
        "_last_lr": last_lrs,
    }


class EvalIdentityDropout(PinnedDropout):
    """Pin training dropout draws while evaluation-mode channel dropout passes through.

    Evaluation runs under ``inference_mode``, where ``feature_dropout`` reaches the
    dispatcher as itself instead of decomposing; with ``training`` False it is an
    identity that draws nothing.
    """

    @override
    def __torch_dispatch__(
        self,
        func: OpOverload[..., object],
        types: tuple[type, ...],
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ) -> object:
        if func is torch.ops.aten.feature_dropout.default and args[2] is False:
            return func(*args, **(kwargs or {}))
        return super().__torch_dispatch__(func, types, args, kwargs)


class ObservedLoss(WellboreLoss):
    """Record each training loss slot result without replacing its arithmetic."""

    class Config(Makes["ObservedLoss"], WellboreLoss.Config, kw_only=True):
        """The unchanged complete posterior objective."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.losses: list[Tensor] = []
        self.details: dict[str, list[Tensor]] = {}

    @override
    def __call__(self, output: object, **preprocessed_batch: object) -> LossOutput:
        result = super().__call__(output, **preprocessed_batch)
        prediction, _ = cast("tuple[Tensor, dict[str, Tensor]]", output)
        if not prediction.requires_grad:
            return result
        self.losses.append(result["loss"].detach()[0].clone())
        for name, value in result.items():
            if name != "loss":
                self.details.setdefault(name, []).append(value.detach().clone())
        return result


class BatchedFoldData(FoldData):
    """Construct complete deterministic batches for training-slot regression tests."""

    class Config(Makes["BatchedFoldData"], FoldData.Config, kw_only=True):
        batches: Callable[[], BatchLoader] = field(default_factory=lambda: tiny_batches)
        """Named constructor, never instantiated tensors in configuration."""

        has_validation: bool = True
        """Exercise either held-out checkpoint selection or training-only state."""

    def __init__(self, config: Config) -> None:
        self.train = config.batches()
        self.validation = config.batches() if config.has_validation else None
        self._held_out = None
        self.timer_epoch = CheckpointableStepTimer()


def tiny_model_config(*, dropout: float) -> SequenceModel.Config:
    """Return all four native stages at the immutable witness geometry.

    Args:
      dropout: Released family's stochastic feature probability.

    Returns:
      config: All stages and heads at the externally minted record dimensions.

    """
    config = SequenceModel.Config()
    shrink_model(config)
    config.channels_in = 3
    config.reference_columns = 32
    config.reference_window = 100.0
    config.target_stats = (1.591691, 15.853182)
    config.dropout = dropout
    return config


def tiny_recipe(family: Family.Config) -> WellboreFoldLoop.Config:
    """Retain every numerical recipe setting at the immutable three-update budget.

    Args:
      family: Complete public family tree.

    Returns:
      recipe: Real CPU loop tree with the complete four-stage witness model.

    """
    config = family.training.copy_tree()
    config.runtime = SingleProcess.Config(device="cpu")
    config.seed = family.seed
    config.max_epochs = 3
    config.dataset = BatchedFoldData.Config()
    config.step.dtype_autocast = None
    config.step.model = tiny_model_config(dropout=family.training.step.model.dropout)
    config.step.loss = ObservedLoss.Config()
    metric = config.metrics_eval[""]
    assert isinstance(metric, PooledRMSE.Config)
    metric.expansion.prefix_rows = 4
    metric.expansion.target_rows = 28
    metric.expansion.downsample = 2
    metric.expansion.target_stats = (1.591691, 15.853182)
    schedule = config.step.lr_schedule
    assert isinstance(schedule, RecursiveCosine.Config)
    schedule.T_max = 3
    return config


def tiny_batches() -> list[dict[str, object]]:
    """Return every complete deterministic source-record input and metadata field.

    Returns:
      batches: Full inputs and original-row labels used to mint the frozen records.

    """
    target = (torch.arange(2 * 16).reshape(2, 16) % 7).float() / 11
    mask = torch.ones(2, 16, dtype=torch.bool)
    mask[:, :2] = False
    probability = (torch.arange(2 * 16 * 32).reshape(2, 16, 32) % 13 + 1).float()
    probability.div_(probability.sum(-1, keepdim=True))
    return [
        {
            "media": (torch.arange(2 * 3 * 16 * 32).reshape(2, 3, 16, 32) % 17).float()
            / 19,
            # SequenceModel.forward reads two typewell channels per example.
            "typewell_aux": torch.zeros(2, 2, 32),
            "z_rel": target / 3,
            "bin_count": torch.full((2, 16), 2.0),
            "label": target,
            "typewell_target_probs": probability,
            "aux_targets": {
                "GR_penalty_error": probability,
                "unet_GR_RMSE": target.abs(),
                "unet_GR_RMSE_mask": mask,
                "gr_corr_weight_gr": target,
                "gr_corr_weight_mask": mask,
            },
            "target_mask": mask,
            "row_mask": torch.ones(2, 16),
            "meta": [
                {
                    "well_id": f"golden{index}",
                    "suffix_len": 28,
                    "submit_indices": np.arange(28),
                    "suffix_z": np.linspace(0, 1, 28, dtype=np.float32),
                    "tvt0": 100.0 + index,
                    "TVT": np.linspace(99, 103, 28, dtype=np.float32),
                }
                for index in range(2)
            ],
        },
    ]


def capture_training(
    *,
    initial_record: Path,
    recipe: WellboreFoldLoop.Config,
    one_ulp: bool = False,
    draw_offset: int = 0,
) -> dict[str, Tensor]:
    """Capture every original record leaf through the actual shared loop tree.

    Args:
      initial_record: Read-only original-source initial state and complete controls.
      recipe: Real finalized loop configuration with the original numerical recipe.
      one_ulp: Perturb the original alignment head to prove complete-state sensitivity.
      draw_offset: Perturb the pinned random input ordering.

    Returns:
      record: Every stored loss, model, optimizer, schedule, EMA and event leaf.

    """
    expected = read_tensors(initial_record)
    loop = recipe.finalized().make()
    step = loop.native_step
    model = step.model
    model.load_state_dict(
        {
            name.removeprefix("initial/")
            .replace(".mlp.fc1.", ".mlp.up_proj.")
            .replace(".mlp.fc2.", ".mlp.down_proj."): value
            for name, value in expected.items()
            if name.startswith("initial/")
        },
    )
    if one_ulp:
        raise_first_alignment_weight_one_ulp(model)
    assert isinstance(step.ema, InitializableEMA)
    step.ema.initialize(model, reset=True)
    pins = EvalIdentityDropout(model, draw_offset=draw_offset)
    with pins:
        loop.train()
    observed = step.loss
    assert isinstance(observed, ObservedLoss)
    if loop.best_state is not None:
        model.load_state_dict(loop.best_state)
    with step.ema.apply_to(model):
        ema_state = named_state(model)
    sites = list(dict.fromkeys(pins.sites))
    schedule = step.config.lr_schedule
    assert isinstance(schedule, RecursiveCosine.Config)
    assert isinstance(step.optimizer, CompositeOptimizer)
    # The original scheduler advanced the rate after each update; the shared
    # step applies it before the next one, so the final rate is applied here.
    step.apply_learning_rate()
    return numerical_tensors(
        {
            "losses": torch.stack(observed.losses),
            "details": {
                name: torch.stack(values) for name, values in observed.details.items()
            },
            "model": named_state(model),
            "optimizer": step.optimizer.optimizers[0].state_dict(),
            "scheduler": historical_cosine_state(
                horizon=schedule.T_max,
                eta_min=schedule.eta_min,
                initial_lrs=[
                    cast("float", group["initial_lr"])
                    for group in step.optimizer.param_groups
                ],
                last_lrs=[
                    cast("float", group["lr"]) for group in step.optimizer.param_groups
                ],
                completed_steps=step.global_step,
            ),
            "ema": ema_state,
            "score": loop.best_score,
            "epoch": loop.best_epoch,
            "history": loop.val_history,
            "pinned_bernoulli_calls": pins.calls,
            "dropout_sites": "|".join(sites),
            "dropout_order": torch.tensor(
                [sites.index(site) for site in pins.sites],
                dtype=torch.uint8,
            ),
        },
    )


def raise_first_alignment_weight_one_ulp(model: nn.Module) -> None:
    """Move the first output-projection weight up by one float32 unit in place."""
    with torch.no_grad():
        bits = model.get_parameter("proj_out.weight").view(torch.int32).flatten()
        bits[0] += 1


def named_state(model: nn.Module) -> dict[str, Tensor]:
    """Map every native MLP and head state key bijectively to its frozen record key.

    Args:
      model: Complete selected or EMA model.

    Returns:
      state: Every unchanged tensor, under its corresponding external record key.

    """
    return {
        name.replace(".mlp.up_proj.", ".mlp.fc1.")
        .replace(".mlp.down_proj.", ".mlp.fc2.")
        .replace("proj_out.", "alignment_head."): value
        for name, value in model.state_dict().items()
    }


class ProbeModel(SequenceModel):
    """Keep a complete posterior output while shrinking only the wiring-test model."""

    class Config(Makes["ProbeModel"], SequenceModel.Config, kw_only=True):
        """Typed small-model constructor with no initialized parameters in config."""

    def __init__(self, config: Config) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.channels_last = config.channels_last
        self.mapping = nn.Linear(2, 28)

    @override
    def forward(self, x: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        """Return four predicted depths and their complete six-candidate logits."""
        values = self.mapping(x)
        return values[:, :4], {"alignment_logits": values[:, 4:].reshape(-1, 4, 6)}


def probe_recipe() -> WellboreFoldLoop.Config:
    """Return a real three-epoch CPU loop with weighted partial accumulation.

    Returns:
      recipe: Shared loop and step slots with independently constructed inputs.

    """
    config = WellboreFoldLoop.Config()
    config.runtime = SingleProcess.Config(device="cpu")
    config.seed = 7
    config.max_epochs = 3
    config.early_stopping = EarlyStopping.Config(
        metric="rmse",
        patience=5,
        min_epochs=3,
    )
    config.dataset = BatchedFoldData.Config(batches=probe_batches)
    config.step.model = ProbeModel.Config()
    config.step.parallelism = NoParallel.Config(device="cpu")
    config.step.dtype_autocast = None
    config.step.compile = None
    config.step.accumulate_grad_batches = 2
    config.step.gradient_clip_norm = 0.2
    config.step.ema = EMA.Config()
    config.step.optimizer = CompositeOptimizer.Config(
        optimizers=[PartialConfig(torch.optim.AdamW, lr=0.002, weight_decay=0.02)],
    )
    config.step.lr_schedule = RecursiveCosine.Config(T_max=6, eta_min=5e-7)
    metric = config.metrics_eval[""]
    assert isinstance(metric, PooledRMSE.Config)
    metric.expansion.prefix_rows = 0
    metric.expansion.target_rows = 4
    metric.expansion.downsample = 1
    metric.expansion.target_stats = (0, 1)
    return config


def probe_batches() -> list[dict[str, object]]:
    """Construct all fields for batches of three, three and two complete samples.

    Returns:
      batches: Unequal complete posterior inputs and original-row metadata.

    """
    batches: list[dict[str, object]] = []
    for count in (3, 3, 2):
        target = torch.arange(count * 4, dtype=torch.float32).reshape(count, 4) / 11
        probabilities = torch.full((count, 4, 6), 1 / 6)
        batches.append(
            {
                "media": torch.arange(count * 2, dtype=torch.float32).reshape(
                    count,
                    2,
                )
                / 7,
                "typewell_aux": torch.zeros(count, 5),
                "z_rel": torch.zeros_like(target),
                "bin_count": torch.ones_like(target),
                "label": target,
                "typewell_target_probs": probabilities,
                "aux_targets": {"GR_penalty_error": probabilities},
                "target_mask": torch.ones_like(target, dtype=torch.bool),
                "row_mask": torch.ones_like(target),
                "meta": [
                    {
                        "well_id": str(index),
                        "suffix_len": 4,
                        "submit_indices": np.arange(4),
                        "suffix_z": np.zeros(4, dtype=np.float32),
                        "tvt0": 0.0,
                        "TVT": target[index].numpy(),
                    }
                    for index in range(count)
                ],
            },
        )
    return batches


def _freeze(value: object, *, path: str, record: Record) -> None:
    if isinstance(value, Tensor):
        record[path] = value.detach().cpu().clone()
    elif isinstance(value, np.ndarray):
        array = cast("NDArray[np.generic]", value)
        record[path] = torch.from_numpy(array.copy())
    elif isinstance(value, np.generic):
        record[path] = torch.from_numpy(np.asarray(value).copy())
    elif isinstance(value, Mapping):
        mapping = cast("Mapping[object, object]", value)
        if not mapping:
            record[path] = "dict:0"
        for key, child in mapping.items():
            _freeze(child, path=f"{path}/{key}".lstrip("/"), record=record)
    elif isinstance(value, (tuple, list)):
        children = cast("tuple[object, ...] | list[object]", value)
        record[f"{path}/@sequence".lstrip("/")] = (
            f"{type(children).__name__}:{len(children)}"
        )
        if not children:
            record[path] = f"{type(children).__name__}:0"
        for index, child in enumerate(children):
            _freeze(child, path=f"{path}/{index}".lstrip("/"), record=record)
    elif value is None or isinstance(value, (str, int, float, bool)):
        record[path] = value
    else:
        raise TypeError(f"Unsupported numerical trace value at {path}: {type(value)}.")
