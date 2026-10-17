"""Evaluate released CHARM checkpoints with the native ARC data and metric.

Examples:
  uv --quiet run --frozen python -m priml.baselines.arcagi.scripts.charm_eval --benchmark arc1 --data-dir /opt/scratch/datasets/charm-release-937316c8 --checkpoint-root /opt/scratch/datasets/charm-release-937316c8/checkpoints/arc-agi-1 --output-dir /opt/scratch/artifacts/summarizer/charm-native-arc1
  uv --quiet run --frozen python -m torch.distributed.run --standalone --nproc_per_node=8 -m priml.baselines.arcagi.scripts.charm_eval --benchmark arc2 --data-dir /opt/scratch/datasets/charm-release-937316c8 --checkpoint-root /opt/scratch/datasets/charm-release-937316c8/checkpoints/arc-agi-2 --output-dir /opt/scratch/artifacts/summarizer/charm-native-arc2

"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, Protocol, TypedDict, cast

import argparse
import json
import os

from torch import Tensor
from treekle import to_plain

import torch
import torch.distributed as dist

from priml.baselines.arcagi.charm_checkpoint import remap_charm_checkpoint
from priml.baselines.arcagi.experiments import (
    CharmTrainLoop,
    exp009,
    exp010,
)
from priml.baselines.arcagi.metric import ParquetPassK
from priml.baselines.arcagi.parquet_data import ArcParquetData
from priml.baselines.sudoku.act import rollout
from priml.model.grid_solver import SingleLatentGridSolver
from priml.paths import validated_output_path
from priml.runtime import MultiProcess, RuntimeProtocol, SingleProcess
from priml.train.parallelism import NoParallel


if TYPE_CHECKING:
    from priml.baselines.arcagi.parquet_data import ArcBatch, ArcParquetData


Benchmark = Literal["arc1", "arc2"]
SOURCE_REVISION: Final = "728bf26903684ef5e8bcb476cb8e55e42cf35b9d"
RELEASE_REVISION: Final = "937316c8f766920724293600bdfd8ed3d45ebdc6"
RELEASE_ROOT: Final = Path("/opt/scratch/datasets/charm-release-937316c8")
RELEASE_DATA_PREFIX: Final = Path("data/augmented/v2")
OUTPUT_ROOT: Final = Path("/opt/scratch")
CHECKPOINT_INTERVAL: Final = 10_000


@dataclass(frozen=True, slots=True, kw_only=True)
class _Preset:
    checkpoint_directory: str
    start_step: int
    stop_step: int
    experiment: Literal["exp009", "exp010"]


_PRESETS: Final[dict[Benchmark, _Preset]] = {
    "arc1": _Preset(
        checkpoint_directory="arc-agi-1",
        start_step=1_510_000,
        stop_step=1_590_001,
        experiment="exp009",
    ),
    "arc2": _Preset(
        checkpoint_directory="arc-agi-2",
        start_step=2_290_000,
        stop_step=2_380_001,
        experiment="exp010",
    ),
}


class Flags(Protocol):
    """Parsed native evaluation options."""

    benchmark: str
    data_dir: Path
    checkpoint_root: Path
    checkpoint_step: int | None
    output_dir: Path


class CheckpointResult(TypedDict):
    """Scores and provenance for one loaded checkpoint event."""

    step: int
    checkpoint: str
    checkpoint_sha256: str
    act_steps: int
    current_scores: dict[str, float]
    accumulated_scores: dict[str, float]


def main() -> int:
    """Parse options and evaluate the selected released checkpoint window.

    Returns:
      status: Process status code, zero after successful evaluation.

    """
    parser = argparse.ArgumentParser(
        description=(__doc__ or "").strip(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_arguments(parser)
    flags = cast(Flags, parser.parse_args())
    benchmark = cast(Benchmark, flags.benchmark)
    preset = _PRESETS[benchmark]
    data_dir = flags.data_dir.expanduser().resolve()
    checkpoint_root = flags.checkpoint_root.expanduser().resolve()
    output_dir = validated_output_path(
        flags.output_dir,
        protected=(data_dir, checkpoint_root),
    )
    if not output_dir.is_relative_to(OUTPUT_ROOT):
        raise ValueError(f"output directory must be beneath {OUTPUT_ROOT}.")
    if output_dir.is_relative_to(data_dir) or output_dir.is_relative_to(
        checkpoint_root,
    ):
        raise ValueError("output directory cannot be inside a release input tree.")
    if checkpoint_root.name != preset.checkpoint_directory:
        raise ValueError(
            f"{benchmark} requires checkpoint root named "
            f"{preset.checkpoint_directory!r}; got {checkpoint_root.name!r}.",
        )
    steps = select_checkpoint_steps(
        benchmark,
        checkpoint_step=flags.checkpoint_step,
    )
    checkpoints = _checkpoint_paths(checkpoint_root, steps)
    runtime_config = _runtime_config()
    runtime = cast(RuntimeProtocol, runtime_config.make())
    runtime.initialize()
    try:
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        torch.manual_seed(0 + rank)
        recipe = _recipe(preset, data_dir=data_dir, world_size=world_size)
        recipe.finalize()
        dataset = recipe.dataset.make()
        with torch.device(runtime.device):
            model = recipe.step.model.make()
            parallelism = NoParallel.Config(device=runtime.device).make()
            model = cast(SingleLatentGridSolver, parallelism(model))
            aggregate_config = cast(
                ParquetPassK.Config,
                recipe.metrics_eval[""],
            )
            accumulated_metric = aggregate_config.make()
            current_config = aggregate_config.copy_tree()
            current_config.aggregated_voting = False
            current_metric = current_config.make()
        input_paths = _dataset_paths(recipe.dataset, data_dir=data_dir)
        dataset_hashes = {str(path): _file_sha256(path) for path in input_paths}
        provenance: dict[str, object] = {
            "benchmark": benchmark,
            "experiment": preset.experiment,
            "source_revision": SOURCE_REVISION,
            "release_revision_expected": RELEASE_REVISION,
            "release_data_root": str(data_dir),
            "release_checkpoint_root": str(checkpoint_root),
            "dataset_sha256": dataset_hashes,
            "checkpoint_steps": steps,
            "global_batch_size": recipe.global_batch_size,
            "data_parallel_size": world_size,
            "per_rank_eval_batch_size": recipe.dataset.eval_batch_size,
            "seed": recipe.seed,
            "canvas_size": recipe.dataset.canvas_size,
            "num_workers": recipe.dataset.num_workers,
            "model_width": recipe.step.model.channels_in,
            "model_layers": recipe.step.model.num_layers,
            "model_vocab_size": recipe.step.model.vocab_size,
            "prefix_tokens": recipe.step.model.prefix.num_tokens
            if recipe.step.model.prefix is not None
            else 0,
            "recurrent_passes": recipe.step.model.recurrence.slow_cycles
            * recipe.step.model.recurrence.fast_cycles
            if recipe.step.model.recurrence is not None
            else 1,
            "act_steps": recipe.step.pool.max_steps,
            "compile_core": "core/default/fullgraph=False",
            "metric_pass_ks": list(accumulated_metric.config.pass_ks),
            "metric_recent_window_size": (accumulated_metric.config.recent_window_size),
        }
        results = evaluate_checkpoints(
            model,
            dataset,
            accumulated_metric,
            current_metric,
            checkpoints,
            device=runtime.device,
            output_dir=output_dir,
            max_steps=recipe.step.pool.max_steps,
            submission_k=100,
            provenance=provenance,
        )
        if rank == 0:
            print(f"Evaluated {len(results)} CHARM checkpoints in {output_dir}.")
    finally:
        runtime.destroy()
    return 0


def select_checkpoint_steps(
    benchmark: Benchmark,
    *,
    checkpoint_step: int | None,
) -> list[int]:
    """Return the released checkpoint window or one aligned step.

    Args:
      benchmark: ARC release preset to select.
      checkpoint_step: Optional single step within that preset's window.

    Returns:
      steps: Ordered checkpoint steps to evaluate.

    """
    preset = _PRESETS[benchmark]
    if checkpoint_step is None:
        return list(range(preset.start_step, preset.stop_step, CHECKPOINT_INTERVAL))
    if (
        checkpoint_step < preset.start_step
        or checkpoint_step >= preset.stop_step
        or (checkpoint_step - preset.start_step) % CHECKPOINT_INTERVAL
    ):
        raise ValueError(
            f"checkpoint step {checkpoint_step} is outside the released window "
            f"for {benchmark}.",
        )
    return [checkpoint_step]


def evaluate_checkpoints(
    model: SingleLatentGridSolver,
    dataset: ArcParquetData,
    accumulated_metric: ParquetPassK,
    current_metric: ParquetPassK,
    checkpoints: Sequence[Path],
    *,
    device: torch.device,
    output_dir: Path,
    max_steps: int,
    submission_k: int,
    provenance: Mapping[str, object],
) -> list[CheckpointResult]:
    """Load live-plus-EMA states and evaluate one or more ACT checkpoints.

    The accumulated metric object survives every checkpoint and retains its
    configured recent event window. The current metric is reset to measure each
    checkpoint in isolation. Every distributed rank calls both metrics in the
    same order so their rank-ordered vote gather remains synchronized.

    Args:
      model: Native model populated from each released checkpoint.
      dataset: Native evaluation data loader and distributed sampler.
      accumulated_metric: Persistent metric retaining the recent event window.
      current_metric: Metric reset for each checkpoint event.
      checkpoints: Ordered source checkpoints to evaluate.
      device: Runtime-owned inference device.
      output_dir: Scratch directory for metrics and submissions.
      max_steps: Number of carried ACT model calls per batch.
      submission_k: Number of attempts written per test input.
      provenance: Release, recipe, input, and runtime metadata to record.

    Returns:
      results: Per-checkpoint scores and measured SHA256 values.

    """
    if max_steps < 1:
        raise ValueError("ACT max_steps must be positive.")
    if not checkpoints:
        raise ValueError("At least one checkpoint is required.")
    output_dir.mkdir(parents=True, exist_ok=True)
    loader = dataset.eval_dataloader()
    results: list[CheckpointResult] = []
    for checkpoint_path in checkpoints:
        step = _step_from_path(checkpoint_path)
        checkpoint_sha = _file_sha256(checkpoint_path)
        _load_released_checkpoint(
            checkpoint_path,
            expected_step=step,
            model=model,
        )
        accumulated_metric.reset()
        current_metric.reset()
        dataset.eval_dataset.set_epoch(0)
        model.eval()
        with torch.inference_mode():
            for batch in loader:
                packed = _predict_batch(
                    model,
                    batch,
                    device=device,
                    max_steps=max_steps,
                )
                accumulated_metric.update(packed, **batch)
                current_metric.update(packed, **batch)
        current_scores = current_metric.compute()
        accumulated_scores = accumulated_metric.compute()
        current_submission = current_metric.submission(
            submission_k=submission_k,
        )
        accumulated_submission = accumulated_metric.submission(
            submission_k=submission_k,
        )
        result = CheckpointResult(
            step=step,
            checkpoint=str(checkpoint_path.resolve()),
            checkpoint_sha256=checkpoint_sha,
            act_steps=max_steps,
            current_scores=current_scores,
            accumulated_scores=accumulated_scores,
        )
        results.append(result)
        if not dist.is_initialized() or dist.get_rank() == 0:
            _write_json(
                output_dir / "submissions" / f"step_{step}.json",
                _source_submission(accumulated_submission),
            )
            _write_json(
                output_dir / "submissions" / f"current_step_{step}.json",
                _source_submission(current_submission),
            )
            _write_json(
                output_dir / "submission.json",
                _source_submission(accumulated_submission),
            )
            _write_json(output_dir / "metrics.json", results)
            _write_json(
                output_dir / "provenance.json",
                {
                    **provenance,
                    "checkpoint_results": results,
                    "submission_file": "submission.json",
                },
            )
    return results


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--benchmark",
        choices=("arc1", "arc2"),
        required=True,
        help="Released ARC checkpoint window to evaluate.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=RELEASE_ROOT,
        help="Pinned CHARM release root containing data/ and checkpoints/.",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        required=True,
        help="Directory containing the selected released step_*.pt files.",
    )
    parser.add_argument(
        "--checkpoint-step",
        type=int,
        default=None,
        help="Evaluate one step from the selected released window.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New run directory beneath /opt/scratch for scores and submissions.",
    )


def _runtime_config() -> MultiProcess.Config | SingleProcess.Config:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        return MultiProcess.Config(
            mesh_topology={"dp": world_size, "pp": 1, "tp": 1},
        )
    return SingleProcess.Config()


def _recipe(
    preset: _Preset,
    *,
    data_dir: Path,
    world_size: int,
) -> CharmTrainLoop.Config:
    recipe = exp009() if preset.experiment == "exp009" else exp010()
    if world_size > 1:
        recipe.runtime = MultiProcess.Config(
            mesh_topology={"dp": world_size, "pp": 1, "tp": 1},
        )
    else:
        recipe.runtime = SingleProcess.Config()
    recipe.dataset.base_dir = data_dir
    for source in recipe.dataset.train_sources:
        source.path = str(RELEASE_DATA_PREFIX / Path(source.path).name)
    eval_path = str(RELEASE_DATA_PREFIX / Path(recipe.dataset.eval_source).name)
    recipe.dataset.eval_source = eval_path
    metric = cast(ParquetPassK.Config, recipe.metrics_eval[""])
    metric.base_dir = data_dir
    metric.eval_source = eval_path
    return recipe


def _dataset_paths(dataset: ArcParquetData.Config, *, data_dir: Path) -> list[Path]:
    paths = {
        (data_dir / Path(source.path)).resolve() for source in dataset.train_sources
    }
    paths.add((data_dir / Path(dataset.eval_source)).resolve())
    missing = [path for path in sorted(paths) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing CHARM parquet inputs: {missing}.")
    return sorted(paths)


def _checkpoint_paths(checkpoint_root: Path, steps: Sequence[int]) -> list[Path]:
    if not checkpoint_root.is_dir():
        raise FileNotFoundError(f"checkpoint root does not exist: {checkpoint_root}.")
    paths = [checkpoint_root / f"step_{step}.pt" for step in steps]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing released CHARM checkpoints: {missing}.")
    return paths


def _load_released_checkpoint(
    path: Path,
    *,
    expected_step: int,
    model: SingleLatentGridSolver,
) -> None:
    payload_value = cast(
        object,
        torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        ),
    )
    payload = _mapping(payload_value, label="checkpoint payload")
    step = payload.get("step")
    if not isinstance(step, int) or isinstance(step, bool) or step != expected_step:
        raise ValueError(
            f"checkpoint {path} has step {step!r}; expected {expected_step}.",
        )
    live = _tensor_mapping(payload.get("model_state_dict"), label="model_state_dict")
    ema = _tensor_mapping(payload.get("ema_state"), label="ema_state")
    state = remap_charm_checkpoint(live, model=model, ema_state=ema)
    model.load_state_dict(state, strict=True)


def _predict_batch(
    model: SingleLatentGridSolver,
    batch: ArcBatch,
    *,
    device: torch.device,
    max_steps: int,
) -> Tensor:
    if max_steps <= 0:
        raise ValueError("CHARM evaluation requires at least one ACT step.")
    media = batch["media"].to(device)
    puzzle_idxs = batch["puzzle_idxs"].to(device)
    dih_idxs = batch["dih_idxs"].to(device)
    colorperm_slots = batch["colorperm_slots"].to(device)
    puzzle_identifiers = batch["puzzle_identifiers"].to(device)
    logits, halt = rollout(
        model,
        media=media,
        max_steps=max_steps,
        carry=None,
        prefix_kwargs={
            "category_indices": puzzle_idxs,
            "augmentation_indices": dih_idxs,
            "slot_permutation": colorperm_slots,
            "puzzle_identifiers": puzzle_identifiers,
        },
    )
    return torch.hstack(
        (
            _halt_column(halt),
            _prediction_tokens(logits),
        ),
    )


def _halt_column(halt: Tensor) -> Tensor:
    return halt.to(torch.float32).reshape(-1, 1)


def _prediction_tokens(logits: Tensor) -> Tensor:
    return logits.argmax(dim=-1).to(torch.float32)


def _step_from_path(path: Path) -> int:
    step_text = path.stem.removeprefix("step_")
    if not path.stem.startswith("step_") or not step_text.isdigit():
        raise ValueError(f"invalid CHARM checkpoint filename: {path.name}.")
    return int(step_text)


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(to_plain(value), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _source_submission(
    submission: dict[str, list[dict[str, list[list[int]]]]],
) -> dict[str, list[dict[str, list[list[int]]]]]:
    """Copy candidates so source JSON contains no treekle alias tags."""
    return {
        puzzle: [
            {attempt: [list(row) for row in grid] for attempt, grid in attempts.items()}
            for attempts in puzzle_attempts
        ]
        for puzzle, puzzle_attempts in submission.items()
    }


def _mapping(value: object, *, label: str) -> Mapping[object, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping.")
    return cast(Mapping[object, object], value)


def _tensor_mapping(value: object, *, label: str) -> Mapping[str, Tensor]:
    raw = _mapping(value, label=label)
    narrowed: dict[str, Tensor] = {}
    for key, item in raw.items():
        if not isinstance(key, str) or not isinstance(item, Tensor):
            raise TypeError(f"{label} must contain string keys and tensor values.")
        narrowed[key] = item
    return narrowed
