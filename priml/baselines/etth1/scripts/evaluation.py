"""Score exp000's best checkpoint on the held-out ETTh1 test split."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import argparse
import hashlib
import json

from torch import Tensor, nn

import numpy as np
import torch

from priml.baselines.etth1.experiments import exp000
from priml.lib.codec import from_plain, loads
from priml.paths import validated_output_path


if TYPE_CHECKING:
    from collections.abc import Iterable


def evaluate(
    model: nn.Module,
    batches: Iterable[dict[str, Tensor]],
) -> dict[str, float | int]:
    """Compute source-style float32 MSE/MAE over concatenated held-out forecasts.

    Args:
      model: The best validation checkpoint, in evaluation mode.
      batches: Ordered held-out batches with media and label tensors.

    Returns:
      metrics: Source MSE/MAE and sample counts, plus mean batch MSE/MAE.

    """
    predictions: list[Tensor] = []
    targets: list[Tensor] = []
    batch_mse: list[float] = []
    batch_mae: list[float] = []
    model.eval()
    with torch.no_grad():
        for batch in batches:
            output = cast(Tensor, model(batch["media"]))
            target = batch["label"]
            predictions.append(output.cpu())
            targets.append(target.cpu())
            batch_mse.append(torch.nn.functional.mse_loss(output, target=target).item())
            batch_mae.append(torch.nn.functional.l1_loss(output, target=target).item())
    if not predictions:
        raise ValueError("No complete test batches to evaluate.")
    prediction = torch.cat(predictions).numpy()
    target = torch.cat(targets).numpy()
    return {
        "mse": float(np.mean((prediction - target) ** 2)),
        "mae": float(np.mean(np.abs(prediction - target))),
        "mean_batch_mse": sum(batch_mse) / len(batch_mse),
        "mean_batch_mae": sum(batch_mae) / len(batch_mae),
        "batches": len(predictions),
        "windows": len(prediction),
        "elements": prediction.size,
    }


def main() -> int:
    """Load the explicitly selected checkpoint and print a reproducible report.

    Returns:
      code: 0 on success.

    """
    cfg = exp000()
    parser = argparse.ArgumentParser(description=(__doc__ or "").strip())
    _add_arguments(parser)
    flags = cast(_Flags, parser.parse_args())
    checkpoint = flags.checkpoint
    protected = [flags.directory / "ETTh1.csv"]
    if checkpoint.is_dir():
        selector = checkpoint / "best.json"
        protected.append(selector)
        best = from_plain(loads(selector.read_text()), dict[str, object])
        if from_plain(best["metric"], str) != "total_loss":
            raise ValueError("Expected a validation total_loss best-checkpoint record.")
        checkpoint = checkpoint / f"step_{from_plain(best['step'], int):08d}.pt"
    protected.append(checkpoint)
    output_path = (
        validated_output_path(
            flags.output,
            protected=protected,
        )
        if flags.output is not None
        else None
    )
    state = from_plain(
        cast(object, torch.load(checkpoint, map_location="cpu", weights_only=True)),
        dict[str, object],
    )
    step = from_plain(state["step"], dict[str, object])
    cfg.dataset.base_dir = None
    cfg.dataset.working_dir = flags.directory
    dataset = cfg.dataset.make()
    model = cfg.step.model.make()
    model.load_state_dict(from_plain(step["model"], dict[str, Tensor]))
    torch.set_num_threads(1)
    result: dict[str, object] = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "dataset_sha256": hashlib.sha256(
            (flags.directory / "ETTh1.csv").read_bytes(),
        ).hexdigest(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "step": from_plain(step["timer_step"], dict[str, object])["global_count"],
        **evaluate(model, batches=dataset.test_dataloader()),
    }
    rendered = json.dumps(result, indent=2) + "\n"
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered)
    print(rendered, end="")
    return 0


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="A .pt checkpoint, or its directory to select best.json.",
    )
    parser.add_argument(
        "--directory",
        type=Path,
        default=Path(exp000().copy_tree().finalize().dataset.working_dir),
    )
    parser.add_argument("--output", type=Path)


class _Flags(Protocol):
    checkpoint: Path
    directory: Path
    output: Path | None
