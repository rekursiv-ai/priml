"""Trivial validation baselines a trained world model must beat.

The design's third base criterion: next-frame cell accuracy above copying the
current frame, and action, reward, and done NLL below their empirical
frequencies. ``tally`` counts one packed batch: the model's summed NLL per
modality, histograms of the action, reward, and done targets, and how many
next-frame cells the model's per-field argmax and a copy of the current frame
each get exactly right. ``report`` turns the summed tallies into mean NLLs,
the entropy of each target's own empirical distribution (the NLL of the best
frequency-only predictor of those same targets), and the accuracies.
"""

from collections.abc import Sequence

from torch import Tensor
from torch.nn import functional

import torch

from priml.baselines.craftax.world_model.batch import PackedBatch
from priml.baselines.craftax.world_model.model import (
    WorldModel,
    WorldModelLogits,
)
from priml.lib.codec import PlainTree


@torch.no_grad()
def tally(
    model: WorldModel,
    batch: PackedBatch,
    *,
    logits: WorldModelLogits | None = None,
) -> dict[str, Tensor]:
    """Count one batch's model NLL, target histograms, and cell predictions.

    Args:
      model: The world model.
      batch: A packed validation micro-batch.
      logits: The model's logits of ``batch``; computed when None.

    Returns:
      counts: ``nll/<modality>`` and ``count/<modality>`` sums; ``action``,
        ``reward``, and ``done`` target histograms; ``cells`` scored next-frame
        cells and the ``model_correct`` and ``copy_correct`` among them.

    """
    logits = model.logits(batch) if logits is None else logits
    terms = model.target_terms(batch, logits)
    counts: dict[str, Tensor] = {}
    for name, (value, scored) in terms.items():
        counts[f"nll/{name}"] = value.nll[scored].double().sum()
        counts[f"count/{name}"] = scored.sum()
    is_act = ~batch.job_is_start
    low, high = model.schema.prefix_ranges[0]
    counts["action"] = torch.bincount(
        batch.action.roll(-1, dims=-1)[terms["action"][1]].long(),
        minlength=model.action_head.out_features,
    )
    counts["reward"] = torch.bincount(
        batch.job_reward[is_act].long() + model.scalar_offset - low,
        minlength=high - low + 1,
    )
    counts["done"] = torch.bincount(batch.job_done[is_act].long(), minlength=2)
    scored = is_act & (batch.job_next >= 0) & (batch.job_memory >= 0)
    target = batch.cells[batch.job_next[scored].long()].long()
    current = batch.cells[batch.job_memory[scored].long()].long()
    prefix = len(model.schema.prefix_ranges)
    board = logits.local[scored, prefix : prefix + model.schema.cell_slots]
    fields = functional.pad(board, (0, 1), value=float("-inf"))[..., model.cell_index]
    correct = (fields.argmax(-1) == target).all(-1)
    counts["cells"] = torch.tensor(correct.numel())
    counts["model_correct"] = correct.sum()
    counts["copy_correct"] = (current == target).all(-1).sum()
    return counts


def report(tallies: Sequence[dict[str, Tensor]]) -> dict[str, PlainTree]:
    """Return model and baseline scores of summed batch tallies.

    Args:
      tallies: ``tally`` of each validation batch.

    Returns:
      report: ``model_nll`` per modality and ``empirical_nll`` of action,
        reward, and done, in nats per target; ``cell_accuracy`` of the model
        and of copying; ``beats``, whether the model wins each comparison; and
        ``targets``, the scored targets per modality.

    """
    total = {
        key: torch.stack([t[key].double().cpu() for t in tallies]).sum(0)
        for key in tallies[0]
    }
    modalities = [key.removeprefix("nll/") for key in total if key.startswith("nll/")]
    model_nll = {m: float(total[f"nll/{m}"] / total[f"count/{m}"]) for m in modalities}
    empirical = {name: _entropy(total[name]) for name in ("action", "reward", "done")}
    accuracy = {
        name: float(total[f"{name}_correct"] / total["cells"])
        for name in ("model", "copy")
    }
    beats: dict[str, PlainTree] = {
        name: model_nll[name] < nll for name, nll in empirical.items()
    }
    beats["cells"] = accuracy["model"] > accuracy["copy"]
    return {
        "model_nll": dict(model_nll),
        "empirical_nll": dict(empirical),
        "cell_accuracy": dict(accuracy),
        "beats": beats,
        "targets": {m: int(total[f"count/{m}"]) for m in modalities},
    }


def _entropy(counts: Tensor) -> float:
    """Return the entropy in nats of a histogram's empirical distribution."""
    p = counts[counts > 0] / counts.sum()
    return float(-(p * p.log()).sum())
