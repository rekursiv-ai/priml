"""Per-token and per-sample training losses of the sudoku train step."""

from __future__ import annotations

from configgle import Fig
from torch import Tensor, nn

import torch

from priml.baselines.sudoku.puzzle_spec import SudokuSpec, sudoku_group_indices


def token_cross_entropy(
    logits: Tensor,
    labels: Tensor,
    *,
    label_smoothing: float,
    ignore_index: int,
) -> Tensor:
    """Return per-token cross-entropy; ignored positions are zero.

    Args:
      logits: ``[B, G, V]`` token logits.
      labels: ``[B, G]`` targets.
      label_smoothing: Label smoothing epsilon.
      ignore_index: Label excluded from the loss.

    Returns:
      loss: ``[B, G]`` per-token loss.

    """
    return nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1).long(),
        label_smoothing=label_smoothing,
        ignore_index=ignore_index,
        reduction="none",
    ).reshape(logits.shape[0], -1)


def csp_cardinality_per_sample(
    logits: Tensor,
    labels: Tensor,
    group_indices: Tensor,
    temperature: float = 1.0,
) -> Tensor:
    """Mean squared row/column/box cardinality violation per sample.

    The cardinality loss of arXiv 2307.04895 on the model's per-cell token
    distributions: within every constraint group, each solution symbol's
    probability mass must sum to exactly 1 over the group's 9 cells. The 9
    solution symbols are read from ``labels`` per sample rather than
    hardcoded to tokens 2-10: the on-the-fly augmentation's legacy branch
    permutes token values 1-9 (blank marker included), so which 9 of the 10
    non-pad tokens are digits is sample-dependent. The symbol absent from
    the label (the blank marker) gets target 0. The softmax runs in fp32 so
    bf16-autocast logits do not quantize the small per-group residuals.

    Args:
      logits: ``[B, 81, 11]`` prefix-stripped grid logits.
      labels: ``[B, 81]`` label tokens. Rows holding only ignore labels
        (atomic-mode tail padding) yield an all-zero target and must be
        masked out by the caller.
      group_indices: ``[27, 9]`` cell indices from
        :func:`sudoku_group_indices`, on ``logits``'s device.
      temperature: Softmax temperature for the per-cell distributions (1.0 is
        the published form and the recipe value).

    Returns:
      per_sample: ``[B]`` mean over the 27 groups x 10 non-pad tokens of
        ``(group mass - target) ** 2``.

    References:
      https://arxiv.org/abs/2307.04895
        Yang, Lee & Park. Learning to Solve Constraint Satisfaction Problems
        with Recurrent Transformer. ICLR 2023.

    """
    if logits.shape[-2] != group_indices.shape[-1] ** 2:
        raise ValueError("logits must contain the prefix-stripped grid cells")
    probs = (logits.float() / temperature).softmax(dim=-1)[..., 1:]
    mass = probs[:, group_indices].sum(dim=-2)
    with torch.no_grad():
        present = nn.functional.one_hot(
            labels.long().clamp(min=0),
            num_classes=logits.shape[-1],
        ).sum(dim=-2)
        target = (present > 0).float()[..., 1:].unsqueeze(-2)
    return (mass - target).square().mean(dim=(-2, -1))


class CardinalityLoss:
    """Add the row, column, and box cardinality term to the token loss.

    The term (:func:`csp_cardinality_per_sample`) is reduced over the slots
    that trained exactly as the token loss is, then weighted; it joins the
    loss only once ``warmup_steps`` optimizer steps have passed.
    """

    class Config(Fig["CardinalityLoss"]):
        """Weight, warmup, temperature, and the grid it constrains."""

        weight: float = 1.0
        """Weight relative to the token loss."""

        warmup_steps: int = 0
        """Optimizer steps before the term joins the loss."""

        temperature: float = 1.0
        """Softmax temperature of the per-cell distributions."""

        spec: SudokuSpec | None = None
        """Grid, box, and vocabulary geometry; filled by the loop from its data."""

    def __init__(self, config: Config) -> None:
        spec = config.spec
        if spec is None:
            raise ValueError("The cardinality loss spec must be filled by its loop.")
        rows, cols = spec.grid_shape
        box_rows, box_cols = spec.box_shape
        if (
            rows != cols
            or min(rows, cols, box_rows, box_cols) < 1
            or rows % box_rows
            or cols % box_cols
            or box_rows * box_cols != rows
            or spec.vocab_size != rows + 2
        ):
            raise ValueError(
                "The cardinality loss requires sudoku grid, boxes, and vocabulary "
                "to agree.",
            )
        self.weight = config.weight
        self.warmup_steps = config.warmup_steps
        self.temperature = config.temperature
        self.groups = sudoku_group_indices(spec)

    def to(self, device: torch.device) -> None:
        """Move the constraint-group table to ``device``."""
        self.groups = self.groups.to(device)

    # Only active slots contribute; rows whose labels are entirely the ignore label
    # (atomic tail padding) are zeroed too.
    def __call__(
        self,
        logits: Tensor,
        labels: Tensor,
        *,
        active: Tensor,
        ignore_index: int,
    ) -> Tensor:
        """Return the unweighted term reduced over the active slots.

        Args:
          logits: ``[B, G, V]`` prefix-stripped grid logits.
          labels: ``[B, G]`` labels; ``ignore_index`` marks cells that do not count.
          active: ``[B]`` slots that trained this step.
          ignore_index: Label of cells that do not count.

        Returns:
          loss: Scalar mean over the active slots.

        """
        per_sample = csp_cardinality_per_sample(
            logits,
            labels,
            self.groups,
            temperature=self.temperature,
        )
        contributes = active & (labels != ignore_index).any(dim=-1)
        per_sample = torch.where(contributes, per_sample, torch.zeros_like(per_sample))
        return per_sample.sum() / active.sum().clamp(min=1)
