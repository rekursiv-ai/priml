"""Per-token and per-sample training losses of the TRM trainer."""

from __future__ import annotations

from torch import Tensor, nn

import torch


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
