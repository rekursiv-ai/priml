"""Pure loss functions: cross-entropy variants and the stablemax surrogate.

Configgleable losses that consume these live in :mod:`priml.loss`; a
config slot takes one of these directly via ``PartialConfig``.
"""

from __future__ import annotations

from torch import Tensor, nn

import torch

from priml.math.numeric import log_modulus


__all__ = [
    "cross_entropy_with_batched_smoothing",
    "log_stablemax",
    "stablemax_cross_entropy",
]


def cross_entropy_with_batched_smoothing(
    input: Tensor,
    target: Tensor,
    *,
    weight: Tensor | None = None,
    ignore_index: int = -100,
    reduction: str = "mean",
    label_smoothing: float | Tensor = 0.0,
) -> Tensor:
    """Cross-entropy loss supporting per-element label smoothing.

    When ``label_smoothing`` is a scalar, delegates to
    ``nn.functional.cross_entropy``. When it is a Tensor broadcastable
    with ``target``, each element gets its own smoothing value.

    Args:
      input: Logits of shape (N, C) or (N, C, ...).
      target: Class indices of shape (N,) or (N, ...).
      weight: Per-class weight tensor.
      ignore_index: Target value to ignore.
      reduction: "none", "mean", or "sum".
      label_smoothing: Scalar or per-element Tensor in [0, 1].

    Returns:
      loss: Cross-entropy loss.

    """
    if isinstance(label_smoothing, (int, float)):
        return nn.functional.cross_entropy(
            input,
            target,
            weight=weight,
            ignore_index=ignore_index,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )

    # Per-element (tensor) smoothing. The class dimension is dim=1 for
    # (N, C, ...) logits, matching ``nn.functional.cross_entropy``.
    num_classes = input.shape[1]
    log_probs = input.log_softmax(dim=1)

    # ``ignore_index`` may exceed ``num_classes``; clamp the index used for
    # gathering and zero those positions out via ``mask`` afterwards.
    mask = (target != ignore_index).to(input.dtype)
    safe_target = target.clamp(0, num_classes - 1)
    gathered = log_probs.gather(1, safe_target.unsqueeze(1)).squeeze(1)

    # PyTorch weights both the NLL and the smoothing term by the per-class
    # weight, then normalizes by the summed weight of non-ignored targets.
    if weight is not None:
        target_weight: Tensor = weight[safe_target]
        nll = -target_weight * gathered
        weight_shape = (1, num_classes, *([1] * (input.ndim - 2)))
        smooth = -(log_probs * weight.view(weight_shape)).sum(dim=1) / num_classes
    else:
        target_weight = mask
        nll = -gathered
        smooth = -log_probs.mean(dim=1)

    loss = ((1.0 - label_smoothing) * nll + label_smoothing * smooth) * mask

    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    # "mean"
    return loss.sum() / (target_weight * mask).sum().clamp(min=1)


def log_stablemax(x: Tensor, dim: int = -1) -> Tensor:
    """Log of the stablemax-normalized distribution.

    Stablemax is a softmax whose ``exp`` is replaced by the surrogate::

          s(x) = 1 / (1 - x)  if x < 0
                 1 + x        if x >= 0

    Args:
      x: Logits.
      dim: Dimension to normalize over.

    Returns:
      logprobs: Log-probabilities, normalized along ``dim``.

    Derivation::
      log[s(x)] =
      = where(x < 0, -log1p(-x), log1p(x))
      = sign(x) * log1p(|x|)
      = log_modulus(x)

      Hence,

      log stablemax(x) =
      = log[ s(x) / sum(s(x)) ]
      = log[s(x)] - logsumexp(log[s(x)])
      = log_modulus(x) - logsumexp(log_modulus(x))
      = log_softmax(log_modulus(x))

    """
    return torch.log_softmax(log_modulus(x), dim=dim)


def stablemax_cross_entropy(
    logits: Tensor,
    labels: Tensor,
    ignore_index: int = -100,
    valid_mask: Tensor | None = None,
) -> Tensor:
    """Per-token stablemax cross-entropy, in the dtype of ``logits``.

    The stablemax surrogate replaces the standard softmax:

        s(x) = 1 / (1 - x)   if x < 0
               x + 1         if x >= 0

    It is monotone, positive, and avoids the softmax's extreme-logit
    instability when logits run unbounded under tiny initial scale.

    Args:
      logits: [..., C] logits over vocab.
      labels: [...] integer labels in [0, C); positions equal to
        ``ignore_index`` are masked out (zero loss contribution).
      ignore_index: Label value marking positions to ignore.
      valid_mask: Optional precomputed mask; same shape as labels.

    Returns:
      loss: Same shape as labels, dtype of ``logits``. A caller reducing over
        many tokens owns the accumulation dtype -- pass ``dtype=`` to its
        ``sum``; summing hundreds of bf16 terms in bf16 costs far more than
        bf16's own rounding.

    """
    logprobs = log_stablemax(logits, dim=-1)
    if valid_mask is None:
        valid_mask = labels != ignore_index
    transformed_labels = torch.where(valid_mask, labels, 0)
    prediction_logprobs = torch.gather(
        logprobs,
        index=transformed_labels.to(torch.long).unsqueeze(-1),
        dim=-1,
    ).squeeze(-1)
    return -torch.where(valid_mask, prediction_logprobs, 0.0)
