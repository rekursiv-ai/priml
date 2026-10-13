"""Pure loss functions: cross-entropy variants and the stablemax surrogate.

Configgleable losses that consume these live in :mod:`priml.loss`; a
config slot takes one of these directly via ``PartialConfig``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import math

from torch import Tensor, nn

import torch

from priml.math.basic import reduction_dims
from priml.math.numeric import log_modulus


if TYPE_CHECKING:
    from collections.abc import Sequence


__all__ = [
    "cross_entropy_logz",
    "cross_entropy_with_batched_smoothing",
    "log_stablemax",
    "stablemax_cross_entropy",
]


def cross_entropy_logz(
    input: Tensor,
    target: Tensor,
    *,
    weight: Tensor | None = None,
    ignore_index: int = -100,
    reduction: Literal["none", "mean", "sum"] = "mean",
    label_smoothing: float = 0.0,
    dim: int | Sequence[int] = 1,
) -> tuple[Tensor, Tensor]:
    """``nn.functional.cross_entropy`` that also returns its log-normalizer.

    The log-normalizer ``log Z`` is what z-loss penalizes, ``mean(log Z ** 2)``,
    which keeps the logits from drifting where bf16 cannot resolve them; one
    ``logsumexp`` serves both. A ``-inf`` logit takes no probability, so masking
    entries to ``-inf`` restricts the softmax to the rest.

    ``log Z`` is not recovered from ``F.cross_entropy`` as ``loss + x_t``: that
    adds the loss's rounding to the logit's, measured up to 5.8x the error of a
    direct ``logsumexp`` on bfloat16 logits.

    It computes in the logits' dtype. On bfloat16 logits ``log Z`` rounds at its
    own magnitude before the subtraction, and the gradient through it with it,
    several times further from exact than ``F.cross_entropy`` on the same input;
    pass float32 logits where that matters.

    Args:
      input: Logits; the class axes are ``dim``.
      target: Class indices, ``input``'s shape without ``dim``. Over several
        axes, a row-major index into them in ``input``'s axis order.
      weight: Per-class weight, ``[C]``.
      ignore_index: Target value that contributes no loss and no gradient.
      reduction: "none", "mean", or "sum", as in ``F.cross_entropy``.
      label_smoothing: Mass in ``[0, 1]`` spread uniformly over the classes.
      dim: Class axis, or axes taking one joint softmax. ``1`` is
        ``F.cross_entropy``'s; pass ``0`` for unbatched logits.

    Returns:
      loss: ``F.cross_entropy`` with the same arguments.
      logz: ``logsumexp`` over ``dim``, shaped like ``target``, never reduced:
        z-loss squares it first. ``0`` where ``target`` is ignored.

    References:
      https://arxiv.org/abs/2204.02311
        Chowdhery et al. 2022. PaLM: z-loss on the softmax normalizer.

    Derivation::
      With w = weight (or ones), a = label_smoothing, and C classes,
      -log_softmax(x)_c = log Z - x_c, so
      nll = w_t * (log Z - x_t)
      smooth = -sum_c w_c * log_softmax(x)_c / C = (log Z * sum(w) - x @ w) / C
      loss = lerp(nll, smooth, a)

    """
    dims = reduction_dims(dim, ndim=input.ndim)
    num_classes = math.prod(input.shape[d] for d in dims)
    logits = input.movedim(dims, tuple(range(-len(dims), 0))).reshape(-1, num_classes)
    flat_target = target.reshape(-1)
    logz = logits.logsumexp(-1)
    valid = flat_target != ignore_index
    index = torch.where(valid, flat_target, 0)
    loss = logz - logits.gather(-1, index[:, None])[:, 0]
    count = valid if weight is None else torch.where(valid, weight[index], 0.0)
    if weight is not None:
        loss = count * loss
    # Skipped at zero, not multiplied by it: one ``-inf`` logit makes ``smooth``
    # infinite, and ``0 * inf`` is NaN.
    if label_smoothing:
        if weight is None:
            smooth = logz - logits.mean(-1)
        else:
            smooth = (logz * weight.sum() - logits @ weight) / num_classes
        loss = loss.lerp(smooth, label_smoothing)
    # ``where``, not a multiply: an ignored target may sit on a ``-inf`` logit,
    # and ``inf * 0`` would turn every reduction into NaN.
    loss = torch.where(valid, loss, 0.0)
    logz = torch.where(valid, logz, 0.0).reshape(target.shape)
    if reduction == "none":
        return loss.reshape(target.shape), logz
    if reduction == "sum":
        return loss.sum(), logz
    return loss.sum() / count.sum(), logz


def cross_entropy_with_batched_smoothing(
    input: Tensor,
    target: Tensor,
    *,
    weight: Tensor | None = None,
    ignore_index: int = -100,
    reduction: Literal["none", "mean", "sum"] = "mean",
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

    # Only ignored targets may be outside the class range; gather rejects others.
    mask = target != ignore_index
    safe_target = torch.where(mask, target, 0)
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
    return loss.sum() / (target_weight * mask).sum()


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
    class_dim = logits.ndim - 1
    logprobs = log_stablemax(logits)
    if valid_mask is None:
        valid_mask = labels != ignore_index
    transformed_labels = torch.where(valid_mask, labels, torch.zeros_like(labels))
    prediction_logprobs = torch.gather(
        logprobs,
        index=transformed_labels.unsqueeze(class_dim),
        dim=class_dim,
    ).squeeze(class_dim)
    return -torch.where(valid_mask, prediction_logprobs, 0.0)
