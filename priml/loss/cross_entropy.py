"""Cross-entropy variants."""

from __future__ import annotations

from torch import Tensor, nn


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
