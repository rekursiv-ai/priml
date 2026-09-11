"""Learning-rate scaling helpers shared by every optimizer in this package."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol, runtime_checkable

import math

from torch import Tensor, nn

import torch


@runtime_checkable
class HasParamGroups(Protocol):
    """An optimizer whose per-group learning rates can be rewritten.

    The whole contract the rate helpers below need. Stated as a shape rather
    than as ``torch.optim.Optimizer`` so a composite or a wrapper that never
    inherits that class is still accepted.
    """

    param_groups: list[dict[str, Any]]


def lr_scale(
    step: int,
    total_steps: int,
    warmup_steps: int = 0,
    min_ratio: float = 0.0,
) -> float:
    """Learning rate scale factor with linear warmup and cosine decay.

    Args:
      step: Current training step.
      total_steps: Total number of training steps.
      warmup_steps: Number of linear warmup steps.
      min_ratio: Minimum LR ratio at end of cosine decay.

    Returns:
      scale: Multiplicative factor in [min_ratio, 1.0].

    """
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    # Clamp progress to 1 so steps past ``total_steps`` hold the cosine floor
    # rather than cycling back up (cos is periodic; progress > 1 re-ascends).
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return min_ratio + (1 - min_ratio) * cosine


def remember_initial_lrs(optimizers: Iterable[HasParamGroups]) -> None:
    """Store each optimizer group's original learning rate.

    Args:
      optimizers: Optimizers whose parameter groups should record
        ``initial_lr``. Existing ``initial_lr`` values are preserved.

    """
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            group.setdefault("initial_lr", group["lr"])


def apply_lr_scale(
    optimizers: Iterable[HasParamGroups],
    scale: float,
) -> None:
    """Apply an LR multiplier to optimizer groups.

    Takes the narrowest shape it uses rather than ``torch.optim.Optimizer``:
    only ``param_groups`` is read, so a wrapper or a composite that never
    inherits torch's class still qualifies. Declared here rather than imported
    from ``train`` because dependencies point down, and this layer sits below
    it.

    Args:
      optimizers: Optimizers whose parameter-group ``lr`` values should be
        updated.
      scale: Multiplicative factor applied to each group's ``initial_lr``.

    Raises:
      KeyError: If a parameter group has no ``initial_lr``. Call
        ``remember_initial_lrs`` after constructing optimizers.

    """
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * scale


def step_optimizers(optimizers: Iterable[torch.optim.Optimizer]) -> None:
    """Step multiple optimizers.

    Args:
      optimizers: Optimizers to step in order.

    """
    for optimizer in optimizers:
        optimizer.step()


def zero_optimizers(
    optimizers: Iterable[torch.optim.Optimizer],
    *,
    set_to_none: bool = True,
) -> None:
    """Zero gradients for multiple optimizers.

    Args:
      optimizers: Optimizers to zero in order.
      set_to_none: Whether gradients should be set to None.

    """
    for optimizer in optimizers:
        optimizer.zero_grad(set_to_none=set_to_none)


def clip_grad_norm(
    parameters: Iterable[nn.Parameter],
    max_norm: float | None,
) -> Tensor | None:
    """Clip gradient norm when enabled.

    Args:
      parameters: Parameters whose gradients should be clipped.
      max_norm: Maximum norm, or None to disable clipping.

    Returns:
      grad_norm: Total norm before clipping, or None when disabled.

    """
    if max_norm is None:
        return None
    return torch.nn.utils.clip_grad_norm_(parameters, max_norm)
