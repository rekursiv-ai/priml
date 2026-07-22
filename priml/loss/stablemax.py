"""Stablemax cross-entropy in fp64.

Ported verbatim from `TinyRecursiveModels/models/losses.py` for byte-by-byte
parity. The stablemax surrogate replaces the standard softmax:

    s(x) = 1 / (1 - x + eps)   if x < 0
           x + 1               if x >= 0

It is monotone, positive, and avoids the softmax's extreme-logit instability
when logits run unbounded under tiny initial scale. Reference operates in
fp64 inside the function; the call site is responsible for upcasting/restoring
dtype as needed.

Per-token loss is returned; the caller sums and divides by per-sample
valid-token count to match reference's training reduction.
"""

from __future__ import annotations

from torch import Tensor

import torch


def log_stablemax(x: Tensor, dim: int = -1) -> Tensor:
    """Log of the stablemax-normalized distribution."""
    s_x = _s(x)
    return torch.log(s_x / torch.sum(s_x, dim=dim, keepdim=True))


def _s(x: Tensor, epsilon: float = 1e-30) -> Tensor:
    """Stablemax surrogate (monotone positive replacement for exp)."""
    return torch.where(x < 0, 1.0 / (1.0 - x + epsilon), x + 1.0)


def stablemax_cross_entropy(
    logits: Tensor,
    labels: Tensor,
    ignore_index: int = -100,
    valid_mask: Tensor | None = None,
) -> Tensor:
    """Per-token stablemax cross-entropy, returned in fp64.

    Args:
      logits: [..., C] logits over vocab.
      labels: [...] integer labels in [0, C); positions equal to
        ``ignore_index`` are masked out (zero loss contribution).
      ignore_index: Label value marking positions to ignore.
      valid_mask: Optional precomputed mask; same shape as labels.

    Returns:
      loss: Same shape as labels; fp64. Caller reduces.

    """
    logprobs = log_stablemax(logits.to(torch.float64), dim=-1)
    if valid_mask is None:
        valid_mask = labels != ignore_index
    transformed_labels = torch.where(valid_mask, labels, 0)
    prediction_logprobs = torch.gather(
        logprobs,
        index=transformed_labels.to(torch.long).unsqueeze(-1),
        dim=-1,
    ).squeeze(-1)
    return -torch.where(valid_mask, prediction_logprobs, 0.0)
