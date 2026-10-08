"""Grouped cross-entropy over the world model's restricted softmaxes.

Every target is scored by a softmax over only the IDs its slot may hold. A
board cell is eight such softmaxes, one per cell field, gathered from the 461
logits through ``FrameSchema.cell_index_table``; its NLL is their sum, the
cell's log-likelihood under the factorized heads. Every other slot adds a
``0``/``-inf`` bias from ``FrameSchema.local_allowed`` and takes one softmax.

The training objective is the unweighted sum of per-modality means, so the 99
cells of a frame do not bury the action, reward, and terminal. z-loss
(``weight * logZ**2``, OLMo 2's regularizer) is averaged the same way.
"""

from collections.abc import Mapping

import dataclasses

from torch import Tensor
from torch.nn import functional

import torch

from priml.math.loss import cross_entropy_logz


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Nll:
    """Per-target negative log-likelihood and squared log-normalizer.

    Attributes:
      nll: ``-log p(target)``, one per scored target.
      logz_sq: The softmax's squared log-normalizer, summed like ``nll``.

    """

    nll: Tensor
    logz_sq: Tensor


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ModalityLoss:
    """Summed NLL, z-loss, and target counts per modality.

    Attributes:
      nll: Summed NLL of the scored targets, per modality.
      logz_sq: Summed squared log-normalizers, per modality.
      count: Number of scored targets, per modality.
      objective: Sum over modalities of the mean NLL; 0 for an empty modality.
      z_loss: Weighted sum over modalities of the mean ``logZ**2``.
      loss: ``objective + z_loss``, the value to differentiate.

    """

    nll: dict[str, Tensor]
    logz_sq: dict[str, Tensor]
    count: dict[str, Tensor]
    objective: Tensor
    z_loss: Tensor
    loss: Tensor


def cross_entropy(logits: Tensor, target: Tensor) -> Nll:
    """Return ``target``'s NLL under ``softmax(logits)``, ``-inf`` logits excluded."""
    nll, logz = cross_entropy_logz(logits, target)
    return Nll(nll=nll, logz_sq=logz.square())


def cell_nll(logits: Tensor, *, index_table: Tensor, target: Tensor) -> Nll:
    """Return each cell's NLL: the sum of its per-field restricted softmaxes.

    Row ``k`` of ``index_table`` lists field ``k``'s allowed IDs in the game's value
    order, so a game value is directly its column in that row.

    Args:
      logits: Vocabulary scores per cell, ``[..., V]``.
      index_table: ``FrameSchema.cell_index_table()``, ``[K, W]``, padded with V.
      target: The game's value of each field, ``[..., K]``.

    Returns:
      nll: Per-cell NLL and ``logZ**2``, each summed over the ``K`` fields, ``[...]``.

    """
    padded = functional.pad(logits, (0, 1), value=float("-inf"))
    field = cross_entropy(padded[..., index_table], target)
    return Nll(nll=field.nll.sum(-1), logz_sq=field.logz_sq.sum(-1))


def scalar_nll(logits: Tensor, *, allowed: Tensor, target: Tensor) -> Nll:
    """Return the NLL of single-ID slots, each restricted to its allowed IDs.

    Args:
      logits: Vocabulary scores per slot, ``[..., S, V]``.
      allowed: Allowed IDs per slot, bool ``[S, V]``.
      target: Target ID per slot, ``[..., S]``.

    Returns:
      nll: Per-slot NLL and ``logZ**2``, ``[..., S]``.

    """
    bias = torch.zeros(allowed.shape, dtype=logits.dtype, device=logits.device)
    return cross_entropy(logits + bias.masked_fill(~allowed, float("-inf")), target)


def modality_loss(
    terms: Mapping[str, tuple[Nll, Tensor]],
    *,
    z_loss: float,
) -> ModalityLoss:
    """Reduce per-target NLLs to modality sums and the training objective.

    Args:
      terms: Per modality, the targets' NLL and a bool mask of those scored,
        broadcastable to the NLL's shape.
      z_loss: Weight of the mean squared log-normalizer, per modality.

    Returns:
      loss: Sums, counts, the sum of modality means, and the weighted z-loss.

    """
    nll: dict[str, Tensor] = {}
    logz_sq: dict[str, Tensor] = {}
    count: dict[str, Tensor] = {}
    for name, (value, mask) in terms.items():
        scored = mask.expand(value.nll.shape)
        # ``where``, not a multiply: an unscored target may sit on a ``-inf``
        # logit, and ``inf * 0`` would turn the whole sum into NaN.
        nll[name] = torch.where(scored, value.nll, 0.0).sum()
        logz_sq[name] = torch.where(scored, value.logz_sq, 0.0).sum()
        count[name] = scored.sum().to(value.nll.dtype)
    objective = torch.stack([nll[k] / count[k].clamp(min=1) for k in nll]).sum()
    regularizer = torch.stack([logz_sq[k] / count[k].clamp(min=1) for k in nll]).sum()
    return ModalityLoss(
        nll=nll,
        logz_sq=logz_sq,
        count=count,
        objective=objective,
        z_loss=z_loss * regularizer,
        loss=objective + z_loss * regularizer,
    )
