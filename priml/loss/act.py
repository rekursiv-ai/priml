"""Sequence loss and binary halting objective for adaptive computation."""

from __future__ import annotations

from dataclasses import field
from typing import NamedTuple

import math

from configgle import Fig, Makeable, Makes
from torch import Tensor

import torch

from priml.loss.custom_types import TokenLoss
from priml.loss.simple_loss import SimpleLoss, bce_with_logits
from priml.loss.token import StablemaxTokens


def _summed_halt_loss() -> SimpleLoss.Config:
    """Configure BCE as a summed batch loss."""
    config = SimpleLoss.Config()
    config.loss_fn = bce_with_logits
    config.target_key = "halt_target"
    config.kwargs = {"reduction": "sum"}
    return config


class ACTLossOutput(NamedTuple):
    """Unreduced ACT losses and sequence-level training diagnostics."""

    loss: Tensor
    """Token objective plus the weighted summed halt loss."""

    lm_loss: Tensor
    """Sum of per-sequence normalized token losses."""

    sequence_loss: Tensor
    """One token loss per sequence, each divided by its counted-token count."""

    halt_loss: Tensor
    """Summed binary cross-entropy over the halt logits."""

    correct: Tensor
    """Exact-match halt target; an all-ignored sequence is correct."""

    loss_counts: Tensor
    """Count of non-ignored target cells for each sequence."""


class ACTLoss:
    """Combine per-sequence token loss with the ACT halt objective."""

    class Config(Makes["ACTLoss"], Fig["ACTLoss"]):
        """Token-loss, halt-loss, and ignore-label settings."""

        token_loss: Makeable[TokenLoss] = field(
            default_factory=StablemaxTokens.Config,
        )
        """Per-token loss; default keeps float64 accumulation."""

        halt_loss: Makeable[SimpleLoss] = field(
            default_factory=_summed_halt_loss,
        )
        """Simple summed BCE for source-compatible halt supervision."""

        ignore_index: int = -100
        """Target token excluded from loss and exact-match scoring."""

        halt_weight: float = 0.5
        """Coefficient on summed halt BCE."""

    def __init__(self, config: Config) -> None:
        if not math.isfinite(config.halt_weight) or config.halt_weight < 0:
            raise ValueError("halt_weight must be finite and nonnegative.")
        self.token_loss: TokenLoss = config.token_loss.make()
        self.halt_loss = config.halt_loss.make()
        self.ignore_index = config.ignore_index
        self.halt_weight = config.halt_weight

    def __call__(
        self,
        *,
        logits: Tensor,
        labels: Tensor,
        halt_logits: Tensor,
        active: Tensor | None = None,
    ) -> ACTLossOutput:
        """Compute per-sequence token losses and summed halt BCE.

        Args:
          logits: ``[B, S, V]`` token logits.
          labels: ``[B, S]`` targets; ignored cells use ``ignore_index``.
          halt_logits: ``[B]`` ACT halt logits.
          active: Optional ``[B]`` mask; inactive rows contribute no loss,
            halt gradient, or diagnostics.

        Returns:
          output: Total loss, its components, exact-match targets, and counts.

        """
        if logits.ndim != 3 or labels.shape != logits.shape[:2]:
            raise ValueError(
                "ACTLoss expects logits [B,S,V] and labels [B,S] with matching "
                "leading dimensions.",
            )
        if halt_logits.shape != (logits.shape[0],):
            raise ValueError("ACTLoss expects halt_logits with shape [B].")
        if active is None:
            active = torch.ones(logits.shape[0], dtype=torch.bool, device=logits.device)
        elif active.shape != (logits.shape[0],):
            raise ValueError("ACTLoss expects active with shape [B].")

        counted = (labels != self.ignore_index) & active.unsqueeze(-1)
        token_loss = self.token_loss(
            logits,
            labels,
            ignore_index=self.ignore_index,
        )
        token_loss = torch.where(counted, token_loss, torch.zeros_like(token_loss))
        loss_counts = counted.sum(dim=1)
        normalized_tokens = token_loss / loss_counts.clamp_min(1)[:, None]
        sequence_loss = normalized_tokens.sum(dim=1)
        lm_loss = normalized_tokens.sum()
        with torch.no_grad():
            correct = (
                ((logits.argmax(dim=2) == labels) | ~counted).all(dim=1)
            ) & active
        if bool(active.all()):
            halt_loss = self.halt_loss(
                halt_logits,
                halt_target=correct.to(halt_logits.dtype),
            )["loss"]
        else:
            halt_loss = self.halt_loss(
                halt_logits[active],
                halt_target=correct[active].to(halt_logits.dtype),
            )["loss"]
        total = lm_loss + self.halt_weight * halt_loss
        return ACTLossOutput(
            loss=total,
            lm_loss=lm_loss,
            sequence_loss=sequence_loss,
            halt_loss=halt_loss,
            correct=correct,
            loss_counts=loss_counts,
        )
