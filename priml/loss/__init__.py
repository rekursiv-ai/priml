"""Loss functions for autoencoders and generative models."""

from __future__ import annotations

from priml.loss.custom_types import LossOutput, SimpleLossFn
from priml.loss.diffusion import DiffusionLoss
from priml.loss.lpips_loss import LPIPSLoss
from priml.loss.simple_loss import SimpleLoss
from priml.loss.stablemax import (
    log_stablemax,
    stablemax_cross_entropy,
)
from priml.loss.weighted_loss import WeightedSum


__all__ = [
    "DiffusionLoss",
    "LPIPSLoss",
    "LossOutput",
    "SimpleLoss",
    "SimpleLossFn",
    "WeightedSum",
    "log_stablemax",
    "stablemax_cross_entropy",
]
