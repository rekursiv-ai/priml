"""Optimizers and learning rate utilities."""

from __future__ import annotations

from priml.optimizers.adam_atan2 import AdamATan2
from priml.optimizers.composite import (
    CompositeOptimizer,
    Selector,
    complement,
    everything,
    excluding,
    matching,
)
from priml.optimizers.fused_adamw import FusedAdamW
from priml.optimizers.lr import (
    HasParamGroups,
    apply_lr_scale,
    clip_grad_norm,
    lr_scale,
    remember_initial_lrs,
    step_optimizers,
    zero_optimizers,
)
from priml.optimizers.muon import Muon
from priml.optimizers.newton import Newton
from priml.optimizers.normuon import NorMuon
from priml.optimizers.sign_sgd import SignSGD


__all__ = [
    "AdamATan2",
    "CompositeOptimizer",
    "FusedAdamW",
    "HasParamGroups",
    "Muon",
    "Newton",
    "NorMuon",
    "Selector",
    "SignSGD",
    "apply_lr_scale",
    "clip_grad_norm",
    "complement",
    "everything",
    "excluding",
    "lr_scale",
    "matching",
    "remember_initial_lrs",
    "step_optimizers",
    "zero_optimizers",
]
