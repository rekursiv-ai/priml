"""Muon and AdamW parameter partition for SpeedrunDiT."""

from __future__ import annotations

from configgle import PartialConfig

import torch

from priml.optimizers.composite import CompositeOptimizer
from priml.optimizers.muon import Muon
from priml.optimizers.parameter_filter import complement, excluding


def speedrundit_optimizer(
    *, reference_numerics: bool = True
) -> CompositeOptimizer.Config:
    """Use the REG AdamW/Muon split, selecting source arithmetic for exp000."""
    on_muon = excluding(
        Muon.eligible_tensor,
        "x_embedder",
        "t_embedder",
        "y_embedder",
        "final_layer",
        "cls_projector",
        "adaLN_modulation",
    )
    adamw = PartialConfig(torch.optim.AdamW)
    adamw.lr = 1e-4
    adamw.betas = (0.9, 0.999)
    adamw.weight_decay = 0.0
    adamw.eps = 1e-15

    muon = Muon.Config()
    muon.lr = 1e-3
    muon.momentum = 0.95
    muon.weight_decay = 0.0
    muon.nesterov = True
    muon.ns_steps = 5
    muon.reference_numerics = reference_numerics

    config = CompositeOptimizer.Config()
    config.optimizers = [adamw, muon]
    config.select = [complement(on_muon), on_muon]
    return config
