"""Unit tests for ARC1 parameter routing."""

from __future__ import annotations

from configgle import PartialConfig
from torch import nn

import torch

from priml.baselines.arcagi1.optimizer import adamw_muon, with_ndim
from priml.optimizers.muon import Muon
from priml.optimizers.parameter_filter import excluding


def test_rank_filter_and_optimizer_recipe() -> None:
    select = excluding(Muon.eligible_tensor, "head")
    rank2 = with_ndim(select, 2)
    rank3 = with_ndim(select, 3)
    assert rank2("body", nn.Parameter(torch.zeros(2, 3)))
    assert not rank2("body", nn.Parameter(torch.zeros(2, 3, 4)))
    assert rank3("body", nn.Parameter(torch.zeros(2, 3, 4)))
    assert rank2 == with_ndim(select, 2)
    assert rank2 != rank3
    assert hash(rank2) == hash(with_ndim(select, 2))
    assert "with_ndim" in repr(rank2)
    config = adamw_muon(adamw_lr=1e-4, muon_lr=1e-2)
    assert len(config.optimizers) == 3
    assert config.drop_empty
    adamw, muon_2d, muon_3d = config.optimizers
    assert adamw == PartialConfig(
        torch.optim.AdamW,
        lr=1e-4,
        betas=(0.9, 0.95),
        weight_decay=1.0,
    )
    assert isinstance(muon_2d, Muon.Config)
    assert isinstance(muon_3d, Muon.Config)
    for muon in (muon_2d, muon_3d):
        assert muon.lr == 1e-2
        assert muon.momentum == 0.6
        assert muon.ns_steps == 3
        assert muon.weight_decay == 0.01
    assert muon_2d.ensemble_dims == 0
    assert muon_3d.ensemble_dims == 1

    rank2_weight = nn.Parameter(torch.zeros(2, 3))
    rank3_weight = nn.Parameter(torch.zeros(2, 3, 4))
    vector = nn.Parameter(torch.zeros(2))
    assert [select("body.weight", rank2_weight) for select in config.select] == [
        False,
        True,
        False,
    ]
    assert [select("body.weight", rank3_weight) for select in config.select] == [
        False,
        False,
        True,
    ]
    for name in ("embed_tokens.weight", "head.weight", "register_tokens"):
        assert [select(name, rank2_weight) for select in config.select] == [
            True,
            False,
            False,
        ]
    assert [select("body.bias", vector) for select in config.select] == [
        True,
        False,
        False,
    ]


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
