"""Unit tests for ARC1 parameter routing."""

from __future__ import annotations

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


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
