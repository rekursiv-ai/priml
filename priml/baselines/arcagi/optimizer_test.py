"""Unit tests for the rank filter that splits Muon's members."""

from __future__ import annotations

from torch import nn

import torch

from priml.baselines.arcagi.optimizer import with_ndim
from priml.optimizers.muon import Muon
from priml.optimizers.parameter_filter import excluding


def test_rank_filter_accepts_one_rank_and_compares_by_value() -> None:
    select = excluding(Muon.eligible_tensor, "head")
    rank2 = with_ndim(select, 2)
    rank3 = with_ndim(select, 3)
    assert rank2("body", nn.Parameter(torch.zeros(2, 3)))
    assert not rank2("body", nn.Parameter(torch.zeros(2, 3, 4)))
    assert rank3("body", nn.Parameter(torch.zeros(2, 3, 4)))
    assert not rank2("head", nn.Parameter(torch.zeros(2, 3)))
    assert rank2 == with_ndim(select, 2)
    assert rank2 != rank3
    assert hash(rank2) == hash(with_ndim(select, 2))
    assert "with_ndim" in repr(rank2)


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
