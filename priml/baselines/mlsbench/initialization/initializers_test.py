"""Tests for the ``dl-weight-initialization`` initializers, against their closed forms."""

from __future__ import annotations

from typing import cast

from torch import Tensor

import pytest
import torch

from priml.baselines.mlsbench.initialization.initializers import (
    fixup,
    orthogonal_relu,
    small_normal,
)
from priml.baselines.mlsbench.pytorch_vision.model import BasicBlock, ResNet


def test_small_normal_draws_at_one_hundredth() -> None:
    w = torch.empty(64, 65)
    small_normal(w)
    assert w.mean().item() == pytest.approx(0.0, abs=1e-3)
    assert w.std().item() == pytest.approx(0.01, rel=0.05)


def test_orthogonal_relu_rows_are_orthogonal_at_relus_gain() -> None:
    w = torch.empty(2, 3, 4, 5)
    orthogonal_relu(w)
    flat = w.flatten(1).double()
    # Float32 storage of an orthogonal draw leaves rounding near 1e-7 per entry.
    gram = flat @ flat.T
    assert gram == pytest.approx(2 * torch.eye(2, dtype=torch.float64), abs=1e-6)


def test_fixup_scales_each_last_conv_and_zeros_its_norm_only() -> None:
    config = ResNet.Config(channels_in=3, channels_out=2)
    config.channels_hidden = (4, 5, 6)
    config.blocks_per_stage = (1, 2, 1)
    model = config.make()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    state = torch.get_rng_state()
    fixup(model)
    assert torch.equal(torch.get_rng_state(), state)
    blocks = {n: m for n, m in model.named_modules() if isinstance(m, BasicBlock)}
    assert len(blocks) == 4
    for name, block in blocks.items():
        want = before[f"{name}.conv2.weight"] * 4 ** (-0.5)
        assert torch.equal(block.conv2.weight, want)
        assert not cast(Tensor, block.norm2.weight).any()
    touched = {f"{n}.{p}.weight" for n in blocks for p in ("conv2", "norm2")}
    after = model.state_dict()
    assert all(torch.equal(after[k], v) for k, v in before.items() if k not in touched)


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
