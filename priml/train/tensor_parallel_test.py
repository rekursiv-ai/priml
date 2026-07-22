"""Tests for the generic tensor-parallel applier (#301).

tp=1 must be a structural no-op (forward bit-for-bit). Multirank correctness
(sharded == dense) is proven in #304's cpu:gloo test.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast, override

from torch import nn

import torch

from priml.model.embedding import Embedding
from priml.model.linear import Linear
from priml.train.tensor_parallel import apply_tensor_parallel


if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh


class TwoLinear(nn.Module):
    """colwise -> rowwise pair: the canonical MLP TP shape."""

    def __init__(self) -> None:
        super().__init__()
        self.up = Linear.Config(channels_in=8, channels_out=16, shard="colwise").make()
        self.down = Linear.Config(
            channels_in=16,
            channels_out=8,
            shard="rowwise",
        ).make()

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.up(x))


class _TpOneSubmesh:
    def size(self) -> int:
        return 1


class _TpOneMesh:
    def __getitem__(self, name: str) -> _TpOneSubmesh:
        assert name == "tp"
        return _TpOneSubmesh()


def test_shard_style_stored_on_runtime_module() -> None:
    up = Linear.Config(channels_in=8, channels_out=16, shard="colwise").make()
    down = Linear.Config(channels_in=16, channels_out=8, shard="rowwise").make()
    plain = Linear.Config(channels_in=8, channels_out=8).make()
    embed = Embedding.Config(num_embeddings=10, channels_out=8, shard="vocab").make()
    assert up.shard == "colwise"
    assert down.shard == "rowwise"
    assert plain.shard == "none"
    assert embed.shard == "vocab"


def test_tp1_applier_is_structural_noop() -> None:
    """tp=1: applier returns the model untouched; forward bit-for-bit."""
    torch.manual_seed(0)
    model = TwoLinear()
    x = torch.randn(4, 8)
    expected = model(x)
    sharded = apply_tensor_parallel(model, cast("DeviceMesh", _TpOneMesh()))
    assert sharded is model
    assert not any(p.__class__.__name__ == "DTensor" for p in sharded.parameters())
    assert torch.equal(sharded(x), expected)


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
