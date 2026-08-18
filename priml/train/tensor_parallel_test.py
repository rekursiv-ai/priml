"""Tests for the generic tensor-parallel applier (#301).

tp=1 must be a structural no-op (forward bit-for-bit). Multirank correctness
(sharded == dense) is proven in #304's cpu:gloo test.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast, override

from torch import nn

import pytest
import torch

from priml.model.custom_types import ShardStyle
from priml.model.embedding import Embedding
from priml.model.linear import Linear
from priml.train import tensor_parallel
from priml.train.tensor_parallel import (
    TensorParallel,
    _shard_style,
    apply_tensor_parallel,
)


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
    device_type = "cpu"
    mesh_dim_names = ("tp",)

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
    assert plain.shard is None
    assert embed.shard == "vocab"


def test_unknown_shard_style_is_refused() -> None:
    """A style outside the declared set raises instead of silently replicating.

    The annotation rules this out statically, so the value is cast in: a config
    from JSON or a ``--override`` is unchecked text, and the runtime guard is
    what catches it.
    """
    config = Linear.Config(channels_in=8, channels_out=8)
    config.shard = cast("ShardStyle", "colwize")
    with pytest.raises(ValueError, match="Unknown shard style"):
        _shard_style(config.make())


def test_meta_model_is_materialized_rather_than_copied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A meta-built model materializes here, as it does under every other strategy.

    ``device_init="meta"`` hands the placement strategy a module with no
    storage. ``Module.to`` cannot move one -- torch raises and names
    ``to_empty`` -- so a strategy that copies instead of materializing refuses
    the lazy construction the meta path exists for.
    """
    monkeypatch.setattr(
        tensor_parallel,
        "global_device_mesh",
        lambda: cast("DeviceMesh", _TpOneMesh()),
    )
    strategy = TensorParallel.Config().make()
    with torch.device("meta"):
        model = Linear.Config(channels_in=8, channels_out=8).finalize().make()
    assert model.weight.is_meta

    placed = strategy(model)

    assert not any(t.is_meta for t in placed.parameters())
    assert not torch.isnan(placed.weight).any()


def test_the_configured_mesh_dim_is_the_one_sharded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The strategy shards over ``mesh_dim``, not over a hard-coded name.

    The field is validated against the mesh at construction, so a config
    naming another dimension is accepted -- and then ignored, which shards
    over the wrong axis or raises on a mesh that has no ``tp`` at all.
    """
    asked: list[object] = []

    class _NamedMesh(_TpOneMesh):
        mesh_dim_names = ("model",)

        @override
        def __getitem__(self, name: str) -> _TpOneSubmesh:
            asked.append(name)
            return _TpOneSubmesh()

    monkeypatch.setattr(
        tensor_parallel,
        "global_device_mesh",
        lambda: cast("DeviceMesh", _NamedMesh()),
    )
    strategy = TensorParallel.Config(mesh_dim="model").make()

    strategy(Linear.Config(channels_in=8, channels_out=8).finalize().make())

    assert asked == ["model"], asked


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
