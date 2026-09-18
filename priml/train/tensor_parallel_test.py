"""Tests for the generic tensor-parallel applier (#301).

tp=1 must be a structural no-op (forward bit-for-bit). Multirank correctness
(sharded == dense) is proven in #304's cpu:gloo test.
"""

from __future__ import annotations

from typing import cast, override

from torch import Tensor, nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Replicate
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    ParallelStyle,
    RowwiseParallel,
)

import pytest
import torch

from priml.model.embedding import Embedding
from priml.model.linear import EnsembleLinear, Linear, _EnsembleParallel
from priml.train import tensor_parallel
from priml.train.tensor_parallel import (
    TensorParallel,
    _shard_style,
    apply_tensor_parallel,
)


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
    def forward(self, x: Tensor) -> Tensor:
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
    embed = Embedding.Config(channels_in=10, channels_out=8, shard="vocab").make()
    assert up.shard == "colwise"
    assert down.shard == "rowwise"
    assert plain.shard is None
    assert embed.shard == "vocab"


def test_unknown_shard_style_is_refused() -> None:
    """A style outside the declared set raises instead of silently replicating.

    The annotation rules this out statically; ``setattr`` is the seam a config
    from JSON or a ``--override`` comes through -- unchecked text the runtime
    guard, not the checker, has to catch.
    """
    config = Linear.Config(channels_in=8, channels_out=8)
    setattr(config, "shard", "colwize")  # noqa: B010 -- Bypasses the static type deliberately; see docstring.
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
        lambda: cast(DeviceMesh, _TpOneMesh()),
    )
    strategy = TensorParallel.Config().make()
    with torch.device("meta"):
        model = Linear.Config(channels_in=8, channels_out=8).finalize().make()
    assert model.weight.is_meta

    placed = strategy(model)

    assert not any(t.is_meta for t in placed.parameters())
    assert isinstance(placed, Linear)
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
        lambda: cast(DeviceMesh, _NamedMesh()),
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
    sharded = apply_tensor_parallel(model, cast(DeviceMesh, _TpOneMesh()))
    assert sharded is model
    assert isinstance(sharded, TwoLinear)
    assert not any(p.__class__.__name__ == "DTensor" for p in sharded.parameters())
    assert torch.equal(sharded(x), expected)


class _TpTwoSubmesh:
    def size(self) -> int:
        return 2


class _TpTwoMesh:
    device_type = "cpu"
    mesh_dim_names = ("tp",)

    def __getitem__(self, name: str) -> _TpTwoSubmesh:
        assert name == "tp"
        return _TpTwoSubmesh()


class _Validated(nn.Module):
    """A sharded block that also checks its own tensor-parallel preconditions."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = Linear.Config(channels_in=8, channels_out=8, shard="colwise").make()
        self.checked = 0

    def assert_tensor_parallel_compatible(self) -> None:
        self.checked += 1

    @override
    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x)


def test_tp2_applier_plans_every_declared_style_then_validates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Over a real tp axis the plan names each sharded leaf by its module path."""
    plans: list[dict[str, ParallelStyle]] = []

    def record(
        module: nn.Module,
        mesh: object,
        plan: dict[str, ParallelStyle],
    ) -> nn.Module:
        del mesh
        plans.append(dict(plan))
        return module

    monkeypatch.setattr(tensor_parallel, "parallelize_module", record)
    model = nn.Sequential(TwoLinear(), _Validated())

    sharded = apply_tensor_parallel(model, cast(DeviceMesh, _TpTwoMesh()))

    assert sharded is model
    assert len(plans) == 1
    assert {name: type(style) for name, style in plans[0].items()} == {
        "0.up": ColwiseParallel,
        "0.down": RowwiseParallel,
        "1.proj": ColwiseParallel,
    }
    validated = model[1]
    assert isinstance(validated, _Validated)
    assert validated.checked == 1


def test_tp2_applier_leaves_a_model_with_no_shard_declarations_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []

    def record(module: nn.Module, mesh: object, plan: object) -> nn.Module:
        calls.append((mesh, plan))
        return module

    monkeypatch.setattr(tensor_parallel, "parallelize_module", record)
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU())

    assert apply_tensor_parallel(model, cast(DeviceMesh, _TpTwoMesh())) is model
    assert calls == []


def test_strategy_requires_a_distributed_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tensor_parallel, "global_device_mesh", lambda: None)
    with pytest.raises(RuntimeError, match="requires distributed mode"):
        TensorParallel.Config().make()


def test_strategy_rejects_a_mesh_dim_the_mesh_lacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        tensor_parallel,
        "global_device_mesh",
        lambda: cast(DeviceMesh, _TpOneMesh()),
    )
    with pytest.raises(ValueError, match="'model' not in \\('tp',\\)"):
        TensorParallel.Config(mesh_dim="model").make()


def test_builtin_styles_dispatch_on_the_declared_shard() -> None:
    colwise = _shard_style(
        Linear.Config(channels_in=8, channels_out=8, shard="colwise").make(),
    )
    rowwise = _shard_style(
        Linear.Config(channels_in=8, channels_out=8, shard="rowwise").make(),
    )
    assert isinstance(colwise, ColwiseParallel)
    assert isinstance(rowwise, RowwiseParallel)
    assert _shard_style(Linear.Config(channels_in=8, channels_out=8).make()) is None
    assert _shard_style(nn.ReLU()) is None


def test_vocab_shard_replicates_ids_in_and_logits_out() -> None:
    """An embedding takes replicated ids; an lm head returns replicated logits."""
    table = _shard_style(
        Embedding.Config(channels_in=10, channels_out=8, shard="vocab").make(),
    )
    head = _shard_style(
        Linear.Config(channels_in=8, channels_out=10, shard="vocab").make(),
    )
    assert isinstance(table, RowwiseParallel)
    assert table.input_layouts == (Replicate(),)
    assert isinstance(head, ColwiseParallel)
    assert head.output_layouts == (Replicate(),)


def test_a_custom_layer_supplies_its_own_style_only_when_sharded() -> None:
    sharded = EnsembleLinear.Config(
        channels_in=8,
        channels_out=4,
        num_ensemble=2,
        shard="colwise",
    ).make()
    replicated = EnsembleLinear.Config(
        channels_in=8,
        channels_out=4,
        num_ensemble=2,
    ).make()
    assert isinstance(_shard_style(sharded), _EnsembleParallel)
    assert _shard_style(replicated) is None


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
