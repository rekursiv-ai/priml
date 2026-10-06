"""Tests for parallelism strategies."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast, override

import functools
import importlib
import math
import tempfile

from torch import Tensor, nn
from torch.distributed._composable.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor import DTensor

import pytest
import torch

from priml import runtime
from priml.model.attention.gated_delta_net import GatedDeltaNet
from priml.model.norm import CenteredRMSNorm
from priml.train import parallelism
from priml.train.parallelism import (
    DataParallel,
    FullySharded,
    HybridSharded,
    NoParallel,
    RecursiveSharded,
    _create_mp_policy,
    _mesh_device,
    _module_mp_policy,
    _shard,
    materialize_meta,
)


if TYPE_CHECKING:
    from collections.abc import Callable

    from configgle import Makeable
    from torch.distributed.device_mesh import DeviceMesh

    from priml.distributed.testing import WarmPoolGetter
    from priml.train.custom_types import ParallelStrategyProtocol


class _FakeMesh:
    """The mesh surface a strategy reads while choosing its device and plan.

    A real ``DeviceMesh`` needs an initialized process group, which the
    placement decisions under test all precede.
    """

    device_type = "cpu"
    mesh_dim_names = ("dp", "tp")
    shape = (1, 1)

    def __getitem__(self, key: object) -> _FakeMesh:
        del key
        return self

    def get_group(self, name: str) -> None:
        """Return the process group for ``name``; there is none here."""
        del name

    def size(self) -> int:
        """Ranks along this dimension."""
        return 1


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)

    def reset_parameters(self) -> None:
        # Owns re-initializing the child it constructed (recursive ownership).
        self.linear.reset_parameters()

    @override
    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)


class _TwoLinear(nn.Module):
    """Two stacked linears with recursive-ownership reset (no bare nn.Sequential)."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(8, 8)
        self.fc2 = nn.Linear(8, 8)

    def reset_parameters(self) -> None:
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()

    @override
    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.fc1(x))


def test_no_parallel_places_on_device():
    model = SimpleModel()
    config = NoParallel.Config(device="cpu")
    strategy = config.make()
    result = strategy(model)
    assert next(result.parameters()).device.type == "cpu"


def test_no_parallel_exposes_device():
    """T-046: every strategy must expose ``.device`` (used by Learnable)."""
    strategy = NoParallel.Config(device="cpu").make()
    assert strategy.device == torch.device("cpu")


def test_no_parallel_forwards_the_configured_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested: list[torch.device | str | None] = []

    def resolve(device: torch.device | str | None) -> torch.device:
        requested.append(device)
        return torch.device("cpu")

    monkeypatch.setattr(parallelism, "get_device", resolve)
    strategy = NoParallel.Config(device="meta").make()

    assert requested == ["meta"]
    assert strategy.device == torch.device("cpu")


def test_data_parallel_forwards_configuration_and_returns_placed_model(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    mesh = _FakeMesh()
    group = object()
    groups: list[str] = []

    def get_group(name: str) -> object:
        groups.append(name)
        return group

    monkeypatch.setattr(parallelism, "global_device_mesh", lambda: mesh)
    monkeypatch.setattr(mesh, "get_group", get_group)
    calls: list[tuple[nn.Module, dict[str, object]]] = []

    def record(model: nn.Module, **kwargs: object) -> None:
        calls.append((model, kwargs))

    monkeypatch.setattr(parallelism, "replicate", record)
    model = SimpleModel()
    strategy = DataParallel.Config(
        mesh_dim="tp",
        bucket_cap_mb=7,
        find_unused_parameters=True,
        gradient_as_bucket_view=True,
    ).make()

    with caplog.at_level("INFO"):
        result = strategy(model)

    assert groups == ["tp"]
    assert strategy.device == torch.device("cpu")
    assert strategy.bucket_cap_mb == 7
    assert strategy.find_unused_parameters is True
    assert strategy.gradient_as_bucket_view is True
    assert calls == [
        (
            model,
            {
                "process_group": group,
                "bucket_cap_mb": 7,
                "find_unused_parameters": True,
                "gradient_as_bucket_view": True,
            },
        ),
    ]
    assert result is model
    assert caplog.records[-1].message == (
        "Applied DataParallel: mesh_dim=tp, bucket_cap_mb=7, "
        "gradient_as_bucket_view=True"
    )


class _BufferOnly(nn.Module):
    """Module with only a floating-point buffer."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("stat", torch.zeros(4))

    def reset_parameters(self) -> None:
        stat = self._buffers["stat"]
        assert isinstance(stat, Tensor)
        stat.fill_(1.0)


def test_no_parallel_materializes_meta_buffer_only_model() -> None:
    """NoParallel materializes meta buffers even when no parameters exist."""
    with torch.device("meta"):
        model = _BufferOnly()
    model_stat: object = model._buffers["stat"]
    assert isinstance(model_stat, Tensor)
    assert model_stat.is_meta

    result = NoParallel.Config(device="cpu").make()(model)

    result_stat: object = result._buffers["stat"]
    assert isinstance(result_stat, Tensor)
    assert not result_stat.is_meta
    assert result_stat.device.type == "cpu"
    torch.testing.assert_close(result_stat, torch.ones(4))


def test_no_parallel_materializes_meta_model():
    """D-002: a meta-initialized model must be materialized to real tensors."""
    with torch.device("meta"):
        model = SimpleModel()
    assert next(model.parameters()).is_meta

    strategy = NoParallel.Config(device="cpu").make()
    result = strategy(model)

    param = next(result.parameters())
    assert not param.is_meta, "meta params were never materialized"
    assert param.device.type == "cpu"
    # reset_parameters must have run post-materialization: to_empty leaves
    # uninitialized memory which can contain NaN/inf; a proper reset yields
    # finite values.
    assert torch.isfinite(param).all(), "materialized params not re-initialized"


def test_data_parallel_requires_distributed():
    config = DataParallel.Config()
    with pytest.raises(RuntimeError) as exc_info:
        config.make()
    assert str(exc_info.value) == (
        "DataParallel requires distributed mode. Initialize with MultiProcess runtime."
    )


def test_fully_sharded_requires_distributed():
    config = FullySharded.Config()
    with pytest.raises(RuntimeError) as exc_info:
        config.make()
    assert str(exc_info.value) == (
        "FullySharded requires distributed mode. Initialize with MultiProcess runtime."
    )


def test_hybrid_sharded_requires_distributed():
    config = HybridSharded.Config()
    with pytest.raises(RuntimeError) as exc_info:
        config.make()
    assert str(exc_info.value) == (
        "HybridSharded requires distributed mode. Initialize with MultiProcess runtime."
    )


def test_recursive_sharded_requires_distributed():
    config = RecursiveSharded.Config()
    with pytest.raises(RuntimeError) as exc_info:
        config.make()
    assert str(exc_info.value) == (
        "RecursiveSharded requires distributed mode. "
        "Initialize with MultiProcess runtime."
    )


class _DpOnlyMesh(_FakeMesh):
    mesh_dim_names = ("dp",)


@pytest.mark.parametrize(
    "build",
    [
        lambda: DataParallel.Config(mesh_dim="tp"),
        lambda: FullySharded.Config(mesh_dim="tp"),
        lambda: RecursiveSharded.Config(mesh_dim="tp", module_types=(nn.Linear,)),
    ],
    ids=["data_parallel", "fully_sharded", "recursive_sharded"],
)
def test_strategies_reject_a_mesh_dim_the_mesh_lacks(
    monkeypatch: pytest.MonkeyPatch,
    build: Callable[[], Makeable[ParallelStrategyProtocol]],
) -> None:
    monkeypatch.setattr(parallelism, "global_device_mesh", _DpOnlyMesh)
    with pytest.raises(ValueError, match=r"'tp' not in \('dp',\)"):
        build().make()


def test_hybrid_sharded_names_every_missing_dimension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(parallelism, "global_device_mesh", _DpOnlyMesh)
    with pytest.raises(ValueError, match=r"\['replica', 'tp'\] not in \('dp',\)"):
        HybridSharded.Config(replicate_dim="replica", shard_dim="tp").make()


def test_recursive_sharded_refuses_a_plan_that_matches_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def ignore(
        module: nn.Module,
        mesh: DeviceMesh,
        mp_policy: MixedPrecisionPolicy | None,
        reshard_after_forward: bool,
    ) -> None:
        del module, mesh, mp_policy, reshard_after_forward

    monkeypatch.setattr(parallelism, "global_device_mesh", _FakeMesh)
    monkeypatch.setattr(parallelism, "_shard", ignore)
    strategy = RecursiveSharded.Config(module_types=(nn.Conv2d,)).make()
    with pytest.raises(ValueError, match="found 0 modules matching"):
        strategy(SimpleModel())


def test_mesh_device_pins_the_current_cuda_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every rank reduces on its own GPU; a shared index would pile onto one."""

    class _CudaMesh(_FakeMesh):
        device_type = "cuda"

    # Scoped: the autouse ``cleanup_cuda`` teardown synchronizes the CURRENT
    # device before monkeypatch unwinds, and index 3 is not a real ordinal.
    with monkeypatch.context() as patch:
        patch.setattr(torch.cuda, "current_device", lambda: 3)
        assert _mesh_device(cast("DeviceMesh", _CudaMesh())) == torch.device("cuda", 3)
    assert _mesh_device(cast("DeviceMesh", _FakeMesh())) == torch.device("cpu")


def test_mixed_precision_policy_is_built_only_when_a_dtype_is_set() -> None:
    assert _create_mp_policy(None, None, None) is None
    for dtypes in (
        (torch.bfloat16, None, None),
        (None, torch.float16, None),
        (None, None, torch.float32),
    ):
        assert _create_mp_policy(*dtypes) is not None

    policy = _create_mp_policy(torch.bfloat16, None, torch.float32)
    assert policy is not None
    assert policy.param_dtype == torch.bfloat16
    assert policy.reduce_dtype is None
    assert policy.output_dtype == torch.float32


def test_shard_passes_a_policy_only_when_one_resolves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def record(module: nn.Module, **kwargs: object) -> None:
        calls.append({"module": module, **kwargs})

    monkeypatch.setattr(parallelism, "fully_shard", record)
    mesh = cast("DeviceMesh", _FakeMesh())
    base = MixedPrecisionPolicy(param_dtype=torch.bfloat16)
    linear = nn.Linear(2, 2)
    norm = nn.BatchNorm1d(2)

    _shard(linear, mesh=mesh, mp_policy=None, reshard_after_forward=True)
    _shard(linear, mesh=mesh, mp_policy=base, reshard_after_forward=False)
    _shard(norm, mesh=mesh, mp_policy=base, reshard_after_forward=True)

    assert calls[0] == {"module": linear, "mesh": mesh, "reshard_after_forward": True}
    assert calls[1] == {
        "module": linear,
        "mesh": mesh,
        "mp_policy": base,
        "reshard_after_forward": False,
    }
    bn_policy = calls[2]["mp_policy"]
    assert isinstance(bn_policy, MixedPrecisionPolicy)
    assert bn_policy.param_dtype == torch.float32


def test_materialize_barriers_when_a_group_is_initialized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ranks re-init in lockstep so no rank reads a peer's shard early."""
    barriers: list[int] = []
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)

    def gather(errors: list[str | None], error: str | None) -> None:
        errors[0] = error
        barriers.append(1)

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    with torch.device("meta"):
        model = SimpleModel()

    materialize_meta(model, torch.device("cpu"))

    assert barriers == [1]
    assert torch.isfinite(next(model.parameters())).all()


def test_materialize_meta_is_a_noop_on_an_allocated_model() -> None:
    model = SimpleModel()
    before = next(model.parameters()).detach().clone()
    materialize_meta(model, torch.device("cpu"))
    assert torch.equal(next(model.parameters()), before)


def test_recursive_sharded_requires_module_types(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty ``module_types`` is refused when the strategy is built.

    Asserting the config's default instead proves only that the field starts
    empty -- the guard could be deleted and that assertion would still pass.
    The guard runs in ``__init__``, ahead of every collective, so a stand-in
    mesh is all it takes to reach.
    """
    monkeypatch.setattr(parallelism, "global_device_mesh", _FakeMesh)
    with pytest.raises(ValueError, match="requires module_types") as exc_info:
        RecursiveSharded.Config().make()
    assert str(exc_info.value) == (
        "RecursiveSharded requires module_types to be specified. "
        "Provide tuple of module classes to shard (e.g., (TransformerBlock,))."
    )


def test_recursive_sharded_shards_the_root_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A root that is itself a matched type is sharded once, not twice.

    ``model.modules()`` yields the root, so a root matching ``module_types``
    is sharded by the loop and then again by the explicit root shard below
    it. Composable FSDP rejects the second application, and the failure needs
    a model shaped like the plan to appear at all.
    """
    monkeypatch.setattr(parallelism, "global_device_mesh", _FakeMesh)
    sharded: list[nn.Module] = []

    def record(
        module: nn.Module,
        mesh: DeviceMesh,
        mp_policy: MixedPrecisionPolicy | None,
        reshard_after_forward: bool,
    ) -> None:
        del mesh, mp_policy, reshard_after_forward
        sharded.append(module)

    monkeypatch.setattr(parallelism, "_shard", record)
    strategy = RecursiveSharded.Config(module_types=(SimpleModel,)).make()

    strategy(SimpleModel())

    assert len(sharded) == len(set(map(id, sharded))), "a module was sharded twice"


def test_recursive_sharding_passes_child_and_root_plans(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    mesh = _FakeMesh()
    monkeypatch.setattr(parallelism, "global_device_mesh", lambda: mesh)
    calls: list[tuple[nn.Module, dict[str, object]]] = []

    def record_shard(module: nn.Module, **kwargs: object) -> None:
        calls.append((module, kwargs))

    placed: list[tuple[nn.Module, torch.device]] = []

    def record_place(model: nn.Module, device: torch.device) -> nn.Module:
        placed.append((model, device))
        return model

    monkeypatch.setattr(parallelism, "_shard", record_shard)
    monkeypatch.setattr(parallelism, "place", record_place)

    class NestedLinear(nn.Linear):
        def __init__(self) -> None:
            super().__init__(2, 3)
            self.child = nn.Linear(3, 4)
            self.second_child = nn.Linear(4, 5)
            self.unmatched = nn.ReLU()

    model = NestedLinear()
    strategy = RecursiveSharded.Config(
        module_types=(nn.Linear,),
        reshard_after_forward=True,
        mp_param_dtype=torch.bfloat16,
        mp_reduce_dtype=torch.float16,
        mp_output_dtype=torch.float64,
    ).make()

    with caplog.at_level("INFO"):
        assert strategy(model) is model

    assert placed == [(model, torch.device("cpu"))]
    assert [module for module, _kwargs in calls] == [
        model.second_child,
        model.child,
        model,
    ]
    assert [kwargs for _module, kwargs in calls[:2]] == [
        {
            "mesh": mesh,
            "mp_policy": strategy.mp_policy,
            "reshard_after_forward": True,
        },
        {
            "mesh": mesh,
            "mp_policy": strategy.mp_policy,
            "reshard_after_forward": True,
        },
    ]
    assert calls[2] == (
        model,
        {
            "mesh": mesh,
            "mp_policy": strategy.mp_policy,
            "reshard_after_forward": False,
        },
    )
    assert strategy.mp_policy is not None
    assert strategy.mp_policy.param_dtype == torch.bfloat16
    assert strategy.mp_policy.reduce_dtype == torch.float16
    assert strategy.mp_policy.output_dtype == torch.float64
    assert caplog.records[-1].message == (
        "Applied RecursiveSharded: sharded 3 modules matching ['Linear'], mesh_dim=dp"
    )


def test_sharded_strategies_shard_before_placement(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    mesh = _FakeMesh()
    monkeypatch.setattr(parallelism, "global_device_mesh", lambda: mesh)
    events: list[str] = []
    shard_calls: list[tuple[nn.Module, dict[str, object]]] = []

    def record_shard(module: nn.Module, **kwargs: object) -> None:
        events.append("shard")
        shard_calls.append((module, kwargs))

    def record_place(model: nn.Module, device: torch.device) -> nn.Module:
        events.append("place")
        assert device == torch.device("cpu")
        return model

    monkeypatch.setattr(parallelism, "_shard", record_shard)
    monkeypatch.setattr(parallelism, "place", record_place)
    fully_model = SimpleModel()
    hybrid_model = SimpleModel()
    fully = FullySharded.Config(
        reshard_after_forward=False,
        mp_param_dtype=torch.bfloat16,
        mp_reduce_dtype=torch.float16,
        mp_output_dtype=torch.float64,
    ).make()
    hybrid = HybridSharded.Config(
        replicate_dim="dp",
        shard_dim="tp",
        reshard_after_forward=False,
        mp_param_dtype=torch.bfloat16,
        mp_reduce_dtype=torch.float16,
        mp_output_dtype=torch.float64,
    ).make()

    with caplog.at_level("INFO"):
        assert fully(fully_model) is fully_model
        assert hybrid(hybrid_model) is hybrid_model

    assert events == ["shard", "place", "shard", "place"]
    assert shard_calls[0] == (
        fully_model,
        {
            "mesh": mesh,
            "mp_policy": fully.mp_policy,
            "reshard_after_forward": False,
        },
    )
    assert isinstance(fully.mp_policy, MixedPrecisionPolicy)
    assert fully.mp_policy.param_dtype == torch.bfloat16
    assert fully.mp_policy.reduce_dtype == torch.float16
    assert fully.mp_policy.output_dtype == torch.float64
    assert shard_calls[1] == (
        hybrid_model,
        {
            "mesh": mesh,
            "mp_policy": hybrid.mp_policy,
            "reshard_after_forward": False,
        },
    )
    assert isinstance(hybrid.mp_policy, MixedPrecisionPolicy)
    assert hybrid.mp_policy.param_dtype == torch.bfloat16
    assert hybrid.mp_policy.reduce_dtype == torch.float16
    assert hybrid.mp_policy.output_dtype == torch.float64
    assert [record.message for record in caplog.records] == [
        "Applied FullySharded: mesh_dim=dp, reshard_after_forward=False",
        "Applied HybridSharded: replicate_dim=dp, shard_dim=tp, mesh_shape=(1, 1)",
    ]


def test_distributed_strategies_place_an_eager_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every strategy moves an already-built model to its device.

    The module's own contract makes device assignment step one, and a model
    built eagerly is already on the host: a strategy that only materializes
    meta state leaves it there, and the first forward meets parameters on one
    device and a batch on another.
    """

    def replicate_in_place(model: nn.Module, **kwargs: object) -> nn.Module:
        """Stand in for DDP's ``replicate``, which needs a process group."""
        del kwargs
        return model

    def ignore(
        module: nn.Module,
        mesh: DeviceMesh,
        mp_policy: MixedPrecisionPolicy | None,
        reshard_after_forward: bool,
    ) -> None:
        """Stand in for ``_shard``; placement is what this test measures."""
        del module, mesh, mp_policy, reshard_after_forward

    monkeypatch.setattr(parallelism, "global_device_mesh", _FakeMesh)
    monkeypatch.setattr(parallelism, "replicate", replicate_in_place)
    monkeypatch.setattr(parallelism, "_shard", ignore)
    configs = (
        DataParallel.Config(),
        FullySharded.Config(),
        HybridSharded.Config(replicate_dim="dp", shard_dim="tp"),
        RecursiveSharded.Config(module_types=(nn.Linear,)),
    )
    for config in configs:
        strategy = config.make()
        strategy.device = torch.device("meta")  # A device the model is NOT on.
        placed = strategy(SimpleModel())
        assert next(placed.parameters()).is_meta, type(config).__qualname__


# ``result_dir`` is bound via ``functools.partial`` and pickled with the worker, so it
# survives a forkserver started by an earlier test (an env var would be stale in that
# forkserver's snapshotted environment). Writes ``ok`` or ``FAIL:<reason>`` to
# ``rank_<r>`` under ``result_dir``.
def _fsdp_materialize_worker(result_dir: str, mesh: DeviceMesh) -> None:
    """Worker: shard a meta model under FSDP, record materialize outcome."""
    result_path = Path(result_dir)
    rank = mesh.get_rank()
    try:
        runtime._device_mesh = mesh
        torch.manual_seed(0)
        with torch.device("meta"):
            model = _TwoLinear()
        out = FullySharded.Config(mesh_dim="dp").make()(model)
        param = next(out.parameters())
        local = param.to_local() if isinstance(param, DTensor) else param
        if param.is_meta:
            (result_path / f"rank_{rank}").write_text("FAIL:still-meta")
        elif not torch.isfinite(local).all():
            (result_path / f"rank_{rank}").write_text("FAIL:non-finite")
        else:
            (result_path / f"rank_{rank}").write_text("ok")
    except (RuntimeError, OSError, ValueError, TypeError, KeyError) as e:
        (result_path / f"rank_{rank}").write_text(f"FAIL:{e!r}")
    finally:
        runtime._device_mesh = None


@pytest.mark.compute_distributed
def test_fully_sharded_materializes_meta_model_multirank(
    warm_pools: WarmPoolGetter,
) -> None:
    """FSDP shards a meta model across 2 ranks; each shard must materialize."""
    pool = warm_pools({"dp": 2})
    with tempfile.TemporaryDirectory() as tmp:
        pool(functools.partial(_fsdp_materialize_worker, tmp))
        results = {p.name: p.read_text() for p in Path(tmp).iterdir() if p.is_file()}

    assert results == {"rank_0": "ok", "rank_1": "ok"}, results


class _BNBlock(nn.Module):
    """A shardable block containing a BatchNorm whose stats must stay fp32."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(8, 8)
        self.bn = nn.BatchNorm1d(8)

    def reset_parameters(self) -> None:
        self.fc.reset_parameters()
        self.bn.reset_parameters()

    @override
    def forward(self, x: Tensor) -> Tensor:
        return self.bn(self.fc(x))


class _BNModel(nn.Module):
    """Root model wrapping a BN block (root distinct from the sharded block)."""

    def __init__(self) -> None:
        super().__init__()
        self.block = _BNBlock()
        self.head = nn.Linear(8, 8)

    def reset_parameters(self) -> None:
        self.block.reset_parameters()
        self.head.reset_parameters()

    @override
    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.block(x))


def test_module_mp_policy_forces_batchnorm_fp32() -> None:
    """#324: a bf16 base policy is overridden to float32 for BatchNorm modules.

    BatchNorm accumulates running statistics by reduction; doing that in bf16
    drifts the stats. ``_module_mp_policy`` must return an all-float32 policy
    for BatchNorm while leaving the base policy untouched for other modules.
    """
    base = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
    )
    bn_policy = _module_mp_policy(nn.BatchNorm1d(8), base)
    assert bn_policy is not None
    assert bn_policy.param_dtype == torch.float32
    assert bn_policy.reduce_dtype == torch.float32
    assert bn_policy.output_dtype == torch.float32
    # Non-BatchNorm keeps the base policy unchanged.
    assert _module_mp_policy(nn.Linear(8, 8), base) is base
    # No base policy -> no override (full precision everywhere already).
    assert _module_mp_policy(nn.BatchNorm1d(8), None) is None


def _bn_shard_worker(result_dir: str, mesh: DeviceMesh) -> None:
    """Worker: shard a BN-bearing model under bf16; confirm BN materializes ok."""
    result_path = Path(result_dir)
    rank = mesh.get_rank()
    try:
        runtime._device_mesh = mesh
        torch.manual_seed(0)
        model = _BNModel()
        out = RecursiveSharded.Config(
            mesh_dim="dp",
            module_types=(_BNBlock,),
            mp_param_dtype=torch.bfloat16,
            mp_reduce_dtype=torch.bfloat16,
        ).make()(model)
        bn = next(m for m in out.modules() if isinstance(m, nn.BatchNorm1d))
        weight = bn.weight
        local = weight.to_local() if isinstance(weight, DTensor) else weight
        ok = torch.isfinite(local).all()
        (result_path / f"rank_{rank}").write_text("ok" if ok else "FAIL:non-finite")
    except (RuntimeError, OSError, ValueError, TypeError, KeyError) as e:
        (result_path / f"rank_{rank}").write_text(f"FAIL:{e!r}")
    finally:
        runtime._device_mesh = None


@pytest.mark.compute_distributed
def test_recursive_sharded_shards_batchnorm_multirank(
    warm_pools: WarmPoolGetter,
) -> None:
    """#324: a BN-bearing model shards under bf16 FSDP without crashing."""
    pool = warm_pools({"dp": 2})
    with tempfile.TemporaryDirectory() as tmp:
        pool(functools.partial(_bn_shard_worker, tmp))
        results = {p.name: p.read_text() for p in Path(tmp).iterdir() if p.is_file()}
    assert results == {"rank_0": "ok", "rank_1": "ok"}, results


def test_tensor_parallel_lives_in_lib() -> None:
    """#298 reverses the staging: mesh-native TP is part of lib's surface."""
    train_pkg = importlib.import_module("priml.train")
    assert hasattr(train_pkg, "TensorParallel")
    assert hasattr(train_pkg, "apply_tensor_parallel")


def test_materialize_meta_initializes_centered_rmsnorm() -> None:
    """CenteredRMSNorm.weight (custom param, no torch reset) must init.

    Stay as ``to_empty`` garbage, after meta materialization.
    """
    with torch.device("meta"):
        norm = CenteredRMSNorm(CenteredRMSNorm.Config(channels_in=8))
    materialize_meta(norm, torch.device("cpu"))
    assert torch.isfinite(norm.weight).all()
    # Weight is used as ``1.0 + weight``; its init is zeros.
    # torch.equal not assert_close: to_empty garbage can be near-zero.
    assert torch.equal(norm.weight, torch.zeros(8))


def test_materialize_meta_initializes_gated_delta_net_raw_params() -> None:
    """GatedDeltaNet.dt_bias / A_log.

    Raw nn.Parameters, no reset) must hold their intended init after meta
    materialization, not garbage.
    """
    with torch.device("meta"):
        gdn = GatedDeltaNet(
            GatedDeltaNet.Config(
                channels_in=32,
                num_heads_k=2,
                num_heads_v=4,
                channels_k_head=8,
                channels_v_head=8,
            ).finalize(),
        )
    materialize_meta(gdn, torch.device("cpu"))
    assert torch.isfinite(gdn.dt_bias).all()
    assert torch.isfinite(gdn.A_log).all()
    torch.testing.assert_close(gdn.dt_bias, torch.ones_like(gdn.dt_bias))
    # A_log = log(uniform(0, 16)); finite and <= log(16).
    assert (gdn.A_log <= math.log(16.0) + 1e-4).all()


def test_materialize_meta_raises_on_uninitialized_param() -> None:
    """A param-bearing module whose params are never reset must fail loudly.

    Silently keep ``to_empty`` garbage.
    """

    class _Uninit(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(4))  # No reset_parameters.

    with torch.device("meta"):
        mod = _Uninit()
    with pytest.raises(
        RuntimeError,
        match="not initialized after materialize",
    ) as exc_info:
        materialize_meta(mod, torch.device("cpu"))
    assert str(exc_info.value) == (
        "State not initialized after materialize (a module did not reset a "
        "parameter or buffer it constructed): ['weight']"
    )


def test_materialize_meta_raises_on_uninitialized_buffer() -> None:
    """A registered buffer left unwritten by reset_parameters must fail loudly.

    Ship ``to_empty`` garbage. The audit covers buffers, not just params.
    """

    class _UninitBuf(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(4))
            self.register_buffer("stat", torch.zeros(4))

        def reset_parameters(self) -> None:
            nn.init.ones_(self.weight)  # Writes the param, forgets the buffer.

    with torch.device("meta"):
        mod = _UninitBuf()
    with pytest.raises(RuntimeError, match="not initialized after materialize"):
        materialize_meta(mod, torch.device("cpu"))


def test_materialize_meta_raises_on_partial_param_init() -> None:
    """A reset that writes only a slice leaves the rest as poison NaN.

    The ``.any()`` audit must catch the partial write, not pass it as initialized.
    """

    class _Partial(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(4))

        def reset_parameters(self) -> None:
            with torch.no_grad():
                self.weight[:2] = 1.0  # Only half written.

    with torch.device("meta"):
        mod = _Partial()
    with pytest.raises(RuntimeError, match="not initialized after materialize"):
        materialize_meta(mod, torch.device("cpu"))


def test_materialize_meta_allows_integer_buffer() -> None:
    """Materialize an integer buffer whose owner resets its allocated storage."""

    class _IntBuf(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(4))
            self.register_buffer("count", torch.zeros(1, dtype=torch.long))

        def reset_parameters(self) -> None:
            nn.init.ones_(self.weight)
            self.get_buffer("count").zero_()

    with torch.device("meta"):
        mod = _IntBuf()
    materialize_meta(mod, torch.device("cpu"))
    assert torch.isfinite(mod.weight).all()
    assert torch.equal(mod.get_buffer("count"), torch.zeros(1, dtype=torch.long))


def test_materialization_rejects_mixed_storage_before_discarding_weights() -> None:
    model = _TwoLinear()
    before = model.fc1.weight.detach().clone()
    model.fc2 = nn.Linear(8, 8, device="meta")
    with pytest.raises(ValueError, match="mixed"):
        NoParallel.Config(device="cpu").make()(model)
    assert torch.equal(model.fc1.weight, before)
    assert model.fc2.weight.is_meta


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.complex64])
def test_materialization_audits_every_nan_capable_dtype(dtype: torch.dtype) -> None:
    class ForgottenBuffer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("forgotten", torch.zeros(2, dtype=dtype))

    with torch.device("meta"):
        model = ForgottenBuffer()
    with pytest.raises(RuntimeError, match="forgotten"):
        materialize_meta(model, torch.device("cpu"))


@pytest.mark.parametrize(
    ("dtype", "extreme"),
    [(torch.uint8, 255), (torch.int8, -128), (torch.int64, -(2**63))],
)
def test_materialization_accepts_integer_extremes(
    dtype: torch.dtype,
    extreme: int,
) -> None:
    """Every integer value is a legal initial state; none can mark "unwritten"."""

    class ExtremeBuffer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("table", torch.empty(3, dtype=dtype))

        def reset_parameters(self) -> None:
            self.get_buffer("table").fill_(extreme)

    with torch.device("meta"):
        model = ExtremeBuffer()
    materialize_meta(model, torch.device("cpu"))
    assert torch.equal(
        model.get_buffer("table"),
        torch.full((3,), extreme, dtype=dtype),
    )


@pytest.mark.parametrize(
    "strategy_type",
    [FullySharded, HybridSharded, RecursiveSharded],
)
@pytest.mark.parametrize("dtype", [None, torch.bfloat16])
def test_batchnorm_is_separate_only_with_mixed_precision(
    monkeypatch: pytest.MonkeyPatch,
    strategy_type: type[FullySharded | HybridSharded | RecursiveSharded],
    dtype: torch.dtype | None,
) -> None:
    monkeypatch.setattr(parallelism, "global_device_mesh", _FakeMesh)
    sharded: list[nn.Module] = []

    def shard(module: nn.Module, **kwargs: object) -> None:
        del kwargs
        sharded.append(module)
        monkeypatch.setattr(module, "_get_fsdp_state", lambda: None, raising=False)

    monkeypatch.setattr(parallelism, "fully_shard", shard)
    config = strategy_type.Config()
    config.mp_param_dtype = dtype
    if isinstance(config, RecursiveSharded.Config):
        config.module_types = (_BNBlock,)
    model = _BNModel()
    config.make()(model)
    assert (model.block.bn in sharded) is (dtype is not None)
    assert sharded[-1] is model
    if dtype is not None:
        assert sharded.count(model.block.bn) == 1


def test_zero_match_plan_does_not_mutate_batchnorm_children(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(parallelism, "global_device_mesh", _FakeMesh)
    sharded: list[nn.Module] = []

    def shard(module: nn.Module, **kwargs: object) -> None:
        del kwargs
        sharded.append(module)

    monkeypatch.setattr(parallelism, "fully_shard", shard)
    config = RecursiveSharded.Config()
    config.module_types = (nn.Conv2d,)
    config.mp_param_dtype = torch.bfloat16
    with pytest.raises(ValueError, match="found 0"):
        config.make()(_BNModel())
    assert sharded == []


@pytest.mark.parametrize("dtype", [torch.bool, torch.uint8])
def test_materialization_accepts_initialized_discrete_buffers(
    dtype: torch.dtype,
) -> None:
    class InitializedBuffer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("state", torch.empty(2, dtype=dtype))

        def reset_parameters(self) -> None:
            self.get_buffer("state").fill_(True if dtype == torch.bool else 0)

    with torch.device("meta"):
        model = InitializedBuffer()
    materialize_meta(model, torch.device("cpu"))
    assert torch.equal(
        model.get_buffer("state"),
        torch.full((2,), True if dtype == torch.bool else 0, dtype=dtype),
    )


@pytest.mark.parametrize("rank", [0, 1])
def test_materialization_failure_reaches_every_rank(
    monkeypatch: pytest.MonkeyPatch,
    rank: int,
) -> None:
    class FailingReset(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.empty(2, 3))

        def reset_parameters(self) -> None:
            if rank == 0:
                raise RuntimeError("injected reset failure")
            nn.init.ones_(self.weight)

    observed: list[str | None] = []

    def gather(errors: list[str | None], error: str | None) -> None:
        observed.append(error)
        errors[:] = ["injected reset failure", None]

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    with torch.device("meta"):
        model = FailingReset()
    with pytest.raises(RuntimeError, match="injected reset failure"):
        materialize_meta(model, torch.device("cpu"))
    assert observed == (["injected reset failure"] if rank == 0 else [None])


class _RankFailingReset(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(2, 3))

    def reset_parameters(self) -> None:
        if torch.distributed.get_rank() == 0:
            raise RuntimeError("injected reset failure")
        nn.init.ones_(self.weight)


def _materialization_failure_worker(result_dir: str, mesh: DeviceMesh) -> None:
    with torch.device("meta"):
        model = _RankFailingReset()
    with pytest.raises(RuntimeError, match="injected reset failure"):
        materialize_meta(model, torch.device("cpu"))
    (Path(result_dir) / f"rank_{mesh.get_rank()}").write_text("ok")


@pytest.mark.compute_distributed
def test_materialization_failure_reaches_real_peers(
    warm_pools: WarmPoolGetter,
    tmp_path: Path,
) -> None:
    """Propagate a rank-zero reset failure across a real two-rank group."""
    pool = warm_pools({"dp": 2})
    pool(functools.partial(_materialization_failure_worker, str(tmp_path)))
    assert {p.name: p.read_text() for p in tmp_path.iterdir()} == {
        "rank_0": "ok",
        "rank_1": "ok",
    }


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
