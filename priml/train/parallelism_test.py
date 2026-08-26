"""Tests for parallelism strategies."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, override

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
    _module_mp_policy,
    materialize_meta,
)


if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh

    from priml.distributed.testing import WarmPoolGetter


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


class _BufferOnly(nn.Module):
    """Module with only a floating-point buffer."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("stat", torch.zeros(4))

    def reset_parameters(self) -> None:
        self.stat.fill_(1.0)


def test_no_parallel_materializes_meta_buffer_only_model() -> None:
    """NoParallel materializes meta buffers even when no parameters exist."""
    with torch.device("meta"):
        model = _BufferOnly()
    assert model.stat.is_meta

    result = NoParallel.Config(device="cpu").make()(model)

    assert not result.stat.is_meta
    assert result.stat.device.type == "cpu"
    torch.testing.assert_close(result.stat, torch.ones(4))


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
    with pytest.raises(RuntimeError, match="DataParallel requires distributed mode"):
        config.make()


def test_fully_sharded_requires_distributed():
    config = FullySharded.Config()
    with pytest.raises(RuntimeError, match="FullySharded requires distributed mode"):
        config.make()


def test_hybrid_sharded_requires_distributed():
    config = HybridSharded.Config()
    with pytest.raises(RuntimeError, match="HybridSharded requires distributed mode"):
        config.make()


def test_recursive_sharded_requires_distributed():
    config = RecursiveSharded.Config()
    with pytest.raises(
        RuntimeError,
        match="RecursiveSharded requires distributed mode",
    ):
        config.make()


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
    with pytest.raises(ValueError, match="requires module_types"):
        RecursiveSharded.Config().make()


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


def test_distributed_strategies_place_an_eager_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every strategy moves an already-built model to its device.

    The module's own contract makes device assignment step one, and a model
    built eagerly is already on the host: a strategy that only materializes
    meta state leaves it there, and the first forward meets parameters on one
    device and a batch on another.
    """

    def replicate_in_place(model: nn.Module, **kwargs: Any) -> nn.Module:
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
        strategy.device = torch.device("meta")  # a device the model is NOT on
        placed = strategy(SimpleModel())
        assert next(placed.parameters()).is_meta, type(config).__qualname__


def _fsdp_materialize_worker(result_dir: str, mesh: DeviceMesh) -> None:
    """Worker: shard a meta model under FSDP, record materialize outcome.

    ``result_dir`` is bound via ``functools.partial`` and pickled with the
    worker, so it survives a forkserver started by an earlier test (an env
    var would be stale in that forkserver's snapshotted environment).
    Writes ``ok`` or ``FAIL:<reason>`` to ``rank_<r>`` under ``result_dir``.
    """
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
    except Exception as e:  # noqa: BLE001  -- surface any worker error to parent
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
        ok = local is not None and torch.isfinite(local).all()
        (result_path / f"rank_{rank}").write_text("ok" if ok else "FAIL:non-finite")
    except Exception as e:  # noqa: BLE001  -- surface any worker error to parent
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
    """CenteredRMSNorm.weight (custom param, no torch reset) must init, not stay
    as ``to_empty`` garbage, after meta materialization.
    """
    with torch.device("meta"):
        norm = CenteredRMSNorm(CenteredRMSNorm.Config(channels_in=8))
    materialize_meta(norm, torch.device("cpu"))
    assert torch.isfinite(norm.weight).all()
    # weight is used as ``1.0 + weight``; its init is zeros.
    # torch.equal not assert_close: to_empty garbage can be near-zero.
    assert torch.equal(norm.weight, torch.zeros(8))


def test_materialize_meta_initializes_gated_delta_net_raw_params() -> None:
    """GatedDeltaNet.dt_bias / A_log (raw nn.Parameters, no reset) must hold
    their intended init after meta materialization, not garbage.
    """
    with torch.device("meta"):
        gdn = GatedDeltaNet(
            GatedDeltaNet.Config(
                channels_in=32,
                num_heads_k=2,
                num_heads_v=4,
                channels_k_head=8,
                channels_v_head=8,
            ).finalize()
        )
    materialize_meta(gdn, torch.device("cpu"))
    assert torch.isfinite(gdn.dt_bias).all()
    assert torch.isfinite(gdn.A_log).all()
    torch.testing.assert_close(gdn.dt_bias, torch.ones_like(gdn.dt_bias))
    # A_log = log(uniform(0, 16)); finite and <= log(16).
    assert (gdn.A_log <= math.log(16.0) + 1e-4).all()


def test_materialize_meta_raises_on_uninitialized_param() -> None:
    """A param-bearing module whose params are never reset must fail loudly,
    not silently keep ``to_empty`` garbage.
    """

    class _Uninit(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(4))  # no reset_parameters

    with torch.device("meta"):
        mod = _Uninit()
    with pytest.raises(RuntimeError, match="not initialized after materialize"):
        materialize_meta(mod, torch.device("cpu"))


def test_materialize_meta_raises_on_uninitialized_buffer() -> None:
    """A registered buffer left unwritten by reset_parameters must fail loudly,
    not ship ``to_empty`` garbage. The audit covers buffers, not just params.
    """

    class _UninitBuf(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(4))
            self.register_buffer("stat", torch.zeros(4))

        def reset_parameters(self) -> None:
            nn.init.ones_(self.weight)  # writes the param, forgets the buffer

    with torch.device("meta"):
        mod = _UninitBuf()
    with pytest.raises(RuntimeError, match="not initialized after materialize"):
        materialize_meta(mod, torch.device("cpu"))


def test_materialize_meta_raises_on_partial_param_init() -> None:
    """A reset that writes only a slice leaves the rest as poison NaN; the
    ``.any()`` audit must catch the partial write, not pass it as initialized.
    """

    class _Partial(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(4))

        def reset_parameters(self) -> None:
            with torch.no_grad():
                self.weight[:2] = 1.0  # only half written

    with torch.device("meta"):
        mod = _Partial()
    with pytest.raises(RuntimeError, match="not initialized after materialize"):
        materialize_meta(mod, torch.device("cpu"))


def test_materialize_meta_allows_integer_buffer() -> None:
    """Integer buffers (e.g. counters) cannot hold NaN and must not trip the
    float-only poison audit; a valid module with one must materialize cleanly.
    """

    class _IntBuf(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(4))
            self.register_buffer("count", torch.zeros(1, dtype=torch.long))

        def reset_parameters(self) -> None:
            nn.init.ones_(self.weight)

    with torch.device("meta"):
        mod = _IntBuf()
    materialize_meta(mod, torch.device("cpu"))
    assert torch.isfinite(mod.weight).all()


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
