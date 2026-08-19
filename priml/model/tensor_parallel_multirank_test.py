"""Multirank tensor-parallel correctness (#304): sharded output == dense.

Runs the generic ``apply_tensor_parallel`` applier over a cpu:gloo ``tp=2``
mesh (via :class:`WorkerPool`) and asserts each sharded model's forward equals
its dense forward within float reassociation tolerance. This proves the
declared shard plan is *correct*, not merely non-crashing.

Tensor-parallel attention requires a DTensor-compatible kernel: the fused
flash kernel has no DTensor sharding strategy, so these models use
``SdpaNaive``. The fused-kernel guard is asserted separately.

This file covers MLA only in its replicated (``shard="none"``) form: a
``tp=2`` MLA model with no shard style runs replicated and equals dense
trivially. The head-parallel MLA style (``shard="colwise"``, #307) is proven
sharded==dense in ``mla_tensor_parallel_test.py``.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import functools
import tempfile

from torch.distributed.tensor import DTensor

import pytest
import torch

from priml import runtime
from priml.model.attention import SdpaNaive, SelfAttention
from priml.model.causal_lm import CausalLM
from priml.model.mla import MultiHeadLatentAttention
from priml.model.moe import MoE, Router
from priml.model.swiglu import SwiGLU
from priml.model.transformer import TransformerBlock
from priml.train.tensor_parallel import apply_tensor_parallel


if TYPE_CHECKING:
    from torch import nn
    from torch.distributed.device_mesh import DeviceMesh

    from priml.distributed.testing import WarmPoolGetter


pytestmark = pytest.mark.network_huggingface


def _swiglu() -> tuple[nn.Module, torch.Tensor]:
    return SwiGLU.Config(channels_in=32, shard="colwise").make(), torch.randn(2, 5, 32)


def _self_attention() -> tuple[nn.Module, torch.Tensor]:
    attn = SelfAttention.Config(
        channels_in=32,
        heads=4,
        num_heads_kv=2,
        attn_kernel=SdpaNaive.Config(),
    ).make()
    return attn, torch.randn(2, 6, 32)


def _moe() -> tuple[nn.Module, torch.Tensor]:
    moe = MoE.Config(
        channels_in=32,
        router=Router.Config(num_experts=4, top_k=2),
    ).make()
    return moe, torch.randn(2, 5, 32)


def _transformer_block() -> tuple[nn.Module, torch.Tensor]:
    block = TransformerBlock.Config(
        channels_in=32,
        attn=SelfAttention.Config(
            heads=4,
            num_heads_kv=2,
            attn_kernel=SdpaNaive.Config(),
        ),
    ).make()
    return block, torch.randn(2, 6, 32)


def _causal_lm() -> tuple[nn.Module, torch.Tensor]:
    model = CausalLM.Config(
        vocab_size=64,
        channels=32,
        num_layers=2,
        block=TransformerBlock.Config(
            attn=SelfAttention.Config(
                heads=4,
                num_heads_kv=2,
                attn_kernel=SdpaNaive.Config(),
            ),
        ),
    ).make()
    return model, torch.randint(0, 64, (2, 6))


def _mla_replicated() -> tuple[nn.Module, torch.Tensor]:
    # MLA with shard="none": the applier leaves it replicated, so sharded ==
    # dense holds trivially. The head-parallel style is covered separately in
    # mla_tensor_parallel_test.py.
    mla = MultiHeadLatentAttention.Config(channels_in=32, heads=4).make()
    return mla, torch.randn(2, 6, 32)


_CASES: dict[str, Callable[[], tuple[nn.Module, torch.Tensor]]] = {
    "swiglu": _swiglu,
    "self_attention": _self_attention,
    "moe": _moe,
    "transformer_block": _transformer_block,
    "causal_lm": _causal_lm,
    "mla_replicated": _mla_replicated,
}

# Control cases that are SUPPOSED to stay replicated (shard="none"); the
# silent-replication guard must not fire on them.
_REPLICATED_CASES = frozenset({"mla_replicated"})


def _first(out: torch.Tensor | tuple[torch.Tensor, object]) -> torch.Tensor:
    """Unwrap a model output that may be a ``(tensor, cache)`` tuple."""
    return out[0] if isinstance(out, tuple) else out


def _all_cases_worker(result_dir_str: str, mesh: DeviceMesh) -> None:
    """Run every sharded==dense case plus the fused-kernel guard in one pool.

    Each ``WorkerPool`` spawn is expensive and repeated spawns in a single
    process corrupt the multiprocessing forkserver, so all cases share one
    pool. ``result_dir_str`` is bound via ``functools.partial`` and pickled
    with the worker, so it survives a forkserver started by an earlier test
    (an env var would be stale in that forkserver's snapshotted environment).
    Writes ``<case>_rank<r>`` files the parent collates.
    """
    result_dir = Path(result_dir_str)
    rank = mesh.get_rank()
    runtime._device_mesh = mesh
    try:
        for case, build in _CASES.items():
            _record_case(result_dir, case, rank, build, mesh)
        _record_fused_kernel_guard(result_dir, rank, mesh)
    finally:
        runtime._device_mesh = None


def _record_case(
    result_dir: Path,
    case: str,
    rank: int,
    build: Callable[[], tuple[nn.Module, torch.Tensor]],
    mesh: DeviceMesh,
) -> None:
    """Assert one case's sharded forward equals dense; record the outcome."""
    target = result_dir / f"{case}_rank{rank}"
    try:
        torch.manual_seed(0)
        model, x = build()
        dense = _first(model(x))
        sharded = apply_tensor_parallel(model, mesh)
        # Guard against silent replication: for a case meant to shard,
        # sharded==dense is vacuous if the plan was a no-op. At least one
        # parameter must be a real DTensor for the equality to mean anything.
        # Replicated control cases (shard="none") are exempt by design.
        sharding_expected = case not in _REPLICATED_CASES
        has_dtensor = any(isinstance(p, DTensor) for p in sharded.parameters())
        if sharding_expected and not has_dtensor:
            target.write_text("FAIL:no-dtensor-param (silently replicated?)")
            return
        out = _first(sharded(x))
        full = out.full_tensor() if isinstance(out, DTensor) else out
        if torch.allclose(full, dense, rtol=1e-4, atol=1e-5):
            target.write_text("ok")
        else:
            target.write_text(f"FAIL:max={float((full - dense).abs().max()):.2e}")
    except Exception as e:  # noqa: BLE001  -- surface any worker error to parent
        target.write_text(f"FAIL:{e!r}")


def _record_fused_kernel_guard(result_dir: Path, rank: int, mesh: DeviceMesh) -> None:
    """A sharded SelfAttention with the fused flash kernel must raise clearly."""
    target = result_dir / f"fused_guard_rank{rank}"
    try:
        torch.manual_seed(0)
        attn = SelfAttention.Config(channels_in=32, heads=4, num_heads_kv=2).make()
        try:
            apply_tensor_parallel(attn, mesh)
        except RuntimeError as e:
            ok = "DTensor-compatible attention kernel" in str(e)
            target.write_text("ok" if ok else f"FAIL:{e!r}")
        else:
            target.write_text("FAIL:no-raise")
    except Exception as e:  # noqa: BLE001  -- surface any worker error to parent
        target.write_text(f"FAIL:{e!r}")


def _run_all_cases(get_pool: WarmPoolGetter) -> dict[str, str]:
    pool = get_pool({"dp": 1, "tp": 2})
    with tempfile.TemporaryDirectory() as tmp:
        pool(functools.partial(_all_cases_worker, tmp))
        return {p.name: p.read_text() for p in Path(tmp).iterdir() if p.is_file()}


def test_sharded_equals_dense_and_guard_tp2(warm_pools: WarmPoolGetter) -> None:
    """All TP cases produce sharded==dense; the fused-kernel guard fires."""
    results = _run_all_cases(warm_pools)
    expected_cases = [*_CASES, "fused_guard"]
    expected = {
        f"{case}_rank{rank}": "ok" for case in expected_cases for rank in (0, 1)
    }
    assert results == expected, results


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
