"""Shard declarations and multirank correctness for tensor parallelism.

#303: each block wires its children's shard style in ``__init__`` (mechanism
A), exactly where it wires ``channels_in``/``depth``; these tests assert the
built module carries the right ``shard`` style.

#302: ``EnsembleLinear``'s einsum cannot shard over a naive DTensor weight, so
it provides a custom ensemble-dim ``ParallelStyle``. The multirank test proves
its sharded output equals the dense output (cpu:gloo, tp=2).

#304: the generic applier runs over a cpu:gloo ``tp=2`` mesh and compares each
sharded model's output with its dense output. Attention uses ``SdpaNaive``
because the fused flash kernel has no DTensor sharding strategy; its guard is
asserted separately. MLA is covered here only while replicated.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, override

import functools
import tempfile

from torch import Tensor, nn
from torch.distributed.tensor import DTensor

import pytest
import torch

from priml import runtime
from priml.model.attention.kernel import SdpaNaive
from priml.model.attention.mla import MultiHeadLatentAttention
from priml.model.attention.multi_stream import MultiStreamAttention
from priml.model.attention.self_attention import SelfAttention
from priml.model.linear import EnsembleLinear, Linear
from priml.model.moe import MoE, Router
from priml.model.swiglu import SwiGLU
from priml.model.transformer.block import TransformerBlock
from priml.model.transformer.causal_lm import CausalLM
from priml.train.tensor_parallel import TensorParallel, apply_tensor_parallel


if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh

    from priml.distributed.testing import WarmPoolGetter


def test_self_attention_declares_qkv_colwise_out_rowwise() -> None:
    attn = SelfAttention.Config(channels_in=32, num_heads=4).make()
    assert attn.proj_qkv.shard == "colwise"
    assert attn.proj_out.shard == "rowwise"


def test_multistream_attention_declares_qkv_colwise_out_rowwise() -> None:
    attn = MultiStreamAttention.Config(
        channels_in=32, num_heads=4, num_streams=2
    ).make()
    assert all(qkv.shard == "colwise" for qkv in attn.proj_qkvs)
    assert all(out.shard == "rowwise" for out in attn.proj_outs)


def test_swiglu_declares_shard_via_transformer_block() -> None:
    block = TransformerBlock.Config(channels_in=32).make()
    assert isinstance(block.ffn, SwiGLU)
    assert block.ffn.shard == "colwise"
    assert block.ffn.up_proj.shard is None  # the SwiGLU style shards children
    assert block.ffn.down_proj.shard is None


def test_standalone_swiglu_defaults_to_no_shard() -> None:
    ffn = SwiGLU.Config(channels_in=32).make()
    assert ffn.shard is None


def test_moe_experts_inherit_swiglu_shard() -> None:
    block = TransformerBlock.Config(
        channels_in=32,
        ffn=MoE.Config(router=MoE.Config().router),
    ).make()
    assert isinstance(block.ffn, MoE)
    assert all(expert.shard == "colwise" for expert in block.ffn.experts)


def test_causal_lm_declares_embedding_and_head_vocab() -> None:
    model = CausalLM.Config(vocab_size=64, channels_in=32, num_layers=1).make()
    assert model.embed.shard == "vocab"
    assert isinstance(model.lm_head, Linear)
    assert model.lm_head.shard == "vocab"


def _ensemble_tp_worker(result_dir_str: str, mesh: DeviceMesh) -> None:
    """Worker: sharded EnsembleLinear output must equal the dense output.

    ``result_dir_str`` is bound via ``functools.partial`` and pickled with the
    worker, so it survives a forkserver started by an earlier test (an env var
    would be stale in that forkserver's snapshotted environment). Writes ``ok``
    or ``FAIL:<reason>`` to ``rank_<r>`` under it.
    """
    result_dir = Path(result_dir_str)
    rank = mesh.get_rank()
    try:
        runtime._device_mesh = mesh
        torch.manual_seed(0)
        model = EnsembleLinear.Config(
            channels_in=8,
            channels_out=5,
            num_ensemble=4,
            bias=True,
            shard="colwise",
        ).make()
        x = torch.randn(2, 3, 8)
        dense = model(x)
        sharded = apply_tensor_parallel(model, mesh)
        out = sharded(x)
        full = out.full_tensor() if hasattr(out, "full_tensor") else out
        if not torch.allclose(full, dense, rtol=1e-4, atol=1e-5):
            (result_dir / f"rank_{rank}").write_text(
                f"FAIL:mismatch max={float((full - dense).abs().max()):.2e}",
            )
        else:
            (result_dir / f"rank_{rank}").write_text("ok")
    except Exception as e:  # noqa: BLE001  -- surface any worker error to parent
        (result_dir / f"rank_{rank}").write_text(f"FAIL:{e!r}")
    finally:
        runtime._device_mesh = None


@pytest.mark.network_huggingface
def test_ensemble_linear_sharded_equals_dense_tp2(
    warm_pools: WarmPoolGetter,
) -> None:
    """#302: ensemble-dim ParallelStyle makes sharded output == dense (tp=2)."""
    pool = warm_pools({"dp": 1, "tp": 2})
    with tempfile.TemporaryDirectory() as tmp:
        pool(functools.partial(_ensemble_tp_worker, tmp))
        results = {p.name: p.read_text() for p in Path(tmp).iterdir() if p.is_file()}
    assert results == {"rank_0": "ok", "rank_1": "ok"}, results


@pytest.mark.network_huggingface
def test_meta_built_model_materializes_and_shards_tp2(
    warm_pools: WarmPoolGetter,
) -> None:
    """A ``device_init="meta"`` model survives placement AND real sharding."""
    pool = warm_pools({"dp": 1, "tp": 2})
    with tempfile.TemporaryDirectory() as tmp:
        pool(functools.partial(_meta_tp_worker, tmp))
        results = {p.name: p.read_text() for p in Path(tmp).iterdir() if p.is_file()}
    assert results == {"rank_0": "ok", "rank_1": "ok"}, results


class _ColwiseRowwisePair(nn.Module):
    """The canonical MLP TP shape: a colwise projection into a rowwise one.

    Declares ``reset_parameters`` because it CONSTRUCTS the two linears:
    ``materialize_meta`` drives init through the root alone, so a container
    that does not recurse into what it built leaves those children holding
    ``to_empty``'s garbage.
    """

    def __init__(self) -> None:
        super().__init__()
        self.up = Linear.Config(channels_in=8, channels_out=16, shard="colwise").make()
        self.down = Linear.Config(
            channels_in=16,
            channels_out=8,
            shard="rowwise",
        ).make()

    def reset_parameters(self) -> None:
        """Re-initialize both children."""
        self.up.reset_parameters()
        self.down.reset_parameters()

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Project up then back down."""
        return self.down(self.up(x))


def _meta_tp_worker(result_dir_str: str, mesh: DeviceMesh) -> None:
    """Worker: a meta-built model materializes AND shards, and still matches.

    The strategy's placement and the applier's sharding meet only here. A
    single-rank run exercises the materialize branch but returns before any
    plan is built (``tp=1`` is a structural no-op), so it cannot see a
    materialized tensor reaching ``parallelize_module``.
    """
    result_dir = Path(result_dir_str)
    rank = mesh.get_rank()
    try:
        runtime._device_mesh = mesh
        # One seed drives both builds, and each draws the same sequence from
        # it: the dense reference by constructing, the sharded one by
        # materializing. Copying weights in afterwards is not available -- a
        # sharded parameter is a DTensor and rejects a plain tensor -- so the
        # streams are made to agree instead.
        torch.manual_seed(0)
        reference = _ColwiseRowwisePair()
        x = torch.randn(4, 8)
        dense = reference(x)

        torch.manual_seed(0)
        with torch.device("meta"):
            model = _ColwiseRowwisePair()
        assert all(p.is_meta for p in model.parameters())
        placed = TensorParallel.Config().make()(model)

        out = placed(x)
        full = out.full_tensor() if hasattr(out, "full_tensor") else out
        deviation = float((full - dense).abs().max())
        if any(p.is_meta for p in placed.parameters()):
            (result_dir / f"rank_{rank}").write_text("FAIL:still meta")
        elif deviation > 1e-5:
            (result_dir / f"rank_{rank}").write_text(
                f"FAIL:mismatch max={deviation:.2e}"
            )
        else:
            (result_dir / f"rank_{rank}").write_text("ok")
    except Exception as e:  # noqa: BLE001  -- surface any worker error to parent
        (result_dir / f"rank_{rank}").write_text(f"FAIL:{e!r}")
    finally:
        runtime._device_mesh = None


def _swiglu() -> tuple[nn.Module, Tensor]:
    return SwiGLU.Config(channels_in=32, shard="colwise").make(), torch.randn(2, 5, 32)


def _self_attention() -> tuple[nn.Module, Tensor]:
    attn = SelfAttention.Config(
        channels_in=32,
        num_heads=4,
        num_heads_kv=2,
        attn_kernel=SdpaNaive.Config(),
    ).make()
    return attn, torch.randn(2, 6, 32)


def _moe() -> tuple[nn.Module, Tensor]:
    moe = MoE.Config(
        channels_in=32,
        router=Router.Config(num_experts=4, top_k=2),
    ).make()
    return moe, torch.randn(2, 5, 32)


def _transformer_block() -> tuple[nn.Module, Tensor]:
    block = TransformerBlock.Config(
        channels_in=32,
        attn=SelfAttention.Config(
            num_heads=4,
            num_heads_kv=2,
            attn_kernel=SdpaNaive.Config(),
        ),
    ).make()
    return block, torch.randn(2, 6, 32)


def _causal_lm() -> tuple[nn.Module, Tensor]:
    model = CausalLM.Config(
        vocab_size=64,
        channels_in=32,
        num_layers=2,
        block=TransformerBlock.Config(
            attn=SelfAttention.Config(
                num_heads=4,
                num_heads_kv=2,
                attn_kernel=SdpaNaive.Config(),
            ),
        ),
    ).make()
    return model, torch.randint(0, 64, (2, 6))


def _mla_replicated() -> tuple[nn.Module, Tensor]:
    # MLA with shard="none": the applier leaves it replicated, so sharded ==
    # dense holds trivially. The head-parallel style is covered separately in
    # attention/mla_test.py.
    mla = MultiHeadLatentAttention.Config(channels_in=32, num_heads=4).make()
    return mla, torch.randn(2, 6, 32)


_CASES: dict[str, Callable[[], tuple[nn.Module, Tensor]]] = {
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


def _first(out: Tensor | tuple[Tensor, object]) -> Tensor:
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
    build: Callable[[], tuple[nn.Module, Tensor]],
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
        attn = SelfAttention.Config(channels_in=32, num_heads=4, num_heads_kv=2).make()
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


@pytest.mark.network_huggingface
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
