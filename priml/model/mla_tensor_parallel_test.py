"""Multirank tensor-parallel correctness for MLA (#307): sharded == dense.

MLA is the hardest tensor-parallel layer in this library: its absorb-math
forward reshapes ``kv_b_proj.weight`` per head and contracts the q-path in
the latent space, so the head dim cannot be sharded by the generic
colwise/rowwise styles alone. The custom :class:`MultiHeadLatentAttention`
``ParallelStyle`` shards ``q_proj``/``q_b_proj`` (colwise, head dim) and
``o_proj`` (rowwise), leaves the head-shared latent path replicated, and makes
the absorb-math rank-local (each rank expands only its ``heads // tp`` heads).

This test runs that style over a cpu:gloo ``tp=2`` mesh (via
:class:`WorkerPool`) and asserts the sharded forward equals the dense forward
within float reassociation tolerance, proving the plan is *correct*. The
``heads`` count must divide ``tp``; the indivisible case is asserted to raise.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import functools
import tempfile

from torch.distributed.tensor import DTensor

import pytest
import torch

from priml import runtime
from priml.model.mla import MultiHeadLatentAttention
from priml.model.rope import RoPE
from priml.train.tensor_parallel import apply_tensor_parallel


if TYPE_CHECKING:
    from torch import nn
    from torch.distributed.device_mesh import DeviceMesh

    from priml.distributed.testing import WarmPoolGetter


pytestmark = pytest.mark.integration


def _mla(*, q_lora_rank: int | None = None) -> tuple[nn.Module, torch.Tensor]:
    """Small MLA exercising the head-parallel q/o paths and latent absorb."""
    mla = MultiHeadLatentAttention.Config(
        channels_in=16,
        heads=2,
        channels_qk_nope_head=4,
        channels_qk_rope_head=4,
        channels_v_head=4,
        kv_lora_rank=8,
        q_lora_rank=q_lora_rank,
        rope=RoPE.Config(channels_head=4, base=10_000),
        shard="colwise",
    ).make()
    return mla, torch.randn(1, 3, 16)


def _first(out: torch.Tensor | tuple[torch.Tensor, object]) -> torch.Tensor:
    """Unwrap a model output that may be a ``(tensor, cache)`` tuple."""
    return out[0] if isinstance(out, tuple) else out


def _record_case(
    result_dir: Path,
    case: str,
    rank: int,
    q_lora_rank: int | None,
    mesh: DeviceMesh,
) -> None:
    """Assert one MLA variant's sharded forward equals dense; record outcome."""
    target = result_dir / f"{case}_rank{rank}"
    try:
        torch.manual_seed(0)
        model, x = _mla(q_lora_rank=q_lora_rank)
        dense = _first(model(x))
        sharded = apply_tensor_parallel(model, mesh)
        # Guard against a silent replicated no-op passing trivially: the q-path
        # weight must genuinely become a head-sharded DTensor.
        q = sharded.q_proj if sharded.q_proj is not None else sharded.q_b_proj
        if not isinstance(q.weight, DTensor):
            target.write_text("FAIL:q-path-not-sharded")
            return
        out = _first(sharded(x))
        full = out.full_tensor() if isinstance(out, DTensor) else out
        if torch.allclose(full, dense, rtol=1e-4, atol=1e-5):
            target.write_text("ok")
        else:
            target.write_text(f"FAIL:max={float((full - dense).abs().max()):.2e}")
    except Exception as e:  # noqa: BLE001  -- surface any worker error to parent
        target.write_text(f"FAIL:{e!r}")


def _record_indivisible_guard(result_dir: Path, rank: int, mesh: DeviceMesh) -> None:
    """``tp`` must divide ``heads``; otherwise the style raises clearly."""
    target = result_dir / f"indivisible_rank{rank}"
    try:
        torch.manual_seed(0)
        mla = MultiHeadLatentAttention.Config(
            channels_in=16,
            heads=3,
            channels_qk_nope_head=4,
            channels_qk_rope_head=4,
            channels_v_head=4,
            kv_lora_rank=8,
            shard="colwise",
        ).make()
        try:
            apply_tensor_parallel(mla, mesh)
        except ValueError as e:
            ok = "heads" in str(e) and "divis" in str(e).lower()
            target.write_text("ok" if ok else f"FAIL:{e!r}")
        else:
            target.write_text("FAIL:no-raise")
    except Exception as e:  # noqa: BLE001  -- surface any worker error to parent
        target.write_text(f"FAIL:{e!r}")


def _mla_tp_worker(result_dir_str: str, mesh: DeviceMesh) -> None:
    """Run every MLA sharded==dense case plus the indivisible guard in one pool.

    ``result_dir_str`` is bound via ``functools.partial`` and pickled with the
    worker, so it survives a forkserver started by an earlier test (an env var
    would be stale in that forkserver's snapshotted environment). Writes
    ``<case>_rank<r>`` files the parent collates.
    """
    result_dir = Path(result_dir_str)
    rank = mesh.get_rank()
    runtime._device_mesh = mesh
    try:
        _record_case(result_dir, "no_lora", rank, None, mesh)
        _record_case(result_dir, "q_lora", rank, 8, mesh)
        _record_indivisible_guard(result_dir, rank, mesh)
    finally:
        runtime._device_mesh = None


def test_mla_sharded_equals_dense_tp2(warm_pools: WarmPoolGetter) -> None:
    """#307: the custom MLA style makes sharded output == dense (tp=2)."""
    pool = warm_pools({"dp": 1, "tp": 2})
    with tempfile.TemporaryDirectory() as tmp:
        pool(functools.partial(_mla_tp_worker, tmp))
        results = {p.name: p.read_text() for p in Path(tmp).iterdir() if p.is_file()}
    expected = {
        f"{case}_rank{rank}": "ok"
        for case in ("no_lora", "q_lora", "indivisible")
        for rank in (0, 1)
    }
    assert results == expected, results


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
