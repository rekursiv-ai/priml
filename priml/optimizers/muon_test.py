"""Tests for Muon optimizer and lr_scale."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

import functools
import tempfile

from torch import nn
from torch.distributed.tensor import DTensor, Shard, distribute_tensor

import pytest
import torch

from priml import runtime
from priml.optimizers import lr_scale
from priml.optimizers.muon import Muon
from priml.testing.fixtures import (
    cleanup_cuda,  # noqa: F401 -- autouse fixture; imported for pytest collection
)


if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh

    from priml.distributed.testing import WarmPoolGetter


class TestLrScale:
    def test_warmup_start(self):
        assert lr_scale(0, total_steps=100, warmup_steps=10) == 0.0

    def test_warmup_midpoint(self):
        assert lr_scale(5, total_steps=100, warmup_steps=10) == 0.5

    def test_warmup_end(self):
        assert lr_scale(10, total_steps=100, warmup_steps=10) == 1.0

    def test_cosine_end(self):
        result = lr_scale(100, total_steps=100, warmup_steps=0)
        assert abs(result) < 1e-7

    def test_cosine_midpoint(self):
        result = lr_scale(50, total_steps=100, warmup_steps=0)
        assert abs(result - 0.5) < 1e-7

    def test_min_ratio(self):
        result = lr_scale(100, total_steps=100, warmup_steps=0, min_ratio=0.1)
        assert abs(result - 0.1) < 1e-7

    def test_no_warmup(self):
        assert lr_scale(0, total_steps=100, warmup_steps=0) == 1.0

    def test_zero_total_steps(self):
        # Should not raise.
        lr_scale(0, total_steps=0, warmup_steps=0)

    def test_monotone_decay(self):
        values = [lr_scale(s, total_steps=100, warmup_steps=0) for s in range(101)]
        for i in range(len(values) - 1):
            assert values[i] >= values[i + 1]

    def test_warmup_then_decay(self):
        # Warmup phase should be monotonically increasing
        warmup = [lr_scale(s, total_steps=100, warmup_steps=10) for s in range(11)]
        for i in range(len(warmup) - 1):
            assert warmup[i] <= warmup[i + 1]
        # Decay phase should be monotonically decreasing
        decay = [lr_scale(s, total_steps=100, warmup_steps=10) for s in range(10, 101)]
        for i in range(len(decay) - 1):
            assert decay[i] >= decay[i + 1]


class TestMuon:
    def _make_model(self, in_features: int = 8, out_features: int = 4) -> nn.Linear:
        return nn.Linear(in_features, out_features, bias=False)

    def test_basic_step(self):
        model = self._make_model()
        opt = Muon(model.parameters(), lr=0.02)

        x = torch.randn(2, 8)
        loss = model(x).sum()
        loss.backward()
        opt.step()
        opt.zero_grad()

    def test_weight_changes(self):
        model = self._make_model()
        opt = Muon(model.parameters(), lr=0.02)
        w_before = model.weight.data.clone()

        x = torch.randn(2, 8)
        loss = model(x).sum()
        loss.backward()
        opt.step()

        assert not torch.equal(model.weight.data, w_before)

    def test_multiple_steps(self):
        model = self._make_model()
        opt = Muon(model.parameters(), lr=0.02)

        for _ in range(5):
            x = torch.randn(2, 8)
            loss = model(x).sum()
            loss.backward()
            opt.step()
            opt.zero_grad()

    def test_weight_decay(self):
        model = self._make_model()
        opt_wd = Muon(model.parameters(), lr=0.02, weight_decay=0.1)

        x = torch.randn(2, 8)
        loss = model(x).sum()
        loss.backward()
        opt_wd.step()

    def test_no_weight_decay(self):
        model = self._make_model()
        opt = Muon(model.parameters(), lr=0.02, weight_decay=None)

        x = torch.randn(2, 8)
        loss = model(x).sum()
        loss.backward()
        opt.step()

    def test_nesterov_false(self):
        model = self._make_model()
        opt = Muon(model.parameters(), lr=0.02, nesterov=False)

        x = torch.randn(2, 8)
        loss = model(x).sum()
        loss.backward()
        opt.step()

    def test_rejects_1d_params(self):
        param = nn.Parameter(torch.randn(8))
        opt = Muon([param], lr=0.02)
        param.grad = torch.randn(8)

        with pytest.raises(ValueError, match="ndim >= 2"):
            opt.step()

    def test_invalid_lr(self):
        model = self._make_model()
        with pytest.raises(ValueError, match="learning rate"):
            Muon(model.parameters(), lr=-1.0)

    def test_invalid_momentum(self):
        model = self._make_model()
        with pytest.raises(ValueError, match="momentum"):
            Muon(model.parameters(), lr=0.02, momentum=-0.1)

    def test_invalid_weight_decay(self):
        model = self._make_model()
        with pytest.raises(ValueError, match="weight_decay"):
            Muon(model.parameters(), lr=0.02, weight_decay=-0.1)

    def test_nesterov_requires_momentum(self):
        model = self._make_model()
        with pytest.raises(ValueError, match="Nesterov"):
            Muon(model.parameters(), lr=0.02, momentum=0, nesterov=True)

    def test_adjust_lr_match_rms_adamw(self):
        model = self._make_model()
        opt = Muon(model.parameters(), lr=0.02, adjust_lr_fn="match_rms_adamw")

        x = torch.randn(2, 8)
        loss = model(x).sum()
        loss.backward()
        opt.step()

    def test_adjust_lr_conv_heuristic(self):
        model = self._make_model()
        opt = Muon(model.parameters(), lr=0.02, adjust_lr_fn="conv_heuristic")

        x = torch.randn(2, 8)
        loss = model(x).sum()
        loss.backward()
        opt.step()

    def test_state_dict_roundtrip(self):
        model = self._make_model()
        opt = Muon(model.parameters(), lr=0.02)

        x = torch.randn(2, 8)
        loss = model(x).sum()
        loss.backward()
        opt.step()

        state = opt.state_dict()
        opt2 = Muon(model.parameters(), lr=0.02)
        opt2.load_state_dict(state)


def _muon_shard_worker(result_dir: str, mesh: DeviceMesh) -> None:
    """Worker: one Muon step on a row-sharded param vs the full-tensor update.

    Newton-Schulz is a *global* spectral op: ``g.norm`` and ``g @ g.T`` must
    see the whole matrix. Muon passes the DTensor grad straight into the NS
    kernel; torch's DTensor dispatch redistributes ``Shard(0) @ Shard(0).T``
    into a collective, so the sharded update matches the single-device update
    (maxdiff ~0). This test pins that property: a regression to *naive
    per-shard* NS (independent NS on each 4-row block) would diverge by ~0.24.
    The worker writes ``ok`` or ``FAIL:<reason>`` (max-abs divergence) per rank.
    """
    result_path = Path(result_dir)
    rank = mesh.get_rank()
    try:
        runtime._device_mesh = mesh
        torch.manual_seed(0)
        weight = torch.randn(4, 4)
        grad = torch.randn(4, 4)

        full_param = nn.Parameter(weight.clone())
        full_param.grad = grad.clone()
        Muon([full_param], lr=0.1, momentum=0.0, nesterov=False).step()

        weight_dt: DTensor = distribute_tensor(weight.clone(), mesh, [Shard(0)])
        sharded = nn.Parameter(weight_dt)
        sharded.grad = distribute_tensor(grad.clone(), mesh, [Shard(0)])
        # Guard against a silent world_size=1 (no real shard): each rank must
        # hold exactly 2 of the 4 rows, else the test proves nothing.
        if tuple(weight_dt.to_local().shape) != (2, 4):
            (result_path / f"rank_{rank}").write_text(
                f"FAIL:not-sharded local={tuple(weight_dt.to_local().shape)}",
            )
            return
        Muon([sharded], lr=0.1, momentum=0.0, nesterov=False).step()

        got = cast("DTensor", sharded.data).full_tensor()
        max_abs = (got - full_param.detach()).abs().max().item()
        if max_abs > 1e-2:
            (result_path / f"rank_{rank}").write_text(f"FAIL:maxdiff={max_abs:.4f}")
        else:
            (result_path / f"rank_{rank}").write_text("ok")
    except Exception as e:  # noqa: BLE001 -- worker subprocess; any failure must reach the parent as a file
        (result_path / f"rank_{rank}").write_text(f"FAIL:{e!r}")
    finally:
        runtime._device_mesh = None


@pytest.mark.integration
def test_muon_sharded_matches_replicated_multirank(
    warm_pools: WarmPoolGetter,
) -> None:
    """Muon on a 2-way row-sharded param must equal the single-device update."""
    pool = warm_pools({"dp": 2})
    with tempfile.TemporaryDirectory() as tmp:
        pool(functools.partial(_muon_shard_worker, tmp))
        results = {p.name: p.read_text() for p in Path(tmp).iterdir() if p.is_file()}
    assert results == {"rank_0": "ok", "rank_1": "ok"}, results


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
