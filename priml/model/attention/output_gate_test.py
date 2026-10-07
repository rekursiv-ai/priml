"""Tests for attention module."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pytest
import torch

from priml.cost import cost
from priml.model.attention.attention import Attention
from priml.model.attention.kernel import SdpaNaive
from priml.model.attention.kvcache import KVCache, alloc_layer_cache
from priml.model.attention.output_gate import OutputGate
from priml.model.attention.rope import RoPE
from priml.model.norm import RMSNorm
from priml.testing.bfb import assert_bfb_against_golden, bfb_devices
from priml.testing.cost import assert_cost_matches_torch
from priml.testing.golden import assert_pprint_golden


_CWD: Final = Path(__file__).resolve().parent


def test_output_gate_config_pprint() -> None:
    config = OutputGate.Config(
        channels_in=16,
        inner=Attention.Config(num_heads=2, channels_head=8),
    )
    assert_pprint_golden(
        test_file=__file__,
        name="output_gate",
        config=config,
    )


def test_output_gate_mismatched_widths_reject_at_make() -> None:
    config = OutputGate.Config()
    config.channels_in = 8
    config.channels_out = 16

    with pytest.raises(ValueError, match="channels_in=8 must equal channels_out=16"):
        config.make()


def test_output_gate_basic():
    m = OutputGate.Config(
        channels_in=4,
        inner=Attention.Config(
            channels_in=4,
            num_heads=2,
            channels_head=2,
            causal=True,
        ),
    ).make()
    with torch.no_grad():
        m.gate_proj.weight.copy_(torch.tensor([[1.0, 0, 0, 0]] * 4))
    x = torch.tensor([[[0.0, 1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0]]])
    out = m(x)
    expected = m.inner(x) * torch.sigmoid(m.gate_proj(x))
    assert torch.equal(out, expected)
    assert out.shape == (1, 2, 4)


def test_output_gate_cached():
    m = OutputGate.Config(
        channels_in=64,
        inner=Attention.Config(
            channels_in=64,
            num_heads=4,
            channels_head=16,
            causal=True,
        ),
        depth_index=((0, 1),),
    ).make()
    cache = alloc_layer_cache(m, batch=2, max_seq=8)

    out = m(torch.randn(2, 8, 64), cache=cache)

    assert out.shape == (2, 8, 64)
    assert isinstance(m.inner, Attention)
    state = cache[m.inner.depth_index]
    assert isinstance(state, KVCache)
    assert state.length == 8


def test_output_gate_passthrough_kwargs():
    rope = RoPE.Config(channels_head=16).make()
    m = OutputGate.Config(
        channels_in=64,
        inner=Attention.Config(
            channels_in=64,
            num_heads=4,
            channels_head=16,
        ),
    ).make()
    x = torch.randn(2, 8, 64)
    cos, sin = rope(torch.arange(8))
    out = m(x, cos_sin=(cos, sin))
    assert out.shape == (2, 8, 64)


def test_output_gate_reset():
    m = OutputGate.Config(
        channels_in=64,
        inner=Attention.Config(channels_in=64, num_heads=4, channels_head=16),
    ).make()
    m.reset_parameters()


def test_output_gate_finalize_propagates():
    cfg = OutputGate.Config(
        channels_in=128,
        inner=Attention.Config(num_heads=4, channels_head=32),
    ).finalize()
    assert isinstance(cfg.inner, Attention.Config)
    assert cfg.inner.channels_in == 128


def test_output_gate_geometry_rejects_an_unheaded_inner() -> None:
    """The gate mirrors its inner attention's geometry, never a stand-in."""
    cfg = OutputGate.Config(channels_in=8)
    cfg.inner = RMSNorm.Config(channels_in=8)

    with pytest.raises(AttributeError, match="num_heads"):
        _ = cfg.num_heads
    with pytest.raises(AttributeError, match="channels_head"):
        _ = cfg.channels_head


@pytest.mark.parametrize("device", bfb_devices(), ids=str)
def test_output_gate_bfb(device: str) -> None:
    assert_bfb_against_golden(
        golden_dir=_CWD / "testdata",
        golden_name="output_gate",
        build_module=lambda: (
            OutputGate.Config(
                channels_in=4,
                inner=Attention.Config(num_heads=2, channels_head=2),
            )
            .make()
            .to(device)
        ),
        build_input=lambda: torch.randn(2, 3, 4),
        seed=0,
    )


def test_output_gate_cost_is_the_inner_plus_a_square_gate_matmul() -> None:
    config = OutputGate.Config(
        channels_in=16,
        inner=Attention.Config(num_heads=2, channels_head=8),
    )
    config.inner = Attention.Config(
        num_heads=2,
        channels_head=8,
        attn_kernel=SdpaNaive.Config(),
    )
    model_cost = assert_cost_matches_torch(
        config,
        build_input=lambda: torch.randn(2, 3, 16, requires_grad=True),
        seq_len=3,
        batch_size=2,
        dtype=None,
    )
    inner = cost(
        config.copy_tree().finalize().inner,
        seq_len=3,
        batch_size=2,
        dtype=None,
    )
    gate = 16 * 16
    assert (
        model_cost["flops", "primal", "matmul"].sum()
        == inner[
            "flops",
            "primal",
            "matmul",
        ].sum()
        + 4 * 3 * gate
    )
    assert (
        model_cost["flops", "adjoint", "matmul"].sum()
        == inner[
            "flops",
            "adjoint",
            "matmul",
        ].sum()
        + 8 * 3 * gate
    )
    assert model_cost.params == inner.params + gate
    assert model_cost.bytes_state == inner.bytes_state


def test_output_gate_traffic_prices_projection_and_scalar_operands() -> None:
    config = OutputGate.Config()
    config.channels_in = 8
    config.inner = RMSNorm.Config()
    config = config.copy_tree().finalize()
    inner = cost(config.inner, seq_len=4, batch_size=1, dtype=torch.bfloat16)
    actual = config.cost(seq_len=4, batch_size=1, dtype=torch.bfloat16)
    assert actual["bytes", "primal", "matmul"].sum() == 2 * (4 * 8 + 4 * 8 + 64)
    assert (
        actual["bytes", "primal", "elementwise"].sum()
        - inner["bytes", "primal", "elementwise"].sum()
        == 2 * 5 * 8 * 4
    )
    assert (
        actual["bytes", "adjoint", "elementwise"].sum()
        - inner["bytes", "adjoint", "elementwise"].sum()
        == 2 * 12 * 8 * 4
    )
    wide = config.cost(seq_len=4, batch_size=1, dtype=None)
    assert (
        wide["bytes", torch.float32].sum() == actual["bytes", torch.bfloat16].sum() * 2
    )


def test_output_gate_allocates_one_logical_cache_slot() -> None:
    model = OutputGate.Config(
        channels_in=12,
        inner=Attention.Config(num_heads=3, channels_head=4),
        depth_index=((0, 1),),
    ).make()
    cache = alloc_layer_cache(
        model,
        batch=2,
        max_seq=5,
        device="meta",
        dtype=torch.bfloat16,
    )
    assert isinstance(model.inner, Attention)
    assert list(cache) == [model.inner.depth_index]
    state = cache[model.inner.depth_index]
    assert isinstance(state, KVCache)
    assert state.k.shape == (2, 3, 5, 4)
    assert state.v.shape == (2, 3, 5, 4)
    assert state.k.device == state.v.device == torch.device("meta")
    assert state.k.dtype == state.v.dtype == torch.bfloat16


def test_output_gate_cached_forwards_kwargs_and_multiplies_by_gate() -> None:
    model = OutputGate.Config(
        channels_in=12,
        inner=Attention.Config(
            num_heads=3,
            channels_head=4,
            attn_kernel=SdpaNaive.Config(),
        ),
        depth_index=((0, 1),),
    ).make()
    with torch.no_grad():
        model.gate_proj.weight.zero_()
    x = torch.arange(2 * 5 * 12, dtype=torch.float32).reshape(2, 5, 12) / 100
    # Attention masks are [batch, heads, sequence, sequence].
    mask = torch.full((2, 3, 5, 5), -float("inf"))
    mask[..., 0] = 0
    actual_cache = alloc_layer_cache(model, batch=2, max_seq=5)
    reference_cache = alloc_layer_cache(model, batch=2, max_seq=5)

    actual = model(x, cache=actual_cache, attn_mask=mask)
    assert isinstance(model.inner, Attention)
    inner = model.inner(x, cache=reference_cache, attn_mask=mask)
    torch.testing.assert_close(actual, inner * torch.sigmoid(model.gate_proj(x)))
    assert not torch.equal(actual, inner / torch.sigmoid(model.gate_proj(x)))
    updated = actual_cache[model.inner.depth_index]
    reference_updated = reference_cache[model.inner.depth_index]
    assert isinstance(updated, KVCache)
    assert isinstance(reference_updated, KVCache)
    assert torch.equal(updated.k, reference_updated.k)
    assert torch.equal(updated.v, reference_updated.v)
    assert updated.length == reference_updated.length == 5


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
