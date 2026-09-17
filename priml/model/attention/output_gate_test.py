"""Tests for attention module."""

from __future__ import annotations

from pathlib import Path
from typing import Final

from configgle.testing import assert_pprint_golden

import pytest
import torch

from priml.model.attention.kernel import SdpaNaive
from priml.model.attention.output_gate import OutputGate
from priml.model.attention.rope import RoPE
from priml.model.attention.self_attention import SelfAttention
from priml.model.cost import cost
from priml.model.norm import RMSNorm
from priml.testing.bfb import assert_bfb_against_golden, bfb_devices
from priml.testing.cost import assert_cost_matches_torch


_CWD: Final = Path(__file__).resolve().parent


def test_output_gate_config_pprint() -> None:
    config = OutputGate.Config(
        channels_in=16,
        inner=SelfAttention.Config(num_heads=2, channels_head=8),
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
        channels_in=64,
        inner=SelfAttention.Config(
            channels_in=64,
            num_heads=4,
            channels_head=16,
            causal=True,
        ),
    ).make()
    x = torch.randn(2, 8, 64)
    out = m(x)
    assert out.shape == (2, 8, 64)


def test_output_gate_cached():
    m = OutputGate.Config(
        channels_in=64,
        inner=SelfAttention.Config(
            channels_in=64,
            num_heads=4,
            channels_head=16,
            causal=True,
        ),
    ).make()
    cache = m.alloc_kv_cache(batch=2, max_seq=8)

    out, cache = m.forward_cached(torch.randn(2, 8, 64), cache=cache)

    assert out.shape == (2, 8, 64)
    assert cache.length == 8


def test_output_gate_passthrough_kwargs():
    rope = RoPE.Config(channels_head=16).make()
    m = OutputGate.Config(
        channels_in=64,
        inner=SelfAttention.Config(
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
        inner=SelfAttention.Config(channels_in=64, num_heads=4, channels_head=16),
    ).make()
    m.reset_parameters()


def test_output_gate_finalize_propagates():
    cfg = OutputGate.Config(
        channels_in=128,
        inner=SelfAttention.Config(num_heads=4, channels_head=32),
    ).finalize()
    assert isinstance(cfg.inner, SelfAttention.Config)
    assert cfg.inner.channels_in == 128


def test_output_gate_geometry_falls_back_for_an_unheaded_inner() -> None:
    cfg = OutputGate.Config(channels_in=8)
    cfg.inner = RMSNorm.Config(channels_in=8)

    assert cfg.num_heads == 1
    assert cfg.channels_head == 8


@pytest.mark.parametrize("device", bfb_devices(), ids=str)
def test_output_gate_bfb(device: str) -> None:
    assert_bfb_against_golden(
        golden_dir=_CWD / "testdata",
        golden_name="output_gate",
        build_module=lambda: (
            OutputGate.Config(
                channels_in=16,
                inner=SelfAttention.Config(num_heads=2, channels_head=8),
            )
            .make()
            .to(device)
        ),
        build_input=lambda: torch.randn(2, 4, 16),
        seed=0,
    )


def test_output_gate_cost_is_the_inner_plus_a_square_gate_matmul() -> None:
    config = OutputGate.Config(
        channels_in=16,
        inner=SelfAttention.Config(num_heads=2, channels_head=8),
    )
    config.inner = SelfAttention.Config(
        num_heads=2,
        channels_head=8,
        attn_kernel=SdpaNaive.Config(),
    )
    model_cost = assert_cost_matches_torch(
        config,
        build_input=lambda: torch.randn(1, 8, 16, requires_grad=True),
        seq_len=8,
        batch_size=1,
        dtype=None,
    )
    inner = cost(
        config.copy_tree().finalize().inner,
        seq_len=8,
        batch_size=1,
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
        + 2 * gate
    )
    assert (
        model_cost["flops", "adjoint", "matmul"].sum()
        == inner[
            "flops",
            "adjoint",
            "matmul",
        ].sum()
        + 4 * gate
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
    assert actual["bytes", "primal", "matmul"].sum() == 2 * (8 + 8 + 64 / 4)
    assert (
        actual["bytes", "primal", "elementwise"].sum()
        - inner["bytes", "primal", "elementwise"].sum()
        == 2 * 5 * 8
    )
    assert (
        actual["bytes", "adjoint", "elementwise"].sum()
        - inner["bytes", "adjoint", "elementwise"].sum()
        == 2 * 12 * 8
    )
    wide = config.cost(seq_len=4, batch_size=1, dtype=None)
    assert (
        wide["bytes", :, :, torch.float32].sum()
        == actual["bytes", :, :, torch.bfloat16].sum() * 2
    )


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
