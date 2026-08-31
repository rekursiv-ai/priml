"""Tests for attention module."""

from __future__ import annotations

from pathlib import Path

from configgle.testing import assert_pprint_golden

import pytest
import torch

from priml.model.attention.output_gate import OutputGate
from priml.model.attention.rope import RoPE
from priml.model.attention.self_attention import SelfAttention
from priml.model.norm import RMSNorm
from priml.testing.bfb import assert_bfb_against_golden, bfb_devices
from priml.testing.fixtures import (
    cleanup_cuda,  # noqa: F401 -- pytest fixture, injected by name not called
)


_TESTDATA = Path(__file__).parent.resolve() / "testdata"


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
        golden_dir=_TESTDATA,
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


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
