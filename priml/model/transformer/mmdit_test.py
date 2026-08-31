"""Tests for MMDiT block.

Regenerate bit-for-bit goldens after an intentional numeric change::

    BFB_REGENERATE=1 uv --quiet run --frozen pytest \
        priml/model/transformer/mmdit_test.py

Run regeneration through pytest so priml's conftest establishes the required
math environment before torch imports.
"""

from __future__ import annotations

from pathlib import Path

from configgle.testing import assert_pprint_golden

import pytest
import torch

from priml.model.attention.multi_stream import MultiStreamAttention
from priml.model.attention.rope import RoPE
from priml.model.transformer.mmdit import AdaLNZero, MMDiTBlock
from priml.testing.bfb import (
    assert_bfb_against_golden,
    bfb_devices,
    move_to_device,
)
from priml.testing.fixtures import (
    cleanup_cuda,  # noqa: F401 -- pytest fixture, injected by name not called
)


_TESTDATA = Path(__file__).parent.resolve() / "testdata"


def _cfg(
    channels_in: int = 64,
    num_streams: int = 2,
    num_heads: int = 4,
    **kwargs: object,
) -> MMDiTBlock.Config:
    """Helper to build MMDiTBlock.Config with attention params."""
    cfg = MMDiTBlock.Config(channels_in=channels_in, num_streams=num_streams)
    cfg.attn = MultiStreamAttention.Config(num_heads=num_heads)
    for k, v in kwargs.items():
        setattr(cfg, k, v)
    return cfg


def _canonical_adaln_config() -> AdaLNZero.Config:
    return AdaLNZero.Config(channels_in=8, cond_dim=4)


def _canonical_mmdit_config() -> MMDiTBlock.Config:
    config = MMDiTBlock.Config(channels_in=8, num_streams=2)
    config.attn = MultiStreamAttention.Config(num_heads=2, channels_head=4)
    return config


def test_adaln_zero_config_pprint() -> None:
    assert_pprint_golden(
        test_file=__file__,
        name="ada_ln_zero",
        config=_canonical_adaln_config(),
    )


def test_adaln_zero_bfb() -> None:
    assert_bfb_against_golden(
        golden_dir=_TESTDATA,
        golden_name="ada_ln_zero",
        build_module=lambda: _canonical_adaln_config().make(),
        build_input=lambda: torch.randn(2, 4),
        seed=0,
        run=lambda module, conditioning: torch.cat(module(conditioning), dim=-1),
    )


def test_mmdit_block_config_pprint() -> None:
    assert_pprint_golden(
        test_file=__file__,
        name="mmdit_block",
        config=_canonical_mmdit_config(),
    )


# -- AdaLNZero tests -------------------------------------------------


def test_adaln_zero_init():
    m = AdaLNZero.Config(channels_in=32, cond_dim=64).make()
    c = torch.randn(2, 64)
    params = m(c)
    # Gates (indices 2 and 5) should be near-zero at init.
    assert params[2].abs().max() < 1e-6
    assert params[5].abs().max() < 1e-6
    assert params[0].shape == (2, 1, 32)


# -- MMDiTBlock tests ------------------------------------------------


def test_2_streams():
    m = _cfg(num_streams=2).make()
    x0 = torch.randn(2, 8, 64)
    x1 = torch.randn(2, 12, 64)
    y0, y1 = m([x0, x1])
    assert y0.shape == (2, 8, 64)
    assert y1.shape == (2, 12, 64)


def test_3_streams():
    m = _cfg(num_streams=3).make()
    x0 = torch.randn(1, 4, 64)
    x1 = torch.randn(1, 8, 64)
    x2 = torch.randn(1, 6, 64)
    y0, y1, y2 = m([x0, x1, x2])
    assert y0.shape == (1, 4, 64)
    assert y1.shape == (1, 8, 64)
    assert y2.shape == (1, 6, 64)


def test_1_stream():
    """Single stream degenerates to a standard transformer block."""
    m = _cfg(num_streams=1).make()
    x = torch.randn(2, 16, 64)
    (y,) = m([x])
    assert y.shape == (2, 16, 64)


def test_with_adaln():
    m = _cfg(cond_dim=32).make()
    x0 = torch.randn(2, 8, 64)
    x1 = torch.randn(2, 12, 64)
    c = torch.randn(2, 32)
    y0, y1 = m([x0, x1], c=c)
    assert y0.shape == (2, 8, 64)
    assert y1.shape == (2, 12, 64)


def test_per_stream_conditioning():
    m = _cfg(cond_dim=32).make()
    x0 = torch.randn(2, 8, 64)
    x1 = torch.randn(2, 12, 64)
    c0 = torch.randn(2, 32)
    c1 = torch.randn(2, 32)
    y0, y1 = m([x0, x1], c=[c0, c1])
    assert y0.shape == (2, 8, 64)
    assert y1.shape == (2, 12, 64)


def test_identity_at_init():
    """With adaLN zero-init, block should be near-identity."""
    m = _cfg(cond_dim=32).make()
    x0 = torch.randn(1, 8, 64)
    x1 = torch.randn(1, 12, 64)
    c = torch.randn(1, 32)
    with torch.no_grad():
        y0, y1 = m([x0, x1], c=c)
    assert torch.allclose(y0, x0, atol=1e-5)
    assert torch.allclose(y1, x1, atol=1e-5)


def test_conditioning_is_required_when_adaln_is_configured():
    """A configured AdaLN with no conditioning is not the identity it claims.

    Skipping modulation adds both sublayers ungated.
    """
    m = _cfg(cond_dim=32).make()
    with pytest.raises(ValueError, match="conditioning"):
        m([torch.randn(1, 8, 64), torch.randn(1, 12, 64)])


def test_conditioning_count_must_match_the_streams():
    """One conditioning short leaves the last stream silently unmodulated."""
    m = _cfg(cond_dim=32, num_streams=3).make()
    xs = [torch.randn(1, 4, 64), torch.randn(1, 4, 64), torch.randn(1, 4, 64)]
    with pytest.raises(ValueError, match="conditioning"):
        m(xs, c=[torch.randn(1, 32), torch.randn(1, 32)])


def test_no_adaln():
    m = _cfg(cond_dim=0).make()
    x0 = torch.randn(2, 8, 64)
    x1 = torch.randn(2, 12, 64)
    y0, y1 = m([x0, x1])
    assert y0.shape == (2, 8, 64)
    assert y1.shape == (2, 12, 64)


def test_with_rope():
    """RoPE on stream 0 only (e.g. image with positions, text without)."""
    m = _cfg().make()
    x0 = torch.randn(2, 8, 64)
    x1 = torch.randn(2, 12, 64)

    rope = RoPE.Config(channels_head=16).make()
    cos_sin_0 = rope(torch.arange(8))

    y0, y1 = m([x0, x1], cos_sin=[cos_sin_0, None])
    assert y0.shape == (2, 8, 64)
    assert y1.shape == (2, 12, 64)


def test_adaln_configured_takes_zero_conditioning():
    """Zero conditioning is how an AdaLN block is run unconditioned.

    It routes through the zero-initialized gates, so the block stays the
    identity its docstring promises.
    """
    m = _cfg(cond_dim=32).make()
    x0 = torch.randn(2, 8, 64)
    x1 = torch.randn(2, 12, 64)
    with torch.no_grad():
        y0, y1 = m([x0, x1], c=torch.zeros(2, 32))
    assert torch.allclose(y0, x0, atol=1e-5)
    assert torch.allclose(y1, x1, atol=1e-5)


def test_rope_all_streams():
    """RoPE on every stream."""
    m = _cfg().make()
    x0 = torch.randn(2, 8, 64)
    x1 = torch.randn(2, 12, 64)

    rope = RoPE.Config(channels_head=16).make()
    cs0 = rope(torch.arange(8))
    cs1 = rope(torch.arange(12))

    y0, y1 = m([x0, x1], cos_sin=[cs0, cs1])
    assert y0.shape == (2, 8, 64)
    assert y1.shape == (2, 12, 64)


def test_backward():
    """Verify gradients flow through all streams and conditioning."""
    m = _cfg(cond_dim=32).make()
    x0 = torch.randn(2, 8, 64, requires_grad=True)
    x1 = torch.randn(2, 12, 64, requires_grad=True)
    c = torch.randn(2, 32, requires_grad=True)
    y0, y1 = m([x0, x1], c=c)
    loss = y0.sum() + y1.sum()
    loss.backward()
    assert x0.grad is not None
    assert x1.grad is not None
    assert c.grad is not None


def test_reset_parameters():
    m = _cfg(cond_dim=32).make()
    m.reset_parameters()


def test_attention_inner_width_decoupled_from_residual():
    """Attention inner width (num_heads*channels_head) may differ from residual.

    Post-MODEL-008: ``channels_in`` (residual stream) and
    ``num_heads * channels_head`` (attention inner width) are independent;
    ``proj_outs`` map inner -> residual. The block builds and forwards.
    """
    cfg = _cfg()
    cfg.attn = MultiStreamAttention.Config(
        channels_in=64,
        num_heads=4,
        channels_head=8,
    )
    m = cfg.make()
    assert m.attn.proj_outs[0].weight.shape == (64, 4 * 8)
    y0, y1 = m([torch.randn(2, 8, 64), torch.randn(2, 12, 64)])
    assert y0.shape == (2, 8, 64)
    assert y1.shape == (2, 12, 64)


def test_extra_batch_dims():
    """Verify arbitrary leading batch dimensions work."""
    m = _cfg().make()
    x0 = torch.randn(2, 3, 8, 64)
    x1 = torch.randn(2, 3, 12, 64)
    y0, y1 = m([x0, x1])
    assert y0.shape == (2, 3, 8, 64)
    assert y1.shape == (2, 3, 12, 64)


@pytest.mark.parametrize("device", bfb_devices(), ids=str)
def test_mmdit_block_bfb(device: str) -> None:
    assert_bfb_against_golden(
        golden_dir=_TESTDATA,
        golden_name="mmdit_block",
        build_module=lambda: _canonical_mmdit_config().make().to(device),
        build_input=lambda: move_to_device(
            [torch.randn(2, 3, 8), torch.randn(2, 2, 8)], device
        ),
        seed=0,
        run=lambda module, streams: torch.cat(module(streams), dim=-2),
    )


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
