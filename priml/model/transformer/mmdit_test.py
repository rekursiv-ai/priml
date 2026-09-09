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
from priml.model.attention.self_attention import SelfAttention
from priml.model.norm import RMSNorm
from priml.model.swiglu import SwiGLU
from priml.model.transformer import mmdit
from priml.model.transformer.block import TransformerBlock
from priml.model.transformer.mmdit import AdaLNZero, MMDiTBlock
from priml.testing.bfb import (
    assert_bfb_against_golden,
    bfb_devices,
    host_agnostic_numerics,
    move_to_device,
    randomize_parameters,
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


def _native_stream_config() -> TransformerBlock.Config:
    cfg = TransformerBlock.Config()
    cfg.channels_in = 8
    cfg.attn = SelfAttention.Config()
    cfg.attn.num_heads = 2
    cfg.attn.num_heads_kv = 1
    cfg.attn.channels_head = 4
    cfg.attn.norm_qk = RMSNorm.Config()
    cfg.attn.norm_qk.elementwise_affine = True
    cfg.attn.share_qk_norm = False
    cfg.norm1 = RMSNorm.Config()
    cfg.norm1.elementwise_affine = True
    cfg.norm2 = RMSNorm.Config()
    cfg.norm2.elementwise_affine = True
    cfg.ffn = SwiGLU.Config()
    cfg.ffn.channels_hidden = 12
    return cfg


def test_native_stream_loading_matches_transformer_and_freezes_independently() -> None:
    native_cfg = _native_stream_config()
    source = native_cfg.make()
    randomize_parameters(source, seed=7, std=0.2)
    stream = mmdit.MMDiTStream.Config()
    assert isinstance(native_cfg.attn, SelfAttention.Config)
    stream.attn = native_cfg.attn.copy_tree()
    stream.norm1 = native_cfg.norm1.copy_tree()
    stream.norm2 = native_cfg.norm2.copy_tree()
    stream.ffn = native_cfg.ffn.copy_tree()
    cfg = _cfg(channels_in=8, num_heads=2)
    assert isinstance(cfg.attn, MultiStreamAttention.Config)
    cfg.attn.num_heads_kv = 1
    cfg.attn.channels_head = 4
    cfg.streams = [stream]
    model = cfg.make()
    model.load_stream(0, source=source)
    x = torch.randn(1, 3, 8)
    with host_agnostic_numerics():
        assert torch.equal(model([x])[0], source(x))
    cfg.streams = [stream, stream.copy_tree()]
    mixed = cfg.make()
    randomize_parameters(mixed, seed=11, std=0.2)
    mixed.load_stream(0, source=source)
    frozen = [
        mixed.norms1[0],
        mixed.norms2[0],
        mixed.ffns[0],
        mixed.attn.streams[0],
    ]
    for module in frozen:
        module.requires_grad_(False)
    before = {k: v.clone() for k, v in mixed.state_dict().items()}
    other = torch.randn(1, 2, 8)
    mask = torch.cat((torch.zeros(3, 3), torch.full((3, 2), float("-inf"))), -1)
    with host_agnostic_numerics():
        assert torch.equal(mixed([x, other], attn_mask=[mask, None])[0], source(x))
    y = mixed([x, other])[1]
    y.square().sum().backward()
    assert all(p.grad is None for module in frozen for p in module.parameters())
    gradient = mixed.attn.streams[1].proj_qkv.weight.grad
    assert gradient is not None
    assert gradient.abs().sum() > 0
    assert mixed.ffns[1].up_proj.weight.grad.abs().sum() > 0
    torch.optim.SGD(mixed.parameters(), lr=0.1).step()
    assert torch.equal(before["ffns.0.up_proj.weight"], mixed.ffns[0].up_proj.weight)
    assert not torch.equal(
        before["ffns.1.up_proj.weight"], mixed.ffns[1].up_proj.weight
    )


def test_mixed_conditioning_has_no_unconditioned_parameters() -> None:
    cfg = _cfg(channels_in=8, num_heads=2)
    cfg.streams = [mmdit.MMDiTStream.Config(), mmdit.MMDiTStream.Config()]
    cfg.streams[0].adaln = AdaLNZero.Config()
    cfg.streams[0].adaln.cond_dim = 4
    cfg.streams[0].ffn = SwiGLU.Config()
    cfg.streams[0].ffn.channels_hidden = 12
    cfg.streams[1].ffn = SwiGLU.Config()
    cfg.streams[1].ffn.channels_hidden = 16
    cfg.streams[1].norm1 = RMSNorm.Config()
    cfg.streams[1].norm1.elementwise_affine = True
    model = cfg.make()
    xs = [torch.randn(1, 2, 8), torch.randn(1, 3, 8)]
    actual = model(xs, c=[torch.randn(1, 4), None])
    assert torch.equal(actual[0], xs[0])
    assert not torch.equal(actual[1], xs[1])
    assert not any(name.startswith("adalns.1.") for name, _ in model.named_parameters())
    assert model.ffns[0].up_proj.weight.shape[0] == 24
    assert model.ffns[1].up_proj.weight.shape[0] == 32
    with pytest.raises(ValueError, match="conditioning"):
        model(xs, c=[None, None])
    model.reset_parameters()


def test_default_block_forwards_per_query_masks() -> None:
    model = _cfg(channels_in=8, num_heads=2).make()
    xs = [torch.randn(1, 2, 8), torch.randn(1, 3, 8)]
    masks = [
        torch.cat((torch.zeros(2, 2), torch.full((2, 3), float("-inf"))), -1),
        None,
    ]
    expected = model(xs, attn_mask=masks)[0]
    actual = model([xs[0], xs[1] + 10], attn_mask=masks)[0]
    assert torch.equal(actual, expected)


def test_explicit_streams_do_not_configure_unused_ffn_template() -> None:
    cfg = _cfg(channels_in=8, num_heads=2)
    cfg.depth_index = ((3, 4),)
    cfg.streams = [mmdit.MMDiTStream.Config()]

    finalized = cfg.copy_tree().finalize()

    assert isinstance(finalized.ffn, SwiGLU.Config)
    assert finalized.ffn.channels_in != finalized.channels_in
    assert finalized.ffn.channels_out != finalized.channels_out
    assert finalized.ffn.depth_index != finalized.depth_index
    assert isinstance(finalized.streams[0].ffn, SwiGLU.Config)
    assert finalized.streams[0].ffn.channels_in == 8
    assert finalized.streams[0].ffn.channels_out == 8
    assert finalized.streams[0].ffn.depth_index == ((3, 4),)


def test_stream_attention_geometry_finalizes_before_stream_norm() -> None:
    cfg = _cfg(channels_in=8, num_heads=2)
    cfg.streams = [mmdit.MMDiTStream.Config()]
    cfg.streams[0].attn.norm_qk = RMSNorm.Config()
    cfg.streams[0].attn.norm_qk.elementwise_affine = True
    finalized = cfg.copy_tree().finalize()
    assert isinstance(finalized.streams[0].attn.norm_qk, RMSNorm.Config)
    assert finalized.streams[0].attn.norm_qk.channels_in == 4
    assert finalized.streams[0].attn.channels_head == 4
    finalized.make()([torch.randn(1, 2, 8)])


@pytest.mark.parametrize("index", [-1, 1])
def test_native_loading_rejects_invalid_stream_indices(index: int) -> None:
    source = _native_stream_config().make()
    cfg = _cfg(channels_in=8, num_heads=2)
    cfg.streams = [mmdit.MMDiTStream.Config()]
    cfg.streams[0].adaln = AdaLNZero.Config()
    cfg.streams[0].adaln.cond_dim = 4
    model = cfg.make()
    with pytest.raises(ValueError, match="index"):
        model.load_stream(index, source=source)


def test_native_loading_rejects_shapes_atomically_and_postnorm() -> None:
    source_cfg = _native_stream_config()
    cfg = _cfg(channels_in=8, num_heads=2)
    assert isinstance(cfg.attn, MultiStreamAttention.Config)
    cfg.attn.num_heads_kv = 1
    cfg.streams = [mmdit.MMDiTStream.Config().update(source_cfg, skip_missing=True)]
    model = cfg.make()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    assert isinstance(source_cfg.ffn, SwiGLU.Config)
    source_cfg.ffn.channels_hidden = 16
    with pytest.raises(ValueError, match="shape"):
        model.load_stream(0, source=source_cfg.make())
    assert all(
        torch.equal(before[name], value) for name, value in model.state_dict().items()
    )
    source_cfg.prenorm = False
    with pytest.raises(ValueError, match="prenorm"):
        model.load_stream(0, source=source_cfg.make())
    with pytest.raises(ValueError, match="explicit"):
        _cfg(channels_in=8, num_heads=2).make().load_stream(
            0, source=_native_stream_config().make()
        )


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
