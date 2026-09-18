"""Tests for MMDiT block.

Regenerate bit-for-bit goldens after an intentional numeric change::

    BFB_REGENERATE=1 uv --quiet run --frozen pytest \
        priml/model/transformer/mmdit_test.py

Run regeneration through pytest so priml's conftest establishes the required
math environment before torch imports.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from configgle.testing import assert_pprint_golden
from torch import Tensor, nn

import pytest
import torch

from priml.cost import Cost, cost
from priml.model.attention.kernel import SdpaNaive
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
from priml.testing.cost import assert_cost_matches_torch


_CWD: Final = Path(__file__).resolve().parent


def _cfg(
    channels_in: int = 64,
    num_streams: int = 2,
    num_heads: int = 4,
    **kwargs: object,
) -> MMDiTBlock.Config:
    """Build an MMDiTBlock.Config with attention params."""
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
        golden_dir=_CWD / "testdata",
        golden_name="ada_ln_zero",
        build_module=lambda: _canonical_adaln_config().make(),
        build_input=lambda: torch.randn(2, 4),
        seed=0,
        run=_run_adaln,
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
        golden_dir=_CWD / "testdata",
        golden_name="mmdit_block",
        build_module=lambda: _canonical_mmdit_config().make().to(device),
        build_input=lambda: move_to_device(
            [torch.randn(2, 3, 8), torch.randn(2, 2, 8)],
            device,
        ),
        seed=0,
        run=_run_mmdit,
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
    state = mixed.state_dict()
    before: dict[str, Tensor] = {k: v.clone() for k, v in state.items()}
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
    ffn0 = mixed.ffns[0]
    ffn1 = mixed.ffns[1]
    assert isinstance(ffn0, SwiGLU)
    assert isinstance(ffn1, SwiGLU)
    assert ffn1.up_proj.weight.grad is not None
    assert ffn1.up_proj.weight.grad.abs().sum() > 0
    torch.optim.SGD(mixed.parameters(), lr=0.1).step()
    assert torch.equal(before["ffns.0.up_proj.weight"], ffn0.up_proj.weight)
    assert not torch.equal(
        before["ffns.1.up_proj.weight"],
        ffn1.up_proj.weight,
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
    ffn0 = model.ffns[0]
    ffn1 = model.ffns[1]
    assert isinstance(ffn0, SwiGLU)
    assert isinstance(ffn1, SwiGLU)
    assert ffn0.up_proj.weight.shape[0] == 24
    assert ffn1.up_proj.weight.shape[0] == 32
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
    state = model.state_dict()
    before: dict[str, Tensor] = {name: value.clone() for name, value in state.items()}
    assert isinstance(source_cfg.ffn, SwiGLU.Config)
    source_cfg.ffn.channels_hidden = 16
    with pytest.raises(ValueError, match="shape"):
        model.load_stream(0, source=source_cfg.make())
    current = model.state_dict()
    assert all(torch.equal(before[name], value) for name, value in current.items())
    source_cfg.prenorm = False
    with pytest.raises(ValueError, match="prenorm"):
        model.load_stream(0, source=source_cfg.make())
    with pytest.raises(ValueError, match="explicit"):
        _cfg(channels_in=8, num_heads=2).make().load_stream(
            0,
            source=_native_stream_config().make(),
        )


def test_adaln_zero_cost_is_one_biased_matmul() -> None:
    config = AdaLNZero.Config(channels_in=8, cond_dim=4)
    finalized = config.copy_tree().finalize()
    proj = cost(finalized.proj, seq_len=8, batch_size=1, dtype=None)
    assert proj.params == 4 * 6 * 8 + 6 * 8
    f32 = torch.float32
    assert finalized.cost(seq_len=8, batch_size=1, dtype=None) == proj + Cost(
        cells={
            ("flops", "primal", "elementwise", f32): 5 * 4,
            ("flops", "adjoint", "elementwise", f32): 5 * 4,
            ("bytes", "primal", "elementwise", f32): 4 * 2 * 4,
            ("bytes", "adjoint", "elementwise", f32): 4 * 3 * 4,
        },
    )
    assert proj.params == sum(p.numel() for p in config.make().parameters())


def test_stream_cost_sums_its_branches_and_leaves_attention_to_the_joint() -> None:
    """The joint attention builds ``attn``, so the stream does not price it."""
    config = mmdit.MMDiTStream.Config(channels_in=8)
    config.ffn = SwiGLU.Config(channels_hidden=12)
    config.adaln = AdaLNZero.Config(cond_dim=4)
    config.norm1 = RMSNorm.Config(elementwise_affine=True)
    finalized = config.copy_tree().finalize()
    children = (finalized.norm1, finalized.norm2, finalized.ffn, finalized.adaln)
    f32 = torch.float32
    expected = sum(
        (cost(child, seq_len=8, batch_size=1, dtype=None) for child in children),
        Cost(),
    ) + Cost(
        cells={
            ("flops", "primal", "elementwise", f32): 10 * 8,
            ("flops", "adjoint", "elementwise", f32): 10 * 8,
            ("bytes", "primal", "elementwise", f32): 4 * 28 * 8,
            ("bytes", "adjoint", "elementwise", f32): 4 * 40 * 8,
        },
    )
    assert finalized.cost(seq_len=8, batch_size=1, dtype=None) == expected
    assert expected.params == sum(p.numel() for p in config.make().parameters())
    assert expected.params == 8 + (8 * 24 + 12 * 8) + (4 * 48 + 48)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("conditioned", [False, True])
def test_stream_traffic_counts_residual_and_modulation_operands(
    dtype: torch.dtype,
    conditioned: bool,
) -> None:
    config = mmdit.MMDiTStream.Config()
    config.channels_in = 8
    if conditioned:
        config.adaln = AdaLNZero.Config()
        config.adaln.cond_dim = 4
    config = config.finalize()
    itemsize = dtype.itemsize
    children = (config.norm1, config.norm2, config.ffn)
    child_cost = sum(
        (cost(child, seq_len=1, batch_size=1, dtype=dtype) for child in children),
        Cost(),
    )
    if config.adaln is not None:
        child_cost += cost(config.adaln, seq_len=1, batch_size=1, dtype=dtype)
    actual = config.cost(seq_len=1, batch_size=1, dtype=dtype)
    # Per branch: unary scale offset, scale, shift, gate, residual.
    primal_elements = 2 * (2 + 3 + 3 + 3 + 3) if conditioned else 2 * 3
    adjoint_elements = 2 * (2 + 6 + 3 + 6 + 3) if conditioned else 2 * 3
    assert actual["bytes", "primal", "elementwise"].sum() == (
        child_cost["bytes", "primal", "elementwise"].sum()
        + itemsize * 8 * primal_elements
    )
    assert actual["bytes", "adjoint", "elementwise"].sum() == (
        child_cost["bytes", "adjoint", "elementwise"].sum()
        + itemsize * 8 * adjoint_elements
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("conditioned", [False, True])
def test_explicit_and_implicit_stream_costs_agree(
    dtype: torch.dtype,
    conditioned: bool,
) -> None:
    implicit = _cfg(channels_in=8, num_streams=1, num_heads=2)
    if conditioned:
        implicit.cond_dim = 4
    explicit = implicit.copy_tree()
    stream = mmdit.MMDiTStream.Config()
    stream.ffn = implicit.ffn.copy_tree()
    if conditioned:
        stream.adaln = AdaLNZero.Config()
        stream.adaln.cond_dim = 4
    explicit.streams = [stream]
    assert implicit.finalize().cost(
        seq_len=4,
        batch_size=1,
        dtype=dtype,
    ) == explicit.finalize().cost(seq_len=4, batch_size=1, dtype=dtype)


def test_block_cost_with_implicit_streams_is_the_hand_formula() -> None:
    """Joint scores over ``seq_len`` keys; every stream pays its own branches."""
    config = _cfg(channels_in=8, num_streams=2, num_heads=2, cond_dim=4)
    config.ffn = SwiGLU.Config(channels_hidden=12)
    seq_len = 16
    finalized = config.copy_tree().finalize()
    model_cost = finalized.cost(seq_len=seq_len, batch_size=1, dtype=None)
    attn = cost(finalized.attn, seq_len=seq_len, batch_size=1, dtype=None)
    ffn = cost(finalized.ffn, seq_len=seq_len, batch_size=1, dtype=None)
    adaln = cost(
        AdaLNZero.Config(channels_in=8, cond_dim=4).finalize(),
        seq_len=seq_len,
        batch_size=1,
        dtype=None,
    )
    assert attn.params == 2 * ((2 + 2 * 2) * 8 * 4 + 2 * 4 * 8)
    assert ffn.params == 8 * 24 + 12 * 8
    assert adaln.params == 4 * 48 + 48
    assert model_cost["flops", "primal", "matmul"].sum() == (
        attn["flops", "primal", "matmul"].sum()
        + 2
        * (
            ffn["flops", "primal", "matmul"].sum()
            + adaln["flops", "primal", "matmul"].sum()
        )
    )
    assert model_cost["flops", "adjoint", "matmul"].sum() == (
        attn["flops", "adjoint", "matmul"].sum()
        + 2
        * (
            ffn["flops", "adjoint", "matmul"].sum()
            + adaln["flops", "adjoint", "matmul"].sum()
        )
    )
    assert model_cost.params == attn.params + 2 * (ffn.params + adaln.params)
    assert model_cost.params == sum(p.numel() for p in config.make().parameters())
    # Two streams per position, each caching its own K and V.
    assert model_cost.bytes_state == 4 * 2 * 2 * 2 * 4


def test_block_cost_with_explicit_streams_prices_each_once() -> None:
    config = _cfg(channels_in=8, num_heads=2)
    config.streams = [mmdit.MMDiTStream.Config(), mmdit.MMDiTStream.Config()]
    config.streams[0].adaln = AdaLNZero.Config(cond_dim=4)
    config.streams[0].ffn = SwiGLU.Config(channels_hidden=12)
    config.streams[1].ffn = SwiGLU.Config(channels_hidden=16)
    config.streams[1].attn.norm_qk = RMSNorm.Config(elementwise_affine=True)
    finalized = config.copy_tree().finalize()
    model_cost = finalized.cost(seq_len=8, batch_size=1, dtype=None)
    expected = cost(finalized.attn, seq_len=8, batch_size=1, dtype=None) + sum(
        (
            cost(stream, seq_len=8, batch_size=1, dtype=None)
            for stream in finalized.streams
        ),
        Cost(),
    )
    assert model_cost == expected
    assert model_cost.params == sum(p.numel() for p in config.make().parameters())


def test_block_cost_matches_torch_without_conditioning() -> None:
    """Two streams of four tokens: joint attention plus each stream's FFN.

    Unconditioned, because adaLN's projection runs once per SEQUENCE while
    ``cost`` prices it per token (its documented upper bound); measured, the
    per-token figure overstates a four-token sequence by exactly ``3/4`` of
    the projection.
    """
    config = _cfg(channels_in=8, num_streams=2, num_heads=2)
    config.attn = MultiStreamAttention.Config(
        num_heads=2,
        attn_kernel=SdpaNaive.Config(),
    )
    config.ffn = SwiGLU.Config(channels_hidden=12)
    assert_cost_matches_torch(
        config,
        build_input=lambda: tuple(
            torch.randn(1, 4, 8, requires_grad=True) for _ in range(2)
        ),
        seq_len=4,
        batch_size=1,
        num_tokens=4,
        dtype=None,
        run=lambda module, xs: _run_mmdit(module, list(xs)).sum(),
    )


def _run_adaln(module: nn.Module, conditioning: Tensor) -> Tensor:
    """Run AdaLN-Zero with its typed tensor output."""
    assert isinstance(module, AdaLNZero)
    return torch.cat(module(conditioning), dim=-1)


def _run_mmdit(module: nn.Module, streams: list[Tensor]) -> Tensor:
    """Run MMDiT and concatenate its stream outputs."""
    assert isinstance(module, MMDiTBlock)
    return torch.cat(module(streams), dim=-2)


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
