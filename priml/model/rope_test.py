"""Tests for rope module."""

from __future__ import annotations

import math

import pytest
import torch

from priml.model.rope import RoPE, RoPEMixed, YarnScaling
from priml.testing.fixtures import (
    cleanup_cuda,  # noqa: F401 -- pytest fixture, injected by name not called
)


def test_rope_1d():
    m = RoPE.Config(32).make()
    pos = torch.arange(16)
    cos, sin = m(pos)
    assert cos.shape == (16, 1, 16)
    assert sin.shape == (16, 1, 16)


def test_rope_nd():
    m = RoPE.Config([16, 16]).make()
    pos = torch.stack([torch.arange(8), torch.arange(8)], dim=-1)
    cos, _sin = m(pos)
    assert cos.shape == (8, 1, 16)


def test_rope_rotate():
    q = torch.randn(2, 8, 4, 32)
    k = torch.randn(2, 8, 4, 32)
    cos = torch.randn(8, 1, 16)
    sin = torch.randn(8, 1, 16)
    q_rot, k_rot = RoPE.rotate(q, k, cos, sin)
    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape


def test_rope_rotate_interleave():
    q = torch.randn(2, 8, 4, 32)
    k = torch.randn(2, 8, 4, 32)
    cos = torch.randn(8, 1, 16)
    sin = torch.randn(8, 1, 16)
    q_rot, _k_rot = RoPE.rotate(q, k, cos, sin, interleave=True)
    assert q_rot.shape == q.shape


def _rotate_reference(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, interleave: bool
) -> torch.Tensor:
    dtype = x.dtype
    shape = x.shape
    if interleave:
        pairs = x.float().reshape(*shape[:-1], -1, 2)
        x0 = pairs[..., 0]
        x1 = pairs[..., 1]
        return (
            torch.stack(
                [x0 * cos - x1 * sin, x1 * cos + x0 * sin],
                dim=-1,
            )
            .reshape(shape)
            .to(dtype)
        )
    pairs = x.float().reshape(*shape[:-1], 2, -1)
    x0 = pairs[..., 0, :]
    x1 = pairs[..., 1, :]
    return (
        torch.stack(
            [x0 * cos - x1 * sin, x1 * cos + x0 * sin],
            dim=-2,
        )
        .reshape(shape)
        .to(dtype)
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("interleave", [False, True])
@pytest.mark.parametrize("head_dim", [8, 64, 6])
def test_rotate_matches_stack_reference_forward_and_backward(
    dtype: torch.dtype,
    interleave: bool,
    head_dim: int,
) -> None:
    """The slice-assign ``_rotate`` is bit-for-bit with the old reshape+stack one.

    The implementation was rewritten (reshape+stack -> strided slice-assign) to
    avoid the ``aten.reshape`` that torchao fp8 axiswise scaling cannot trace.
    The rotation math is unchanged, so output AND all input grads must be
    ``torch.equal`` to the original algorithm across dtypes, both pairing
    conventions, and head dims (incl. the real model's 64 and an odd-half 6).
    This is the regression guard: any future RoPE edit that perturbs the
    numerics fails here rather than silently shifting every downstream model.
    """
    torch.manual_seed(0)
    half = head_dim // 2
    x_ref = torch.randn(2, 5, 3, head_dim, dtype=dtype, requires_grad=True)
    x = x_ref.detach().clone().requires_grad_()
    cos_ref = torch.randn(5, 1, half, dtype=dtype, requires_grad=True)
    cos = cos_ref.detach().clone().requires_grad_()
    sin_ref = torch.randn(5, 1, half, dtype=dtype, requires_grad=True)
    sin = sin_ref.detach().clone().requires_grad_()

    out_ref = _rotate_reference(x_ref, cos_ref, sin_ref, interleave)
    out = RoPE._rotate(x, cos, sin, interleave)
    grad = torch.randn_like(out_ref)
    out_ref.backward(grad)
    out.backward(grad)

    assert torch.equal(out, out_ref)
    assert x.grad is not None
    assert x_ref.grad is not None
    assert cos.grad is not None
    assert cos_ref.grad is not None
    assert sin.grad is not None
    assert sin_ref.grad is not None
    assert torch.equal(x.grad, x_ref.grad)
    assert torch.equal(cos.grad, cos_ref.grad)
    assert torch.equal(sin.grad, sin_ref.grad)


@pytest.mark.parametrize("interleave", [False, True])
def test_rotate_avoids_stack_materialization(
    monkeypatch: pytest.MonkeyPatch,
    interleave: bool,
) -> None:
    def fail_stack(*args: object, **kwargs: object) -> torch.Tensor:
        assert args or kwargs
        raise AssertionError("RoPE._rotate should not call torch.stack")

    monkeypatch.setattr(torch, "stack", fail_stack)
    x = torch.randn(2, 5, 3, 8, dtype=torch.bfloat16, requires_grad=True)
    cos = torch.randn(5, 1, 4, dtype=torch.bfloat16, requires_grad=True)
    sin = torch.randn(5, 1, 4, dtype=torch.bfloat16, requires_grad=True)

    RoPE._rotate(x, cos, sin, interleave).float().sum().backward()


@pytest.mark.parametrize("interleave", [False, True])
def test_rotate_does_not_use_empty_like(
    monkeypatch: pytest.MonkeyPatch,
    interleave: bool,
) -> None:
    """``_rotate`` must not allocate output via ``torch.empty_like``.

    Regression guard for a CUDA eval hang: the previous in-place form
    (``out = torch.empty_like(x); out[..., 0::2] = ...``) wrote rotated values
    into UNINITIALIZED memory, which wedged a compiled ``torch.inference_mode``
    eval on GPU (an eval-only resume froze mid-pass; a fresh-process allocator's
    garbage made it non-deterministic, so a warm training-process eval slipped
    through). The output must be built from initialized memory only -- ``cat`` for
    the production half-split path, ``zeros_like`` for the interleave path.
    """

    def fail_empty_like(*args: object, **kwargs: object) -> torch.Tensor:
        assert args or kwargs
        raise AssertionError("RoPE._rotate should not call torch.empty_like")

    monkeypatch.setattr(torch, "empty_like", fail_empty_like)
    x = torch.randn(2, 5, 3, 8, dtype=torch.float32)
    cos = torch.randn(5, 1, 4, dtype=torch.float32)
    sin = torch.randn(5, 1, 4, dtype=torch.float32)

    out = RoPE._rotate(x, cos, sin, interleave)
    assert torch.isfinite(out).all()


def test_rope_rotate_padding():
    """cos/sin shorter than seq_len should be padded with identity."""
    q = torch.randn(2, 16, 1, 32)
    k = torch.randn(2, 16, 1, 32)
    cos = torch.randn(8, 1, 16)  # shorter than seq_len=16
    sin = torch.randn(8, 1, 16)
    q_rot, k_rot = RoPE.rotate(q, k, cos, sin)
    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape


def test_rope_sum_mode():
    m = RoPE.Config([16, 16], reduction_mode="sum").make()
    pos = torch.stack([torch.arange(8), torch.arange(8)], dim=-1)
    cos, _sin = m(pos)
    assert cos.shape == (8, 1, 8)  # sum reduces to single set


def test_rope_auto_split_dim():
    """Scalar dim + multi base auto-splits in cat mode."""
    m = RoPE.Config(128, base=[10000.0, 10000.0, 10000.0]).make()
    assert m.channels_head == (44, 42, 42)


def test_rope_smallest_recommended_base():
    b = RoPE.smallest_recommended_base(64, 1024)
    assert isinstance(b, float)
    assert b > 0


def test_rope_smallest_recommended_base_nd():
    bases = RoPE.smallest_recommended_base([64, 64], [1024, 512])
    assert isinstance(bases, tuple)
    assert len(bases) == 2


def test_rope_validate_odd_dim():
    with pytest.raises(ValueError, match="even"):
        RoPE.Config(33).make()


def test_rope_validate_small_dim():
    with pytest.raises(ValueError, match="at least 2"):
        RoPE.Config(1).make()


def test_rope_split_dim():
    assert RoPE._split_dim(128, 3) == [44, 42, 42]
    assert RoPE._split_dim(64, 2) == [32, 32]


def test_rope_zero_dim_axis():
    """dim=0 skips an axis."""
    m = RoPE.Config([16, 0]).make()
    pos = torch.stack([torch.arange(8), torch.arange(8)], dim=-1)
    cos, _sin = m(pos)
    assert cos.shape == (8, 1, 8)


def test_rope_mixed_basic():
    m = RoPEMixed.Config(32, heads=4).make()
    pos = torch.arange(8)
    cos, _sin = m(pos)
    assert cos.shape[-2] == 4  # per-head


def test_rope_mixed_learnable():
    m = RoPEMixed.Config(32, heads=4, learnable=True).make()
    # Learnable freqs should be parameters
    param_count = sum(1 for p in m.parameters() if p.requires_grad)
    assert param_count > 0


def test_rope_mixed_single_head():
    m = RoPEMixed.Config(32, heads=1).make()
    pos = torch.arange(8)
    cos, _sin = m(pos)
    assert cos.shape[-2] == 1


def test_yarn_kimi_k2_mscale_unity():
    """Kimi-K2 YaRN has mscale = mscale_all_dim = 1.0 → ratio collapses to 1.0."""
    yarn = YarnScaling(
        factor=32.0,
        original_max_position_embeddings=4096,
        beta_fast=1.0,
        beta_slow=1.0,
        mscale=1.0,
        mscale_all_dim=1.0,
    )
    m = RoPE.Config(channels_head=64, base=50_000, yarn=yarn).make()
    assert math.isclose(m._mscale, 1.0)


def test_yarn_split_interp_extrap():
    """Low-freq (high-index) channels get inv_freq/factor; high-freq stay original."""
    yarn = YarnScaling(
        factor=8.0,
        original_max_position_embeddings=512,
        beta_fast=32.0,
        beta_slow=1.0,
        mscale=1.0,
        mscale_all_dim=0.0,
    )
    dim = 64
    m = RoPE.Config(channels_head=dim, base=10_000, yarn=yarn).make()
    orig = RoPE._make_inv_freqs(10_000, dim)
    got = m._inv_freqs[0].squeeze(0)
    # First channel (highest freq) stays original; last channel (lowest
    # freq) is interpolated by factor.
    assert torch.isclose(got[0], orig[0], atol=1e-6)
    assert torch.isclose(got[-1], orig[-1] / yarn.factor, atol=1e-6)


def test_yarn_mscale_log_formula():
    """Mscale = 0.1 * log(factor) * mscale + 1 when mscale_all_dim=0."""
    yarn = YarnScaling(
        factor=40.0,
        original_max_position_embeddings=4096,
        mscale=1.0,
        mscale_all_dim=0.0,
    )
    m = RoPE.Config(channels_head=32, base=10_000, yarn=yarn).make()
    expected = 0.1 * math.log(40.0) + 1.0
    assert math.isclose(m._mscale, expected)


def test_yarn_mscale_applied_to_cos_sin():
    """cos/sin are scaled by mscale, so q·k attention gets mscale² (via cos²+sin²≈1)."""
    yarn = YarnScaling(
        factor=32.0,
        original_max_position_embeddings=4096,
        mscale=1.0,
        mscale_all_dim=0.0,
    )
    m_plain = RoPE.Config(channels_head=32, base=10_000).make()
    m_yarn = RoPE.Config(channels_head=32, base=10_000, yarn=yarn).make()
    pos = torch.arange(16)
    cos_p, _ = m_plain(pos)
    cos_y, _ = m_yarn(pos)
    # At position 0 cos is always 1 (plain) -> mscale (yarn).
    assert math.isclose(cos_p[0, 0, 0].item(), 1.0, abs_tol=1e-5)
    assert math.isclose(cos_y[0, 0, 0].item(), m_yarn._mscale, abs_tol=1e-5)


# -- hf_inv_freq tests -------------------------------------------------


def test_hf_inv_freq_matches_hf_formula():
    """hf_inv_freq=True reproduces the HF transformers inv_freq exactly."""
    base = 1_000_000.0
    dim = 16
    m = RoPE.Config(channels_head=dim, base=base, hf_inv_freq=True).make()
    expected = 1.0 / (
        base ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim)
    )
    got = m._inv_freqs[0].squeeze()
    assert torch.equal(got, expected)


def test_hf_inv_freq_default_off():
    """Default (hf_inv_freq=False) uses the higher-precision linspace formula."""
    base = 1_000_000.0
    dim = 16
    m_hf = RoPE.Config(channels_head=dim, base=base, hf_inv_freq=True).make()
    m_default = RoPE.Config(channels_head=dim, base=base).make()
    hf_freq = m_hf._inv_freqs[0].squeeze(-1)
    default_freq = m_default._inv_freqs[0].squeeze(-1)
    # Close but not identical — different computation paths
    assert torch.allclose(hf_freq, default_freq, atol=1e-6)
    assert not torch.equal(hf_freq, default_freq)


def test_hf_inv_freq_cos_sin_exact():
    """cos/sin from hf_inv_freq=True match HF's rotary embedding output."""
    base = 10_000.0
    dim = 32
    m = RoPE.Config(channels_head=dim, base=base, hf_inv_freq=True).make()
    cos, sin = m(torch.arange(8))
    # Reproduce HF's computation manually
    inv_freq = 1.0 / (
        base ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim)
    )
    pos = torch.arange(8).float()
    freqs = torch.outer(pos, inv_freq)  # (8, dim//2)
    assert torch.equal(cos, freqs.cos().unsqueeze(-2))
    assert torch.equal(sin, freqs.sin().unsqueeze(-2))


def test_rope_rotate_partial():
    """Partial rotation: cos/sin covers fewer channels than q/k."""
    q = torch.randn(2, 8, 4, 32)
    k = torch.randn(2, 8, 4, 32)
    cos = torch.randn(8, 1, 4)  # half-dim for rot_dim=8 out of D=32
    sin = torch.randn(8, 1, 4)
    q_rot, k_rot = RoPE.rotate(q, k, cos, sin)
    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape
    # Unrotated suffix should be unchanged.
    assert torch.equal(q_rot[..., 8:], q[..., 8:])
    assert torch.equal(k_rot[..., 8:], k[..., 8:])
    # Rotated prefix should differ.
    assert not torch.equal(q_rot[..., :8], q[..., :8])


def test_rope_rotate_partial_interleave():
    """Partial rotation with interleave=True."""
    q = torch.randn(2, 8, 4, 32)
    k = torch.randn(2, 8, 4, 32)
    cos = torch.randn(8, 1, 4)
    sin = torch.randn(8, 1, 4)
    q_rot, _k_rot = RoPE.rotate(q, k, cos, sin, interleave=True)
    assert q_rot.shape == q.shape
    assert torch.equal(q_rot[..., 8:], q[..., 8:])


def test_rope_rotate_partial_full_coverage():
    """When cos covers all channels, partial path is not taken."""
    q = torch.randn(2, 8, 1, 32)
    k = torch.randn(2, 8, 1, 32)
    cos = torch.randn(8, 1, 16)  # half-dim = D/2
    sin = torch.randn(8, 1, 16)
    q_full, _k_full = RoPE.rotate(q, k, cos, sin)
    # No unrotated suffix — full rotation.
    assert not torch.equal(q_full, q)


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
