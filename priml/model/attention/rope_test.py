"""Tests for rope module."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import math

from configgle import Fig
from configgle.testing import assert_pprint_golden
from torch import Tensor

import pytest
import torch

from priml.model.attention.rope import (
    GeometricFrequencies,
    HuggingFaceFrequencies,
    RoPE,
    RoPEMixed,
    YarnScaling,
)
from priml.testing.bfb import assert_bfb_against_golden, bfb_devices
from priml.testing.fixtures import (
    cleanup_cuda,  # noqa: F401 -- pytest fixture, injected by name not called
)


_TESTDATA = Path(__file__).parent.resolve() / "testdata"


@pytest.mark.parametrize(
    ("name", "config"),
    [
        ("geometric_frequencies", GeometricFrequencies.Config()),
        ("hugging_face_frequencies", HuggingFaceFrequencies.Config()),
        ("yarn_scaling", YarnScaling.Config()),
        ("rope", RoPE.Config(8)),
        ("rope_mixed", RoPEMixed.Config(8)),
    ],
)
def test_rope_config_pprint(
    name: str,
    config: Fig[object],
) -> None:
    assert_pprint_golden(
        test_file=__file__,
        name=name,
        config=config,
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


def _rotate_reference(x: Tensor, cos: Tensor, sin: Tensor, interleave: bool) -> Tensor:
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
    def fail_stack(*args: object, **kwargs: object) -> Tensor:
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

    def fail_empty_like(*args: object, **kwargs: object) -> Tensor:
        assert args or kwargs
        raise AssertionError("RoPE._rotate should not call torch.empty_like")

    monkeypatch.setattr(torch, "empty_like", fail_empty_like)
    x = torch.randn(2, 5, 3, 8, dtype=torch.float32)
    cos = torch.randn(5, 1, 4, dtype=torch.float32)
    sin = torch.randn(5, 1, 4, dtype=torch.float32)

    out = RoPE._rotate(x, cos, sin, interleave)
    assert torch.isfinite(out).all()


def test_rope_rotate_padding() -> None:
    """cos/sin shorter than seq_len should be padded with identity."""
    q = torch.randn(2, 16, 1, 32)
    k = torch.randn(2, 16, 1, 32)
    cos = torch.randn(8, 1, 16)  # shorter than seq_len=16
    sin = torch.randn(8, 1, 16)
    q_rot, k_rot = RoPE.rotate(q, k, cos, sin)
    assert torch.equal(q_rot[:, 8:], q[:, 8:])
    assert torch.equal(k_rot[:, 8:], k[:, 8:])


def test_rope_sum_mode():
    m = RoPE.Config([16, 16], reduction_mode="sum").make()
    pos = torch.stack([torch.arange(8), torch.arange(8)], dim=-1)
    cos, _sin = m(pos)
    assert cos.shape == (8, 1, 8)  # sum reduces to single set


def test_rope_auto_split_dim():
    """Scalar dim + a per-axis table list auto-splits in cat mode."""
    m = RoPE.Config(
        128,
        frequencies=[HuggingFaceFrequencies.Config() for _ in range(3)],
    ).make()
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
    m = RoPEMixed.Config(32, num_heads=4).make()
    pos = torch.arange(8)
    cos, _sin = m(pos)
    assert cos.shape[-2] == 4  # per-head


def test_rope_mixed_learnable():
    m = RoPEMixed.Config(32, num_heads=4, learnable=True).make()
    # Learnable freqs should be parameters
    param_count = sum(1 for p in m.parameters() if p.requires_grad)
    assert param_count > 0


def test_rope_mixed_single_head():
    m = RoPEMixed.Config(32, num_heads=1).make()
    pos = torch.arange(8)
    cos, _sin = m(pos)
    assert cos.shape[-2] == 1


def test_rope_mixed_keeps_its_learned_frequencies_across_a_move():
    """A move must carry the per-head frequencies, not rebuild them.

    ``RoPEMixed._apply`` reaches past ``RoPE._apply`` to the grandparent
    because the parent rebuilds ``_inv_freqs`` from the frequency tables --
    discarding the per-head scaling, and in fact failing outright, since the
    rebuild assigns a plain ``list`` where this class holds an
    ``nn.ParameterList``. Nothing covered that skip, so a well-meaning
    ``super()._apply(...)`` would have looked correct.
    """
    torch.manual_seed(0)
    m = RoPEMixed.Config(32, num_heads=4, learnable=True).make()
    before = [f.detach().clone() for f in m._inv_freqs]

    _ = m.to(torch.device("cpu"))

    after = [f.detach().clone() for f in m._inv_freqs]
    assert all(torch.equal(a, b) for a, b in zip(before, after, strict=True))
    # Per-head, not the unscaled base: a rebuild would drop the head axis.
    assert all(f.shape[0] == 4 for f in after if f.numel())


def test_yarn_kimi_k2_mscale_unity():
    """Kimi-K2 YaRN has mscale = mscale_all_dim = 1.0 → ratio collapses to 1.0."""
    yarn = YarnScaling.Config()
    yarn.factor = 32.0
    yarn.original_max_position_embeddings = 4096
    yarn.beta_fast = 1.0
    yarn.beta_slow = 1.0
    yarn.mscale_all_dim = 1.0
    yarn.inner = HuggingFaceFrequencies.Config(base=50_000.0)
    m = RoPE.Config(channels_head=64, frequencies=yarn).make()
    assert math.isclose(m._mscale, 1.0)


def test_yarn_split_interp_extrap():
    """Low-freq (high-index) channels get inv_freq/factor; high-freq stay original."""
    yarn = YarnScaling.Config()
    yarn.factor = 8.0
    yarn.original_max_position_embeddings = 512
    dim = 64
    yarn.inner = GeometricFrequencies.Config(base=10_000.0)
    m = RoPE.Config(channels_head=dim, frequencies=yarn).make()
    orig, _ = GeometricFrequencies(GeometricFrequencies.Config(base=10_000.0))(
        channels=dim, device=torch.device("cpu")
    )
    got = m._inv_freqs[0].squeeze(0)
    # First channel (highest freq) stays original; last channel (lowest
    # freq) is interpolated by factor.
    assert torch.isclose(got[0], orig[0], atol=1e-6)
    assert torch.isclose(got[-1], orig[-1] / yarn.factor, atol=1e-6)


def test_yarn_mscale_log_formula():
    """Mscale = 0.1 * log(factor) * mscale + 1 when mscale_all_dim=0."""
    yarn = YarnScaling.Config()
    yarn.factor = 40.0
    m = RoPE.Config(channels_head=32, frequencies=yarn).make()
    expected = 0.1 * math.log(40.0) + 1.0
    assert math.isclose(m._mscale, expected)


def test_yarn_mscale_applied_to_cos_sin():
    """cos/sin are scaled by mscale, so q·k attention gets mscale² (via cos²+sin²≈1)."""
    yarn = YarnScaling.Config()
    yarn.factor = 32.0
    m_plain = RoPE.Config(channels_head=32).make()
    m_yarn = RoPE.Config(channels_head=32, frequencies=yarn).make()
    pos = torch.arange(16)
    cos_p, _ = m_plain(pos)
    cos_y, _ = m_yarn(pos)
    # At position 0 cos is always 1 (plain) -> mscale (yarn).
    assert math.isclose(cos_p[0, 0, 0].item(), 1.0, abs_tol=1e-5)
    assert math.isclose(cos_y[0, 0, 0].item(), m_yarn._mscale, abs_tol=1e-5)


# -- frequency-table tests ---------------------------------------------


def test_hf_inv_freq_matches_hf_formula():
    """The HF table reproduces the HF transformers inv_freq exactly."""
    base = 1_000_000.0
    dim = 16
    m = RoPE.Config(
        channels_head=dim, frequencies=HuggingFaceFrequencies.Config(base=base)
    ).make()
    expected = 1.0 / (
        base ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim)
    )
    got = m._inv_freqs[0].squeeze()
    assert torch.equal(got, expected)


def test_the_default_table_is_the_hugging_face_one():
    """Every checkpoint this library loads was trained under HF's formula.

    So it is the default, and a port states only what it changes. The
    higher-precision :class:`GeometricFrequencies` is opt-in: it agrees to
    ~1e-7 but is not bit-identical, which is what checkpoint parity needs.
    """
    base = 10_000.0
    dim = 16
    default = RoPE.Config(channels_head=dim).make()
    explicit = RoPE.Config(
        channels_head=dim, frequencies=HuggingFaceFrequencies.Config(base=base)
    ).make()
    geometric = RoPE.Config(
        channels_head=dim, frequencies=GeometricFrequencies.Config(base=base)
    ).make()
    default_freq = default._inv_freqs[0].squeeze(-1)
    assert torch.equal(default_freq, explicit._inv_freqs[0].squeeze(-1))
    geometric_freq = geometric._inv_freqs[0].squeeze(-1)
    assert torch.allclose(default_freq, geometric_freq, atol=1e-6)
    assert not torch.equal(default_freq, geometric_freq)


def test_the_default_base_and_width_need_no_restating():
    """``RoPE.Config()`` alone is a usable 64-channel, base-1e4 embedding."""
    config = RoPE.Config()
    assert config.channels_head == 64
    assert isinstance(config.frequencies, HuggingFaceFrequencies.Config)
    assert config.frequencies.base == 10_000.0


def test_hf_inv_freq_cos_sin_exact():
    """cos/sin from hf_inv_freq=True match HF's rotary embedding output."""
    base = 10_000.0
    dim = 32
    m = RoPE.Config(
        channels_head=dim, frequencies=HuggingFaceFrequencies.Config(base=base)
    ).make()
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


def test_rope_rotate_rejects_cos_longer_than_sequence():
    """More rope positions than q/k positions is a caller error, not a broadcast.

    The short-cos branch right-pads with identity, but nothing guards the
    opposite. At ``S == 1`` the sequence axis broadcasts instead of erroring,
    so a five-position table applied to a one-position query returned a
    five-position result -- a wrong shape produced silently.
    """
    q = torch.randn(1, 1, 2, 8)
    k = torch.randn(1, 1, 2, 8)
    cos = torch.randn(1, 5, 1, 4)
    sin = torch.randn(1, 5, 1, 4)
    with pytest.raises(ValueError, match="positions"):
        _ = RoPE.rotate(q, k, cos, sin)


def test_rope_rejects_a_base_that_cannot_build_a_table():
    """A base is a scalar config field, so it is checked where it is free.

    ``nan``/``inf`` produced an all-NaN frequency table with no error, and
    ``base=1`` divides by ``log(1)`` inside the YaRN correction. The check
    lives on the TABLE, which is what owns the base.
    """
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ValueError, match="base"):
            _ = HuggingFaceFrequencies.Config(base=bad).make()
    yarn = YarnScaling.Config()
    yarn.factor = 32.0
    yarn.inner = HuggingFaceFrequencies.Config(base=1.0)
    with pytest.raises(ValueError, match="base"):
        _ = RoPE.Config(32, frequencies=yarn).make()


def test_yarn_needs_a_table_whose_spacing_has_a_base():
    """The YaRN ramp is measured in rotations per token, so it needs one.

    A table built from a learned vector or a lookup has no base, which is why
    ``base`` is not a ``FrequencyTable`` parameter -- and why wrapping such a
    table in YaRN is rejected where the wrapping happens.
    """

    class _NoBase:
        class Config(Fig["_NoBase"]):
            pass

        def __init__(self, config: Config) -> None:
            del config

        def __call__(
            self, *, channels: int, device: torch.device
        ) -> tuple[Tensor, float]:
            return torch.ones(channels // 2, device=device), 1.0

    yarn = YarnScaling.Config()
    yarn.inner = _NoBase.Config()
    with pytest.raises(TypeError, match="has none"):
        _ = RoPE.Config(32, frequencies=yarn).make()


def test_rope_rejects_an_integer_factor_dtype():
    """``dtype`` rounds the cos/sin factors, so an integer one destroys them.

    Measured: ``dtype=int64`` quantized every factor to ``{0, 1}`` -- the
    rotation stops being a rotation, silently.
    """
    with pytest.raises(ValueError, match="floating point"):
        _ = RoPE.Config(32, dtype=torch.int64).make()


def test_rope_mixed_rejects_a_non_positive_head_count():
    """``num_heads=0`` built a one-head table rather than failing."""
    for bad in (0, -1):
        with pytest.raises(ValueError, match="num_heads"):
            _ = RoPEMixed.Config(32, num_heads=bad).make()


def test_smallest_recommended_base_rejects_unusable_position_counts():
    """``max_positions <= 1`` has no real base: measured complex, or 0.0."""
    for bad in (0, 1, -5):
        with pytest.raises(ValueError, match="max_positions"):
            _ = RoPE.smallest_recommended_base(64, bad)


def test_yarn_scaling_rejects_degenerate_parameters():
    """Every YaRN parameter that divides must be positive.

    ``factor=0`` is the one worth a test: it raised nothing and warned about
    nothing, dividing the interpolated frequencies to infinity and returning a
    table of NaNs -- a run that trains on garbage rather than one that stops.
    """
    # NaN and inf fail every comparison, so ``value <= 0`` waved them through
    # and the table came out all-NaN with no error anywhere. Validation is in
    # ``__init__``, not ``finalize``, so it fires at ``make()``.
    for bad in (float("nan"), float("inf"), 0.0):
        config = YarnScaling.Config()
        config.factor = bad
        with pytest.raises(ValueError, match="finite and positive"):
            _ = config.make()
    for field in ("original_max_position_embeddings", "beta_fast", "beta_slow"):
        config = YarnScaling.Config()
        config.factor = 32.0
        setattr(config, field, 0)
        with pytest.raises(ValueError, match="finite and positive"):
            _ = config.make()


def test_frequency_scaling_is_an_injection_point_not_a_named_field():
    """A scaling is supplied as a config, so a variant needs no library edit.

    ``yarn`` was a frozen dataclass in the slot, so changing one parameter
    meant ``dataclasses.replace`` and a new object. As a ``Makeable`` slot it
    is built and mutated like every other config in the tree, and anything
    satisfying ``FrequencyTable`` fits -- linear, dynamic-NTK, LongRoPE.
    """
    config = RoPE.Config(64)
    yarn = config.frequencies = YarnScaling.Config()
    yarn.factor = 32.0
    yarn.original_max_position_embeddings = 4096
    # Mutating after assignment is the point: no replace(), no rebuild.
    yarn.factor = 16.0
    model = config.make()
    assert math.isclose(model._mscale, 0.1 * math.log(16.0) + 1.0)

    scaled = model(torch.arange(8))[0]
    assert torch.isfinite(scaled).all()


def test_a_custom_frequency_table_needs_no_library_change():
    """Any callable matching the protocol drops into the slot."""

    class _Halve:
        class Config(Fig["_Halve"]):
            pass

        def __init__(self, config: Config) -> None:
            del config

        def __call__(
            self, *, channels: int, device: torch.device
        ) -> tuple[Tensor, float]:
            inner, _ = HuggingFaceFrequencies(HuggingFaceFrequencies.Config())(
                channels=channels, device=device
            )
            return inner / 2.0, 1.0

    config = RoPE.Config(32)
    config.frequencies = _Halve.Config()
    plain = RoPE.Config(32).make()._inv_freqs[0]
    halved = config.make()._inv_freqs[0]
    torch.testing.assert_close(halved, plain / 2.0)


def test_the_two_builders_differ_only_in_precision():
    """HuggingFace and geometric are the same frequencies, computed differently.

    The HF spelling rounds the exponent to float32 before the power; the
    geometric one keeps float64 until the end. Agreement to one float32 ULP
    is what makes the default a numerical improvement rather than a different
    embedding.
    """
    device = torch.device("cpu")
    geometric, _ = GeometricFrequencies(GeometricFrequencies.Config(base=10_000.0))(
        channels=64, device=device
    )
    hugging_face, _ = HuggingFaceFrequencies(
        HuggingFaceFrequencies.Config(),
    )(channels=64, device=device)
    assert float(((geometric - hugging_face).abs() / geometric).max()) < 2e-7


def test_per_axis_tables_must_agree_on_mscale() -> None:
    """One scalar scales every axis, so a per-axis correction cannot be held.

    Assigning it inside the per-axis loop let the LAST axis win: YaRN on axis
    0 beside a plain table on axis 1 silently produced mscale 1.0, so the run
    trained at the wrong attention temperature with nothing raised.
    """
    config = RoPE.Config([32, 32])
    config.frequencies = [
        YarnScaling.Config(factor=32.0),
        HuggingFaceFrequencies.Config(),
    ]
    with pytest.raises(ValueError, match="disagree on mscale"):
        _ = config.make()
    # Agreeing tables are unaffected, and the correction survives.
    agreeing = RoPE.Config([32, 32])
    agreeing.frequencies = [
        YarnScaling.Config(factor=32.0),
        YarnScaling.Config(factor=32.0),
    ]
    assert agreeing.make()._mscale > 1.0


def test_yarn_scaling_builds_finite_frequencies():
    """The accepted settings still produce a usable table."""
    yarn = YarnScaling.Config()
    yarn.factor = 32.0
    m = RoPE.Config(64, frequencies=yarn).make()
    cos, sin = m(torch.arange(8))
    assert torch.isfinite(cos).all()
    assert torch.isfinite(sin).all()


def test_rope_smallest_recommended_base_rejects_two_channels():
    """``c == 2`` has no finite base, so it must raise rather than divide by zero.

    The exponent is ``c / (c - 2)``, and the class documents every nonzero axis
    as needing ``>= 2`` channels -- so the documented boundary was exactly the
    value that crashed with a bare ZeroDivisionError.
    """
    with pytest.raises(ValueError, match="at least 4"):
        _ = RoPE.smallest_recommended_base(2, 16)


@pytest.mark.parametrize("device", bfb_devices(), ids=str)
@pytest.mark.parametrize(
    ("name", "frequencies"),
    [
        ("geometric_frequencies", GeometricFrequencies.Config()),
        ("hugging_face_frequencies", HuggingFaceFrequencies.Config()),
        ("yarn_scaling", YarnScaling.Config()),
    ],
)
def test_frequency_table_bfb(device: str, name: str, frequencies: object) -> None:
    assert isinstance(
        frequencies,
        (
            GeometricFrequencies.Config,
            HuggingFaceFrequencies.Config,
            YarnScaling.Config,
        ),
    )
    assert_bfb_against_golden(
        golden_dir=_TESTDATA,
        golden_name=name,
        build_module=lambda: RoPE.Config(8, frequencies=frequencies).make().to(device),
        build_input=lambda: torch.arange(4),
        seed=0,
        run=lambda module, positions: torch.cat(cast(RoPE, module)(positions), dim=-1),
    )


@pytest.mark.parametrize("device", bfb_devices(), ids=str)
def test_rope_bfb(device: str) -> None:
    assert_bfb_against_golden(
        golden_dir=_TESTDATA,
        golden_name="rope",
        build_module=lambda: RoPE.Config(8).make().to(device),
        build_input=lambda: torch.arange(4),
        seed=0,
        run=lambda module, positions: torch.cat(cast(RoPE, module)(positions), dim=-1),
    )


@pytest.mark.parametrize("device", bfb_devices(), ids=str)
def test_rope_mixed_bfb(device: str) -> None:
    config = RoPEMixed.Config(8)
    config.num_heads = 2
    config.learnable = True
    assert_bfb_against_golden(
        golden_dir=_TESTDATA,
        golden_name="rope_mixed",
        build_module=lambda: config.make().to(device),
        build_input=lambda: torch.arange(4),
        seed=0,
        run=lambda module, positions: torch.cat(
            cast(RoPEMixed, module)(positions), dim=-1
        ),
    )


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
