"""Tests for init module."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import override

import inspect

from torch import nn

import pytest
import torch

from priml.model.attention.gated_delta_net import GatedDeltaNet
from priml.model.attention.mla import MultiHeadLatentAttention
from priml.model.attention.rope import RoPE, RoPEMixed
from priml.model.attention.self_attention import SelfAttention
from priml.model.custom_types import DepthIndex
from priml.model.init import (
    call_init,
    dirac,
    kaiming_normal,
    kaiming_uniform,
    mup_output,
    normal,
    truncated_normal,
    unit_fan_in_uniform,
    xavier_normal,
    xavier_uniform,
)
from priml.model.linear import EnsembleLinear, Linear
from priml.model.moe import MoE, Router
from priml.model.norm import CenteredRMSNorm
from priml.model.transformer.block import TransformerBlock
from priml.testing.bfb import assert_bfb_against_golden
from priml.testing.fixtures import (
    cleanup_cuda,  # noqa: F401 -- pytest fixture, injected by name not called
)
from priml.testing.golden import assert_text_golden


_TESTDATA = Path(__file__).parent.resolve() / "testdata"
_INIT_FUNCTIONS = (
    call_init,
    kaiming_uniform,
    kaiming_normal,
    xavier_uniform,
    xavier_normal,
    normal,
    truncated_normal,
    unit_fan_in_uniform,
    mup_output,
    dirac,
)


class _InitModule(nn.Module):
    @override
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        initialized: list[torch.Tensor] = []
        for initializer in (
            kaiming_uniform,
            kaiming_normal,
            xavier_uniform,
            xavier_normal,
            normal,
            truncated_normal,
            unit_fan_in_uniform,
            mup_output,
        ):
            tensor = torch.empty_like(input)
            call_init(initializer, tensor, depth_index=((3, 4),))
            initialized.append(tensor.flatten())
        convolution = input.new_empty(2, 2, 3, 3)
        call_init(dirac, convolution, depth_index=((3, 4),))
        initialized.append(convolution.flatten())
        return torch.cat(initialized)


def test_init_api_text(request: pytest.FixtureRequest) -> None:
    assert_text_golden(
        request,
        test_file=__file__,
        name="init",
        rendered="\n".join(
            f"{function.__name__}{inspect.signature(function)}"
            for function in _INIT_FUNCTIONS
        ),
    )


def test_init_bfb() -> None:
    assert_bfb_against_golden(
        golden_dir=_TESTDATA,
        golden_name="init",
        build_module=_InitModule,
        build_input=lambda: torch.arange(16, dtype=torch.float32).reshape(4, 4),
        seed=0,
    )


def test_call_init_with_depth():
    w = torch.empty(16, 16)
    kaiming_uniform(w, depth_index=((3, 4),))
    assert w.std() > 0


def test_call_init_passes_a_positional_or_keyword_depth_index() -> None:
    seen: list[DepthIndex] = []

    def initializer(tensor: torch.Tensor, depth_index: DepthIndex = ()) -> None:
        del tensor
        seen.append(depth_index)

    call_init(initializer, torch.empty(1), depth_index=((1, 2),))

    assert seen == [((1, 2),)]


def test_call_init_passes_all_kwargs_to_variadic_initializer() -> None:
    seen: dict[str, object] = {}

    def initializer(tensor: torch.Tensor, **kwargs: object) -> None:
        del tensor
        seen.update(kwargs)

    call_init(initializer, torch.empty(1), depth_index=((1, 2),))

    assert seen == {"depth_index": ((1, 2),)}


def test_call_init_without_depth():
    """call_init skips depth kwarg for fns that don't accept it."""
    w = torch.empty(16, 16)
    call_init(nn.init.xavier_uniform_, w, depth_index=((3, 4),))
    assert w.std() > 0


def test_call_init_signature_inspection_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """call_init handles uninspectable callables gracefully."""
    called = False

    def initializer(tensor: torch.Tensor) -> None:
        nonlocal called
        called = True
        nn.init.zeros_(tensor)

    def fail_signature(fn: object) -> None:
        del fn
        raise ValueError("uninspectable")

    monkeypatch.setattr(inspect, "signature", fail_signature)
    w = torch.empty(4, 4)
    call_init(initializer, w, depth_index=((2, 3),))

    assert called
    assert w.abs().sum() == 0


def test_all_init_fns():
    for fn in (
        kaiming_uniform,
        kaiming_normal,
        xavier_uniform,
        xavier_normal,
        normal,
        truncated_normal,
        mup_output,
    ):
        w = torch.empty(32, 32)
        fn(w, depth_index=((2, 3),))
        assert w.std() > 0, f"{fn.__name__} produced zero std"


def test_depth_scaling():
    torch.manual_seed(0)
    w0 = torch.empty(64, 64)
    kaiming_uniform(w0, depth_index=((0, 1),))

    torch.manual_seed(0)
    w3 = torch.empty(64, 64)
    kaiming_uniform(w3, depth_index=((3, 4),))

    assert w0.std() > w3.std()


def test_depth_zero_no_scaling():
    """depth_index=((0, 1),) and depth_index=() should produce no scaling."""
    torch.manual_seed(0)
    w_neg = torch.empty(64, 64)
    kaiming_uniform(w_neg, depth_index=())

    torch.manual_seed(0)
    w_zero = torch.empty(64, 64)
    kaiming_uniform(w_zero, depth_index=((0, 1),))

    assert torch.allclose(w_neg, w_zero)


def test_truncated_normal_variance_correction_realizes_requested_std():
    """With correction on, realized std equals the request; off, it undershoots."""
    torch.manual_seed(0)
    corrected = torch.empty(400_000)
    truncated_normal(corrected, std=1.0, depth_index=(), variance_correction=True)
    torch.manual_seed(0)
    uncorrected = torch.empty(400_000)
    truncated_normal(uncorrected, std=1.0, depth_index=())

    assert abs(corrected.std().item() - 1.0) < 0.01
    # ~0.88: the truncated tail mass the correction restores.
    assert uncorrected.std().item() < 0.92


def test_truncated_normal_default_is_uncorrected():
    """The flag defaults off, so existing callers keep their init unchanged."""
    torch.manual_seed(0)
    default = torch.empty(4096)
    truncated_normal(default, std=0.02, depth_index=())
    torch.manual_seed(0)
    explicit = torch.empty(4096)
    truncated_normal(explicit, std=0.02, depth_index=(), variance_correction=False)

    assert torch.equal(default, explicit)


def test_truncated_normal_respects_scaled_bounds():
    """Correction scales the truncation bounds along with the std."""
    torch.manual_seed(0)
    w = torch.empty(100_000)
    truncated_normal(w, std=1.0, depth_index=(), variance_correction=True)

    assert w.abs().max().item() <= 2.0 * 1.1372


def test_truncated_normal_corrected_zero_std_zeros_tensor():
    w = torch.ones(16)
    truncated_normal(w, std=0.0, depth_index=(), variance_correction=True)
    assert w.abs().sum() == 0


def test_truncated_normal_corrected_depth_scaling():
    torch.manual_seed(0)
    w0 = torch.empty(4096)
    truncated_normal(w0, std=1.0, depth_index=((0, 1),), variance_correction=True)
    torch.manual_seed(0)
    w3 = torch.empty(4096)
    truncated_normal(w3, std=1.0, depth_index=((3, 4),), variance_correction=True)
    assert torch.allclose(w3, w0 / 2.0)


def test_dirac_conv2d():
    w = torch.empty(8, 8, 3, 3)
    dirac(w)
    # Center pixel of each filter for matching in/out channel should be ~1.
    center = w[:, :, 1, 1]
    assert torch.allclose(center, torch.eye(8), atol=1e-6)
    # Non-center pixels should be zero.
    mask = torch.ones(3, 3, dtype=torch.bool)
    mask[1, 1] = False
    assert torch.allclose(w[:, :, mask], torch.zeros(8, 8, 8))


_BUILDERS: dict[str, Callable[[], nn.Module]] = {
    "linear": lambda: Linear.Config(channels_in=16, channels_out=8, bias=True).make(),
    "ensemble_linear": lambda: EnsembleLinear.Config(
        channels_in=8,
        channels_out=8,
        num_ensemble=2,
        bias=True,
    ).make(),
    "centered_rmsnorm": lambda: CenteredRMSNorm(CenteredRMSNorm.Config(channels_in=8)),
    "self_attention": lambda: SelfAttention.Config(
        channels_in=16,
        num_heads=2,
        channels_head=8,
    ).make(),
    "transformer_block": lambda: TransformerBlock.Config(
        channels_in=16,
        attn=SelfAttention.Config(num_heads=2, channels_head=8),
    ).make(),
    "gated_delta_net": lambda: GatedDeltaNet.Config(
        channels_in=16,
        num_heads_k=2,
        num_heads_v=2,
        channels_k_head=8,
        channels_v_head=8,
    ).make(),
    "moe": lambda: MoE.Config(
        channels_in=16,
        router=Router.Config(num_experts=4, top_k=2),
    ).make(),
    "mla": lambda: MultiHeadLatentAttention.Config(
        channels_in=16,
        num_heads=2,
        channels_qk_nope_head=8,
        channels_qk_rope_head=4,
        channels_v_head=8,
        kv_lora_rank=8,
        rope=RoPE.Config(channels_head=4),
    ).make(),
    "rope_mixed_learnable": lambda: RoPEMixed(
        RoPEMixed.Config(channels_head=8, num_heads=2, learnable=True),
    ),
    # num_heads=1 takes the directions=None branch (no per-head scaling);
    # reduction_mode="sum" shares channels across axes. Both are distinct
    # shape-allocation paths in reset_parameters that must round-trip.
    "rope_mixed_heads1": lambda: RoPEMixed(
        RoPEMixed.Config(channels_head=8, num_heads=1, learnable=True),
    ),
    "rope_mixed_sum": lambda: RoPEMixed(
        RoPEMixed.Config(
            channels_head=8,
            num_heads=2,
            reduction_mode="sum",
            learnable=True,
        ),
    ),
}


@pytest.mark.parametrize("name", list(_BUILDERS))
def test_reset_parameters_reinitializes_every_param(name: str) -> None:
    """After wiping every param and float buffer to NaN, one ``reset_parameters``
    restores all of them to finite values.

    A param or float buffer left NaN means a module owns state that
    ``reset_parameters`` does not initialize -- so it is not the complete single
    source of truth and would ship ``to_empty`` garbage on the meta path. This
    mirrors the production ``materialize_meta`` audit, which poisons and checks
    params AND float buffers (integer buffers cannot hold NaN and are skipped).
    """
    model = _BUILDERS[name]()
    # name -> tensor over params + buffers, matching the materialize audit.
    state = [*model.named_parameters(), *model.named_buffers()]
    with torch.no_grad():
        for _, tensor in state:
            if tensor.is_floating_point():
                tensor.fill_(float("nan"))
    model.reset_parameters()
    for key, tensor in state:
        if not tensor.is_floating_point():
            continue
        assert torch.isfinite(tensor).all(), (
            f"{name}.{key}: reset_parameters left it NaN (incomplete init source)"
        )


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
