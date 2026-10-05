"""Tests for init module."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final, override

import inspect
import math

from torch import nn

import pytest
import torch

from priml.model.attention.attention import Attention
from priml.model.attention.gated_delta_net import GatedDeltaNet
from priml.model.attention.mla import MultiHeadLatentAttention
from priml.model.attention.rope import RoPE, RoPEMixed
from priml.model.custom_types import DepthIndex, HasResetParameters
from priml.model.init import (
    call_init,
    dirac,
    fan_in_truncated_normal,
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
from priml.model.moe import MoE, SoftmaxRouter
from priml.model.norm import CenteredRMSNorm
from priml.model.transformer.block import TransformerBlock
from priml.testing.bfb import assert_bfb_against_golden
from priml.testing.golden import assert_text_golden


if TYPE_CHECKING:
    from collections.abc import Callable


_CWD: Final = Path(__file__).resolve().parent


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
        # Dirac requires the 3x3 convolution to retain its center pixel.
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
            for function in (
                call_init,
                kaiming_uniform,
                kaiming_normal,
                xavier_uniform,
                xavier_normal,
                normal,
                truncated_normal,
                fan_in_truncated_normal,
                unit_fan_in_uniform,
                mup_output,
                dirac,
            )
        ),
    )


def test_init_bfb() -> None:
    assert_bfb_against_golden(
        golden_dir=_CWD / "testdata",
        golden_name="init",
        build_module=_InitModule,
        build_input=lambda: torch.arange(6, dtype=torch.float32).reshape(2, 3),
        seed=0,
    )


def test_call_init_with_depth():
    w = torch.empty(16, 17)
    kaiming_uniform(w, depth_index=((3, 4),))
    assert w.std() > 0


@pytest.mark.parametrize(("shape", "expected_std"), [((5,), 1 / 5), ((5, 7), 1 / 7)])
def test_mup_output_uses_the_last_input_axis(
    monkeypatch: pytest.MonkeyPatch,
    shape: tuple[int, ...],
    expected_std: float,
) -> None:
    calls: list[tuple[torch.Tensor, float]] = []

    def normal_(tensor: torch.Tensor, *, std: float) -> torch.Tensor:
        calls.append((tensor, std))
        return tensor

    monkeypatch.setattr(nn.init, "normal_", normal_)
    weight = torch.empty(shape)

    mup_output(weight)

    assert calls == [(weight, expected_std)]


def test_unit_fan_in_uniform_uses_both_bounds_and_last_axis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[torch.Tensor, float, float]] = []

    def uniform_(tensor: torch.Tensor, low: float, high: float) -> torch.Tensor:
        calls.append((tensor, low, high))
        return tensor

    monkeypatch.setattr(nn.init, "uniform_", uniform_)
    weight = torch.empty(2, 3, 4)

    unit_fan_in_uniform(weight)

    bound = 3**0.5 * 4**-0.5
    assert calls == [(weight, -bound, bound)]


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
    w = torch.empty(16, 17)
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
    w = torch.empty(4, 5)
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
        w = torch.empty(32, 33)
        fn(w, depth_index=((2, 3),))
        assert w.std() > 0, f"{fn.__name__} produced zero std"


def test_depth_scaling():
    torch.manual_seed(0)
    w0 = torch.empty(64, 65)
    kaiming_uniform(w0, depth_index=((0, 1),))

    torch.manual_seed(0)
    w3 = torch.empty(64, 65)
    kaiming_uniform(w3, depth_index=((3, 4),))

    assert w0.std() > w3.std()


def test_depth_zero_no_scaling():
    """depth_index=((0, 1),) and depth_index=() should produce no scaling."""
    torch.manual_seed(0)
    w_neg = torch.empty(64, 65)
    kaiming_uniform(w_neg, depth_index=())

    torch.manual_seed(0)
    w_zero = torch.empty(64, 65)
    kaiming_uniform(w_zero, depth_index=((0, 1),))

    assert torch.allclose(w_neg, w_zero)


def test_depth_one_scales_by_the_square_root_of_two(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fill_ones(tensor: torch.Tensor, **kwargs: object) -> torch.Tensor:
        del kwargs
        return tensor.fill_(1.0)

    monkeypatch.setattr(nn.init, "kaiming_uniform_", fill_ones)
    weight = torch.empty(2, 3)

    kaiming_uniform(weight, depth_index=((1, 2),))

    torch.testing.assert_close(weight, torch.full_like(weight, 2**-0.5))


def test_truncated_normal_variance_correction_passes_closed_form_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[torch.Tensor, float, float, float]] = []

    def trunc_normal_(
        tensor: torch.Tensor,
        *,
        std: float,
        a: float,
        b: float,
    ) -> torch.Tensor:
        calls.append((tensor, std, a, b))
        return tensor

    monkeypatch.setattr(nn.init, "trunc_normal_", trunc_normal_)
    weight = torch.empty(2, 3)
    requested_std = 0.75
    lower, upper = -1.25, 2.0
    sqrt2 = 2.0**0.5
    z = (math.erf(upper / sqrt2) - math.erf(lower / sqrt2)) / 2.0
    inv_sqrt_2pi = 1.0 / (2.0 * math.pi) ** 0.5
    pdf_u = inv_sqrt_2pi * math.exp(-0.5 * upper * upper)
    pdf_l = inv_sqrt_2pi * math.exp(-0.5 * lower * lower)
    ratio = (pdf_u - pdf_l) / z
    corrected_std = (
        requested_std
        / (1.0 - (upper * pdf_u - lower * pdf_l) / z - ratio * ratio) ** 0.5
    )

    truncated_normal(
        weight,
        std=requested_std,
        lower=lower,
        upper=upper,
        variance_correction=True,
    )

    assert len(calls) == 1
    tensor, std, a, b = calls[0]
    assert tensor is weight
    assert std == pytest.approx(corrected_std)
    assert a == pytest.approx(lower * corrected_std)
    assert b == pytest.approx(upper * corrected_std)


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


@pytest.mark.parametrize("width", [1, 3, 4, 7, 512, 921])
def test_fan_in_truncated_normal_is_truncated_normal_at_fan_in_std(width: int) -> None:
    """Each form equals the explicit draw it names, at awkward widths too."""
    std = width**-0.5

    def drawn(init: Callable[[torch.Tensor], object]) -> torch.Tensor:
        w = torch.empty(3, width)
        torch.manual_seed(0)
        init(w)
        return w

    assert torch.equal(
        drawn(fan_in_truncated_normal),
        drawn(lambda w: truncated_normal(w, std=std)),
    )
    assert torch.equal(
        drawn(lambda w: fan_in_truncated_normal(w, variance_correction=True)),
        drawn(lambda w: truncated_normal(w, std=std, variance_correction=True)),
    )
    assert torch.equal(
        drawn(lambda w: fan_in_truncated_normal(w, absolute_bounds=True)),
        drawn(lambda w: nn.init.trunc_normal_(w, std=std)),
    )


def test_fan_in_truncated_normal_ignores_depth() -> None:
    w0, w3 = torch.empty(5, 6), torch.empty(5, 6)
    torch.manual_seed(0)
    fan_in_truncated_normal(w0)
    torch.manual_seed(0)
    fan_in_truncated_normal(w3, depth_index=((3, 4),))
    assert torch.equal(w0, w3)


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


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float8_e4m3fn])
@pytest.mark.parametrize(
    "initializer",
    [
        kaiming_uniform,
        kaiming_normal,
        xavier_uniform,
        xavier_normal,
        normal,
        truncated_normal,
        unit_fan_in_uniform,
        mup_output,
    ],
)
def test_a_narrow_tensor_holds_the_fp32_draw_rounded_once(
    initializer: Callable[..., None],
    dtype: torch.dtype,
) -> None:
    """``call_init`` draws in fp32 and rounds once, whatever the tensor's dtype.

    Drawn in bf16, torch's CPU generator fills a uniform from 8 random bits
    (256 values per tensor; fp16 keeps 11) and runs the normal's Box-Muller in
    bf16, which never passes ~3.3 sigma. The depth scale rides along, so it
    rounds once too.
    """
    torch.manual_seed(0)
    wide = torch.empty(64, 65)
    call_init(initializer, wide, depth_index=((3, 4),))
    torch.manual_seed(0)
    narrow = torch.empty(64, 65, dtype=dtype)
    call_init(initializer, narrow, depth_index=((3, 4),))
    assert torch.equal(narrow, wide.to(dtype))


@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.float64, torch.int16, torch.bool, torch.complex64],
)
def test_other_dtypes_are_initialized_in_place(dtype: torch.dtype) -> None:
    """Only floats narrower than fp32 widen: every other dtype gets the tensor itself."""
    tensor = torch.zeros(4, dtype=dtype)
    seen: list[torch.Tensor] = []
    call_init(seen.append, tensor)
    assert len(seen) == 1
    assert seen[0] is tensor


def test_a_bf16_linear_holds_the_fp32_linears_init_rounded_once() -> None:
    torch.manual_seed(0)
    wide = Linear.Config(64, 64, bias=True, init_bias=normal).make()
    torch.manual_seed(0)
    narrow = Linear.Config(
        64,
        64,
        bias=True,
        init_bias=normal,
        dtype=torch.bfloat16,
    ).make()
    assert narrow.weight.dtype == torch.bfloat16
    assert torch.equal(narrow.weight, wide.weight.bfloat16())
    assert narrow.bias is not None
    assert wide.bias is not None
    assert torch.equal(narrow.bias, wide.bias.bfloat16())


def test_dirac_conv2d():
    w = torch.empty(8, 10, 3, 5)
    dirac(w)
    # Center pixel of each filter for matching in/out channel should be ~1.
    center = w[:, :, 1, 2]
    expected = torch.cat((torch.eye(8), torch.zeros(8, 2)), dim=1)
    assert torch.allclose(center, expected, atol=1e-6)
    # Non-center pixels should be zero.
    mask = torch.ones(3, 5, dtype=torch.bool)
    mask[1, 2] = False
    assert torch.allclose(w[:, :, mask], torch.zeros(8, 10, 14))


@pytest.mark.parametrize(
    "name",
    [
        "linear",
        "ensemble_linear",
        "centered_rmsnorm",
        "self_attention",
        "transformer_block",
        "gated_delta_net",
        "moe",
        "mla",
        "rope_mixed_learnable",
        "rope_mixed_heads1",
        "rope_mixed_sum",
    ],
)
def test_reset_parameters_reinitializes_every_param(name: str) -> None:
    """After wiping every param and float buffer to NaN.

    One ``reset_parameters`` restores all of them to finite values.

        A param or float buffer left NaN means a module owns state that
        ``reset_parameters`` does not initialize -- so it is not the complete single
        source of truth and would ship ``to_empty`` garbage on the meta path. This
        mirrors the production ``materialize_meta`` audit, which poisons and checks
        params AND float buffers (integer buffers cannot hold NaN and are skipped).
    """
    builders: dict[str, Callable[[], nn.Module]] = {
        "linear": lambda: Linear.Config(
            channels_in=16,
            channels_out=8,
            bias=True,
        ).make(),
        "ensemble_linear": lambda: EnsembleLinear.Config(
            channels_in=8,
            channels_out=8,
            num_ensemble=2,
            bias=True,
        ).make(),
        "centered_rmsnorm": lambda: CenteredRMSNorm(
            CenteredRMSNorm.Config(channels_in=8),
        ),
        "self_attention": lambda: Attention.Config(
            channels_in=16,
            num_heads=2,
            channels_head=8,
        ).make(),
        "transformer_block": lambda: TransformerBlock.Config(
            channels_in=16,
            attn=Attention.Config(num_heads=2, channels_head=8),
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
            router=SoftmaxRouter.Config(num_experts=4, top_k=2),
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
    model = builders[name]()
    # Name -> tensor over params + buffers, matching the materialize audit.
    state = [*model.named_parameters(), *model.named_buffers()]
    with torch.no_grad():
        for _, tensor in state:
            if tensor.is_floating_point():
                tensor.fill_(float("nan"))
    assert isinstance(model, HasResetParameters)
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
