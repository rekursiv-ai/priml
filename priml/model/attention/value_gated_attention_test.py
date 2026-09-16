"""Tests for attention module."""

from __future__ import annotations

from pathlib import Path
from typing import Final, cast

from configgle import Fig, PartialConfig
from configgle.testing import assert_pprint_golden
from torch import Tensor

import pytest
import torch

from priml.model.attention.kernel import SdpaNaive
from priml.model.attention.rope import RoPE
from priml.model.attention.value_gated_attention import (
    SdpaCausal,
    ValueGatedAttention,
)
from priml.model.cost import Compute, Cost, Flops, cost
from priml.model.norm import RMSNorm
from priml.testing.bfb import assert_bfb_against_golden, bfb_devices
from priml.testing.cost import assert_cost_matches_torch


_CWD: Final = Path(__file__).resolve().parent


def test_value_gated_attention_config_pprint() -> None:
    config = ValueGatedAttention.Config(
        channels_in=16,
        num_heads=2,
        channels_head=8,
        gate_channels=4,
        window=4,
    )
    assert_pprint_golden(
        test_file=__file__,
        name="value_gated_attention",
        config=config,
    )


def test_value_gated_attention_forwards_the_open_message_bus() -> None:
    messages: list[object] = []

    def kernel(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        *,
        message: object,
        **kwargs: object,
    ) -> Tensor:
        del k, v, kwargs
        messages.append(message)
        return q

    attention = ValueGatedAttention.Config(
        channels_in=16,
        num_heads=2,
        channels_head=8,
        gate_channels=4,
        window=4,
        kernel=PartialConfig(kernel),
    ).make()
    message = object()
    cos_sin = RoPE.Config(channels_head=8).make()(torch.arange(4))

    attention(torch.randn(1, 4, 16), cos_sin=cos_sin, message=message)

    assert messages == [message]


def test_value_gated_attention_uses_the_value_embedding() -> None:
    attention = ValueGatedAttention.Config(
        channels_in=16,
        num_heads=2,
        channels_head=8,
        gate_channels=4,
        window=4,
    ).make()
    cos_sin = RoPE.Config(channels_head=8).make()(torch.arange(4))
    x = torch.randn(1, 4, 16)
    value_embedding = torch.randn(1, 4, 16)

    assert attention(x, cos_sin=cos_sin, value_embedding=value_embedding).shape == (
        1,
        4,
        16,
    )


def test_value_gated_attention_ungated_reset() -> None:
    attention = ValueGatedAttention.Config(
        channels_in=16,
        num_heads=2,
        channels_head=8,
        gate_channels=-1,
        gated=False,
    ).make()

    attention.reset_parameters()
    assert attention.value_gate is None


def test_value_gated_attention_reset_initializes_affine_qk_norms() -> None:
    config = ValueGatedAttention.Config(
        channels_in=16,
        num_heads=2,
        channels_head=8,
        gate_channels=4,
        norm_qk=RMSNorm.Config(elementwise_affine=True),
    )
    attention = config.make()
    assert isinstance(attention.norm_q, RMSNorm)
    assert isinstance(attention.norm_k, RMSNorm)
    assert attention.norm_q.weight is not None
    assert attention.norm_k.weight is not None
    with torch.no_grad():
        attention.norm_q.weight.fill_(float("nan"))
        attention.norm_k.weight.fill_(float("nan"))

    attention.reset_parameters()

    assert torch.equal(attention.norm_q.weight, torch.ones(8))
    assert torch.equal(attention.norm_k.weight, torch.ones(8))


@pytest.mark.parametrize(
    ("config", "match"),
    [
        (
            ValueGatedAttention.Config(channels_in=16, channels_head=7),
            "must be even",
        ),
        (
            ValueGatedAttention.Config(channels_in=15, channels_head=8),
            "not divisible",
        ),
        (
            ValueGatedAttention.Config(
                channels_in=16,
                num_heads=2,
                channels_head=8,
                gate_channels=17,
            ),
            "at most channels_in",
        ),
    ],
)
def test_value_gated_attention_invalid_config_prints_before_make_rejects(
    config: ValueGatedAttention.Config,
    match: str,
) -> None:
    rendered = config.pformat(hide_default_values=False)

    assert "ValueGatedAttention.Config" in rendered
    with pytest.raises(ValueError, match=match):
        config.make()


@pytest.mark.parametrize("device", bfb_devices(), ids=str)
def test_value_gated_attention_bfb(device: str) -> None:
    assert_bfb_against_golden(
        golden_dir=_CWD / "testdata",
        golden_name="value_gated_attention",
        build_module=lambda: (
            ValueGatedAttention.Config(
                channels_in=16,
                num_heads=2,
                channels_head=8,
                gate_channels=4,
                window=2,
            )
            .make()
            .to(device)
        ),
        build_input=lambda: torch.randn(2, 4, 16),
        seed=0,
        run=lambda module, x: cast(ValueGatedAttention, module)(
            x,
            cos_sin=RoPE.Config(8).make()(torch.arange(4)),
            value_embedding=x,
        ),
    )


def test_value_gated_attention_cost_is_projections_gate_and_the_kernel() -> None:
    """Four projections plus the gate follow the matrix rule; the kernel prices itself."""
    config = ValueGatedAttention.Config(
        channels_in=16,
        num_heads=2,
        channels_head=8,
        gate_channels=4,
        window=4,
    )
    finalized = config.copy_tree().finalize()
    model_cost = finalized.cost(seq_len=32)
    kernel = cost(finalized.kernel, seq_len=32, num_heads=2, channels_head=8, window=4)
    projections = 3 * 16 * 16 + 16 * 16
    gate = 4 * 2
    assert kernel.primal.flops.matmul == 4 * 2 * 8 * 4  # Scores stop at the window.
    assert model_cost.params == projections + gate
    assert model_cost.params == sum(p.numel() for p in config.make().parameters())
    assert model_cost.primal.flops.matmul == (
        2 * (projections + gate) + kernel.primal.flops.matmul
    )
    assert model_cost.adjoint.flops.matmul == (
        4 * (projections + gate) + kernel.adjoint.flops.matmul
    )
    assert model_cost.bytes_state == 2 * 2 * 8
    # The gate's gradient reduces over each head's channels; the norm runs on
    # every q and k head row.
    norm = cost(finalized.norm_qk).tile(2 * 2, copies=2)
    assert model_cost.adjoint.flops.reduction == (
        kernel.adjoint.flops.reduction + norm.adjoint.flops.reduction + 2 * (8 - 1)
    )


def test_value_gated_attention_cost_matches_torch_through_a_naive_kernel() -> None:
    """Projections, the gate, and the kernel's two bmms are torch's whole count.

    ``SdpaCausal`` dispatches to the CPU SDPA op, which ``FlopCounterMode``
    does not register; the naive kernel makes the same products countable.
    """
    config = ValueGatedAttention.Config(
        channels_in=16,
        num_heads=2,
        channels_head=8,
        gate_channels=4,
        kernel=SdpaNaive.Config(),
    )
    assert_cost_matches_torch(
        config,
        build_input=lambda: torch.randn(1, 8, 16, requires_grad=True),
        num_tokens=8,
        bus={"seq_len": 8},
        run=lambda module, x: cast(ValueGatedAttention, module)(
            x,
            cos_sin=RoPE.Config(8).make()(torch.arange(8)),
            value_embedding=x,
        ),
    )


def test_value_gated_attention_cost_without_a_gate_or_window() -> None:
    config = ValueGatedAttention.Config(
        channels_in=16,
        num_heads=2,
        channels_head=8,
        gate_channels=-1,
        gated=False,
        norm_qk=RMSNorm.Config(elementwise_affine=True),
    )
    finalized = config.copy_tree().finalize()
    model_cost = finalized.cost(seq_len=32)
    kernel = cost(finalized.kernel, seq_len=32, num_heads=2, channels_head=8, window=-1)
    norm = cost(finalized.norm_qk)
    assert model_cost.params == 4 * 16 * 16 + 2 * norm.params  # norm_q and norm_k.
    assert model_cost.params == sum(p.numel() for p in config.make().parameters())
    assert model_cost.primal.flops.matmul == (
        2 * 4 * 16 * 16 + kernel.primal.flops.matmul
    )


class _FlatKernel:
    """A kernel that reports a fixed, recognizable cost."""

    class Config(Fig["_FlatKernel"]):
        def cost(self, **kwargs: object) -> Cost:
            del kwargs
            return Cost(primal=Compute(flops=Flops(matmul=1_000_003)))

    def __init__(self, config: Config) -> None:
        del config

    def __call__(self, q: Tensor, k: Tensor, v: Tensor, **kwargs: object) -> Tensor:
        del k, v, kwargs
        return q


def test_value_gated_attention_cost_prices_an_injected_kernel() -> None:
    """The slot's own cost reaches the total; the owner does not re-derive it."""
    config = ValueGatedAttention.Config(
        channels_in=16,
        num_heads=2,
        channels_head=8,
        window=4,
        kernel=_FlatKernel.Config(),
    )
    default = config.copy_tree()
    default.kernel = SdpaCausal.Config()
    injected = config.copy_tree().finalize().cost(seq_len=32)
    baseline = default.finalize().cost(seq_len=32)
    default_kernel = cost(
        SdpaCausal.Config(),
        seq_len=32,
        num_heads=2,
        channels_head=8,
        window=4,
    )
    assert injected.primal.flops.matmul - baseline.primal.flops.matmul == (
        1_000_003 - default_kernel.primal.flops.matmul
    )


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
