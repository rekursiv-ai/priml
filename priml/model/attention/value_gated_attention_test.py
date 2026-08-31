"""Tests for attention module."""

from __future__ import annotations

from pathlib import Path
from typing import cast

from configgle import PartialConfig
from configgle.testing import assert_pprint_golden
from torch import Tensor

import pytest
import torch

from priml.model.attention.rope import RoPE
from priml.model.attention.value_gated_attention import ValueGatedAttention
from priml.testing.bfb import assert_bfb_against_golden, bfb_devices
from priml.testing.fixtures import (
    cleanup_cuda,  # noqa: F401 -- pytest fixture, injected by name not called
)


_TESTDATA = Path(__file__).parent.resolve() / "testdata"


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
        golden_dir=_TESTDATA,
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


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
