"""Tests for quantization strategies."""

from __future__ import annotations

import dataclasses

from torch import nn
from torchao.float8.float8_linear import Float8Linear

import pytest
import torch

from priml.train import quantization
from priml.train.quantization import (
    Float8ModelQuantization,
    NoModelQuantization,
    _check_float8_available,
)


def test_no_quantization_is_identity() -> None:
    model = nn.Linear(16, 16)
    out = NoModelQuantization.Config().make()(model)
    assert out is model


def _force_enabled() -> Float8ModelQuantization:
    """Build a Float8 strategy, skipping if float8 isn't available here."""
    available, reason = _check_float8_available()
    if not available:
        pytest.skip(f"float8 unavailable: {reason}")
    return Float8ModelQuantization.Config().make()


def test_default_filter_is_pure() -> None:
    """T-026: the module filter predicate must not mutate conversion state."""
    quant = _force_enabled()
    quant.converted_count = 0
    module = nn.Linear(16, 16)

    # Calling the predicate repeatedly (as torchao may) must not inflate count.
    quant._default_module_filter(module, "linear")
    quant._default_module_filter(module, "linear")

    assert quant.converted_count == 0, (
        "filter predicate mutated converted_count (counts filter calls, not conversions)"
    )


def test_disabled_instance_has_consistent_attributes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T-027: a disabled instance must not leave attributes unset."""
    monkeypatch.setattr(
        "priml.train.quantization._check_float8_available",
        lambda: (False, "forced-disabled"),
    )
    quant = Float8ModelQuantization.Config().make()

    assert quant.enabled is False
    # These must exist even on the disabled path (no AttributeError on access).
    assert hasattr(quant, "converted_count")
    assert hasattr(quant, "module_filter")
    # And a disabled strategy is an identity transform.
    model = nn.Linear(16, 16)
    assert quant(model) is model


def test_fsdp_all_gather_uses_dataclass_replace() -> None:
    """T-028: enabling FSDP all-gather must produce a valid Float8 config."""
    available, reason = _check_float8_available()
    if not available:
        pytest.skip(f"float8 unavailable: {reason}")

    quant = Float8ModelQuantization.Config(
        recipe="tensorwise",
        enable_fsdp_float8_all_gather=True,
    ).make()

    assert dataclasses.is_dataclass(quant.float8_config)
    assert quant.float8_config.enable_fsdp_float8_all_gather is True


def test_float8_linear_forward_keeps_autocast_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The surrounding autocast must reach ``Float8Linear.forward`` unchanged.

    torchao's ``Float8Linear.forward`` reads ``torch.is_autocast_enabled()`` and
    casts its input to the autocast dtype before the fp8 matmul; quantization
    must NOT disable autocast around it (an earlier wrapper did, which left the
    input fp32 and crashed the fp8 matmul with an unsupported ``aten.to.dtype``
    on the Float8 tensor subclass).
    """
    monkeypatch.setattr(quantization, "_check_float8_available", lambda: (True, ""))

    observed: list[bool] = []

    def _spy_forward(self: Float8Linear, input: torch.Tensor) -> torch.Tensor:
        del self
        observed.append(torch.is_autocast_enabled("cpu"))
        return input

    monkeypatch.setattr(Float8Linear, "forward", _spy_forward)
    model = nn.Sequential(nn.Linear(16, 16, bias=False))
    quantized = Float8ModelQuantization.Config().make()(model)

    with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
        quantized(torch.randn(2, 16))

    assert observed == [True]


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
