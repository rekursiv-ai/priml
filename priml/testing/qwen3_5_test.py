"""Tests for the shared Qwen3.5 parity helpers."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial, wraps
from types import FunctionType
from typing import cast, override

import sys

from torch import Tensor, nn

import pytest
import torch

from priml.model.swiglu import SwiGLU
from priml.model.transformer.block import TransformerBlock
from priml.testing.qwen3_5 import (
    hf_config,
    hf_logits,
    hf_reference,
    hf_tensor,
    native_qwen35_config,
    torch_reference,
)


def test_hf_config_pins_cpu_tiny_architecture() -> None:
    config = hf_config()

    assert config == {
        "model_type": "qwen3_5_text",
        "vocab_size": 32,
        "hidden_size": 16,
        "intermediate_size": 24,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "linear_num_key_heads": 1,
        "linear_num_value_heads": 2,
        "linear_key_head_dim": 4,
        "linear_value_head_dim": 4,
        "linear_conv_kernel_dim": 4,
        "layer_types": ["linear_attention", "full_attention"],
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": 10_000.0,
            "partial_rotary_factor": 0.5,
            "mrope_section": [1, 1, 0],
        },
        "hidden_act": "silu",
        "rms_norm_eps": 1e-6,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "tie_word_embeddings": False,
    }


def test_native_config_has_split_swiglu_projection_for_every_block() -> None:
    config = native_qwen35_config()

    assert isinstance(config.block, list)
    assert len(config.block) == 2
    for block in config.block:
        assert isinstance(block, TransformerBlock.Config)
        assert isinstance(block.ffn, SwiGLU.Config)
        assert block.ffn.split_gate_projection is True


def test_hf_tensor_and_logits_extract_exact_tensors() -> None:
    tensor = torch.arange(6).reshape(2, 3)

    assert hf_tensor(tensor) is tensor

    class Output:
        logits = tensor

    assert hf_logits(Output()) is tensor


def test_hf_tensor_rejects_non_tensor() -> None:
    with pytest.raises(AssertionError):
        hf_tensor(object())


def test_hf_logits_rejects_missing_logits() -> None:
    with pytest.raises(AssertionError):
        hf_logits(object())


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [(None, torch.float32), (torch.float64, torch.float64)],
)
def test_hf_reference_casts_dtype_and_adapts_linear_attention(
    monkeypatch: pytest.MonkeyPatch,
    dtype: torch.dtype | None,
    expected: torch.dtype,
) -> None:
    class Reference(nn.Module):
        config = object()

        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(2))
            self.linear_attn = nn.Identity()

        @override
        def forward(self, value: Tensor) -> Tensor:
            return value * self.weight

    reference = Reference()
    adapted: list[nn.Module] = []
    monkeypatch.setattr(
        "priml.testing.qwen3_5.torch_reference",
        adapted.append,
    )

    result = hf_reference(reference, dtype=dtype)

    assert result is reference
    assert reference.weight.dtype == expected
    assert adapted == [reference.linear_attn]


def test_hf_reference_rejects_non_module() -> None:
    with pytest.raises(AssertionError):
        hf_reference(object())


type _Forward = Callable[..., Tensor]


def torch_chunk_gated_delta_rule(value: Tensor) -> Tensor:
    return value


def torch_recurrent_gated_delta_rule(value: Tensor) -> Tensor:
    return value


def causal_conv1d_fn(value: Tensor) -> Tensor:
    return value


def causal_conv1d_update(value: Tensor) -> Tensor:
    return value


def _reference_forward(self: nn.Module, value: Tensor) -> Tensor:
    del self
    return causal_conv1d_update(
        causal_conv1d_fn(
            torch_recurrent_gated_delta_rule(torch_chunk_gated_delta_rule(value)),
        ),
    )


class _Reference(nn.Module):
    @override
    @wraps(_reference_forward)
    def forward(self, value: Tensor) -> Tensor:
        return _reference_forward(self, value)


def _module_fallback(called: list[str]) -> nn.Module:
    def base(value: Tensor) -> Tensor:
        called.append("base")
        return value + 1

    @wraps(base)
    def func(value: Tensor) -> Tensor:
        called.append("wrapper")
        return value + 100

    class Kernel(nn.Module):
        @override
        def forward(self, value: Tensor) -> Tensor:
            return func(value)

    return Kernel()


class _DefaultsReference(nn.Module):
    bias: int = 5

    @override
    def forward(self, value: Tensor) -> Tensor:
        return value


def _reference_with_defaults(captured: int) -> nn.Module:
    def original(
        self: _DefaultsReference,
        value: Tensor,
        scale: int = 2,
        *,
        offset: int = 3,
    ) -> Tensor:
        return (
            causal_conv1d_update(
                causal_conv1d_fn(
                    torch_recurrent_gated_delta_rule(
                        torch_chunk_gated_delta_rule(value),
                    ),
                ),
            )
            + captured
            + self.bias
            + scale
            + offset
        )

    @wraps(original)
    def wrapped(
        self: _DefaultsReference,
        value: Tensor,
        scale: int = 2,
        *,
        offset: int = 3,
    ) -> Tensor:
        return original(self, value, scale, offset=offset)

    class Reference(_DefaultsReference):
        forward = wrapped

    return Reference()


def test_torch_reference_unwraps_module_kernel_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called: list[str] = []
    monkeypatch.setattr(
        sys.modules[__name__],
        "torch_chunk_gated_delta_rule",
        _module_fallback(called),
    )

    reference = torch_reference(_Reference())
    value = torch.zeros(2, 3)

    torch.testing.assert_close(reference(value), value + 1)
    assert called == ["base"]


def test_torch_reference_preserves_defaults_closure_and_receiver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called: list[str] = []
    monkeypatch.setattr(
        sys.modules[__name__],
        "torch_chunk_gated_delta_rule",
        _module_fallback(called),
    )

    reference = torch_reference(_reference_with_defaults(captured=7))
    value = torch.zeros(2, 3)

    torch.testing.assert_close(reference(value), value + 18)
    torch.testing.assert_close(reference(value, 4, offset=2), value + 19)
    assert called == ["base", "base"]


def _renamed(old: str, new: str) -> Callable[[_Forward], _Forward]:
    """Return a decorator reading ``old=`` as ``new=``, as HF's ``deprecate_kwarg`` does."""

    def decorate(func: _Forward) -> _Forward:
        @wraps(func)
        def wrapped(*args: object, **kwargs: object) -> Tensor:
            if old in kwargs:
                kwargs[new] = kwargs.pop(old)
            return func(*args, **kwargs)

        return wrapped

    return decorate


def _scaled_forward(self: nn.Module, value: Tensor, *, scale: int = 1) -> Tensor:
    del self
    return torch.mul(torch_chunk_gated_delta_rule(value), scale)


# An attribute of the function's own, as a decorator may set one.
_scaled_forward.__dict__["marker"] = "kept"

type _Rename = Callable[[str, str], Callable[[_Forward], _Forward]]

# The same decorator defined where no kernel is, as HF's ``deprecate_kwarg`` is.
_foreign_renamed = cast(_Rename, FunctionType(_renamed.__code__, {"wraps": wraps}))


class _RenamedReference(nn.Module):
    forward = _renamed("factor", "scale")(
        _foreign_renamed("size", "scale")(_scaled_forward),
    )


def test_torch_reference_keeps_the_forward_s_decorators(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HF renames ``cache_params`` to ``cache`` in a decorator: dropping it drops the cache.

    Each copy keeps its name and attributes, and reaches the copy it wraps.
    """
    called: list[str] = []
    monkeypatch.setattr(
        sys.modules[__name__],
        "torch_chunk_gated_delta_rule",
        _module_fallback(called),
    )
    reference = torch_reference(_RenamedReference())
    value = torch.zeros(2, 3)
    torch.testing.assert_close(reference(value, factor=3), (value + 1) * 3)
    assert called == ["base"]
    forward = reference.forward
    assert isinstance(forward, partial)
    outer = cast(FunctionType, forward.func)
    assert outer.__name__ == "_scaled_forward"
    assert cast(object, outer.__dict__["marker"]) == "kept"
    inner = cast(FunctionType, outer.__dict__["__wrapped__"])
    original = cast(FunctionType, _RenamedReference.forward)
    assert inner is not cast(object, original.__dict__["__wrapped__"])
    assert inner.__globals__ is not original.__globals__
    assert cast(object, inner.__dict__["marker"]) == "kept"


class _CallableForward:
    """A callable object: ``forward`` that is not a plain function."""

    def __call__(self, value: Tensor) -> Tensor:
        return value


def test_torch_reference_rejects_a_forward_that_is_not_a_function(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = _Reference()
    monkeypatch.setattr(reference, "forward", _CallableForward())
    with pytest.raises(
        TypeError,
        match=r"^Expected the reference forward to be a plain function\.$",
    ):
        torch_reference(reference)


def test_torch_reference_rejects_non_callable_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys.modules[__name__],
        "torch_chunk_gated_delta_rule",
        object(),
    )

    with pytest.raises(TypeError) as error:
        torch_reference(_Reference())

    assert str(error.value) == "Expected callable(kernel)."


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
