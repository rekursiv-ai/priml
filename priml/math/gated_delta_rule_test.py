"""Exact parity against the Transformers 5.17 PyTorch delta-rule kernels."""

from collections.abc import Callable
from typing import TYPE_CHECKING, cast

import inspect

from torch import Tensor, nn

import pytest
import torch

from priml.math.gated_delta_rule import (
    chunk_gated_delta_rule,
    recurrent_gated_delta_rule,
)


if TYPE_CHECKING:
    from transformers.models.qwen3_5 import modeling_qwen3_5
else:
    from wrapt import lazy_import

    # Avoids the measured 3.3--3.5s fresh-process Qwen reference import.
    modeling_qwen3_5 = lazy_import("transformers.models.qwen3_5.modeling_qwen3_5")


def _reference(name: str) -> Callable[..., tuple[Tensor, Tensor | None]]:
    pytest.importorskip("transformers")
    kernel = cast(object, getattr(modeling_qwen3_5, name))
    if isinstance(kernel, nn.Module):
        kernel = cast(
            object,
            inspect.getclosurevars(kernel.forward).nonlocals["func"],
        )
    assert callable(kernel)
    return cast(Callable[..., tuple[Tensor, Tensor | None]], inspect.unwrap(kernel))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("length", [1, 7, 65])
@pytest.mark.parametrize("cached", [False, True])
def test_chunk_reference_forward_backward(
    dtype: torch.dtype,
    length: int,
    cached: bool,
) -> None:
    inputs = (
        torch.randn(1, length, 2, 4, dtype=dtype, requires_grad=True),
        torch.randn(1, length, 2, 4, dtype=dtype, requires_grad=True),
        torch.randn(1, length, 2, 3, dtype=dtype, requires_grad=True),
        (-torch.rand(1, length, 2)).requires_grad_(),
        torch.rand(1, length, 2, requires_grad=True),
    )
    state = torch.randn(1, 2, 4, 3, requires_grad=True) if cached else None
    query, key, value, g, beta = inputs
    expected, expected_state = _reference("torch_chunk_gated_delta_rule")(
        query=query,
        key=key,
        value=value,
        g=g,
        beta=beta,
        initial_state=state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    actual, actual_state = chunk_gated_delta_rule(
        query=query,
        key=key,
        value=value,
        g=g,
        beta=beta,
        initial_state=state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    assert torch.equal(actual, expected)
    assert actual_state is not None
    assert expected_state is not None
    assert torch.equal(actual_state, expected_state)
    assert actual_state.dtype == torch.float32
    differentiable = inputs if state is None else (*inputs, state)
    expected_grad = torch.autograd.grad(
        expected.float().square().sum() + expected_state.square().sum(),
        differentiable,
        retain_graph=True,
    )
    actual_grad = torch.autograd.grad(
        actual.float().square().sum() + actual_state.square().sum(),
        differentiable,
    )
    for actual_value, expected_value in zip(actual_grad, expected_grad, strict=True):
        assert torch.equal(actual_value, expected_value)


@pytest.mark.parametrize("normalize", [False, True])
def test_recurrent_reference_and_state_ownership(normalize: bool) -> None:
    inputs = (
        torch.randn(1, 3, 2, 4),
        torch.randn(1, 3, 2, 4),
        torch.randn(1, 3, 2, 3),
        -torch.rand(1, 3, 2),
        torch.rand(1, 3, 2),
    )
    state = torch.randn(1, 2, 4, 3)
    original = state.clone()
    query, key, value, g, beta = inputs
    expected, expected_state = _reference("torch_recurrent_gated_delta_rule")(
        query=query,
        key=key,
        value=value,
        g=g,
        beta=beta,
        initial_state=state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=normalize,
    )
    actual, actual_state = recurrent_gated_delta_rule(
        query=query,
        key=key,
        value=value,
        g=g,
        beta=beta,
        initial_state=state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=normalize,
    )
    assert torch.equal(actual, expected)
    assert actual_state is not None
    assert expected_state is not None
    assert torch.equal(actual_state, expected_state)
    assert torch.equal(state, original)


def test_closed_decay_tail_and_optional_state() -> None:
    query = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]])
    value = torch.tensor([[[[2.0]], [[3.0]]]])
    for kernel in (chunk_gated_delta_rule, recurrent_gated_delta_rule):
        output, state = kernel(
            query=query,
            key=query,
            value=value,
            g=torch.full((1, 2, 1), -1_000.0),
            beta=torch.ones(1, 2, 1),
        )
        assert state is None
        assert torch.isfinite(output).all()
        torch.testing.assert_close(output, value * 2**-0.5, rtol=0, atol=0)


@pytest.mark.parametrize("chunk_size", [0, -1])
def test_chunk_rejects_nonpositive_chunk_size(chunk_size: int) -> None:
    query = torch.ones(1, 1, 1, 1)
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        chunk_gated_delta_rule(
            query=query,
            key=query,
            value=query,
            g=torch.zeros(1, 1, 1),
            beta=torch.ones(1, 1, 1),
            chunk_size=chunk_size,
        )


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
