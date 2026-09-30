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
        torch.randn(2, length, 3, 4, dtype=dtype, requires_grad=True),
        torch.randn(2, length, 3, 4, dtype=dtype, requires_grad=True),
        torch.randn(2, length, 3, 5, dtype=dtype, requires_grad=True),
        (-torch.rand(2, length, 3)).requires_grad_(),
        torch.rand(2, length, 3, requires_grad=True),
    )
    state = torch.randn(2, 3, 4, 5, requires_grad=True) if cached else None
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
        torch.randn(2, 3, 4, 5),
        torch.randn(2, 3, 4, 5),
        torch.randn(2, 3, 4, 6),
        -torch.rand(2, 3, 4),
        torch.rand(2, 3, 4),
    )
    state = torch.randn(2, 4, 5, 6)
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
    batch, sequence, heads, key_width, value_width = 2, 3, 4, 5, 6
    query = torch.eye(sequence, key_width)[None, :, None, :].expand(
        batch,
        -1,
        heads,
        -1,
    )
    value = torch.ones(batch, sequence, heads, value_width)
    for kernel in (chunk_gated_delta_rule, recurrent_gated_delta_rule):
        output, state = kernel(
            query=query,
            key=query,
            value=value,
            g=torch.full((2, 3, 4), -1_000.0),
            beta=torch.ones(2, 3, 4),
        )
        assert state is None
        assert torch.isfinite(output).all()
        assert output.shape == value.shape


@pytest.mark.parametrize("chunk_size", [0, -1])
def test_chunk_rejects_nonpositive_chunk_size(chunk_size: int) -> None:
    query = torch.ones(2, 3, 4, 5)
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        chunk_gated_delta_rule(
            query=query,
            key=query,
            value=query,
            g=torch.zeros(2, 3, 4),
            beta=torch.ones(2, 3, 4),
            chunk_size=chunk_size,
        )


@pytest.mark.parametrize("initial_state", [False, True])
@pytest.mark.parametrize("chunk_size", [1, 3, 64])
@pytest.mark.parametrize("length", [1, 7, 63, 65, 129])
def test_chunk_matches_recurrent(
    length: int,
    chunk_size: int,
    initial_state: bool,
) -> None:
    """Chunk and recurrent scans compute the identical operation.

    Unlike the HF-parity tests above, this imports no optional dependency --
    the export build's only coverage of the chunked scan, a chunk boundary
    (``length`` need not divide ``chunk_size``), state carry, and non-trivial
    decay.
    """
    torch.manual_seed(length * 1_000 + chunk_size)
    batch, heads, key_width, value_width = 2, 3, 4, 5
    query = torch.randn(batch, length, heads, key_width)
    key = torch.randn(batch, length, heads, key_width)
    value = torch.randn(batch, length, heads, value_width)
    g = -torch.rand(batch, length, heads)
    beta = torch.rand(batch, length, heads)
    state = torch.randn(batch, heads, key_width, value_width) if initial_state else None

    chunk_out, chunk_state = chunk_gated_delta_rule(
        query=query,
        key=key,
        value=value,
        g=g,
        beta=beta,
        chunk_size=chunk_size,
        initial_state=state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    recurrent_out, recurrent_state = recurrent_gated_delta_rule(
        query=query,
        key=key,
        value=value,
        g=g,
        beta=beta,
        initial_state=state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )

    # Measured worst case over this parametrization: 4.8e-7 (output),
    # 7.2e-7 (state) -- roughly 3x that margin, still tight enough to bite.
    torch.testing.assert_close(chunk_out, recurrent_out, atol=2e-6, rtol=2e-6)
    assert chunk_state is not None
    assert recurrent_state is not None
    torch.testing.assert_close(chunk_state, recurrent_state, atol=2e-6, rtol=2e-6)


def test_delta_rule_matches_closed_form_with_orthonormal_keys() -> None:
    """No decay, full update, orthonormal keys collapse the scan to a sum.

    With ``beta=1`` and ``g=0``, each new key direction is orthogonal to
    every earlier one, so the delta correction recovers ``v_j`` exactly and
    the running state after ``t`` steps is ``sum_{j<=t} k_j (x) v_j`` --
    giving ``out_t = d**-0.5 * sum_{j<=t} (q_t . k_j) v_j``, a closed form
    independent of both kernels under test.
    """
    torch.manual_seed(0)
    batch, sequence, heads, key_width, value_width = 2, 5, 3, 6, 4
    key = torch.eye(sequence, key_width)[None, :, None, :].expand(batch, -1, heads, -1)
    query = torch.randn(batch, sequence, heads, key_width)
    value = torch.randn(batch, sequence, heads, value_width)
    g = torch.zeros(batch, sequence, heads)
    beta = torch.ones(batch, sequence, heads)

    causal = torch.tril(torch.ones(sequence, sequence))[:, None, :]
    weights = torch.einsum("btha,bsha->bths", query, key) * causal
    expected = torch.einsum("bths,bshv->bthv", weights, value) * key_width**-0.5

    chunk_actual, _ = chunk_gated_delta_rule(
        query=query,
        key=key,
        value=value,
        g=g,
        beta=beta,
        chunk_size=3,
    )
    recurrent_actual, _ = recurrent_gated_delta_rule(
        query=query,
        key=key,
        value=value,
        g=g,
        beta=beta,
    )

    # Measured: 6.0e-8 (chunk), 1.2e-7 (recurrent).
    torch.testing.assert_close(chunk_actual, expected, atol=5e-7, rtol=5e-7)
    torch.testing.assert_close(recurrent_actual, expected, atol=5e-7, rtol=5e-7)


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
