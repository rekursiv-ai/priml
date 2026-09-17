"""Tests for NanoChat causal attention and fused QK/RoPE."""

from __future__ import annotations

from collections.abc import Iterator
from importlib.metadata import version
from types import ModuleType
from typing import cast

import sys

from configgle import PartialConfig
from torch import Tensor, nn

import pytest
import torch

from priml.baselines.nanochat.attention import (
    CausalAttention,
    Flash4Attention,
    _qk_backward,
    _qk_backward_fake,
    _qk_backward_reference,
    _qk_fake,
    _qk_reference,
    fused_qk_norm_rope,
)
from priml.model.attention.kernel import SdpaNaive
from priml.model.cost import cost
from priml.model.linear import Linear
from priml.model.norm import RMSNorm
from priml.model.special import Identity
from priml.testing.cost import assert_cost_matches_torch

import priml.baselines.nanochat.attention


def test_flash_backends_belong_to_attention() -> None:
    for name in ("Flash3Attention", "Flash4Attention"):
        backend = cast(object, vars(priml.baselines.nanochat.attention)[name])
        assert isinstance(backend, type)
        assert backend.__module__ == priml.baselines.nanochat.attention.__name__


def test_head_gate_inherits_full_input_width() -> None:
    config = CausalAttention.Config()
    config.channels_out = 16
    config.channels_head = 8
    config.gate_channels = -1
    config.max_seq_len = 4
    config.window = 4
    config.head_gate = Linear.Config()
    attention = config.make()
    assert attention.head_gate is not None
    assert attention.head_gate.weight.shape == (2, 16)
    rotation = (torch.ones(4, 1, 4), torch.zeros(4, 1, 4))
    assert attention(torch.ones(1, 4, 16), cos_sin=rotation).shape == (1, 4, 16)


def test_causal_attention_reset_initializes_affine_output_norm() -> None:
    config = CausalAttention.Config()
    config.channels_in = 16
    config.channels_head = 8
    config.gate_channels = 4
    config.norm_out = RMSNorm.Config(elementwise_affine=True)
    attention = config.make()
    assert isinstance(attention.norm_out, RMSNorm)
    assert attention.norm_out.weight is not None
    with torch.no_grad():
        attention.norm_out.weight.fill_(float("nan"))

    attention.reset_parameters()

    assert torch.equal(attention.norm_out.weight, torch.ones(8))


@pytest.mark.parametrize("normalization", ["affine", "epsilon", "custom"])
def test_fused_attention_rejects_unsupported_normalization(normalization: str) -> None:
    config = CausalAttention.Config()
    config.channels_in = 16
    config.channels_head = 8
    config.gate_channels = 4
    config.fused_qk_rope = True
    norm = RMSNorm.Config()
    norm.eps = torch.finfo(torch.float32).eps
    if normalization == "affine":
        norm.elementwise_affine = True
    elif normalization == "epsilon":
        norm.eps = 0.01
    config.norm_qk = Identity.Config() if normalization == "custom" else norm

    with pytest.raises(ValueError, match=r"fused_qk_rope.*norm_qk"):
        config.make()


def test_causal_attention_forwards_unconsumed_messages() -> None:
    config = CausalAttention.Config()
    config.channels_in = 16
    config.channels_head = 8
    config.gate_channels = 4
    config.kernel = PartialConfig(_message_kernel)
    attention = config.make()
    x = torch.ones(1, 4, 16)
    output = attention(
        x,
        cos_sin=(torch.ones(4, 1, 4), torch.zeros(4, 1, 4)),
        window=2,
        message=123,
        bigram_value=None,
        trigram_value=None,
        fused_tables=None,
    )
    assert output.shape == x.shape


def _message_kernel(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    window: int,
    message: object,
) -> Tensor:
    assert q.shape == k.shape == v.shape
    assert window == 2
    assert message == 123
    return q


@pytest.mark.parametrize(
    ("gate_channels", "bigram", "trigram", "required_slices"),
    [
        (9, True, False, 2),
        (6, False, True, 3),
        (-1, True, False, 2),
        (-1, False, True, 3),
    ],
)
def test_memory_gate_slices_must_fit_the_residual_stream(
    gate_channels: int,
    bigram: bool,
    trigram: bool,
    required_slices: int,
) -> None:
    config = CausalAttention.Config()
    config.channels_in = 16
    config.channels_head = 8
    config.gate_channels = gate_channels
    config.bigram = bigram
    config.trigram = trigram

    with pytest.raises(ValueError, match=rf"{required_slices} gate slices.*16"):
        config.make()


def test_qk_forward_and_backward_match_fp32_math() -> None:
    torch.manual_seed(19)
    q = torch.randn(2, 3, 2, 8, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    phase = torch.randn(1, 3, 1, 4)
    cos, sin = phase.cos(), phase.sin()
    outputs = fused_qk_norm_rope(q, k, cos, sin)
    references: list[torch.Tensor] = []
    for value in (q, k):
        normalized = value * torch.rsqrt(
            value.square().mean(-1, keepdim=True) + torch.finfo(torch.float32).eps,
        )
        first, second = normalized.chunk(2, dim=-1)
        references.append(
            torch.cat((first * cos + second * sin, second * cos - first * sin), dim=-1),
        )
    gradients = [torch.randn_like(q), torch.randn_like(k)]
    for actual, expected in zip(outputs, references, strict=True):
        torch.testing.assert_close(actual, expected)
    expected_grads = torch.autograd.grad(references, (q, k), gradients)
    actual_grads = torch.autograd.grad(outputs, (q, k), gradients)
    for actual, expected in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("layout", [torch.contiguous_format, torch.channels_last])
def test_qk_backward_preserves_reference_bits_and_declared_layout(
    dtype: torch.dtype,
    layout: torch.memory_format,
) -> None:
    """Keep channels-last cotangents from violating the compiled stride contract."""
    q = torch.randn(2, 4, 2, 8, dtype=dtype).to(memory_format=layout)
    k = torch.randn_like(q, memory_format=torch.contiguous_format)
    phase = torch.randn(4, 1, 4, dtype=dtype)
    cos, sin = phase.cos(), phase.sin()
    gradient = torch.randn_like(q, memory_format=torch.channels_last)
    actual = _qk_backward(gradient, gradient, q, k, [cos, sin])
    declared = _qk_backward_fake(gradient, gradient, q, k, [cos, sin])
    for result, fake, value in zip(actual, declared, (q, k), strict=True):
        assert torch.equal(result, _qk_backward_reference(gradient, value, cos, sin))
        assert result.is_contiguous()
        assert result.stride() == fake.stride()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("layout", [torch.contiguous_format, torch.channels_last])
def test_qk_forward_preserves_reference_bits_and_declared_layout(
    dtype: torch.dtype,
    layout: torch.memory_format,
) -> None:
    """Expose the same contiguous output contract as the CUDA kernels."""
    q = torch.randn(2, 4, 2, 8, dtype=dtype).to(memory_format=layout)
    k = torch.randn_like(q, memory_format=torch.contiguous_format)
    phase = torch.randn(4, 1, 4, dtype=dtype)
    cos, sin = phase.cos(), phase.sin()
    actual = fused_qk_norm_rope(q, k, cos, sin)
    declared = _qk_fake(q, k, cos, sin)
    for result, fake, value in zip(actual, declared, (q, k), strict=True):
        assert torch.equal(result, _qk_reference(value, cos, sin))
        assert result.is_contiguous()
        assert result.stride() == fake.stride()


@pytest.mark.gpu_torch_cuda
@pytest.mark.gpu_triton
def test_cuda_qk_forward_and_backward_match_fp32_math() -> None:
    if not torch.cuda.is_available():
        pytest.skip("Requires a CUDA device.")
    torch.manual_seed(29)
    q = torch.randn(2, 17, 2, 8, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    phase = torch.randn(1, 17, 1, 4, device="cuda")
    cos, sin = phase.cos(), phase.sin()
    gradients = [torch.randn_like(q), torch.randn_like(k)]

    reference_inputs = [value.detach().requires_grad_() for value in (q, k)]
    references: list[torch.Tensor] = []
    for value in reference_inputs:
        fp32 = value.float()
        normalized = fp32 * torch.rsqrt(
            fp32.square().mean(-1, keepdim=True) + torch.finfo(torch.float32).eps,
        )
        first, second = normalized.chunk(2, dim=-1)
        references.append(
            torch.cat(
                (first * cos + second * sin, second * cos - first * sin),
                dim=-1,
            ).to(value.dtype),
        )
    expected_grads = torch.autograd.grad(references, reference_inputs, gradients)

    actual_inputs = [value.detach().requires_grad_() for value in (q, k)]
    outputs = fused_qk_norm_rope(actual_inputs[0], actual_inputs[1], cos, sin)
    actual_grads = torch.autograd.grad(outputs, actual_inputs, gradients)
    for actual, expected in zip(outputs, references, strict=True):
        torch.testing.assert_close(actual, expected)
    for actual, expected in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual, expected)


@pytest.fixture
def external_module(monkeypatch: pytest.MonkeyPatch) -> Iterator[_FakeInterface]:
    module = _FakeInterface()
    monkeypatch.setitem(sys.modules, "flash_attn.cute.interface", module)
    priml.baselines.nanochat.attention._make_flash4_ops.cache_clear()
    torch.compiler.reset()
    yield module
    torch.compiler.reset()
    priml.baselines.nanochat.attention._make_flash4_ops.cache_clear()


def test_optional_dependency_resolves_at_make(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "flash_attn.cute.interface", None)
    priml.baselines.nanochat.attention._make_flash4_ops.cache_clear()
    config = Flash4Attention.Config()
    assert "Flash4Attention" in config.pformat(hide_default_values=False)
    with pytest.raises(ModuleNotFoundError):
        config.make()


@pytest.mark.parametrize("window", [-1, 0, 1, 5, 16])
def test_layout_window_tuple_and_gradients(
    external_module: _FakeInterface,
    window: int,
) -> None:
    attention = Flash4Attention.Config().make()
    q, k, v = [torch.randn(2, 5, 3, 8, requires_grad=True) for _ in range(3)]
    output = attention(q, k, v, window=window)
    torch.testing.assert_close(output, q + 2 * k + 3 * v)
    output.sum().backward()
    for tensor, scale in zip((q, k, v), (1, 2, 3), strict=True):
        torch.testing.assert_close(tensor.grad, torch.full_like(tensor, scale))
    history = None if window < 0 or window >= q.shape[1] else window
    assert external_module.windows == [
        (None, None) if history is None else (history, 0),
        (history, 0),
    ]


@pytest.mark.gpu_flash_attention
@pytest.mark.gpu_torch_cuda
def test_cuda_matches_official_autograd() -> None:
    if not torch.cuda.is_available():
        pytest.skip("Requires an SM90 or SM100 CUDA device and flash-attn-4==4.0.0b29.")
    if torch.cuda.get_device_capability() not in ((9, 0), (10, 0)):
        pytest.skip("Requires an SM90 or SM100 CUDA device.")
    assert version("flash-attn-4") == "4.0.0b29"
    interface = cast(
        object,
        __import__("flash_attn.cute.interface", fromlist=["interface"]),
    )
    assert isinstance(
        interface,
        priml.baselines.nanochat.attention._Flash4Interface,
    )
    attention = Flash4Attention.Config().make()
    torch.manual_seed(42)
    tensors = [
        torch.randn(2, 129, 2, 128, device="cuda", dtype=torch.bfloat16)
        for _ in range(3)
    ]
    cotangent = torch.randn_like(tensors[0])
    reference_inputs = [tensor.detach().requires_grad_() for tensor in tensors]
    reference, _ = interface.flash_attn_func(
        *reference_inputs,
        causal=True,
        window_size=(64, 0),
        return_lse=True,
    )
    reference.backward(cotangent)
    inputs = [tensor.detach().requires_grad_() for tensor in tensors]
    output = attention(*inputs, window=64)
    output.backward(cotangent)
    torch.testing.assert_close(output, reference, rtol=0, atol=0)
    for actual, expected in zip(inputs, reference_inputs, strict=True):
        assert actual.grad is not None
        assert expected.grad is not None
        torch.testing.assert_close(actual.grad, expected.grad, rtol=0, atol=0)


class _FakeInterface(ModuleType):
    def __init__(self) -> None:
        super().__init__("flash_attn.cute.interface")
        self.windows: list[tuple[int | None, int | None]] = []

    def flash_attn_func(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        *,
        causal: bool,
        window_size: tuple[int | None, int | None],
        return_lse: bool,
    ) -> tuple[Tensor, Tensor]:
        assert causal
        assert return_lse
        self.windows.append(window_size)
        return q + 2 * k + 3 * v, torch.zeros(q.shape[0], q.shape[2], q.shape[1])

    def _flash_attn_bwd(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        *saved: Tensor,
        causal: bool,
        window_size_left: int | None,
        window_size_right: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        output, grad_output, lse = saved
        assert causal
        assert q.shape == k.shape == v.shape == output.shape == grad_output.shape
        assert lse.shape == (q.shape[0], q.shape[2], q.shape[1])
        self.windows.append((window_size_left, window_size_right))
        return grad_output.clone(), 2 * grad_output, 3 * grad_output


@pytest.mark.parametrize("feature", ["bigram", "trigram", "head_gate", "norm_out"])
def test_cost_prices_each_causal_attention_extension(feature: str) -> None:
    config = CausalAttention.Config()
    config.channels_in = 12
    config.channels_head = 4
    config.gate_channels = 4
    config.gated = False
    baseline = cost(config.copy_tree().finalize(), seq_len=4, rows=8, itemsize=2)
    if feature == "head_gate":
        config.head_gate = Linear.Config()
    elif feature == "norm_out":
        config.norm_out = RMSNorm.Config(elementwise_affine=True)
    else:
        setattr(config, feature, True)
    config = config.finalize()
    counted = cost(config, seq_len=4, rows=8, itemsize=2)
    assert counted.params == sum(p.numel() for p in config.make().parameters())
    if feature == "norm_out":
        extra = cost(config.norm_out, rows=24, itemsize=2).tile(3)
        assert counted == baseline + extra
    else:
        assert counted.params - baseline.params == 12
        assert counted.primal.flops.matmul - baseline.primal.flops.matmul == 24
        assert counted.adjoint.flops.matmul - baseline.adjoint.flops.matmul == 48
        assert counted.primal.bytes.matmul - baseline.primal.bytes.matmul == 2 * (
            4 + 3 + 12 / 8
        )
        assert counted.adjoint.flops.reduction - baseline.adjoint.flops.reduction == 9
        assert counted.adjoint.bytes.reduction - baseline.adjoint.bytes.reduction == 30
        assert counted.primal.flops.elementwise - baseline.primal.flops.elementwise == (
            15 + (12 if feature == "head_gate" else 24)
        )


def test_cost_extension_dtype_and_fusion_preserve_the_analytical_algorithm() -> None:
    config = CausalAttention.Config()
    config.channels_in = 12
    config.channels_head = 4
    config.gate_channels = 4
    config.bigram = config.trigram = True
    config.head_gate = Linear.Config()
    config.norm_out = RMSNorm.Config(elementwise_affine=True)
    config = config.finalize()
    narrow = cost(config, seq_len=4, rows=8, itemsize=2)
    wide = cost(config, seq_len=4, rows=8, itemsize=4)
    assert wide.training.bytes == 2 * narrow.training.bytes
    assert wide.training.flops == narrow.training.flops
    config.fused_qk_rope = True
    assert cost(config, seq_len=4, rows=8, itemsize=2) == narrow


def test_cost_extension_matmuls_match_executed_forward_and_backward() -> None:
    config = CausalAttention.Config()
    config.kernel = SdpaNaive.Config()
    config.channels_in = 12
    config.channels_head = 4
    config.gate_channels = 4
    config.bigram = config.trigram = True
    config.head_gate = Linear.Config()
    config.norm_out = RMSNorm.Config(elementwise_affine=True)
    assert_cost_matches_torch(
        config,
        build_input=lambda: torch.randn(2, 4, 12, requires_grad=True),
        num_tokens=8,
        bus={"seq_len": 4},
        run=_run_all_attention_gates,
    )


def _run_all_attention_gates(module: nn.Module, x: Tensor) -> Tensor:
    return cast(CausalAttention, module)(
        x,
        cos_sin=(torch.ones(4, 1, 2), torch.zeros(4, 1, 2)),
        value_embedding=torch.ones_like(x),
        bigram_value=torch.ones_like(x),
        trigram_value=torch.ones_like(x),
    )


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
