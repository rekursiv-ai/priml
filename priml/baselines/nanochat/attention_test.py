"""Tests for NanoChat causal attention and fused QK/RoPE."""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import cast, override

from configgle import PartialConfig
from torch import Tensor, nn

import pytest
import torch

from priml.baselines.nanochat import attention
from priml.baselines.nanochat.attention import (
    CausalAttention,
    _qk_backward,
    _qk_backward_fake,
    _qk_backward_reference,
    _qk_fake,
    _qk_reference,
    fused_qk_norm_rope,
)
from priml.cost import Cost, cost
from priml.model.attention.kernel import SdpaNaive
from priml.model.linear import Linear
from priml.model.norm import RMSNorm
from priml.model.special import Identity
from priml.testing.cost import assert_cost_matches_torch


def test_memory_gates_start_at_zero_without_bias() -> None:
    config = CausalAttention.Config()
    config.channels_in = 12
    config.channels_head = 4
    config.num_heads = 3
    config.gate_channels = 4
    config.bigram = True
    config.trigram = True

    attention = config.make()

    for gate in (attention.bigram_gate, attention.trigram_gate):
        assert gate is not None
        assert torch.equal(gate.weight, torch.zeros_like(gate.weight))
        assert gate.bias is None


def test_head_gate_inherits_full_input_width() -> None:
    config = CausalAttention.Config()
    config.channels_out = 18
    config.channels_head = 6
    config.gate_channels = -1
    config.max_seq_len = 4
    config.window = 4
    config.head_gate = Linear.Config()
    attention = config.make()
    assert attention.head_gate is not None
    assert attention.head_gate.weight.shape == (3, 18)
    # CausalAttention broadcasts rotary values across the head axis.
    rotation = (torch.ones(4, 1, 3), torch.zeros(4, 1, 3))
    assert attention(torch.ones(2, 4, 18), cos_sin=rotation).shape == (2, 4, 18)


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


@pytest.mark.parametrize("normalization", ["affine", "epsilon", "custom", "subclass"])
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
    elif normalization == "subclass":

        class SpecializedRMSNormConfig(RMSNorm.Config):
            pass

        norm = SpecializedRMSNormConfig()
    config.norm_qk = Identity.Config() if normalization == "custom" else norm

    with pytest.raises(
        ValueError,
        match=(
            r"^fused_qk_rope requires norm_qk to be parameter-free RMSNorm "
            r"with epsilon None or float32 epsilon\.$"
        ),
    ):
        config.make()


def _message_kernel(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    window: int,
    message: object,
) -> Tensor:
    del k, v, window, message
    return q


def test_causal_attention_forwards_unconsumed_messages() -> None:
    config = CausalAttention.Config()
    config.channels_in = 18
    config.channels_head = 6
    config.gate_channels = 6
    config.kernel = PartialConfig(_message_kernel)
    attention = config.make()
    x = torch.ones(2, 4, 18)
    output = attention(
        x,
        # CausalAttention broadcasts rotary values across the head axis.
        cos_sin=(torch.ones(4, 1, 3), torch.zeros(4, 1, 3)),
        window=2,
        message=123,
        bigram_value=None,
        trigram_value=None,
        fused_tables=None,
    )
    assert output.shape == x.shape


def test_causal_attention_keeps_the_block_cache_from_its_kernel() -> None:
    """A block hands every attention ``cache``; a bus-naming kernel never sees it."""
    config = CausalAttention.Config()
    config.channels_in = 18
    config.channels_head = 6
    config.gate_channels = 6
    config.kernel = PartialConfig(_message_kernel)
    attention = config.make()
    x = torch.ones(2, 4, 18)
    # CausalAttention broadcasts rotary values across the head axis.
    cos_sin = (torch.ones(4, 1, 3), torch.zeros(4, 1, 3))

    output = attention(x, cos_sin=cos_sin, message=123, cache=None)

    assert output.shape == x.shape
    with pytest.raises(TypeError, match="CausalAttention keeps no decode cache"):
        attention(x, cos_sin=cos_sin, message=123, cache={})


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
    q = torch.randn(3, 4, 2, 8, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    # fused_qk_norm_rope broadcasts phase across batch and head axes.
    phase = torch.randn(
        1,
        4,
        1,
        4,
    )  # fused_qk_norm_rope broadcasts batch and head axes.
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


def test_qk_reference_uses_positive_epsilon_for_zero_inputs() -> None:
    q = torch.zeros(2, 3, 4, 8)
    cos, sin = torch.ones(3, 4), torch.zeros(3, 4)

    q_out, k_out = fused_qk_norm_rope(q, q, cos, sin)

    assert torch.equal(q_out, q)
    assert torch.equal(k_out, q)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("layout", [torch.contiguous_format, torch.channels_last])
def test_qk_backward_preserves_reference_bits_and_declared_layout(
    dtype: torch.dtype,
    layout: torch.memory_format,
) -> None:
    """Keep channels-last cotangents from violating the compiled stride contract."""
    q = torch.randn(2, 3, 4, 8, dtype=dtype).to(memory_format=layout)
    k = torch.randn_like(q, memory_format=torch.contiguous_format)
    # fused_qk_norm_rope broadcasts phase across the head axis.
    phase = torch.randn(3, 1, 4, dtype=dtype)
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
    q = torch.randn(2, 3, 4, 8, dtype=dtype).to(memory_format=layout)
    k = torch.randn_like(q, memory_format=torch.contiguous_format)
    # fused_qk_norm_rope broadcasts phase across the head axis.
    phase = torch.randn(3, 1, 4, dtype=dtype)
    cos, sin = phase.cos(), phase.sin()
    actual = fused_qk_norm_rope(q, k, cos, sin)
    declared = _qk_fake(q, k, cos, sin)
    for result, fake, value in zip(actual, declared, (q, k), strict=True):
        assert torch.equal(result, _qk_reference(value, cos, sin))
        assert result.dtype == value.dtype
        assert result.is_contiguous()
        assert result.stride() == fake.stride()


@pytest.mark.gpu_torch_cuda
@pytest.mark.gpu_triton
def test_cuda_qk_forward_and_backward_match_fp32_math() -> None:
    if not torch.cuda.is_available():
        pytest.skip("Requires a CUDA device.")
    torch.manual_seed(29)
    q = torch.randn(3, 17, 2, 8, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    # fused_qk_norm_rope broadcasts phase across batch and head axes.
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


@pytest.mark.parametrize("feature", ["bigram", "trigram", "head_gate", "norm_out"])
def test_cost_prices_each_causal_attention_extension(feature: str) -> None:
    config = CausalAttention.Config()
    config.channels_in = 12
    config.channels_head = 4
    config.gate_channels = 4
    config.gated = False
    baseline = cost(
        config.copy_tree().finalize(),
        seq_len=8,
        batch_size=1,
        dtype=torch.bfloat16,
    )
    if feature == "head_gate":
        config.head_gate = Linear.Config()
    elif feature == "norm_out":
        config.norm_out = RMSNorm.Config(elementwise_affine=True)
    else:
        setattr(config, feature, True)
    config = config.finalize()
    counted = cost(config, seq_len=8, batch_size=1, dtype=torch.bfloat16)
    assert counted.params == sum(p.numel() for p in config.make().parameters())
    if feature == "norm_out":
        extra = cost(
            config.norm_out,
            seq_len=8,
            batch_size=config.num_heads,
            dtype=torch.bfloat16,
        )
        assert counted == baseline + extra
    else:
        rows = 8
        heads = config.num_heads
        inner = heads * config.channels_head
        itemsize = torch.bfloat16.itemsize
        assert counted.params - baseline.params == 12
        assert (
            counted["flops", "primal", "matmul"].sum()
            - baseline["flops", "primal", "matmul"].sum()
            == 2 * rows * config.gate_channels * heads
        )
        assert (
            counted["flops", "adjoint", "matmul"].sum()
            - baseline["flops", "adjoint", "matmul"].sum()
            == 4 * rows * config.gate_channels * heads
        )
        assert counted["bytes", "primal", "matmul"].sum() - baseline[
            "bytes",
            "primal",
            "matmul",
        ].sum() == itemsize * (rows * config.gate_channels + rows * heads + 12)
        assert counted["flops", "adjoint", "reduction"].sum() - baseline[
            "flops",
            "adjoint",
            "reduction",
        ].sum() == rows * (inner - heads)
        assert counted["bytes", "adjoint", "reduction"].sum() - baseline[
            "bytes",
            "adjoint",
            "reduction",
        ].sum() == itemsize * (rows * inner + rows * heads)
        assert counted["flops", "primal", "elementwise"].sum() - baseline[
            "flops",
            "primal",
            "elementwise",
        ].sum() == rows * (5 * heads + (12 if feature == "head_gate" else 24))


@pytest.mark.parametrize(
    ("add", "primal_elements", "primal_flops"),
    [(False, 165, 105), (True, 225, 135)],
)
def test_value_mix_cost_matches_each_elementwise_and_reduction_cell(
    add: bool,
    primal_elements: int,
    primal_flops: int,
) -> None:
    dtype = torch.float16
    rows, heads, channels_head, channels_in = 5, 3, 2, 11
    result = attention._value_mix_cost(
        heads=heads,
        channels_head=channels_head,
        channels_in=channels_in,
        rows=rows,
        dtype=dtype,
        add=add,
    )
    expected = Cost(
        cells={
            ("bytes", "primal", "elementwise", dtype): 2 * primal_elements,
            ("flops", "primal", "elementwise", dtype): primal_flops,
            ("bytes", "adjoint", "elementwise", dtype): 810,
            ("flops", "adjoint", "elementwise", dtype): 190,
            ("bytes", "adjoint", "reduction", dtype): 90,
            ("flops", "adjoint", "reduction", dtype): 15,
        },
    )

    assert result == expected


def test_cost_extension_dtype_and_fusion_preserve_the_analytical_algorithm() -> None:
    config = CausalAttention.Config()
    config.channels_in = 12
    config.channels_head = 4
    config.gate_channels = 4
    config.bigram = config.trigram = True
    config.head_gate = Linear.Config()
    config.norm_out = RMSNorm.Config(elementwise_affine=True)
    config = config.finalize()
    narrow = cost(config, seq_len=8, batch_size=1, dtype=torch.bfloat16)
    wide = cost(config, seq_len=8, batch_size=1, dtype=None)
    assert (
        wide["bytes", torch.float32].sum() == 2 * narrow["bytes", torch.bfloat16].sum()
    )
    assert wide["bytes", torch.int64] == narrow["bytes", torch.int64]
    assert wide["flops"].sum() == narrow["flops"].sum()
    config.fused_qk_rope = True
    assert cost(config, seq_len=8, batch_size=1, dtype=torch.bfloat16) == narrow


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
        build_input=lambda: torch.randn(3, 4, 12, requires_grad=True),
        seq_len=4,
        batch_size=3,
        dtype=None,
        run=_run_all_attention_gates,
    )


def _run_all_attention_gates(module: nn.Module, x: Tensor) -> Tensor:
    return cast(CausalAttention, module)(
        x,
        # CausalAttention broadcasts rotary values across the head axis.
        cos_sin=(torch.ones(4, 1, 2), torch.zeros(4, 1, 2)),
        value_embedding=torch.ones_like(x),
        bigram_value=torch.ones_like(x),
        trigram_value=torch.ones_like(x),
    )


def test_causal_attention_rejects_non_tensor_memories_and_supports_fused_rope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = attention
    original_fused = module.fused_qk_norm_rope
    fused_calls = 0

    def track_fused(
        q: Tensor,
        k: Tensor,
        cos: Tensor,
        sin: Tensor,
    ) -> tuple[Tensor, Tensor]:
        nonlocal fused_calls
        fused_calls += 1
        return original_fused(q, k, cos, sin)

    monkeypatch.setattr(module, "fused_qk_norm_rope", track_fused)
    config = CausalAttention.Config()
    config.channels_in = 8
    config.channels_head = 4
    config.num_heads = 2
    config.gate_channels = 8
    config.kernel = SdpaNaive.Config()
    config.fused_qk_rope = True
    config.norm_qk = RMSNorm.Config(elementwise_affine=False, eps=None)
    causal = config.make()
    x = torch.randn(2, 3, 8)
    # Rotary factors use a singleton head axis for production broadcasting.
    cos_sin = (torch.ones(3, 1, 2), torch.zeros(3, 1, 2))
    with pytest.raises(TypeError, match=r"^bigram_value must be Tensor or None"):
        causal(x, cos_sin=cos_sin, bigram_value=3)
    with pytest.raises(TypeError, match=r"^trigram_value must be Tensor or None"):
        causal(x, cos_sin=cos_sin, trigram_value=3)
    with pytest.raises(TypeError, match=r"^fused_tables must be list or None"):
        causal(x, cos_sin=cos_sin, fused_tables=(1,))
    output = causal(x, cos_sin=cos_sin)
    assert output.shape == x.shape
    assert fused_calls == 1
    with pytest.raises(TypeError, match=r"^fused_tables must hold NgramSource"):
        causal(x, cos_sin=cos_sin, fused_tables=[3])

    gated = CausalAttention.Config()
    gated.channels_in = 16
    gated.channels_head = 4
    gated.num_heads = 4
    gated.gate_channels = 8
    gated.bigram = True
    gated.kernel = SdpaNaive.Config()
    built = gated.make()
    built.bigram_gate = None
    with pytest.raises(ValueError, match="gate is not None"):
        built(
            torch.randn(2, 3, 16),
            cos_sin=cos_sin,
            bigram_value=torch.randn(2, 3, 16),
        )


def test_qk_validation_rejects_bad_shapes_and_widths() -> None:
    with pytest.raises(ValueError, match="ndim"):
        fused_qk_norm_rope(
            torch.zeros(2, 3, 4),
            torch.zeros(2, 3, 4),
            torch.ones(3, 2),
            torch.zeros(3, 2),
        )
    # Zero width is the explicit degenerate input rejected by this validator.
    with pytest.raises(ValueError, match="half"):
        fused_qk_norm_rope(
            torch.zeros(2, 3, 2, 0),
            torch.zeros(2, 3, 2, 0),
            torch.ones(3, 2),
            torch.zeros(3, 2),
        )
    q = torch.zeros(2, 3, 4, 6)
    with pytest.raises(ValueError, match="shape"):
        fused_qk_norm_rope(q, q[..., :3], torch.ones(3, 2), torch.zeros(3, 2))
    for width, message in ((5, "% 2"), (6, "half")):
        bad = torch.zeros(2, 3, 4, width)
        with pytest.raises(ValueError, match=message):
            fused_qk_norm_rope(bad, bad, torch.ones(3, 2), torch.zeros(3, 2))


def test_qk_cuda_adapters_launch_with_expected_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = attention

    class Launcher:
        def __init__(self) -> None:
            self.arguments: dict[str, object] = {}

        def __getitem__(
            self,
            grid: Callable[[dict[str, int]], tuple[int, ...]],
        ) -> Callable[..., None]:
            def launch(**arguments: object) -> None:
                assert grid({"block": 4}) == (16,)
                self.arguments = arguments

            return launch

    forward, backward = Launcher(), Launcher()

    def cdiv(n: int, b: int) -> int:
        return (n + b - 1) // b

    monkeypatch.setattr(module, "triton", SimpleNamespace(cdiv=cdiv))
    monkeypatch.setattr(module, "_compiled_qk_forward", lambda: forward)
    monkeypatch.setattr(module, "_compiled_qk_backward", lambda: backward)
    q = torch.randn(2, 4, 8, 32)[..., ::2]
    k = torch.randn(2, 4, 8, 16)
    dq, dk = torch.randn_like(q), torch.randn_like(k)
    # _qk_forward_cuda consumes phase with a singleton head axis.
    cos, sin = torch.randn(4, 1, 8), torch.randn(4, 1, 8)

    q_out, k_out = module._qk_forward_cuda(q, k, cos, sin)
    assert q_out.shape == q.shape
    assert q_out.is_contiguous()
    assert k_out.shape == k.shape
    assert k_out.is_contiguous()
    assert forward.arguments["n_rows"] == 64
    assert forward.arguments["geometry"] == (8, 4)
    forward_constants = cast(tuple[float, int, bool], forward.arguments["constants"])
    assert forward_constants == (1.1920928955078125e-07, 8, True)
    assert type(forward_constants[1]) is int
    forward_buffers = cast(tuple[Tensor, ...], forward.arguments["buffers"])
    assert forward_buffers[0].is_contiguous()
    assert forward_buffers[1].is_contiguous()

    q_grad, k_grad = module._qk_backward_cuda(dq, dk, q, k, cos=cos, sin=sin)
    assert q_grad.shape == q.shape
    assert q_grad.is_contiguous()
    assert k_grad.shape == k.shape
    assert k_grad.is_contiguous()
    assert backward.arguments["n_rows"] == 64
    assert backward.arguments["geometry"] == (8, 4)
    backward_constants = cast(
        tuple[float, int, bool],
        backward.arguments["constants"],
    )
    assert backward_constants == (1.1920928955078125e-07, 8, True)
    assert type(backward_constants[1]) is int
    backward_buffers = cast(tuple[Tensor, ...], backward.arguments["buffers"])
    assert all(value.is_contiguous() for value in backward_buffers)


def test_qk_triton_kernels_load_masked_rows_and_store_all_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = attention

    class Symbol:
        def __init__(self, expression: str) -> None:
            self.expression = expression

        @override
        def __repr__(self) -> str:
            return self.expression

        def __getitem__(self, key: object) -> Symbol:
            return Symbol(f"{self.expression}[{key!r}]")

        def to(self, dtype: object) -> Symbol:
            return Symbol(f"{self.expression}.to({dtype!r})")

        def __add__(self, other: object) -> Symbol:
            return Symbol(f"({self.expression}+{other!r})")

        def __radd__(self, other: object) -> Symbol:
            return Symbol(f"({other!r}+{self.expression})")

        def __sub__(self, other: object) -> Symbol:
            return Symbol(f"({self.expression}-{other!r})")

        def __mul__(self, other: object) -> Symbol:
            return Symbol(f"({self.expression}*{other!r})")

        def __rmul__(self, other: object) -> Symbol:
            return Symbol(f"({other!r}*{self.expression})")

        def __truediv__(self, other: object) -> Symbol:
            return Symbol(f"({self.expression}/{other!r})")

        def __floordiv__(self, other: object) -> Symbol:
            return Symbol(f"({self.expression}//{other!r})")

        def __mod__(self, other: object) -> Symbol:
            return Symbol(f"({self.expression}%{other!r})")

        def __lt__(self, other: object) -> Symbol:
            return Symbol(f"({self.expression}<{other!r})")

    loads: list[tuple[Tensor, dict[str, object]]] = []
    stores: list[tuple[Tensor, Symbol, dict[str, object]]] = []

    def load(pointer: Tensor, **kwargs: object) -> Symbol:
        loads.append((pointer, kwargs))
        return Symbol(f"load{len(loads)}")

    def store(pointer: Tensor, value: Symbol, **kwargs: object) -> None:
        stores.append((pointer, value, kwargs))

    def program_id(axis: int) -> Symbol:
        del axis
        return Symbol("pid")

    def arange(start: int, stop: int) -> Symbol:
        return Symbol(f"arange({start},{stop})")

    def sum_symbols(value: Symbol, axis: int) -> Symbol:
        return Symbol(f"sum({value!r},{axis})")

    language = SimpleNamespace(
        float32=object(),
        program_id=program_id,
        arange=arange,
        load=load,
        store=store,
        sum=sum_symbols,
    )
    monkeypatch.setattr(module, "language", language)

    def rsqrt(value: Symbol) -> Symbol:
        return Symbol(f"rsqrt({value!r})")

    monkeypatch.setattr(module, "libdevice", SimpleNamespace(rsqrt=rsqrt))
    pointers = tuple(torch.full((1,), float(index)) for index in range(12))

    forward_kernel = cast(
        Callable[..., object],
        module.__dict__["_qk_norm_rope_fwd_triton"],
    )
    forward_kernel(
        buffers=pointers[:6],
        n_rows=5,
        geometry=(2, 4),
        constants=(1.1920928955078125e-07, 4, False),
        block=4,
    )
    assert len(loads) == 6
    assert all("mask" in kwargs and kwargs["other"] == 0.0 for _, kwargs in loads)
    assert len(stores) == 4
    assert all("mask" in kwargs for _, _, kwargs in stores)
    assert "tensor([0.])" in repr(loads[0][0])
    assert "tensor([1.])" in repr(loads[2][0])
    assert "tensor([2.])" in repr(loads[4][0])

    loads.clear()
    stores.clear()
    backward_kernel = cast(
        Callable[..., object],
        module.__dict__["_qk_norm_rope_bwd_triton"],
    )
    backward_kernel(
        buffers=pointers[:8],
        n_rows=5,
        geometry=(2, 4),
        constants=(1.1920928955078125e-07, 4, False),
        block=4,
    )
    assert len(loads) == 10
    assert all("mask" in kwargs and kwargs["other"] == 0.0 for _, kwargs in loads)
    assert len(stores) == 4
    assert all("mask" in kwargs for _, _, kwargs in stores)
    assert "tensor([2.])" in repr(loads[0][0])
    assert "tensor([0.])" in repr(loads[4][0])
    assert "tensor([4.])" in repr(loads[8][0])


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
