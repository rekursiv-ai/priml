"""Tests for the FlashAttention 4 kernels, on a stand-in FA4 and on the GPU."""

from __future__ import annotations

from dataclasses import field
from types import ModuleType
from typing import TYPE_CHECKING, Final, override

import functools
import math
import sys

from configgle import Fig
from torch import Tensor, nn

import pytest
import torch

from priml.cost import Cost, cost
from priml.model.attention import flash4
from priml.model.attention.flash4 import (
    Flash4Attention,
    Flash4UnavailableError,
    Flash4Varlen,
    _flash4_backward,
    _flash4_backward_fake,
    _flash4_forward_fake,
)
from priml.model.attention.kernel import SdpaFused, SdpaNaive, SdpaVarlen
from priml.model.attention.window import segment_mask
from priml.testing.cost import assert_cost_matches_torch


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


CU_SEQLENS: Final = (0, 3, 7, 8, 8, 8, 12, 16)
"""Two rows of 8: segments of 3, 4 and 1, two empty ones, then 4 and 4."""

FA4_WARNINGS: Final = pytest.mark.filterwarnings(
    "ignore::DeprecationWarning:flash_attn",
    "ignore::DeprecationWarning:cutlass",
    "ignore::DeprecationWarning:quack",
    "ignore::UserWarning:cutlass",
)
"""What a real FA4 run warns from its own and CUTLASS's code, not this repo's.

flash-attn-4 (4.0.0b32 when this was written) imports names CUTLASS deprecates
(``arch.Arch``, ``warpgroup.OperandMajorMode``), and its kernels' JIT hands
CUTLASS an argument it cannot convert; under the suite's warnings-as-errors each
would fail the test. Scoped by module, a warning from priml still fails it."""


@pytest.fixture
def fake_flash4(monkeypatch: pytest.MonkeyPatch) -> Iterator[_FakeInterface]:
    """Serve ``_FakeInterface`` as FA4, with a fresh import and Dynamo cache.

    The device reads as SM90, one FA4 runs on, whatever the host has.
    """
    module = _FakeInterface()
    monkeypatch.setitem(sys.modules, "flash_attn.cute.interface", module)
    monkeypatch.setattr(torch.cuda, "get_device_capability", _sm90)
    flash4._interface.cache_clear()
    torch.compiler.reset()
    yield module
    torch.compiler.reset()
    flash4._interface.cache_clear()


@pytest.mark.parametrize("config", [Flash4Attention.Config(), Flash4Varlen.Config()])
def test_fa4_is_imported_when_a_kernel_is_built(
    monkeypatch: pytest.MonkeyPatch,
    config: Flash4Attention.Config | Flash4Varlen.Config,
) -> None:
    monkeypatch.setitem(sys.modules, "flash_attn.cute.interface", None)
    flash4._interface.cache_clear()
    assert type(config).__qualname__.split(".")[0] in config.pformat()
    with pytest.raises(ModuleNotFoundError):
        config.make()
    flash4._interface.cache_clear()


def test_an_fa4_without_the_entry_points_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        sys.modules,
        "flash_attn.cute.interface",
        ModuleType("flash_attn.cute.interface"),
    )
    flash4._interface.cache_clear()
    with pytest.raises(TypeError, match="FA4 must provide"):
        Flash4Attention.Config().make()
    flash4._interface.cache_clear()


@pytest.mark.usefixtures("fake_flash4")
@pytest.mark.parametrize("config", [Flash4Attention.Config(), Flash4Varlen.Config()])
def test_the_kernels_require_sm90_or_sm100(
    monkeypatch: pytest.MonkeyPatch,
    config: Flash4Attention.Config | Flash4Varlen.Config,
) -> None:
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (12, 0))
    with pytest.raises(Flash4UnavailableError) as error:
        config.make()
    assert str(error.value) == (
        "FlashAttention 4 requires SM90 or SM100; this device is SM120. "
        "Inject a portable kernel such as SdpaFused or SdpaVarlen instead."
    )


@pytest.mark.parametrize("window", [-1, 0, 1, 5, 16])
def test_dense_windows_and_gradients_reach_fa4(
    fake_flash4: _FakeInterface,
    window: int,
) -> None:
    attention = Flash4Attention.Config().make()
    assert attention.max_logit is None
    q, k, v = (torch.randn(2, 5, 3, 8, requires_grad=True) for _ in range(3))
    output = attention(q, k, v, window=window)
    torch.testing.assert_close(output, q + 2 * k + 3 * v)
    output.sum().backward()
    for tensor, scale in zip((q, k, v), (1, 2, 3), strict=True):
        torch.testing.assert_close(tensor.grad, torch.full_like(tensor, scale))
    # A window reaching the whole row is no window.
    history = None if window < 0 or window >= q.shape[1] else window
    assert fake_flash4.windows == [
        (None, None) if history is None else (history, 0),
        (history, 0),
    ]


def test_dense_hands_its_scale_to_both_directions(fake_flash4: _FakeInterface) -> None:
    q, k, v = (torch.randn(2, 5, 3, 8, requires_grad=True) for _ in range(3))
    Flash4Attention.Config().make()(q, k, v, scale=0.3).sum().backward()
    assert fake_flash4.scales == [0.3, 0.3]


@pytest.mark.usefixtures("fake_flash4")
def test_dense_flattens_any_leading_axes_into_rows() -> None:
    q, k, v = (torch.randn(2, 3, 5, 4, 8) for _ in range(3))
    out = Flash4Attention.Config().make()(q, k, v)
    torch.testing.assert_close(out, q + 2 * k + 3 * v)


@pytest.mark.compute_torch_compile
def test_dense_compiles_as_one_opaque_op_under_fullgraph(
    fake_flash4: _FakeInterface,
) -> None:
    # The stand-in refuses to run while Dynamo traces, so passing shows that
    # compile kept each direction as one opaque op and ran it afterwards.
    compiled = torch.compile(
        Flash4Attention.Config().make(),
        fullgraph=True,
        backend="aot_eager",
    )
    q, k, v = (torch.randn(2, 5, 3, 8, requires_grad=True) for _ in range(3))
    out = compiled(q, k, v, window=2)
    torch.testing.assert_close(out, q + 2 * k + 3 * v)
    out.square().sum().backward()
    for tensor, scale in zip((q, k, v), (1, 2, 3), strict=True):
        torch.testing.assert_close(tensor.grad, 2 * scale * out.detach())
    assert fake_flash4.windows == [(2, 0), (2, 0)]


def test_the_dense_forward_fake_declares_a_contiguous_output() -> None:
    # Head-major storage seen through the kernel's [B, S, H, D] layout.
    q = torch.empty(2, 3, 5, 4).transpose(1, 2)
    out, lse = _flash4_forward_fake(q, q, q, None, 0, -1, None)
    assert out.is_contiguous()
    assert out.shape == q.shape
    assert lse.shape == (2, 3, 5)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "layout",
    ["contiguous", "transposed", "aligned", "misaligned"],
)
def test_the_backward_owns_the_contiguous_layout_its_fake_declares(
    monkeypatch: pytest.MonkeyPatch,
    dtype: torch.dtype,
    layout: str,
) -> None:
    """Check the backward against FA4's input normalization, without native execution."""
    value = torch.arange(192, dtype=dtype).reshape(2, 4, 3, 8)
    if layout == "transposed":
        value = value.reshape(2, 4, 8, 3).transpose(-1, -2)
    elif layout == "aligned":
        value = value.reshape(2, 3, 4, 8).transpose(1, 2)
    elif layout == "misaligned":
        value = torch.arange(193, dtype=dtype)[1:].reshape(2, 3, 4, 8).transpose(1, 2)
    gradient = torch.randn(value.shape, dtype=dtype)
    saved = [value, value, value, torch.empty_like(gradient), torch.empty(2, 3, 4)]
    backend = _LayoutInterface()
    monkeypatch.setitem(sys.modules, "flash_attn.cute.interface", backend)
    monkeypatch.setattr(torch.cuda, "get_device_capability", _sm90)
    flash4._interface.cache_clear()
    actual = _flash4_backward(saved, gradient, 0, -1, None)
    declared = _flash4_backward_fake(saved, gradient, 0, -1, None)
    flash4._interface.cache_clear()
    for result, fake, reference in zip(
        actual,
        declared,
        backend.gradients,
        strict=True,
    ):
        assert result.is_contiguous()
        assert result.stride() == fake.stride()
        assert result.dtype == fake.dtype == dtype
        assert result.shape == fake.shape == value.shape
        assert torch.equal(
            result.view(torch.int16),
            reference.contiguous().view(torch.int16),
        )


@pytest.mark.usefixtures("fake_flash4")
@pytest.mark.parametrize("window", [-1, 0, 2])
@pytest.mark.parametrize("scale", [None, 0.3])
def test_varlen_matches_sdpa_varlen_with_grouped_heads(
    *,
    window: int,
    scale: float | None,
) -> None:
    cu_seqlens = torch.tensor(CU_SEQLENS, dtype=torch.int32)
    inputs = [torch.randn(2, 8, heads, 4) for heads in (6, 3, 3)]
    cotangent = torch.randn(2, 8, 6, 4)
    expected = _output_and_grads(
        SdpaVarlen.Config().make(),
        inputs,
        cotangent,
        cu_seqlens=cu_seqlens,
        window=window,
        scale=scale,
    )
    actual = _output_and_grads(
        Flash4Varlen.Config().make(),
        inputs,
        cotangent,
        cu_seqlens=cu_seqlens,
        window=window,
        scale=scale,
    )
    for got, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(got, want)


@pytest.mark.usefixtures("fake_flash4")
def test_varlen_bounds_the_max_logit_by_its_log_normalizer() -> None:
    cu_seqlens = torch.tensor(CU_SEQLENS, dtype=torch.int32)
    q, k, v = torch.randn(2, 8, 6, 4), torch.randn(2, 8, 3, 4), torch.randn(2, 8, 3, 4)
    reference = SdpaVarlen.Config().make()
    reference(q, k, v, cu_seqlens=cu_seqlens, record_max_logit=True)
    kernel = Flash4Varlen.Config().make()
    assert kernel.max_logit is None
    kernel(q, k, v, cu_seqlens=cu_seqlens)
    assert kernel.max_logit is None
    kernel(q, k, v, cu_seqlens=cu_seqlens, record_max_logit=True)
    assert reference.max_logit is not None
    assert kernel.max_logit is not None
    exact, bound = float(reference.max_logit), float(kernel.max_logit)
    # The longest segment has 4 keys.
    assert exact <= bound <= exact + math.log(4)


@pytest.mark.compute_torch_compile
@pytest.mark.usefixtures("fake_flash4")
def test_varlen_compiles_as_one_opaque_op_under_fullgraph() -> None:
    # The stand-in's forward is ``torch.compiler.disable``d, as FA4's CuTe DSL is
    # opaque to Dynamo, so tracing into it fails under ``fullgraph=True``.
    cu_seqlens = torch.tensor(CU_SEQLENS, dtype=torch.int32)
    inputs = [torch.randn(2, 8, heads, 4) for heads in (6, 3, 3)]
    cotangent = torch.randn(2, 8, 6, 4)
    kernel = Flash4Varlen.Config().make()
    compiled = torch.compile(kernel, fullgraph=True, backend="eager")
    expected = _output_and_grads(kernel, inputs, cotangent, cu_seqlens=cu_seqlens)
    actual = _output_and_grads(compiled, inputs, cotangent, cu_seqlens=cu_seqlens)
    for got, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(got, want)


@pytest.mark.usefixtures("fake_flash4")
@pytest.mark.parametrize("kernel", [Flash4Attention.Config(), Flash4Varlen.Config()])
@pytest.mark.parametrize(
    ("is_causal", "attn_mask", "dropout_p"),
    [(False, None, 0.0), (True, torch.zeros(2, 3), 0.0), (True, None, 0.1)],
)
def test_the_kernels_refuse_what_fa4_cannot_express(
    kernel: Flash4Attention.Config | Flash4Varlen.Config,
    *,
    is_causal: bool,
    attn_mask: Tensor | None,
    dropout_p: float,
) -> None:
    q = torch.randn(2, 8, 3, 4)
    with pytest.raises(ValueError, match="causal"):
        # The dense kernel reads no segments; they pass through its bus.
        kernel.make()(
            q,
            q,
            q,
            cu_seqlens=torch.tensor(CU_SEQLENS, dtype=torch.int32),
            is_causal=is_causal,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
        )


@pytest.mark.parametrize("config", [Flash4Attention.Config(), Flash4Varlen.Config()])
@pytest.mark.parametrize("window", [-1, 0, 1, 4, 8, 12])
def test_the_analytical_cost_matches_a_torch_reference(
    config: Flash4Attention.Config | Flash4Varlen.Config,
    window: int,
) -> None:
    """Exercise the real cost bodies; FA4 itself never runs."""
    reference = _FlashCostReference.Config()
    reference.estimate = config
    reference.window = window
    seq_len = 8 if window < 0 else min(window + 1, 8)
    assert_cost_matches_torch(
        reference,
        build_input=lambda: tuple(
            torch.randn(2, seq_len, 3, 4, requires_grad=True) for _ in range(3)
        ),
        seq_len=seq_len,
        batch_size=2,
        dtype=None,
        num_heads=3,
        channels_head=4,
    )


@FA4_WARNINGS
@pytest.mark.gpu_flash_attention
@pytest.mark.gpu_torch_cuda
@pytest.mark.compute_torch_compile
@pytest.mark.parametrize("window", [64, 256])
@pytest.mark.parametrize("transposed", [False, True])
def test_cuda_dense_matches_fa4s_own_autograd_eager_and_compiled(
    *,
    window: int,
    transposed: bool,
) -> None:
    _cuda_interface()
    torch.manual_seed(42)
    shape = (3, 2, 129, 128) if transposed else (3, 129, 2, 128)
    tensors = [
        torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(3)
    ]
    if transposed:
        # Head-major storage, which FA4 normalizes before its kernels run.
        tensors = [tensor.transpose(1, 2) for tensor in tensors]
    cotangent = torch.randn_like(tensors[0])
    expected = _output_and_grads(_fa4_dense, tensors, cotangent, window=window)
    attention = Flash4Attention.Config().make()
    eager = _output_and_grads(attention, tensors, cotangent, window=window)
    # Inductor trusts the gradients' declared layout; a mismatch shows here.
    compiled = _output_and_grads(
        torch.compile(attention, fullgraph=True),
        tensors,
        cotangent,
        window=window,
    )
    for got, eager_got, want in zip(compiled, eager, expected, strict=True):
        torch.testing.assert_close(eager_got, want, rtol=0, atol=0)
        torch.testing.assert_close(got, want, rtol=0, atol=0)


@FA4_WARNINGS
@pytest.mark.gpu_flash_attention
@pytest.mark.gpu_torch_cuda
@pytest.mark.parametrize("window", [-1, 0, 64])
def test_cuda_dense_with_grouped_heads_matches_sdpa(window: int) -> None:
    _cuda_interface()
    torch.manual_seed(42)
    q = torch.randn(3, 129, 6, 128, device="cuda", dtype=torch.bfloat16)
    k, v = (
        torch.randn(3, 129, 2, 128, device="cuda", dtype=torch.bfloat16)
        for _ in range(2)
    )
    inputs, cotangent = [q, k, v], torch.randn_like(q)
    _assert_within_flash_attention_error(
        _output_and_grads(
            Flash4Attention.Config().make(),
            inputs,
            cotangent,
            window=window,
        ),
        exact=_output_and_grads(
            functools.partial(_in_float32, kernel=_grouped_sdpa),
            inputs,
            cotangent,
            window=window,
        ),
        rounded=_output_and_grads(_grouped_sdpa, inputs, cotangent, window=window),
    )


@FA4_WARNINGS
@pytest.mark.gpu_flash_attention
@pytest.mark.gpu_torch_cuda
@pytest.mark.parametrize("window", [0, 64])
def test_cuda_varlen_window_matches_sdpa_varlen(window: int) -> None:
    _cuda_interface()
    torch.manual_seed(42)
    lengths = torch.tensor([0, 37, 128, 129, 256], device="cuda", dtype=torch.int32)
    cu_seqlens = torch.cat([lengths, lengths[1:] + 256])
    q = torch.randn(2, 256, 6, 128, device="cuda", dtype=torch.bfloat16)
    k, v = (
        torch.randn(2, 256, 3, 128, device="cuda", dtype=torch.bfloat16)
        for _ in range(2)
    )
    inputs, cotangent = [q, k, v], torch.randn_like(q)
    reference = SdpaVarlen.Config().make()
    _assert_within_flash_attention_error(
        _output_and_grads(
            Flash4Varlen.Config().make(),
            inputs,
            cotangent,
            cu_seqlens=cu_seqlens,
            window=window,
        ),
        exact=_output_and_grads(
            functools.partial(_in_float32, kernel=reference),
            inputs,
            cotangent,
            cu_seqlens=cu_seqlens,
            window=window,
        ),
        rounded=_output_and_grads(
            reference,
            inputs,
            cotangent,
            cu_seqlens=cu_seqlens,
            window=window,
        ),
    )


@FA4_WARNINGS
@pytest.mark.gpu_flash_attention
@pytest.mark.gpu_torch_cuda
@pytest.mark.compute_torch_compile
def test_cuda_varlen_matches_fa4s_own_autograd_eager_and_compiled() -> None:
    interface = _cuda_interface()
    torch.manual_seed(42)
    lengths = torch.tensor([0, 37, 128, 129, 256], device="cuda", dtype=torch.int32)
    cu_seqlens = torch.cat([lengths, lengths[1:] + 256])
    q = torch.randn(2, 256, 6, 128, device="cuda", dtype=torch.bfloat16)
    k, v = (
        torch.randn(2, 256, 3, 128, device="cuda", dtype=torch.bfloat16)
        for _ in range(2)
    )
    cotangent = torch.randn_like(q)
    leaves = [t.flatten(0, 1).detach().requires_grad_() for t in (q, k, v)]
    reference, _ = interface.flash_attn_varlen_func(
        *leaves,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=256,
        max_seqlen_k=256,
        softmax_scale=None,
        causal=True,
        window_size=(None, None),
        return_lse=True,
    )
    reference.backward(cotangent.flatten(0, 1))
    expected = [reference.unflatten(0, (2, 256))] + [
        t.grad.unflatten(0, (2, 256)) for t in leaves if t.grad is not None
    ]
    kernel = Flash4Varlen.Config().make()
    eager = _output_and_grads(kernel, [q, k, v], cotangent, cu_seqlens=cu_seqlens)
    # Inductor trusts the gradients' declared layout; a mismatch shows here.
    compiled = _output_and_grads(
        torch.compile(kernel),
        [q, k, v],
        cotangent,
        cu_seqlens=cu_seqlens,
    )
    for got, eager_got, want in zip(compiled, eager, expected, strict=True):
        torch.testing.assert_close(eager_got, want, rtol=0, atol=0)
        torch.testing.assert_close(got, want, rtol=0, atol=0)


class _FlashCostReference(nn.Module):
    """Price a FA4 config while a torch kernel does the work, FA4 never built."""

    class Config(Fig["_FlashCostReference"]):
        estimate: Flash4Attention.Config | Flash4Varlen.Config = field(
            default_factory=Flash4Attention.Config,
        )
        """The FA4 config whose analytical cost is under test."""

        window: int = -1
        """Previous keys admitted in addition to the current position."""

        def cost(self, **kwargs: object) -> Cost:
            """Delegate accounting without constructing FA4."""
            return cost(self.estimate, window=self.window, **kwargs)

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.window = config.window
        self.reference = SdpaNaive.Config().make()

    @override
    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        return self.reference(q, k, v, is_causal=True, window=self.window)


def _output_and_grads(
    kernel: Callable[..., Tensor],
    inputs: list[Tensor],
    cotangent: Tensor,
    **kwargs: object,
) -> list[Tensor]:
    """Return a kernel's output and the gradients of ``q``, ``k``, ``v``."""
    leaves = [tensor.clone().requires_grad_() for tensor in inputs]
    out = kernel(*leaves, **kwargs)
    return [out.detach(), *torch.autograd.grad(out, leaves, cotangent)]


def _fa4_dense(q: Tensor, k: Tensor, v: Tensor, *, window: int) -> Tensor:
    """Attend through FA4's own autograd; a window reaching the whole row is none."""
    whole = window < 0 or window >= q.shape[1]
    out, _ = flash4._interface().flash_attn_func(
        q,
        k,
        v,
        softmax_scale=None,
        causal=True,
        window_size=(None, None) if whole else (window, 0),
        return_lse=True,
    )
    return out


def _grouped_sdpa(q: Tensor, k: Tensor, v: Tensor, *, window: int) -> Tensor:
    """Attend causally with each key and value head repeated for the queries it serves."""
    groups = q.shape[-2] // k.shape[-2]
    keys, values = (t.repeat_interleave(groups, dim=-2) for t in (k, v))
    return SdpaFused.Config().make()(q, keys, values, is_causal=True, window=window)


def _in_float32(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    kernel: Callable[..., Tensor],
    **kwargs: object,
) -> Tensor:
    """Run ``kernel`` in float32 and round its output to the inputs' dtype."""
    return kernel(q.float(), k.float(), v.float(), **kwargs).to(q.dtype)


# Each may stray from the float32 reference, rounded to its dtype, at most twice as far
# as the same reference computed in that dtype does, plus twice the dtype's rounding
# near the reference's values: the bound FlashAttention's own forward and backward
# checks use (flash-attention's tests/cute/test_flash_attn.py).
def _assert_within_flash_attention_error(
    actual: list[Tensor],
    *,
    exact: list[Tensor],
    rounded: list[Tensor],
) -> None:
    """Hold each FA4 tensor to FlashAttention's own accuracy criterion."""
    for got, want, baseline in zip(actual, exact, rounded, strict=True):
        error = float((got - want).abs().max())
        bound = float(
            2 * (baseline - want).abs().max()
            + 2 * (want + 0.3 - 0.3 - want).abs().max(),
        )
        assert error <= bound, f"error {error} exceeds {bound}"


def _cuda_interface() -> flash4._Flash4Interface:
    """Return the installed FA4, skipping where it cannot run."""
    if not torch.cuda.is_available():
        pytest.skip(
            "Requires an SM90 or SM100 CUDA device and flash-attn-4.",
        )
    if torch.cuda.get_device_capability() not in {(9, 0), (10, 0)}:
        pytest.skip("Requires an SM90 or SM100 CUDA device.")
    pytest.importorskip("flash_attn.cute.interface", reason="Requires flash-attn-4.")
    flash4._interface.cache_clear()
    return flash4._interface()


def _sm90(device: object = None) -> tuple[int, int]:
    """Stand in for ``get_device_capability``; CUDA's lazy init passes a device."""
    del device
    return 9, 0


class _FakeInterface(ModuleType):
    """A stand-in FA4: arithmetic over rows, and real attention over segments.

    The dense entry points compute ``q + 2k + 3v`` and its gradients, so a test
    sees the layout, the windows and the scales the kernel hands over. The varlen
    ones compute segment attention densely at the given scale, with its gradients
    written out from the saved lse, so the kernel can be held to ``SdpaVarlen``.
    """

    def __init__(self) -> None:
        super().__init__("flash_attn.cute.interface")
        self.windows: list[tuple[int | None, int | None]] = []
        self.scales: list[float | None] = []

    def flash_attn_func(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        *,
        softmax_scale: float | None,
        causal: bool,
        window_size: tuple[int | None, int | None],
        return_lse: bool,
    ) -> tuple[Tensor, Tensor]:
        # FA4's CuTe DSL is opaque to Dynamo; only the custom op may call it.
        assert not torch.compiler.is_compiling()
        assert causal
        assert return_lse
        self.windows.append(window_size)
        self.scales.append(softmax_scale)
        return q + 2 * k + 3 * v, torch.zeros(q.shape[0], q.shape[2], q.shape[1])

    def flash_attn_varlen_func(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        *,
        cu_seqlens_q: Tensor,
        cu_seqlens_k: Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        softmax_scale: float | None,
        causal: bool,
        window_size: tuple[int | None, int | None],
        return_lse: bool,
    ) -> tuple[Tensor, Tensor]:
        assert causal
        assert return_lse
        assert cu_seqlens_k is cu_seqlens_q
        assert max_seqlen_q == max_seqlen_k
        self.windows.append(window_size)
        self.scales.append(softmax_scale)
        window = -1 if window_size[0] is None else window_size[0]
        return _opaque_segment_attention(
            q,
            k,
            v,
            cu_seqlens_q,
            window=window,
            scale=softmax_scale,
        )

    def _flash_attn_bwd(  # noqa: PLR0917 -- FA4's backward takes six positional tensors.
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        out: Tensor,
        grad_out: Tensor,
        lse: Tensor,
        /,
        *,
        softmax_scale: float | None,
        causal: bool,
        window_size_left: int | None,
        window_size_right: int,
        cu_seqlens_q: Tensor | None,
        cu_seqlens_k: Tensor | None,
        max_seqlen_q: int | None,
        max_seqlen_k: int | None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        assert not torch.compiler.is_compiling()
        assert causal
        assert cu_seqlens_k is cu_seqlens_q
        assert max_seqlen_q == max_seqlen_k
        self.windows.append((window_size_left, window_size_right))
        self.scales.append(softmax_scale)
        if cu_seqlens_q is None:
            return grad_out.clone(), 2 * grad_out, 3 * grad_out
        # A custom op's body runs below autograd, so the gradient is written out:
        # P from the saved lse, dS = P * (dO V^T - rowsum(dO * O)), then GQA sums.
        window = -1 if window_size_left is None else window_size_left
        mask = segment_mask(cu_seqlens_q, rows=1, length=len(q), window=window)[0]
        groups = q.shape[-2] // k.shape[-2]
        keys, values = (t.repeat_interleave(groups, dim=-2) for t in (k, v))
        scale = float(q.shape[-1]) ** -0.5 if softmax_scale is None else softmax_scale
        scores = torch.einsum("qhd,khd->hqk", q, keys) * scale
        probs = (scores - lse[..., None]).exp().masked_fill(~mask, 0.0)
        d_probs = torch.einsum("qhd,khd->hqk", grad_out, values)
        d_scores = probs * (d_probs - (grad_out * out).sum(-1).T[..., None])
        dq = torch.einsum("hqk,khd->qhd", d_scores * scale, keys)
        dk, dv = (
            torch.einsum("hqk,qhd->khd", p, x).unflatten(-2, (-1, groups)).sum(-2)
            for p, x in ((d_scores * scale, q), (probs, grad_out))
        )
        return dq, dk, dv


class _LayoutInterface(ModuleType):
    """Emulate only FA4's input normalization and gradient allocation."""

    def __init__(self) -> None:
        super().__init__("flash_attn.cute.interface")
        self.gradients: list[Tensor] = []

    def flash_attn_func(self, *args: object, **kwargs: object) -> tuple[Tensor, Tensor]:
        del args, kwargs
        raise NotImplementedError

    def flash_attn_varlen_func(
        self,
        *args: object,
        **kwargs: object,
    ) -> tuple[Tensor, Tensor]:
        del args, kwargs
        raise NotImplementedError

    def _flash_attn_bwd(  # noqa: PLR0917 -- FA4's backward takes six positional tensors.
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        out: Tensor,
        grad_out: Tensor,
        lse: Tensor,
        /,
        **kwargs: object,
    ) -> tuple[Tensor, Tensor, Tensor]:
        del out, lse, kwargs
        # flash-attn-4 4.0.0b32: cute_dsl_utils.py:70-99; interface.py:2080-2081,
        # 2208-2219.
        for scale, value in enumerate((q, k, v), start=1):
            aligned = value.data_ptr() % 16 == 0
            strides_aligned = value.stride(-1) == 1 and all(
                stride % (16 // value.element_size()) == 0
                for stride in value.stride()[:-1]
            )
            if not aligned:
                normalized = value.clone(memory_format=torch.contiguous_format)
            elif value.is_contiguous() or strides_aligned:
                normalized = value
            else:
                normalized = value.contiguous()
            self.gradients.append(torch.empty_like(normalized).copy_(grad_out * scale))
        return self.gradients[0], self.gradients[1], self.gradients[2]


# Both are detached, so a gradient can only come from ``_flash_attn_bwd``.
@torch.compiler.disable
def _opaque_segment_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cu_seqlens: Tensor,
    *,
    window: int,
    scale: float | None,
) -> tuple[Tensor, Tensor]:
    """Return segment attention of ``[N, H, D]`` inputs and each query's lse ``[H, N]``."""
    reference = SdpaVarlen.Config().make()
    out = reference(
        q[None],
        k[None],
        v[None],
        cu_seqlens=cu_seqlens,
        window=window,
        scale=scale,
    )
    mask = segment_mask(cu_seqlens, rows=1, length=len(q), window=window)[0]
    keys = k.repeat_interleave(q.shape[-2] // k.shape[-2], dim=-2)
    logit_scale = float(q.shape[-1] ** -0.5) if scale is None else scale
    scores = torch.einsum("qhd,khd->hqk", q, keys) * logit_scale
    lse = scores.masked_fill(~mask, float("-inf")).logsumexp(-1)
    return out[0].detach(), lse.detach()


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
