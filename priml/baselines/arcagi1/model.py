"""Reference-TRM pieces the ARC recipes inject into the shared puzzle solver.

The solver is :class:`~priml.baselines.sudoku.model.SudokuNet`; everything
here fills one of its slots:

* :class:`ConvSwiGLU` -- a feed-forward for the transformer block's ``ffn``
  slot that mixes a short window of neighbouring tokens on the gated branch.
* :class:`UrmRecurrence` -- a recurrence for the ``recurrence`` slot that
  carries ONE latent state instead of TRM's slow/fast pair.

Together they are the Universal Reasoning Model's architecture on the TRM
chassis: the same embedding, heads, halting, and puzzle prefix, a different
update rule and a different block.
"""

from __future__ import annotations

from dataclasses import KW_ONLY
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Protocol, Self, override

from configgle import Fig, Makeable, Makes
from torch import Tensor, nn

import torch

from priml.baselines.sudoku.model import DeepRecurrence
from priml.cost import Cost, cost, elementwise_cost
from priml.math.basic import ceil_multiple
from priml.model.conv import Conv1d
from priml.model.custom_types import ChannelsIn, TensorModule
from priml.model.init import InitFn, kaiming_uniform
from priml.model.linear import Linear
from priml.model.swiglu import SwiGLU, silu


if TYPE_CHECKING:
    from collections.abc import Mapping

    from priml.baselines.sudoku.model import MixFn


REFERENCE_NAMES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "embed_tokens.": "embedding.embed_tokens.",
        "embed_feedback": "embedding.channels.0.embed_feedback",
        "q_head.": "halt_head.",
        "register_tokens": "prefix.register_tokens",
        "puzzle_emb.weights": "prefix.weights",
    },
)
"""Parameter-name prefixes of the reference implementation's checkpoints, to ours.

Every other name -- the reasoning blocks, the output head -- is shared."""


def from_reference_name(name: str) -> str:
    """Return the priml name of a reference checkpoint's parameter ``name``."""
    for old, new in REFERENCE_NAMES.items():
        if name.startswith(old):
            return new + name.removeprefix(old)
    return name


class ShortConvFn(Protocol):
    """Depthwise short convolution over the sequence axis of ``[B, S, C]``."""

    def __call__(self, conv: nn.Conv1d, x: Tensor, *, causal: bool) -> Tensor:
        """Apply to the input."""
        ...


def depthwise_conv(conv: nn.Conv1d, x: Tensor, *, causal: bool) -> Tensor:
    """Run the depthwise ``Conv1d`` kernel, then SiLU.

    Non-causal pads ``kernel // 2`` on both sides and trims the right overhang
    to the input length, so each token mixes a small symmetric window; causal
    left-pads ``kernel - 1``.

    Args:
      conv: Depthwise convolution (``groups == channels``), unpadded.
      x: ``[B, S, C]`` gated activations.
      causal: Restrict the window to positions at or before each token.

    Returns:
      mixed: ``[B, S, C]`` in ``x``'s dtype.

    """
    kernel = conv.kernel_size[0]
    s = x.shape[1]
    xt = x.transpose(1, 2)
    if causal:
        xt = conv(nn.functional.pad(xt, (kernel - 1, 0)).to(conv.weight.dtype))
    else:
        padded = nn.functional.pad(xt, (kernel // 2, kernel // 2))
        xt = conv(padded.to(conv.weight.dtype))[..., :s]
    return nn.functional.silu(xt).transpose(1, 2).to(x.dtype)


def depthwise_shift(conv: nn.Conv1d, x: Tensor, *, causal: bool) -> Tensor:
    """Evaluate the same depthwise window as shifted elementwise taps, then SiLU.

    Reads ``conv``'s weight and bias but never its kernel, so the taps stay
    fused elementwise work inside a compiled core. It agrees with
    :func:`depthwise_conv` up to the order of the per-tap additions: bit-equal
    in bfloat16 at kernel 2, one float32 ULP apart otherwise.

    Args:
      conv: Depthwise convolution whose ``[C, 1, K]`` weight supplies the taps.
      x: ``[B, S, C]`` gated activations.
      causal: Restrict the window to positions at or before each token.

    Returns:
      mixed: ``[B, S, C]``; SiLU applied after the cast back to ``x``'s dtype.

    """
    weight = conv.weight
    kernel = conv.kernel_size[0]
    h = x.to(weight.dtype)
    s = h.shape[1]
    left_pad = kernel - 1 if causal else kernel // 2
    out = h * weight[:, 0, left_pad]
    for k in range(kernel):
        shift = k - left_pad
        if shift < 0:
            shifted = nn.functional.pad(h, (0, 0, -shift, 0))[:, :s, :]
        elif shift > 0:
            shifted = nn.functional.pad(h[:, shift:, :], (0, 0, 0, shift))[:, :s, :]
        else:
            continue
        out = out + shifted * weight[:, 0, k]
    if conv.bias is not None:
        out = out + conv.bias
    return nn.functional.silu(out.to(x.dtype))


class ConvSwiGLU(nn.Module):
    """SwiGLU feed-forward with a depthwise short convolution on the gated output.

    ``down(short_conv(g(gate, up)))`` where ``gate, up = up_proj(x).chunk(2)``
    and ``g`` is ``silu(gate) * up`` without a norm, or the modified
    ``sigmoid(gate) * norm(gate * up)`` with one -- the form that keeps a
    high-rate Muon body stable.

    References:
      https://arxiv.org/abs/2512.14693
        Gao et al. Universal Reasoning Model.

    """

    class Config(Fig["ConvSwiGLU"], kw_only=False):
        """Widths, the short window, and the optional gate norm."""

        channels_in: int = -1
        """Input width; -1 infers it from ``channels_out``."""

        channels_out: int = -1
        """Output width; -1 infers it from ``channels_in``."""

        _: KW_ONLY

        channels_hidden: int = -1
        """Hidden width; -1 rounds ``channels_in * expansion`` up to ``round_to``."""

        expansion: float = 8 / 3
        """Hidden-to-input ratio when ``channels_hidden`` is inferred."""

        round_to: int = 256
        """Grid the inferred hidden width is rounded up to."""

        kernel_size: int = 2
        """Short-convolution window."""

        causal: bool = False
        """Left-only window; the default mixes a symmetric one."""

        bias: bool = False
        """Bias on the two projections. The convolution always has one."""

        short_conv: ShortConvFn = depthwise_conv
        """Evaluates the window; :func:`depthwise_shift` keeps it elementwise."""

        norm: Makeable[TensorModule] | None = None
        """Gate norm; ``None`` is the plain SiLU gate."""

        init_weight: InitFn = kaiming_uniform
        """Initializer for both projections."""

        @override
        def finalize(self) -> Self:
            if self.channels_in == -1:
                self.channels_in = self.channels_out
            if self.channels_out == -1:
                self.channels_out = self.channels_in
            if self.channels_hidden == -1:
                self.channels_hidden = int(
                    ceil_multiple(self.channels_in * self.expansion, self.round_to),
                )
            if isinstance(self.norm, ChannelsIn) and self.norm.channels_in == -1:
                self.norm.channels_in = self.channels_hidden
            return super().finalize()

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost the gated projections, the short window, and its SiLU.

            The gate and projections are :class:`SwiGLU`'s with the same widths.
            :func:`depthwise_shift` runs the window as ``kernel_size`` shifted
            taps, so it is elementwise work; :func:`depthwise_conv` runs one
            depthwise convolution over the padded sequence.

            Args:
              seq_len: Tokens per sequence.
              batch_size: Sequences per step.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus, unread.

            Returns:
              cost: Integer FLOPs and logical bytes for the complete invocation.

            """
            del kwargs
            gated = SwiGLU.Config(
                channels_in=self.channels_in,
                channels_out=self.channels_out,
                channels_hidden=self.channels_hidden,
                bias=self.bias,
                norm=self.norm,
            )
            hidden = self.channels_hidden
            rows = seq_len * batch_size
            if self.short_conv is depthwise_shift:
                tap = elementwise_cost(
                    primal=hidden * rows,
                    adjoint=2 * hidden * rows,
                    channels=hidden,
                    params=hidden,
                    rows=rows,
                    dtype=dtype,
                )
                combine = elementwise_cost(
                    primal=(self.kernel_size - 1) * hidden * rows,
                    adjoint=(self.kernel_size - 1) * hidden * rows,
                    channels=(self.kernel_size - 1) * hidden,
                    inputs=2,
                    rows=rows,
                    dtype=dtype,
                )
                conv_bias = elementwise_cost(
                    primal=hidden * rows,
                    adjoint=hidden * rows,
                    channels=hidden,
                    params=hidden,
                    rows=rows,
                    dtype=dtype,
                )
                window = tap.tile(self.kernel_size, copies=self.kernel_size)
                window += combine + conv_bias
            else:
                pad = (
                    self.kernel_size - 1 if self.causal else 2 * (self.kernel_size // 2)
                )
                window = cost(
                    Conv1d.Config(
                        channels_in=hidden,
                        channels_out=hidden,
                        kernel_size=self.kernel_size,
                        padding=0,
                        groups=hidden,
                        bias=True,
                    ),
                    input_grid=seq_len + pad,
                    batch_size=batch_size,
                    dtype=dtype,
                )
            return (
                cost(gated, seq_len=seq_len, batch_size=batch_size, dtype=dtype)
                + window
                + cost(silu, channels=hidden * rows, dtype=dtype)
            )

    def __init__(self, config: Config) -> None:
        super().__init__()
        c_h = config.channels_hidden
        self.causal = config.causal
        self.short_conv = config.short_conv
        # Construction order is the init draw order: up, down, conv.
        self.up_proj = Linear.Config(
            channels_in=config.channels_in,
            channels_out=c_h * 2,
            bias=config.bias,
            init_weight=config.init_weight,
        ).make()
        self.down_proj = Linear.Config(
            channels_in=c_h,
            channels_out=config.channels_out,
            bias=config.bias,
            init_weight=config.init_weight,
        ).make()
        self.conv = Conv1d.Config(
            channels_in=c_h,
            channels_out=c_h,
            kernel_size=config.kernel_size,
            padding=0,
            groups=c_h,
            bias=True,
        ).make()
        self.norm = None if config.norm is None else config.norm.make()

    def reset_parameters(self) -> None:
        """Initialize every parameter in place."""
        self.up_proj.reset_parameters()
        self.down_proj.reset_parameters()
        self.conv.reset_parameters()

    @override
    def forward(self, x: Tensor, **kwargs: object) -> Tensor:
        del kwargs
        gate, up = self.up_proj(x).chunk(2, dim=-1)
        if self.norm is None:
            gated = nn.functional.silu(gate) * up
        else:
            gated = torch.sigmoid(gate) * self.norm(gate * up)
        return self.down_proj(self.short_conv(self.conv, gated, causal=self.causal))


class UrmRecurrence(DeepRecurrence):
    """Refine ONE latent by ``fast_cycles`` block passes per core application.

    The state rides in the ``z_slow`` slot, which the heads read; ``z_fast`` is
    carried through untouched so the two-latent pool contract holds. Truncated
    backprop is the parent's ``slow_cycles`` split by default; with
    ``inner_grad_loops`` it moves inside the core instead.

    References:
      https://arxiv.org/abs/2512.14693
        Gao et al. Universal Reasoning Model.

    """

    class Config(Makes["UrmRecurrence"], DeepRecurrence.Config):
        """Cycle counts plus the in-core truncation length."""

        inner_grad_loops: int = 0
        """Passes that carry gradient inside one core application.

        0 keeps every pass differentiable, so truncation is the parent's
        ``slow_cycles`` split. Positive runs the first
        ``fast_cycles - inner_grad_loops`` passes without gradient, the
        paper's in-core truncation; pair it with ``slow_cycles=1``."""

    @override
    def refine(
        self,
        mix: MixFn,
        input_emb: Tensor,
        z_slow: Tensor,
        z_fast: Tensor,
        cos_sin: tuple[Tensor, Tensor] | None,
    ) -> tuple[Tensor, Tensor]:
        """Re-inject the input before every pass over the single state."""
        config = self.config
        assert isinstance(config, UrmRecurrence.Config)
        h = z_slow
        grad_loops = config.inner_grad_loops
        if grad_loops > 0:
            with torch.no_grad():
                for _ in range(max(config.fast_cycles - grad_loops, 0)):
                    h = mix(h.detach() + input_emb.detach(), cos_sin)
            h = h.detach()
            for _ in range(grad_loops):
                h = mix(h + input_emb, cos_sin)
        else:
            for _ in range(config.fast_cycles):
                h = mix(h + input_emb, cos_sin)
        return h, z_fast
