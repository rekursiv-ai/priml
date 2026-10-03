"""FlashAttention 4 kernels: causal attention over dense rows or packed segments.

FA4's ``flash_attn_func`` and ``flash_attn_varlen_func`` are
``autograd.Function``s over its CuTe DSL, which Dynamo cannot trace: under
``torch.compile(fullgraph=True)`` they raise at the first step. Each direction
therefore runs as a torch custom op, which compile keeps as one opaque node,
and FA4's own backward entry point computes the gradients.

Both kernels take the ``AttentionKernel`` layout,
``[..., S, num_heads, channels_head]``, and attend causally, optionally within
``window`` previous keys. Keys and values may carry fewer heads than the
queries, a divisor of theirs, which FA4 groups. An additive mask, attention
dropout and non-causal attention have no path here; each is refused rather
than dropped.

``flash-attn-4`` is Linux-only and optional. A kernel imports it when built, so
a missing install fails at ``make()`` rather than mid-step, and importing this
module needs nothing.
"""

# No ``from __future__ import annotations``: ``torch.library.custom_op`` infers
# each op's schema from its annotations, which that import would make strings.

from collections.abc import Callable
from typing import Protocol, override, runtime_checkable

import functools
import importlib

from configgle import Fig
from torch import Tensor, nn

import torch

from priml.cost import Cost
from priml.model.attention.kernel import attention_kernel_cost


class Flash4Attention(nn.Module):
    """Causal FlashAttention 4 over dense rows, optionally within a sliding window.

    Attributes:
      max_logit: For the last forward that asked for it (``record_max_logit``),
        the largest log-normalizer of a query over the keys it attends to,
        detached float32: an upper bound on the largest scaled ``q·k``, at most
        ``log`` of a query's key count above it. FA4 never forms the logits.
        None otherwise.

    """

    class Config(Fig["Flash4Attention"]):
        """Nothing to set: FA4 reads every shape from its inputs."""

        @classmethod
        def cost(
            cls,
            *,
            seq_len: int,
            batch_size: int = 1,
            dtype: torch.dtype | None,
            num_heads: int,
            channels_head: int,
            channels_v_head: int = -1,
            window: int = -1,
            dropout_p: float = 0.0,
            rows: int = -1,
            **kwargs: object,
        ) -> Cost:
            """Cost the kernel from the shapes its owner hands it.

            See :func:`attention_kernel_cost` for every argument.

            Args:
              seq_len: Tokens per sequence.
              batch_size: Sequences in this invocation.
              dtype: Activation dtype; ``None`` is torch's default.
              num_heads: Query heads.
              channels_head: Width of each query/key head.
              channels_v_head: Value width; -1 uses the query/key width.
              window: Previous keys each query reaches, plus itself; negative is unbounded.
              dropout_p: Attention dropout rate.
              rows: Query rows sharing K/V; negative uses the modeled key count.
              **kwargs: The rest of the owner's bus, unread.

            Returns:
              cost: Whole-invocation cost of the kernel.

            """
            del kwargs
            return attention_kernel_cost(
                seq_len=seq_len,
                batch_size=batch_size,
                dtype=dtype,
                num_heads=num_heads,
                channels_head=channels_head,
                channels_v_head=channels_v_head,
                window=window,
                dropout_p=dropout_p,
                rows=rows,
            )

    def __init__(self, config: Config) -> None:
        del config
        super().__init__()
        _interface()
        self.max_logit: Tensor | None = None

    @override
    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        *,
        window: int = -1,
        is_causal: bool = True,
        attn_mask: Tensor | None = None,
        dropout_p: float = 0.0,
        scale: float | None = None,
        record_max_logit: bool = False,
        **kwargs: object,
    ) -> Tensor:
        """Attend causally over each row.

        Args:
          q: Queries ``[..., S, H, D]``.
          k: Keys ``[..., S, H_kv, D]``; ``H_kv`` divides ``H``.
          v: Values, shaped like ``k``.
          window: Previous keys each query reaches, plus itself; -1 for all.
          is_causal: Must be True.
          attn_mask: Must be None.
          dropout_p: Must be 0.
          scale: Logit scale; None is ``D**-0.5``.
          record_max_logit: Set ``max_logit``.
          **kwargs: The rest of the bus, unread.

        Returns:
          out: ``[..., S, H, D]``.

        Raises:
          ValueError: Non-causal attention, a mask, or dropout was asked for.

        """
        del kwargs
        if not is_causal or attn_mask is not None or dropout_p:
            raise ValueError("Flash4Attention is causal and takes no mask or dropout.")
        rows = (-1, *q.shape[-3:])
        out, lse = _flash4_forward(
            q.reshape(rows),
            k.reshape(-1, *k.shape[-3:]),
            v.reshape(-1, *v.shape[-3:]),
            None,
            0,
            -1 if window >= q.shape[-3] else window,
            scale,
        )
        self.max_logit = lse.detach().amax().float() if record_max_logit else None
        return out.view(q.shape)


class Flash4Varlen(nn.Module):
    """Causal FlashAttention 4 within packed segments, optionally within a window.

    ``cu_seqlens`` bounds the segments of the flattened leading and ``S`` axes,
    none crossing a row, as ``SdpaVarlen``'s does; that kernel is the portable
    reference this one must match.

    Attributes:
      max_logit: As ``Flash4Attention.max_logit``, over each query's segment.

    """

    class Config(Fig["Flash4Varlen"]):
        """Nothing to set: the segments come from ``cu_seqlens`` at call time."""

        @classmethod
        def cost(
            cls,
            *,
            seq_len: int,
            batch_size: int = 1,
            dtype: torch.dtype | None,
            num_heads: int,
            channels_head: int,
            channels_v_head: int = -1,
            window: int = -1,
            dropout_p: float = 0.0,
            rows: int = -1,
            **kwargs: object,
        ) -> Cost:
            """Cost the kernel as if each row were one segment, an upper bound.

            See :func:`attention_kernel_cost` for every argument.

            Args:
              seq_len: Tokens per row.
              batch_size: Rows in this invocation.
              dtype: Activation dtype; ``None`` is torch's default.
              num_heads: Query heads.
              channels_head: Width of each query/key head.
              channels_v_head: Value width; -1 uses the query/key width.
              window: Previous keys each query reaches, plus itself; negative is unbounded.
              dropout_p: Attention dropout rate.
              rows: Query rows sharing K/V; negative uses the modeled key count.
              **kwargs: The rest of the owner's bus, unread.

            Returns:
              cost: Whole-invocation cost of the kernel.

            """
            del kwargs
            return attention_kernel_cost(
                seq_len=seq_len,
                batch_size=batch_size,
                dtype=dtype,
                num_heads=num_heads,
                channels_head=channels_head,
                channels_v_head=channels_v_head,
                window=window,
                dropout_p=dropout_p,
                rows=rows,
            )

    def __init__(self, config: Config) -> None:
        del config
        super().__init__()
        _interface()
        self.max_logit: Tensor | None = None

    @override
    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        *,
        cu_seqlens: Tensor,
        window: int = -1,
        is_causal: bool = True,
        attn_mask: Tensor | None = None,
        dropout_p: float = 0.0,
        scale: float | None = None,
        record_max_logit: bool = False,
        **kwargs: object,
    ) -> Tensor:
        """Attend causally within each segment.

        Args:
          q: Queries ``[..., S, H, D]``.
          k: Keys ``[..., S, H_kv, D]``; ``H_kv`` divides ``H``.
          v: Values, shaped like ``k``.
          cu_seqlens: Int32 segment boundaries of the flattened leading and
            ``S`` axes.
          window: Previous keys each query reaches, plus itself; -1 for all.
          is_causal: Must be True.
          attn_mask: Must be None; the segments are the mask.
          dropout_p: Must be 0.
          scale: Logit scale; None is ``D**-0.5``.
          record_max_logit: Set ``max_logit``.
          **kwargs: The rest of the bus, unread.

        Returns:
          out: ``[..., S, H, D]``.

        Raises:
          ValueError: Non-causal attention, a mask, or dropout was asked for.

        """
        del kwargs
        if not is_causal or attn_mask is not None or dropout_p:
            raise ValueError("Flash4Varlen is causal and takes no mask or dropout.")
        length = q.shape[-3]
        # No segment crosses a row, so ``length`` bounds them all and no device
        # sync measures the longest.
        out, lse = _flash4_forward(
            q.reshape(-1, *q.shape[-2:]),
            k.reshape(-1, *k.shape[-2:]),
            v.reshape(-1, *v.shape[-2:]),
            cu_seqlens,
            length,
            -1 if window >= length else window,
            scale,
        )
        self.max_logit = lse.detach().amax().float() if record_max_logit else None
        return out.view(q.shape)


@torch.library.custom_op("priml::flash4_forward", mutates_args=())
def _flash4_forward(  # noqa: PLR0917 -- A custom op's schema is positional.
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cu_seqlens: Tensor | None,
    max_seqlen: int,
    window: int,
    scale: float | None,
) -> tuple[Tensor, Tensor]:
    """Run FA4's causal forward on ``[B, S, H, D]``, or ``[N, H, D]`` with segments."""
    interface = _interface()
    window_size = (None, None) if window < 0 else (window, 0)
    if cu_seqlens is None:
        out, lse = interface.flash_attn_func(
            q,
            k,
            v,
            softmax_scale=scale,
            causal=True,
            window_size=window_size,
            return_lse=True,
        )
    else:
        out, lse = interface.flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            softmax_scale=scale,
            causal=True,
            window_size=window_size,
            return_lse=True,
        )
    if lse is None:
        raise ValueError("FA4 returns lse when return_lse is set.")
    return out, lse


@_flash4_forward.register_fake
def _flash4_forward_fake(  # noqa: PLR0917 -- A custom op's schema is positional.
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cu_seqlens: Tensor | None,
    max_seqlen: int,
    window: int,
    scale: float | None,
) -> tuple[Tensor, Tensor]:
    del k, v, max_seqlen, window, scale
    out = torch.empty_like(q, memory_format=torch.contiguous_format)
    # FA4's lse is ``[B, H, S]`` over rows and ``[H, N]`` over segments.
    shape = (
        (q.shape[0], q.shape[2], q.shape[1]) if cu_seqlens is None else q.shape[1::-1]
    )
    return out, q.new_empty(shape, dtype=torch.float32)


@torch.library.custom_op("priml::flash4_backward", mutates_args=())
def _flash4_backward(
    saved: list[Tensor],
    grad_out: Tensor,
    max_seqlen: int,
    window: int,
    scale: float | None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Run FA4's causal backward; ``saved`` is q, k, v, out, lse, then any segments."""
    q, k, v, out, lse, *segments = saved
    cu_seqlens = segments[0] if segments else None
    dq, dk, dv = _interface()._flash_attn_bwd(  # noqa: SLF001 -- FA4 exposes its backward only through this entry point.
        q,
        k,
        v,
        out,
        grad_out,
        lse,
        softmax_scale=scale,
        causal=True,
        window_size_left=None if window < 0 else window,
        window_size_right=0,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=None if cu_seqlens is None else max_seqlen,
        max_seqlen_k=None if cu_seqlens is None else max_seqlen,
    )
    # FA4 normalizes some input layouts before allocating the gradients, so
    # their strides can differ from the inputs'; the fake declares contiguous
    # ones, and a compiled graph trusts the declaration.
    return dq.contiguous(), dk.contiguous(), dv.contiguous()


@_flash4_backward.register_fake
def _flash4_backward_fake(
    saved: list[Tensor],
    grad_out: Tensor,
    max_seqlen: int,
    window: int,
    scale: float | None,
) -> tuple[Tensor, Tensor, Tensor]:
    del grad_out, max_seqlen, window, scale
    q, k, v = saved[:3]
    return (
        torch.empty_like(q, memory_format=torch.contiguous_format),
        torch.empty_like(k, memory_format=torch.contiguous_format),
        torch.empty_like(v, memory_format=torch.contiguous_format),
    )


def _flash4_setup(
    ctx: "_Flash4Context",
    inputs: tuple[Tensor, Tensor, Tensor, Tensor | None, int, int, float | None],
    output: tuple[Tensor, Tensor],
) -> None:
    q, k, v, cu_seqlens, ctx.max_seqlen, ctx.window, ctx.scale = inputs
    segments = () if cu_seqlens is None else (cu_seqlens,)
    ctx.save_for_backward(q, k, v, *output, *segments)


def _register_flash4_autograd[FunctionT: Callable[..., object]](
    function: FunctionT,
) -> FunctionT:
    _flash4_forward.register_autograd(function, setup_context=_flash4_setup)
    return function


@_register_flash4_autograd
def _flash4_grad(
    ctx: "_Flash4Context",
    grad_out: Tensor,
    grad_lse: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor, None, None, None, None]:
    # The lse leaves only as the detached max-logit bound, so it has no gradient.
    del grad_lse
    dq, dk, dv = _flash4_backward(
        list(ctx.saved_tensors),
        grad_out,
        ctx.max_seqlen,
        ctx.window,
        ctx.scale,
    )
    return dq, dk, dv, None, None, None, None


@functools.cache
def _interface() -> "_Flash4Interface":
    """Import FA4's CuTe interface and check the entry points this module calls."""
    module = importlib.import_module("flash_attn.cute.interface")
    if not isinstance(module, _Flash4Interface):
        raise TypeError(
            "FA4 must provide flash_attn_func, flash_attn_varlen_func and "
            "_flash_attn_bwd.",
        )
    return module


class _Flash4Context(Protocol):
    max_seqlen: int
    window: int
    scale: float | None
    saved_tensors: tuple[Tensor, ...]

    def save_for_backward(self, *tensors: Tensor) -> None: ...


@runtime_checkable
class _Flash4Interface(Protocol):
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
    ) -> tuple[Tensor, Tensor | None]: ...

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
    ) -> tuple[Tensor, Tensor | None]: ...

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
    ) -> tuple[Tensor, Tensor, Tensor]: ...
