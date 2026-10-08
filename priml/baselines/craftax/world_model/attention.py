"""The world model's global attention: causal within the episode segments of a window.

The global transformer attends causally within each episode segment of a
packed window and never across segments. ``PackedBatch.cu_seqlens`` names the
segment boundaries of the flattened ``[B·t_g]`` batch, and every window ends on
a boundary, so no segment crosses windows. priml's varlen kernels read it:
``Flash4Varlen`` (FlashAttention 4, Linux and CUDA) and ``SdpaVarlen``, the
reference that runs anywhere. The kernel is an explicit config slot, never
chosen from the environment: CPU runs and tests use ``SdpaVarlen``, and GPU
experiments set ``Flash4Varlen``.

Each forward also records its largest attention logit (scaled ``q·k``) for
logging, as a detached device scalar, so reading it costs no host sync until
the logger asks: exact under ``SdpaVarlen``, and under ``Flash4Varlen`` the
largest log-normalizer, at most ``log(t_g)`` above it. The frame encoder and
local decoder use SDPA on fixed-length rows: priml's ``Attention``, for both
self-attention and cross-attention.
"""

from dataclasses import KW_ONLY, field, replace
from typing import Protocol, override

from configgle import Makeable, Makes
from torch import Tensor

import torch

from priml.cost import Cost, cost, resolve_dtype
from priml.model.attention.attention import AttentionProjections
from priml.model.attention.kernel import SdpaVarlen
from priml.model.attention.rope import RoPE


class VarlenKernel(Protocol):
    """Causal attention within the segments ``cu_seqlens`` names, keeping its max logit."""

    max_logit: Tensor | None
    """The last forward's max logit when it was asked to record one."""

    def __call__(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        *,
        cu_seqlens: Tensor,
        record_max_logit: bool,
    ) -> Tensor:
        """Attend ``[B, T, H, D]`` queries to ``[B, T, H_kv, D]`` keys and values."""
        ...


def row_cu_seqlens(rows: int, length: int, *, device: torch.device) -> Tensor:
    """Return ``cu_seqlens`` making each of ``rows`` windows one segment."""
    return torch.arange(rows + 1, device=device, dtype=torch.int32) * length


class VarlenAttention(AttentionProjections):
    """Causal self-attention within packed segments, with built-in GQA.

    Parameters match priml's ``Attention`` one for one, so its weight
    remapping (``qwen3.remap_hf_state_dict``) applies unchanged. Keys and values
    stay at ``num_heads_kv`` heads; the kernel does the grouping.

    Attributes:
      max_logit: The last forward's max attention logit (see the module
        docstring), a detached float32 scalar; None before the first forward.

    """

    class Config(
        Makes["VarlenAttention"],
        AttentionProjections.Config,
        kw_only=False,
    ):
        """``causal`` is fixed True; the kernel slot picks FA4 or the reference."""

        _: KW_ONLY

        causal: bool = True
        """Always True; the varlen kernels are causal and nothing else is implemented."""

        attn_kernel: Makeable[VarlenKernel] = field(default_factory=SdpaVarlen.Config)
        """Varlen kernel: ``SdpaVarlen`` anywhere, ``Flash4Varlen`` on Linux CUDA."""

        @override
        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            memory_len: int = -1,
            **kwargs: object,
        ) -> Cost:
            """Add the kernel's products and the per-token KV cache, as ``Attention``'s.

            The kernel is costed as one segment per row, an upper bound.

            Args:
              seq_len: Tokens per row.
              batch_size: Rows per step.
              dtype: Activation dtype; ``None`` is torch's default.
              memory_len: -1: a varlen attention attends to no memory.
              **kwargs: The open bus, forwarded to every child.

            Returns:
              cost: Whole-invocation cost over ``seq_len`` and ``batch_size``.

            """
            if memory_len >= 0:
                raise ValueError("A varlen attention attends to no memory.")
            kernel = cost(
                self.attn_kernel,
                seq_len=seq_len,
                batch_size=batch_size,
                dtype=dtype,
                num_heads=self.num_heads,
                channels_head=self.channels_head,
                dropout_p=self.dropout,
            )
            projections = super().cost(
                seq_len=seq_len,
                batch_size=batch_size,
                dtype=dtype,
                memory_len=memory_len,
                **kwargs,
            )
            itemsize = resolve_dtype(dtype).itemsize
            return replace(
                projections + kernel,
                bytes_state=itemsize * 2 * self.num_heads_kv * self.channels_head,
            )

    def __init__(self, config: Config) -> None:
        if not config.causal or config.dropout:
            raise ValueError("VarlenAttention is causal and has no dropout.")
        super().__init__(config)
        self.attn_kernel = config.attn_kernel.make()
        self.max_logit: Tensor | None = None

    @override
    def forward(
        self,
        x: Tensor,
        *,
        positions: Tensor,
        cu_seqlens: Tensor,
        **kwargs: object,
    ) -> Tensor:
        """Attend within the segments of a packed batch.

        Args:
          x: Inputs ``[B, T, channels_in]``.
          positions: RoPE position of each input within its segment, ``[B, T]``.
          cu_seqlens: Segment boundaries of the flattened ``[B·T]`` batch, int32.
          **kwargs: Open message bus; ignored.

        Returns:
          out: ``[B, T, channels_in]``.

        """
        del kwargs
        if self.split_qkv_projection:
            q, k, v = self.split_qkv(x)
        else:
            q, k, v = self.proj_qkv(x).split(
                [self.num_heads, self.num_heads_kv, self.num_heads_kv],
                dim=-2,
            )
        if self.norm_q is not None:
            q = self.norm_q(q)
        if self.norm_k is not None:
            k = self.norm_k(k)
        if self.rope is not None:
            cos, sin = self.rope(positions.unsqueeze(-1))
            q, k = RoPE.rotate(q, k, cos, sin)
        out = self.attn_kernel(
            q,
            k,
            v.contiguous(),
            cu_seqlens=cu_seqlens,
            record_max_logit=True,
        )
        self.max_logit = self.attn_kernel.max_logit
        out = out.flatten(-2)
        if self.norm_out is not None:
            out = self.norm_out(out)
        return self.proj_out(out)
