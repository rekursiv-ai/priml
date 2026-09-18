"""Qwen3.5 gated delta attention with serializable convolution and recurrent state."""

from dataclasses import field
from typing import TypeGuard, override

import math

from configgle import Fig, Makeable, Makes
from torch import Tensor, nn

import torch
import torch.nn.functional

from priml.cost import (
    Cost,
    elementwise_cost,
)
from priml.math.gated_delta_rule import (
    chunk_gated_delta_rule,
    recurrent_gated_delta_rule,
)
from priml.model.attention.gated_delta_net import GatedDeltaNet
from priml.model.custom_types import TensorModule
from priml.model.init import InitFn
from priml.model.norm import RMSNorm


class Qwen35RMSNormGated(nn.Module):
    """Normalize per value head, then apply the Qwen3.5 fp32 SiLU gate."""

    class Config(Fig["Qwen35RMSNormGated"]):
        channels_in: int = -1
        """Per-head value width, inherited from the attention layer."""

        eps: float = 1e-6
        """Epsilon added to the fp32 mean square."""

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Count an affine RMSNorm, a SiLU on the gate, and their product.

            Per row: square, mean, epsilon and rsqrt, normalize, scale, SiLU
            (a sigmoid and a product), and the gating product, around one sum
            over the row. The adjoint pulls back through the product, SiLU's
            saved-sigmoid derivative, and the affine norm; the scale's gradient
            reduces over rows.

            Args:
              seq_len: Tokens per sequence.
              batch_size: Sequences per step.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus, forwarded to every child.

            Returns:
              cost: Per-token cost of this module.

            """
            width = self.channels_in
            norm = RMSNorm.Config()
            norm.channels_in = width
            norm.elementwise_affine = True
            return norm.cost(
                seq_len=seq_len,
                batch_size=batch_size,
                dtype=dtype,
                **kwargs,
            ) + elementwise_cost(
                primal=6 * width,
                adjoint=7 * width,
                channels=width,
                inputs=3,
                outputs=2,
                adjoint_inputs=6,
                adjoint_outputs=3,
                dtype=dtype,
                rows=seq_len * batch_size,
            )

    def __init__(self, config: Config) -> None:
        if not math.isfinite(config.eps) or config.eps <= 0:
            raise ValueError("eps must be finite and positive.")
        super().__init__()
        self.eps = config.eps
        self.weight = nn.Parameter(torch.ones(config.channels_in))

    def reset_parameters(self) -> None:
        """Initialize the learned scale to one."""
        nn.init.ones_(self.weight)

    @override
    def forward(self, x: Tensor, **kwargs: object) -> Tensor:
        """Apply normalization and gating with reference rounding boundaries.

        Args:
          x: Per-head values whose final axis is the configured head width.
          **kwargs: Required ``gate`` tensor with x's shape.

        Returns:
          output: Gated normalized values with x's shape and dtype.

        """
        gate = kwargs["gate"]
        assert isinstance(gate, Tensor)
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x = self.weight * x.to(dtype)
        return (x * torch.nn.functional.silu(gate.float())).to(dtype)


def _init_decay(weight: Tensor) -> None:
    with torch.no_grad():
        weight.copy_(torch.empty_like(weight).uniform_(0.01, 16).log_())


class Qwen35GatedDeltaNet(GatedDeltaNet):
    """Reuse native projections with Qwen3.5 arithmetic and incremental state."""

    class Config(Makes["Qwen35GatedDeltaNet"], GatedDeltaNet.Config):
        norm: Makeable[TensorModule] = field(default_factory=Qwen35RMSNormGated.Config)
        """Complete post-delta transform; the default normalizes then gates."""

        init_decay: InitFn = _init_decay
        """Initialize log decay rates from the public Qwen3.5 reference range."""

        @override
        def _output_gate_cost(self, *, rows: float, dtype: torch.dtype | None) -> Cost:
            """Leave the complete post-delta transform to the injected norm."""
            del rows, dtype
            return Cost()

    @override
    def forward(self, x: Tensor, **kwargs: object) -> Tensor:
        """Apply delta attention, updating caller-owned cache tensors when present.

        Args:
          x: Hidden states shaped ``[..., sequence, channels]``.
          **kwargs: Optional ``cache`` dictionary holding ``conv_state`` and
            ``recurrent_state``, and ``attention_mask`` whose trailing positions
            match x. Other messages from the enclosing block are unused.

        Returns:
          output: Hidden states with the same shape as x.

        """
        cache = _validated_cache(kwargs.get("cache"))
        attention_mask = kwargs.get("attention_mask")
        if attention_mask is not None and not isinstance(attention_mask, Tensor):
            raise TypeError("attention_mask must be a Tensor or None.")
        shape = x.shape
        sequence = shape[-2]
        x = x.reshape(-1, sequence, shape[-1])
        if attention_mask is not None:
            mask = attention_mask[..., -sequence:].reshape(*x.shape[:-1], 1)
            x = (x * mask).to(x.dtype)
        batch = x.shape[0]
        qkv = self.proj_qkv(x).transpose(1, 2)
        z = self.proj_z(x).reshape(batch, sequence, -1, self.channels_v_head)
        beta = self.proj_b(x).sigmoid()
        a = self.proj_a(x)
        warm = cache is not None and "recurrent_state" in cache
        self._validate_cache_state(cache, qkv=qkv)
        qkv = self._convolve(qkv, cache=cache, warm=warm).transpose(1, 2)
        key_width = self.num_heads_k * self.channels_k_head
        value_width = self.num_heads_v * self.channels_v_head
        query, key, value = qkv.split([key_width, key_width, value_width], dim=-1)
        query = query.reshape(batch, sequence, -1, self.channels_k_head)
        key = key.reshape(batch, sequence, -1, self.channels_k_head)
        value = value.reshape(batch, sequence, -1, self.channels_v_head)
        g = -self.A_log.float().exp() * torch.nn.functional.softplus(
            a.float() + self.dt_bias,
        )
        repeats = self.num_heads_v // self.num_heads_k
        if repeats > 1:
            query = query.repeat_interleave(repeats, dim=2)
            key = key.repeat_interleave(repeats, dim=2)
        kernel = (
            recurrent_gated_delta_rule
            if warm and sequence == 1
            else chunk_gated_delta_rule
        )
        output, state = kernel(
            query=query,
            key=key,
            value=value,
            g=g,
            beta=beta,
            initial_state=cache["recurrent_state"]
            if cache is not None and warm
            else None,
            output_final_state=cache is not None,
            use_qk_l2norm_in_kernel=True,
        )
        if cache is not None:
            if state is None:
                raise ValueError("Expected state is not None.")
            cache["recurrent_state"] = state
        output = self.norm(
            output.reshape(-1, self.channels_v_head),
            gate=z.reshape(-1, self.channels_v_head),
        )
        return self.proj_out(output.reshape(batch, sequence, value_width)).reshape(
            shape,
        )

    def alloc_kv_cache(
        self,
        *,
        batch: int | tuple[int, ...],
        max_seq: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> dict[str, Tensor]:
        """Allocate a lazy cache, materialized from the first input's placement.

        Args:
          batch: Ignored until the first input materializes the cache.
          max_seq: Ignored because the delta cache grows from the input shape.
          device: Ignored until the first input materializes the cache.
          dtype: Ignored until the first input materializes the cache.

        Returns:
          cache: Empty mutable delta cache.

        """
        del batch, max_seq, device, dtype
        return {}

    def forward_cached(
        self,
        x: Tensor,
        *,
        cache: dict[str, Tensor],
        **kwargs: object,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Apply a cached chunk and return the same cache object.

        Args:
          x: Hidden states shaped ``[..., sequence, channels]``.
          cache: Mutable delta cache updated in place.
          **kwargs: Messages forwarded to delta attention.

        Returns:
          output: Hidden states with x's shape.
          cache: The updated input cache.

        """
        return self.forward(x, cache=cache, **kwargs), cache

    def _validate_cache_state(
        self,
        cache: dict[str, Tensor] | None,
        *,
        qkv: Tensor,
    ) -> None:
        """Reject materialized cache tensors incompatible with the current input."""
        if cache is None or not cache:
            return
        conv_state = cache["conv_state"]
        recurrent_state = cache["recurrent_state"]
        if (
            tuple(conv_state.shape) != (*qkv.shape[:-1], self.conv1d.kernel_size[0])
            or conv_state.dtype != qkv.dtype
            or conv_state.device != qkv.device
            or conv_state.layout != torch.strided
            or tuple(recurrent_state.shape)
            != (
                qkv.shape[0],
                self.num_heads_v,
                self.channels_k_head,
                self.channels_v_head,
            )
            or recurrent_state.dtype != torch.float32
            or recurrent_state.device != qkv.device
            or recurrent_state.layout != torch.strided
        ):
            raise ValueError("cache state is incompatible with input.")

    def _convolve(
        self,
        qkv: Tensor,
        *,
        cache: dict[str, Tensor] | None,
        warm: bool,
    ) -> Tensor:
        sequence = qkv.shape[-1]
        kernel_size = self.conv1d.kernel_size[0]
        padding = kernel_size - 1
        if cache is not None:
            if warm:
                qkv = torch.cat([cache["conv_state"], qkv], dim=-1)
                if sequence == 1:
                    padding = 0
            elif sequence < kernel_size:
                qkv = torch.nn.functional.pad(qkv, (kernel_size - sequence, 0))
            cache["conv_state"] = qkv[..., -kernel_size:].clone()
        output = torch.nn.functional.conv1d(
            qkv.to(self.conv1d.weight.dtype),
            weight=self.conv1d.weight,
            bias=self.conv1d.bias,
            padding=padding,
            groups=qkv.shape[-2],
        )
        if padding:
            output = output[..., : qkv.shape[-1]]
        if warm and sequence == 1:
            # SiLU selects different CPU vector kernels for the extra context.
            # Decode must slice first to reproduce the reference rounding.
            output = output[..., -sequence:]
        return torch.nn.functional.silu(output)[..., -sequence:].to(qkv.dtype)


def _validated_cache(value: object) -> dict[str, Tensor] | None:
    """Return a mutable delta cache after validating its complete state boundary."""
    if value is None:
        return None
    if _is_empty_cache(value) or _is_complete_cache(value):
        return value
    raise TypeError(
        "cache must be empty or contain only Tensor conv_state and recurrent_state.",
    )


def _is_empty_cache(value: object) -> TypeGuard[dict[str, Tensor]]:
    """Return whether value is a mutable cache without materialized state."""
    return isinstance(value, dict) and not value


def _is_complete_cache(value: object) -> TypeGuard[dict[str, Tensor]]:
    """Return whether value contains exactly the native delta cache state pair."""
    return (
        isinstance(value, dict)
        and value.keys() == {"conv_state", "recurrent_state"}
        and isinstance(value["conv_state"], Tensor)
        and isinstance(value["recurrent_state"], Tensor)
    )
