"""Forecast with replaceable attention or recurrent blocks over history patches."""

from __future__ import annotations

from dataclasses import KW_ONLY, field
from typing import Self, override

import math

from configgle import Fig, Makeable
from torch import Tensor, nn

import torch

from priml.baselines.oscillator.data import STATE_CHANNELS
from priml.cost import Cost, cost
from priml.model.attention.attention import Attention
from priml.model.attention.rope import RoPE
from priml.model.custom_types import (
    ChannelsIn,
    ChannelsOut,
    HasResetParameters,
    TensorModule,
    propagate_attr,
)
from priml.model.linear import Linear
from priml.model.min_gru import MinGRU
from priml.model.sequential import Sequential
from priml.model.swiglu import SwiGLU
from priml.model.transformer.block import TransformerBlock


class PatchSequence(nn.Module):
    """Preserve time order while mixing short patches of the observed history."""

    class Config(Fig["PatchSequence"]):
        channels_in: int = -1
        """Flattened observed position/velocity values, supplied by HistoryNetwork."""

        channels_out: int = -1
        """Flattened forecast values, supplied by HistoryNetwork."""

        _: KW_ONLY

        patch_size: int = 4
        """Consecutive timesteps placed in each token."""

        proj_in: Linear.Config = field(
            default_factory=lambda: Linear.Config(channels_out=32, bias=True),
        )
        """Patch projection; owns the internal width and its initialization."""

        block: Makeable[TensorModule] = field(
            default_factory=lambda: Sequential.Config(
                elements=TransformerBlock.Config(
                    attn=Attention.Config(
                        num_heads=4,
                        rope=RoPE.Config(channels_head=8),
                    ),
                    ffn=SwiGLU.Config(channels_hidden=64, bias=True),
                ),
                repeat=2,
            ),
        )
        """Time-mixing stack; an attention or recurrent module may be supplied."""

        proj_out: Linear.Config = field(
            default_factory=lambda: Linear.Config(bias=True),
        )
        """Head over all encoded tokens, preserving their time order."""

        @override
        def finalize(self) -> Self:
            self.proj_in.channels_in = self.patch_size * STATE_CHANNELS
            width = self.proj_in.channels_out
            propagate_attr(self.block, "channels_in", width, protocol=ChannelsIn)
            propagate_attr(self.block, "channels_out", width, protocol=ChannelsOut)
            self.proj_out.channels_in = width * (
                self.channels_in // max(1, self.patch_size * STATE_CHANNELS)
            )
            self.proj_out.channels_out = self.channels_out
            return super().finalize()

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Count patch projection, sequence mixing, and one flattened head.

            Args:
              seq_len: Flattened histories per invocation; HistoryNetwork passes one.
              batch_size: Independent trajectories.
              dtype: Activation dtype; None uses torch's default.
              **kwargs: Forwarded to the child costs.

            Returns:
              cost: Whole invocation, with each child's parameters counted once.

            """
            tokens = self.channels_in // (self.patch_size * STATE_CHANNELS)
            return (
                cost(
                    self.proj_in,
                    seq_len=tokens * seq_len,
                    batch_size=batch_size,
                    dtype=dtype,
                    **kwargs,
                )
                + cost(
                    self.block,
                    seq_len=tokens * seq_len,
                    batch_size=batch_size,
                    dtype=dtype,
                    **kwargs,
                )
                + cost(
                    self.proj_out,
                    seq_len=seq_len,
                    batch_size=batch_size,
                    dtype=dtype,
                    **kwargs,
                )
            )

    def __init__(self, config: Config) -> None:
        super().__init__()
        if min(config.channels_in, config.channels_out, config.patch_size) <= 0:
            raise ValueError("Patch geometry must be positive.")
        if config.channels_in % (config.patch_size * STATE_CHANNELS):
            raise ValueError(
                "The history must contain complete position/velocity patches.",
            )
        if config.proj_out.channels_in != config.proj_in.channels_out * (
            config.channels_in // (config.patch_size * STATE_CHANNELS)
        ):
            raise ValueError("The sequence head does not match the patch width.")
        self.config = config
        self.proj_in = config.proj_in.make()
        self.block = config.block.make()
        self.proj_out = config.proj_out.make()

    @override
    def forward(self, values: Tensor, **kwargs: object) -> Tensor:
        """Map flattened histories to flattened forecast windows."""
        if values.ndim != 2 or values.shape[-1] != self.config.channels_in:
            raise ValueError("The flattened history has the wrong geometry.")
        tokens = values.reshape(
            len(values),
            -1,
            self.config.patch_size * STATE_CHANNELS,
        )
        encoded = self.block(self.proj_in(tokens), **kwargs)
        return self.proj_out(encoded.flatten(start_dim=1))

    def reset_parameters(self) -> None:
        """Reset the patch projection, sequence stack, and forecast head."""
        for child in (self.proj_in, self.block, self.proj_out):
            child.reset_parameters()


class RecurrentSequence(nn.Module):
    """Expose minGRU's sequence output while keeping carry local to each window."""

    class Config(Fig["RecurrentSequence"]):
        channels_in: int = -1
        """Token width supplied by the patch projection."""

        channels_out: int = -1
        """Sequence output width; defaults to the input width."""

        _: KW_ONLY

        block: MinGRU.Config = field(
            default_factory=lambda: MinGRU.Config(num_layers=2),
        )
        """Recurrent stack; owns its depth and highway configuration."""

        @override
        def finalize(self) -> Self:
            if self.channels_out == -1:
                self.channels_out = self.channels_in
            self.block.channels_in = self.channels_in
            if self.block.channels_hidden == -1:
                self.block.channels_hidden = self.channels_out
            return super().finalize()

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Delegate the full recurrent invocation to its owned stack.

            Args:
              seq_len: Tokens processed per trajectory.
              batch_size: Independent trajectories.
              dtype: Activation dtype; None uses torch's default.
              **kwargs: Forwarded to the recurrent stack's cost calculation.

            Returns:
              cost: Full recurrent invocation, counting its parameters once.

            """
            return cost(
                self.block,
                seq_len=seq_len,
                batch_size=batch_size,
                dtype=dtype,
                **kwargs,
            )

    def __init__(self, config: Config) -> None:
        super().__init__()
        if (
            min(config.channels_in, config.channels_out) <= 0
            or config.block.channels_hidden != config.channels_out
        ):
            raise ValueError("The recurrent output width must match the sequence head.")
        self.block = config.block.make()

    def reset_parameters(self) -> None:
        """Reset the owned recurrent stack through its existing initializer."""
        for child in self.block.modules():
            if not list(child.children()) and isinstance(child, HasResetParameters):
                child.reset_parameters()

    @override
    def forward(self, values: Tensor, **kwargs: object) -> Tensor:
        """Start a fresh carry and return one output per observed token."""
        del kwargs
        return self.block(values)[0]


class RbfKernel:
    """Compare histories with a fixed smooth radial kernel."""

    class Config(Fig["RbfKernel"]):
        length_scale: float = 1.0
        """Distance scale after training-only feature standardization."""

    def __init__(self, config: Config) -> None:
        if (
            math.isnan(config.length_scale)
            or config.length_scale <= 0
            or config.length_scale >= math.inf
        ):
            raise ValueError("The kernel length scale must be finite and positive.")
        self.config = config

    def __call__(self, left: Tensor, right: Tensor) -> Tensor:
        """Return a kernel matrix without allocating pairwise feature differences."""
        distance = (
            left.square().sum(dim=1, keepdim=True)
            + right.square().sum(dim=1)[None]
            - 2 * left @ right.T
        ).clamp_min(0)
        return (-distance / (2 * left.shape[1] * self.config.length_scale**2)).exp()
