"""Convolutional layers with configurable initialization."""

from __future__ import annotations

from dataclasses import KW_ONLY
from typing import Self, override

import math

from configgle import Fig
from torch import Tensor, nn

import torch

from priml.cost import (
    Cost,
    matmul_cost,
)
from priml.model.custom_types import DepthIndex
from priml.model.init import InitFn, call_init, kaiming_uniform


class Conv1d(nn.Conv1d):
    """Conv1d with configurable init."""

    class Config(Fig["Conv1d"], kw_only=False):
        channels_in: int = -1
        """Number of input channels (-1 to infer from channels_out)."""

        channels_out: int = -1
        """Number of output channels (-1 to infer from channels_in)."""

        _: KW_ONLY

        kernel_size: int = 3
        """Size of the convolving kernel."""

        stride: int = 1
        """Stride of the convolution."""

        padding: int | str = "same"
        """Padding added to input ("same", "valid", or int)."""

        dilation: int = 1
        """Spacing between kernel elements."""

        groups: int = 1
        """Number of blocked connections from input to output channels."""

        bias: bool = False
        """Include bias in the convolution."""

        depth_index: DepthIndex = ()
        """Block depth index for depth-scaled init (-1 = no scaling)."""

        device: torch.device | str | None = None
        """Device for parameter allocation."""

        dtype: torch.dtype | None = None
        """Data type for parameters."""

        init_weight: InitFn = kaiming_uniform
        """Weight initialization function."""

        init_bias: InitFn = nn.init.zeros_
        """Bias initialization function."""

        @override
        def finalize(self) -> Self:
            if self.channels_in == -1:
                self.channels_in = self.channels_out
            if self.channels_out == -1:
                self.channels_out = self.channels_in
            return super().finalize()

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Price one output position; the sharing rows are output positions.

            Args:
              seq_len: Tokens per sequence.
              batch_size: Sequences per step.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus, forwarded to every child.

            Returns:
              cost: Per-token cost of this module.

            """
            del kwargs
            return conv_cost(
                channels_in=self.channels_in,
                channels_out=self.channels_out,
                kernel_size=self.kernel_size,
                ndim=1,
                groups=self.groups,
                bias=self.bias,
                rows=seq_len * batch_size,
                dtype=self.dtype if self.dtype is not None else dtype,
            )

    def __init__(self, config: Config) -> None:
        self.depth_index = config.depth_index
        self._init_weight = config.init_weight
        self._init_bias = config.init_bias
        super().__init__(
            in_channels=config.channels_in,
            out_channels=config.channels_out,
            kernel_size=config.kernel_size,
            stride=config.stride,
            padding=config.padding,
            dilation=config.dilation,
            groups=config.groups,
            bias=config.bias,
            device=config.device,
            dtype=config.dtype,
        )

    @override
    def reset_parameters(self) -> None:
        call_init(self._init_weight, self.weight, depth_index=self.depth_index)
        if self.bias is not None:
            call_init(self._init_bias, self.bias, depth_index=self.depth_index)

    @override
    def forward(self, input: Tensor, **kwargs: object) -> Tensor:
        del kwargs
        return super().forward(input)


class Conv2d(nn.Conv2d):
    """Conv2d with configurable init."""

    class Config(Fig["Conv2d"], kw_only=False):
        channels_in: int = -1
        """Number of input channels (-1 to infer from channels_out)."""

        channels_out: int = -1
        """Number of output channels (-1 to infer from channels_in)."""

        _: KW_ONLY

        kernel_size: int | tuple[int, int] = 3
        """Size of the convolving kernel."""

        stride: int | tuple[int, int] = 1
        """Stride of the convolution."""

        padding: int | tuple[int, int] | str = "same"
        """Padding added to input ("same", "valid", or int/tuple)."""

        dilation: int | tuple[int, int] = 1
        """Spacing between kernel elements."""

        groups: int = 1
        """Number of blocked connections from input to output channels."""

        bias: bool = False
        """Include bias in the convolution."""

        depth_index: DepthIndex = ()
        """Block depth index for depth-scaled init (-1 = no scaling)."""

        device: torch.device | str | None = None
        """Device for parameter allocation."""

        dtype: torch.dtype | None = None
        """Data type for parameters."""

        init_weight: InitFn = kaiming_uniform
        """Weight initialization function."""

        init_bias: InitFn = nn.init.zeros_
        """Bias initialization function."""

        @override
        def finalize(self) -> Self:
            if self.channels_in == -1:
                self.channels_in = self.channels_out
            if self.channels_out == -1:
                self.channels_out = self.channels_in
            return super().finalize()

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Price one output position; the sharing rows are output positions.

            Args:
              seq_len: Tokens per sequence.
              batch_size: Sequences per step.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus, forwarded to every child.

            Returns:
              cost: Per-token cost of this module.

            """
            del kwargs
            return conv_cost(
                channels_in=self.channels_in,
                channels_out=self.channels_out,
                kernel_size=self.kernel_size,
                ndim=2,
                groups=self.groups,
                bias=self.bias,
                rows=seq_len * batch_size,
                dtype=self.dtype if self.dtype is not None else dtype,
            )

    def __init__(self, config: Config) -> None:
        self.depth_index = config.depth_index
        self._init_weight = config.init_weight
        self._init_bias = config.init_bias
        super().__init__(
            in_channels=config.channels_in,
            out_channels=config.channels_out,
            kernel_size=config.kernel_size,
            stride=config.stride,
            padding=config.padding,
            dilation=config.dilation,
            groups=config.groups,
            bias=config.bias,
            device=config.device,
            dtype=config.dtype,
        )

    @override
    def reset_parameters(self) -> None:
        call_init(self._init_weight, self.weight, depth_index=self.depth_index)
        if self.bias is not None:
            call_init(self._init_bias, self.bias, depth_index=self.depth_index)

    @override
    def forward(self, input: Tensor, **kwargs: object) -> Tensor:
        del kwargs
        return super().forward(input)


class Conv3d(nn.Conv3d):
    """Conv3d with configurable init."""

    class Config(Fig["Conv3d"], kw_only=False):
        channels_in: int = -1
        """Number of input channels (-1 to infer from channels_out)."""

        channels_out: int = -1
        """Number of output channels (-1 to infer from channels_in)."""

        _: KW_ONLY

        kernel_size: int | tuple[int, int, int] = 3
        """Size of the convolving kernel."""

        stride: int | tuple[int, int, int] = 1
        """Stride of the convolution."""

        padding: int | tuple[int, int, int] | str = "same"
        """Padding added to input ("same", "valid", or int/tuple)."""

        dilation: int | tuple[int, int, int] = 1
        """Spacing between kernel elements."""

        groups: int = 1
        """Number of blocked connections from input to output channels."""

        bias: bool = False
        """Include bias in the convolution."""

        depth_index: DepthIndex = ()
        """Block depth index for depth-scaled init (-1 = no scaling)."""

        device: torch.device | str | None = None
        """Device for parameter allocation."""

        dtype: torch.dtype | None = None
        """Data type for parameters."""

        init_weight: InitFn = kaiming_uniform
        """Weight initialization function."""

        init_bias: InitFn = nn.init.zeros_
        """Bias initialization function."""

        @override
        def finalize(self) -> Self:
            if self.channels_in == -1:
                self.channels_in = self.channels_out
            if self.channels_out == -1:
                self.channels_out = self.channels_in
            return super().finalize()

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Price one output position; the sharing rows are output positions.

            Args:
              seq_len: Tokens per sequence.
              batch_size: Sequences per step.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus, forwarded to every child.

            Returns:
              cost: Per-token cost of this module.

            """
            del kwargs
            return conv_cost(
                channels_in=self.channels_in,
                channels_out=self.channels_out,
                kernel_size=self.kernel_size,
                ndim=3,
                groups=self.groups,
                bias=self.bias,
                rows=seq_len * batch_size,
                dtype=self.dtype if self.dtype is not None else dtype,
            )

    def __init__(self, config: Config) -> None:
        self.depth_index = config.depth_index
        self._init_weight = config.init_weight
        self._init_bias = config.init_bias
        super().__init__(
            in_channels=config.channels_in,
            out_channels=config.channels_out,
            kernel_size=config.kernel_size,
            stride=config.stride,
            padding=config.padding,
            dilation=config.dilation,
            groups=config.groups,
            bias=config.bias,
            device=config.device,
            dtype=config.dtype,
        )

    @override
    def reset_parameters(self) -> None:
        call_init(self._init_weight, self.weight, depth_index=self.depth_index)
        if self.bias is not None:
            call_init(self._init_bias, self.bias, depth_index=self.depth_index)

    @override
    def forward(self, input: Tensor, **kwargs: object) -> Tensor:
        del kwargs
        return super().forward(input)


def conv_cost(
    *,
    channels_in: int,
    channels_out: int,
    kernel_size: int | tuple[int, ...],
    ndim: int,
    groups: int,
    bias: bool,
    rows: float = 1,
    dtype: torch.dtype | None = None,
) -> Cost:
    """Price a convolution at one OUTPUT position.

    Every output element is a dot product over its receptive field, so a
    convolution is the matmul ``[channels_in / groups * prod(kernel_size)] ->
    [channels_out]`` applied once per output position. The bus carries no
    spatial extent, so the "token" here is one output position; a caller with
    a grid multiplies by its size. Stride, padding, and dilation move where the
    products land, not how many there are per position. Traffic counts each
    group's receptive-field operand independently at every output position;
    overlapping patches are reread, with no im2col workspace or cache model.
    This is logical operand I/O, not a lower bound on whole-convolution HBM.

    Args:
      channels_in: Input channels, before grouping.
      channels_out: Output channels.
      kernel_size: One extent for every axis, or one per axis.
      ndim: Spatial rank, which a scalar ``kernel_size`` is raised to.
      groups: Blocked connections; each output sees ``channels_in / groups``.
      bias: Whether a bias vector is owned.
      rows: Output positions sharing the weights and bias.
      dtype: Element type of every operand; ``None`` is torch's default.

    Returns:
      cost: The matmul's price; parameters equal the weight plus bias numel.

    """
    taps = math.prod(
        kernel_size if isinstance(kernel_size, tuple) else (kernel_size,) * ndim,
    )
    return matmul_cost(
        channels_in=channels_in // groups * taps,
        channels_out=channels_out // groups,
        bias=bias,
        rows=rows,
        dtype=dtype,
    ).tile(groups, copies=groups)
