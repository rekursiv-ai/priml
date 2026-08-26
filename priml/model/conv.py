"""Convolutional layers with configurable initialization."""

from __future__ import annotations

from dataclasses import KW_ONLY
from typing import Self, override

from configgle import Fig
from torch import Tensor, nn

import torch

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
