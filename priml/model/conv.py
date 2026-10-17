"""Convolutional layers with configurable initialization."""

from __future__ import annotations

from dataclasses import KW_ONLY, field
from typing import Self, override

import math

from configgle import Fig
from torch import Tensor, nn
from torch.nn import functional

import torch

from priml.cost import Cost, cost, elementwise_cost, nbytes, resolve_dtype
from priml.math.custom_types import TensorFn
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
        """Number of blocked connections; -1 selects depthwise channel groups."""

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

        init_bias: InitFn | None = nn.init.zeros_
        """Bias initializer; ``None`` preserves torch's fan-in uniform draw."""

        @override
        def finalize(self) -> Self:
            if self.channels_in == -1:
                self.channels_in = self.channels_out
            if self.channels_out == -1:
                self.channels_out = self.channels_in
            if self.groups == -1:
                if self.channels_in != self.channels_out:
                    raise ValueError(
                        "Depthwise groups require matching input and output "
                        f"channels, got {self.channels_in} and {self.channels_out}.",
                    )
                self.groups = self.channels_in
            return super().finalize()

        def cost(
            self,
            *,
            input_grid: int | tuple[int, ...],
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost one invocation; the output grid follows from the input grid.

            Args:
              input_grid: Spatial input extents; a scalar is the single axis.
              batch_size: Sequences per step.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus, forwarded to every child.

            Returns:
              cost: Integer FLOPs and logical bytes for the complete invocation.

            """
            del kwargs
            return conv_cost(
                channels_in=self.channels_in,
                channels_out=self.channels_out,
                kernel_size=self.kernel_size,
                ndim=1,
                groups=self.groups,
                bias=self.bias,
                input_grid=(input_grid,) if isinstance(input_grid, int) else input_grid,
                batch_size=batch_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
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
            if self._init_bias is None:
                fan_in = self.weight.shape[1] * math.prod(self.weight.shape[2:])
                bound = fan_in**-0.5
                nn.init.uniform_(self.bias, -bound, bound)
            else:
                call_init(self._init_bias, self.bias, depth_index=self.depth_index)

    @override
    def forward(self, input: Tensor, **kwargs: object) -> Tensor:
        del kwargs
        return super().forward(input)


class SequenceConv1d(nn.Module):
    """Apply a configurable Conv1d to the sequence axis of [..., S, C].

    The wrapper's channel fields define its interface to parent configs; the
    nested convolution's explicit widths must agree with them.
    """

    class Config(Fig["SequenceConv1d"], kw_only=False):
        """Expose channel widths to parent configs and configure the child."""

        channels_in: int = -1
        """Input width exposed to parent configs; -1 infers from the child or output."""

        channels_out: int = -1
        """Output interface width; set here when the wrapper changes hidden width."""

        _: KW_ONLY

        conv: Conv1d.Config = field(default_factory=Conv1d.Config)
        """Convolution applied after moving channels to the channel-first axis."""

        act: TensorFn = functional.silu
        """Activation applied after cropping convolution output to input length."""

        @override
        def finalize(self) -> Self:
            """Finalize the wrapper and child channel widths.

            Returns:
              config: This config with wrapper and child channel widths finalized.

            """
            if self.channels_in == -1 and self.conv.channels_in != -1:
                self.channels_in = self.conv.channels_in
            if self.channels_out == -1 and self.conv.channels_out != -1:
                self.channels_out = self.conv.channels_out
            if self.channels_in == -1:
                self.channels_in = self.channels_out
            if self.channels_out == -1:
                self.channels_out = self.channels_in
            if self.conv.channels_in == -1:
                self.conv.channels_in = self.channels_in
            if self.conv.channels_out == -1:
                self.conv.channels_out = self.channels_out
            return super().finalize()

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost the convolution output and its cropped sequence activation.

            Args:
              seq_len: Input sequence positions.
              batch_size: Independent sequences in the invocation.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: Unread cost bus entries.

            Returns:
              cost: Convolution and activation cost for one invocation.

            """
            del kwargs
            _sequence_conv_output_length(
                seq_len,
                kernel_size=self.conv.kernel_size,
                stride=self.conv.stride,
                padding=self.conv.padding,
                dilation=self.conv.dilation,
            )
            convolution = cost(
                self.conv,
                input_grid=seq_len,
                batch_size=batch_size,
                dtype=dtype,
            )
            elements = seq_len * batch_size * self.conv.channels_out
            if self.act is functional.silu:
                activation = elementwise_cost(
                    primal=5 * elements,
                    adjoint=5 * elements,
                    channels=self.conv.channels_out,
                    rows=seq_len * batch_size,
                    dtype=dtype,
                )
            else:
                activation = cost(self.act, channels=elements, dtype=dtype)
            return convolution + activation

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.conv: Conv1d = config.conv.make()
        if (
            self.conv.in_channels != config.channels_in
            or self.conv.out_channels != config.channels_out
        ):
            raise ValueError(
                f"SequenceConv1d channels_in/channels_out ({config.channels_in}, "
                f"{config.channels_out}) do not match child Conv1d widths "
                f"({self.conv.in_channels}, {self.conv.out_channels}).",
            )
        self.act = config.act

    def reset_parameters(self) -> None:
        """Reset the child convolution."""
        self.conv.reset_parameters()

    @override
    def forward(self, input: Tensor, **kwargs: object) -> Tensor:
        length = input.shape[-2]
        leading_shape = input.shape[:-2]
        channels_in = input.shape[-1]
        _sequence_conv_output_length(
            length,
            kernel_size=self.conv.kernel_size,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
        )
        batch = math.prod(leading_shape) if leading_shape else 1
        flattened = input.reshape(batch, length, channels_in)
        mixed = self.conv(
            flattened.transpose(-1, -2).to(self.conv.weight.dtype),
            **kwargs,
        )[..., :length]
        output = self.act(mixed).transpose(-1, -2).to(input.dtype)
        return output.reshape(*leading_shape, length, self.conv.out_channels)


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
            input_grid: int | tuple[int, ...],
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost one invocation; the output grid follows from the input grid.

            Args:
              input_grid: Spatial input extents; a scalar broadcasts to both axes.
              batch_size: Sequences per step.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus, forwarded to every child.

            Returns:
              cost: Integer FLOPs and logical bytes for the complete invocation.

            """
            del kwargs
            return conv_cost(
                channels_in=self.channels_in,
                channels_out=self.channels_out,
                kernel_size=self.kernel_size,
                ndim=2,
                groups=self.groups,
                bias=self.bias,
                input_grid=(input_grid,) * 2
                if isinstance(input_grid, int)
                else input_grid,
                batch_size=batch_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
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
            input_grid: int | tuple[int, ...],
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost one invocation; the output grid follows from the input grid.

            Args:
              input_grid: Spatial input extents; a scalar broadcasts to all axes.
              batch_size: Sequences per step.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus, forwarded to every child.

            Returns:
              cost: Integer FLOPs and logical bytes for the complete invocation.

            """
            del kwargs
            return conv_cost(
                channels_in=self.channels_in,
                channels_out=self.channels_out,
                kernel_size=self.kernel_size,
                ndim=3,
                groups=self.groups,
                bias=self.bias,
                input_grid=(input_grid,) * 3
                if isinstance(input_grid, int)
                else input_grid,
                batch_size=batch_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
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


def conv_output_grid(
    input_grid: tuple[int, ...],
    *,
    kernel_size: int | tuple[int, ...],
    stride: int | tuple[int, ...],
    padding: int | str | tuple[int, ...],
    dilation: int | tuple[int, ...],
) -> tuple[int, ...]:
    """Return the per-axis output grid torch's convolution writes.

    ``padding="same"`` keeps every input position as an output position, and
    ``"valid"`` is zero padding. Any other string is rejected.

    Args:
      input_grid: Per-axis spatial input extents, without batch or channels.
      kernel_size: One extent for every axis, or one per axis.
      stride: Window step, scalar or per axis.
      padding: ``"same"``, ``"valid"``, an int, or one per axis.
      dilation: Kernel spacing, scalar or per axis.

    Returns:
      grid: Per-axis output extents.

    """
    ndim = len(input_grid)
    if padding == "same":
        return input_grid
    pad_value = 0 if padding == "valid" else padding
    if isinstance(pad_value, str):
        raise TypeError(f"Unsupported convolution padding: {pad_value!r}.")
    taps = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,) * ndim
    steps = stride if isinstance(stride, tuple) else (stride,) * ndim
    spacings = dilation if isinstance(dilation, tuple) else (dilation,) * ndim
    pads = pad_value if isinstance(pad_value, tuple) else (pad_value,) * ndim
    return tuple(
        (size + 2 * pad - spacing * (tap - 1) - 1) // step + 1
        for size, pad, spacing, tap, step in zip(
            input_grid,
            pads,
            spacings,
            taps,
            steps,
            strict=True,
        )
    )


def conv_cost(
    *,
    channels_in: int,
    channels_out: int,
    kernel_size: int | tuple[int, ...],
    ndim: int,
    groups: int,
    bias: bool,
    input_grid: tuple[int, ...],
    batch_size: int,
    stride: int | tuple[int, ...] = 1,
    padding: int | str | tuple[int, ...] = "same",
    dilation: int | tuple[int, ...] = 1,
    dtype: torch.dtype | None = None,
    input_grad: bool = True,
    weight_grad: bool = True,
    bias_grad: bool = True,
) -> Cost:
    """Cost one complete convolution invocation and its dense dot products.

    The output grid is derived from ``input_grid`` and the window geometry, so
    stride, padding, and dilation enter only through the positions they leave;
    a caller never pre-divides. The primal reads input and weight tensors once
    and writes the output once. The joint backward reads the incoming gradient
    once, the input only for a weight gradient, and the weight only for an input
    gradient. Each requested gradient is written once. Bias operands belong to
    elementwise/reduction. These are logical tensor bytes, not measured cache or
    HBM transactions.

    Args:
      channels_in: Input channels, before grouping.
      channels_out: Output channels.
      kernel_size: One extent for every axis, or one per axis.
      ndim: Spatial rank, which a scalar ``kernel_size`` is raised to.
      groups: Blocked connections; each output sees ``channels_in / groups``.
      bias: Whether a bias vector is owned.
      input_grid: Per-axis spatial input extents, without batch or channels.
      batch_size: Sequences in this invocation.
      stride: Window step, scalar or per axis.
      padding: ``"same"``, ``"valid"``, an int, or one per axis.
      dilation: Kernel spacing, scalar or per axis.
      dtype: Operand dtype; ``None`` is torch's default.
      input_grad: Compute the input gradient.
      weight_grad: Compute the weight gradient.
      bias_grad: Compute a gradient for an owned bias.

    Returns:
      cost: Integer FLOPs and operand bytes for the invocation, plus parameters.

    """
    if len(input_grid) != ndim:
        raise ValueError(
            f"input_grid has {len(input_grid)} axes; ndim is {ndim}.",
        )
    output_grid = conv_output_grid(
        input_grid,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )
    rows = batch_size * math.prod(output_grid)
    input_rows = batch_size * math.prod(input_grid)
    _validate_row_count(rows)
    _validate_row_count(input_rows)
    taps = math.prod(
        kernel_size if isinstance(kernel_size, tuple) else (kernel_size,) * ndim,
    )
    weights = channels_out * (channels_in // groups) * taps
    biases = channels_out if bias else 0
    dt = resolve_dtype(dtype)
    inputs = channels_in * input_rows
    gradients = int(input_grad) + int(weight_grad)
    backward = channels_out * rows + gradients * (inputs + weights) if gradients else 0
    return Cost(
        cells={
            ("flops", "primal", "matmul", dt): 2 * rows * weights,
            ("flops", "adjoint", "matmul", dt): 2 * rows * weights * gradients,
            ("bytes", "primal", "matmul", dt): nbytes(
                inputs + weights + rows * channels_out,
                dt,
            ),
            ("bytes", "adjoint", "matmul", dt): nbytes(backward, dt),
            ("flops", "primal", "elementwise", dt): rows * biases,
            ("bytes", "primal", "elementwise", dt): nbytes(biases, dt),
            ("flops", "adjoint", "reduction", dt): biases * (rows - 1)
            if bias_grad
            else 0,
            ("bytes", "adjoint", "reduction", dt): nbytes(rows * biases + biases, dt)
            if bias_grad
            else 0,
        },
        params=weights + biases,
        params_active=weights + biases,
    )


def _validate_row_count(rows: object) -> None:
    if not isinstance(rows, int) or isinstance(rows, bool):
        raise TypeError("Convolution row counts must be integers.")
    if rows < 1:
        raise ValueError("Convolution input and output rows must be positive.")


def _sequence_conv_output_length(
    input_length: int,
    *,
    kernel_size: int | tuple[int, ...],
    stride: int | tuple[int, ...],
    padding: int | str | tuple[int, ...],
    dilation: int | tuple[int, ...],
) -> int:
    """Require the convolution to produce every input sequence position."""
    output_length = conv_output_grid(
        (input_length,),
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )[0]
    if output_length < input_length:
        raise ValueError(
            "SequenceConv1d convolution output length "
            f"{output_length} is shorter than input length {input_length}.",
        )
    return output_length
