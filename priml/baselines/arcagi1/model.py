"""ARC model components that extend the shared priml puzzle solver."""

from __future__ import annotations

from typing import TYPE_CHECKING, Self, override

from configgle import Makes
from torch import Tensor

import torch

from priml.baselines.sudoku.model import CoreOutput, SudokuNet
from priml.cost import Cost, cost, elementwise_cost
from priml.model.conv import Conv1d
from priml.model.init import InitFn, unit_fan_in_uniform
from priml.model.linear import Linear
from priml.model.swiglu import SwiGLU, silu


if TYPE_CHECKING:
    from torch.distributed.tensor.parallel import ParallelStyle


class HPSURM(SudokuNet):
    """Single-state URM core, using priml's shared puzzle model machinery.

    The slow state is the one carried hidden state. The fast state remains an
    inert compatibility slot because the shared recurrence driver carries two
    tensors; HPS's inner loop is implemented here and injects the input at each
    pass, as in the historical URM.
    """

    class Config(Makes["HPSURM"], SudokuNet.Config):
        """Shared SudokuNet shape and slots with HPSURM construction."""

        @override
        def cost(self, *, batch_size: int, dtype: torch.dtype | None) -> Cost:
            """Remove the shared two-state core's unused stack and state adds."""
            total = SudokuNet.Config.cost(self, batch_size=batch_size, dtype=dtype)
            rows = batch_size * self.total_seq_len
            stack = cost(
                self.block,
                seq_len=self.total_seq_len,
                batch_size=batch_size,
                dtype=dtype,
            ).tile(self.num_layers, copies=self.num_layers)
            add = elementwise_cost(
                primal=self.channels_in * rows,
                adjoint=self.channels_in * rows,
                channels=self.channels_in,
                inputs=2,
                rows=rows,
                dtype=dtype,
            )
            excess = stack + add.tile(2)
            slow_cycles = 1 if self.recurrence is None else self.recurrence.slow_cycles
            excess += excess.only("primal").tile(slow_cycles - 1)
            return Cost(
                cells={
                    key: value - excess.cells.get(key, 0)
                    for key, value in total.cells.items()
                },
                params=total.params,
                params_active=total.params_active,
                bytes_state=total.bytes_state,
            )

    @override
    def _core(
        self,
        input_emb: Tensor,
        z_slow: Tensor,
        z_fast: Tensor,
        cos_sin: tuple[Tensor, Tensor] | None = None,
    ) -> CoreOutput:
        """Run the URM inner loop over its single carried latent."""
        cycles = (
            1 if self.config.recurrence is None else self.config.recurrence.fast_cycles
        )
        hidden = z_slow
        for _ in range(cycles):
            hidden = self._mix(hidden + input_emb, cos_sin)
        logits = self.head(hidden)
        halt_logits = self.halt_head(hidden[:, 0]).to(torch.float32)
        halt = (
            halt_logits.squeeze(-1)
            if self.config.halt_outputs == 1
            else halt_logits[..., 0]
        )
        return CoreOutput(logits, halt, hidden, z_fast)


class ConvSwiGLU(SwiGLU):
    """SwiGLU with the HPS depthwise short convolution between gate and head."""

    class Config(Makes["ConvSwiGLU"], SwiGLU.Config):
        """SwiGLU settings plus URM's depthwise convolution width."""

        if TYPE_CHECKING:

            @override
            def make(self) -> ConvSwiGLU:
                """Build the configured convolutional SwiGLU."""
                ...

        init_weight_out: InitFn = unit_fan_in_uniform
        kernel_size: int = 2
        shift_conv: bool = True

        @override
        def finalize(self) -> Self:
            if not self.gate or self.split_gate_projection or self.act is not silu:
                raise ValueError("ConvSwiGLU requires the fused gated projection")
            if self.init_weight_out is unit_fan_in_uniform:
                self.init_weight_out = self.init_weight
            elif self.init_weight_out is not self.init_weight:
                raise ValueError("ConvSwiGLU uses init_weight for both projections")
            if self.shard not in (None, "colwise"):
                raise ValueError(f"unsupported ConvSwiGLU shard policy: {self.shard}")
            return super().finalize()

        @override
        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost the gated projections and the depthwise convolution.

            Args:
              seq_len: Sequence length entering the block.
              batch_size: Number of examples.
              dtype: Activation dtype, or ``None`` for the default.
              **kwargs: Ignored additional cost context.

            Returns:
              result: Forward and backward operation estimate.

            """
            del kwargs
            base = SwiGLU.Config(
                channels_in=self.channels_in,
                channels_out=self.channels_out,
                channels_hidden=self.channels_hidden,
                bias=self.bias,
                norm=self.norm,
                init_weight=self.init_weight,
                init_weight_out=self.init_weight,
            )
            hidden = self.channels_hidden
            rows = seq_len * batch_size
            if self.shift_conv:
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
                bias = elementwise_cost(
                    primal=hidden * rows,
                    adjoint=hidden * rows,
                    channels=hidden,
                    params=hidden,
                    rows=rows,
                    dtype=dtype,
                )
                conv = tap.tile(self.kernel_size, copies=self.kernel_size)
                conv += combine + bias
            else:
                conv = cost(
                    Conv1d.Config(
                        channels_in=hidden,
                        channels_out=hidden,
                        kernel_size=self.kernel_size,
                        padding=0,
                        groups=hidden,
                        bias=True,
                    ),
                    input_grid=seq_len + 2 * (self.kernel_size // 2),
                    batch_size=batch_size,
                    dtype=dtype,
                )
            return (
                cost(
                    base,
                    seq_len=seq_len,
                    batch_size=batch_size,
                    dtype=dtype,
                )
                + conv
                + cost(silu, channels=hidden * rows, dtype=dtype)
            )

    def __init__(self, config: Config) -> None:
        torch.nn.Module.__init__(self)
        self.up_proj = Linear.Config(
            channels_in=config.channels_in,
            channels_out=config.channels_hidden * 2,
            bias=config.bias,
            init_weight=config.init_weight,
            depth_index=config.depth_index,
        ).make()
        self.down_proj = Linear.Config(
            channels_in=config.channels_hidden,
            channels_out=config.channels_out,
            bias=config.bias,
            init_weight=config.init_weight,
            depth_index=config.depth_index,
        ).make()
        self.conv = Conv1d.Config(
            channels_in=config.channels_hidden,
            channels_out=config.channels_hidden,
            kernel_size=config.kernel_size,
            padding=0,
            groups=config.channels_hidden,
            bias=True,
            depth_index=config.depth_index,
        ).make()
        self.norm = config.norm.make() if config.norm is not None else None
        self.shift_conv = config.shift_conv
        self.shard = config.shard

    @override
    def reset_parameters(self) -> None:
        """Reset the projections, normalization, and depthwise convolution."""
        super().reset_parameters()
        self.conv.reset_parameters()

    @override
    def tensor_parallel_style(self) -> ParallelStyle:
        """Reject tensor parallelism until the depthwise convolution can shard."""
        raise NotImplementedError(
            "ConvSwiGLU tensor parallelism needs a depthwise convolution plan.",
        )

    @override
    def forward(self, x: Tensor, *args: object, **kwargs: object) -> Tensor:
        """Apply gated expansion, short convolution, and output projection.

        Args:
          x: Hidden sequence shaped ``[batch, sequence, channels]``.
          *args: Ignored compatibility arguments.
          **kwargs: Ignored compatibility options.

        Returns:
          output: Transformed hidden sequence with ``channels_out`` channels.

        """
        del args, kwargs
        gate, up = self.up_proj(x).chunk(2, dim=-1)
        gated = (
            torch.sigmoid(gate) * self.norm(gate * up)
            if self.norm is not None
            else torch.nn.functional.silu(gate) * up
        )
        length = gated.shape[1]
        if not self.shift_conv:
            convolved = torch.nn.functional.pad(
                gated.transpose(1, 2),
                (self.conv.kernel_size[0] // 2,) * 2,
            )
            convolved = torch.nn.functional.silu(
                self.conv(convolved.to(self.conv.weight.dtype))[..., :length],
            )
            return self.down_proj(convolved.transpose(1, 2).to(gated.dtype))
        weight = self.conv.weight
        values = gated.to(weight.dtype)
        left_pad = self.conv.kernel_size[0] // 2
        convolved = values * weight[:, 0, left_pad]
        for tap in range(weight.shape[-1]):
            shift = tap - left_pad
            if shift == 0:
                continue
            shifted = (
                torch.nn.functional.pad(values, (0, 0, -shift, 0))
                if shift < 0
                else torch.nn.functional.pad(values[:, shift:], (0, 0, 0, shift))
            )[:, :length]
            convolved = convolved + shifted * weight[:, 0, tap]
        if self.conv.bias is not None:
            convolved = convolved + self.conv.bias
        convolved = torch.nn.functional.silu(convolved.to(gated.dtype))
        return self.down_proj(convolved)
