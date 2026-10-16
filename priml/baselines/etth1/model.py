"""DLinear model for ETTh1 forecasting."""

from __future__ import annotations

from dataclasses import KW_ONLY, field
from typing import Self, cast, override

from configgle import Fig, Makeable, Makes
from torch import Tensor, nn

import torch

from priml.cost import Cost, cost, elementwise_cost, matmul_cost, traffic
from priml.model.custom_types import ChannelsIn, ChannelsOut, propagate_attr
from priml.model.pool import avg_pool_cost


class MovingAverage(nn.Module):
    """Moving-average block used by DLinear."""

    class Config(Fig["MovingAverage"]):
        """Moving-average configuration."""

        kernel_size: int = 25
        """Width of the moving-average filter."""

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            channels: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost overlapping averages and their endpoint-folded adjoint.

            Each output averages kernel_size values. The adjoint first spreads
            those scaled gradients, then accumulates S*K contributions into S
            original positions, including repeated endpoint padding.

            Args:
              seq_len: Timesteps per sequence.
              batch_size: Sequences per step.
              channels: Variables per timestep.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus, unread here.

            Returns:
              cost: Forward and adjoint operations of one invocation.

            """
            del kwargs
            elements = batch_size * channels * seq_len
            return avg_pool_cost(
                channels=channels,
                positions=self.kernel_size,
                batch_size=batch_size * seq_len,
                dtype=dtype,
            ) + traffic(
                "adjoint",
                "reduction",
                elements=elements * (self.kernel_size + 1),
                flops=elements * (self.kernel_size - 1),
                dtype=dtype,
            )

    def __init__(self, config: Config) -> None:
        super().__init__()

        if config.kernel_size <= 0 or config.kernel_size % 2 == 0:
            raise ValueError("Moving-average kernel_size must be positive and odd.")
        self.kernel_size = config.kernel_size
        self.avg = nn.AvgPool1d(
            kernel_size=config.kernel_size,
            stride=1,
            padding=0,
        )

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Apply endpoint-padded moving averaging."""
        front = x[:, 0:1, :].repeat(
            1,
            (self.kernel_size - 1) // 2,
            1,
        )
        end = x[:, -1:, :].repeat(
            1,
            (self.kernel_size - 1) // 2,
            1,
        )

        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))

        return x.permute(0, 2, 1)


class SeriesDecomposition(nn.Module):
    """Split a sequence into seasonal and trend components."""

    class Config(Fig["SeriesDecomposition"]):
        """Series-decomposition configuration."""

        moving_average: Makeable[nn.Module] = field(
            default_factory=MovingAverage.Config,
        )
        """Moving-average trend extractor."""

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            channels: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost the average, residual subtraction, and two adjoint joins.

            Args:
              seq_len: Timesteps per sequence.
              batch_size: Sequences per step.
              channels: Variables per timestep.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: Forwarded to the moving-average cost.

            Returns:
              cost: Forward and adjoint operations of one invocation.

            """
            elements = batch_size * seq_len * channels
            return cost(
                self.moving_average,
                seq_len=seq_len,
                batch_size=batch_size,
                channels=channels,
                dtype=dtype,
                **kwargs,
            ) + elementwise_cost(
                primal=elements,
                adjoint=2 * elements,
                channels=channels,
                rows=batch_size * seq_len,
                inputs=2,
                adjoint_outputs=2,
                dtype=dtype,
            )

    def __init__(self, config: Config) -> None:
        super().__init__()

        self.moving_average = config.moving_average.make()

    @override
    def forward(
        self,
        x: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return residual seasonal component and moving-average trend."""
        moving_mean = cast(Tensor, self.moving_average(x))
        residual = x - moving_mean

        return residual, moving_mean


class ReferenceLinear(nn.Linear):
    """Linear projection with the reference's random initialization."""

    class Config(Fig["ReferenceLinear"], kw_only=False):
        channels_in: int = -1
        """Input width, supplied by DLinear."""

        channels_out: int = -1
        """Output width, supplied by DLinear."""

        _: KW_ONLY

        bias: bool = True
        """Include a random bias."""

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Count this projection's operations and parameters.

            Args:
              seq_len: Rows projected per sequence.
              batch_size: Sequences per step.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus, unread here.

            Returns:
              cost: Operations and parameters of one invocation.

            """
            del kwargs
            return matmul_cost(
                channels_in=self.channels_in,
                channels_out=self.channels_out,
                bias=self.bias,
                rows=seq_len * batch_size,
                dtype=dtype,
            )

    def __init__(self, config: Config) -> None:
        super().__init__(
            in_features=config.channels_in,
            out_features=config.channels_out,
            bias=config.bias,
        )


class MeanLinear(ReferenceLinear):
    """Start each forecast at the input mean, keeping the random bias."""

    class Config(Makes["MeanLinear"], ReferenceLinear.Config):
        pass

    @override
    def reset_parameters(self) -> None:
        super().reset_parameters()
        with torch.no_grad():
            self.weight.copy_(torch.ones_like(self.weight) / self.in_features)


class DLinear(nn.Module):
    """Decomposition-linear model for multivariate forecasting."""

    class Config(Fig["DLinear"]):
        """DLinear forecasting configuration."""

        seq_len: int = 336
        """Input sequence length."""

        pred_len: int = 96
        """Forecast horizon."""

        channels: int = 7
        """Number of input channels."""

        decomposition: Makeable[nn.Module] = field(
            default_factory=SeriesDecomposition.Config,
        )
        """Split the input into seasonal and trend values."""

        seasonal: Makeable[nn.Module] = field(default_factory=MeanLinear.Config)
        """Project the seasonal values into the forecast."""

        trend: Makeable[nn.Module] = field(default_factory=MeanLinear.Config)
        """Project the trend values into the forecast."""

        decoder: Makeable[nn.Module] = field(default_factory=ReferenceLinear.Config)
        """Unused layer retained for reference initialization."""

        @override
        def finalize(self) -> Self:
            for projection in (self.seasonal, self.trend, self.decoder):
                propagate_attr(
                    projection,
                    "channels_in",
                    self.seq_len,
                    protocol=ChannelsIn,
                )
                propagate_attr(
                    projection,
                    "channels_out",
                    self.pred_len,
                    protocol=ChannelsOut,
                )
            return super().finalize()

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost both active projections and own the unused reference decoder.

            Args:
              seq_len: Input history; must equal the configured ``seq_len``.
              batch_size: Sequences per step.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: Forwarded to every submodule cost.

            Returns:
              cost: Operations of one forward and backward pass, plus the
                decoder's parameters.

            Raises:
              ValueError: ``seq_len`` differs from the configured history.

            """
            if seq_len != self.seq_len:
                raise ValueError(
                    "Cost sequence length must match the model's configured history.",
                )
            return (
                cost(
                    self.decomposition,
                    seq_len=seq_len,
                    batch_size=batch_size,
                    channels=self.channels,
                    dtype=dtype,
                    **kwargs,
                )
                + cost(
                    self.seasonal,
                    seq_len=self.channels,
                    batch_size=batch_size,
                    dtype=dtype,
                    **kwargs,
                )
                + cost(
                    self.trend,
                    seq_len=self.channels,
                    batch_size=batch_size,
                    dtype=dtype,
                    **kwargs,
                )
                + elementwise_cost(
                    primal=batch_size * self.channels * self.pred_len,
                    adjoint=0,
                    channels=self.pred_len,
                    rows=batch_size * self.channels,
                    inputs=2,
                    adjoint_inputs=1,
                    adjoint_outputs=2,
                    dtype=dtype,
                )
                + Cost(
                    params=cost(
                        self.decoder,
                        seq_len=self.channels,
                        batch_size=batch_size,
                        dtype=dtype,
                        **kwargs,
                    ).params,
                )
            )

    def __init__(self, config: Config) -> None:
        super().__init__()

        if min(config.seq_len, config.pred_len, config.channels) <= 0:
            raise ValueError(
                "Sequence length, forecast horizon, and channels must be positive.",
            )
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.channels = config.channels

        self.decomposition = config.decomposition.make()
        self.seasonal = config.seasonal.make()
        self.trend = config.trend.make()

        # Keep this unused layer because it affects the reference RNG sequence.
        self.decoder = config.decoder.make()

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Forecast future values from an input sequence."""
        seasonal_init, trend_init = cast(tuple[Tensor, Tensor], self.decomposition(x))

        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)

        seasonal_output = cast(Tensor, self.seasonal(seasonal_init))
        trend_output = cast(Tensor, self.trend(trend_init))

        output = seasonal_output + trend_output

        return output.permute(0, 2, 1)
