"""DLinear model for ETTh1 forecasting."""

from __future__ import annotations

from dataclasses import field
from typing import override

from configgle import Fig
from torch import Tensor, nn

import torch

from priml.cost import Cost, elementwise_cost, matmul_cost, traffic
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

        moving_average: MovingAverage.Config = field(
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
            """Cost the average, residual subtraction, and two adjoint joins."""
            elements = batch_size * seq_len * channels
            return self.moving_average.cost(
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

        self.moving_average = MovingAverage(config.moving_average)

    @override
    def forward(
        self,
        x: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return residual seasonal component and moving-average trend."""
        moving_mean = self.moving_average(x)
        residual = x - moving_mean

        return residual, moving_mean


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

        kernel_size: int = 25
        """Moving-average kernel width."""

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost both active projections and own the unused reference decoder."""
            if seq_len != self.seq_len:
                raise ValueError(
                    "Cost sequence length must match the model's configured history.",
                )
            decomposition = SeriesDecomposition.Config()
            decomposition.moving_average.kernel_size = self.kernel_size
            projection = matmul_cost(
                channels_in=self.seq_len,
                channels_out=self.pred_len,
                bias=True,
                rows=batch_size * self.channels,
                dtype=dtype,
            )
            return (
                decomposition.cost(
                    seq_len=seq_len,
                    batch_size=batch_size,
                    channels=self.channels,
                    dtype=dtype,
                    **kwargs,
                )
                + projection.tile(2, copies=2)
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
                + Cost(params=(self.seq_len + 1) * self.pred_len)
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

        decomposition = SeriesDecomposition.Config()
        decomposition.moving_average.kernel_size = config.kernel_size

        self.decomposition = SeriesDecomposition(decomposition)

        self.seasonal = nn.Linear(
            config.seq_len,
            out_features=config.pred_len,
        )

        self.trend = nn.Linear(
            config.seq_len,
            out_features=config.pred_len,
        )

        # Keep this unused layer because it affects the reference RNG sequence.
        self.decoder = nn.Linear(
            config.seq_len,
            out_features=config.pred_len,
        )

        with torch.no_grad():
            self.seasonal.weight.copy_(
                torch.ones_like(self.seasonal.weight) / config.seq_len,
            )
            self.trend.weight.copy_(
                torch.ones_like(self.trend.weight) / config.seq_len,
            )

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Forecast future values from an input sequence."""
        seasonal_init, trend_init = self.decomposition(x)

        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)

        seasonal_output = self.seasonal(seasonal_init)
        trend_output = self.trend(trend_init)

        output = seasonal_output + trend_output

        return output.permute(0, 2, 1)
