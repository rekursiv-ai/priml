"""Global pools of the MLS-Bench ``cv-pooling-aggregation`` baselines.

Each reduces the last stage's ``(B, C, H, W)`` to ``(B, C)``, as the task's
editable ``CustomPool`` does, and fills
:attr:`~priml.baselines.mlsbench.dl.vision.model.ResNet.Config.pool`. The
pristine average pool is
:class:`~priml.baselines.mlsbench.dl.vision.model.GlobalAvgPool`.

Every expression keeps the reference's operation order: a pool that must
reproduce the reference's values bit for bit follows its arithmetic, so an
algebraically equal rewrite is not a refactor.
"""

from __future__ import annotations

from typing import override

from configgle import Fig
from torch import Tensor, nn
from torch.nn import functional

import torch

from priml.cost import Cost, elementwise_cost
from priml.model.pool import avg_pool_cost, max_pool_cost


class GlobalMaxPool(nn.Module):
    """Take every channel's largest activation over its spatial grid."""

    class Config(Fig["GlobalMaxPool"]):
        """No parameters; a config so the pool is a slot value."""

        def cost(
            self,
            *,
            channels: int,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost one window-covering max per channel and its routed gradient.

            Args:
              channels: Pooled channels.
              seq_len: Spatial positions reduced per channel.
              batch_size: Images in this invocation.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus; unused.

            Returns:
              cost: Whole-invocation FLOPs and logical bytes; owns nothing.

            """
            del kwargs
            return max_pool_cost(
                channels=channels,
                kernel_size=(seq_len, 1),
                rows=batch_size,
                dtype=dtype,
            )

    def __init__(self, config: Config) -> None:
        del config
        super().__init__()

    @override
    def forward(self, x: Tensor) -> Tensor:
        return functional.adaptive_max_pool2d(x, 1).flatten(1)


class GeneralizedMeanPool(nn.Module):
    """Generalized mean ``mean(x ** p) ** (1 / p)`` with one learned exponent.

    ``p = 1`` is the average pool and ``p -> inf`` the max pool. Activations
    are floored at ``eps`` so a fractional power of a negative is never taken.

    References:
      https://arxiv.org/abs/1711.02512
        Radenovic, Tolias and Chum 2018, "Fine-tuning CNN Image Retrieval with
        No Human Annotation."

    """

    class Config(Fig["GeneralizedMeanPool"]):
        """Initial exponent and the floors of the exponent and activations."""

        p: float = 3.0
        """Exponent the parameter starts at."""

        p_min: float = 1.0
        """Floor of the exponent in the forward, so the pool is never below a mean."""

        eps: float = 1e-6
        """Floor of every activation before the power."""

        def cost(
            self,
            *,
            channels: int,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost the clamped power, the global mean, the root and the exponent's gradient.

            Args:
              channels: Pooled channels.
              seq_len: Spatial positions reduced per channel.
              batch_size: Images in this invocation.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus; unused.

            Returns:
              cost: Whole-invocation FLOPs and logical bytes; owns the exponent.

            """
            del kwargs
            elements = channels * batch_size * seq_len
            # Clamp and power on every element, then the mean, then one root per
            # channel; the exponent is one owned parameter whose gradient is a
            # reduction over every element.
            return (
                elementwise_cost(
                    primal=3 * elements,
                    adjoint=4 * elements,
                    channels=elements,
                    params=1,
                    dtype=dtype,
                    inputs=2,
                )
                + avg_pool_cost(
                    channels=channels,
                    positions=seq_len,
                    batch_size=batch_size,
                    dtype=dtype,
                )
                + elementwise_cost(
                    primal=2 * channels * batch_size,
                    adjoint=3 * channels * batch_size,
                    channels=channels * batch_size,
                    dtype=dtype,
                    inputs=2,
                )
            )

    def __init__(self, config: Config) -> None:
        super().__init__()
        # Built as the reference builds it; a parameter registered here draws
        # nothing, so the generator stays where the parameter-free pools leave it.
        self.p = nn.Parameter(torch.ones(1) * config.p)
        self.p_min = config.p_min
        self.eps = config.eps

    @override
    def forward(self, x: Tensor) -> Tensor:
        p = self.p.clamp(min=self.p_min)
        x = x.clamp(min=self.eps)
        return functional.adaptive_avg_pool2d(x.pow(p), 1).pow(1.0 / p).flatten(1)


class AvgMaxPool(nn.Module):
    """Average the global average pool and the global max pool."""

    class Config(Fig["AvgMaxPool"]):
        """No parameters; a config so the pool is a slot value."""

        def cost(
            self,
            *,
            channels: int,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost both global pools and the average of their outputs.

            Args:
              channels: Pooled channels.
              seq_len: Spatial positions reduced per channel.
              batch_size: Images in this invocation.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus; unused.

            Returns:
              cost: Whole-invocation FLOPs and logical bytes; owns nothing.

            """
            del kwargs
            pooled = channels * batch_size
            return (
                avg_pool_cost(
                    channels=channels,
                    positions=seq_len,
                    batch_size=batch_size,
                    dtype=dtype,
                )
                + max_pool_cost(
                    channels=channels,
                    kernel_size=(seq_len, 1),
                    rows=batch_size,
                    dtype=dtype,
                )
                + elementwise_cost(
                    primal=2 * pooled,
                    adjoint=2 * pooled,
                    channels=pooled,
                    dtype=dtype,
                    inputs=2,
                    adjoint_outputs=2,
                )
            )

    def __init__(self, config: Config) -> None:
        del config
        super().__init__()

    @override
    def forward(self, x: Tensor) -> Tensor:
        avg = functional.adaptive_avg_pool2d(x, 1)
        mx = functional.adaptive_max_pool2d(x, 1)
        return ((avg + mx) / 2).flatten(1)
