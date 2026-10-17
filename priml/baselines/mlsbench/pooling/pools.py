"""Global pools of the MLS-Bench ``cv-pooling-aggregation`` baselines.

Each reduces the last stage's ``(B, C, H, W)`` to ``(B, C)``, as the task's
editable ``CustomPool`` does, and fills
:attr:`~priml.baselines.mlsbench.pytorch_vision.model.ResNet.Config.pool`. The
pristine average pool is
:class:`~priml.baselines.mlsbench.pytorch_vision.model.GlobalAvgPool`.

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


class GlobalMaxPool(nn.Module):
    """Take every channel's largest activation over its spatial grid."""

    class Config(Fig["GlobalMaxPool"]):
        """No parameters; a config so the pool is a slot value."""

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

    def __init__(self, config: Config) -> None:
        del config
        super().__init__()

    @override
    def forward(self, x: Tensor) -> Tensor:
        avg = functional.adaptive_avg_pool2d(x, 1)
        mx = functional.adaptive_max_pool2d(x, 1)
        return ((avg + mx) / 2).flatten(1)
