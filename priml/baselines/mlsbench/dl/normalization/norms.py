"""Normalization layers of the MLS-Bench ``dl-normalization`` baselines.

Each builds from a channel count alone, as the task's editable ``CustomNorm``
does, so each fills
:attr:`~priml.baselines.mlsbench.dl.vision.model.ResNet.Config.norm`.
The task's pristine ``CustomNorm`` wraps ``nn.BatchNorm2d``, the slot's default.

Every forward keeps the reference's operation order and every layer registers
its parameters in the reference's order: an algebraically equal rewrite rounds
differently, and the port must reproduce the reference bit for bit. The
reference's ``initialize_weights`` sets each layer's scale to one and shift to
zero, the values these layers are built with, so no initializer is needed.
"""

from __future__ import annotations

from typing import Final, override

from torch import Tensor, nn
from torch.nn import functional

import torch

from priml.cost import Cost, cost, elementwise_cost, reduction_cost
from priml.model.norm import GroupNorm2d


MAX_GROUPS: Final = 32
"""Most groups :class:`GroupNorm` divides the channels into, as the reference caps it."""


class GroupNorm(nn.GroupNorm):
    """Group normalization with the most groups, up to :data:`MAX_GROUPS`, dividing the channels.

    References:
      https://arxiv.org/abs/1803.08494
        Wu and He 2018, "Group Normalization."

    """

    def __init__(self, num_features: int) -> None:
        super().__init__(groups_for(num_features), num_features)

    @classmethod
    def cost(
        cls,
        *,
        channels: int,
        seq_len: int,
        batch_size: int,
        dtype: torch.dtype | None,
        **kwargs: object,
    ) -> Cost:
        """Cost one affine group norm over ``seq_len`` positions per image.

        Args:
          channels: Normalized channels.
          seq_len: Spatial positions per image.
          batch_size: Images in this invocation.
          dtype: Activation dtype; ``None`` is torch's default.
          **kwargs: The open bus, forwarded.

        Returns:
          cost: Whole-invocation FLOPs, logical bytes and ownership.

        """
        return cost(
            GroupNorm2d.Config(
                channels,
                num_groups=groups_for(channels),
                elementwise_affine=True,
            ),
            seq_len=seq_len,
            batch_size=batch_size,
            dtype=dtype,
            **kwargs,
        )


def groups_for(num_features: int) -> int:
    """Return the most groups, up to :data:`MAX_GROUPS`, that divide ``num_features``."""
    groups = min(MAX_GROUPS, num_features)
    while num_features % groups:
        groups -= 1
    return groups


class BatchInstanceNorm(nn.Module):
    """Per channel, a learned sigmoid gate between batch and instance statistics.

    Both statistics come from the current batch in training and in evaluation
    alike: the layer keeps no running estimates.

    References:
      https://arxiv.org/abs/1805.07925
        Name and Kim 2018, "Batch-Instance Normalization for Adaptively
        Style-Invariant Neural Networks."

    """

    @classmethod
    def cost(
        cls,
        *,
        channels: int,
        seq_len: int,
        batch_size: int,
        dtype: torch.dtype | None,
        **kwargs: object,
    ) -> Cost:
        """Cost batch and instance statistics, the gated mix and the affine.

        Per element the forward centers and scales twice, squares twice for the
        variances, mixes three ways and applies the affine: 13 operations,
        twice that backward.

        Args:
          channels: Normalized channels.
          seq_len: Spatial positions per image.
          batch_size: Images in this invocation.
          dtype: Activation dtype; ``None`` is torch's default.
          **kwargs: The open bus; unused.

        Returns:
          cost: Whole-invocation FLOPs, logical bytes and ``3 * channels`` owned.

        """
        del kwargs
        return _mixed_norm_cost(
            channels=channels,
            rows=seq_len * batch_size,
            per_element=13,
            groups=(channels, channels * batch_size),
            params=3 * channels,
            dtype=dtype,
        )

    def __init__(self, num_features: int, *, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.rho = nn.Parameter(torch.ones(num_features))

    @override
    def forward(self, x: Tensor) -> Tensor:
        gate = torch.sigmoid(self.rho).view(1, -1, 1, 1)
        mean_bn = x.mean(dim=(0, 2, 3), keepdim=True)
        var_bn = x.var(dim=(0, 2, 3), keepdim=True, correction=0)
        mean_in = x.mean(dim=(2, 3), keepdim=True)
        var_in = x.var(dim=(2, 3), keepdim=True, correction=0)
        x_bn = (x - mean_bn) / (var_bn + self.eps).sqrt()
        x_in = (x - mean_in) / (var_in + self.eps).sqrt()
        x_norm = gate * x_bn + (1 - gate) * x_in
        return x_norm * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class SwitchableNorm(nn.Module):
    """Mean and variance as learned softmax mixtures of instance, layer and batch statistics.

    Every statistic comes from the current batch in training and in evaluation
    alike: the layer keeps no running estimates.

    References:
      https://arxiv.org/abs/1806.10779
        Luo et al. 2019, "Differentiable Learning-to-Normalize via Switchable
        Normalization."

    """

    @classmethod
    def cost(
        cls,
        *,
        channels: int,
        seq_len: int,
        batch_size: int,
        dtype: torch.dtype | None,
        **kwargs: object,
    ) -> Cost:
        """Cost instance, layer and batch statistics, the mixture and the affine.

        Per element the forward squares three times for the variances,
        centers, scales and applies the affine: 10 operations, twice that
        backward.

        Args:
          channels: Normalized channels.
          seq_len: Spatial positions per image.
          batch_size: Images in this invocation.
          dtype: Activation dtype; ``None`` is torch's default.
          **kwargs: The open bus; unused.

        Returns:
          cost: Whole-invocation FLOPs, logical bytes and ``2 * channels + 6``
            owned.

        """
        del kwargs
        return _mixed_norm_cost(
            channels=channels,
            rows=seq_len * batch_size,
            per_element=10,
            groups=(channels * batch_size, batch_size, channels),
            params=2 * channels + 6,
            dtype=dtype,
        )

    def __init__(self, num_features: int, *, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.mean_weight = nn.Parameter(torch.ones(3))
        self.var_weight = nn.Parameter(torch.ones(3))

    @override
    def forward(self, x: Tensor) -> Tensor:
        mean_w = functional.softmax(self.mean_weight, dim=0)
        var_w = functional.softmax(self.var_weight, dim=0)
        mean_in = x.mean(dim=(2, 3), keepdim=True)
        var_in = x.var(dim=(2, 3), keepdim=True, correction=0)
        mean_ln = x.mean(dim=(1, 2, 3), keepdim=True)
        var_ln = x.var(dim=(1, 2, 3), keepdim=True, correction=0)
        mean_bn = x.mean(dim=(0, 2, 3), keepdim=True)
        var_bn = x.var(dim=(0, 2, 3), keepdim=True, correction=0)
        mean = mean_w[0] * mean_in + mean_w[1] * mean_ln + mean_w[2] * mean_bn
        var = var_w[0] * var_in + var_w[1] * var_ln + var_w[2] * var_bn
        x_norm = (x - mean) / (var + self.eps).sqrt()
        return x_norm * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


def _mixed_norm_cost(
    *,
    channels: int,
    rows: int,
    per_element: int,
    groups: tuple[int, ...],
    params: int,
    dtype: torch.dtype | None,
) -> Cost:
    """Cost a norm mixing several statistics: one mean and variance per entry of ``groups``."""
    elements = channels * rows
    total = elementwise_cost(
        primal=per_element * elements,
        adjoint=2 * per_element * elements,
        channels=channels,
        params=params,
        rows=rows,
        dtype=dtype,
    )
    for count in groups:
        for phase in ("primal", "adjoint"):
            total += reduction_cost(
                input_elements=elements,
                output_groups=count,
                dtype=dtype,
                phase=phase,
            ).tile(2)
    return total
