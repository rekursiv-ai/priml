"""Normalization layers of the MLS-Bench ``dl-normalization`` baselines.

Each builds from a channel count alone, as the task's editable ``CustomNorm``
does, so each fills
:attr:`~priml.baselines.mlsbench.pytorch_vision.model.ResNet.Config.norm`.
The task's pristine ``CustomNorm`` wraps ``nn.BatchNorm2d``, the slot's default.

Every forward keeps the reference's operation order and every layer registers
its parameters in the reference's order: an algebraically equal rewrite rounds
differently, and the port must reproduce the reference bit for bit. The
reference's ``initialize_weights`` sets each layer's scale to one and shift to
zero, the values these layers are built with, so no initializer is needed.
"""

from __future__ import annotations

from typing import override

from torch import Tensor, nn
from torch.nn import functional

import torch


class GroupNorm(nn.GroupNorm):
    """Group normalization with the most groups, up to ``max_groups``, dividing the channels.

    References:
      https://arxiv.org/abs/1803.08494
        Wu and He 2018, "Group Normalization."

    """

    def __init__(self, num_features: int, *, max_groups: int = 32) -> None:
        groups = min(max_groups, num_features)
        while num_features % groups:
            groups -= 1
        super().__init__(groups, num_features)


class BatchInstanceNorm(nn.Module):
    """Per channel, a learned sigmoid gate between batch and instance statistics.

    Both statistics come from the current batch in training and in evaluation
    alike: the layer keeps no running estimates.

    References:
      https://arxiv.org/abs/1805.07925
        Name and Kim 2018, "Batch-Instance Normalization for Adaptively
        Style-Invariant Neural Networks."

    """

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
