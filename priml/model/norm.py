"""Normalization layers."""

from __future__ import annotations

from dataclasses import KW_ONLY
from typing import Any, Protocol, override, runtime_checkable

from configgle import Fig
from torch import Tensor, nn

import torch


@runtime_checkable
class NormProtocol(Protocol):
    """Protocol for normalization layers."""

    def __call__(self, input: Tensor, *args: Any, **kwargs: Any) -> Tensor: ...


class NormConfigProtocol(Protocol):
    """Protocol for normalization layer configs.

    Captures the fields common to all norm configs, useful for
    propagating ``channels_in`` or ``elementwise_affine`` into
    nested norm configs without knowing the concrete norm type.
    """

    channels_in: int
    eps: float
    elementwise_affine: bool
    device: torch.device | str | None
    dtype: torch.dtype | None


class RMSNorm(nn.RMSNorm):
    """Root Mean Square Layer Normalization."""

    class Config(Fig["RMSNorm"], kw_only=False):
        channels_in: int = -1
        """Number of input channels (normalized shape)."""

        _: KW_ONLY

        eps: float = 1e-6
        """Epsilon for numerical stability."""

        elementwise_affine: bool = False
        """Learn per-channel scale parameters."""

        device: torch.device | str | None = None
        """Device for parameter allocation."""

        dtype: torch.dtype | None = None
        """Data type for parameters."""

    def __init__(self, config: Config) -> None:
        super().__init__(
            normalized_shape=config.channels_in,
            eps=config.eps,
            elementwise_affine=config.elementwise_affine,
            device=config.device,
            dtype=config.dtype,
        )

    @override
    def forward(self, input: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        del args, kwargs
        return super().forward(input)


class CenteredRMSNorm(nn.Module):
    """RMSNorm with ``(1 + weight)`` scaling (weight initialized to zeros).

    Equivalent to standard RMSNorm with ``elementwise_affine=True`` except
    the learnable scale is parameterized as ``1 + w`` where ``w`` starts at
    zero. This keeps the layer near-identity at init, which improves
    training stability in deep networks (used by Gemma, Qwen3.5, etc.).
    Computation is done in float32 for numerical precision.
    """

    class Config(Fig["CenteredRMSNorm"], kw_only=False):
        channels_in: int = -1
        _: KW_ONLY
        eps: float = 1e-6

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.eps = config.eps
        self.weight = nn.Parameter(torch.zeros(config.channels_in))

    def reset_parameters(self) -> None:
        # weight is used as ``1.0 + weight``; zeros makes the init identity.
        nn.init.zeros_(self.weight)

    @override
    def forward(self, input: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        del args, kwargs
        x_f32 = input.float()
        normed = x_f32 * torch.rsqrt(x_f32.pow(2).mean(-1, keepdim=True) + self.eps)
        return ((1.0 + self.weight.float()) * normed).type_as(input)


class LayerNorm(nn.LayerNorm):
    """Layer Normalization."""

    class Config(Fig["LayerNorm"], kw_only=False):
        channels_in: int = -1
        """Number of input channels (normalized shape)."""

        _: KW_ONLY

        eps: float = 1e-5
        """Epsilon for numerical stability."""

        elementwise_affine: bool = False
        """Learn per-channel scale and shift parameters."""

        device: torch.device | str | None = None
        """Device for parameter allocation."""

        dtype: torch.dtype | None = None
        """Data type for parameters."""

    def __init__(self, config: Config) -> None:
        super().__init__(
            normalized_shape=config.channels_in,
            eps=config.eps,
            elementwise_affine=config.elementwise_affine,
            device=config.device,
            dtype=config.dtype,
        )

    @override
    def forward(self, input: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        del args, kwargs
        return super().forward(input)


class BatchNorm(nn.BatchNorm1d):
    """BatchNorm1d for (B, L, C) input."""

    class Config(Fig["BatchNorm"], kw_only=False):
        channels_in: int = -1
        """Number of input channels."""

        _: KW_ONLY

        momentum: float = 0.1
        """Running stats exponential moving average factor."""

        eps: float = 1e-5
        """Epsilon for numerical stability."""

        elementwise_affine: bool = False
        """Learn per-channel scale and shift parameters."""

        device: torch.device | str | None = None
        """Device for parameter allocation."""

        dtype: torch.dtype | None = None
        """Data type for parameters."""

    def __init__(self, config: Config) -> None:
        super().__init__(
            config.channels_in,
            momentum=config.momentum,
            eps=config.eps,
            affine=config.elementwise_affine,
            device=config.device,
            dtype=config.dtype,
        )

    @override
    def forward(self, input: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        del args, kwargs
        shape = input.shape
        return super().forward(input.reshape(-1, shape[-1])).reshape(shape)


class BatchNorm2d(nn.BatchNorm2d):
    """BatchNorm2d for (B, C, H, W) input."""

    class Config(Fig["BatchNorm2d"], kw_only=False):
        channels_in: int = -1
        """Number of input channels."""

        _: KW_ONLY

        momentum: float = 0.1
        """Running stats exponential moving average factor."""

        eps: float = 1e-5
        """Epsilon for numerical stability."""

        elementwise_affine: bool = False
        """Learn per-channel scale and shift parameters."""

        device: torch.device | str | None = None
        """Device for parameter allocation."""

        dtype: torch.dtype | None = None
        """Data type for parameters."""

    def __init__(self, config: Config) -> None:
        super().__init__(
            config.channels_in,
            momentum=config.momentum,
            eps=config.eps,
            affine=config.elementwise_affine,
            device=config.device,
            dtype=config.dtype,
        )

    @override
    def forward(self, input: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        del args, kwargs
        return super().forward(input)


class GroupNorm2d(nn.GroupNorm):
    """GroupNorm for (B, C, H, W) input.

    Batch-independent (no running statistics), so train and eval modes
    are identical -- the norm of choice for weight-shared recursive
    cores where BatchNorm's single running-stat set cannot represent
    per-iteration activation distributions.
    """

    class Config(Fig["GroupNorm2d"], kw_only=False):
        channels_in: int = -1
        """Number of input channels."""

        _: KW_ONLY

        num_groups: int = 8
        """Number of groups to divide channels into."""

        eps: float = 1e-5
        """Epsilon for numerical stability."""

        elementwise_affine: bool = False
        """Learn per-channel scale and shift parameters."""

        device: torch.device | str | None = None
        """Device for parameter allocation."""

        dtype: torch.dtype | None = None
        """Data type for parameters."""

    def __init__(self, config: Config) -> None:
        super().__init__(
            config.num_groups,
            config.channels_in,
            eps=config.eps,
            affine=config.elementwise_affine,
            device=config.device,
            dtype=config.dtype,
        )

    @override
    def forward(self, input: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        del args, kwargs
        return super().forward(input)


class GroupNorm(nn.GroupNorm):
    """GroupNorm for (B, L, C) input."""

    class Config(Fig["GroupNorm"], kw_only=False):
        channels_in: int = -1
        """Number of input channels."""

        _: KW_ONLY

        num_groups: int = 8
        """Number of groups to divide channels into."""

        eps: float = 1e-5
        """Epsilon for numerical stability."""

        elementwise_affine: bool = False
        """Learn per-channel scale and shift parameters."""

        device: torch.device | str | None = None
        """Device for parameter allocation."""

        dtype: torch.dtype | None = None
        """Data type for parameters."""

    def __init__(self, config: Config) -> None:
        super().__init__(
            config.num_groups,
            config.channels_in,
            eps=config.eps,
            affine=config.elementwise_affine,
            device=config.device,
            dtype=config.dtype,
        )

    @override
    def forward(self, input: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        del args, kwargs
        shape = input.shape
        x = input.reshape(-1, *shape[-2:]).movedim(-2, -1)
        return super().forward(x).movedim(-2, -1).reshape(shape)
