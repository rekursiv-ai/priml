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

        eps: float | None = 1e-6
        """Epsilon for numerical stability; None takes the dtype's own.

        ``None`` is not "no epsilon" -- torch substitutes ``finfo(dtype).eps``,
        which is ~1.19e-7 in float32 and 7.8e-3 in bfloat16. That is what a
        bare ``functional.rms_norm`` call uses, so a port reproducing one has
        to say ``None`` rather than any fixed number."""

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
        """Number of input channels (normalized shape)."""

        _: KW_ONLY

        eps: float = 1e-6
        """Epsilon for numerical stability."""

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


class BatchRenorm(nn.Module):
    """Batch statistics a small or shifting batch can trust.

    Batch normalization uses the current batch's statistics while training and
    running averages afterwards, so a network behaves differently in the two
    modes -- badly, when batches are small or their contents shift. This
    corrects each batch's statistics TOWARD the running ones with a clipped
    affine term, and lets the correction engage only once those running
    estimates are worth correcting toward.

    The correction is deliberately crude and detached from the backward pass.
    That is the point: it moves training-time behavior toward inference-time
    behavior without letting a stale estimate inject gradients of its own.

    Why a reinforcement learner cares: a network trained on its own changing
    policy sees the input distribution shift under it, which is the case plain
    batch normalization handles worst. Stabilizing that is what lets a
    Q-learner drop its target network.

    References:
      https://arxiv.org/abs/1702.03275
        Ioffe 2017. Batch renormalization: towards reducing minibatch
        dependence in batch-normalized models.

    """

    class Config(Fig["BatchRenorm"]):
        """Configure the normalization."""

        channels_in: int = 1
        """Width of the axis being normalized."""

        momentum: float = 0.999
        """Retention of the running statistics per update.

        High, because the estimate has to survive a distribution that shifts
        with the policy; a fast average would track the current batch and
        defeat the purpose."""

        eps: float = 1e-3
        """Added under every square root, so a constant feature is finite."""

        warmup_steps: int = 1_000
        """Updates spent as plain batch normalization before correcting.

        The correction is measured AGAINST the running statistics, so applying
        it before those mean anything would correct toward noise."""

        max_ratio: float = 3.0
        """Bound on the scale correction, applied both ways."""

        max_drift: float = 5.0
        """Bound on the mean correction, in running standard deviations."""

    def __init__(self, config: Config) -> None:
        """Build the learned affine and the running statistics.

        Args:
          config: Width and correction bounds.

        Raises:
          ValueError: A dimension or bound is invalid.

        """
        super().__init__()
        if config.channels_in <= 0:
            raise ValueError("channels_in must be positive")
        if config.max_ratio < 1.0:
            raise ValueError("max_ratio must be at least one")
        if config.max_drift < 0.0:
            raise ValueError("max_drift must be non-negative")
        if not 0.0 <= config.momentum < 1.0:
            raise ValueError("momentum must be in [0, 1)")

        self.config = config
        self.weight = nn.Parameter(torch.ones(config.channels_in))
        self.bias = nn.Parameter(torch.zeros(config.channels_in))
        # Buffers, not parameters: they are estimates the forward pass
        # maintains, and an optimizer must never step them.
        #
        # Annotated as well as registered. ``register_buffer`` types its
        # result as ``Tensor | Module | None``, so every arithmetic use below
        # would otherwise infer as ``Any`` and take the shapes with it.
        self.running_mean: Tensor
        self.running_var: Tensor
        self.steps: Tensor
        self.register_buffer("running_mean", torch.zeros(config.channels_in))
        self.register_buffer("running_var", torch.ones(config.channels_in))
        self.register_buffer("steps", torch.zeros((), dtype=torch.int64))

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Normalize ``x`` over every axis but the last.

        Args:
          x: Input with features last, ``[..., channels_in]``.

        Returns:
          normalized: Same shape, affinely transformed.

        """
        if not self.training:
            mean, variance = self.running_mean, self.running_var
            return self._affine(x, mean=mean, variance=variance)

        axes = tuple(range(x.dim() - 1))
        batch_mean = x.mean(dim=axes)
        batch_var = ((x - batch_mean) ** 2).mean(dim=axes)

        mean, variance = self._corrected(batch_mean, batch_var)
        self._accumulate(batch_mean, batch_var)
        return self._affine(x, mean=mean, variance=variance)

    def _corrected(
        self, batch_mean: Tensor, batch_var: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Return the statistics to normalize by, corrected once warm."""
        config = self.config
        deviation = (batch_var + config.eps).sqrt()
        running_deviation = (self.running_var + config.eps).sqrt()

        # Detached: these correct the normalization toward inference-time
        # behavior, and are not a path the loss should differentiate through.
        ratio = (
            (deviation / running_deviation)
            .detach()
            .clamp(
                1.0 / config.max_ratio,
                config.max_ratio,
            )
        )
        drift = (
            ((batch_mean - self.running_mean) / running_deviation)
            .detach()
            .clamp(-config.max_drift, config.max_drift)
        )

        warm = bool(self.steps >= config.warmup_steps)
        if not warm:
            return batch_mean, batch_var
        return batch_mean - drift * deviation / ratio, batch_var / ratio**2

    @torch.no_grad()
    def _accumulate(self, batch_mean: Tensor, batch_var: Tensor) -> None:
        """Fold this batch into the running statistics."""
        momentum = self.config.momentum
        self.running_mean.mul_(momentum).add_(batch_mean, alpha=1.0 - momentum)
        self.running_var.mul_(momentum).add_(batch_var, alpha=1.0 - momentum)
        self.steps.add_(1)

    def _affine(self, x: Tensor, *, mean: Tensor, variance: Tensor) -> Tensor:
        """Standardize by the supplied statistics, then scale and shift."""
        normalized = (x - mean) / (variance + self.config.eps).sqrt()
        return normalized * self.weight + self.bias


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
