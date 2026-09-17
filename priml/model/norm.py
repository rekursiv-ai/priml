"""Normalization layers."""

from __future__ import annotations

from dataclasses import KW_ONLY
from typing import Self, override

from configgle import Fig
from torch import Tensor, nn
from torch.nn import functional as f

import torch

from priml.model.cost import (
    Bytes,
    Compute,
    Cost,
    elementwise_cost,
    reduction_cost,
)
from priml.model.custom_types import infer_same_width


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    class Config(Fig["RMSNorm"], kw_only=False):
        channels_in: int = -1
        """Number of input channels (normalized shape)."""

        channels_out: int = -1
        """Number of output channels (-1 to infer from channels_in)."""

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

        @override
        def finalize(self) -> Self:
            infer_same_width(self)
            return super().finalize()

        def cost(
            self,
            *,
            rows: float = 1,
            itemsize: int = 4,
            **kwargs: object,
        ) -> Cost:
            """Count square/mean/rsqrt/scale and its saved-rsqrt derivative.

            The adjoint forms sum(g*x), scales by r**3 / width, then subtracts
            the radial component; the affine adds one product each way. The
            two sums over the row are the reductions.

            Args:
              rows: Rows sharing each parameter and its gradient reduction.
              itemsize: Uniform bytes per operand element.
              **kwargs: The open message bus, forwarded to every child.

            Returns:
              cost: Per-token cost of this module.

            """
            del kwargs
            width = self.channels_in
            params = width if self.elementwise_affine else 0
            return elementwise_cost(
                primal=2 * width + 3 + params,
                adjoint=4 * width + 4 + 2 * params,
                channels=width,
                inputs=2,
                outputs=2,
                adjoint_inputs=6,
                adjoint_outputs=4,
                params=params,
                rows=rows,
                itemsize=itemsize,
            ) + Cost(
                primal=Compute(
                    bytes=Bytes(elementwise=itemsize * (7 + 2 * params)),
                )
                + reduction_cost(input_elements=width, itemsize=itemsize),
                adjoint=Compute(
                    bytes=Bytes(elementwise=itemsize * (12 + 4 * params)),
                )
                + reduction_cost(input_elements=width, itemsize=itemsize),
            )

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.normalized_shape = (config.channels_in,)
        self.eps = config.eps
        self.elementwise_affine = config.elementwise_affine
        self.weight: nn.Parameter | None
        if config.elementwise_affine:
            self.weight = nn.Parameter(
                torch.empty(
                    config.channels_in,
                    device=config.device,
                    dtype=config.dtype,
                ),
            )
        else:
            self.register_parameter("weight", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize every parameter in place."""
        if self.weight is not None:
            nn.init.ones_(self.weight)

    @override
    def forward(self, input: Tensor, **kwargs: object) -> Tensor:
        del kwargs
        return f.rms_norm(input, self.normalized_shape, self.weight, self.eps)


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

        channels_out: int = -1
        """Number of output channels (-1 to infer from channels_in)."""

        _: KW_ONLY

        eps: float = 1e-6
        """Epsilon for numerical stability."""

        @override
        def finalize(self) -> Self:
            infer_same_width(self)
            return super().finalize()

        def cost(
            self,
            *,
            rows: float = 1,
            itemsize: int = 4,
            **kwargs: object,
        ) -> Cost:
            """Count affine RMSNorm plus one ``1 + weight`` fold shared by every row.

            Args:
              rows: Rows sharing each parameter and its gradient reduction.
              itemsize: Uniform bytes per operand element.
              **kwargs: The open message bus, forwarded to every child.

            Returns:
              cost: Per-token cost of this module.

            """
            del kwargs
            width = self.channels_in
            return elementwise_cost(
                primal=3 * width + 3 + width / rows,
                adjoint=6 * width + 4,
                channels=width,
                inputs=3,
                outputs=3,
                adjoint_inputs=8,
                adjoint_outputs=6,
                params=width,
                rows=rows,
                itemsize=itemsize,
            ) + Cost(
                primal=Compute(
                    bytes=Bytes(elementwise=itemsize * (7 + 2 * width / rows)),
                )
                + reduction_cost(input_elements=width, itemsize=itemsize),
                adjoint=Compute(
                    bytes=Bytes(elementwise=itemsize * 12),
                )
                + reduction_cost(input_elements=width, itemsize=itemsize),
            )

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.eps = config.eps
        self.weight = nn.Parameter(torch.zeros(config.channels_in))

    def reset_parameters(self) -> None:
        """Initialize every parameter in place."""
        # Weight is used as ``1.0 + weight``; zeros makes the init identity.
        nn.init.zeros_(self.weight)

    @override
    def forward(self, input: Tensor, **kwargs: object) -> Tensor:
        del kwargs
        x_f32 = input.float()
        normed = x_f32 * torch.rsqrt(x_f32.pow(2).mean(-1, keepdim=True) + self.eps)
        return ((1.0 + self.weight.float()) * normed).type_as(input)


class LayerNorm(nn.LayerNorm):
    """Layer Normalization."""

    class Config(Fig["LayerNorm"], kw_only=False):
        channels_in: int = -1
        """Number of input channels (normalized shape)."""

        channels_out: int = -1
        """Number of output channels (-1 to infer from channels_in)."""

        _: KW_ONLY

        eps: float = 1e-5
        """Epsilon for numerical stability."""

        elementwise_affine: bool = False
        """Learn per-channel scale and shift parameters."""

        device: torch.device | str | None = None
        """Device for parameter allocation."""

        dtype: torch.dtype | None = None
        """Data type for parameters."""

        @override
        def finalize(self) -> Self:
            infer_same_width(self)
            return super().finalize()

        def cost(
            self,
            *,
            rows: float = 1,
            itemsize: int = 4,
            **kwargs: object,
        ) -> Cost:
            """Count centered statistics, normalization, and affine gradients.

            Args:
              rows: Rows sharing each parameter and its gradient reduction.
              itemsize: Uniform bytes per operand element.
              **kwargs: The open message bus, forwarded to every child.

            Returns:
              cost: Per-token cost of this module.

            """
            del kwargs
            return _normalization_cost(
                channels=self.channels_in,
                groups_per_token=1,
                params=2 * self.channels_in if self.elementwise_affine else 0,
                rows=rows,
                itemsize=itemsize,
            )

    def __init__(self, config: Config) -> None:
        super().__init__(
            normalized_shape=config.channels_in,
            eps=config.eps,
            elementwise_affine=config.elementwise_affine,
            device=config.device,
            dtype=config.dtype,
        )

    @override
    def forward(self, input: Tensor, **kwargs: object) -> Tensor:
        del kwargs
        return super().forward(input)


class BatchNorm(nn.BatchNorm1d):
    """BatchNorm1d for (B, L, C) input."""

    class Config(Fig["BatchNorm"], kw_only=False):
        channels_in: int = -1
        """Number of input channels."""

        channels_out: int = -1
        """Number of output channels (-1 to infer from channels_in)."""

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

        @override
        def finalize(self) -> Self:
            infer_same_width(self)
            return super().finalize()

        def cost(
            self,
            *,
            rows: float = 1,
            itemsize: int = 4,
            **kwargs: object,
        ) -> Cost:
            """Estimate training statistics and running updates per position.

            ``rows`` is batch times spatial/sequence positions. The
            analytical population-variance algorithm includes six operations
            per channel for the two running averages and two for the unbiased
            variance correction. One-row defaults are shape estimates only;
            a training batch norm requires more than one sample per channel.

            Args:
              rows: Rows sharing each parameter and its gradient reduction.
              itemsize: Uniform bytes per operand element.
              **kwargs: The open message bus, forwarded to every child.

            Returns:
              cost: Per-token cost of this module.

            """
            del kwargs
            return _normalization_cost(
                channels=self.channels_in,
                groups_per_token=self.channels_in / rows,
                params=2 * self.channels_in if self.elementwise_affine else 0,
                rows=rows,
                itemsize=itemsize,
            ) + elementwise_cost(
                primal=8 * self.channels_in / rows,
                adjoint=0,
                channels=self.channels_in / rows,
                inputs=10,
                outputs=8,
                adjoint_inputs=0,
                adjoint_outputs=0,
                itemsize=itemsize,
            )

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
    def forward(self, input: Tensor, **kwargs: object) -> Tensor:
        del kwargs
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

        channels_in: int = -1
        """Width of the axis being normalized (-1 to infer from channels_out)."""

        channels_out: int = -1
        """Number of output channels (-1 to infer from channels_in)."""

        _: KW_ONLY

        @override
        def finalize(self) -> Self:
            """Fill and validate the preserved channel width."""
            infer_same_width(self)
            return super().finalize()

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

        def cost(
            self,
            *,
            rows: float = 1,
            itemsize: int = 4,
            **kwargs: object,
        ) -> Cost:
            """Estimate warm training, including detached corrections and updates.

            Warmup skips five scalar operations per channel; this prices the
            post-warmup path. Detached correction coefficients do not receive
            gradients. The backward estimate includes differentiating the
            corrected mean and variance in addition to ordinary normalization.

            Args:
              rows: Rows sharing each parameter and its gradient reduction.
              itemsize: Uniform bytes per operand element.
              **kwargs: The open message bus, forwarded to every child.

            Returns:
              cost: Per-token cost of this module.

            """
            del kwargs
            width = self.channels_in
            return _normalization_cost(
                channels=width,
                groups_per_token=width / rows,
                params=2 * width,
                rows=rows,
                itemsize=itemsize,
            ) + elementwise_cost(
                primal=18 * width / rows,
                adjoint=5 * width / rows,
                channels=width / rows,
                inputs=26,
                outputs=20,
                adjoint_inputs=8,
                adjoint_outputs=5,
                itemsize=itemsize,
            )

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
        if config.momentum < 0.0 or config.momentum >= 1.0:
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

    def reset_parameters(self) -> None:
        """Initialize every parameter in place."""
        # The running estimates reset too: they are learned state, and the
        # correction this layer applies is measured against them, so leaving
        # them warm would reinitialize into the previous run's statistics.
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)
        nn.init.zeros_(self.running_mean)
        nn.init.ones_(self.running_var)
        nn.init.zeros_(self.steps)

    @override
    def forward(self, x: Tensor, **kwargs: object) -> Tensor:
        """Normalize ``x`` over every axis but the last.

        Args:
          x: Input with features last, ``[..., channels_in]``.
          **kwargs: Open message bus ignored by this terminal layer.

        Returns:
          normalized: Same shape, affinely transformed.

        """
        del kwargs
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
        self,
        batch_mean: Tensor,
        batch_var: Tensor,
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

        channels_out: int = -1
        """Number of output channels."""

        _: KW_ONLY

        @override
        def finalize(self) -> Self:
            """Fill and validate the preserved channel width."""
            infer_same_width(self)
            return super().finalize()

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

        def cost(
            self,
            *,
            rows: float = 1,
            itemsize: int = 4,
            **kwargs: object,
        ) -> Cost:
            """Estimate training statistics and running updates per position.

            ``rows`` is batch times spatial/sequence positions. The
            analytical population-variance algorithm includes six operations
            per channel for the two running averages and two for the unbiased
            variance correction. One-row defaults are shape estimates only;
            a training batch norm requires more than one sample per channel.

            Args:
              rows: Rows sharing each parameter and its gradient reduction.
              itemsize: Uniform bytes per operand element.
              **kwargs: The open message bus, forwarded to every child.

            Returns:
              cost: Per-token cost of this module.

            """
            del kwargs
            return _normalization_cost(
                channels=self.channels_in,
                groups_per_token=self.channels_in / rows,
                params=2 * self.channels_in if self.elementwise_affine else 0,
                rows=rows,
                itemsize=itemsize,
            ) + elementwise_cost(
                primal=8 * self.channels_in / rows,
                adjoint=0,
                channels=self.channels_in / rows,
                inputs=10,
                outputs=8,
                adjoint_inputs=0,
                adjoint_outputs=0,
                itemsize=itemsize,
            )

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
    def forward(self, input: Tensor, **kwargs: object) -> Tensor:
        del kwargs
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

        channels_out: int = -1
        """Number of output channels."""

        _: KW_ONLY

        @override
        def finalize(self) -> Self:
            """Fill and validate the preserved channel width."""
            infer_same_width(self)
            return super().finalize()

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

        def cost(
            self,
            *,
            rows: float = 1,
            seq_len: int = 1,
            itemsize: int = 4,
            **kwargs: object,
        ) -> Cost:
            """Count groups spanning ``seq_len`` positions within each sample.

            For images, ``seq_len`` is the product of spatial extents;
            ``rows`` is batch times that extent for affine gradients.

            Args:
              rows: Rows sharing each parameter and its gradient reduction.
              itemsize: Uniform bytes per operand element.
              seq_len: Positions each normalization group spans within one sample.
              **kwargs: The open message bus, forwarded to every child.

            Returns:
              cost: Per-token cost of this module.

            """
            del kwargs
            return _normalization_cost(
                channels=self.channels_in,
                groups_per_token=self.num_groups / seq_len,
                params=2 * self.channels_in if self.elementwise_affine else 0,
                rows=rows,
                itemsize=itemsize,
            )

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
    def forward(self, input: Tensor, **kwargs: object) -> Tensor:
        del kwargs
        return super().forward(input)


class GroupNorm(nn.GroupNorm):
    """GroupNorm for (B, L, C) input."""

    class Config(Fig["GroupNorm"], kw_only=False):
        channels_in: int = -1
        """Number of input channels."""

        channels_out: int = -1
        """Number of output channels."""

        _: KW_ONLY

        @override
        def finalize(self) -> Self:
            """Fill and validate the preserved channel width."""
            infer_same_width(self)
            return super().finalize()

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

        def cost(
            self,
            *,
            rows: float = 1,
            seq_len: int = 1,
            itemsize: int = 4,
            **kwargs: object,
        ) -> Cost:
            """Count groups spanning ``seq_len`` positions within each sample.

            For images, ``seq_len`` is the product of spatial extents;
            ``rows`` is batch times that extent for affine gradients.

            Args:
              rows: Rows sharing each parameter and its gradient reduction.
              itemsize: Uniform bytes per operand element.
              seq_len: Positions each normalization group spans within one sample.
              **kwargs: The open message bus, forwarded to every child.

            Returns:
              cost: Per-token cost of this module.

            """
            del kwargs
            return _normalization_cost(
                channels=self.channels_in,
                groups_per_token=self.num_groups / seq_len,
                params=2 * self.channels_in if self.elementwise_affine else 0,
                rows=rows,
                itemsize=itemsize,
            )

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
    def forward(self, input: Tensor, **kwargs: object) -> Tensor:
        del kwargs
        shape = input.shape
        x = input.reshape(-1, *shape[-2:]).movedim(-2, -1)
        return super().forward(x).movedim(-2, -1).reshape(shape)


# A group of R elements uses 5R+2 primal operations: mean, centering, square/mean
# variance, epsilon/rsqrt, and scaling. The adjoint uses 7R: sum(g), sum(g*y), two
# means, y scaling, two subtractions and rsqrt scaling. Each pass holds two sums over
# the group, R-1 apiece, in the reduction silo. Affine adds a scale/shift and one
# gradient product per parameter; the reduction over rows is the primitive's.
def _normalization_cost(
    *,
    channels: int,
    groups_per_token: float,
    params: int,
    rows: float,
    itemsize: int,
) -> Cost:
    """Count unfused centered statistics, scalar broadcasts, and affine maps."""
    sums = 2 * (channels - groups_per_token)
    return elementwise_cost(
        primal=5 * channels + 2 * groups_per_token + params - sums,
        adjoint=7 * channels + params - sums,
        channels=channels,
        inputs=3,
        outputs=3,
        adjoint_inputs=7,
        adjoint_outputs=5,
        params=params,
        rows=rows,
        itemsize=itemsize,
    ) + Cost(
        primal=Compute(
            bytes=Bytes(elementwise=itemsize * (10 * groups_per_token + 2 * params)),
        )
        + 2
        * reduction_cost(
            input_elements=channels,
            output_groups=groups_per_token,
            itemsize=itemsize,
        ),
        adjoint=Compute(
            bytes=Bytes(
                elementwise=itemsize
                * (7 * groups_per_token + 1.5 * params - params / (2 * rows)),
            ),
        )
        + 2
        * reduction_cost(
            input_elements=channels,
            output_groups=groups_per_token,
            itemsize=itemsize,
        ),
    )
