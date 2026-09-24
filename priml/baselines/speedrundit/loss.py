"""Flow-matching objective for SR-DiT: velocity, alignment, and contrast.

Four terms, each weighted and each reported separately so a run can be read
after the fact:

- ``denoising``  -- squared error against the rectified-flow velocity.
- ``cls``        -- the same, for the diffused class token.
- ``projection`` -- REPA alignment of encoder tokens to a frozen visual
  encoder's features, as a negative cosine similarity.
- ``cfm``        -- contrastive flow matching, a NEGATIVE squared error against
  the batch neighbour's velocity, so minimizing the total pushes the field away
  from what the neighbour wants.

Four things vary independently, so each is a slot rather than a mode string:
``interpolant`` (the path between data and noise), ``time_sampler`` (where
along it a step lands), ``time_transform`` (how that time is reparameterized
for the resolution), and ``cfm_weight`` (what the contrastive term costs at
each time). Two interpolants ship: :func:`linear_path`, which writes
``alpha = 1 - t`` directly, and :func:`rectified_flow_path`, which routes
through :mod:`priml.math.diffusion`'s log-SNR schedule. They compute one
function and differ in the last bits.

References:
  https://arxiv.org/abs/2209.03003
    Liu et al. 2022, "Flow Straight and Fast."
  https://arxiv.org/abs/2410.06940
    Yu et al. 2024, "Representation Alignment for Generation" (REPA).
  https://arxiv.org/abs/2506.05350
    Stoica et al. 2025, "Contrastive Flow Matching."

"""

from __future__ import annotations

from dataclasses import KW_ONLY, field
from typing import TYPE_CHECKING, NamedTuple, Protocol, cast

import math

from configgle import Fig, Makeable, PartialConfig
from torch import Tensor

import torch

from priml.cost import Cost, elementwise_cost, map_cost, reduction_cost, set_cost
from priml.math.custom_types import TensorFn
from priml.math.diffusion import (
    compute_log_alpha,
    log_sigma_from_log_snr_per_rectified_flow,
    log_snr_from_log_time_per_logit,
)
from priml.math.probability import random_logit_normal


if TYPE_CHECKING:
    from collections.abc import Sequence

    from priml.baselines.speedrundit.model import SpeedrunDiT


__all__ = [
    "Interpolant",
    "InterpolantFn",
    "SpeedrunDiTLoss",
    "TimeSamplerFn",
    "TimeTransformFn",
    "VelocityField",
    "as_callable",
    "cosine_path",
    "linear_path",
    "linear_time_weight",
    "logit_normal_time",
    "mean_flat",
    "rectified_flow_path",
    "resolution_time_shift",
    "uniform_time",
    "uniform_time_weight",
]


def mean_flat(x: Tensor) -> Tensor:
    """Average over every axis but the batch.

    Args:
      x: Any tensor whose first axis is the batch.

    Returns:
      reduced: ``[batch]`` for rank two or more; a scalar for rank one, where
        the empty axis list makes torch reduce everything.

    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))


class Interpolant(NamedTuple):
    """Coefficients of a probability path at one time."""

    alpha: Tensor
    """Weight on the clean sample."""

    sigma: Tensor
    """Weight on the noise."""

    d_alpha: Tensor | float
    """Time derivative of ``alpha``."""

    d_sigma: Tensor | float
    """Time derivative of ``sigma``."""


class InterpolantFn(Protocol):
    """Maps a time to the path coefficients and their derivatives."""

    def __call__(self, t: Tensor) -> Interpolant:
        """Evaluate the path.

        Args:
          t: Flow times, broadcastable to the sample shape.

        Returns:
          interpolant: Path coefficients at ``t``.

        """
        ...


class TimeTransformFn(Protocol):
    """Reparameterizes flow time before the path is evaluated."""

    def __call__(self, t: Tensor, *, shape: Sequence[int]) -> Tensor:
        """Transform sampled times.

        Args:
          t: Sampled times in ``[0, 1]``.
          shape: Shape of one clean sample, without the batch axis.

        Returns:
          t: Transformed times, still in ``[0, 1]``.

        """
        ...


class VelocityField(Protocol):
    """Predicts both streams' velocities; the model, or a wrapper around it."""

    def __call__(
        self,
        media: Tensor,
        time: Tensor,
        label: Tensor,
        cls_token: Tensor,
        /,
    ) -> SpeedrunDiT.Output:
        """Predict velocities at one time.

        Args:
          media: Noised latents.
          time: Flow times, ``[batch]``.
          label: Class indices.
          cls_token: Noised class token.

        Returns:
          output: Velocities and alignment projections.

        """
        ...


class TimeSamplerFn(Protocol):
    """Draws the flow times one step lands on."""

    def __call__(self, batch: int) -> Tensor:
        """Draw one step's times.

        Args:
          batch: Samples per step.

        Returns:
          t: ``[batch, 1, 1, 1]`` times in ``[0, 1)``.

        """
        ...


def linear_path(t: Tensor) -> Interpolant:
    """Straight path between data and noise: ``x_t = (1 - t) x + t eps``.

    Args:
      t: Flow times.

    Returns:
      interpolant: ``(1 - t, t, -1, 1)``.

    """
    # Integer derivatives, not tensors: ``-1 * x + 1 * eps`` is an exact
    # negation and an exact subtraction, so the target carries no rounding.
    return Interpolant(alpha=1 - t, sigma=t, d_alpha=-1, d_sigma=1)


def rectified_flow_path(t: Tensor) -> Interpolant:
    """Walk the straight path through Priml's log-SNR schedule instead.

    Args:
      t: Flow times.

    Returns:
      interpolant: ``(1 - t, t, -1, 1)``, reached through log-SNR.

    """
    # Same function as ``linear_path``, off by the last bits: a logit into a
    # sigmoid rounds twice where ``1 - t`` rounds once. Inject this when
    # upstream parity is not the point.
    log_snr = log_snr_from_log_time_per_logit(t.log())
    log_sigma = log_sigma_from_log_snr_per_rectified_flow(log_snr)
    log_alpha = compute_log_alpha(log_snr, log_sigma)
    return Interpolant(
        alpha=torch.exp(log_alpha),
        sigma=torch.exp(log_sigma),
        d_alpha=-1,
        d_sigma=1,
    )


def cosine_path(t: Tensor) -> Interpolant:
    """Quarter-cosine path: ``x_t = cos(pi t / 2) x + sin(pi t / 2) eps``.

    Args:
      t: Flow times.

    Returns:
      interpolant: Coefficients and derivatives of the cosine path.

    """
    half_pi = math.pi / 2
    return Interpolant(
        alpha=torch.cos(t * half_pi),
        sigma=torch.sin(t * half_pi),
        d_alpha=-half_pi * torch.sin(t * half_pi),
        d_sigma=half_pi * torch.cos(t * half_pi),
    )


def uniform_time(
    batch: int,
    *,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Draw flow times uniformly on ``[0, 1)``.

    Args:
      batch: Samples per step.
      generator: Optional generator; ``None`` uses the CPU default stream.

    Returns:
      t: ``[batch, 1, 1, 1]`` times.

    """
    # Drawn on the CPU stream, not the batch's device: that is where the
    # reference draws, and the device move happens once, afterwards.
    return torch.rand((batch, 1, 1, 1), generator=generator)


def logit_normal_time(batch: int) -> Tensor:
    """Draw flow times from a logit-normal, concentrating mass mid-path.

    Args:
      batch: Samples per step.

    Returns:
      t: ``[batch, 1, 1, 1]`` times.

    """
    # The reference spells this ``exp(z) / (1 + exp(z))``, which is
    # ``sigmoid(z)`` off by the last bits. Off the parity path either way:
    # the pinned recipe draws uniformly.
    return random_logit_normal(batch, 1, 1, 1)


def resolution_time_shift(
    t: Tensor,
    *,
    shape: Sequence[int],
    base: int = 4096,
) -> Tensor:
    """Shift time toward noise in proportion to the sample's element count.

    A larger latent carries more redundancy, so the same nominal time destroys
    less information; the shift restores a comparable difficulty across
    resolutions. ``base`` is the element count at which the shift is the
    identity.

    Args:
      t: Sampled times in ``[0, 1]``.
      shape: Shape of one clean sample, without the batch axis.
      base: Element count defining the unshifted resolution.

    Returns:
      t: Shifted times, clamped to ``[0, 1]``.

    """
    elements = 1
    for dim in shape:
        elements *= dim
    # Correctly rounded, as the reference computes it: ``** 0.5`` is a libm
    # ``pow`` and lands on a different last bit for some element counts.
    shift = math.sqrt(elements / base)  # noqa: TID251 -- Bit parity: correctly rounded sqrt, not libm pow.
    return torch.clamp((shift * t) / (1 + (shift - 1) * t), 0.0, 1.0)


@set_cost(map_cost(primal=1, adjoint=1))
def uniform_time_weight(t: Tensor) -> Tensor:
    """Weight every time equally in the contrastive term.

    Args:
      t: Flow times.

    Returns:
      weight: A scalar one, broadcasting against any error tensor.

    """
    return torch.ones((), device=t.device, dtype=t.dtype)


@set_cost(map_cost(primal=1, adjoint=1))
def linear_time_weight(t: Tensor) -> Tensor:
    """Weight the contrastive term by time, sparing the near-data end.

    Args:
      t: Flow times.

    Returns:
      weight: ``t`` itself.

    """
    return t


class SpeedrunDiTLoss:
    """The four-term SR-DiT objective, reported term by term."""

    class Output(NamedTuple):
        """One evaluation of the objective."""

        loss: Tensor
        """Weighted total, a scalar."""

        denoising: Tensor
        """Per-sample latent velocity error, ``[batch]``."""

        cls: Tensor
        """Per-sample class-token velocity error, ``[batch]``."""

        projection: Tensor
        """Alignment term, a scalar."""

        cfm: Tensor
        """Contrastive term for the latents, a scalar."""

        cfm_cls: Tensor
        """Contrastive term for the class token, a scalar. Reported but not
        summed into ``loss``, matching the reference."""

        time: Tensor
        """The times drawn for this step, ``[batch, 1, 1, 1]``."""

        noise: Tensor
        """The latent noise drawn for this step."""

    class Config(Fig["SpeedrunDiTLoss"]):
        """Configuration for SpeedrunDiTLoss."""

        _: KW_ONLY

        interpolant: InterpolantFn = linear_path
        """Probability path between data and noise."""

        time_sampler: Makeable[TimeSamplerFn] | TimeSamplerFn = uniform_time
        """Draws the per-sample flow times."""

        time_transform: Makeable[TimeTransformFn] | TimeTransformFn | None = field(
            default_factory=lambda: PartialConfig(resolution_time_shift, base=4096),
        )
        """Reparameterizes time before the path is evaluated; ``None`` leaves
        the sampled times alone."""

        cfm_weight: Makeable[TensorFn] | TensorFn = uniform_time_weight
        """Per-time weight on the contrastive term."""

        projection_coeff: float = 0.5
        """Weight on the alignment term."""

        cls_coeff: float = 0.03
        """Weight on the class-token velocity term."""

        cfm_coeff: float = 0.05
        """Weight on the contrastive term."""

        def cost(
            self,
            *,
            batch_size: int,
            channels_in: int,
            image_size: int,
            channels_cls: int,
            seq_len: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost one objective evaluation, excluding the model forward.

            Args:
              batch_size: Samples per step.
              channels_in: Latent channels.
              image_size: Latent grid side.
              channels_cls: Class-token width.
              seq_len: Tokens per sequence.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus, unread here.

            Returns:
              cost: Whole-invocation cost of the objective.

            """
            del kwargs
            latent = channels_in * image_size * image_size
            # Path construction, both targets, both squared errors, and the
            # contrastive pair: nine elementwise sweeps over the latent.
            latents = elementwise_cost(
                primal=9,
                adjoint=9,
                channels=latent,
                rows=batch_size,
                dtype=dtype,
                inputs=2,
            )
            cls = elementwise_cost(
                primal=9,
                adjoint=9,
                channels=channels_cls,
                rows=batch_size,
                dtype=dtype,
                inputs=2,
            )
            reductions = reduction_cost(
                input_elements=batch_size * latent * 2,
                output_groups=batch_size,
                dtype=dtype,
            ) + reduction_cost(
                input_elements=batch_size * seq_len * channels_cls,
                output_groups=batch_size * seq_len,
                dtype=dtype,
            )
            return latents + cls + reductions

    def __init__(self, config: Config) -> None:
        self.config = config
        self.interpolant = config.interpolant
        self.time_sampler: TimeSamplerFn = as_callable(config.time_sampler)
        self.time_transform: TimeTransformFn | None = (
            None
            if config.time_transform is None
            else as_callable(config.time_transform)
        )
        self.cfm_weight: TensorFn = as_callable(config.cfm_weight)

    def __call__(
        self,
        model: VelocityField,
        *,
        media: Tensor,
        label: Tensor,
        cls_token: Tensor,
        features: Sequence[Tensor] = (),
        time: Tensor | None = None,
        noise: Tensor | None = None,
        noise_cls: Tensor | None = None,
    ) -> Output:
        """Evaluate the objective on one batch.

        The draw order is part of the contract: time first, on the CPU stream,
        then the latent noise and the class-token noise on the batch's device.
        The model's own draws -- label dropout, token drop, path drop -- follow
        inside the forward. Reordering any of them changes every sample a run
        ever sees.

        Args:
          model: The velocity field.
          media: Clean latents, ``[batch, channels, size, size]``.
          label: Class indices, ``[batch]``.
          cls_token: Clean class-token targets, ``[batch, channels_cls]``.
          features: Frozen encoder features to align to, one per target.
          time: Times to use instead of drawing them.
          noise: Latent noise to use instead of drawing it.
          noise_cls: Class-token noise to use instead of drawing it.

        Returns:
          output: The weighted total and every term that built it.

        """
        if time is None:
            time = self.time_sampler(media.shape[0])
        if self.time_transform is not None:
            time = self.time_transform(time, shape=tuple(media.shape[1:]))
        time = time.to(device=media.device, dtype=media.dtype)
        if noise is None:
            noise = torch.randn_like(media)
        if noise_cls is None:
            noise_cls = torch.randn_like(cls_token)

        path = self.interpolant(time)
        flat = path.alpha.squeeze(-1).squeeze(-1), path.sigma.squeeze(-1).squeeze(-1)
        media_noisy = path.alpha * media + path.sigma * noise
        cls_noisy = flat[0] * cls_token + flat[1] * noise_cls
        media_target = path.d_alpha * media + path.d_sigma * noise
        cls_target = path.d_alpha * cls_token + path.d_sigma * noise_cls

        output = model(media_noisy, time.flatten(), label, cls_noisy)

        denoising = mean_flat((output.velocity - media_target) ** 2)
        cls_error = mean_flat((output.cls_velocity - cls_target) ** 2)
        projection = self._projection(features, output.projections, media)

        # A scalar weight multiplies by exactly one, which IEEE-754 leaves
        # untouched, so the uniform case costs an op and no accuracy. A
        # per-sample weight arrives shaped like ``time`` and has to lose the
        # two spatial axes before it meets the class token.
        weight = self.cfm_weight(time)
        cls_weight = weight if weight.ndim == 0 else weight.squeeze(-1).squeeze(-1)
        cfm_error = (output.velocity - torch.roll(media_target, 1, 0)) ** 2
        cfm = -(cfm_error * weight).mean()
        cfm_cls = -(
            ((output.cls_velocity - torch.roll(cls_target, 1, 0)) ** 2) * cls_weight
        ).mean()

        cfg = self.config
        total = (
            denoising.mean()
            + cfg.projection_coeff * projection.mean()
            + cfg.cls_coeff * cls_error.mean()
            + cfg.cfm_coeff * cfm.mean()
        )
        return SpeedrunDiTLoss.Output(
            loss=total,
            denoising=denoising,
            cls=cls_error,
            projection=projection,
            cfm=cfm,
            cfm_cls=cfm_cls,
            time=time,
            noise=noise,
        )

    # Accumulated one sample at a time, in batch order. That is slower than a single
    # batched reduction and it is what the reference does; a batched sum associates the
    # additions differently and lands on a different last bit.
    def _projection(
        self,
        features: Sequence[Tensor],
        projections: Sequence[Tensor],
        media: Tensor,
    ) -> Tensor:
        """Average negative cosine similarity between features and projections."""
        if not features:
            return torch.zeros((), device=media.device, dtype=media.dtype)
        total = torch.zeros((), device=media.device, dtype=media.dtype)
        batch = features[0].shape[0]
        for target, predicted in zip(features, projections, strict=False):
            for target_row, predicted_row in zip(target, predicted, strict=True):
                normalized = torch.nn.functional.normalize(predicted_row, dim=-1)
                reference = torch.nn.functional.normalize(target_row, dim=-1)
                total = total + mean_flat(-(reference * normalized).sum(dim=-1))
        return total / (len(features) * batch)


def as_callable[T](slot: Makeable[T] | T) -> T:
    """Build a slot holding a config, or pass a callable through.

    Args:
      slot: A config such as a ``PartialConfig``, or the callable itself.

    Returns:
      callable: What the slot holds, built if it was a config.

    """
    if isinstance(slot, Makeable):
        # ``Makeable`` is runtime-checkable, so isinstance erases its type
        # parameter: ``make`` reads as returning ``object`` without the cast.
        return cast(T, slot.make())
    return slot
