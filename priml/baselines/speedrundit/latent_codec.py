"""Storage codecs: how a raw latent corpus is written to disk and read back.

A codec is the third of three separate transforms. The autoencoder produces a
raw floating latent; a codec turns it into stored bytes and back; a latent
normalizer (on the train step) maps the recovered raw latent into diffusion
space. The codec never sees the normalizer, so a corpus can be re-normalized
without re-encoding, and the autoencoder never sees the codec, so RAE's
``encode`` stays float.

Two families:

* :class:`FloatCodec` rounds to a floating dtype. ``float32`` is the identity
  the REG corpora use; ``float16`` halves them, and beats ``bfloat16`` at the
  same cost for bounded latents (three more mantissa bits, range never used).
* :class:`ScalarTableCodec` stores one ``uint8`` index per scalar against a
  fitted ``[C, 256]`` table of reconstruction levels: 4x smaller than float32.
  How the table is fitted -- Lloyd-Max, Gaussian companding, quantiles, uniform
  -- is an injected :class:`TableFit`, and which channels share a table is an
  injected :class:`ChannelGroups`, so every benchmark candidate is a value in
  one slot rather than a class of its own.

A fitted codec's table is DATA, produced by the preparer and stored beside the
corpus; its config states only how to fit it. Codecs are plain objects, not
modules: they transform data on its way to a model and hold no parameters.
"""

from __future__ import annotations

from dataclasses import field
from typing import TYPE_CHECKING, Final, Protocol, runtime_checkable

import math

from configgle import Fig, Makeable
from torch import Tensor

import torch

from priml.math.scalar_quantization import (
    dequantize,
    gaussian_levels,
    high_resolution_levels,
    lloyd_max,
    midpoints,
    quantize,
)


if TYPE_CHECKING:
    from collections.abc import Mapping


NUM_LEVELS: Final = 256
"""Levels a ``uint8`` index addresses."""


@runtime_checkable
class LatentCodec(Protocol):
    """Maps raw ``[..., C, H, W]`` latents to a stored tensor and back."""

    stored_dtype: torch.dtype
    """Dtype of what :meth:`encode` returns and :meth:`decode` accepts."""

    def encode(self, latent: Tensor, /) -> Tensor:
        """Return the stored representation of a raw latent."""
        ...

    def decode(self, stored: Tensor, /) -> Tensor:
        """Return the float32 raw latent a stored tensor approximates."""
        ...


@runtime_checkable
class FittedCodec(LatentCodec, Protocol):
    """A codec whose parameters are fitted to a sample of the corpus."""

    num_fit_images: int
    """Images the preparer encodes to fit the codec."""

    def fit(self, sample: Tensor, /) -> None:
        """Fit the codec's parameters to ``[N, C, H, W]`` raw latents."""
        ...

    def table(self) -> dict[str, Tensor]:
        """Return the fitted parameters, for storage beside the corpus."""
        ...

    def load_table(self, table: Mapping[str, Tensor], /) -> None:
        """Restore parameters :meth:`table` returned."""
        ...


class FloatCodec:
    """Round to a floating dtype; decode widens back to float32."""

    class Config(Fig["FloatCodec"]):
        """The stored dtype."""

        dtype: torch.dtype = torch.float32
        """``float32`` is the identity; ``float16`` or ``bfloat16`` halve storage."""

    def __init__(self, config: Config) -> None:
        if not config.dtype.is_floating_point:
            raise ValueError(f"FloatCodec needs a floating dtype; got {config.dtype}.")
        self.stored_dtype = config.dtype

    def encode(self, latent: Tensor, /) -> Tensor:
        """Round to the stored dtype.

        Args:
          latent: Raw floating latent.

        Returns:
          stored: The latent in ``stored_dtype``.

        Raises:
          ValueError: A value overflows the stored dtype.

        """
        stored = latent.to(self.stored_dtype)
        # float16 saturates at 65504 to inf; an overflow is a corpus that decodes to
        # garbage, so it fails here rather than at training time.
        if not bool(torch.isfinite(stored).all()):
            raise ValueError(
                f"Latent does not fit {self.stored_dtype}: a value is non-finite "
                "or out of range.",
            )
        return stored

    def decode(self, stored: Tensor, /) -> Tensor:
        """Widen to float32.

        Args:
          stored: Latent in ``stored_dtype``.

        Returns:
          latent: float32 latent.

        """
        return stored.float()


@runtime_checkable
class TableFit(Protocol):
    """Chooses ``NUM_LEVELS`` reconstruction levels for one sample of scalars."""

    def __call__(self, values: Tensor, /) -> Tensor:
        """Return ``[NUM_LEVELS]`` non-decreasing float64 levels for 1-D ``values``."""
        ...


class LloydMaxFit:
    """Minimum-squared-error levels, started at the cube-root density."""

    class Config(Fig["LloydMaxFit"]):
        """Iteration budget."""

        max_iterations: int = 500
        """Upper bound on Lloyd iterations per table."""

        tolerance: float = 1e-7
        """Stop once an iteration lowers distortion by less than this fraction."""

    def __init__(self, config: Config) -> None:
        self.config = config

    def __call__(self, values: Tensor, /) -> Tensor:
        """Return Lloyd-Max levels for ``values``.

        Args:
          values: 1-D sample.

        Returns:
          levels: ``[NUM_LEVELS]`` float64.

        """
        return lloyd_max(
            values,
            high_resolution_levels(values, NUM_LEVELS),
            max_iterations=self.config.max_iterations,
            tolerance=self.config.tolerance,
        )


class GaussianFit:
    """The standard-normal Lloyd-Max table, shifted and scaled to the sample."""

    class Config(Fig["GaussianFit"]):
        """No parameters: mean and standard deviation come from the sample."""

    def __init__(self, config: Config) -> None:
        del config
        self.unit = gaussian_levels(NUM_LEVELS)

    def __call__(self, values: Tensor, /) -> Tensor:
        """Return ``mean + std * gaussian_levels``.

        Args:
          values: 1-D sample.

        Returns:
          levels: ``[NUM_LEVELS]`` float64.

        """
        x = values.to(torch.float64)
        return x.mean() + x.std() * self.unit


class QuantileFit:
    """Levels at equal-probability quantiles: every index equally likely.

    Maximizes index entropy, not accuracy -- the level density is ``p`` rather
    than the squared-error optimum ``p ** (1 / 3)``, so the tails are coarse.
    """

    class Config(Fig["QuantileFit"]):
        """No parameters."""

    def __init__(self, config: Config) -> None:
        del config

    def __call__(self, values: Tensor, /) -> Tensor:
        """Return the ``(k + 1/2) / NUM_LEVELS`` sample quantiles.

        Args:
          values: 1-D sample.

        Returns:
          levels: ``[NUM_LEVELS]`` float64.

        """
        x = values.to(torch.float64).sort().values
        position = (torch.arange(NUM_LEVELS, dtype=torch.float64) + 0.5) / NUM_LEVELS
        index = (position * x.numel()).long().clamp(max=x.numel() - 1)
        return x[index]


class LinearFit:
    """Evenly spaced levels over the sample's range, or a clipped range."""

    class Config(Fig["LinearFit"]):
        """Range policy."""

        clip_sigmas: float | None = None
        """Span ``mean +- clip_sigmas * std``; ``None`` spans min to max."""

    def __init__(self, config: Config) -> None:
        if config.clip_sigmas is not None and config.clip_sigmas <= 0:
            raise ValueError("clip_sigmas must be positive.")
        self.clip_sigmas = config.clip_sigmas

    def __call__(self, values: Tensor, /) -> Tensor:
        """Return cell centres of a uniform partition of the range.

        Args:
          values: 1-D sample.

        Returns:
          levels: ``[NUM_LEVELS]`` float64.

        """
        x = values.to(torch.float64)
        if self.clip_sigmas is None:
            low, high = x.min(), x.max()
        else:
            spread = self.clip_sigmas * x.std()
            low, high = x.mean() - spread, x.mean() + spread
        step = (high - low) / NUM_LEVELS
        return low + (torch.arange(NUM_LEVELS, dtype=torch.float64) + 0.5) * step


@runtime_checkable
class ChannelGroups(Protocol):
    """Decides which channels share one table, from each channel's scale."""

    def __call__(self, scale: Tensor, /) -> Tensor:
        """Return ``[C]`` int64 group ids for ``[C]`` per-channel standard deviations."""
        ...


class PerChannel:
    """Every channel gets its own table."""

    class Config(Fig["PerChannel"]):
        """No parameters."""

    def __init__(self, config: Config) -> None:
        del config

    def __call__(self, scale: Tensor, /) -> Tensor:
        """Return ``arange(C)``."""
        return torch.arange(scale.numel())


class SharedTable:
    """One table for every channel."""

    class Config(Fig["SharedTable"]):
        """No parameters."""

    def __init__(self, config: Config) -> None:
        del config

    def __call__(self, scale: Tensor, /) -> Tensor:
        """Return all zeros."""
        return torch.zeros(scale.numel(), dtype=torch.int64)


class ScaleGroups:
    """Channels ranked by log standard deviation, cut into equal-size groups."""

    class Config(Fig["ScaleGroups"]):
        """How many groups."""

        num_groups: int = 8
        """Tables shared across channels of similar scale."""

    def __init__(self, config: Config) -> None:
        if config.num_groups < 1:
            raise ValueError("num_groups must be positive.")
        self.num_groups = config.num_groups

    def __call__(self, scale: Tensor, /) -> Tensor:
        """Return each channel's scale-rank group.

        Args:
          scale: ``[C]`` per-channel standard deviations.

        Returns:
          groups: ``[C]`` int64 in ``[0, num_groups)``.

        """
        rank = torch.empty(scale.numel(), dtype=torch.int64)
        rank[scale.argsort()] = torch.arange(scale.numel())
        return rank * self.num_groups // scale.numel()


class ScalarTableCodec:
    """One ``uint8`` index per scalar against a fitted ``[C, 256]`` level table.

    Encoding picks each value's nearest level (ties to the lower one) and
    saturates past the outermost levels; decoding is a table lookup. Tables are
    float32; the boundaries between levels are stored with them, so encoding
    reproduces exactly the cells the fit used.
    """

    class Config(Fig["ScalarTableCodec"]):
        """How the table is fitted and shared, and how much data fits it."""

        fit: Makeable[TableFit] = field(default_factory=LloydMaxFit.Config)
        """Chooses each table's levels from its sample."""

        groups: Makeable[ChannelGroups] = field(default_factory=PerChannel.Config)
        """Which channels share a table."""

        num_fit_images: int = 4_096
        """Images encoded to fit the tables (the stability benchmark sizes this)."""

    def __init__(self, config: Config) -> None:
        if config.num_fit_images < 1:
            raise ValueError("num_fit_images must be positive.")
        self.stored_dtype = torch.uint8
        self.num_fit_images = config.num_fit_images
        self.fit_levels = config.fit.make()
        self.groups = config.groups.make()
        self.levels: Tensor | None = None
        self.thresholds: Tensor | None = None

    def fit(self, sample: Tensor, /) -> None:
        """Fit one table per channel group.

        Args:
          sample: ``[N, C, H, W]`` raw latents.

        Raises:
          ValueError: The sample is not 4-D or not finite.

        """
        if sample.ndim != 4:
            raise ValueError(f"fit expects [N, C, H, W]; got {tuple(sample.shape)}.")
        channels = sample.shape[1]
        scale = torch.stack([sample[:, c].double().std() for c in range(channels)])
        groups = self.groups(scale)
        levels = torch.empty(channels, NUM_LEVELS, dtype=torch.float32)
        for group in groups.unique():
            members = (groups == group).nonzero().flatten()
            values = sample[:, members].reshape(-1)
            levels[members] = self.fit_levels(values).float()
        self.load_table({"levels": levels, "thresholds": midpoints(levels)})

    def table(self) -> dict[str, Tensor]:
        """Return the fitted table.

        Returns:
          table: ``levels`` ``[C, 256]`` and ``thresholds`` ``[C, 255]``, float32.

        Raises:
          RuntimeError: The codec was never fitted or loaded.

        """
        levels, thresholds = self._fitted()
        return {"levels": levels.clone(), "thresholds": thresholds.clone()}

    def load_table(self, table: Mapping[str, Tensor], /) -> None:
        """Restore a fitted table.

        Args:
          table: ``levels`` and ``thresholds`` as :meth:`table` returns them.

        Raises:
          ValueError: The table is malformed or its levels decrease.

        """
        levels = table["levels"].float()
        thresholds = table["thresholds"].float()
        if levels.ndim != 2 or levels.shape[1] != NUM_LEVELS:
            raise ValueError(
                f"levels must be [C, {NUM_LEVELS}]; got {tuple(levels.shape)}.",
            )
        if thresholds.shape != (levels.shape[0], NUM_LEVELS - 1):
            raise ValueError("thresholds must be [C, 255], one row per level row.")
        if not bool((levels[:, 1:] >= levels[:, :-1]).all()):
            raise ValueError("levels must be non-decreasing in every row.")
        self.levels = levels
        self.thresholds = thresholds

    def encode(self, latent: Tensor, /) -> Tensor:
        """Return each scalar's nearest-level index.

        Args:
          latent: ``[..., C, H, W]`` finite raw latent.

        Returns:
          stored: ``[..., C, H, W]`` uint8.

        """
        _, thresholds = self._fitted()
        rows = latent.movedim(-3, 0)
        indices = quantize(rows.reshape(rows.shape[0], -1).float(), thresholds)
        return indices.to(torch.uint8).reshape(rows.shape).movedim(0, -3)

    def decode(self, stored: Tensor, /) -> Tensor:
        """Look up each index's level.

        Args:
          stored: ``[..., C, H, W]`` uint8 indices.

        Returns:
          latent: ``[..., C, H, W]`` float32.

        """
        levels, _ = self._fitted()
        rows = stored.movedim(-3, 0)
        values = dequantize(rows.reshape(rows.shape[0], -1), levels)
        return values.reshape(rows.shape).movedim(0, -3)

    def saturated(self, latent: Tensor, /) -> Tensor:
        """Return where ``latent`` lies outside its table's outermost levels.

        Args:
          latent: ``[..., C, H, W]`` raw latent.

        Returns:
          outside: Boolean tensor shaped like ``latent``.

        """
        levels, _ = self._fitted()
        shape = (-1,) + (1,) * 2
        low = levels[:, 0].view(shape)
        high = levels[:, -1].view(shape)
        return (latent < low) | (latent > high)

    def _fitted(self) -> tuple[Tensor, Tensor]:
        """Return the table, or raise if there is none."""
        if self.levels is None or self.thresholds is None:
            raise RuntimeError("ScalarTableCodec has no table: fit or load one first.")
        return self.levels, self.thresholds


def bits_per_scalar(codec: LatentCodec) -> int:
    """Return the stored bits per latent scalar, ignoring any table.

    Args:
      codec: A built codec.

    Returns:
      bits: ``8 * itemsize`` of the stored dtype.

    """
    return 8 * torch.empty(0, dtype=codec.stored_dtype).element_size()


def entropy_bits(stored: Tensor) -> float:
    """Return the empirical entropy of a uint8 index tensor, in bits per index.

    Below 8 means an entropy coder could shrink the corpus further.

    Args:
      stored: uint8 indices.

    Returns:
      entropy: Bits per index.

    """
    counts = torch.bincount(stored.flatten().long(), minlength=NUM_LEVELS).double()
    probability = counts[counts > 0] / counts.sum()
    return (
        float(-(probability * probability.log2()).sum())
        if probability.numel()
        else math.nan
    )
