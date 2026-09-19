"""Configurable ARC view generation and spatial packing for prepared datasets."""

from __future__ import annotations

from dataclasses import field
from typing import TYPE_CHECKING, cast

import functools
import hashlib
import math

from configgle import Fig, Makeable
from torch import Tensor

import numpy as np
import torch


if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray


class ColorDihedral:
    """Sample a shared color permutation and symmetry for one puzzle's examples."""

    class Config(Fig["ColorDihedral"]):
        transforms: tuple[int, ...] = tuple(range(8))
        """Dihedral transform identifiers to sample uniformly."""

        colors: tuple[int, ...] = tuple(range(1, 10))
        """Colors to permute; unlisted colors remain fixed."""

        separator: str = "|||"
        """Separator encoding the transform in a prepared puzzle identifier."""

    def __init__(self, config: Config) -> None:
        if not config.transforms or any(t < 0 or t > 7 for t in config.transforms):
            raise ValueError("transforms must contain identifiers in 0..7.")
        if len(set(config.colors)) != len(config.colors) or any(
            c < 0 or c > 9 for c in config.colors
        ):
            raise ValueError("colors must contain distinct identifiers in 0..9.")
        if not config.separator:
            raise ValueError("separator must be nonempty.")
        self.config = config

    def sample(
        self,
        name: str,
        *,
        rng: np.random.Generator,
    ) -> tuple[str, Callable[[NDArray[np.uint8]], NDArray[np.uint8]]]:
        """Return an encoded name and the transformation shared by its examples.

        Args:
          name: Original puzzle identifier.
          rng: Dataset builder's shared random stream.

        Returns:
          name: Identifier encoding the sampled symmetry and color permutation.
          transform: Callable applying that same view to inputs and labels.

        """
        tid = self.config.transforms[int(rng.integers(0, len(self.config.transforms)))]
        mapping = np.arange(10, dtype=np.uint8)
        colors = np.array(self.config.colors, dtype=np.uint8)
        mapping[colors] = rng.permutation(colors)

        def transform(grid: NDArray[np.uint8]) -> NDArray[np.uint8]:
            return dihedral_transform(np.take(mapping, grid), tid=tid)

        suffix = "".join(str(int(value)) for value in mapping)
        separator = self.config.separator
        return f"{name}{separator}t{tid}{separator}{suffix}", transform

    def inverse(
        self,
        name: str,
    ) -> tuple[str, Callable[[NDArray[np.uint8]], NDArray[np.uint8]]]:
        """Decode a prepared identifier and return its inverse grid transform.

        Args:
          name: Bare puzzle name or encoded augmented identifier.

        Returns:
          name: Original puzzle identifier.
          transform: Callable restoring the original colors and orientation.

        """
        separator = self.config.separator
        if separator not in name:
            return name, lambda grid: grid
        tid_text, permutation = name.split(separator)[-2:]
        if len(permutation) != 10 or set(permutation) != set("0123456789"):
            raise ValueError("Encoded colors must be a permutation of 0..9.")
        tid = int(tid_text.removeprefix("t"))
        if tid < 0 or tid > 7:
            raise ValueError("Encoded transform must be in 0..7.")
        inverse_tid = (0, 3, 2, 1, 4, 5, 6, 7)[tid]
        inverse_colors = np.argsort(list(permutation)).astype(np.uint8)

        def transform(grid: NDArray[np.uint8]) -> NDArray[np.uint8]:
            return inverse_colors[dihedral_transform(grid, tid=inverse_tid)]

        return name.split(separator, maxsplit=1)[0], transform

    def augment_tokens(
        self,
        inputs: Tensor,
        labels: Tensor,
        *,
        vocab_size: int,
        token_offset: int,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Apply paired symbol and grid permutations to a batch of square token grids.

        Args:
          inputs: Flattened square grids, one per row.
          labels: Matching target grids.
          vocab_size: Number of encoded token values.
          token_offset: Offset from raw symbols to encoded tokens.
          generator: Torch random stream; independent of offline NumPy sampling.

        Returns:
          inputs: Augmented input rows.
          labels: Target rows under the same transformations.

        """
        batch = inputs.shape[0]
        side = math.isqrt(inputs.shape[1])
        if side * side != inputs.shape[1]:
            raise ValueError("Token rows must represent square grids.")
        symbols = (
            torch.tensor(self.config.colors, device=inputs.device, dtype=torch.long)
            + token_offset
        )
        permutations = (
            torch.arange(vocab_size, device=inputs.device, dtype=torch.long)
            .expand(batch, -1)
            .clone()
        )
        draws = torch.rand(
            batch,
            len(self.config.colors),
            device=inputs.device,
            generator=generator,
        )
        permutations[:, symbols] = symbols[draws.argsort(dim=1)]
        inputs_aug = torch.gather(permutations, 1, inputs.long()).to(inputs.dtype)
        labels_aug = torch.gather(permutations, 1, labels.long()).to(labels.dtype)
        symmetries = _token_symmetries(side, self.config.transforms, inputs.device)
        choice = torch.randint(
            0,
            len(self.config.transforms),
            (batch,),
            device=inputs.device,
            generator=generator,
        )
        selected = symmetries[choice]
        return torch.gather(inputs_aug, 1, selected), torch.gather(
            labels_aug,
            1,
            selected,
        )


class SpatialAugmentation:
    """Scale and translate paired grids, then encode them as square token rows."""

    class Config(Fig["SpatialAugmentation"]):
        max_grid: int = 30
        """Side length of the padded token grid."""

        train_scale_weights: dict[int, float] = field(default_factory=lambda: {1: 1.0})
        """Relative sampling weights for integer training scales."""

        translation_prob: float = 1.0
        """Probability of translating an eligible training example."""

        scale_prob: float = 1.0
        """Probability of sampling a scale for an eligible training example."""

    def __init__(self, config: Config) -> None:
        if config.max_grid < 1:
            raise ValueError("max_grid must be positive.")
        for name, probability in (
            ("translation_prob", config.translation_prob),
            ("scale_prob", config.scale_prob),
        ):
            if not math.isfinite(probability) or probability < 0 or probability > 1:
                raise ValueError(f"{name} must be finite and in [0, 1].")
        weights = config.train_scale_weights
        if (
            not weights
            or any(
                scale < 1 or not math.isfinite(weight) or weight < 0
                for scale, weight in weights.items()
            )
            or sum(weights.values()) <= 0
        ):
            raise ValueError(
                "Scale weights require positive scales and finite nonnegative weights with a positive total.",
            )
        self.config = config
        total = sum(weight for _, weight in sorted(weights.items()) if weight > 0)
        self.weights = {
            scale: weight / total
            for scale, weight in sorted(weights.items())
            if weight > 0
        }

    def pack(
        self,
        inp: np.ndarray[tuple[int, ...], np.dtype[np.uint8]],
        out: np.ndarray[tuple[int, ...], np.dtype[np.uint8]],
        *,
        training: bool,
        rng: np.random.Generator,
    ) -> list[NDArray[np.uint8]]:
        """Pack paired grids with one shared scale and translation.

        Args:
          inp: Input colors in 0..9.
          out: Target colors in 0..9.
          training: Apply spatial augmentation; false preserves the canonical view.
          rng: Dataset builder's shared random stream.

        Returns:
          rows: Input and target token rows; 0 is padding, 1 EOS, 2..11 colors.

        """
        side = self.config.max_grid
        if max(*inp.shape, *out.shape) > side:
            raise ValueError(f"Grid shape exceeds max_grid={side}.")
        scale = (
            self._sample_scale(inp, out, rng=rng)
            if training and _bernoulli(self.config.scale_prob, rng=rng)
            else 1
        )
        if scale > 1:
            inp = cast(
                np.ndarray[tuple[int, ...], np.dtype[np.uint8]],
                np.repeat(np.repeat(inp, scale, axis=0), scale, axis=1),
            )
            out = cast(
                np.ndarray[tuple[int, ...], np.dtype[np.uint8]],
                np.repeat(np.repeat(out, scale, axis=0), scale, axis=1),
            )
        pad_r = pad_c = 0
        if training and _bernoulli(self.config.translation_prob, rng=rng):
            pad_r = int(rng.integers(0, side - max(inp.shape[0], out.shape[0]) + 1))
            pad_c = int(rng.integers(0, side - max(inp.shape[1], out.shape[1]) + 1))
        result: list[NDArray[np.uint8]] = []
        for grid in (inp, out):
            nrow, ncol = grid.shape
            padded = np.pad(
                grid + 2,
                ((pad_r, side - pad_r - nrow), (pad_c, side - pad_c - ncol)),
                constant_values=0,
            )
            eos_row, eos_col = pad_r + nrow, pad_c + ncol
            if eos_row < side:
                padded[eos_row, pad_c:eos_col] = 1
            if eos_col < side:
                padded[pad_r:eos_row, eos_col] = 1
            result.append(padded.flatten())
        return result

    def _sample_scale(
        self,
        inp: np.ndarray[tuple[int, ...], np.dtype[np.uint8]],
        out: np.ndarray[tuple[int, ...], np.dtype[np.uint8]],
        *,
        rng: np.random.Generator,
    ) -> int:
        fitting = [
            (scale, weight)
            for scale, weight in self.weights.items()
            if max(*inp.shape, *out.shape) * scale <= self.config.max_grid
        ]
        # The identity path consumes no draw, preserving the source RNG stream.
        if not fitting or [scale for scale, _ in fitting] == [1]:
            return 1
        scales = np.array([scale for scale, _ in fitting], dtype=np.int64)
        weights = np.array([weight for _, weight in fitting], dtype=np.float64)
        return int(rng.choice(scales, p=weights / weights.sum()))


class ArcAugmentation:
    """Own the offline ARC augmentation recipe and its injectable transforms."""

    class Config(Fig["ArcAugmentation"]):
        num_aug: int = 1_000
        """Maximum distinct augmented views per puzzle, besides its original view."""

        seed: int = 42
        """Seed for source shuffling, view generation, and spatial packing."""

        retries_factor: int = 5
        """Maximum sampling attempts per requested distinct view."""

        transform: Makeable[ColorDihedral] = field(default_factory=ColorDihedral.Config)
        """Color and symmetry policy shared by a puzzle's input/output examples."""

        spatial: SpatialAugmentation.Config = field(
            default_factory=SpatialAugmentation.Config,
        )
        """Training-only scale/translation and token packing policy."""

    def __init__(self, config: Config) -> None:
        if config.num_aug < 0 or config.retries_factor < 1:
            raise ValueError("num_aug must be nonnegative and retries_factor positive.")
        self.config = config
        self.transform = config.transform.make()
        self.spatial = config.spatial.make()


def dihedral_transform[T: np.generic](arr: NDArray[T], *, tid: int) -> NDArray[T]:
    """Apply one of the eight square symmetries to a grid.

    Args:
      arr: Two-dimensional color grid.
      tid: Symmetry identifier in 0..7.

    Returns:
      grid: Rotated, reflected, or transposed grid.

    """
    if tid == 0:
        return arr
    if tid == 1:
        return np.rot90(arr, k=1)
    if tid == 2:
        return np.rot90(arr, k=2)
    if tid == 3:
        return np.rot90(arr, k=3)
    if tid == 4:
        return np.fliplr(arr)
    if tid == 5:
        return np.flipud(arr)
    if tid == 6:
        return arr.T
    if tid == 7:
        return np.fliplr(np.rot90(arr, k=1))
    raise ValueError("Dihedral transform must be in 0..7.")


def grid_hash(grid: NDArray[np.uint8]) -> str:
    """Hash a uint8 grid's shape and content."""
    return hashlib.sha256(bytes(grid.shape) + grid.tobytes()).hexdigest()


def arc_grid_to_np(grid: list[list[int]], *, max_grid: int) -> NDArray[np.uint8]:
    """Validate a source color grid before narrowing it to uint8.

    Args:
      grid: Rectangular source grid with colors in 0..9.
      max_grid: Largest permitted side length.

    Returns:
      grid: Validated uint8 array.

    """
    arr = np.array(grid, dtype=np.int64)
    if arr.ndim != 2 or max(arr.shape) > max_grid:
        raise ValueError("Source grid must be two-dimensional and fit max_grid.")
    if not np.all((arr >= 0) & (arr <= 9)):
        raise ValueError("Source grid colors must be in 0..9.")
    return arr.astype(np.uint8)


@functools.cache
def _token_symmetries(
    side: int,
    transforms: tuple[int, ...],
    device: torch.device,
) -> Tensor:
    grid = np.arange(side * side, dtype=np.int64).reshape(side, side)
    return torch.from_numpy(
        np.stack([dihedral_transform(grid, tid=tid).reshape(-1) for tid in transforms]),
    ).to(device)


def _bernoulli(prob: float, *, rng: np.random.Generator) -> bool:
    if prob >= 1.0:
        return True
    if prob <= 0.0:
        return False
    return bool(rng.random() < prob)
