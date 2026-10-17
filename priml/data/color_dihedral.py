"""A shared color permutation and square symmetry for a puzzle's grids.

:class:`ColorDihedral` samples one color permutation and one of the eight
symmetries of the square per puzzle, encodes the pair into the puzzle's
identifier, and decodes it back; :meth:`ColorDihedral.augment_tokens` applies
fresh draws per row to batches of flattened square token grids.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import functools
import math

from configgle import Fig
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

        separator: str = ""
        """Identifier delimiter; filled by the owning ARC augmentation."""

    def __init__(self, config: Config) -> None:
        if not config.transforms or any(t < 0 or t > 7 for t in config.transforms):
            raise ValueError("transforms must contain identifiers in 0..7.")
        if len(set(config.colors)) != len(config.colors) or any(
            c < 0 or c > 9 for c in config.colors
        ):
            raise ValueError("colors must contain distinct identifiers in 0..9.")
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
        if not self.config.separator:
            raise ValueError("identifier separator must be filled before sampling.")
        tid = self.config.transforms[int(rng.integers(len(self.config.transforms)))]
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
        if not separator:
            raise ValueError("identifier separator must be filled before decoding.")
        if separator not in name:
            return name, lambda grid: grid
        tid_text, permutation = name.split(separator)[-2:]
        tid = int(tid_text.removeprefix("t"))
        # A negative id would index the inverse table from the end and decode silently.
        if tid < 0 or tid > 7:
            raise ValueError("Encoded transform must be in 0..7.")
        # A non-bijective suffix would misroute colors and silently miscompare hashes.
        if len(permutation) != 10 or set(permutation) != set("0123456789"):
            raise ValueError(
                f"invalid color-permutation suffix {permutation!r} in identifier "
                f"{name!r}; expected a permutation of '0123456789'.",
            )
        inverse_tid = (0, 3, 2, 1, 4, 5, 6, 7)[tid]
        inverse_colors = np.argsort(list(permutation)).astype(np.uint8)

        def transform(grid: NDArray[np.uint8]) -> NDArray[np.uint8]:
            return inverse_colors[dihedral_transform(grid, tid=inverse_tid)]

        return name.partition(separator)[0], transform

    def apply_explicit(
        self,
        grid: NDArray[np.uint8],
        *,
        tid: int,
        colorperm_slots: tuple[int, ...],
    ) -> NDArray[np.uint8]:
        """Apply one recorded D4 id and color-slot permutation."""
        permutation = _explicit_color_permutation(colorperm_slots)
        return dihedral_transform(permutation[grid], tid=tid)

    def inverse_explicit(
        self,
        grid: NDArray[np.uint8],
        *,
        tid: int,
        colorperm_slots: tuple[int, ...],
    ) -> NDArray[np.uint8]:
        """Restore a canonical grid from its recorded D4 and color view.

        Args:
          grid: Augmented colors in transformed spatial order.
          tid: Recorded dihedral transform index.
          colorperm_slots: Source color-slot ordering, with background omitted.

        Returns:
          canonical: Restored two-dimensional ARC color grid.

        """
        if tid < 0 or tid > 7:
            raise ValueError(f"Invalid dihedral tid={tid}; must be in 0..7.")
        inverse_tid = (0, 3, 2, 1, 4, 5, 6, 7)[tid]
        inverse_colors = np.argsort(
            _explicit_color_permutation(colorperm_slots),
        ).astype(np.uint8)
        return inverse_colors[dihedral_transform(grid, tid=inverse_tid)]

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
        return np.rot90(arr)
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
        return np.fliplr(np.rot90(arr))
    raise ValueError(f"Invalid dihedral tid={tid}; must be in 0..7.")


def _explicit_color_permutation(colorperm_slots: tuple[int, ...]) -> NDArray[np.uint8]:
    """Return a validated map from original colors to explicit color slots."""
    if len(colorperm_slots) != 9 or set(colorperm_slots) != set(range(9)):
        raise ValueError("colorperm_slots must be a permutation of 0..8.")
    return np.array([0, *(slot + 1 for slot in colorperm_slots)], dtype=np.uint8)


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
