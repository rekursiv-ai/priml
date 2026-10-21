"""Configurable ARC view generation and spatial packing for prepared datasets.

Token layout: ``0`` pad, ``1`` EOS (content boundary), ``2``-``11`` colors. An
augmented puzzle name encodes its view as
``"{name}|||t{tid}|||{''.join(str(x) for x in mapping)}"``: a dihedral id and a
color permutation that fixes 0. :func:`inverse_aug` decodes it for voting.
"""

from __future__ import annotations

from dataclasses import field
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Self, cast, override

import hashlib
import math

from configgle import Fig, Makeable

import numpy as np

from priml.data.color_dihedral import ColorDihedral


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from numpy.typing import NDArray


class ArcSpec(Fig["ArcSpec"]):
    """The dataset-owned ARC token format and packed grid geometry."""

    max_grid: int = 30
    """Square side of a packed ARC grid."""

    vocab_pad: int = 0
    """Pad token id."""

    vocab_eos: int = 1
    """End-of-sequence token id."""

    vocab_color_offset: int = 2
    """First color token id; colors occupy ``[offset, offset + 10)``."""

    puzzle_id_separator: str = "|||"
    """Delimiter joining the name, dihedral transform, and color permutation."""

    @property
    def vocab_size(self) -> int:
        """Total token count, including ten colors."""
        return self.vocab_color_offset + 10

    @property
    def grid_shape(self) -> tuple[int, ...]:
        """Flat packed token-row shape."""
        return (self.max_grid**2,)


NO_TRAIN_SCALE_WEIGHTS: Final[Mapping[int, float]] = MappingProxyType({1: 1.0})
"""Identity scale distribution (factor 1): the scale gate never changes a grid."""


class SpatialAugmentation:
    """Scale and translate paired grids, then encode them as square token rows."""

    class Config(Fig["SpatialAugmentation"]):
        spec: ArcSpec | None = None
        """Dataset-owned token vocabulary and packed grid geometry."""

        train_scale_weights: dict[int, float] = field(default_factory=lambda: {1: 1.0})
        """Relative sampling weights for integer training scales."""

        translation_prob: float = 1.0
        """Probability of translating an eligible training example."""

        scale_prob: float = 1.0
        """Probability of sampling a scale for an eligible training example."""

    def __init__(self, config: Config) -> None:
        if config.spec is None:
            raise ValueError("Spatial augmentation spec must be filled by its owner.")
        if config.spec.max_grid < 1:
            raise ValueError("max_grid must be positive.")
        self.spec = config.spec
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
        rows, _ = self.pack_with_tags(inp, out=out, training=training, rng=rng)
        return rows

    def pack_with_tags(
        self,
        inp: np.ndarray[tuple[int, ...], np.dtype[np.uint8]],
        *,
        out: np.ndarray[tuple[int, ...], np.dtype[np.uint8]],
        training: bool,
        rng: np.random.Generator,
    ) -> tuple[list[NDArray[np.uint8]], tuple[int, int, int]]:
        """Pack paired grids and return the spatial tag they share.

        Draws exactly what :meth:`pack` draws, in the same order.

        Args:
          inp: Input colors in 0..9.
          out: Target colors in 0..9.
          training: Apply spatial augmentation; false preserves the canonical view.
          rng: Dataset builder's shared random stream.

        Returns:
          rows: Input and target token rows.
          tag: Shared ``(scale, row_offset, col_offset)``.

        """
        side = self.spec.max_grid
        if max(*inp.shape, *out.shape) > side:
            raise ValueError(
                f"grid shape exceeds max_grid={side}: inp={inp.shape}, out={out.shape}.",
            )
        scale = (
            sample_scale_factor(
                inp,
                out,
                self.config.train_scale_weights,
                rng,
                max_grid=side,
            )
            if training and bernoulli(self.config.scale_prob, rng)
            else 1
        )
        pad_r = pad_c = 0
        if training and bernoulli(self.config.translation_prob, rng):
            rows = max(inp.shape[0], out.shape[0]) * scale
            cols = max(inp.shape[1], out.shape[1]) * scale
            pad_r = int(rng.integers(side - rows + 1))
            pad_c = int(rng.integers(side - cols + 1))
        tag = (scale, pad_r, pad_c)
        return self.pack_at(inp, out=out, tag=tag), tag

    def pack_at(
        self,
        inp: np.ndarray[tuple[int, ...], np.dtype[np.uint8]],
        *,
        out: np.ndarray[tuple[int, ...], np.dtype[np.uint8]],
        tag: tuple[int, int, int],
    ) -> list[NDArray[np.uint8]]:
        """Pack both grids under one preselected spatial tag.

        Args:
          inp: Input colors in 0..9.
          out: Target colors in 0..9.
          tag: Shared ``(scale, row_offset, col_offset)``.

        Returns:
          rows: Input and target token rows under the same transform.

        """
        side = self.spec.max_grid
        scale, pad_r, pad_c = tag
        if scale < 1 or pad_r < 0 or pad_c < 0:
            raise ValueError(f"tag needs scale >= 1 and offsets >= 0; got {tag}.")
        if max(*inp.shape, *out.shape) > side:
            raise ValueError(
                f"grid shape exceeds max_grid={side}: inp={inp.shape}, out={out.shape}.",
            )
        if (
            pad_r + max(inp.shape[0], out.shape[0]) * scale > side
            or pad_c + max(inp.shape[1], out.shape[1]) * scale > side
        ):
            raise ValueError(f"tag {tag} places the grid outside max_grid={side}.")
        return [self.pack_one_at(grid, tag=tag) for grid in (inp, out)]

    def pack_one_at(
        self,
        grid: np.ndarray[tuple[int, ...], np.dtype[np.uint8]],
        *,
        tag: tuple[int, int, int],
    ) -> NDArray[np.uint8]:
        """Pack one grid at a preselected CanvasEOS placement.

        Args:
          grid: Canonical grid whose colors are shifted into the token vocabulary.
          tag: Scale, row offset, and column offset shared with its paired target.

        Returns:
          media: Flattened CanvasEOS token grid.

        """
        side = self.spec.max_grid
        scale, pad_r, pad_c = tag
        if scale < 1 or pad_r < 0 or pad_c < 0:
            raise ValueError(f"tag needs scale >= 1 and offsets >= 0; got {tag}.")
        if max(*grid.shape) > side:
            raise ValueError(f"grid shape exceeds max_grid={side}: grid={grid.shape}.")
        if scale != 1:
            grid = cast(
                np.ndarray[tuple[int, ...], np.dtype[np.uint8]],
                scale_grid(grid, scale),
            )
        nrow, ncol = grid.shape
        if nrow > side or ncol > side:
            raise ValueError(f"tag {tag} places the grid outside max_grid={side}.")
        if pad_r + nrow > side or pad_c + ncol > side:
            raise ValueError(f"tag {tag} places the grid outside max_grid={side}.")
        padded = np.pad(
            grid + self.spec.vocab_color_offset,
            ((pad_r, side - pad_r - nrow), (pad_c, side - pad_c - ncol)),
            constant_values=self.spec.vocab_pad,
        )
        eos_row, eos_col = pad_r + nrow, pad_c + ncol
        if eos_row < side:
            padded[eos_row, pad_c:eos_col] = self.spec.vocab_eos
        if eos_col < side:
            padded[pad_r:eos_row, eos_col] = self.spec.vocab_eos
        return padded.flatten()


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

        spec: ArcSpec | None = None
        """Token vocabulary and packed grid geometry of this dataset."""

        spatial: SpatialAugmentation.Config = field(
            default_factory=SpatialAugmentation.Config,
        )
        """Training-only scale/translation and token packing policy."""

        spatial_eval_views: bool = False
        """Add one spatially transformed view of every test puzzle when preparing."""

        spatial_eval_scale: int = 2
        """Integer scale of the spatial evaluation views."""

        @override
        def finalize(self) -> Self:
            """Push the dataset's geometry into its transform and spatial packer."""
            self.spatial.spec = self.spec
            if (
                self.spec is not None
                and isinstance(self.transform, ColorDihedral.Config)
                and not self.transform.separator
            ):
                self.transform.separator = self.spec.puzzle_id_separator
            return super().finalize()

    def __init__(self, config: Config) -> None:
        if config.spec is None:
            raise ValueError("ARC augmentation spec must be filled by its dataset.")
        if config.num_aug < 0 or config.retries_factor < 1:
            raise ValueError("num_aug must be nonnegative and retries_factor positive.")
        self.spec = config.spec
        self.config = config
        self.transform = config.transform.make()
        self.spatial = config.spatial.make()


def inverse_aug(
    name: str,
    *,
    spec: ArcSpec,
) -> tuple[str, Callable[[NDArray[np.uint8]], NDArray[np.uint8]]]:
    """Decode an augmented identifier into its original name and inverse view.

    Args:
      name: Encoded identifier like ``"abc|||t3|||0123456789"``, or a bare name.
      spec: Dataset token layout and identifier separator.

    Returns:
      name: The portion before the first separator.
      inverse: Maps a grid in the augmented frame back to the canonical frame:
        inverse dihedral first, then the inverse color permutation.

    """
    return ColorDihedral.Config(separator=spec.puzzle_id_separator).make().inverse(name)


def untranslate_unscale(
    flat: NDArray[np.uint8],
    *,
    scale: int,
    pad_r: int,
    pad_c: int,
) -> NDArray[np.uint8]:
    """Invert :meth:`SpatialAugmentation.pack`'s scale+translate on a flat grid.

    Slices the content window at ``(pad_r, pad_c)``, keeps the top-left token of
    each ``scale x scale`` block, then re-pads to the square, top-left anchored
    frame :func:`crop_grid` expects. A model prediction need not have constant
    blocks; it is judged by each block's top-left token.

    Args:
      flat: Square flat token grid in the augmented frame.
      scale: Forward block-upscale factor (>= 1).
      pad_r: Forward top-left row pad.
      pad_c: Forward top-left column pad.

    Returns:
      flat: Square flat token grid in the canonical frame.

    """
    if scale < 1:
        raise ValueError(f"scale must be >= 1, got {scale}.")
    if pad_r < 0 or pad_c < 0:
        raise ValueError(f"pads must be >= 0, got ({pad_r}, {pad_c}).")
    # Validated before the identity fast path so non-square input always fails.
    side = square_side(len(flat), who="untranslate_unscale")
    if scale == 1 and pad_r == 0 and pad_c == 0:
        return flat
    shifted = flat.reshape(side, side)[pad_r:, pad_c:][::scale, ::scale]
    return np.pad(
        shifted,
        ((0, side - shifted.shape[0]), (0, side - shifted.shape[1])),
    ).flatten()


def grid_hash(grid: NDArray[np.generic]) -> str:
    """Hash a 2D uint8 grid's shape and content."""
    if grid.ndim != 2:
        raise ValueError("Expected grid.ndim == 2.")
    if grid.dtype != np.uint8:
        raise ValueError("Expected grid.dtype == np.uint8.")
    return hashlib.sha256(bytes(grid.shape) + grid.tobytes()).hexdigest()


def normalize_scale_weights(
    train_scale_weights: Mapping[int, float],
) -> dict[int, float]:
    """Validate integer scale weights and normalize them to probabilities.

    Args:
      train_scale_weights: Scale factor -> nonnegative weight.

    Returns:
      weights: Scale -> probability over the positive-weight scales, ascending.

    """
    if not train_scale_weights:
        raise ValueError("train_scale_weights must contain at least one scale.")
    total = 0.0
    normalized: dict[int, float] = {}
    for scale, weight in sorted(train_scale_weights.items()):
        if scale <= 0:
            raise ValueError(f"scale factors must be positive: {train_scale_weights}.")
        if weight < 0 or math.isnan(weight) or math.isinf(weight):
            raise ValueError(
                f"scale weights must be finite and nonnegative: {train_scale_weights}.",
            )
        if weight == 0:
            continue
        normalized[int(scale)] = float(weight)  # pyrefly: ignore[unnecessary-type-conversion] -- The stub's type hides the runtime value (numpy/torch scalar or non-str key); the conversion is load-bearing.
        total += float(weight)
    if total <= 0:
        raise ValueError(
            f"scale weights must include a positive weight: {train_scale_weights}.",
        )
    # Rescale only on overflow: the recipe sentinels record the result, so finite
    # totals must keep their exact quotients or a built tree stops matching.
    if math.isinf(total):
        peak = max(normalized.values())
        normalized = {scale: weight / peak for scale, weight in normalized.items()}
        total = sum(normalized.values())
    return {scale: weight / total for scale, weight in normalized.items()}


def scale_grid[T: np.generic](grid: NDArray[T], scale: int) -> NDArray[T]:
    """Nearest-neighbor upscale a grid by an integer factor."""
    return np.repeat(np.repeat(grid, scale, axis=0), scale, axis=1)


# Skipping the draw at the boundaries keeps an always/never policy's stream identical
# to a build without the gate.
def bernoulli(prob: float, rng: np.random.Generator) -> bool:
    """Draw Bernoulli(``prob``), consuming no randomness when ``prob`` is 0 or 1."""
    if prob >= 1.0:
        return True
    if prob <= 0.0:
        return False
    return rng.random() < prob


def sample_scale_factor(
    inp: NDArray[np.uint8],
    out: NDArray[np.uint8],
    train_scale_weights: Mapping[int, float],
    rng: np.random.Generator,
    *,
    max_grid: int,
) -> int:
    """Sample a scale factor that fits both grids of a pair in ``max_grid``.

    Draws nothing when the only fitting factor is 1 (or none fits): the result
    cannot vary, and the identity policy must reproduce the unscaled stream. A
    singleton ``{k > 1}`` still draws.

    Args:
      inp: Input grid.
      out: Output grid.
      train_scale_weights: Scale factor -> sampling weight.
      rng: Dataset builder's shared random stream.
      max_grid: Square side the packed grid must fit.

    Returns:
      scale: Sampled factor.

    """
    normalized = normalize_scale_weights(train_scale_weights)
    inp_rows, inp_cols = cast(tuple[int, int], inp.shape)
    out_rows, out_cols = cast(tuple[int, int], out.shape)
    rows, cols = max(inp_rows, out_rows), max(inp_cols, out_cols)
    fitting = [
        (scale, weight)
        for scale, weight in normalized.items()
        if rows * scale <= max_grid and cols * scale <= max_grid
    ]
    if not fitting or [scale for scale, _ in fitting] == [1]:
        return 1
    scales = tuple(scale for scale, _ in fitting)
    weights = tuple(weight for _, weight in fitting)
    return int(rng.choice(scales, p=np.array(weights) / sum(weights)))


def crop_grid(
    flat: NDArray[np.uint8],
    *,
    spec: ArcSpec,
    row: int = 0,
    col: int = 0,
) -> NDArray[np.uint8]:
    """Recover the largest all-color rectangle from a token-grid origin.

    Args:
      flat: Square flat token grid; the side is inferred from its length.
      spec: Dataset token vocabulary.
      row: First CanvasEOS row containing the grid.
      col: First CanvasEOS column containing the grid.

    Returns:
      grid: Color grid with values 0..9.

    """
    side = square_side(len(flat), who="crop_grid")
    if row < 0 or col < 0 or row >= side or col >= side:
        raise ValueError(f"crop origin ({row}, {col}) is outside side {side}.")
    grid = flat.reshape(side, side)
    values = cast(list[int], flat.tolist())
    max_area = max_height = max_width = 0
    num_c = side - col
    for height in range(1, side - row + 1):
        row_values = values[(row + height - 1) * side + col : (row + height) * side]
        for width in range(1, num_c + 1):
            token = row_values[width - 1]
            if token < spec.vocab_color_offset or token >= spec.vocab_size:
                num_c = width - 1
                break
        if height * num_c > max_area:
            max_area, max_height, max_width = height * num_c, height, num_c
    return (
        grid[row : row + max_height, col : col + max_width] - spec.vocab_color_offset
    ).astype(np.uint8)


def square_side(length: int, *, who: str) -> int:
    """Return ``sqrt(length)`` for a square flat grid; reject any other length."""
    side = math.isqrt(length)
    if side * side != length:
        raise ValueError(f"{who} expects a square flat grid, got length {length}.")
    return side


def arc_grid_to_np(
    grid: Sequence[Sequence[int | float]],
    *,
    max_grid: int,
) -> NDArray[np.uint8]:
    """Validate a source color grid before narrowing it to uint8.

    Args:
      grid: Rectangular source grid with colors in 0..9.
      max_grid: Largest permitted side length.

    Returns:
      grid: Validated uint8 array.

    """
    arr = np.array(grid, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("Expected arr.ndim == 2.")
    if arr.shape[0] > max_grid:
        raise ValueError(f"Expected arr.shape[0] <= max_grid={max_grid}.")
    if arr.shape[1] > max_grid:
        raise ValueError(f"Expected arr.shape[1] <= max_grid={max_grid}.")
    # Checked before any integer cast, so 256 cannot wrap and 9.9 cannot truncate
    # into a valid color.
    if not np.all(np.isin(arr, np.arange(10))):
        raise ValueError("ARC grid colors must be in 0..9.")
    return arr.astype(np.uint8)
