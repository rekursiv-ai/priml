"""Configurable ARC view generation and spatial packing for prepared datasets.

Token layout: ``0`` pad, ``1`` EOS (content boundary), ``2``-``11`` colors. An
augmented puzzle name encodes its view as
``"{name}|||t{tid}|||{''.join(str(x) for x in mapping)}"``: a dihedral id and a
color permutation that fixes 0. :func:`inverse_aug` decodes it for voting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, cast

import functools
import hashlib
import math

from configgle import Fig, Makeable
from torch import Tensor

import numpy as np
import torch

from priml.lib.custom_json import ListCodec


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from numpy.typing import NDArray


@dataclass(slots=True, kw_only=True, frozen=True)
class ArcSpec:
    """The ARC token format, grouped into one named scope."""

    puzzle_id_separator: str = "|||"
    """Delimiter joining ``{name}``, ``t{tid}``, and the color permutation."""

    max_grid: int = 30
    """Square side an ARC grid is packed/padded to (the real 900-token grid)."""

    vocab_pad: int = 0
    """Pad token id."""

    vocab_eos: int = 1
    """End-of-sequence token id (content boundary marker)."""

    vocab_color_offset: int = 2
    """First color token id; colors occupy ``[offset, offset + 10)``."""

    aug_retries_factor: int = 5
    """Rejection-sampling retry budget per augmentation, as a multiple of count."""

    @property
    def vocab_size(self) -> int:
        """Total token count: pad + eos + 10 colors (12 for the default offset)."""
        return self.vocab_color_offset + 10


ARC: Final = ArcSpec()
"""The ARC token format shared by builders, loaders, and pass@K voting."""

NO_TRAIN_SCALE_WEIGHTS: Final[Mapping[int, float]] = MappingProxyType({1: 1.0})
"""Identity scale distribution (factor 1): the scale gate never changes a grid."""


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
        side = self.config.max_grid
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
            pad_r = int(rng.integers(0, side - rows + 1))
            pad_c = int(rng.integers(0, side - cols + 1))
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
        side = self.config.max_grid
        scale, pad_r, pad_c = tag
        if scale < 1 or pad_r < 0 or pad_c < 0:
            raise ValueError(f"tag needs scale >= 1 and offsets >= 0; got {tag}.")
        if max(*inp.shape, *out.shape) > side:
            raise ValueError(
                f"grid shape exceeds max_grid={side}: inp={inp.shape}, out={out.shape}.",
            )
        if scale > 1:
            inp = cast(
                np.ndarray[tuple[int, ...], np.dtype[np.uint8]],
                scale_grid(inp, scale),
            )
            out = cast(
                np.ndarray[tuple[int, ...], np.dtype[np.uint8]],
                scale_grid(out, scale),
            )
        if (
            pad_r + max(inp.shape[0], out.shape[0]) > side
            or pad_c + max(inp.shape[1], out.shape[1]) > side
        ):
            raise ValueError(f"tag {tag} places the grid outside max_grid={side}.")
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

        spatial_eval_views: bool = False
        """Add one spatially transformed view of every test puzzle when preparing."""

        spatial_eval_scale: int = 2
        """Integer scale of the spatial evaluation views."""

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
    raise ValueError(f"Invalid dihedral tid={tid}; must be in 0..7.")


def inverse_dihedral_transform[T: np.generic](
    arr: NDArray[T],
    *,
    tid: int,
) -> NDArray[T]:
    """Undo :func:`dihedral_transform` with the same ``tid``."""
    # Reflections and rot180 are self-inverse; rot90 and rot270 swap.
    return dihedral_transform(arr, tid=(0, 3, 2, 1, 4, 5, 6, 7)[tid])


def inverse_aug(
    name: str,
) -> tuple[str, Callable[[NDArray[np.uint8]], NDArray[np.uint8]]]:
    """Decode an augmented identifier into its original name and inverse view.

    Args:
      name: Encoded identifier like ``"abc|||t3|||0123456789"``, or a bare name.

    Returns:
      name: The portion before the first separator.
      inverse: Maps a grid in the augmented frame back to the canonical frame:
        inverse dihedral first, then the inverse color permutation.

    """
    separator = ARC.puzzle_id_separator
    if separator not in name:
        return name, lambda x: x
    tid_str, perm_str = name.split(separator)[-2:]
    tid = int(tid_str[1:])
    # A non-bijective suffix would misroute colors and silently miscompare hashes.
    if len(perm_str) != 10 or set(perm_str) != set("0123456789"):
        raise ValueError(
            f"invalid color-permutation suffix {perm_str!r} in identifier {name!r}; "
            "expected a permutation of '0123456789'.",
        )
    inv_perm = np.argsort(list(perm_str)).astype(np.uint8)

    def _map_grid(grid: NDArray[np.uint8]) -> NDArray[np.uint8]:
        return inv_perm[inverse_dihedral_transform(grid, tid=tid)]

    return name.split(separator, maxsplit=1)[0], _map_grid


def canonicalize_arc_grid(
    tokens: Tensor,
    *,
    name: str,
    spatial_tags: Tensor,
    transform: ColorDihedral | None = None,
) -> tuple[str, Tensor]:
    """Undo a view's spatial tag, crop to its colors, and invert its color/symmetry.

    Args:
      tokens: Flattened square input or predicted token grid.
      name: Augmented puzzle identifier encoding the color/dihedral view.
      spatial_tags: ``(scale, row_offset, col_offset)`` of this view.
      transform: Policy that encoded ``name``; ``None`` is the default one.

    Returns:
      original_name: Source puzzle name.
      canonical: Cropped raw-color grid in 0..9, on ``tokens``'s device.

    """
    flat = tokens.detach().to("cpu", torch.uint8).reshape(-1).numpy()
    tag = ListCodec.coerce(spatial_tags.reshape(-1).tolist(), int)
    if len(tag) != 3:
        raise ValueError(f"spatial tags hold (scale, row, col); got {tag}.")
    scale, pad_r, pad_c = tag
    colors = crop_grid(untranslate_unscale(flat, scale=scale, pad_r=pad_r, pad_c=pad_c))
    policy = transform if transform is not None else ColorDihedral.Config().make()
    original_name, inverse = policy.inverse(name)
    canonical = np.array(inverse(colors), copy=True)
    return original_name, torch.from_numpy(canonical).to(tokens.device)


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
    shifted = flat.reshape(side, side)[pad_r:, pad_c:]
    if scale > 1:
        shifted = shifted[::scale, ::scale]
    return np.pad(
        shifted,
        ((0, side - shifted.shape[0]), (0, side - shifted.shape[1])),
    ).flatten()


def grid_hash(grid: NDArray[np.uint8]) -> str:
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
        normalized[int(scale)] = float(weight)
        total += float(weight)
    if total <= 0:
        raise ValueError(
            f"scale weights must include a positive weight: {train_scale_weights}.",
        )
    return {scale: weight / total for scale, weight in normalized.items()}


def scale_weights_slug(train_scale_weights: Mapping[int, float]) -> str:
    """Return a stable path slug such as ``1w0p5-2w0p5`` for scale weights."""
    return "-".join(
        f"{scale}w{str(weight).replace('.', 'p')}"
        for scale, weight in normalize_scale_weights(train_scale_weights).items()
    )


def parse_scale_weights(values: Sequence[str]) -> dict[int, float]:
    """Parse and normalize CLI scale weights.

    Args:
      values: Strings written ``SCALE=WEIGHT``, such as ``"2=0.5"``.

    Returns:
      weights: Normalized scale -> probability.

    """
    result: dict[int, float] = {}
    for value in values:
        scale_text, sep, weight_text = value.partition("=")
        if not sep:
            raise ValueError(f"scale weight must be SCALE=WEIGHT, got {value!r}.")
        result[int(scale_text)] = float(weight_text)
    return normalize_scale_weights(result)


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
    return bool(rng.random() < prob)


def sample_scale_factor(
    inp: NDArray[np.uint8],
    out: NDArray[np.uint8],
    train_scale_weights: Mapping[int, float],
    rng: np.random.Generator,
    *,
    max_grid: int = ARC.max_grid,
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
    inp_rows, inp_cols = ListCodec.coerce(list(inp.shape), int)
    out_rows, out_cols = ListCodec.coerce(list(out.shape), int)
    rows, cols = max(inp_rows, out_rows), max(inp_cols, out_cols)
    fitting = [
        (scale, weight)
        for scale, weight in normalized.items()
        if rows * scale <= max_grid and cols * scale <= max_grid
    ]
    if not fitting or [scale for scale, _ in fitting] == [1]:
        return 1
    scales = np.array([scale for scale, _ in fitting], dtype=np.int64)
    weights = np.array([weight for _, weight in fitting], dtype=np.float64)
    return int(rng.choice(scales, p=weights / weights.sum()))


def crop_grid(flat: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Recover the largest top-left all-color rectangle from a flat token grid.

    Args:
      flat: Square flat token grid; the side is inferred from its length.

    Returns:
      grid: Color grid with values 0..9.

    """
    side = square_side(len(flat), who="crop_grid")
    grid = flat.reshape(side, side)
    values = ListCodec.coerce(cast(object, flat.tolist()), int)
    max_area = max_nr = max_nc = 0
    num_c = side
    for num_r in range(1, side + 1):
        row = values[(num_r - 1) * side : num_r * side]
        for c in range(1, num_c + 1):
            if row[c - 1] < ARC.vocab_color_offset or row[c - 1] >= ARC.vocab_size:
                num_c = c - 1
                break
        if num_r * num_c > max_area:
            max_area, max_nr, max_nc = num_r * num_c, num_r, num_c
    return (grid[:max_nr, :max_nc] - ARC.vocab_color_offset).astype(np.uint8)


def square_side(length: int, *, who: str) -> int:
    """Return ``sqrt(length)`` for a square flat grid; reject any other length."""
    side = math.isqrt(length)
    if side * side != length:
        raise ValueError(f"{who} expects a square flat grid, got length {length}.")
    return side


def arc_grid_to_np(grid: list[list[int]], *, max_grid: int) -> NDArray[np.uint8]:
    """Validate a source color grid before narrowing it to uint8.

    Args:
      grid: Rectangular source grid with colors in 0..9.
      max_grid: Largest permitted side length.

    Returns:
      grid: Validated uint8 array.

    """
    arr = np.array(grid, dtype=np.int64)
    if arr.ndim != 2:
        raise ValueError("Expected arr.ndim == 2.")
    if arr.shape[0] > max_grid:
        raise ValueError("Expected arr.shape[0] <= ARC.max_grid.")
    if arr.shape[1] > max_grid:
        raise ValueError("Expected arr.shape[1] <= ARC.max_grid.")
    # Checked on the wide dtype, so 256 is rejected rather than wrapping to a color.
    if not np.all((arr >= 0) & (arr <= 9)):
        raise ValueError("ARC grid colors must be in 0..9.")
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
