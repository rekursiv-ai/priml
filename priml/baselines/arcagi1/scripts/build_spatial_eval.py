"""Expand a prepared ARC eval split with clip-safe scale/translation views.

A post-process over an already-built tree. Each test puzzle is kept unchanged
(identity spatial) and followed by up to ``spatial_views - 1`` variants that
apply one forward scale+translate to every input and label row. A variant
reuses its source puzzle id -- same task, same learned embedding -- and records
``[scale, pad_r, pad_c]`` in ``test/all__spatial_tags.npy``, so the identifier
vocabulary is unchanged. Only variants fitting the square grid are emitted.
The train split and root files are copied through.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import dataclasses
import json
import logging
import shutil
import tempfile

import numpy as np

from priml.baselines.arcagi1.augmentation import ArcSpec, normalize_scale_weights
from priml.baselines.arcagi1.scripts.build_dataset import (
    DEFAULT_SCALE_WEIGHTS,
    RECIPE_FILE,
    check_recipe,
    stamp_recipe,
)
from priml.data.distributed_build import run_rank_zero_build
from priml.data.ensure import DataSpec, EnsureResult, FileSpec, ensure_data
from priml.lib.codec import from_plain, loads
from priml.paths import resolve_working_dir, validated_output_path


if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from numpy.typing import NDArray


logger = logging.getLogger(__name__)

SPLIT_FILES: Final = (
    "all__inputs.npy",
    "all__labels.npy",
    "all__puzzle_indices.npy",
    "all__group_indices.npy",
    "all__puzzle_identifiers.npy",
)
"""Per-split arrays a spatial-eval tree carries."""


def spatial_eval_slug(*, spatial_views: int) -> str:
    """Return the path slug for a spatial-eval dataset, e.g. ``spatialeval-v2``."""
    if spatial_views < 1:
        raise ValueError(f"spatial_views must be >= 1, got {spatial_views}.")
    return f"spatialeval-v{spatial_views}"


def spatial_eval_dataset_dir(
    *,
    spatial_views: int,
    source_name: str | None = None,
    base_dir: Path | str | None = None,
    working_dir: Path | str = "/datasets",
) -> Path:
    """Resolve the spatial-eval directory for a source tree.

    Args:
      spatial_views: Views per source puzzle, identity included.
      source_name: Source dataset directory name; distinct sources never clobber.
      base_dir: Resource root.
      working_dir: Logical root beneath ``base_dir``.

    Returns:
      path: ``<root>/<source_name>-spatialeval-v<N>``.

    """
    slug = spatial_eval_slug(spatial_views=spatial_views)
    base = source_name or "arc1concept-aug-1000"
    return resolve_working_dir(base_dir, working_dir) / f"{base}-{slug}"


_ensure_cache: set[tuple[Path, str]] = set()
"""``(target, serialized recipe)`` ensures already completed in this process."""


def ensure_spatial_eval_data(
    *,
    source_dir: Path,
    spatial_views: int,
    scale_weights: Mapping[int, float] = DEFAULT_SCALE_WEIGHTS,
    seed: int = 42,
    target: Path | None = None,
    spec: ArcSpec | None = None,
) -> Path:
    """Ensure the spatial-eval expansion of an existing ``source_dir``.

    Rank-safe and process-cached. The manifest lists every file the build
    writes, train split included, so a preempted build is never adopted. The
    tree's recipe sentinel must match every input (see ``check_recipe``), so a
    path never serves an expansion it was not built by.

    Args:
      source_dir: A built ARC tree.
      spatial_views: Views per source puzzle, identity included.
      scale_weights: Scale factors sampled per variant (fit-gated).
      seed: Sampler seed.
      target: Destination; ``None`` resolves under ``/opt/scratch``.
      spec: Dataset-owned grid geometry; ``None`` uses the default ARC spec.

    Returns:
      target: The spatial-eval dataset directory.

    Raises:
      ValueError: The tree at ``target`` was built by a different recipe.

    """
    if target is None:
        target = spatial_eval_dataset_dir(
            spatial_views=spatial_views,
            source_name=source_dir.name,
            base_dir="/opt/scratch",
        )
    if spec is None:
        spec = ArcSpec()
    recipe = _spatial_recipe(
        source_dir=source_dir,
        spatial_views=spatial_views,
        scale_weights=scale_weights,
        seed=seed,
        spec=spec,
    )
    key = (target, json.dumps(recipe, sort_keys=True))
    if key in _ensure_cache:
        return target
    if any(cached == target for cached, _ in _ensure_cache):
        raise ValueError(
            f"{target} was built by a different recipe than requested earlier "
            "in this process.",
        )
    manifest = [
        FileSpec(rel_path="identifiers.json"),
        FileSpec(rel_path="test_puzzles.json"),
        *(FileSpec(rel_path=f"test/{name}") for name in SPLIT_FILES),
        FileSpec(rel_path="test/all__spatial_tags.npy"),
        FileSpec(rel_path="test/dataset.json"),
    ]
    if (source_dir / "train").is_dir():
        # train/dataset.json is written last, so it closes the manifest.
        manifest += [
            *(FileSpec(rel_path=f"train/{name}") for name in SPLIT_FILES),
            FileSpec(rel_path="train/dataset.json"),
        ]
    fetch = _SpatialEvalBuild(
        source_dir=source_dir,
        target_dir=target,
        spatial_views=spatial_views,
        scale_weights=scale_weights,
        seed=seed,
        spec=spec,
    )
    data_spec = DataSpec(target_dir=target, manifest=manifest, fetch=fetch)

    def _build() -> None:
        check_recipe(target, recipe)
        if ensure_data(data_spec) is EnsureResult.DOWNLOADED:
            stamp_recipe(target, recipe)

    run_rank_zero_build(name="ensure_spatial_eval_data", build=_build)
    _ensure_cache.add(key)
    return target


def build_spatial_eval(
    *,
    spatial_views: int,
    source_dir: Path,
    target_dir: Path,
    scale_weights: Mapping[int, float] = DEFAULT_SCALE_WEIGHTS,
    seed: int = 42,
    spec: ArcSpec | None = None,
) -> None:
    """Write a spatial-expanded copy of ``source_dir`` to ``target_dir``.

    Args:
      spatial_views: Views per source puzzle, identity included.
      source_dir: A built ARC tree.
      target_dir: Destination.
      scale_weights: Scale factor weights (normalized here).
      seed: Sampler seed.
      spec: Dataset-owned grid geometry; ``None`` uses the default ARC spec.

    """
    if spatial_views < 1:
        raise ValueError(f"spatial_views must be >= 1, got {spatial_views}.")
    target_dir = validated_output_path(target_dir, protected=(source_dir,))
    if spec is None:
        spec = ArcSpec()
    scale_weights = normalize_scale_weights(scale_weights)
    rng = np.random.default_rng(seed)
    src_test = source_dir / "test"
    inputs = _rows(_load(src_test / "all__inputs.npy"))
    labels = _rows(_load(src_test / "all__labels.npy"))
    puzzle_indices = _int_list(_load(src_test / "all__puzzle_indices.npy"))
    group_indices = _int_list(_load(src_test / "all__group_indices.npy"))
    puzzle_ids = _int_list(_load(src_test / "all__puzzle_identifiers.npy"))
    identifiers = from_plain(
        loads((source_dir / "identifiers.json").read_text()),
        list[str],
    )

    new_inputs: list[NDArray[np.int64]] = []
    new_labels: list[NDArray[np.int64]] = []
    new_puzzle_indices = [0]
    new_group_indices = [0]
    new_puzzle_ids: list[int] = []
    new_spatial_tags: list[tuple[int, int, int]] = []
    example_count = 0
    group_ptr = 1
    for p, pid in enumerate(puzzle_ids):
        lo, hi = puzzle_indices[p], puzzle_indices[p + 1]
        for r in range(lo, hi):
            new_inputs.append(inputs[r])
            new_labels.append(labels[r])
            example_count += 1
        new_puzzle_indices.append(example_count)
        new_puzzle_ids.append(pid)
        new_spatial_tags.append((1, 0, 0))
        # One transform covers the whole puzzle, so every row must fit it.
        puzzle_grids = [*inputs[lo:hi], *labels[lo:hi]]
        for _ in range(spatial_views - 1):
            sampled = _sample_spatial(puzzle_grids, scale_weights, rng, spec=spec)
            if sampled is None:
                continue
            scale, pad_r, pad_c = sampled
            for r in range(lo, hi):
                new_inputs.append(
                    _forward_spatial(
                        inputs[r],
                        scale=scale,
                        pad_r=pad_r,
                        pad_c=pad_c,
                        spec=spec,
                    ),
                )
                new_labels.append(
                    _forward_spatial(
                        labels[r],
                        scale=scale,
                        pad_r=pad_r,
                        pad_c=pad_c,
                        spec=spec,
                    ),
                )
                example_count += 1
            new_puzzle_indices.append(example_count)
            new_puzzle_ids.append(pid)
            new_spatial_tags.append((scale, pad_r, pad_c))
        if p + 1 == group_indices[group_ptr]:
            new_group_indices.append(len(new_puzzle_ids))
            group_ptr += 1

    target_dir.parent.mkdir(parents=True, exist_ok=True)
    # Written beside the target and moved in only once complete, so a failure
    # mid-build never leaves a mixed tree at a path its readers trust.
    with tempfile.TemporaryDirectory(
        dir=target_dir.parent,
        prefix=f".{target_dir.name}-",
    ) as temporary:
        staging = Path(temporary)
        test_out = staging / "test"
        test_out.mkdir()
        np.save(test_out / "all__inputs.npy", np.stack(new_inputs))
        np.save(test_out / "all__labels.npy", np.stack(new_labels))
        for name, values in (
            ("all__puzzle_indices.npy", new_puzzle_indices),
            ("all__group_indices.npy", new_group_indices),
            ("all__puzzle_identifiers.npy", new_puzzle_ids),
        ):
            np.save(test_out / name, np.array(values, dtype=np.int32))
        np.save(
            test_out / "all__spatial_tags.npy",
            np.array(new_spatial_tags, dtype=np.int32),
        )
        _copy_dataset_json(
            src_test,
            test_out,
            num_ids=len(identifiers),
            num_puzzles=len(new_puzzle_ids),
            num_examples=example_count,
        )
        (staging / "identifiers.json").write_text(json.dumps(identifiers))
        shutil.copy(source_dir / "test_puzzles.json", staging / "test_puzzles.json")
        src_train = source_dir / "train"
        if src_train.is_dir():
            dst_train = staging / "train"
            dst_train.mkdir()
            for name in SPLIT_FILES:
                shutil.copy(src_train / name, dst_train / name)
            _copy_dataset_json(src_train, dst_train, num_ids=len(identifiers))
        _publish(staging, target_dir)
    logger.info(
        "spatial eval: %d puzzles (from %d), %d examples -> %s",
        len(new_puzzle_ids),
        len(puzzle_ids),
        example_count,
        target_dir,
    )


def _spatial_recipe(
    *,
    source_dir: Path,
    spatial_views: int,
    scale_weights: Mapping[int, float],
    seed: int,
    spec: ArcSpec,
) -> dict[str, object]:
    """Return every input that changes an expansion's bytes, as JSON."""
    source_recipe = source_dir / RECIPE_FILE
    return from_plain(
        loads(
            json.dumps(
                {
                    "builder": "spatial_eval",
                    "source": str(source_dir),
                    "source_recipe": (
                        from_plain(
                            loads(source_recipe.read_text()),
                            dict[str, object],
                        )
                        if source_recipe.is_file()
                        else None
                    ),
                    "spatial_views": spatial_views,
                    "scale_weights": [
                        [scale, weight]
                        for scale, weight in normalize_scale_weights(
                            scale_weights,
                        ).items()
                    ],
                    "seed": seed,
                    "spec": dataclasses.asdict(spec),
                },
            ),
        ),
        dict[str, object],
    )


class _SpatialEvalBuild:
    """Fetch closure building the whole spatial-eval tree once."""

    def __init__(
        self,
        *,
        source_dir: Path,
        target_dir: Path,
        spatial_views: int,
        scale_weights: Mapping[int, float],
        seed: int,
        spec: ArcSpec,
    ) -> None:
        self._spec = spec
        self._source_dir = source_dir
        self._target_dir = target_dir
        self._spatial_views = spatial_views
        self._scale_weights = scale_weights
        self._seed = seed
        self._built = False

    def __call__(self, *, rel_path: str, dest: Path) -> None:
        del rel_path, dest
        if self._built:
            return
        build_spatial_eval(
            spatial_views=self._spatial_views,
            source_dir=self._source_dir,
            target_dir=self._target_dir,
            scale_weights=self._scale_weights,
            seed=self._seed,
            spec=self._spec,
        )
        self._built = True


def _load(path: Path) -> NDArray[np.int64]:
    return cast("NDArray[np.int64]", np.load(path))


def _rows(array: NDArray[np.int64]) -> list[NDArray[np.int64]]:
    return list(cast("Iterable[NDArray[np.int64]]", array))


def _int_list(array: NDArray[np.int64]) -> list[int]:
    return cast(list[int], array.tolist())


def _content_shape(flat: NDArray[np.int64], *, spec: ArcSpec) -> tuple[int, int]:
    """Return the color content's bounding ``(rows, cols)`` from the top-left."""
    color = flat.reshape(spec.max_grid, spec.max_grid) >= spec.vocab_color_offset
    if not color.any():
        return 0, 0
    rows = _int_list(np.nonzero(color.any(axis=1))[0])
    cols = _int_list(np.nonzero(color.any(axis=0))[0])
    return rows[-1] + 1, cols[-1] + 1


def _forward_spatial(
    flat: NDArray[np.int64],
    *,
    scale: int,
    pad_r: int,
    pad_c: int,
    spec: ArcSpec,
) -> NDArray[np.int64]:
    """Crop content, block-upscale, pad to ``(pad_r, pad_c)``, and mark EOS."""
    nr, nc = _content_shape(flat, spec=spec)
    content = flat.reshape(spec.max_grid, spec.max_grid)[:nr, :nc]
    scaled = np.repeat(np.repeat(content, scale, axis=0), scale, axis=1)
    sh, sw = nr * scale, nc * scale
    out = np.full((spec.max_grid, spec.max_grid), spec.vocab_pad, dtype=flat.dtype)
    out[pad_r : pad_r + sh, pad_c : pad_c + sw] = scaled
    if pad_r + sh < spec.max_grid:
        out[pad_r + sh, pad_c : pad_c + sw] = spec.vocab_eos
    if pad_c + sw < spec.max_grid:
        out[pad_r : pad_r + sh, pad_c + sw] = spec.vocab_eos
    return out.flatten()


def _sample_spatial(
    grids: Sequence[NDArray[np.int64]],
    scale_weights: Mapping[int, float],
    rng: np.random.Generator,
    *,
    spec: ArcSpec,
) -> tuple[int, int, int] | None:
    """Sample ``(scale, pad_r, pad_c)`` fitting every grid; None for identity."""
    shapes = [_content_shape(grid, spec=spec) for grid in grids]
    if not shapes:
        return None
    max_r = max(r for r, _ in shapes)
    max_c = max(c for _, c in shapes)
    if max_r == 0:
        return None
    factors = [
        s
        for s in scale_weights
        if s * max_r <= spec.max_grid and s * max_c <= spec.max_grid
    ]
    weights: NDArray[np.float64] = np.array([scale_weights[s] for s in factors])
    scale = int(rng.choice(factors, p=weights / weights.sum())) if factors else 1
    pad_r = int(rng.integers(spec.max_grid - scale * max_r + 1))
    pad_c = int(rng.integers(spec.max_grid - scale * max_c + 1))
    if scale == 1 and pad_r == 0 and pad_c == 0:
        return None
    return scale, pad_r, pad_c


# ``num_puzzles`` is ``None`` for a split copied verbatim, whose puzzle counts stand.
def _copy_dataset_json(
    source: Path,
    destination: Path,
    *,
    num_ids: int,
    num_puzzles: int | None = None,
    num_examples: int = 0,
) -> None:
    """Copy a split's ``dataset.json``, restating the counts the expansion changed."""
    meta = from_plain(
        loads((source / "dataset.json").read_text()),
        dict[str, object],
    )
    meta["num_puzzle_identifiers"] = num_ids
    if num_puzzles is not None:
        meta["total_puzzles"] = num_puzzles
        meta["mean_puzzle_examples"] = num_examples / num_puzzles
    (destination / "dataset.json").write_text(json.dumps(meta))


def _publish(staging: Path, target_dir: Path) -> None:
    """Move every staged file into ``target_dir``, replacing what it held."""
    for path in sorted(staging.rglob("*")):
        if path.is_file():
            destination = target_dir / path.relative_to(staging)
            destination.parent.mkdir(parents=True, exist_ok=True)
            path.replace(destination)
