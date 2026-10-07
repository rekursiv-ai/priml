"""Materialize the dataset-owned ARC augmentation recipe from local source JSON.

Tree schema (``TinyRecursiveModels/dataset/build_arc_dataset.py``)::

  <split>/all__inputs.npy             [n_examples, seq_len]
  <split>/all__labels.npy             [n_examples, seq_len]
  <split>/all__puzzle_identifiers.npy [n_puzzles]
  <split>/all__puzzle_indices.npy     [n_puzzles + 1] example offsets
  <split>/all__group_indices.npy      [n_groups + 1] puzzle offsets
  test/all__spatial_tags.npy          [n_puzzles, 3] scale and offsets; spatial eval only
  <split>/dataset.json                split metadata
  identifiers.json                    list[str] (index = id, 0 = "<blank>")
  test_puzzles.json                   {name: original puzzle dict}
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, cast

import dataclasses
import hashlib
import json
import logging
import math
import shlex
import subprocess
import tempfile

from numpy.typing import NDArray

import numpy as np

from priml.baselines.arcagi1.augmentation import (
    ArcAugmentation,
    ArcSpec,
    arc_grid_to_np,
    grid_hash,
    normalize_scale_weights,
    scale_weights_slug,
)
from priml.data.distributed_build import run_rank_zero_build
from priml.data.ensure import DataSpec, EnsureResult, FileSpec, ensure_data
from priml.lib.codec import from_plain, loads
from priml.paths import resolve_working_dir


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


logger = logging.getLogger(__name__)

SOURCE_URL: Final = "https://github.com/SamsungSAILMontreal/TinyRecursiveModels.git"
"""Pinned upstream source repository."""

SOURCE_REVISION: Final = "c01103738605ba39d1430519b1ee0c62f4c707f8"
"""Immutable upstream commit containing the ARC source files."""

DEFAULT_SCALE_WEIGHTS: Final[Mapping[int, float]] = MappingProxyType({2: 1.0})
"""Scale distribution sampled when an aug policy's ``scale_prob`` gate fires.

Mass only on factor 1 would make the gate a silent no-op, which
:func:`aug_policy_slug` rejects."""


@dataclass(slots=True, kw_only=True)
class _Puzzle:
    name: str
    examples: list[tuple[NDArray[np.uint8], NDArray[np.uint8]]]


def build_arc_dataset(
    *,
    target_dir: Path,
    input_file_prefix: str | None,
    augmentation: ArcAugmentation,
) -> None:
    """Build a fresh, recipe-stamped ARC tree without overwriting an existing one.

    Args:
      target_dir: Destination for the complete dataset tree.
      input_file_prefix: Local prefix of the challenge/solution JSON files;
        ``None`` clones the pinned source.
      augmentation: Dataset-owned augmentation recipe and transforms.

    Raises:
      FileExistsError: The destination already contains data.

    """
    if target_dir.exists() and any(target_dir.iterdir()):
        raise FileExistsError(f"Dataset destination is not empty: {target_dir}")
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=target_dir.parent,
        prefix=f".{target_dir.name}-",
    ) as temporary:
        staging = Path(temporary) / "dataset"
        _write_from_source(
            output_dir=staging,
            augmentation=augmentation,
            input_file_prefix=input_file_prefix,
        )
        stamp_recipe(
            staging,
            arc_recipe(augmentation, input_file_prefix=input_file_prefix),
        )
        staging.replace(target_dir)


def arc_manifest() -> list[FileSpec]:
    """Return the existence-only manifest every ARC tree shares.

    Returns:
      manifest: Root JSON files, then each split's arrays and metadata.

    """
    split_files = (
        "all__inputs.npy",
        "all__labels.npy",
        "all__puzzle_indices.npy",
        "all__group_indices.npy",
        "all__puzzle_identifiers.npy",
        "dataset.json",
    )
    manifest = [
        FileSpec(rel_path=name) for name in ("identifiers.json", "test_puzzles.json")
    ]
    manifest += [
        FileSpec(rel_path=f"{split}/{name}")
        for split in ("train", "test")
        for name in split_files
    ]
    return manifest


RECIPE_FILE: Final = "_build_params.json"
"""Root-level sentinel recording every input that shaped a tree's bytes."""

ARC1_SUBSETS: Final = ("training", "evaluation", "concept")
"""Kaggle subsets the ARC-AGI-1 ``arc1concept`` tree is built from."""

ARC1_TEST_SET: Final = "evaluation"
"""The subset whose test pairs form the ARC-AGI-1 test split."""


def arc_recipe(
    augmentation: ArcAugmentation,
    *,
    input_file_prefix: str | None,
    subsets: Sequence[str] = ARC1_SUBSETS,
    test_set_name: str = ARC1_TEST_SET,
) -> dict[str, object]:
    """Return the JSON recipe of a tree: every input that changes its bytes.

    Args:
      augmentation: Recipe the tree is built with.
      input_file_prefix: Local source prefix; ``None`` names the pinned source.
      subsets: Kaggle subsets read, in order.
      test_set_name: Subset whose test pairs form the test split.

    Returns:
      recipe: JSON-ready mapping, compared verbatim against a tree's sentinel.

    """
    return from_plain(
        loads(
            json.dumps(
                {
                    "builder": "arc",
                    "source": input_file_prefix or f"{SOURCE_URL}@{SOURCE_REVISION}",
                    "subsets": list(subsets),
                    "test_set_name": test_set_name,
                    "augmentation": dataclasses.asdict(augmentation.config),
                },
            ),
        ),
        dict[str, object],
    )


def check_recipe(target_dir: Path, recipe: Mapping[str, object]) -> None:
    """Refuse a tree stamped by another recipe; adopt an unstamped one, warning.

    Args:
      target_dir: Dataset root that may hold :data:`RECIPE_FILE`.
      recipe: Recipe the caller is about to read the tree as.

    Raises:
      ValueError: The tree's sentinel records a different recipe.

    """
    path = target_dir / RECIPE_FILE
    if not path.is_file():
        if any((target_dir / spec.rel_path).is_file() for spec in arc_manifest()):
            logger.warning(
                "%s has no recipe sentinel; adopting it as built by the requested "
                "recipe. Stamp it with: %s",
                target_dir,
                _stamp_command(path, recipe),
            )
        return
    stamped = from_plain(loads(path.read_text()), dict[str, object])
    if stamped != recipe:
        raise ValueError(
            f"{target_dir} was built by a different recipe than requested; "
            f"stamped {stamped}, requested {dict(recipe)}. Point working_dir at a "
            "fresh directory, or delete the tree to rebuild it.",
        )


def stamp_recipe(target_dir: Path, recipe: Mapping[str, object]) -> None:
    """Write ``recipe`` as ``target_dir``'s :data:`RECIPE_FILE` sentinel."""
    (target_dir / RECIPE_FILE).write_text(json.dumps(recipe, sort_keys=True))


_ensure_cache: dict[Path, tuple[dict[str, object], EnsureResult]] = {}
"""Per-target recipe and outcome, so repeated finalize/make passes build once."""


def ensure_arc_dataset(
    *,
    target_dir: Path,
    augmentation: ArcAugmentation,
    input_file_prefix: str | None = None,
) -> EnsureResult:
    """Ensure the ARC tree is present at ``target_dir``, building it if not.

    Rank-safe: rank 0 builds and :func:`run_rank_zero_build`'s all-reduce is
    the cross-rank sync. Cached per target, so every rank sees one build. The
    tree's :data:`RECIPE_FILE` must match ``augmentation`` (see
    :func:`check_recipe`), so a path never serves a recipe it was not built by.

    Args:
      target_dir: Destination dataset root.
      augmentation: Recipe the tree is built with.
      input_file_prefix: Local source prefix; ``None`` clones the pinned source.

    Returns:
      result: Outcome of the underlying :func:`ensure_data` call.

    Raises:
      ValueError: The tree at ``target_dir`` was built by a different recipe.

    """
    recipe = arc_recipe(augmentation, input_file_prefix=input_file_prefix)
    if target_dir in _ensure_cache:
        cached, result = _ensure_cache[target_dir]
        if cached != recipe:
            raise ValueError(
                f"{target_dir} was built by a different recipe than requested "
                "earlier in this process.",
            )
        return result
    spec = DataSpec(
        target_dir=target_dir,
        manifest=arc_manifest(),
        fetch=_ArcBuild(
            target_dir=target_dir,
            augmentation=augmentation,
            input_file_prefix=input_file_prefix,
        ),
    )
    result = EnsureResult.PRESENT

    def _build() -> None:
        nonlocal result
        check_recipe(target_dir, recipe)
        result = ensure_data(spec)
        if result is EnsureResult.DOWNLOADED:
            stamp_recipe(target_dir, recipe)

    run_rank_zero_build(name="ensure_arc_dataset", build=_build)
    _ensure_cache[target_dir] = (recipe, result)
    return result


class KaggleSource:
    """Clone the pinned source and yield its ARC challenge/solution prefix."""

    def __init__(self) -> None:
        self._temporary: tempfile.TemporaryDirectory[str] | None = None

    def __enter__(self) -> Path:
        """Clone at :data:`SOURCE_REVISION`, verify it, and return the prefix."""
        self._temporary = tempfile.TemporaryDirectory(prefix="arcagi1-source-")
        clone = Path(self._temporary.name) / "TinyRecursiveModels"
        subprocess.run(  # noqa: S603 -- Fixed Git executable and repository URL.
            ["git", "clone", "--no-checkout", SOURCE_URL, str(clone)],  # noqa: S607 -- Fixed Git executable.
            check=True,
        )
        subprocess.run(  # noqa: S603 -- Fixed Git executable and revision.
            ["git", "-C", str(clone), "checkout", SOURCE_REVISION],  # noqa: S607 -- Fixed Git executable.
            check=True,
        )
        actual = subprocess.run(  # noqa: S603 -- Fixed Git executable and revision.
            ["git", "-C", str(clone), "rev-parse", "HEAD"],  # noqa: S607 -- Fixed Git executable.
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if actual != SOURCE_REVISION:
            raise RuntimeError(
                f"source revision mismatch: got {actual}, expected {SOURCE_REVISION}",
            )
        return clone / "kaggle" / "combined" / "arc-agi"

    def __exit__(self, *exc: object) -> None:
        """Remove the temporary clone."""
        del exc
        if self._temporary is not None:
            self._temporary.cleanup()


def aug_policy_slug(
    *,
    translation_prob: float,
    scale_prob: float,
    train_scale_weights: Mapping[int, float] = DEFAULT_SCALE_WEIGHTS,
    num_aug: int = 1_000,
    seed: int = 42,
) -> str:
    """Return a stable path slug for a translation/scale Bernoulli policy.

    Every input that changes the built bytes enters the slug, except the
    weights when ``scale_prob == 0`` (the gate never fires).

    Args:
      translation_prob: Per-example translation probability in ``[0, 1]``.
      scale_prob: Per-example scale probability in ``[0, 1]``.
      train_scale_weights: Scale distribution sampled when the gate fires.
      num_aug: Augmentations per puzzle.
      seed: Build seed.

    Returns:
      slug: Such as ``tr0p1-sc0p1-2w1p0-n1000-s42``.

    """
    _validate_prob("translation_prob", translation_prob)
    _validate_prob("scale_prob", scale_prob)
    # Rounding keeps IEEE-754 repr noise (0.1 + 0.2) from splitting equal policies.
    # Two recipes that round alike share a name; the tree's recipe sentinel then
    # refuses the second one instead of serving it the first one's arrays.
    tr = str(round(translation_prob, 6)).replace(".", "p")
    sc = str(round(scale_prob, 6)).replace(".", "p")
    if scale_prob == 0:
        return f"tr{tr}-sc{sc}-n{num_aug}-s{seed}"
    _validate_scalable("scale_prob", scale_prob, train_scale_weights)
    scale = scale_weights_slug(train_scale_weights)
    return f"tr{tr}-sc{sc}-{scale}-n{num_aug}-s{seed}"


def aug_policy_template(
    *,
    translation_prob: float,
    scale_prob: float,
    train_scale_weights: Mapping[int, float] = DEFAULT_SCALE_WEIGHTS,
    num_aug: int = 1_000,
    seed: int = 42,
) -> str:
    """Return the logical ``/datasets/...`` working_dir for an aug policy.

    Args:
      translation_prob: Per-example translation probability in ``[0, 1]``.
      scale_prob: Per-example scale probability in ``[0, 1]``.
      train_scale_weights: Scale distribution sampled when the gate fires.
      num_aug: Augmentations per puzzle; must match the build.
      seed: Build seed; must match the build.

    Returns:
      working_dir: ``/datasets/arc1concept-aug-1000-<slug>``. The prefix names
        the reference build family, as published trees spell it; ``num_aug``
        itself is the slug's ``n<num_aug>`` field.

    """
    slug = aug_policy_slug(
        translation_prob=translation_prob,
        scale_prob=scale_prob,
        train_scale_weights=train_scale_weights,
        num_aug=num_aug,
        seed=seed,
    )
    return f"/datasets/arc1concept-aug-1000-{slug}"


def aug_policy_dataset_dir(
    *,
    translation_prob: float,
    scale_prob: float,
    train_scale_weights: Mapping[int, float] = DEFAULT_SCALE_WEIGHTS,
    num_aug: int = 1_000,
    seed: int = 42,
    base_dir: Path | str | None = None,
    working_dir: Path | str = "/datasets",
) -> Path:
    """Resolve an aug-policy dataset directory beneath ``base_dir``.

    Experiments should carry :func:`aug_policy_template` instead, so the
    config's own ``finalize`` resolves the path.

    Args:
      translation_prob: Per-example translation probability in ``[0, 1]``.
      scale_prob: Per-example scale probability in ``[0, 1]``.
      train_scale_weights: Scale distribution sampled when the gate fires.
      num_aug: Augmentations per puzzle.
      seed: Build seed.
      base_dir: Resource root.
      working_dir: Logical root beneath ``base_dir``.

    Returns:
      dataset_dir: The aug-policy dataset root.

    """
    name = Path(
        aug_policy_template(
            translation_prob=translation_prob,
            scale_prob=scale_prob,
            train_scale_weights=train_scale_weights,
            num_aug=num_aug,
            seed=seed,
        ),
    ).name
    return resolve_working_dir(base_dir, working_dir) / name


def build(
    *,
    dataset_dir: str | None = None,
    input_file_prefix: str | None = None,
    translation_prob: float = 0.1,
    scale_prob: float = 0.1,
    train_scale_weights: Mapping[int, float] = DEFAULT_SCALE_WEIGHTS,
    num_aug: int = 1_000,
    seed: int = 42,
    spec: ArcSpec | None = None,
) -> None:
    """Ensure an aug-policy ARC tree, defaulting to its slugged scratch path.

    Args:
      dataset_dir: Destination; ``None`` resolves :func:`aug_policy_dataset_dir`.
      input_file_prefix: Local source prefix; ``None`` clones the pinned source.
      translation_prob: Per-example translation probability in ``[0, 1]``.
      scale_prob: Per-example scale probability in ``[0, 1]``.
      train_scale_weights: Scale distribution sampled when the gate fires.
      num_aug: Augmentations per puzzle.
      seed: Build seed.
      spec: Packed-grid geometry; None uses ARC's production geometry.

    """
    _validate_prob("translation_prob", translation_prob)
    _validate_prob("scale_prob", scale_prob)
    if scale_prob > 0:
        _validate_scalable("scale_prob", scale_prob, train_scale_weights)
    root = (
        Path(dataset_dir)
        if dataset_dir
        else aug_policy_dataset_dir(
            translation_prob=translation_prob,
            scale_prob=scale_prob,
            train_scale_weights=train_scale_weights,
            num_aug=num_aug,
            seed=seed,
            base_dir="/opt/scratch",
        )
    )
    augmentation = ArcAugmentation.Config()
    augmentation.spec = spec if spec is not None else ArcSpec()
    augmentation.num_aug = num_aug
    augmentation.seed = seed
    augmentation.spatial.train_scale_weights = dict(train_scale_weights)
    augmentation.spatial.translation_prob = translation_prob
    augmentation.spatial.scale_prob = scale_prob
    ensure_arc_dataset(
        target_dir=root,
        augmentation=augmentation.make(),
        input_file_prefix=input_file_prefix,
    )
    logger.info("Aug-policy ARC dataset at %s", root)


def write_arc_tree(
    *,
    input_file_prefix: str,
    output_dir: Path,
    augmentation: ArcAugmentation,
    subsets: Sequence[str] = ARC1_SUBSETS,
    test_set_name: str = ARC1_TEST_SET,
) -> None:
    """Write an ARC tree's arrays and root JSON from Kaggle-format sources.

    Writes no recipe sentinel: the caller owns where the tree is published and
    stamps it with :func:`stamp_recipe` once complete.

    Args:
      input_file_prefix: Prefix of ``<prefix>_<subset>_challenges.json`` files.
      output_dir: Directory receiving the tree; created as needed.
      augmentation: Dataset-owned augmentation recipe and transforms.
      subsets: Kaggle subsets to read, in order; they share one id space.
      test_set_name: Subset whose test pairs form the test split.

    """
    rng = np.random.default_rng(augmentation.config.seed)
    results: dict[str, dict[str, list[list[_Puzzle]]]] = {}
    test_puzzles: dict[str, object] = {}
    train_dest = ("train", "all")
    test_dest = ("test", "all")
    for subset_name in subsets:
        raw_puzzles = from_plain(
            loads(
                Path(f"{input_file_prefix}_{subset_name}_challenges.json").read_text(),
            ),
            dict[str, object],
        )
        puzzles = {
            pid: {
                key: [
                    from_plain(example, dict[str, object])
                    for example in from_plain(examples, list[object])
                ]
                for key, examples in from_plain(puzzle, dict[str, object]).items()
            }
            for pid, puzzle in raw_puzzles.items()
        }
        solutions = Path(f"{input_file_prefix}_{subset_name}_solutions.json")
        if solutions.is_file():
            raw_solutions = from_plain(
                loads(solutions.read_text()),
                dict[str, object],
            )
            for pid, puzzle in puzzles.items():
                for index, grid in enumerate(
                    from_plain(raw_solutions[pid], list[object]),
                ):
                    puzzle["test"][index]["output"] = _int_grid(grid)
        else:
            logger.warning("%s solutions not found, filling with dummy", subset_name)
            for puzzle in puzzles.values():
                for example in puzzle["test"]:
                    example.setdefault("output", [[0]])
        puzzle_list = list(puzzles.items())
        rng.shuffle(puzzle_list)
        for name, puzzle in puzzle_list:
            is_test = subset_name == test_set_name
            if is_test:
                test_puzzles[name] = puzzle
            _convert_puzzle(
                results,
                name,
                puzzle,
                dest_map={
                    "train": train_dest,
                    "test": test_dest if is_test else train_dest,
                },
                augmentation=augmentation,
                rng=rng,
            )

    id_map: dict[str, int] = {}
    for split in results.values():
        for subset in split.values():
            for group in subset:
                for puzzle in group:
                    if puzzle.name not in id_map:
                        id_map[puzzle.name] = len(id_map) + 1
    for split_name, split in results.items():
        _write_split(
            output_dir,
            split_name,
            split,
            id_map,
            augmentation=augmentation,
            rng=rng,
            spatial_rng=np.random.default_rng(augmentation.config.seed),
        )
    inverse_id_map = {value: key for key, value in id_map.items()}
    identifiers = [inverse_id_map.get(i, "<blank>") for i in range(len(id_map) + 1)]
    (output_dir / "identifiers.json").write_text(json.dumps(identifiers))
    (output_dir / "test_puzzles.json").write_text(json.dumps(test_puzzles))


def _write_split(
    out_path: Path,
    split_name: str,
    split: dict[str, list[list[_Puzzle]]],
    id_map: dict[str, int],
    *,
    augmentation: ArcAugmentation,
    rng: np.random.Generator,
    spatial_rng: np.random.Generator,
) -> None:
    split_path = out_path / split_name
    # An ensure rebuild may find a partial tree from an interrupted build.
    split_path.mkdir(parents=True, exist_ok=True)
    total_examples = total_puzzles = total_groups = 0
    subset_names: list[str] = []
    for subset_name, subset in split.items():
        subset_names.append(subset_name)
        all_inputs: list[NDArray[np.uint8]] = []
        all_labels: list[NDArray[np.uint8]] = []
        puzzle_tags: list[tuple[int, int, int]] = []
        all_puzzle_ids: list[int] = []
        puzzle_indices = [0]
        group_indices = [0]
        puzzle_count = example_count = 0
        for group in subset:
            for puzzle in group:
                no_aug_idx = int(rng.integers(len(puzzle.examples)))
                for index, (inp, out) in enumerate(puzzle.examples):
                    inp_seq, out_seq = augmentation.spatial.pack(
                        inp,
                        out,
                        training=split_name == "train" and index != no_aug_idx,
                        rng=rng,
                    )
                    all_inputs.append(inp_seq)
                    all_labels.append(out_seq)
                    example_count += 1
                    total_examples += 1
                puzzle_indices.append(example_count)
                all_puzzle_ids.append(id_map[puzzle.name])
                puzzle_count += 1
                total_puzzles += 1
                puzzle_tags.append((1, 0, 0))
                eval_tag = (
                    _spatial_eval_tag(
                        puzzle,
                        augmentation=augmentation,
                        rng=spatial_rng,
                    )
                    if split_name == "test" and augmentation.config.spatial_eval_views
                    else None
                )
                if eval_tag is not None:
                    # A second puzzle under the same id: voting inverts the tag, so
                    # its ballots land with the canonical view's.
                    for inp, out in puzzle.examples:
                        inp_seq, out_seq = augmentation.spatial.pack_at(
                            inp,
                            out=out,
                            tag=eval_tag,
                        )
                        all_inputs.append(inp_seq)
                        all_labels.append(out_seq)
                        example_count += 1
                        total_examples += 1
                    puzzle_indices.append(example_count)
                    all_puzzle_ids.append(id_map[puzzle.name])
                    puzzle_count += 1
                    total_puzzles += 1
                    puzzle_tags.append(eval_tag)
            group_indices.append(puzzle_count)
            total_groups += 1
        np.save(split_path / f"{subset_name}__inputs.npy", np.stack(all_inputs))
        np.save(split_path / f"{subset_name}__labels.npy", np.stack(all_labels))
        # Written only with spatial views, so a plain tree stays byte-identical to
        # the reference build; the layout is build_spatial_eval.py's, one row per
        # puzzle, which is what PuzzleData reads.
        if split_name == "test" and augmentation.config.spatial_eval_views:
            np.save(
                split_path / f"{subset_name}__spatial_tags.npy",
                np.asarray(puzzle_tags, dtype=np.int32),
            )
        np.save(
            split_path / f"{subset_name}__puzzle_indices.npy",
            np.array(puzzle_indices, dtype=np.int32),
        )
        np.save(
            split_path / f"{subset_name}__group_indices.npy",
            np.array(group_indices, dtype=np.int32),
        )
        np.save(
            split_path / f"{subset_name}__puzzle_identifiers.npy",
            np.array(all_puzzle_ids, dtype=np.int32),
        )
    (split_path / "dataset.json").write_text(
        json.dumps(
            {
                "pad_id": augmentation.spec.vocab_pad,
                "ignore_label_id": augmentation.spec.vocab_pad,
                "blank_identifier_id": 0,
                "vocab_size": augmentation.spec.vocab_size,
                "seq_len": augmentation.spec.grid_shape[0],
                "num_puzzle_identifiers": len(id_map) + 1,
                "total_groups": total_groups,
                "mean_puzzle_examples": total_examples / total_puzzles,
                "total_puzzles": total_puzzles,
                "sets": subset_names,
            },
        ),
    )


# Returns ``None`` when the draw is the identity, so no duplicate view is written.
def _spatial_eval_tag(
    puzzle: _Puzzle,
    *,
    augmentation: ArcAugmentation,
    rng: np.random.Generator,
) -> tuple[int, int, int] | None:
    """Choose one scale and translation that fits every grid of ``puzzle``."""
    shapes = [_shape(grid) for pair in puzzle.examples for grid in pair]
    max_rows = max(rows for rows, _ in shapes)
    max_cols = max(cols for _, cols in shapes)
    side = augmentation.spec.max_grid
    scale = augmentation.config.spatial_eval_scale
    if scale < 1:
        raise ValueError(f"spatial_eval_scale must be positive; got {scale}.")
    # The single-option choice still consumes a draw; the published trees include it.
    scale = (
        int(rng.choice([scale], p=[1.0]))
        if scale * max_rows <= side and scale * max_cols <= side
        else 1
    )
    pad_r = int(rng.integers(side - scale * max_rows + 1))
    pad_c = int(rng.integers(side - scale * max_cols + 1))
    tag = (scale, pad_r, pad_c)
    return None if tag == (1, 0, 0) else tag


def _shape(grid: NDArray[np.uint8]) -> tuple[int, int]:
    rows, cols = cast("tuple[int, int]", grid.shape)
    return rows, cols


def _convert_puzzle(
    results: dict[str, dict[str, list[list[_Puzzle]]]],
    name: str,
    puzzle: dict[str, list[dict[str, object]]],
    *,
    dest_map: dict[str, tuple[str, str]],
    augmentation: ArcAugmentation,
    rng: np.random.Generator,
) -> None:
    destinations = set(dest_map.values())
    converted = {
        destination: _Puzzle(name=name, examples=[]) for destination in destinations
    }
    for example_type, examples in puzzle.items():
        converted[dest_map[example_type]].examples.extend(
            [
                (
                    arc_grid_to_np(
                        _int_grid(example["input"]),
                        max_grid=augmentation.spec.max_grid,
                    ),
                    arc_grid_to_np(
                        _int_grid(example["output"]),
                        max_grid=augmentation.spec.max_grid,
                    ),
                )
                for example in examples
            ],
        )
    group = [converted]
    if augmentation.config.num_aug:
        hashes = {_puzzle_group_hash(converted)}
        for _ in range(
            augmentation.config.retries_factor * augmentation.config.num_aug,
        ):
            aug_name, apply = augmentation.transform.sample(name, rng=rng)
            augmented = {
                destination: _Puzzle(
                    name=aug_name,
                    examples=[(apply(inp), apply(out)) for inp, out in value.examples],
                )
                for destination, value in converted.items()
            }
            digest = _puzzle_group_hash(augmented)
            if digest not in hashes:
                hashes.add(digest)
                group.append(augmented)
            if len(group) >= augmentation.config.num_aug + 1:
                break
    for destination in destinations:
        split, subset = destination
        results.setdefault(split, {}).setdefault(subset, []).append(
            [view[destination] for view in group],
        )


def _puzzle_group_hash(group: Mapping[tuple[str, str], _Puzzle]) -> str:
    hashes = sorted(
        f"{grid_hash(inp)}|{grid_hash(out)}"
        for puzzle in group.values()
        for inp, out in puzzle.examples
    )
    return hashlib.sha256("|".join(hashes).encode()).hexdigest()


def _int_grid(value: object) -> list[list[int]]:
    return [from_plain(row, list[int]) for row in from_plain(value, list[object])]


class _ArcBuild:
    """Fetch closure building the whole tree on the first missing file."""

    def __init__(
        self,
        *,
        target_dir: Path,
        augmentation: ArcAugmentation,
        input_file_prefix: str | None,
    ) -> None:
        self._target_dir = target_dir
        self._augmentation = augmentation
        self._input_file_prefix = input_file_prefix
        self._built = False

    def __call__(self, *, rel_path: str, dest: Path) -> None:
        del dest
        if self._built:
            return
        logger.info(
            "ensure_arc_dataset: building %s (missing %s)",
            self._target_dir,
            rel_path,
        )
        _write_from_source(
            output_dir=self._target_dir,
            augmentation=self._augmentation,
            input_file_prefix=self._input_file_prefix,
        )
        self._built = True


def _write_from_source(
    *,
    output_dir: Path,
    augmentation: ArcAugmentation,
    input_file_prefix: str | None,
) -> None:
    """Write the ARC-AGI-1 tree from a local prefix, or from a pinned-source clone."""
    if input_file_prefix is not None:
        write_arc_tree(
            input_file_prefix=input_file_prefix,
            output_dir=output_dir,
            augmentation=augmentation,
        )
        return
    with KaggleSource() as prefix:
        write_arc_tree(
            input_file_prefix=str(prefix),
            output_dir=output_dir,
            augmentation=augmentation,
        )


def _stamp_command(path: Path, recipe: Mapping[str, object]) -> str:
    """Return a shell command writing exactly the sentinel :func:`stamp_recipe` would."""
    text = json.dumps(recipe, sort_keys=True)
    return f"printf %s {shlex.quote(text)} > {shlex.quote(str(path))}"


def _validate_prob(name: str, prob: float) -> None:
    if math.isnan(prob) or prob < 0.0 or prob > 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {prob}.")


def _validate_scalable(
    name: str,
    prob: float,
    train_scale_weights: Mapping[int, float],
) -> None:
    """Reject a firing scale gate whose distribution can never scale above 1."""
    if not any(factor > 1 for factor in normalize_scale_weights(train_scale_weights)):
        raise ValueError(
            f"{name}={prob} > 0 but train_scale_weights {dict(train_scale_weights)} "
            "has no mass on a factor > 1, so the scale gate is a silent no-op; "
            "add a factor > 1 or set scale_prob=0.",
        )
