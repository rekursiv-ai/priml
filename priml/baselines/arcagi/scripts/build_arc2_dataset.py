"""Build the ARC-AGI-2 TRM dataset (``arc2concept-aug-1000``).

Same builder, augmentation math, and on-disk schema as the ARC-AGI-1
``arc1concept-aug-1000`` tree; only the Kaggle subsets differ, mirroring the
upstream TinyRecursiveModels ARC-AGI-2 recipe (README lines 47-56):
``training2`` (1,000 tasks) + ``evaluation2`` (120 tasks, the test split) +
``concept`` (160 ConceptARC tasks). One epoch therefore spans 1,280 groups.

Source data is the Kaggle JSON in the pinned TinyRecursiveModels revision,
NOT the arcprize/ARC-AGI-2 GitHub repo: the two differ on 6 evaluation tasks
(172 test outputs across 120 tasks here, 167 there), and the published TRM
numbers were scored on this packaging.

LEAKAGE: ARC-AGI-2 ``training2`` contains 376/400 ARC-AGI-1 public eval tasks,
and 6/120 ``evaluation2`` tasks are re-partitions of ARC-AGI-1 public eval
tasks. A model trained on this tree must never be scored on ARC-AGI-1.

:class:`~priml.baselines.arcagi.puzzle_data.Arc2PuzzleDataset` stages the
tree at ``__init__`` when its ``num_puzzle_identifiers`` is positive.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Final

import fcntl
import json
import logging

from treekle import from_plain
from treekle.wire import json_loads

from priml.baselines.arcagi.augmentation import (
    NO_TRAIN_SCALE_WEIGHTS,
    ArcAugmentation,
    ArcSpec,
    normalize_scale_weights,
)
from priml.baselines.arcagi.scripts.build_dataset import (
    RECIPE_FILE,
    KaggleSource,
    arc_manifest,
    write_arc_tree,
)
from priml.data.distributed_build import run_rank_zero_build


if TYPE_CHECKING:
    from collections.abc import Generator, Mapping


logger = logging.getLogger(__name__)

_ensure_cache: dict[Path, Path] = {}
_ARC2_SUBSETS: Final = ("training2", "evaluation2", "concept")
_ARC2_TEST_SET: Final = "evaluation2"


def ensure_arc2_dataset(
    *,
    target: Path,
    train_scale_weights: Mapping[int, float] = NO_TRAIN_SCALE_WEIGHTS,
    translation_prob: float = 1.0,
    scale_prob: float = 1.0,
    num_aug: int = 1_000,
    seed: int = 42,
    input_file_prefix: str | None = None,
) -> Path:
    """Ensure the ARC-AGI-2 dataset is staged at ``target`` (build if missing).

    Rank-safe and process-cached per target. ``target`` is the caller-resolved
    dataset root (the dataset config's finalized ``dataset_dir``) so the build
    target and the path the loader reads from cannot diverge. A completed
    build is recognized by the ``_build_params.json`` sentinel matching every
    build input AND every manifest file existing on disk; anything else
    triggers a rebuild under a cross-rank filesystem lock.

    Args:
      target: Destination dataset root (parent of train/ and test/).
      train_scale_weights: Integer scale distribution for the train split.
      translation_prob: Per-example train translation Bernoulli probability.
      scale_prob: Per-example train scale Bernoulli probability.
      num_aug: Augmentations per puzzle (changes the tree).
      seed: Build RNG seed (changes the tree).
      input_file_prefix: Kaggle JSON prefix override (tests); ``None`` uses the
        pinned TinyRecursiveModels source.

    Returns:
      root: The staged dataset root (== ``target``).

    """
    root = Path(target).expanduser()
    want = _build_params(
        train_scale_weights=train_scale_weights,
        translation_prob=translation_prob,
        scale_prob=scale_prob,
        num_aug=num_aug,
        seed=seed,
        input_file_prefix=input_file_prefix,
    )
    if root in _ensure_cache and _params_match(root, want):
        return _ensure_cache[root]

    def build() -> None:
        with _build_lock(root):
            if _params_match(root, want):
                return
            # Invalidate any previous build's sentinel BEFORE mutating the
            # tree in place: a crash mid-rebuild must leave a tree that
            # matches NO params (forcing a rebuild), never the old ones.
            (root / RECIPE_FILE).unlink(missing_ok=True)
            logger.info("ensure_arc2_dataset: building %s", root)
            if input_file_prefix is not None:
                _build_arc2_tree(
                    root,
                    prefix=input_file_prefix,
                    train_scale_weights=train_scale_weights,
                    translation_prob=translation_prob,
                    scale_prob=scale_prob,
                    num_aug=num_aug,
                    seed=seed,
                    want=want,
                )
            else:
                with KaggleSource() as prefix:
                    _build_arc2_tree(
                        root,
                        prefix=str(prefix),
                        train_scale_weights=train_scale_weights,
                        translation_prob=translation_prob,
                        scale_prob=scale_prob,
                        num_aug=num_aug,
                        seed=seed,
                        want=want,
                    )

    run_rank_zero_build(name="ensure_arc2_dataset", build=build)
    _ensure_cache[root] = root
    return root


def _build_arc2_tree(
    root: Path,
    *,
    prefix: str,
    train_scale_weights: Mapping[int, float],
    translation_prob: float,
    scale_prob: float,
    num_aug: int,
    seed: int,
    want: dict[str, object],
) -> None:
    """Run the canonical ARC builder with ARC-AGI-2 subsets, then stamp it."""
    augmentation = ArcAugmentation.Config()
    augmentation.spec = ArcSpec()
    augmentation.num_aug = num_aug
    augmentation.seed = seed
    augmentation.spatial.train_scale_weights = dict(train_scale_weights)
    augmentation.spatial.translation_prob = translation_prob
    augmentation.spatial.scale_prob = scale_prob
    root.mkdir(exist_ok=True)
    write_arc_tree(
        input_file_prefix=prefix,
        output_dir=root,
        augmentation=augmentation.make(),
        subsets=_ARC2_SUBSETS,
        test_set_name=_ARC2_TEST_SET,
    )
    with (root / RECIPE_FILE).open("w") as f:
        json.dump(want, f)
    logger.info("Done. ARC-AGI-2 dataset at %s", root)


def _build_params(
    *,
    train_scale_weights: Mapping[int, float],
    translation_prob: float,
    scale_prob: float,
    num_aug: int,
    seed: int,
    input_file_prefix: str | None,
) -> dict[str, object]:
    """Return the build sentinel payload (every input that shapes the tree)."""
    return {
        "subsets": list(_ARC2_SUBSETS),
        "test_set_name": _ARC2_TEST_SET,
        "train_scale_weights": [
            [scale, weight]
            for scale, weight in normalize_scale_weights(
                train_scale_weights,
            ).items()
        ],
        "translation_prob": translation_prob,
        "scale_prob": scale_prob,
        "num_aug": num_aug,
        "seed": seed,
        # A tree from a non-pinned source (a test fixture, an alternate
        # ARC-AGI-2 packaging) must never be silently accepted as the
        # canonical build: the packagings differ in content (172 vs 167
        # evaluation outputs).
        "source": input_file_prefix or "vendored-trm-kaggle",
    }


# Requires BOTH the matching params sentinel AND every ARC manifest file on disk: a
# params file alone is not proof of a complete tree (a crash can write params before --
# or independent of -- the arrays), so the data must be present too, or ensure would
# skip a build the loader then fails on.
def _params_match(root: Path, want: dict[str, object]) -> bool:
    """Whether ``root`` holds a completed compatible build."""
    path = root / RECIPE_FILE
    if not path.is_file():
        return False
    try:
        got = from_plain(json_loads(path.read_text()), dict[str, object])
    except (json.JSONDecodeError, TypeError):
        return False
    if got != want:
        return False
    return all((root / spec.rel_path).is_file() for spec in arc_manifest())


@contextmanager
def _build_lock(root: Path) -> Generator[None]:
    """Hold the cross-rank filesystem lock for one ARC-AGI-2 dataset root."""
    lock_path = root.with_name(root.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
