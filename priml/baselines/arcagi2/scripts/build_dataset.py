#!/bin/sh
# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 "$0" "$@"
Build the ARC-AGI-2 TRM dataset (``arc2concept-aug-1000``).

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

Prebuild from the CLI (the dataset also stages itself at ``__init__``):

    uv --quiet run --frozen python "$0" [OPTIONS]
'''
# fmt: on

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, cast

import argparse
import fcntl
import json
import logging

from priml.baselines.arcagi1.augmentation import (
    NO_TRAIN_SCALE_WEIGHTS,
    ArcAugmentation,
    ArcSpec,
    normalize_scale_weights,
)
from priml.baselines.arcagi1.scripts.build_dataset import (
    DEFAULT_SCALE_WEIGHTS,
    KaggleSource,
    _build_arc_dataset,
    arc_manifest,
    aug_policy_slug,
)
from priml.baselines.arcagi1.scripts.build_spatial_eval import spatial_eval_slug
from priml.data.distributed_build import run_rank_zero_build
from priml.lib.custom_json import parse


if TYPE_CHECKING:
    from collections.abc import Generator, Mapping


logger = logging.getLogger(__name__)

_BUILD_PARAMS_FILE: Final = "_build_params.json"
_ensure_cache: dict[Path, Path] = {}
_ARC2_SUBSETS: Final = ("training2", "evaluation2", "concept")
_ARC2_TEST_SET: Final = "evaluation2"


def arc2_aug_policy_template(
    *,
    translation_prob: float,
    scale_prob: float,
    train_scale_weights: Mapping[int, float] = DEFAULT_SCALE_WEIGHTS,
    num_aug: int = 1_000,
    seed: int = 42,
) -> str:
    """Return the logical ``working_dir`` for a translation/scale policy variant.

    Mirrors ``priml.baselines.arcagi1.scripts.build_dataset.aug_policy_template`` with
    the ARC-AGI-2 base slug. The policy touches grid serialization only, not
    puzzle names, so the variant keeps the plain tree's puzzle-identifier
    vocabulary (``num_puzzle_identifiers`` is unchanged).

    Args:
      translation_prob: Per-example train translation Bernoulli probability.
      scale_prob: Per-example train scale Bernoulli probability.
      train_scale_weights: Integer scale distribution for the train split.
      num_aug: Augmentations per puzzle.
      seed: Build RNG seed.

    Returns:
      path: Logical dataset working directory path.

    """
    slug = aug_policy_slug(
        translation_prob=translation_prob,
        scale_prob=scale_prob,
        train_scale_weights=train_scale_weights,
        num_aug=num_aug,
        seed=seed,
    )
    return f"/datasets/arc2concept-aug-1000-{slug}"


def arc2_spatial_eval_template(
    *,
    spatial_views: int,
    translation_prob: float,
    scale_prob: float,
    train_scale_weights: Mapping[int, float] = DEFAULT_SCALE_WEIGHTS,
    num_aug: int = 1_000,
    seed: int = 42,
) -> str:
    """Return the logical ``working_dir`` for an aug-policy tree's spatial expansion.

    Composes :func:`arc2_aug_policy_template` with the
    ``build_spatial_eval`` naming (``<source>-spatialeval-vS``), so a config's
    ``working_dir`` resolves to the exact directory
    :func:`priml.baselines.arcagi1.scripts.build_spatial_eval.ensure_spatial_eval_data`
    stages for that source tree.

    Args:
      spatial_views: Number of spatial augmentation views.
      translation_prob: Per-example train translation Bernoulli probability.
      scale_prob: Per-example train scale Bernoulli probability.
      train_scale_weights: Integer scale distribution for the train split.
      num_aug: Augmentations per puzzle.
      seed: Build RNG seed.

    Returns:
      path: Logical dataset working directory path with spatial expansion.

    """
    source = arc2_aug_policy_template(
        translation_prob=translation_prob,
        scale_prob=scale_prob,
        train_scale_weights=train_scale_weights,
        num_aug=num_aug,
        seed=seed,
    )
    return f"{source}-{spatial_eval_slug(spatial_views=spatial_views)}"


def arc2_num_puzzle_identifiers(dataset_dir: Path) -> int:
    """Return ``len(identifiers.json)`` for a built ARC-AGI-2 tree."""
    path = Path(dataset_dir).expanduser() / "identifiers.json"
    return len(parse(path.read_text(), list[str]))


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
      input_file_prefix: Kaggle JSON prefix override (tests / CLI); ``None``
        uses the vendored TinyRecursiveModels source.

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
            (root / _BUILD_PARAMS_FILE).unlink(missing_ok=True)
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


def main() -> int:
    """Run the program; return the process exit code.

    Returns:
      code: Process exit code (0 on success).

    """
    if __doc__ is None:
        raise ValueError("Expected __doc__ is not None.")
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", 2)[2],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_arguments(parser)
    args = cast(_Flags, parser.parse_args())
    logging.basicConfig(level=logging.INFO)
    plain = args.translation_prob == 1.0 and args.scale_prob == 1.0
    if args.target_dir:
        target = Path(args.target_dir)
    elif plain and args.num_aug == 1_000 and args.seed == 42:
        # Data-helper-only path-owner (CLI prebuild), not for experiments:
        # configs carry the "/datasets/..." working_dir instead. The CLI is
        # the top-level path owner, so it injects base_dir="/opt/scratch".
        target = Path("/opt/scratch/datasets/arc2concept-aug-1000")
    elif not plain:
        # Fold the policy into the slug so a policy prebuild can NEVER
        # rebuild the plain tree in place (distinct policies get distinct
        # dirs). Mirrors the experiment-side arc2_aug_policy_template.
        slug = arc2_aug_policy_template(
            translation_prob=args.translation_prob,
            scale_prob=args.scale_prob,
            num_aug=args.num_aug,
            seed=args.seed,
        ).removeprefix("/datasets/")
        target = Path(f"/opt/scratch/datasets/{slug}")
    else:
        parser.error(
            "a plain-policy build with non-default --num-aug/--seed has no "
            "canonical directory; pass --target-dir explicitly.",
        )
    root = ensure_arc2_dataset(
        target=target,
        train_scale_weights=(
            NO_TRAIN_SCALE_WEIGHTS if plain else DEFAULT_SCALE_WEIGHTS
        ),
        translation_prob=args.translation_prob,
        scale_prob=args.scale_prob,
        num_aug=args.num_aug,
        seed=args.seed,
        input_file_prefix=args.input_file_prefix,
    )
    logger.info(
        "ARC-AGI-2 dataset at %s (num_puzzle_identifiers=%d)",
        root,
        arc2_num_puzzle_identifiers(root),
    )
    return 0


class _Flags(Protocol):
    """Parsed command-line flags."""

    target_dir: str
    translation_prob: float
    scale_prob: float
    num_aug: int
    seed: int
    input_file_prefix: str | None


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register flags on ``parser``."""
    parser.add_argument(
        "--target-dir",
        default="",
        help=(
            "Dataset root override. Empty resolves the canonical scratch dir: "
            "the plain slug at default flags, or the aug-policy slug (with "
            "the standard {2: 1.0} scale weights) when a policy flag is set."
        ),
    )
    parser.add_argument("--translation-prob", type=float, default=1.0)
    parser.add_argument("--scale-prob", type=float, default=1.0)
    parser.add_argument("--num-aug", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--input-file-prefix",
        default=None,
        help="Kaggle JSON prefix override; default uses the vendored TRM source.",
    )


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
    _build_arc_dataset(
        input_file_prefix=prefix,
        output_dir=root,
        augmentation=augmentation.make(),
        subsets=_ARC2_SUBSETS,
        test_set_name=_ARC2_TEST_SET,
    )
    with (root / _BUILD_PARAMS_FILE).open("w") as f:
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
        # A tree from a non-vendored source (a test fixture, an alternate
        # ARC-AGI-2 packaging) must never be silently accepted as the
        # canonical build: the revisions differ in content (172 vs 167 eval
        # outputs across packagings -- see RESEARCH.md).
        "source": input_file_prefix or "vendored-trm-kaggle",
    }


# Requires BOTH the matching params sentinel AND every ARC manifest file on disk: a
# params file alone is not proof of a complete tree (a crash can write params before --
# or independent of -- the arrays), so the data must be present too, or ensure would
# skip a build the loader then fails on.
def _params_match(root: Path, want: dict[str, object]) -> bool:
    """Whether ``root`` holds a completed compatible build."""
    path = root / _BUILD_PARAMS_FILE
    if not path.is_file():
        return False
    try:
        got = parse(path.read_text(), dict[str, object])
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


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
