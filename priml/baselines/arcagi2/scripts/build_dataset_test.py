"""Tests for the ARC-AGI-2 dataset builder (slugs, sentinel, tiny build)."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import json

import numpy as np
import pytest

from priml.baselines.arcagi1.scripts.build_dataset import arc_manifest
from priml.baselines.arcagi2.scripts.build_dataset import (
    arc2_aug_policy_template,
    arc2_num_puzzle_identifiers,
    arc2_spatial_eval_template,
    ensure_arc2_dataset,
)
from priml.lib.custom_json import DictCodec, ListCodec, loads


if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray


def write_tiny_kaggle_source(root: Path) -> str:
    """Write a minimal 3-subset Kaggle-format ARC-AGI-2 source; return prefix.

    Two tasks per subset, one of them multi-test-input, so the build
    exercises subset routing and the multi-output test split.
    """
    grid_a = [[0, 1], [2, 3]]
    grid_b = [[4, 5, 6], [7, 8, 9]]
    prefix = root / "arc-agi"

    def _task(
        n_test: int,
    ) -> tuple[dict[str, list[dict[str, list[list[int]]]]], list[list[list[int]]]]:
        challenges = {
            "train": [{"input": grid_a, "output": grid_b}],
            "test": [{"input": grid_a} for _ in range(n_test)],
        }
        solutions = [grid_b for _ in range(n_test)]
        return challenges, solutions

    for subset in ("training2", "evaluation2", "concept"):
        challenges: dict[str, object] = {}
        solutions: dict[str, object] = {}
        for i, n_test in enumerate((1, 2)):
            name = f"{subset}_task{i}"
            challenges[name], solutions[name] = _task(n_test)
        (root / f"arc-agi_{subset}_challenges.json").write_text(json.dumps(challenges))
        (root / f"arc-agi_{subset}_solutions.json").write_text(json.dumps(solutions))
    return str(prefix)


def test_templates_are_pure_strings() -> None:
    aug = arc2_aug_policy_template(translation_prob=0.2, scale_prob=0.2)
    assert aug.startswith("/datasets/arc2concept-aug-1000-tr0p2-sc0p2")
    spatial = arc2_spatial_eval_template(
        spatial_views=2,
        translation_prob=0.2,
        scale_prob=0.2,
    )
    assert spatial == f"{aug}-spatialeval-v2"


def test_ensure_builds_tiny_tree_and_sentinel(tmp_path: Path) -> None:
    prefix = write_tiny_kaggle_source(tmp_path)
    target = tmp_path / "arc2concept-tiny"
    root = ensure_arc2_dataset(
        target=target,
        num_aug=2,
        seed=7,
        input_file_prefix=prefix,
    )
    assert root == target
    for spec in arc_manifest():
        assert (root / spec.rel_path).is_file(), spec.rel_path
    params = DictCodec.coerce(loads((root / "_build_params.json").read_text()))
    assert params["subsets"] == ["training2", "evaluation2", "concept"]
    assert params["test_set_name"] == "evaluation2"
    assert params["num_aug"] == 2
    assert params["seed"] == 7
    assert params["source"] == prefix
    # 6 source tasks, each <= 1 + num_aug identifiers, + the blank sentinel.
    n_ids = arc2_num_puzzle_identifiers(root)
    assert 1 < n_ids <= 1 + 6 * 3
    identifiers = ListCodec.coerce(loads((root / "identifiers.json").read_text()), str)
    assert identifiers[0] == "<blank>"
    # Only the evaluation2 tasks form the test split.
    test_puzzles = DictCodec.coerce(loads((root / "test_puzzles.json").read_text()))
    assert set(test_puzzles) == {"evaluation2_task0", "evaluation2_task1"}
    task = DictCodec.coerce(test_puzzles["evaluation2_task1"])
    assert len(ListCodec.coerce(task["test"])) == 2


def test_ensure_noops_on_matching_sentinel(tmp_path: Path) -> None:
    prefix = write_tiny_kaggle_source(tmp_path)
    target = tmp_path / "arc2concept-tiny"
    ensure_arc2_dataset(target=target, num_aug=2, seed=7, input_file_prefix=prefix)
    before = cast("NDArray[np.int32]", np.load(target / "train" / "all__inputs.npy"))
    mtime = (target / "train" / "all__inputs.npy").stat().st_mtime_ns
    ensure_arc2_dataset(target=target, num_aug=2, seed=7, input_file_prefix=prefix)
    assert (target / "train" / "all__inputs.npy").stat().st_mtime_ns == mtime
    after = cast("NDArray[np.int32]", np.load(target / "train" / "all__inputs.npy"))
    np.testing.assert_array_equal(before, after)


def test_ensure_rebuilds_on_param_change(tmp_path: Path) -> None:
    prefix = write_tiny_kaggle_source(tmp_path)
    target = tmp_path / "arc2concept-tiny"
    ensure_arc2_dataset(target=target, num_aug=2, seed=7, input_file_prefix=prefix)
    mtime = (target / "train" / "all__inputs.npy").stat().st_mtime_ns
    ensure_arc2_dataset(target=target, num_aug=3, seed=7, input_file_prefix=prefix)
    assert (target / "train" / "all__inputs.npy").stat().st_mtime_ns != mtime
    params = DictCodec.coerce(loads((target / "_build_params.json").read_text()))
    assert params["num_aug"] == 3


def test_incomplete_tree_triggers_rebuild(tmp_path: Path) -> None:
    prefix = write_tiny_kaggle_source(tmp_path)
    target = tmp_path / "arc2concept-tiny"
    ensure_arc2_dataset(target=target, num_aug=2, seed=7, input_file_prefix=prefix)
    (target / "test" / "all__labels.npy").unlink()
    ensure_arc2_dataset(target=target, num_aug=2, seed=7, input_file_prefix=prefix)
    assert (target / "test" / "all__labels.npy").is_file()


def test_different_source_triggers_rebuild(tmp_path: Path) -> None:
    prefix = write_tiny_kaggle_source(tmp_path)
    other = tmp_path / "other"
    other.mkdir()
    other_prefix = write_tiny_kaggle_source(other)
    target = tmp_path / "arc2concept-tiny"
    ensure_arc2_dataset(target=target, num_aug=2, seed=7, input_file_prefix=prefix)
    mtime = (target / "train" / "all__inputs.npy").stat().st_mtime_ns
    ensure_arc2_dataset(
        target=target,
        num_aug=2,
        seed=7,
        input_file_prefix=other_prefix,
    )
    assert (target / "train" / "all__inputs.npy").stat().st_mtime_ns != mtime


def test_crashed_rebuild_invalidates_stale_sentinel(tmp_path: Path) -> None:
    """A rebuild that dies mid-way must not leave a tree the OLD params accept."""
    prefix = write_tiny_kaggle_source(tmp_path)
    target = tmp_path / "arc2concept-tiny"
    ensure_arc2_dataset(target=target, num_aug=2, seed=7, input_file_prefix=prefix)
    # Trigger a params-changing rebuild that fails partway (missing source).
    with pytest.raises(FileNotFoundError):
        ensure_arc2_dataset(
            target=target,
            num_aug=3,
            seed=7,
            input_file_prefix=str(tmp_path / "nonexistent" / "arc-agi"),
        )
    # The old sentinel is gone, so the next ensure with the ORIGINAL params
    # rebuilds instead of adopting the possibly-mixed tree.
    assert not (target / "_build_params.json").is_file()
    ensure_arc2_dataset(target=target, num_aug=2, seed=7, input_file_prefix=prefix)
    assert (target / "_build_params.json").is_file()


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
