"""Tests for the ARC-AGI-2 dataset builder (sentinel, tiny build)."""

from __future__ import annotations

from contextlib import nullcontext
from typing import TYPE_CHECKING, cast

import inspect
import json

from treekle import from_plain, loads

import numpy as np
import pytest

from priml.baselines.arcagi.augmentation import NO_TRAIN_SCALE_WEIGHTS
from priml.baselines.arcagi.scripts import build_arc2_dataset
from priml.baselines.arcagi.scripts.build_arc2_dataset import (
    _build_params,
    _params_match,
    ensure_arc2_dataset,
)
from priml.baselines.arcagi.scripts.build_dataset import arc_manifest


if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from numpy.typing import NDArray


def num_identifiers(root: Path) -> int:
    """Return the length of a built tree's ``identifiers.json``."""
    return len(from_plain(loads((root / "identifiers.json").read_text()), list[str]))


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


def test_ensure_defaults_remain_explicit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    defaults = inspect.signature(ensure_arc2_dataset).parameters
    assert (
        cast(object, defaults["train_scale_weights"].default) == NO_TRAIN_SCALE_WEIGHTS
    )
    assert cast(object, defaults["translation_prob"].default) == 1.0
    assert cast(object, defaults["scale_prob"].default) == 1.0
    assert cast(object, defaults["num_aug"].default) == 1_000
    assert cast(object, defaults["seed"].default) == 42
    assert cast(object, defaults["input_file_prefix"].default) is None

    builds: list[dict[str, object]] = []

    def run_build(**kwargs: object) -> None:
        cast("Callable[[], None]", kwargs["build"])()

    def record_tree(root: Path, **kwargs: object) -> None:
        del root
        builds.append(kwargs)

    monkeypatch.setattr(build_arc2_dataset, "run_rank_zero_build", run_build)
    monkeypatch.setattr(build_arc2_dataset, "_build_arc2_tree", record_tree)
    ensure_arc2_dataset(target=tmp_path / "default", input_file_prefix="source")

    assert len(builds) == 1
    assert builds[0]["num_aug"] == 1_000
    assert builds[0]["seed"] == 42


def test_ensure_passes_every_argument_to_the_vendored_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "vendored"
    source_prefix = str(tmp_path / "source" / "arc-agi")
    calls: list[tuple[Path, dict[str, object]]] = []

    def record_build(root: Path, **kwargs: object) -> None:
        calls.append((root, kwargs))

    def run_build(**kwargs: object) -> None:
        cast("Callable[[], None]", kwargs["build"])()

    monkeypatch.setattr(
        build_arc2_dataset,
        "KaggleSource",
        lambda: nullcontext(source_prefix),
    )
    monkeypatch.setattr(build_arc2_dataset, "run_rank_zero_build", run_build)
    monkeypatch.setattr(build_arc2_dataset, "_build_arc2_tree", record_build)

    ensure_arc2_dataset(
        target=target,
        train_scale_weights={2: 1.0, 3: 3.0},
        translation_prob=0.25,
        scale_prob=0.75,
        num_aug=6,
        seed=31,
    )

    assert calls == [
        (
            target,
            {
                "prefix": source_prefix,
                "train_scale_weights": {2: 1.0, 3: 3.0},
                "translation_prob": 0.25,
                "scale_prob": 0.75,
                "num_aug": 6,
                "seed": 31,
                "want": {
                    "subsets": ["training2", "evaluation2", "concept"],
                    "test_set_name": "evaluation2",
                    "train_scale_weights": [[2, 0.25], [3, 0.75]],
                    "translation_prob": 0.25,
                    "scale_prob": 0.75,
                    "num_aug": 6,
                    "seed": 31,
                    "source": "vendored-trm-kaggle",
                },
            },
        ),
    ]


def test_build_params_records_every_tree_input() -> None:
    assert _build_params(
        train_scale_weights={3: 2.0, 2: 2.0},
        translation_prob=0.25,
        scale_prob=0.75,
        num_aug=4,
        seed=9,
        input_file_prefix="/source/arc-agi",
    ) == {
        "subsets": ["training2", "evaluation2", "concept"],
        "test_set_name": "evaluation2",
        "train_scale_weights": [[2, 0.5], [3, 0.5]],
        "translation_prob": 0.25,
        "scale_prob": 0.75,
        "num_aug": 4,
        "seed": 9,
        "source": "/source/arc-agi",
    }
    assert (
        _build_params(
            train_scale_weights=NO_TRAIN_SCALE_WEIGHTS,
            translation_prob=1.0,
            scale_prob=1.0,
            num_aug=1_000,
            seed=42,
            input_file_prefix=None,
        )["source"]
        == "vendored-trm-kaggle"
    )


def test_ensure_builds_tiny_tree_and_sentinel(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level("INFO")
    prefix = write_tiny_kaggle_source(tmp_path)
    target = tmp_path / "parent" / "nested" / "arc2concept-tiny"
    root = ensure_arc2_dataset(
        target=target,
        num_aug=2,
        seed=7,
        input_file_prefix=prefix,
    )
    assert root == target
    assert f"ensure_arc2_dataset: building {target}" in caplog.messages
    assert f"Done. ARC-AGI-2 dataset at {target}" in caplog.messages
    for spec in arc_manifest():
        assert (root / spec.rel_path).is_file(), spec.rel_path
    params = from_plain(
        loads((root / "_build_params.json").read_text()),
        dict[str, object],
    )
    assert params["subsets"] == ["training2", "evaluation2", "concept"]
    assert params["test_set_name"] == "evaluation2"
    assert params["train_scale_weights"] == [[1, 1.0]]
    assert params["translation_prob"] == 1.0
    assert params["scale_prob"] == 1.0
    assert params["num_aug"] == 2
    assert params["seed"] == 7
    assert params["source"] == prefix
    assert _params_match(root, params)
    sentinel = root / "_build_params.json"
    sentinel.write_text("null")
    assert not _params_match(root, {})
    sentinel.write_text(json.dumps(params))
    # 6 source tasks, each <= 1 + num_aug identifiers, + the blank sentinel.
    n_ids = num_identifiers(root)
    assert 1 < n_ids <= 1 + 6 * 3
    identifiers = from_plain(
        loads((root / "identifiers.json").read_text()),
        list[str],
    )
    assert identifiers[0] == "<blank>"
    # Only the evaluation2 tasks form the test split.
    test_puzzles = from_plain(
        loads((root / "test_puzzles.json").read_text()),
        dict[str, object],
    )
    assert set(test_puzzles) == {"evaluation2_task0", "evaluation2_task1"}
    task = from_plain(test_puzzles["evaluation2_task1"], dict[str, object])
    assert len(from_plain(task["test"], list[object])) == 2
    second_root = ensure_arc2_dataset(
        target=tmp_path / "same-seed",
        num_aug=2,
        seed=7,
        input_file_prefix=prefix,
    )
    for split in ("train", "test"):
        for filename in ("all__inputs.npy", "all__labels.npy"):
            np.testing.assert_array_equal(
                cast("NDArray[np.int32]", np.load(root / split / filename)),
                cast("NDArray[np.int32]", np.load(second_root / split / filename)),
            )


def test_ensure_dispatches_rank_zero_build_with_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prefix = write_tiny_kaggle_source(tmp_path)
    names: list[str] = []

    def run_build(*, name: str, build: Callable[[], None]) -> None:
        names.append(name)
        build()

    monkeypatch.setattr(build_arc2_dataset, "run_rank_zero_build", run_build)
    ensure_arc2_dataset(
        target=tmp_path / "rank-safe",
        num_aug=1,
        input_file_prefix=prefix,
    )
    assert names == ["ensure_arc2_dataset"]


def test_ensure_noops_on_matching_sentinel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prefix = write_tiny_kaggle_source(tmp_path)
    target = tmp_path / "arc2concept-tiny"
    ensure_arc2_dataset(target=target, num_aug=2, seed=7, input_file_prefix=prefix)
    before = cast("NDArray[np.int32]", np.load(target / "train" / "all__inputs.npy"))
    mtime = (target / "train" / "all__inputs.npy").stat().st_mtime_ns

    def fail_build(**kwargs: object) -> None:
        pytest.fail(f"cached build repeated: {kwargs}")

    monkeypatch.setattr(build_arc2_dataset, "run_rank_zero_build", fail_build)
    cached_root = ensure_arc2_dataset(
        target=target,
        num_aug=2,
        seed=7,
        input_file_prefix=prefix,
    )
    assert cached_root == target
    assert (target / "train" / "all__inputs.npy").stat().st_mtime_ns == mtime
    after = cast("NDArray[np.int32]", np.load(target / "train" / "all__inputs.npy"))
    np.testing.assert_array_equal(before, after)


def test_disk_sentinel_survives_a_cold_process_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prefix = write_tiny_kaggle_source(tmp_path)
    target = tmp_path / "arc2concept-tiny"
    ensure_arc2_dataset(target=target, num_aug=2, seed=7, input_file_prefix=prefix)
    inputs = target / "train" / "all__inputs.npy"
    mtime = inputs.stat().st_mtime_ns
    monkeypatch.setattr(build_arc2_dataset, "_ensure_cache", {})

    result = ensure_arc2_dataset(
        target=target,
        num_aug=2,
        seed=7,
        input_file_prefix=prefix,
    )

    assert result == target
    assert inputs.stat().st_mtime_ns == mtime


def test_ensure_rebuilds_on_param_change(tmp_path: Path) -> None:
    prefix = write_tiny_kaggle_source(tmp_path)
    target = tmp_path / "arc2concept-tiny"
    ensure_arc2_dataset(target=target, num_aug=2, seed=7, input_file_prefix=prefix)
    mtime = (target / "train" / "all__inputs.npy").stat().st_mtime_ns
    ensure_arc2_dataset(target=target, num_aug=3, seed=7, input_file_prefix=prefix)
    assert (target / "train" / "all__inputs.npy").stat().st_mtime_ns != mtime
    params = from_plain(
        loads((target / "_build_params.json").read_text()),
        dict[str, object],
    )
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
