"""Tests for the ARC-AGI-2 dataset builder (slugs, sentinel, tiny build)."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import argparse
import inspect
import json
import logging

import numpy as np
import pytest

from priml.baselines.arcagi1.augmentation import NO_TRAIN_SCALE_WEIGHTS
from priml.baselines.arcagi1.scripts.build_dataset import (
    DEFAULT_SCALE_WEIGHTS,
    arc_manifest,
)
from priml.baselines.arcagi2.scripts import build_dataset
from priml.baselines.arcagi2.scripts.build_dataset import (
    _add_arguments,
    _build_params,
    _params_match,
    arc2_aug_policy_template,
    arc2_num_puzzle_identifiers,
    arc2_spatial_eval_template,
    ensure_arc2_dataset,
)
from priml.lib.custom_json import DictCodec, ListCodec, loads


if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray


class _Flags(Protocol):
    target_dir: str
    translation_prob: float
    scale_prob: float
    num_aug: int
    seed: int
    input_file_prefix: str | None


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
    assert arc2_aug_policy_template(translation_prob=0.2, scale_prob=0.2) == (
        "/datasets/arc2concept-aug-1000-tr0p2-sc0p2-2w1p0-n1000-s42"
    )
    aug = arc2_aug_policy_template(
        translation_prob=0.2,
        scale_prob=0.2,
        train_scale_weights={3: 2.0},
        num_aug=4,
        seed=9,
    )
    assert aug == "/datasets/arc2concept-aug-1000-tr0p2-sc0p2-3w1p0-n4-s9"
    spatial = arc2_spatial_eval_template(
        spatial_views=3,
        translation_prob=0.2,
        scale_prob=0.2,
        train_scale_weights={3: 2.0},
        num_aug=4,
        seed=9,
    )
    assert spatial == f"{aug}-spatialeval-v3"
    assert (
        arc2_spatial_eval_template(
            spatial_views=2,
            translation_prob=0.2,
            scale_prob=0.2,
        )
        == "/datasets/arc2concept-aug-1000-tr0p2-sc0p2-2w1p0-n1000-s42-spatialeval-v2"
    )


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

    monkeypatch.setattr(build_dataset, "run_rank_zero_build", run_build)
    monkeypatch.setattr(build_dataset, "_build_arc2_tree", record_tree)
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
        build_dataset,
        "KaggleSource",
        lambda: nullcontext(source_prefix),
    )
    monkeypatch.setattr(build_dataset, "run_rank_zero_build", run_build)
    monkeypatch.setattr(build_dataset, "_build_arc2_tree", record_build)

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


def test_argument_defaults_and_cli_main_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level("INFO")
    parser = argparse.ArgumentParser()
    _add_arguments(parser)
    defaults = cast(_Flags, parser.parse_args([]))
    help_text = parser.format_help()
    assert "Dataset root override." in help_text
    assert "--translation-prob" in help_text
    assert "--scale-prob" in help_text
    assert "--num-aug" in help_text
    assert "--seed" in help_text
    assert "--input-file-prefix" in help_text
    assert (
        defaults.target_dir,
        defaults.translation_prob,
        defaults.scale_prob,
        defaults.num_aug,
        defaults.seed,
        defaults.input_file_prefix,
    ) == ("", 1.0, 1.0, 1_000, 42, None)
    calls: list[dict[str, object]] = []

    def record_ensure(**kwargs: object) -> Path:
        calls.append(kwargs)
        return tmp_path

    def count_identifiers(root: Path) -> int:
        if root != tmp_path:
            pytest.fail(f"wrong root: {root}")
        return 7

    monkeypatch.setattr(build_dataset, "ensure_arc2_dataset", record_ensure)
    monkeypatch.setattr(build_dataset, "arc2_num_puzzle_identifiers", count_identifiers)
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_dataset",
            "--target-dir",
            str(tmp_path),
            "--translation-prob",
            "0.25",
            "--scale-prob",
            "0.5",
            "--num-aug",
            "3",
            "--seed",
            "17",
            "--input-file-prefix",
            "/source/arc-agi",
        ],
    )
    basic_config: list[dict[str, object]] = []

    def record_basic_config(**kwargs: object) -> None:
        basic_config.append(kwargs)

    monkeypatch.setattr(logging, "basicConfig", record_basic_config)
    assert build_dataset.main() == 0
    assert basic_config == [{"level": logging.INFO}]
    assert (
        f"ARC-AGI-2 dataset at {tmp_path} (num_puzzle_identifiers=7)" in caplog.messages
    )
    assert calls == [
        {
            "target": tmp_path,
            "train_scale_weights": DEFAULT_SCALE_WEIGHTS,
            "translation_prob": 0.25,
            "scale_prob": 0.5,
            "num_aug": 3,
            "seed": 17,
            "input_file_prefix": "/source/arc-agi",
        },
    ]


def test_main_requires_module_docstring(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(build_dataset, "__doc__", None)
    with pytest.raises(
        ValueError,
        match=r"^Expected __doc__ is not None[.]$",
    ):
        build_dataset.main()


def test_cli_help_includes_builder_description(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("sys.argv", ["build_dataset", "--help"])
    with pytest.raises(SystemExit) as exc:
        build_dataset.main()
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    description = help_text.split("usage:", 1)[1].split("\n\n", 1)[1]
    assert description.startswith("Build the ARC-AGI-2 TRM dataset")
    assert "Same builder, augmentation math, and on-disk schema" in description


def test_default_cli_uses_canonical_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def record_ensure(**kwargs: object) -> Path:
        calls.append(kwargs)
        return Path("built")

    def count_identifiers(root: Path) -> int:
        del root
        return 4

    monkeypatch.setattr(build_dataset, "ensure_arc2_dataset", record_ensure)
    monkeypatch.setattr(build_dataset, "arc2_num_puzzle_identifiers", count_identifiers)
    monkeypatch.setattr("sys.argv", ["build_dataset"])
    assert build_dataset.main() == 0
    assert calls == [
        {
            "target": Path("/opt/scratch/datasets/arc2concept-aug-1000"),
            "train_scale_weights": NO_TRAIN_SCALE_WEIGHTS,
            "translation_prob": 1.0,
            "scale_prob": 1.0,
            "num_aug": 1_000,
            "seed": 42,
            "input_file_prefix": None,
        },
    ]


def test_one_nondefault_probability_selects_policy_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def record_ensure(**kwargs: object) -> Path:
        calls.append(kwargs)
        return Path("built")

    def count_identifiers(root: Path) -> int:
        del root
        return 4

    monkeypatch.setattr(build_dataset, "ensure_arc2_dataset", record_ensure)
    monkeypatch.setattr(build_dataset, "arc2_num_puzzle_identifiers", count_identifiers)
    monkeypatch.setattr(
        "sys.argv",
        ["build_dataset", "--translation-prob", "0.2"],
    )
    assert build_dataset.main() == 0
    assert calls[0]["target"] == Path(
        "/opt/scratch/datasets/arc2concept-aug-1000-tr0p2-sc1p0-2w1p0-n1000-s42",
    )
    assert calls[0]["train_scale_weights"] == DEFAULT_SCALE_WEIGHTS


def test_policy_cli_uses_distinct_slugged_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def record_ensure(**kwargs: object) -> Path:
        calls.append(kwargs)
        return Path("built")

    def count_identifiers(root: Path) -> int:
        del root
        return 4

    monkeypatch.setattr(build_dataset, "ensure_arc2_dataset", record_ensure)
    monkeypatch.setattr(build_dataset, "arc2_num_puzzle_identifiers", count_identifiers)
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_dataset",
            "--translation-prob",
            "0.2",
            "--scale-prob",
            "0.3",
            "--num-aug",
            "5",
            "--seed",
            "8",
        ],
    )
    assert build_dataset.main() == 0
    assert calls == [
        {
            "target": Path(
                "/opt/scratch/datasets/arc2concept-aug-1000-tr0p2-sc0p3-2w1p0-n5-s8",
            ),
            "train_scale_weights": DEFAULT_SCALE_WEIGHTS,
            "translation_prob": 0.2,
            "scale_prob": 0.3,
            "num_aug": 5,
            "seed": 8,
            "input_file_prefix": None,
        },
    ]


def test_plain_nondefault_cli_requires_target_dir(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("sys.argv", ["build_dataset", "--num-aug", "3"])
    with pytest.raises(SystemExit) as exc:
        build_dataset.main()
    assert exc.value.code == 2
    assert capsys.readouterr().err.splitlines()[-1] == (
        "build_dataset: error: a plain-policy build with non-default "
        "--num-aug/--seed has no canonical directory; pass --target-dir explicitly."
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
    params = DictCodec.coerce(loads((root / "_build_params.json").read_text()))
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
    n_ids = arc2_num_puzzle_identifiers(root)
    assert 1 < n_ids <= 1 + 6 * 3
    identifiers = ListCodec.coerce(loads((root / "identifiers.json").read_text()), str)
    assert identifiers[0] == "<blank>"
    # Only the evaluation2 tasks form the test split.
    test_puzzles = DictCodec.coerce(loads((root / "test_puzzles.json").read_text()))
    assert set(test_puzzles) == {"evaluation2_task0", "evaluation2_task1"}
    task = DictCodec.coerce(test_puzzles["evaluation2_task1"])
    assert len(ListCodec.coerce(task["test"])) == 2
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

    monkeypatch.setattr(build_dataset, "run_rank_zero_build", run_build)
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

    monkeypatch.setattr(build_dataset, "run_rank_zero_build", fail_build)
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
    monkeypatch.setattr(build_dataset, "_ensure_cache", {})

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
