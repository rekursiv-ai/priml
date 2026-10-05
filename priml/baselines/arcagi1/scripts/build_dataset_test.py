"""Prepared ARC trees preserve the source recipe and existing destinations."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import hashlib
import json
import re
import subprocess
import tempfile

import numpy as np
import pytest
import torch

from priml.baselines.arcagi1.augmentation import (
    ArcAugmentation,
    ArcSpec,
    ColorDihedral,
    canonicalize_arc_grid,
    grid_hash,
)
from priml.baselines.arcagi1.scripts import build_dataset
from priml.baselines.arcagi1.scripts.build_dataset import (
    KaggleSource,
    _ArcBuild,
    _int_grid,
    _Puzzle,
    _puzzle_group_hash,
    _shape,
    _spatial_eval_tag,
    _validate_prob,
    _validate_scalable,
    _write_split,
    arc_manifest,
    aug_policy_dataset_dir,
    aug_policy_slug,
    aug_policy_template,
    build,
    build_arc_dataset,
    ensure_arc_dataset,
)
from priml.data.ensure import DataSpec, EnsureResult
from priml.lib.custom_json import ReadError, convert, parse


if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray


def _load_array(path: Path) -> NDArray[np.generic]:
    with path.open("rb") as stream:
        return np.lib.format.read_array(stream, allow_pickle=False)


@pytest.fixture
def source_prefix(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    source.mkdir()
    for subset in ("training", "evaluation", "concept"):
        puzzles = {
            f"{subset}-{index}": {
                "train": [
                    {"input": [[0, index + 1, 2], [3, 4, 5]], "output": [[6], [7]]},
                    {"input": [[8, 9]], "output": [[index, 0], [2, 3]]},
                ],
                "test": [{"input": [[1, 2], [3, 4]]}],
            }
            for index in range(3)
        }
        (source / f"arc_{subset}_challenges.json").write_text(json.dumps(puzzles))
        (source / f"arc_{subset}_solutions.json").write_text(
            json.dumps({name: [[[4, 3], [2, 1]]] for name in puzzles}),
        )
    return source / "arc"


def _example(
    *,
    rows: int,
    columns: int,
) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
    grid: NDArray[np.uint8] = np.zeros((rows, columns), dtype=np.uint8)
    return grid, grid


def test_kaggle_source_rejects_a_different_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    revision = "a" * 40

    def run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        del kwargs
        stdout = revision if args[-1] == "HEAD" else ""
        return subprocess.CompletedProcess(args, 0, stdout=stdout)

    monkeypatch.setattr(
        "priml.baselines.arcagi1.scripts.build_dataset.subprocess.run",
        run,
    )
    source = KaggleSource()
    with pytest.raises(
        RuntimeError,
        match=(
            "source revision mismatch: got "
            f"{revision}, expected c01103738605ba39d1430519b1ee0c62f4c707f8"
        ),
    ):
        source.__enter__()
    assert source._temporary is not None
    source._temporary.cleanup()


def test_kaggle_source_clones_pinned_revision_and_cleans_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []
    options: list[dict[str, object]] = []

    def run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(args)
        options.append(kwargs)
        return subprocess.CompletedProcess(
            args,
            0,
            stdout="c01103738605ba39d1430519b1ee0c62f4c707f8\n",
        )

    monkeypatch.setattr(
        "priml.baselines.arcagi1.scripts.build_dataset.subprocess.run",
        run,
    )
    source = KaggleSource()
    assert source._temporary is None
    with source as prefix:
        temporary = source._temporary
        assert temporary is not None
        temporary_root = Path(temporary.name)
        assert (
            prefix
            == temporary_root
            / "TinyRecursiveModels"
            / "kaggle"
            / "combined"
            / "arc-agi"
        )
        assert temporary_root.is_dir()
        assert temporary_root.name.startswith("arcagi1-source-")
    assert not temporary_root.exists()
    clone = str(temporary_root / "TinyRecursiveModels")
    revision = "c01103738605ba39d1430519b1ee0c62f4c707f8"
    assert commands == [
        [
            "git",
            "clone",
            "--no-checkout",
            "https://github.com/SamsungSAILMontreal/TinyRecursiveModels.git",
            clone,
        ],
        ["git", "-C", clone, "checkout", revision],
        ["git", "-C", clone, "rev-parse", "HEAD"],
    ]
    assert options == [
        {"check": True},
        {"check": True},
        {"check": True, "capture_output": True, "text": True},
    ]


def test_build_defaults_and_seed_match_the_documented_recipe(
    tmp_path: Path,
    source_prefix: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[ArcAugmentation] = []

    def capture_ensure(
        *,
        target_dir: Path,
        augmentation: ArcAugmentation,
        input_file_prefix: str | None = None,
    ) -> EnsureResult:
        del target_dir, input_file_prefix
        captured.append(augmentation)
        return EnsureResult.PRESENT

    with monkeypatch.context() as patch:
        patch.setattr(build_dataset, "ensure_arc_dataset", capture_ensure)
        build(
            dataset_dir=str(tmp_path / "defaults"),
            input_file_prefix=str(source_prefix),
        )

    assert len(captured) == 1
    defaults = captured[0].config
    assert defaults.spatial.translation_prob == 0.1
    assert defaults.spatial.scale_prob == 0.1
    assert defaults.num_aug == 1_000
    assert defaults.seed == 42

    actual = tmp_path / "build"
    build(
        dataset_dir=str(actual),
        input_file_prefix=str(source_prefix),
        translation_prob=0,
        scale_prob=0,
        num_aug=0,
    )
    config = ArcAugmentation.Config(spec=ArcSpec())
    config.num_aug = 0
    config.spatial.translation_prob = 0
    config.spatial.scale_prob = 0
    expected = tmp_path / "reference"
    build_arc_dataset(
        target_dir=expected,
        input_file_prefix=str(source_prefix),
        augmentation=config.make(),
    )
    for relative_path in (
        Path("identifiers.json"),
        Path("test_puzzles.json"),
        Path("train/all__inputs.npy"),
        Path("train/all__labels.npy"),
        Path("test/all__inputs.npy"),
        Path("test/all__labels.npy"),
    ):
        actual_path = actual / relative_path
        expected_path = expected / relative_path
        if actual_path.suffix == ".npy":
            assert np.array_equal(_load_array(actual_path), _load_array(expected_path))
        else:
            assert actual_path.read_bytes() == expected_path.read_bytes()


def test_missing_solution_files_fill_dummy_outputs_and_log_context(
    tmp_path: Path,
    source_prefix: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    solution_file = source_prefix.parent / "arc_evaluation_solutions.json"
    solution_file.unlink()
    config = ArcAugmentation.Config(spec=ArcSpec())
    config.num_aug = 0
    logger_name = "priml.baselines.arcagi1.scripts.build_dataset"
    caplog.set_level("WARNING", logger=logger_name)

    target = tmp_path / "dataset"
    build_arc_dataset(
        target_dir=target,
        input_file_prefix=str(source_prefix),
        augmentation=config.make(),
    )

    warnings = [record.getMessage() for record in caplog.records]
    assert warnings == ["evaluation solutions not found, filling with dummy"]
    test_puzzles = parse(
        (target / "test_puzzles.json").read_text(),
        dict[str, object],
    )
    outputs = [
        convert(
            convert(
                convert(puzzle, dict[str, object])["test"],
                list[object],
            )[0],
            dict[str, object],
        )["output"]
        for puzzle in test_puzzles.values()
    ]
    assert outputs == [[[0]], [[0]], [[0]]]


def test_grid_and_validation_helpers() -> None:
    grid = [[1, 2, 3], [4, 5, 6]]
    assert _int_grid(grid) == grid
    with pytest.raises(ReadError):
        _int_grid([[1, "2", 3], [4, 5, 6]])
    array = np.asarray(grid, dtype=np.uint8)
    assert _shape(array) == (2, 3)
    with pytest.raises(
        ValueError,
        match=r"translation_prob must be in \[0, 1\], got -0.1\.",
    ):
        _validate_prob("translation_prob", -0.1)
    with pytest.raises(
        ValueError,
        match=r"scale_prob must be in \[0, 1\], got 1.1\.",
    ):
        _validate_prob("scale_prob", 1.1)
    _validate_prob("translation_prob", 0.0)
    _validate_prob("scale_prob", 1.0)
    with pytest.raises(ValueError, match="nan"):
        _validate_prob("translation_prob", float("nan"))
    message = (
        "scale_prob=0.5 > 0 but train_scale_weights {1: 1.0} has no mass on a "
        "factor > 1, so the scale gate is a silent no-op; add a factor > 1 or "
        "set scale_prob=0."
    )
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        _validate_scalable("scale_prob", 0.5, {1: 1.0})


def test_spatial_tag_bounds_scale_and_fallback() -> None:
    config = ArcAugmentation.Config(spec=ArcSpec())
    config.spatial_eval_scale = 2
    augmentation = config.make()
    puzzle = _Puzzle(name="puzzle", examples=[_example(rows=2, columns=3)])
    tag = _spatial_eval_tag(
        puzzle,
        augmentation=augmentation,
        rng=np.random.default_rng(3),
    )
    assert tag is not None
    scale, row, col = tag
    assert (scale, row, col) == (2, 4, 5)

    config.spatial_eval_scale = 1
    unit = config.make()
    tag = _spatial_eval_tag(puzzle, augmentation=unit, rng=np.random.default_rng(3))
    assert tag is not None
    assert tag == (1, 5, 6)

    config.spatial_eval_scale = 1
    identity = _Puzzle(
        name="identity",
        examples=[_example(rows=30, columns=29)],
    )
    assert (
        _spatial_eval_tag(
            identity,
            augmentation=config.make(),
            rng=np.random.default_rng(3),
        )
        is None
    )

    config.spatial_eval_scale = 2
    too_large = _Puzzle(
        name="large",
        examples=[_example(rows=16, columns=15)],
    )
    tag = _spatial_eval_tag(
        too_large,
        augmentation=config.make(),
        rng=np.random.default_rng(3),
    )
    assert tag is not None
    assert tag == (1, 12, 1)

    row_boundary = _Puzzle(
        name="row-boundary",
        examples=[_example(rows=15, columns=14)],
    )
    tag = _spatial_eval_tag(
        row_boundary,
        augmentation=config.make(),
        rng=np.random.default_rng(3),
    )
    assert tag is not None
    assert tag[0] == 2
    assert tag[1] == 0
    assert 0 <= tag[2] <= 2

    col_boundary = _Puzzle(
        name="col-boundary",
        examples=[_example(rows=14, columns=15)],
    )
    tag = _spatial_eval_tag(
        col_boundary,
        augmentation=config.make(),
        rng=np.random.default_rng(3),
    )
    assert tag is not None
    assert tag[0] == 2
    assert 0 <= tag[1] <= 2
    assert tag[2] == 0

    col_overflow = _Puzzle(
        name="col-overflow",
        examples=[_example(rows=2, columns=16)],
    )
    tag = _spatial_eval_tag(
        col_overflow,
        augmentation=config.make(),
        rng=np.random.default_rng(3),
    )
    assert tag is not None
    assert tag[0] == 1


def test_spatial_tag_rejects_nonpositive_scale_with_context() -> None:
    config = ArcAugmentation.Config(spec=ArcSpec())
    config.spatial_eval_scale = 0
    puzzle = _Puzzle(name="puzzle", examples=[_example(rows=2, columns=3)])
    with pytest.raises(
        ValueError,
        match=r"spatial_eval_scale must be positive; got 0\.",
    ):
        _spatial_eval_tag(
            puzzle,
            augmentation=config.make(),
            rng=np.random.default_rng(3),
        )


def test_write_split_only_augments_training_and_rebuilds_partial_splits(
    tmp_path: Path,
) -> None:
    config = ArcAugmentation.Config(spec=ArcSpec())
    config.num_aug = 0
    config.spatial.translation_prob = 1
    config.spatial.scale_prob = 0
    augmentation = config.make()
    grid = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.uint8)
    puzzle = _Puzzle(name="puzzle", examples=[(grid, grid), (grid, grid)])
    split = {"all": [[puzzle]]}
    target = tmp_path / "nested" / "tree"
    _write_split(
        target,
        "train",
        split,
        {"puzzle": 1},
        augmentation=augmentation,
        rng=np.random.default_rng(3),
        spatial_rng=np.random.default_rng(3),
    )
    train_inputs = _array(target / "train" / "all__inputs.npy")
    assert np.array_equal(train_inputs[:, 0], [0, 2])
    _write_split(
        target,
        "train",
        split,
        {"puzzle": 1},
        augmentation=augmentation,
        rng=np.random.default_rng(3),
        spatial_rng=np.random.default_rng(3),
    )
    _write_split(
        target,
        "test",
        split,
        {"puzzle": 1},
        augmentation=augmentation,
        rng=np.random.default_rng(3),
        spatial_rng=np.random.default_rng(3),
    )
    assert _load_array(target / "test" / "all__inputs.npy")[:, 0].tolist() == [2, 2]


def test_write_split_leaves_exactly_one_training_example_unaugmented(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = ArcAugmentation.Config(spec=ArcSpec())
    config.num_aug = 0
    augmentation = config.make()
    grid = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.uint8)
    training_flags: list[bool] = []

    def pack(
        inp: np.ndarray,
        out: np.ndarray,
        *,
        training: bool,
        rng: np.random.Generator,
    ) -> list[NDArray[np.uint8]]:
        del rng
        training_flags.append(training)
        return [inp, out]

    monkeypatch.setattr(augmentation.spatial, "pack", pack)
    no_aug_idx = int(np.random.default_rng(0).integers(2))
    _write_split(
        tmp_path,
        "train",
        {"all": [[_Puzzle(name="p", examples=[(grid, grid), (grid, grid)])]]},
        {"p": 1},
        augmentation=augmentation,
        rng=np.random.default_rng(0),
        spatial_rng=np.random.default_rng(0),
    )
    assert training_flags == [index != no_aug_idx for index in range(2)]


def test_puzzle_group_hash_tracks_all_grids() -> None:
    first = np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
    second = np.asarray([[1, 2, 3], [4, 5, 7]], dtype=np.uint8)
    group = {
        ("train", "first"): _Puzzle(name="first", examples=[(first, first)]),
        ("test", "second"): _Puzzle(name="second", examples=[(second, first)]),
    }
    other = {
        ("train", "first"): _Puzzle(name="first", examples=[(first, second)]),
        ("test", "second"): _Puzzle(name="second", examples=[(second, first)]),
    }
    assert _puzzle_group_hash(group) != _puzzle_group_hash(other)
    expected_hashes = sorted(
        (grid_hash(first), grid_hash(first), grid_hash(second), grid_hash(first)),
    )
    assert (
        _puzzle_group_hash(group)
        == hashlib.sha256(
            "|".join(expected_hashes).encode(),
        ).hexdigest()
    )
    assert _puzzle_group_hash(group) == _puzzle_group_hash(
        dict(reversed(tuple(group.items()))),
    )


def test_arc_manifest_lists_every_required_artifact_in_order() -> None:
    assert [file.rel_path for file in arc_manifest()] == [
        "identifiers.json",
        "test_puzzles.json",
        *(
            f"{split}/{name}"
            for split in ("train", "test")
            for name in (
                "all__inputs.npy",
                "all__labels.npy",
                "all__puzzle_indices.npy",
                "all__group_indices.npy",
                "all__puzzle_identifiers.npy",
                "dataset.json",
            )
        ),
    ]


def test_aug_policy_slug_and_paths() -> None:
    assert aug_policy_slug(translation_prob=0.1, scale_prob=0.1) == (
        "tr0p1-sc0p1-2w1p0-n1000-s42"
    )
    assert (
        aug_policy_slug(
            translation_prob=0.1234567,
            scale_prob=0.1234567,
        )
        == "tr0p123457-sc0p123457-2w1p0-n1000-s42"
    )
    with pytest.raises(ValueError, match="translation_prob must be"):
        aug_policy_slug(translation_prob=-0.1, scale_prob=0)
    with pytest.raises(ValueError, match="scale_prob must be"):
        aug_policy_slug(translation_prob=0, scale_prob=1.1)
    with pytest.raises(ValueError, match=r"scale_prob=0\.1"):
        aug_policy_slug(
            translation_prob=0,
            scale_prob=0.1,
            train_scale_weights={1: 1},
        )
    assert (
        aug_policy_slug(
            translation_prob=0.25,
            scale_prob=0,
            train_scale_weights={1: 1},
            num_aug=3,
            seed=5,
        )
        == "tr0p25-sc0-n3-s5"
    )
    slug = "tr0p1-sc0p1-2w1p0-n1000-s42"
    assert aug_policy_template(translation_prob=0.1, scale_prob=0.1) == (
        f"/datasets/arc1concept-aug-1000-{slug}"
    )
    assert (
        aug_policy_template(
            translation_prob=0.25,
            scale_prob=0,
            train_scale_weights={1: 1},
            num_aug=3,
            seed=5,
        )
        == "/datasets/arc1concept-aug-1000-tr0p25-sc0-n3-s5"
    )
    assert (
        aug_policy_template(
            translation_prob=0.25,
            scale_prob=0.5,
            train_scale_weights={3: 1},
            num_aug=3,
            seed=5,
        )
        == "/datasets/arc1concept-aug-1000-tr0p25-sc0p5-3w1p0-n3-s5"
    )
    assert aug_policy_dataset_dir(
        translation_prob=0.1,
        scale_prob=0.1,
        base_dir="/scratch",
        working_dir="/datasets",
    ) == Path(f"/scratch/datasets/arc1concept-aug-1000-{slug}")
    assert aug_policy_dataset_dir(
        translation_prob=0.25,
        scale_prob=0,
        train_scale_weights={1: 1},
        num_aug=3,
        seed=5,
        base_dir="/scratch",
    ) == Path("/scratch/datasets/arc1concept-aug-1000-tr0p25-sc0-n3-s5")
    assert aug_policy_dataset_dir(
        translation_prob=0.25,
        scale_prob=0,
        train_scale_weights={1: 1},
        num_aug=3,
        seed=5,
    ) == Path("/datasets/arc1concept-aug-1000-tr0p25-sc0-n3-s5")
    assert aug_policy_dataset_dir(
        translation_prob=0.25,
        scale_prob=0.5,
        train_scale_weights={3: 1},
        num_aug=3,
        seed=5,
        base_dir="/scratch",
    ) == Path("/scratch/datasets/arc1concept-aug-1000-tr0p25-sc0p5-3w1p0-n3-s5")


def test_build_uses_explicit_destination_and_source(
    tmp_path: Path,
    source_prefix: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    target = tmp_path / "nested" / "dataset"
    caplog.set_level(
        "INFO",
        logger="priml.baselines.arcagi1.scripts.build_dataset",
    )
    build(
        dataset_dir=str(target),
        input_file_prefix=str(source_prefix),
        translation_prob=0,
        scale_prob=0,
        num_aug=0,
    )
    assert (target / "train" / "all__inputs.npy").is_file()
    assert (target / "test" / "all__labels.npy").is_file()
    assert caplog.records[-1].getMessage() == f"Aug-policy ARC dataset at {target}"
    assert not list(target.glob("*/all__spatial_tags.npy"))
    custom_target = tmp_path / "custom-grid"
    build(
        dataset_dir=str(custom_target),
        input_file_prefix=str(source_prefix),
        translation_prob=0,
        scale_prob=0,
        num_aug=0,
        spec=ArcSpec(max_grid=5),
    )
    assert _load_array(custom_target / "train" / "all__inputs.npy").shape == (24, 25)
    custom_metadata = parse(
        (custom_target / "train" / "dataset.json").read_text(),
        dict[str, object],
    )
    assert custom_metadata["seq_len"] == 25
    identity_target = tmp_path / "identity-scale-policy"
    build(
        dataset_dir=str(identity_target),
        input_file_prefix=str(source_prefix),
        scale_prob=0,
        train_scale_weights={1: 1},
        num_aug=0,
    )
    assert (identity_target / "train" / "all__inputs.npy").is_file()
    with pytest.raises(ValueError, match="translation_prob must be"):
        build(
            dataset_dir=str(tmp_path / "invalid-translation"),
            input_file_prefix=str(source_prefix),
            translation_prob=-0.1,
            scale_prob=0,
            num_aug=0,
        )
    with pytest.raises(ValueError, match="scale_prob must be"):
        build(
            dataset_dir=str(tmp_path / "invalid-scale"),
            input_file_prefix=str(source_prefix),
            translation_prob=0,
            scale_prob=1.1,
            num_aug=0,
        )


def test_occupied_destination_is_unchanged(tmp_path: Path, source_prefix: Path) -> None:
    target = tmp_path / "dataset"
    target.mkdir()
    sentinel = target / "existing"
    sentinel.write_bytes(b"original")
    with pytest.raises(
        FileExistsError,
        match=f"Dataset destination is not empty: {target}",
    ):
        build_arc_dataset(
            target_dir=target,
            input_file_prefix=str(source_prefix),
            augmentation=ArcAugmentation.Config(spec=ArcSpec()).make(),
        )
    assert {path.name: path.read_bytes() for path in target.iterdir()} == {
        "existing": b"original",
    }


def test_spatial_eval_reuses_puzzle_id_and_canonical_answer(
    tmp_path: Path,
    source_prefix: Path,
) -> None:
    """Each spatial view shares its puzzle's id and restores to the same grids."""
    config = ArcAugmentation.Config(spec=ArcSpec())
    assert isinstance(config.spec, ArcSpec)
    config.num_aug = 0
    config.spatial_eval_views = True
    target = tmp_path / "dataset"
    build_arc_dataset(
        target_dir=target,
        input_file_prefix=str(source_prefix),
        augmentation=config.make(),
    )
    repeated_target = tmp_path / "repeated-dataset"
    build_arc_dataset(
        target_dir=repeated_target,
        input_file_prefix=str(source_prefix),
        augmentation=config.make(),
    )
    test_dir = target / "test"
    inputs = _array(test_dir / "all__inputs.npy")
    labels = _array(test_dir / "all__labels.npy")
    tags = _array(test_dir / "all__spatial_tags.npy")
    assert np.array_equal(
        tags,
        _array(repeated_target / "test" / "all__spatial_tags.npy"),
    )
    indices = _array(test_dir / "all__puzzle_indices.npy")
    ids = _array(test_dir / "all__puzzle_identifiers.npy")
    group_indices = _array(test_dir / "all__group_indices.npy")
    names = parse((target / "identifiers.json").read_text(), list[str])
    assert len(ids) == len(tags) == 6
    assert _ints(indices) == list(range(7))
    assert _ints(group_indices) == [0, 2, 4, 6]
    assert not (target / "train" / "all__spatial_tags.npy").exists()
    assert _load_array(test_dir / "all__spatial_tags.npy").dtype == np.int32
    assert parse(
        (test_dir / "dataset.json").read_text(),
        dict[str, object],
    ) == {
        "pad_id": 0,
        "ignore_label_id": 0,
        "blank_identifier_id": 0,
        "vocab_size": 12,
        "seq_len": 900,
        "num_puzzle_identifiers": 10,
        "total_groups": 3,
        "mean_puzzle_examples": 1.0,
        "total_puzzles": 6,
        "sets": ["all"],
    }
    id_list = _ints(ids)
    tag_rows = [_ints(tags[row : row + 1].reshape(-1)) for row in range(len(id_list))]
    for puzzle in range(0, len(id_list), 2):
        assert id_list[puzzle] == id_list[puzzle + 1]
        assert tag_rows[puzzle] == [1, 0, 0]
        assert tag_rows[puzzle + 1][0] == 2
        for rows in (inputs, labels):
            canonical = [
                canonicalize_arc_grid(
                    torch.from_numpy(rows[row : row + 1].reshape(-1)),
                    name=names[id_list[puzzle]],
                    spatial_tags=torch.from_numpy(tags[row : row + 1].reshape(-1)),
                    spec=config.spec,
                )
                for row in (puzzle, puzzle + 1)
            ]
            assert canonical[0][0] == canonical[1][0]
            assert torch.equal(canonical[0][1], canonical[1][1])


def test_augmented_views_are_grouped_with_their_source(
    tmp_path: Path,
    source_prefix: Path,
) -> None:
    spec = ArcSpec()
    config = ArcAugmentation.Config(spec=spec)
    assert isinstance(config.transform, ColorDihedral.Config)
    config.transform.transforms = (1, 2, 3)
    config.transform.colors = ()
    config.num_aug = 2
    config.retries_factor = 20
    target = tmp_path / "dataset"
    build_arc_dataset(
        target_dir=target,
        input_file_prefix=str(source_prefix),
        augmentation=config.make(),
    )
    train_inputs = _array(target / "train" / "all__inputs.npy")
    assert 0 in train_inputs[:, 0]
    ids = _ints(_array(target / "train" / "all__puzzle_identifiers.npy"))
    groups = _ints(_array(target / "train" / "all__group_indices.npy"))
    names = parse((target / "identifiers.json").read_text(), list[str])
    assert groups == list(range(0, 28, 3))
    assert len(ids) == 27
    for start in range(0, len(ids), 3):
        original, *augmented = (names[ids[index]] for index in range(start, start + 3))
        assert len(set(augmented)) == 2
        assert all(
            name.startswith(f"{original}{spec.puzzle_id_separator}")
            for name in augmented
        )
    single_config = ArcAugmentation.Config(spec=spec)
    assert isinstance(single_config.transform, ColorDihedral.Config)
    single_config.transform.transforms = (1,)
    single_config.transform.colors = ()
    single_config.num_aug = 1
    single_target = tmp_path / "single-augmented-view"
    build_arc_dataset(
        target_dir=single_target,
        input_file_prefix=str(source_prefix),
        augmentation=single_config.make(),
    )
    assert _ints(_array(single_target / "train" / "all__group_indices.npy")) == list(
        range(0, 19, 2),
    )


def test_plain_build_writes_no_spatial_tags(
    tmp_path: Path,
    source_prefix: Path,
) -> None:
    """Without spatial views the tree keeps the reference file set."""
    config = ArcAugmentation.Config(spec=ArcSpec())
    config.num_aug = 0
    target = tmp_path / "dataset"
    build_arc_dataset(
        target_dir=target,
        input_file_prefix=str(source_prefix),
        augmentation=config.make(),
    )
    assert not list(target.glob("*/all__spatial_tags.npy"))
    assert (target / "train" / "all__inputs.npy").is_file()
    assert (target / "train" / "all__labels.npy").is_file()
    repeated_target = tmp_path / "repeated-dataset"
    build_arc_dataset(
        target_dir=repeated_target,
        input_file_prefix=str(source_prefix),
        augmentation=config.make(),
    )
    for split in ("train", "test"):
        for name in (
            "inputs",
            "labels",
            "puzzle_indices",
            "group_indices",
            "puzzle_identifiers",
        ):
            array_path = target / split / f"all__{name}.npy"
            array = _load_array(array_path)
            repeated = _load_array(repeated_target / split / f"all__{name}.npy")
            assert np.array_equal(array, repeated)
            if name in ("puzzle_indices", "group_indices", "puzzle_identifiers"):
                assert array.dtype == np.int32
    assert _ints(_array(target / "train" / "all__puzzle_indices.npy")) == [
        0,
        3,
        6,
        9,
        11,
        13,
        15,
        18,
        21,
        24,
    ]
    assert _ints(_array(target / "train" / "all__group_indices.npy")) == list(
        range(10),
    )
    assert _ints(_array(target / "test" / "all__puzzle_indices.npy")) == [0, 1, 2, 3]
    assert _ints(_array(target / "test" / "all__group_indices.npy")) == [0, 1, 2, 3]
    test_puzzles = parse(
        (target / "test_puzzles.json").read_text(),
        dict[str, object],
    )
    assert set(test_puzzles) == {f"evaluation-{index}" for index in range(3)}
    identifiers = parse(
        (target / "identifiers.json").read_text(),
        list[str],
    )
    assert identifiers[0] == "<blank>"
    assert set(identifiers[1:]) == {
        f"{subset}-{index}"
        for subset in ("training", "evaluation", "concept")
        for index in range(3)
    }
    assert test_puzzles["evaluation-0"] == {
        "train": [
            {"input": [[0, 1, 2], [3, 4, 5]], "output": [[6], [7]]},
            {"input": [[8, 9]], "output": [[0, 0], [2, 3]]},
        ],
        "test": [{"input": [[1, 2], [3, 4]], "output": [[4, 3], [2, 1]]}],
    }
    assert _array(target / "train" / "all__inputs.npy").shape == (24, 900)
    assert _array(target / "test" / "all__inputs.npy").shape == (3, 900)
    assert parse(
        (target / "train" / "dataset.json").read_text(),
        dict[str, object],
    ) == {
        "pad_id": 0,
        "ignore_label_id": 0,
        "blank_identifier_id": 0,
        "vocab_size": 12,
        "seq_len": 900,
        "num_puzzle_identifiers": 10,
        "total_groups": 9,
        "mean_puzzle_examples": 24 / 9,
        "total_puzzles": 9,
        "sets": ["all"],
    }
    assert parse(
        (target / "test" / "dataset.json").read_text(),
        dict[str, object],
    ) == {
        "pad_id": 0,
        "ignore_label_id": 0,
        "blank_identifier_id": 0,
        "vocab_size": 12,
        "seq_len": 900,
        "num_puzzle_identifiers": 10,
        "total_groups": 3,
        "mean_puzzle_examples": 1.0,
        "total_puzzles": 3,
        "sets": ["all"],
    }


def test_ensure_arc_dataset_uses_rank_zero_callback_and_caches_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "dataset"
    names: list[str] = []
    specs: list[DataSpec] = []

    def build_on_rank_zero(*, name: str, build: Callable[[], None]) -> None:
        names.append(name)
        build()

    def ensure_data(spec: DataSpec) -> EnsureResult:
        specs.append(spec)
        return EnsureResult.DOWNLOADED

    monkeypatch.setattr(
        "priml.baselines.arcagi1.scripts.build_dataset._ensure_cache",
        {},
    )
    monkeypatch.setattr(
        "priml.baselines.arcagi1.scripts.build_dataset.run_rank_zero_build",
        build_on_rank_zero,
    )
    monkeypatch.setattr(
        "priml.baselines.arcagi1.scripts.build_dataset.ensure_data",
        ensure_data,
    )
    augmentation = ArcAugmentation.Config(spec=ArcSpec()).make()
    result = ensure_arc_dataset(
        target_dir=target,
        augmentation=augmentation,
        input_file_prefix="source",
    )
    assert result is EnsureResult.DOWNLOADED
    assert names == ["ensure_arc_dataset"]
    assert len(specs) == 1
    assert specs[0].target_dir == target
    assert [file.rel_path for file in specs[0].manifest] == [
        file.rel_path for file in arc_manifest()
    ]
    assert (
        ensure_arc_dataset(
            target_dir=target,
            augmentation=augmentation,
            input_file_prefix="missing-source",
        )
        is result
    )
    assert names == ["ensure_arc_dataset"]


def test_ensure_arc_dataset_returns_present_when_rank_skips_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    names: list[str] = []

    def skip_nonzero_rank(*, name: str, build: Callable[[], None]) -> None:
        names.append(name)
        assert callable(build)

    monkeypatch.setattr(
        "priml.baselines.arcagi1.scripts.build_dataset._ensure_cache",
        {},
    )
    monkeypatch.setattr(
        "priml.baselines.arcagi1.scripts.build_dataset.run_rank_zero_build",
        skip_nonzero_rank,
    )
    result = ensure_arc_dataset(
        target_dir=tmp_path / "dataset",
        augmentation=ArcAugmentation.Config(spec=ArcSpec()).make(),
        input_file_prefix="source",
    )
    assert result is EnsureResult.PRESENT
    assert names == ["ensure_arc_dataset"]


def test_ensure_arc_dataset_returns_and_caches_outcome(
    tmp_path: Path,
    source_prefix: Path,
) -> None:
    target = tmp_path / "dataset"
    config = ArcAugmentation.Config(spec=ArcSpec())
    config.num_aug = 0
    augmentation = config.make()
    result = ensure_arc_dataset(
        target_dir=target,
        augmentation=augmentation,
        input_file_prefix=str(source_prefix),
    )
    assert result is EnsureResult.DOWNLOADED
    assert (
        ensure_arc_dataset(
            target_dir=target,
            augmentation=augmentation,
            input_file_prefix=str(tmp_path / "missing"),
        )
        is result
    )


def test_arc_build_fetch_runs_once_and_logs_missing_path(
    tmp_path: Path,
    source_prefix: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    target = tmp_path / "dataset"
    target.mkdir()
    config = ArcAugmentation.Config(spec=ArcSpec())
    config.num_aug = 0
    augmentation = config.make()
    fetch = _ArcBuild(
        target_dir=target,
        augmentation=augmentation,
        input_file_prefix=str(source_prefix),
    )
    assert fetch._built is False
    caplog.set_level(
        "INFO",
        logger="priml.baselines.arcagi1.scripts.build_dataset",
    )
    fetch(rel_path="train/all__inputs.npy", dest=target / "train" / "all__inputs.npy")
    assert (target / "train" / "all__inputs.npy").is_file()
    assert caplog.records[-1].getMessage() == (
        f"ensure_arc_dataset: building {target} (missing train/all__inputs.npy)"
    )
    for source_file in source_prefix.parent.iterdir():
        source_file.unlink()
    fetch(rel_path="train/all__labels.npy", dest=target / "train" / "all__labels.npy")


def test_arc_build_without_local_prefix_uses_clone_path_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "upstream" / "arc"
    calls: list[tuple[str, Path, ArcAugmentation]] = []

    class SourceContext:
        def __enter__(self) -> Path:
            return source

        def __exit__(self, *exc: object) -> None:
            del exc

    def build_from_source(
        *,
        input_file_prefix: str,
        output_dir: Path,
        augmentation: ArcAugmentation,
    ) -> None:
        calls.append((input_file_prefix, output_dir, augmentation))

    monkeypatch.setattr(build_dataset, "KaggleSource", SourceContext)
    monkeypatch.setattr(build_dataset, "_build_arc_dataset", build_from_source)
    target = tmp_path / "dataset"
    augmentation = ArcAugmentation.Config(spec=ArcSpec()).make()
    fetch = _ArcBuild(
        target_dir=target,
        augmentation=augmentation,
        input_file_prefix=None,
    )

    destination = target / "train" / "all__inputs.npy"
    fetch(rel_path="train/all__inputs.npy", dest=destination)
    fetch(rel_path="train/all__labels.npy", dest=target / "train" / "all__labels.npy")

    assert len(calls) == 1
    prefix, output_dir, built_augmentation = calls[0]
    assert prefix == str(source)
    assert output_dir == target
    assert built_augmentation is augmentation
    assert fetch._built


def test_failed_build_does_not_publish(tmp_path: Path) -> None:
    target = tmp_path / "dataset"
    with pytest.raises(FileNotFoundError):
        build_arc_dataset(
            target_dir=target,
            input_file_prefix=str(tmp_path / "absent"),
            augmentation=ArcAugmentation.Config(spec=ArcSpec()).make(),
        )
    assert not target.exists()
    assert list(tmp_path.iterdir()) == []


def _array(path: Path) -> NDArray[np.int64]:
    return _load_array(path).astype(np.int64)


def _ints(values: NDArray[np.int64]) -> list[int]:
    return [int(value) for value in values.flat]


def test_build_rejects_scale_policy_that_cannot_scale(
    tmp_path: Path,
    source_prefix: Path,
) -> None:
    with pytest.raises(ValueError, match=r"scale_prob=0.5 > 0"):
        build(
            dataset_dir=str(tmp_path / "dataset"),
            input_file_prefix=str(source_prefix),
            translation_prob=0,
            scale_prob=0.5,
            train_scale_weights={1: 1.0},
            num_aug=0,
        )


def test_build_defaults_to_slugged_target_when_destination_is_none(
    tmp_path: Path,
    source_prefix: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "default-dataset"
    calls: list[tuple[Path, ArcAugmentation, str | None]] = []

    def default_dataset_dir(**kwargs: object) -> Path:
        assert kwargs == {
            "translation_prob": 0.1,
            "scale_prob": 0.1,
            "train_scale_weights": {2: 1.0},
            "num_aug": 1_000,
            "seed": 42,
            "base_dir": "/opt/scratch",
        }
        return target

    def ensure_dataset(
        *,
        target_dir: Path,
        augmentation: ArcAugmentation,
        input_file_prefix: str | None,
    ) -> None:
        calls.append((target_dir, augmentation, input_file_prefix))

    monkeypatch.setattr(
        "priml.baselines.arcagi1.scripts.build_dataset.aug_policy_dataset_dir",
        default_dataset_dir,
    )
    monkeypatch.setattr(
        "priml.baselines.arcagi1.scripts.build_dataset.ensure_arc_dataset",
        ensure_dataset,
    )
    build(input_file_prefix=str(source_prefix))
    assert len(calls) == 1
    built_target, augmentation, built_prefix = calls[0]
    assert built_target == target
    assert built_prefix == str(source_prefix)
    assert augmentation.config.num_aug == 1_000
    assert augmentation.config.seed == 42
    assert augmentation.config.spatial.translation_prob == 0.1
    assert augmentation.config.spatial.scale_prob == 0.1


def test_build_arc_dataset_stages_in_target_parent_with_fixed_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "nested" / "parent" / "dataset"
    temporary_roots: list[Path] = []
    staging_paths: list[Path] = []
    original_temporary_directory = tempfile.TemporaryDirectory

    def temporary_directory(
        *,
        dir: Path,
        prefix: str,
    ) -> tempfile.TemporaryDirectory[str]:
        temporary = original_temporary_directory(dir=dir, prefix=prefix)
        temporary_roots.append(Path(temporary.name))
        return temporary

    def build_staging(
        *,
        input_file_prefix: str,
        output_dir: Path,
        augmentation: ArcAugmentation,
    ) -> None:
        del input_file_prefix, augmentation
        staging_paths.append(output_dir)
        output_dir.mkdir()

    monkeypatch.setattr(
        "priml.baselines.arcagi1.scripts.build_dataset.tempfile.TemporaryDirectory",
        temporary_directory,
    )
    monkeypatch.setattr(
        "priml.baselines.arcagi1.scripts.build_dataset._build_arc_dataset",
        build_staging,
    )
    build_arc_dataset(
        target_dir=target,
        input_file_prefix="unused",
        augmentation=ArcAugmentation.Config(spec=ArcSpec()).make(),
    )
    assert len(temporary_roots) == len(staging_paths) == 1
    assert temporary_roots[0].parent == target.parent
    assert temporary_roots[0].name.startswith(f".{target.name}-")
    assert staging_paths[0] == temporary_roots[0] / "dataset"
    assert target.is_dir()


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
