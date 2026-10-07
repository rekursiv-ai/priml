"""Tests for spatial evaluation tree expansion."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast
from unittest.mock import Mock

import logging

import numpy as np
import pytest

from priml.baselines.arcagi1.augmentation import ArcSpec
from priml.baselines.arcagi1.scripts import build_spatial_eval
from priml.data.ensure import EnsureResult
from priml.lib.custom_json import ReadError, parse


if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from priml.data.ensure import DataSpec


def _load_int_array(path: Path) -> NDArray[np.int32] | NDArray[np.int64]:
    array = cast("object", np.load(path))
    assert isinstance(array, np.ndarray)
    if array.dtype == np.dtype(np.int32):
        return cast("NDArray[np.int32]", array)
    assert array.dtype == np.dtype(np.int64)
    return cast("NDArray[np.int64]", array)


def test_spatial_eval_slug_and_dataset_dir(tmp_path: Path) -> None:
    assert build_spatial_eval.spatial_eval_slug(spatial_views=3) == "spatialeval-v3"
    with pytest.raises(ValueError, match="spatial_views must be >= 1, got 0"):
        build_spatial_eval.spatial_eval_slug(spatial_views=0)
    assert build_spatial_eval.spatial_eval_slug(spatial_views=1) == "spatialeval-v1"
    assert (
        build_spatial_eval.spatial_eval_dataset_dir(
            spatial_views=2,
            source_name="source",
            base_dir=tmp_path,
            working_dir="data",
        )
        == tmp_path / "data/source-spatialeval-v2"
    )
    assert build_spatial_eval.spatial_eval_dataset_dir(spatial_views=2) == Path(
        "/datasets/arc1concept-aug-1000-spatialeval-v2",
    )


def test_content_shape_uses_top_left_color_extent() -> None:
    spec = ArcSpec(max_grid=4, vocab_color_offset=2)
    grid = np.array(
        [[2, 3, 0, 0], [0, 0, 0, 0], [2, 0, 0, 0], [0, 0, 0, 0]],
        dtype=np.int64,
    )
    assert build_spatial_eval._content_shape(grid.flatten(), spec=spec) == (3, 2)
    assert build_spatial_eval._content_shape(
        np.zeros(16, dtype=np.int64),
        spec=spec,
    ) == (
        0,
        0,
    )


def test_forward_spatial_scales_pads_and_marks_eos(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = ArcSpec(max_grid=5, vocab_pad=0, vocab_eos=1, vocab_color_offset=2)
    flat = np.array(
        [
            [2, 3, 0, 0, 0],
            [4, 5, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=np.int64,
    ).flatten()
    full = Mock(wraps=np.full)
    monkeypatch.setattr(np, "full", full)
    actual = build_spatial_eval._forward_spatial(
        flat,
        scale=2,
        pad_r=0,
        pad_c=0,
        spec=spec,
        # ARC grids use a square max_grid output buffer.
    ).reshape(5, 5)
    full.assert_called_once_with((5, 5), spec.vocab_pad, dtype=flat.dtype)
    assert actual.dtype == np.int64
    assert np.array_equal(
        actual,
        np.array(
            [
                [2, 2, 3, 3, 1],
                [2, 2, 3, 3, 1],
                [4, 4, 5, 5, 1],
                [4, 4, 5, 5, 1],
                [1, 1, 1, 1, 0],
            ],
            dtype=np.int64,
        ),
    )


def test_sample_spatial_rejects_empty_and_respects_fit() -> None:
    spec = ArcSpec(max_grid=4, vocab_color_offset=2)
    rng = np.random.default_rng(7)
    empty = np.zeros(16, dtype=np.int64)
    assert build_spatial_eval._sample_spatial([empty], {2: 1.0}, rng, spec=spec) is None
    grid = np.array(
        [[2, 3, 0, 0], [4, 5, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        dtype=np.int64,
    ).flatten()
    assert build_spatial_eval._sample_spatial([], {2: 1.0}, rng, spec=spec) is None
    assert build_spatial_eval._sample_spatial([grid], {2: 1.0}, rng, spec=spec) == (
        2,
        0,
        0,
    )
    weighted = build_spatial_eval._sample_spatial(
        [grid],
        {1: 0.01, 2: 0.99},
        np.random.default_rng(2),
        spec=spec,
    )
    assert weighted == (2, 0, 0)
    assert build_spatial_eval._sample_spatial(
        [grid],
        {1: 1.0, 2: 9.0},
        np.random.default_rng(0),
        spec=spec,
    ) == (2, 0, 0)
    padded_spec = ArcSpec(max_grid=5, vocab_color_offset=2)
    padded_grid = np.array(
        [[2, 3, 0, 0, 0], [4, 5, 0, 0, 0], [0] * 5, [0] * 5, [0] * 5],
        dtype=np.int64,
    ).flatten()
    padded = build_spatial_eval._sample_spatial(
        [padded_grid],
        {1: 1.0},
        np.random.default_rng(10),
        spec=padded_spec,
    )
    assert padded == (1, 1, 0)
    fallback = build_spatial_eval._sample_spatial([grid], {3: 1.0}, rng, spec=spec)
    assert fallback is not None
    assert fallback[0] == 1
    assert 0 <= fallback[1] <= 2
    assert 0 <= fallback[2] <= 2
    full = np.full(16, 2, dtype=np.int64)
    assert build_spatial_eval._sample_spatial([full], {1: 1.0}, rng, spec=spec) is None
    too_tall = np.array(
        [[2, 3, 0, 0], [2, 3, 0, 0], [2, 3, 0, 0], [0, 0, 0, 0]],
        dtype=np.int64,
    ).flatten()
    too_tall_result = build_spatial_eval._sample_spatial(
        [too_tall],
        {2: 1.0},
        rng,
        spec=spec,
    )
    assert too_tall_result is not None
    assert too_tall_result[0] == 1
    too_wide = np.array(
        [[2, 2, 2, 2], [3, 3, 3, 3], [0, 0, 0, 0], [0, 0, 0, 0]],
        dtype=np.int64,
    ).flatten()
    too_wide_result = build_spatial_eval._sample_spatial(
        [too_wide],
        {2: 1.0},
        rng,
        spec=spec,
    )
    assert too_wide_result is None or too_wide_result[0] == 1


def test_build_spatial_eval_uses_default_arc_spec(tmp_path: Path) -> None:
    source = tmp_path / "source"
    test = source / "test"
    test.mkdir(parents=True)
    grid = np.zeros(30 * 30, dtype=np.int64)
    grid[0] = 2
    np.save(test / "all__inputs.npy", np.stack([grid]))
    np.save(test / "all__labels.npy", np.stack([grid]))
    np.save(test / "all__puzzle_indices.npy", np.array([0, 1], dtype=np.int32))
    np.save(test / "all__group_indices.npy", np.array([0, 1], dtype=np.int32))
    np.save(test / "all__puzzle_identifiers.npy", np.array([0], dtype=np.int32))
    (source / "identifiers.json").write_text('["puzzle"]')
    (source / "test_puzzles.json").write_text("{}")
    (test / "dataset.json").write_text('{"num_puzzle_identifiers": 1}')

    spec = ArcSpec()
    sampled = build_spatial_eval._sample_spatial(
        [grid, grid],
        {2: 1.0},
        np.random.default_rng(42),
        spec=spec,
    )
    assert sampled is not None
    scale, pad_r, pad_c = sampled
    expected_variant = build_spatial_eval._forward_spatial(
        grid,
        scale=scale,
        pad_r=pad_r,
        pad_c=pad_c,
        spec=spec,
    )
    target = tmp_path / "target"
    build_spatial_eval.build_spatial_eval(
        spatial_views=2,
        source_dir=source,
        target_dir=target,
        scale_weights={2: 1.0},
    )

    assert np.array_equal(
        _load_int_array(target / "test/all__inputs.npy"),
        np.stack([grid, expected_variant]),
    )
    assert np.array_equal(
        _load_int_array(target / "test/all__spatial_tags.npy"),
        np.array([[1, 0, 0], sampled], dtype=np.int32),
    )


def test_build_spatial_eval_continues_after_an_unfittable_view(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    test = source / "test"
    test.mkdir(parents=True)
    spec = ArcSpec(max_grid=4, vocab_pad=0, vocab_eos=1, vocab_color_offset=2)
    grid = np.array(
        [[2, 3, 0, 0], [4, 5, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        dtype=np.int64,
    ).flatten()
    np.save(test / "all__inputs.npy", np.stack([grid]))
    np.save(test / "all__labels.npy", np.stack([grid]))
    np.save(test / "all__puzzle_indices.npy", np.array([0, 1], dtype=np.int32))
    np.save(test / "all__group_indices.npy", np.array([0, 1], dtype=np.int32))
    np.save(test / "all__puzzle_identifiers.npy", np.array([0], dtype=np.int32))
    (source / "identifiers.json").write_text('["puzzle"]')
    (source / "test_puzzles.json").write_text("{}")
    (test / "dataset.json").write_text('{"num_puzzle_identifiers": 1}')
    sampler = Mock(side_effect=[None, (1, 1, 1)])
    monkeypatch.setattr(build_spatial_eval, "_sample_spatial", sampler)

    target = tmp_path / "target"
    build_spatial_eval.build_spatial_eval(
        spatial_views=3,
        source_dir=source,
        target_dir=target,
        spec=spec,
    )

    assert np.array_equal(
        _load_int_array(target / "test/all__puzzle_indices.npy"),
        [0, 1, 2],
    )
    assert np.array_equal(
        _load_int_array(target / "test/all__spatial_tags.npy"),
        [[1, 0, 0], [1, 1, 1]],
    )


def test_build_spatial_eval_copies_metadata_and_expands_pairs(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger=build_spatial_eval.logger.name)
    source = tmp_path / "source"
    test = source / "test"
    train = source / "train"
    test.mkdir(parents=True)
    train.mkdir()
    spec = ArcSpec(max_grid=4, vocab_pad=0, vocab_eos=1, vocab_color_offset=2)
    grid = np.array(
        [[2, 3, 0, 0], [4, 5, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        dtype=np.int64,
    ).flatten()
    source_arrays = (
        ("all__inputs.npy", np.stack([grid, grid, grid, grid])),
        ("all__labels.npy", np.stack([grid, grid, grid, grid])),
        ("all__puzzle_indices.npy", np.array([0, 1, 3, 4], dtype=np.int32)),
        ("all__group_indices.npy", np.array([0, 2, 3], dtype=np.int32)),
        ("all__puzzle_identifiers.npy", np.array([1, 2, 1], dtype=np.int32)),
    )
    for name, array in source_arrays:
        np.save(test / name, array)
        np.save(train / name, array)
    (source / "identifiers.json").write_text('["<blank>", "first", "second"]')
    (source / "test_puzzles.json").write_text('{"first": {}, "second": {}}')
    for split in (test, train):
        (split / "dataset.json").write_text(
            '{"num_puzzle_identifiers": 9, "sentinel": 12}',
        )

    target = tmp_path / "target"
    build_spatial_eval.build_spatial_eval(
        spatial_views=2,
        source_dir=source,
        target_dir=target,
        scale_weights={2: 1.0},
        spec=spec,
    )

    out = target / "test"
    assert np.array_equal(
        _load_int_array(out / "all__puzzle_indices.npy"),
        [0, 1, 2, 4, 6, 7, 8],
    )
    assert np.array_equal(_load_int_array(out / "all__group_indices.npy"), [0, 4, 6])
    assert np.array_equal(
        _load_int_array(out / "all__puzzle_identifiers.npy"),
        [1, 1, 2, 2, 1, 1],
    )
    tags = _load_int_array(out / "all__spatial_tags.npy")
    assert tags.dtype == np.int32
    assert np.array_equal(tags[[0, 2, 4]], [[1, 0, 0]] * 3)
    assert np.array_equal(tags[[1, 3, 5], 0], [2] * 3)
    for name in (
        "all__puzzle_indices.npy",
        "all__group_indices.npy",
        "all__puzzle_identifiers.npy",
    ):
        assert _load_int_array(out / name).dtype == np.int32
    outputs = _load_int_array(out / "all__inputs.npy")
    labels = _load_int_array(out / "all__labels.npy")
    assert outputs.dtype == np.int64
    assert np.array_equal(outputs, labels)
    assert np.array_equal(outputs[2:4], np.stack([grid, grid]))
    assert (target / "identifiers.json").read_text() == '["<blank>", "first", "second"]'
    assert (target / "test_puzzles.json").read_text() == '{"first": {}, "second": {}}'
    assert (out / "dataset.json").read_text() == (
        '{"num_puzzle_identifiers": 3, "sentinel": 12, "total_puzzles": 6, '
        f'"mean_puzzle_examples": {8 / 6}}}'
    )
    output_train = target / "train"
    for name, _ in source_arrays:
        assert np.array_equal(
            _load_int_array(output_train / name),
            _load_int_array(train / name),
        )
    assert (output_train / "dataset.json").read_text() == (
        '{"num_puzzle_identifiers": 3, "sentinel": 12}'
    )
    assert caplog.records[-1].getMessage() == (
        f"spatial eval: 6 puzzles (from 3), 8 examples -> {target}"
    )
    build_spatial_eval.build_spatial_eval(
        spatial_views=2,
        source_dir=source,
        target_dir=target,
        scale_weights={2: 1.0},
        spec=spec,
    )
    seed_target = tmp_path / "seed-target"
    build_spatial_eval.build_spatial_eval(
        spatial_views=2,
        source_dir=source,
        target_dir=seed_target,
        scale_weights={1: 1.0},
        spec=spec,
    )
    assert np.array_equal(
        _load_int_array(seed_target / "test/all__spatial_tags.npy")[1::2],
        [[1, 1, 1], [1, 0, 2], [1, 1, 2]],
    )
    (source / "identifiers.json").write_text('["<blank>", 1, "second"]')
    identity_target = tmp_path / "identity-target"
    with pytest.raises(ReadError):
        build_spatial_eval.build_spatial_eval(
            spatial_views=1,
            source_dir=source,
            target_dir=identity_target,
            spec=spec,
        )


def _tiny_source(source: Path, *, spec: ArcSpec) -> None:
    """Write a two-puzzle, one-task source tree with a train split."""
    grid = np.zeros(spec.max_grid**2, dtype=np.int64)
    grid[:2] = spec.vocab_color_offset + 1
    for split in ("test", "train"):
        out = source / split
        out.mkdir(parents=True)
        np.save(out / "all__inputs.npy", np.stack([grid, grid, grid]))
        np.save(out / "all__labels.npy", np.stack([grid, grid, grid]))
        np.save(out / "all__puzzle_indices.npy", np.array([0, 1, 3], dtype=np.int32))
        np.save(out / "all__group_indices.npy", np.array([0, 2], dtype=np.int32))
        np.save(out / "all__puzzle_identifiers.npy", np.array([1, 1], dtype=np.int32))
        (out / "dataset.json").write_text(
            '{"num_puzzle_identifiers": 2, "total_puzzles": 2, '
            '"mean_puzzle_examples": 1.5, "total_groups": 1}',
        )
    (source / "identifiers.json").write_text('["<blank>", "task"]')
    (source / "test_puzzles.json").write_text('{"task": {}}')


def test_expanded_metadata_counts_the_expanded_puzzles(tmp_path: Path) -> None:
    spec = ArcSpec(max_grid=4)
    source = tmp_path / "source"
    _tiny_source(source, spec=spec)
    target = tmp_path / "target"
    build_spatial_eval.build_spatial_eval(
        spatial_views=2,
        source_dir=source,
        target_dir=target,
        scale_weights={2: 1.0},
        spec=spec,
    )
    meta = parse((target / "test" / "dataset.json").read_text(), dict[str, object])
    ids = _load_int_array(target / "test" / "all__puzzle_identifiers.npy")
    rows = len(_load_int_array(target / "test" / "all__inputs.npy"))
    assert meta["total_puzzles"] == len(ids) == 4
    assert meta["mean_puzzle_examples"] == rows / len(ids)
    assert meta["total_groups"] == 1


@pytest.mark.parametrize("alias", ["same", "symlink"])
def test_build_refuses_to_write_over_its_source(tmp_path: Path, alias: str) -> None:
    spec = ArcSpec(max_grid=4)
    source = tmp_path / "source"
    _tiny_source(source, spec=spec)
    target = source
    if alias == "symlink":
        target = tmp_path / "link"
        target.symlink_to(source)
    before = {p: p.read_bytes() for p in source.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="aliases protected input"):
        build_spatial_eval.build_spatial_eval(
            spatial_views=2,
            source_dir=source,
            target_dir=target,
            scale_weights={2: 1.0},
            spec=spec,
        )
    assert {p: p.read_bytes() for p in source.rglob("*") if p.is_file()} == before


@pytest.mark.parametrize(
    "change",
    [{"seed": 1}, {"spec": ArcSpec(max_grid=4, vocab_eos=0, vocab_pad=1)}],
)
def test_ensure_refuses_a_tree_built_by_another_recipe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change: dict[str, object],
) -> None:
    """Seed and vocabulary change the bytes, so they are part of the identity."""
    spec = ArcSpec(max_grid=4)
    source = tmp_path / "source"
    _tiny_source(source, spec=spec)
    target = tmp_path / "target"
    first: dict[str, object] = {"seed": 0, "spec": spec}
    build_spatial_eval.ensure_spatial_eval_data(
        source_dir=source,
        target=target,
        spatial_views=2,
        scale_weights={2: 1.0},
        seed=0,
        spec=spec,
    )
    second = {**first, **change}
    for cold in (False, True):
        if cold:
            monkeypatch.setattr(build_spatial_eval, "_ensure_cache", set[object]())
        with pytest.raises(ValueError, match="was built by a different recipe"):
            build_spatial_eval.ensure_spatial_eval_data(
                source_dir=source,
                target=target,
                spatial_views=2,
                scale_weights={2: 1.0},
                seed=cast(int, second["seed"]),
                spec=cast(ArcSpec, second["spec"]),
            )


def test_build_spatial_eval_accepts_identity_only_and_rejects_zero(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="spatial_views must be >= 1, got 0"):
        build_spatial_eval.build_spatial_eval(
            spatial_views=0,
            source_dir=tmp_path / "missing",
            target_dir=tmp_path / "target",
        )


def test_copy_dataset_json_replaces_only_identifier_count(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    destination = tmp_path / "destination"
    destination.mkdir()
    (source / "dataset.json").write_text('{"num_puzzle_identifiers": 8, "other": true}')
    build_spatial_eval._copy_dataset_json(source, destination, num_ids=3)
    assert (
        destination / "dataset.json"
    ).read_text() == '{"num_puzzle_identifiers": 3, "other": true}'


def test_spatial_eval_fetch_builds_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    def capture_build(**kwargs: object) -> None:
        calls.append(kwargs)

    monkeypatch.setattr(build_spatial_eval, "build_spatial_eval", capture_build)
    fetch = build_spatial_eval._SpatialEvalBuild(
        source_dir=Path("source"),
        target_dir=Path("target"),
        spatial_views=2,
        scale_weights={2: 1.0},
        seed=9,
        spec=ArcSpec(max_grid=4),
    )
    assert fetch._built is False
    fetch(rel_path="one", dest=Path("unused"))
    fetch(rel_path="two", dest=Path("unused"))
    assert fetch._built is True
    assert calls == [
        {
            "spatial_views": 2,
            "source_dir": Path("source"),
            "target_dir": Path("target"),
            "scale_weights": {2: 1.0},
            "seed": 9,
            "spec": ArcSpec(max_grid=4),
        },
    ]


def test_ensure_spatial_eval_data_declares_split_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    (source / "train").mkdir(parents=True)
    target = tmp_path / "target"
    specs: list[DataSpec] = []
    rank_builds: list[str] = []

    def ensure_data(spec: DataSpec) -> EnsureResult:
        specs.append(spec)
        return EnsureResult.PRESENT

    def run_build(*, name: str, build: Callable[[], None]) -> None:
        rank_builds.append(name)
        build()

    monkeypatch.setattr(build_spatial_eval, "ensure_data", ensure_data)
    monkeypatch.setattr(build_spatial_eval, "run_rank_zero_build", run_build)
    result = build_spatial_eval.ensure_spatial_eval_data(
        source_dir=source,
        target=target,
        spatial_views=2,
        spec=ArcSpec(max_grid=4),
    )

    assert result == target
    assert (
        build_spatial_eval.ensure_spatial_eval_data(
            source_dir=source,
            target=target,
            spatial_views=2,
            spec=ArcSpec(max_grid=4),
        )
        == target
    )
    other_source = tmp_path / "other-source"
    with pytest.raises(ValueError, match="was built by a different recipe"):
        build_spatial_eval.ensure_spatial_eval_data(
            source_dir=other_source,
            target=target,
            spatial_views=2,
            spec=ArcSpec(max_grid=4),
        )
    other_target = tmp_path / "other-target"
    assert (
        build_spatial_eval.ensure_spatial_eval_data(
            source_dir=source,
            target=other_target,
            spatial_views=2,
            spec=ArcSpec(max_grid=4),
        )
        == other_target
    )
    next_target = tmp_path / "next-target"
    assert (
        build_spatial_eval.ensure_spatial_eval_data(
            source_dir=source,
            target=next_target,
            spatial_views=3,
            spec=ArcSpec(max_grid=4),
        )
        == next_target
    )
    # The refused recipe never reaches a build.
    assert rank_builds == ["ensure_spatial_eval_data"] * 3
    assert len(specs) == 3
    assert specs[0].target_dir == target
    assert isinstance(specs[0].fetch, build_spatial_eval._SpatialEvalBuild)
    assert specs[0].fetch._source_dir == source
    assert specs[0].fetch._target_dir == target
    assert specs[0].fetch._spatial_views == 2
    assert specs[0].fetch._scale_weights == {2: 1.0}
    assert specs[0].fetch._seed == 42
    assert specs[0].fetch._spec == ArcSpec(max_grid=4)
    default_source = tmp_path / "default-source"
    default_target = build_spatial_eval.spatial_eval_dataset_dir(
        spatial_views=1,
        source_name=default_source.name,
        base_dir="/opt/scratch",
    )
    assert (
        build_spatial_eval.ensure_spatial_eval_data(
            source_dir=default_source,
            spatial_views=1,
        )
        == default_target
    )
    assert specs[3].target_dir == default_target
    paths = [file.rel_path for file in specs[0].manifest]
    split_names = (
        "all__inputs.npy",
        "all__labels.npy",
        "all__puzzle_indices.npy",
        "all__group_indices.npy",
        "all__puzzle_identifiers.npy",
    )
    default_paths = [file.rel_path for file in specs[3].manifest]
    assert all(not path.startswith("train/") for path in default_paths)
    assert paths == [
        "identifiers.json",
        "test_puzzles.json",
        *(f"test/{name}" for name in split_names),
        "test/all__spatial_tags.npy",
        "test/dataset.json",
        *(f"train/{name}" for name in split_names),
        "train/dataset.json",
    ]
    assert specs[0].fetch is not None


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
