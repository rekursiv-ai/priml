"""Tests for the Maze-Hard preparer.

Hermetic: every test feeds local CSV text or a stand-in for the hub download, so
nothing here touches the network.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import hashlib

from treekle import json_loads

import numpy as np
import pytest

from priml.baselines.maze.data import (
    CANONICAL,
    CHARSET,
    DIHEDRAL_VIEWS,
    ORIGINAL,
    canonical_labels,
    canonical_path_label,
    tokenize,
)
from priml.baselines.maze.scripts import prepare_data
from priml.data.arrays import load_array
from priml.lib.testing.checks import assert_help_without_docstring


if TYPE_CHECKING:
    from pathlib import Path


MAZES: Final = ("  S   G  ", "S#   #  G")
"""Two 3x3 mazes, row-major; the first is not symmetric under a quarter turn."""


@pytest.fixture
def csv_dir(tmp_path: Path) -> Path:
    """Write train and test CSVs holding both mazes, labeled by the stored rule."""
    rows = ["source,question,answer,rating"]
    for maze in MAZES:
        label = canonical_path_label(tokenize(maze, side=3))
        answer = "".join(CHARSET[token - 1] for token in label.reshape(-1).tolist())
        rows.append(f"synthetic,{maze},{answer},0")
    directory = tmp_path / "csv"
    directory.mkdir()
    for split in ("train", "test"):
        (directory / f"{split}.csv").write_text("\n".join(rows) + "\n")
    return directory


def test_training_mazes_take_eight_views_and_test_mazes_one(
    tmp_path: Path,
    csv_dir: Path,
) -> None:
    out = prepare_data.prepare(tmp_path / "maze", csv_directory=csv_dir)
    for name in (ORIGINAL, CANONICAL):
        for split, views in (("train", 8), ("test", 1)):
            directory = out / name / split
            assert load_array(directory / "all__inputs.npy", np.integer).shape == (
                2 * views,
                9,
            )
            np.testing.assert_array_equal(
                load_array(directory / "all__group_indices.npy", np.integer),
                [0, views, 2 * views],
            )
            marker = json_loads(
                (directory / "dataset.json").read_text(),
                dict[str, object],
            )
            assert (marker["seq_len"], marker["vocab_size"]) == (9, 6)


def test_the_original_tree_moves_each_stored_label_with_its_view(
    tmp_path: Path,
    csv_dir: Path,
) -> None:
    out = prepare_data.prepare(tmp_path / "maze", csv_directory=csv_dir)
    train = out / ORIGINAL / "train"
    label = canonical_path_label(tokenize(MAZES[0], side=3))
    expected = [view(label).reshape(-1) for view in DIHEDRAL_VIEWS]
    np.testing.assert_array_equal(
        load_array(train / "all__labels.npy", np.integer)[:8],
        expected,
    )


def test_the_canonical_tree_relabels_training_views_only(
    tmp_path: Path,
    csv_dir: Path,
) -> None:
    """The test split stays as given, so every recipe is scored on the same grids."""
    out = prepare_data.prepare(tmp_path / "maze", csv_directory=csv_dir)
    original, canonical = out / ORIGINAL, out / CANONICAL
    inputs = load_array(canonical / "train" / "all__inputs.npy", np.uint8)
    relabeled = load_array(canonical / "train" / "all__labels.npy", np.integer)
    np.testing.assert_array_equal(relabeled, canonical_labels(inputs))
    assert not np.array_equal(
        relabeled,
        load_array(original / "train" / "all__labels.npy", np.integer),
    )
    for array in ("all__inputs.npy", "all__labels.npy", "all__group_indices.npy"):
        assert (original / "test" / array).read_bytes() == (
            canonical / "test" / array
        ).read_bytes()


def test_rerunning_leaves_a_prepared_split_alone(tmp_path: Path, csv_dir: Path) -> None:
    out = prepare_data.prepare(tmp_path / "maze", csv_directory=csv_dir)
    labels = out / CANONICAL / "train" / "all__labels.npy"
    before = labels.stat().st_mtime_ns
    prepare_data.prepare(tmp_path / "maze", csv_directory=csv_dir)
    assert labels.stat().st_mtime_ns == before


def test_a_tree_built_from_another_source_is_refused(
    tmp_path: Path,
    csv_dir: Path,
) -> None:
    """Local CSVs are not the pinned revision, so the pinned build will not reuse them."""
    prepare_data.prepare(tmp_path / "maze", csv_directory=csv_dir)
    with pytest.raises(ValueError, match="delete it to rebuild"):
        prepare_data.prepare(tmp_path / "maze")


def test_a_malformed_row_is_reported_by_line(tmp_path: Path, csv_dir: Path) -> None:
    train = csv_dir / "train.csv"
    train.write_text(train.read_text() + "synthetic,S\n")
    with pytest.raises(ValueError, match=r"train\.csv:4: expected 4 fields"):
        prepare_data.prepare(tmp_path / "maze", csv_directory=csv_dir)


def test_the_pinned_csvs_are_fetched_and_verified(
    tmp_path: Path,
    csv_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str, str, str]] = []

    def fetch(repo: str, filename: str, *, repo_type: str, revision: str) -> str:
        calls.append((repo, filename, repo_type, revision))
        return str(csv_dir / filename)

    monkeypatch.setattr(prepare_data, "hf_hub_download", fetch)
    for filename in ("train.csv", "test.csv"):
        digest = hashlib.sha256((csv_dir / filename).read_bytes()).hexdigest()
        monkeypatch.setitem(prepare_data.SOURCE_SHA256, filename, digest)
    prepare_data.prepare(tmp_path / "maze")
    pinned = (prepare_data.SOURCE_REPO, "dataset", prepare_data.SOURCE_REVISION)
    assert calls == [
        (pinned[0], "train.csv", *pinned[1:]),
        (pinned[0], "test.csv", *pinned[1:]),
    ]


def test_a_fetched_csv_with_the_wrong_digest_is_rejected(
    tmp_path: Path,
    csv_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fetch(repo: str, filename: str, *, repo_type: str, revision: str) -> str:
        del repo, repo_type, revision
        return str(csv_dir / filename)

    monkeypatch.setattr(prepare_data, "hf_hub_download", fetch)
    with pytest.raises(RuntimeError, match="the cached copy is corrupt"):
        prepare_data.prepare(tmp_path / "maze")
    assert not (tmp_path / "maze" / ORIGINAL).exists()


def test_main_help_works_without_a_module_docstring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert_help_without_docstring(monkeypatch, prepare_data, prepare_data.main)


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
