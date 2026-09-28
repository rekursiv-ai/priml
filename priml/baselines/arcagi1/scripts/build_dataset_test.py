"""Prepared ARC trees preserve the source recipe and existing destinations."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import json

import numpy as np
import pytest
import torch

from priml.baselines.arcagi1.augmentation import (
    ArcAugmentation,
    canonicalize_arc_grid,
)
from priml.baselines.arcagi1.scripts.build_dataset import build_arc_dataset
from priml.lib.custom_json import ListCodec, loads


if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray


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


def test_occupied_destination_is_unchanged(tmp_path: Path, source_prefix: Path) -> None:
    target = tmp_path / "dataset"
    target.mkdir()
    sentinel = target / "existing"
    sentinel.write_bytes(b"original")
    with pytest.raises(FileExistsError):
        build_arc_dataset(
            target_dir=target,
            input_file_prefix=str(source_prefix),
            augmentation=ArcAugmentation.Config().make(),
        )
    assert {path.name: path.read_bytes() for path in target.iterdir()} == {
        "existing": b"original",
    }


def test_spatial_eval_reuses_puzzle_id_and_canonical_answer(
    tmp_path: Path,
    source_prefix: Path,
) -> None:
    """Each spatial view shares its puzzle's id and restores to the same grids."""
    config = ArcAugmentation.Config()
    config.num_aug = 0
    config.spatial_eval_views = True
    target = tmp_path / "dataset"
    build_arc_dataset(
        target_dir=target,
        input_file_prefix=str(source_prefix),
        augmentation=config.make(),
    )
    test_dir = target / "test"
    inputs = _array(test_dir / "all__inputs.npy")
    labels = _array(test_dir / "all__labels.npy")
    tags = _array(test_dir / "all__spatial_tags.npy")
    indices = _array(test_dir / "all__puzzle_indices.npy")
    ids = _array(test_dir / "all__puzzle_identifiers.npy")
    names = ListCodec.coerce(loads((target / "identifiers.json").read_text()), str)
    assert len(ids) == len(tags) == 6
    assert _ints(indices) == list(range(7))
    assert not (target / "train" / "all__spatial_tags.npy").exists()
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
                )
                for row in (puzzle, puzzle + 1)
            ]
            assert canonical[0][0] == canonical[1][0]
            assert torch.equal(canonical[0][1], canonical[1][1])


def test_plain_build_writes_no_spatial_tags(
    tmp_path: Path,
    source_prefix: Path,
) -> None:
    """Without spatial views the tree keeps the reference file set."""
    config = ArcAugmentation.Config()
    config.num_aug = 0
    target = tmp_path / "dataset"
    build_arc_dataset(
        target_dir=target,
        input_file_prefix=str(source_prefix),
        augmentation=config.make(),
    )
    assert not list(target.glob("*/all__spatial_tags.npy"))


def test_failed_build_does_not_publish(tmp_path: Path) -> None:
    target = tmp_path / "dataset"
    with pytest.raises(FileNotFoundError):
        build_arc_dataset(
            target_dir=target,
            input_file_prefix=str(tmp_path / "absent"),
            augmentation=ArcAugmentation.Config().make(),
        )
    assert not target.exists()
    assert list(tmp_path.iterdir()) == []


def _array(path: Path) -> NDArray[np.int64]:
    return cast("NDArray[np.int64]", np.load(path)).astype(np.int64)


def _ints(values: NDArray[np.int64]) -> list[int]:
    return ListCodec.coerce(cast(object, values.tolist()), int)


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
