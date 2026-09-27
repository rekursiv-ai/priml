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


def test_spatial_eval_reuses_puzzle_id_and_canonical_answer(
    tmp_path: Path,
    source_prefix: Path,
) -> None:
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
    inputs = cast("NDArray[np.int32]", np.load(test_dir / "all__inputs.npy"))
    labels = cast("NDArray[np.int32]", np.load(test_dir / "all__labels.npy"))
    tags = cast("NDArray[np.int16]", np.load(test_dir / "all__spatial_tags.npy"))
    indices = cast("NDArray[np.int32]", np.load(test_dir / "all__puzzle_indices.npy"))
    ids = cast("NDArray[np.int32]", np.load(test_dir / "all__puzzle_identifiers.npy"))
    names = cast("list[str]", json.loads((target / "identifiers.json").read_text()))
    assert len(ids) == 6
    assert indices.tolist() == list(range(7))
    for puzzle in range(0, len(ids), 2):
        assert ids[puzzle] == ids[puzzle + 1]
        assert tuple(cast(int, tags[puzzle, col]) for col in range(3)) == (1, 0, 0)
        assert tags[puzzle + 1, 0] == 2
        for rows in (inputs, labels):
            canonical = [
                canonicalize_arc_grid(
                    torch.from_numpy(cast("NDArray[np.int32]", rows[row])),
                    name=names[cast(int, ids[puzzle])],
                    spatial_tags=torch.from_numpy(
                        cast("NDArray[np.int16]", tags[row]),
                    ),
                )
                for row in (puzzle, puzzle + 1)
            ]
            assert canonical[0][0] == canonical[1][0]
            torch.testing.assert_close(canonical[0][1], canonical[1][1])


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
