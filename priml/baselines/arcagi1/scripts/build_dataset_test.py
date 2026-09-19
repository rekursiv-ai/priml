"""Prepared ARC trees preserve the source recipe and existing destinations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import json

import pytest

from priml.baselines.arcagi1.augmentation import ArcAugmentation
from priml.baselines.arcagi1.scripts.build_dataset import build_arc_dataset


if TYPE_CHECKING:
    from pathlib import Path


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


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
