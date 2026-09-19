"""Hermetic tests for the ARC-AGI1 data preparer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import json

from priml.baselines.arcagi1 import experiments
from priml.baselines.arcagi1.data import ArcData
from priml.baselines.arcagi1.scripts.prepare_data import (
    num_puzzle_identifiers,
    prepare,
)
from priml.baselines.sudoku.prefix import PrefixStack, SparsePuzzleEmbedding


if TYPE_CHECKING:
    from pathlib import Path


def test_dataset_declares_its_augmentation_recipe(tmp_path: Path) -> None:
    config = ArcData.Config()
    config.working_dir = tmp_path / "absent"
    rendered = config.pformat(hide_default_values=False)
    assert "augmentation" in rendered
    assert "num_aug=1_000" in rendered
    assert "translation_prob=1.0" in rendered
    assert not (tmp_path / "absent").exists()


def test_prepare_data_loads_and_sizes_exp000(tmp_path: Path) -> None:
    """Build a tiny source tree, load it, and size exp000 from its metadata."""
    source = tmp_path / "source" / "arc"
    source.parent.mkdir()
    challenge = {
        "task": {
            "train": [{"input": [[0, 1], [1, 0]], "output": [[1, 0], [0, 1]]}],
            "test": [{"input": [[0, 0], [1, 1]], "output": [[1, 1], [0, 0]]}],
        },
    }
    for subset in ("training", "evaluation", "concept"):
        (source.parent / f"arc_{subset}_challenges.json").write_text(
            json.dumps(challenge),
        )
        (source.parent / f"arc_{subset}_solutions.json").write_text(
            json.dumps({"task": [[[1, 1], [0, 0]]]}),
        )

    recipe = ArcData.Config()
    recipe.working_dir = tmp_path / "dataset"
    recipe.augmentation.num_aug = 0
    directory = prepare(recipe, input_file_prefix=source)
    identifier_count = num_puzzle_identifiers(directory)
    assert identifier_count > 1
    assert (directory / "train" / "all__inputs.npy").is_file()
    assert (directory / "test" / "all__labels.npy").is_file()

    config = ArcData.Config()
    config.base_dir = "/"
    config.working_dir = str(directory)
    config.device = "cpu"
    config.batch_size = 1
    batch = next(iter(config.make().train_dataloader()))
    assert batch["puzzle_identifiers"] is not None

    experiment = experiments.exp000()
    experiment.dataset.base_dir = "/"
    experiment.dataset.working_dir = str(directory)
    prefix = experiment.step.model.prefix
    assert isinstance(prefix, PrefixStack.Config)
    table = prefix.parts[0]
    assert isinstance(table, SparsePuzzleEmbedding.Config)
    table.num_puzzles = identifier_count
    finalized = experiment.copy_tree().finalize()
    prefix = finalized.step.model.prefix
    assert isinstance(prefix, PrefixStack.Config)
    table = prefix.parts[0]
    assert isinstance(table, SparsePuzzleEmbedding.Config)
    assert table.num_puzzles == identifier_count


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
