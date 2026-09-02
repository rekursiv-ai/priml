"""The per-task table must be sized for the dataset it will be trained on.

``exp000`` names a real dataset, so the size of its ``SparsePuzzleEmbedding``
is a property of that dataset rather than a free parameter. These tests build
the experiment against a synthetic tree in the on-disk layout and drive a real
batch through the real prefix module -- the first place a table too small for
the identifiers stops being a config discrepancy and becomes an ``IndexError``.
"""

from __future__ import annotations

from pathlib import Path

import json

import numpy as np
import pytest
import torch

from priml.baselines.arcagi1 import experiments
from priml.baselines.arcagi1.data import ArcData
from priml.baselines.sudoku.prefix import PrefixStack, SparsePuzzleEmbedding


TASKS = 3
PUZZLES_PER_TASK = 2
VIEWS_PER_PUZZLE = 2
GRID = 900
NUM_PUZZLE_IDENTIFIERS = TASKS * PUZZLES_PER_TASK + 1
"""Ids the synthetic build assigns: one per puzzle, plus 0 for the blank."""


@pytest.fixture
def dataset_dir(tmp_path: Path) -> Path:
    """Write a tiny ARC tree whose ``num_puzzle_identifiers`` exceeds one.

    Identifiers start at 1 exactly as ``download_data`` assigns them (0 is the
    reserved blank), so the largest id equals the puzzle count -- which is what
    makes a one-row table index out of bounds rather than merely wrong.
    """
    puzzles = TASKS * PUZZLES_PER_TASK
    for split in ("train", "test"):
        directory = tmp_path / split
        directory.mkdir()
        rows = puzzles * VIEWS_PER_PUZZLE
        grids = np.full((rows, GRID), 2, dtype=np.int32)
        np.save(directory / "all__inputs.npy", grids)
        np.save(directory / "all__labels.npy", grids)
        np.save(
            directory / "all__puzzle_indices.npy",
            np.arange(puzzles + 1, dtype=np.int32) * VIEWS_PER_PUZZLE,
        )
        np.save(
            directory / "all__group_indices.npy",
            np.arange(TASKS + 1, dtype=np.int32) * PUZZLES_PER_TASK,
        )
        np.save(
            directory / "all__puzzle_identifiers.npy",
            np.arange(1, puzzles + 1, dtype=np.int32),
        )
        (directory / "dataset.json").write_text(
            json.dumps(
                {
                    "vocab_size": 12,
                    "seq_len": GRID,
                    "ignore_label_id": 0,
                    "num_puzzle_identifiers": NUM_PUZZLE_IDENTIFIERS,
                },
            ),
        )
    return tmp_path


def _table(config: experiments.ArcTrainLoop) -> SparsePuzzleEmbedding.Config:
    """The per-task embedding config from a finalized experiment."""
    prefix = config.step.model.prefix
    assert isinstance(prefix, PrefixStack.Config)
    table = prefix.parts[0]
    assert isinstance(table, SparsePuzzleEmbedding.Config)
    return table


def test_exp000_table_covers_the_datasets_identifiers() -> None:
    """One row per identifier the dataset can emit, or a lookup runs off the end.

    ``exp000`` names ``arc1concept-aug-1000``, whose build assigns 876,403 ids.
    A table shorter than that is not a smaller model -- it is a crash on the
    first batch, which is why this is pinned rather than left to the default.
    """
    config = experiments.exp000().copy_tree().finalize()
    assert _table(config).num_puzzles == experiments.NUM_PUZZLE_IDENTIFIERS


def test_a_real_batch_indexes_the_table_in_bounds(dataset_dir: Path) -> None:
    """The end-to-end contract: dataset ids must index the built table.

    Config equality alone would pass on two numbers that are wrong together.
    This drives the real loader into the real module, so the assertion is that
    a lookup SUCCEEDS -- the exact step that raised ``IndexError: index 876402
    is out of bounds for dimension 0 with size 1``.
    """
    dataset = ArcData.Config()
    dataset.base_dir = "/"
    dataset.working_dir = str(dataset_dir)
    dataset.device = "cpu"
    dataset.batch_size = 4
    batch = next(iter(dataset.make().train_dataloader()))
    identifiers = batch["puzzle_identifiers"]
    assert isinstance(identifiers, torch.Tensor)
    assert int(identifiers.max()) > 0  # The tree really does exercise the table.

    # Only the SIZE axes shrink; ``num_puzzles`` is the field under test and
    # must arrive from the experiment exactly as a real run would build it.
    table = _table(experiments.exp000().copy_tree().finalize())
    table.batch_size = 4
    table.channels = 8
    table.channels_in = 8
    module = table.make()
    module.eval()

    prefix = module(4, puzzle_identifiers=identifiers)
    assert prefix.shape == (4, table.num_tokens, 8)


def test_smoke_keeps_the_full_table_despite_its_task_cap() -> None:
    """Capping TASKS does not cap the ids those tasks carry.

    The tempting shrink -- a table sized to ``num_tasks`` -- is wrong: ids are
    assigned once across the whole build, so a loaded task carries whatever id
    it was given there, not its position in a truncated prefix. Sizing down
    reintroduces the out-of-bounds lookup in the one experiment whose job is to
    prove the installation runs.
    """
    smoke = experiments.exp_smoke().copy_tree().finalize()
    assert smoke.dataset.num_tasks is not None
    assert _table(smoke).num_puzzles == experiments.NUM_PUZZLE_IDENTIFIERS


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
