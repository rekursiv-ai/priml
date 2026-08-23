"""Tests for ARC data loading."""

from __future__ import annotations

from pathlib import Path

import json

from torch import Tensor

import numpy as np
import pytest
import torch

from priml.baselines.arcagi1.data import ArcData


TASKS = 4
PUZZLES_PER_TASK = 2
VIEWS_PER_PUZZLE = 3
GRID = 12


@pytest.fixture
def dataset_dir(tmp_path: Path) -> Path:
    """Write a tiny dataset in the three-level on-disk layout."""
    for split in ("train", "test"):
        directory = tmp_path / split
        directory.mkdir()
        puzzles = TASKS * PUZZLES_PER_TASK
        rows = puzzles * VIEWS_PER_PUZZLE
        # Each row is a distinct constant grid, so a row is identifiable.
        inputs = np.tile(
            np.arange(rows, dtype=np.int32).reshape(rows, 1) % 10 + 2,
            (1, GRID),
        )
        np.save(directory / "all__inputs.npy", inputs)
        np.save(directory / "all__labels.npy", inputs)
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
            np.arange(puzzles, dtype=np.int32),
        )
        (directory / "dataset.json").write_text(
            json.dumps({"vocab_size": 12, "seq_len": GRID, "ignore_label_id": 0}),
        )
    return tmp_path


def _data(dataset_dir: Path, **overrides: object) -> ArcData:
    config = ArcData.Config()
    config.base_dir = "/"
    config.working_dir = str(dataset_dir)
    config.device = "cpu"
    config.batch_size = 4
    for name, value in overrides.items():
        setattr(config, name, value)
    return config.make()


def test_batches_carry_the_puzzle_identity(dataset_dir: Path) -> None:
    """The per-task prefix and the metric both key on it."""
    batch = next(iter(_data(dataset_dir).train_dataloader()))
    assert batch["puzzle_identifiers"].shape == (4,)
    assert batch["media"].shape == (4, GRID)


def test_training_draws_whole_tasks(dataset_dir: Path) -> None:
    """Views of one puzzle arrive together, so a batch votes coherently.

    Sampling rows uniformly would over-weight tasks with more puzzles; the
    benchmark weights every task equally.
    """
    batch = next(iter(_data(dataset_dir).train_dataloader()))
    puzzle_identifiers_raw = batch["puzzle_identifiers"]
    assert isinstance(puzzle_identifiers_raw, Tensor)
    identifiers = puzzle_identifiers_raw.tolist()
    # 4 slots at 3 views per puzzle: at most two puzzles can appear.
    assert len(set(identifiers)) <= 2


def test_eval_walks_every_row_in_order(dataset_dir: Path) -> None:
    """pass@K votes across views, so evaluation must not sample."""
    rows = list(_data(dataset_dir).eval_dataloader())
    seen = torch.cat([b["media"][: b["valid_count"], 0] for b in rows])
    total = TASKS * PUZZLES_PER_TASK * VIEWS_PER_PUZZLE
    assert seen.shape == (total,)
    # Twice through gives the same order.
    again = torch.cat(
        [
            b["media"][: b["valid_count"], 0]
            for b in _data(dataset_dir).eval_dataloader()
        ]
    )
    assert torch.equal(seen, again)


def test_short_final_batch_is_padded(dataset_dir: Path) -> None:
    """Shapes stay constant, and the padding is reported not hidden."""
    batches = list(_data(dataset_dir, batch_size=7).eval_dataloader())
    assert all(b["media"].shape == (7, GRID) for b in batches)
    assert batches[-1]["valid_count"] < 7
    tail = batches[-1]
    assert bool((tail["label"][tail["valid_count"] :] == -100).all())


def test_the_skipped_cell_marker_is_remapped(dataset_dir: Path) -> None:
    """The loss and the halt target both key on -100, so remap once here."""
    batch = next(iter(_data(dataset_dir).eval_dataloader()))
    assert int(batch["label"].min()) >= 2  # nothing was 0 to remap here
    # A padded row carries the marker.
    padded = list(_data(dataset_dir, batch_size=7).eval_dataloader())[-1]
    assert -100 in padded["label"].tolist()[-1]


def test_sampling_is_reproducible_and_advances(dataset_dir: Path) -> None:
    """One seed replays; consecutive passes differ."""
    first = next(iter(_data(dataset_dir, seed=5).train_dataloader()))["media"]
    again = next(iter(_data(dataset_dir, seed=5).train_dataloader()))["media"]
    assert torch.equal(first, again)

    data = _data(dataset_dir, seed=5)
    loader = data.train_dataloader()
    pass_one = next(iter(loader))["media"].clone()
    pass_two = next(iter(loader))["media"].clone()
    assert not torch.equal(pass_one, pass_two)


def test_pass_counter_round_trips(dataset_dir: Path) -> None:
    """Resume continues the sampling sequence rather than replaying it."""
    data = _data(dataset_dir, seed=1)
    loader = data.train_dataloader()
    list(loader)
    state = data.state_dict()
    assert state["passes"] == 1

    restored = _data(dataset_dir, seed=1)
    restored.load_state_dict(state)
    assert torch.equal(
        next(iter(restored.train_dataloader()))["media"],
        next(iter(loader))["media"],
    )


def test_task_cap_trims_whole_tasks(dataset_dir: Path) -> None:
    data = _data(dataset_dir, num_eval_tasks=2)
    rows = sum(b["valid_count"] for b in data.eval_dataloader())
    assert rows == 2 * PUZZLES_PER_TASK * VIEWS_PER_PUZZLE


def test_missing_data_names_the_preparer(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="prepare_data"):
        _data(tmp_path).train_dataloader()


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
