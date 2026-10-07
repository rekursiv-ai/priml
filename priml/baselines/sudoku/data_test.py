"""Tests for sudoku data loading and augmentation."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import json

from torch import Tensor

import numpy as np
import pytest
import torch

from priml.baselines.arcagi1.augmentation import ColorDihedral
from priml.baselines.sudoku import data
from priml.baselines.sudoku.data import SudokuData, _load_split, augment_sudoku
from priml.baselines.sudoku.puzzle_data import PuzzleDataset
from priml.baselines.sudoku.puzzle_spec import SudokuSpec
from priml.lib.codec import from_plain
from priml.math.seed import salt
from priml.runtime import get_device
from priml.testing.golden import assert_tensor_golden, stored


_CWD: Final = Path(__file__).resolve().parent


@pytest.fixture
def dataset_dir(tmp_path: Path) -> Path:
    """Write a tiny two-split dataset in the on-disk layout."""
    for split, puzzles in (("train", 3), ("test", 2)):
        directory = tmp_path / split
        directory.mkdir()
        copies = 2
        rows = puzzles * copies
        # Token 1 is the empty cell; 2..10 are digits, so a row of 2s is valid.
        inputs = np.full((rows, 81), 2, dtype=np.int64)
        inputs[:, 0] = np.arange(rows) % 9 + 2  # `make` rows distinguishable.
        np.save(directory / "all__inputs.npy", inputs)
        np.save(directory / "all__labels.npy", inputs)
        np.save(
            directory / "all__group_indices.npy",
            np.arange(puzzles + 1, dtype=np.int64) * copies,
        )
        (directory / "dataset.json").write_text(
            json.dumps({"vocab_size": 11, "seq_len": 81}),
        )
    return tmp_path


def _data(dataset_dir: Path, **overrides: object) -> SudokuData:
    config = SudokuData.Config()
    config.base_dir = "/"
    config.working_dir = str(dataset_dir)
    config.device = "cpu"
    config.batch_size = 4
    for name, value in overrides.items():
        setattr(config, name, value)
    return config.make()


def test_batches_are_always_full_width(dataset_dir: Path) -> None:
    """A short final batch is padded and reports how many rows are real."""
    data = _data(dataset_dir, augment=False)
    batches = list(data.train_dataloader())
    assert all(b["media"].shape == (4, 81) for b in batches)
    # 6 rows at batch 4: one full batch, one half batch padded up.
    assert [b["valid_count"] for b in batches] == [4, 2]
    tail = batches[-1]
    assert bool((tail["media"][2:] == 0).all())
    assert bool((tail["label"][2:] == 0).all())
    assert sorted(
        row
        for batch in batches
        for row in from_plain(
            batch["media"][: batch["valid_count"], 0].tolist(),
            list[int],
        )
    ) == list(range(2, 8))
    assert torch.equal(tail["media"][:2], tail["label"][:2])


def test_split_loading_returns_tensors_and_logs_counts(
    dataset_dir: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("INFO", logger="priml.baselines.sudoku.data"):
        split = _load_split(dataset_dir, "train")

    assert split["inputs"].dtype == torch.int32
    assert split["labels"].dtype == torch.int32
    assert split["group_indices"].dtype == torch.int64
    assert split["inputs"].shape == (6, 81)
    assert split["labels"].shape == (6, 81)
    assert split["group_indices"].tolist() == [0, 2, 4, 6]
    assert split["vocab_size"] == 11
    assert [record.getMessage() for record in caplog.records] == [
        "loading sudoku split 'train' from " + str(dataset_dir / "train"),
        "sudoku 'train': 6 rows across 3 puzzles",
    ]


def test_eval_is_neither_shuffled_nor_augmented(dataset_dir: Path) -> None:
    """A score must not depend on which transformation was drawn."""
    data = _data(dataset_dir, augment=True, seed=0)
    first = next(iter(data.eval_dataloader()))
    second = next(iter(data.eval_dataloader()))
    assert torch.equal(first["media"], second["media"])
    assert first["media"][:, 0].tolist() == [2, 3, 4, 5]
    assert torch.equal(first["media"], first["label"])


def test_size_two_batches_are_valid(dataset_dir: Path) -> None:
    data = _data(dataset_dir, batch_size=2, eval_batch_size=2, augment=False)
    assert data.batch_size == 2
    assert data.eval_batch_size == 2
    assert [batch["valid_count"] for batch in data.train_dataloader()] == [2, 2, 2]


def test_batch_size_one_is_a_valid_configuration(dataset_dir: Path) -> None:
    assert _data(dataset_dir, batch_size=1).batch_size == 1
    assert _data(dataset_dir, eval_batch_size=1).eval_batch_size == 1


def test_shuffle_factories_receive_resolved_cpu_device(
    dataset_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[torch.device | None] = []
    resolved: list[torch.device | str | None] = []
    original_get_device = get_device
    original_arange = torch.arange
    original_randperm = torch.randperm

    def traced_get_device(
        device: torch.device | str | None = None,
    ) -> torch.device:
        resolved.append(device)
        return original_get_device(device)

    def traced_arange(
        start: int,
        end: int,
        *,
        device: torch.device | None = None,
    ) -> Tensor:
        observed.append(device)
        return original_arange(start, end, device=device)

    def traced_randperm(
        n: int,
        *,
        device: torch.device | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        observed.append(device)
        return original_randperm(n, device=device, generator=generator)

    monkeypatch.setattr(
        data,
        "get_device",
        traced_get_device,
    )
    monkeypatch.setattr(torch, "arange", traced_arange)
    monkeypatch.setattr(torch, "randperm", traced_randperm)
    list(_data(dataset_dir, augment=False, seed=0).train_dataloader())

    assert resolved == ["cpu"]
    assert observed
    assert all(device == torch.device("cpu") for device in observed)


def test_named_generators_include_seed_epoch_and_batch(
    dataset_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_devices: list[torch.device | str | None] = []
    original_generator = torch.Generator

    def traced_generator(
        device: torch.device | str | None = None,
    ) -> torch.Generator:
        observed_devices.append(device)
        return original_generator(device=device)

    monkeypatch.setattr(torch, "Generator", traced_generator)
    loader = _data(
        dataset_dir,
        seed=23,
        augment_seed=29,
        augment=False,
    ).train_dataloader()
    shuffle_generator = loader._shuffle_generator(5)
    augmentation_generator = loader._augmentation_generator(5, 2)
    assert shuffle_generator is not None
    assert augmentation_generator is not None
    assert observed_devices == [torch.device("cpu"), torch.device("cpu")]
    assert shuffle_generator.initial_seed() == salt("sudoku_shuffle", 23, 5)
    assert augmentation_generator.initial_seed() == salt(
        "sudoku_augmentation",
        29,
        5,
        2,
    )


def test_loaders_preserve_device_and_eval_recipe(dataset_dir: Path) -> None:
    data = _data(
        dataset_dir,
        device="meta",
        seed=7,
        augment=True,
        augment_seed=13,
        num_eval_puzzles=1,
    )
    train_loader = data.train_dataloader()
    assert train_loader.inputs.device == torch.device("meta")
    assert train_loader.labels.device == torch.device("meta")
    assert train_loader.bounds.device == torch.device("meta")

    eval_loader = data.eval_dataloader()
    assert eval_loader.inputs.device == torch.device("meta")
    assert eval_loader.labels.device == torch.device("meta")
    assert eval_loader.bounds.device == torch.device("meta")
    assert eval_loader.batch_size == data.eval_batch_size
    assert eval_loader.shuffle is False
    assert eval_loader.seed == 7
    assert eval_loader.epoch == 0
    assert eval_loader.augment is False
    assert eval_loader.augment_seed is None


def test_seeded_shuffle_is_reproducible_and_epoch_varying(
    dataset_dir: Path,
) -> None:
    """One seed fixes each epoch's order, and consecutive epochs differ."""
    orders: list[list[Tensor]] = []
    for _ in range(2):
        data = _data(dataset_dir, augment=False, seed=7)
        loader = data.train_dataloader()
        orders.append([b["media"].clone() for b in loader])
    assert torch.equal(orders[0][0], orders[1][0])
    assert orders[0][0][:, 0].tolist() == [3, 2, 6, 4]

    data = _data(dataset_dir, augment=False, seed=7)
    loader = data.train_dataloader()
    epoch_one = next(iter(loader))["media"].clone()
    epoch_two = next(iter(loader))["media"].clone()
    assert not torch.equal(epoch_one, epoch_two)


def test_missing_loader_state_fields_keep_their_current_values(
    dataset_dir: Path,
) -> None:
    loader = _data(dataset_dir, augment=False).train_dataloader()
    initial = loader.state_dict()
    loader.load_state_dict({})
    assert loader.state_dict() == initial


def test_epoch_counter_round_trips(dataset_dir: Path) -> None:
    """Resume continues the shuffle sequence rather than replaying epoch 0."""
    data = _data(dataset_dir, augment=False, seed=1)
    initial = data.state_dict()
    assert "timer_epoch" in initial
    assert initial["epoch"] == 0
    assert initial["loader"] is None
    assert initial["timer_epoch"] == {"global_count": 0, "global_sec": 0.0}
    loader = data.train_dataloader()
    assert len(loader) == 2
    assert loader.state_dict() == {"epoch": 0, "active_epoch": None, "next_batch": 0}
    list(loader)
    assert loader.state_dict() == {"epoch": 1, "active_epoch": None, "next_batch": 0}
    assert data.state_dict()["epoch"] == 1
    data.timer_epoch.global_count += 1
    state = data.state_dict()

    restored = _data(dataset_dir, augment=False, seed=1)
    restored.load_state_dict(state)
    assert restored.timer_epoch.global_count == 1
    assert torch.equal(
        next(iter(restored.train_dataloader()))["media"],
        next(iter(loader))["media"],
    )


def test_checkpoint_resumes_the_unfinished_epoch(dataset_dir: Path) -> None:
    data = _data(dataset_dir, augment=True, seed=3, augment_seed=7)
    loader = data.train_dataloader()
    iterator = iter(loader)
    first_batch = next(iterator)
    assert first_batch["media"][:, 0].tolist() == [3, 4, 8, 9]
    state = data.state_dict()
    expected = [batch["media"].clone() for batch in iterator]

    restored = _data(dataset_dir, augment=True, seed=3, augment_seed=7)
    restored.load_state_dict(state)
    observed = [batch["media"].clone() for batch in restored.train_dataloader()]

    assert len(observed) == len(expected)
    assert all(
        torch.equal(observed_batch, expected_batch)
        for observed_batch, expected_batch in zip(observed, expected, strict=True)
    )
    assert len(list(restored.train_dataloader())) == 2


def test_seed_and_epoch_are_distinct_named_stream_inputs(dataset_dir: Path) -> None:
    later_epoch = _data(dataset_dir, augment=False, seed=4).train_dataloader()
    list(later_epoch)
    later = [batch["media"].clone() for batch in later_epoch]
    first = [
        batch["media"].clone()
        for batch in _data(dataset_dir, augment=False, seed=5).train_dataloader()
    ]

    assert any(
        not torch.equal(later_batch, first_batch)
        for later_batch, first_batch in zip(later, first, strict=True)
    )


def test_augmentation_preserves_original_outputs_and_rng() -> None:
    """Preserve paired augmentation and the generator's subsequent draws."""
    grid = torch.arange(2 * 4).reshape(2, 4) % 11
    record: dict[str, Tensor] = {}
    for seed in (0, 3, 42):
        generator = torch.Generator().manual_seed(seed)
        inputs, labels = augment_sudoku(
            grid,
            grid.flip(0),
            spec=SudokuSpec(),
            generator=generator,
        )
        record[f"{seed}/inputs"] = stored(inputs)
        record[f"{seed}/labels"] = stored(labels)
        # The next draws pin the generator's position without its 5 KB state.
        record[f"{seed}/rng"] = torch.randint(
            0,
            2_147_483_647,
            (2,),
            generator=generator,
        )
    assert_tensor_golden(_CWD / "testdata" / "augmentation.pt", record)


def test_augmentation_uses_the_dataset_spec() -> None:
    spec = SudokuSpec(grid_shape=(4, 4), box_shape=(2, 2), vocab_size=6)
    grid = torch.arange(32).reshape(2, 16) % 4 + 2
    actual, labels = augment_sudoku(
        grid,
        grid.clone(),
        spec=spec,
        generator=torch.Generator().manual_seed(0),
    )
    assert actual.shape == (2, 16)
    assert torch.equal(actual, labels)
    assert set(from_plain(actual.flatten().tolist(), list[int])) == {2, 3, 4, 5}


def test_augmentation_moves_the_label_with_the_input() -> None:
    """A transformed puzzle must keep a correct solution, or it teaches noise."""
    torch.manual_seed(0)
    # A solved grid: input equals label, so the invariant is checkable directly.
    grid = torch.arange(162).reshape(2, 81) % 9 + 2
    inputs, labels = augment_sudoku(grid, grid.clone(), spec=SudokuSpec())
    assert torch.equal(inputs, labels)


def test_sudoku_augmentation_uses_all_dihedral_transforms() -> None:
    spec = SudokuSpec()
    grid = torch.arange(128 * 81).reshape(128, 81) % 9 + 2
    labels = grid.flip(0)
    generator = torch.Generator().manual_seed(19)
    actual = augment_sudoku(grid, labels, spec=spec, generator=generator)

    config = ColorDihedral.Config()
    config.colors = tuple(range(1, spec.vocab_size - 1))
    config.transforms = (0, 4, 1, 7, 2, 5, 3, 6)
    reference_generator = torch.Generator().manual_seed(19)
    expected = config.make().augment_tokens(
        grid,
        labels,
        vocab_size=spec.vocab_size,
        token_offset=1,
        generator=reference_generator,
    )

    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])


def test_augmentation_preserves_empties_and_padding() -> None:
    """Tokens 0 and 1 are not digits and must survive relabeling."""
    torch.manual_seed(0)
    grid = torch.full((2, 81), 1, dtype=torch.long)  # Every cell empty.
    grid[:, :5] = 0  # Padding.
    inputs, _ = augment_sudoku(grid, grid.clone(), spec=SudokuSpec())
    assert set(from_plain(inputs.flatten().tolist(), list[int])) <= {0, 1}


def test_augmentation_is_seedable() -> None:
    """A dedicated generator makes the stream independent of ambient draws."""
    grid = torch.arange(162).reshape(2, 81) % 9 + 2

    def once(disturb: bool) -> Tensor:
        generator = torch.Generator().manual_seed(3)
        if disturb:
            torch.rand(11)
        return augment_sudoku(
            grid,
            grid.clone(),
            spec=SudokuSpec(),
            generator=generator,
        )[0]

    assert torch.equal(once(disturb=False), once(disturb=True))


def test_seeded_augmentation_resumes_at_the_next_epoch(dataset_dir: Path) -> None:
    data = _data(dataset_dir, seed=1, augment=True, augment_seed=7)
    loader = data.train_dataloader()
    list(loader)
    data.timer_epoch.global_count += 1
    state = data.state_dict()
    expected = [batch["media"].clone() for batch in loader]

    restored = _data(dataset_dir, seed=1, augment=True, augment_seed=7)
    restored.load_state_dict(state)
    observed = [batch["media"].clone() for batch in restored.train_dataloader()]

    assert len(observed) == len(expected)
    assert all(
        torch.equal(observed_batch, expected_batch)
        for observed_batch, expected_batch in zip(observed, expected, strict=True)
    )


@pytest.mark.parametrize(
    "field",
    ["batch_size", "eval_batch_size", "num_train_puzzles", "num_eval_puzzles"],
)
@pytest.mark.parametrize("value", [0, -1])
def test_nonpositive_sizes_are_rejected_at_construction(
    dataset_dir: Path,
    field: str,
    value: int,
) -> None:
    with pytest.raises(ValueError, match="positive") as error:
        _data(dataset_dir, **{field: value})
    assert str(error.value) == f"{field} must be positive; got {value}."


def test_num_puzzles_keeps_a_prefix_of_whole_puzzles(dataset_dir: Path) -> None:
    """Two copies per puzzle, so two puzzles are the first four rows."""
    data = _data(dataset_dir, augment=False, num_eval_puzzles=1, eval_batch_size=4)
    eval_loader = data.eval_dataloader()
    assert eval_loader.inputs.shape == (2, 81)
    (only,) = list(eval_loader)
    assert only["valid_count"] == 2
    assert only["media"][:2, 0].tolist() == [2, 3]

    train = _data(dataset_dir, augment=False, num_train_puzzles=1)
    train_batches = list(train.train_dataloader())
    assert sum(batch["valid_count"] for batch in train_batches) == 2
    assert sorted(
        value
        for batch in train_batches
        for value in from_plain(
            batch["media"][: batch["valid_count"], 0].tolist(),
            list[int],
        )
    ) == [2, 3]

    beyond_split = _data(
        dataset_dir,
        augment=False,
        num_eval_puzzles=4,
    ).eval_dataloader()
    assert beyond_split.inputs.shape == (4, 81)


@pytest.mark.parametrize(
    ("spec", "match"),
    [
        (SudokuSpec(vocab_size=12), "vocabulary"),
        (SudokuSpec(grid_shape=(4, 4)), "grid"),
    ],
)
def test_prepared_data_must_match_the_spec(
    dataset_dir: Path,
    spec: SudokuSpec,
    match: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sudoku = _data(dataset_dir, spec=spec)
    placed: list[object] = []
    monkeypatch.setattr(data, "get_device", placed.append)
    with pytest.raises(ValueError, match="does not match") as error:
        sudoku.train_dataloader()
    # The split is rejected before any of it is copied to the device.
    assert placed == []
    expected_message = (
        "Prepared vocabulary does not match dataset spec."
        if match == "vocabulary"
        else "Prepared grid does not match dataset spec."
    )
    assert str(error.value) == expected_message


def test_a_negative_epoch_is_rejected(dataset_dir: Path) -> None:
    data = _data(dataset_dir)
    live = data.train_dataloader()
    live.epoch = -1
    with pytest.raises(ValueError, match="epoch must be non-negative"):
        data.train_dataloader()


def test_unseeded_augmentation_draws_from_the_ambient_stream(
    dataset_dir: Path,
) -> None:
    def first(seed: int) -> Tensor:
        torch.manual_seed(seed)
        return next(iter(_data(dataset_dir, seed=0).train_dataloader()))["media"]

    assert torch.equal(first(0), first(0))
    assert not torch.equal(first(0), first(1))


def test_a_new_loader_continues_the_live_loaders_epoch(dataset_dir: Path) -> None:
    data = _data(dataset_dir, augment=False, seed=2)
    list(data.train_dataloader())
    assert data.train_dataloader().epoch == 1


def test_restored_epoch_increments_from_its_checkpoint(dataset_dir: Path) -> None:
    data = _data(dataset_dir, augment=False, seed=2)
    data.load_state_dict(
        {
            "epoch": 5,
            "loader": {"epoch": 5, "active_epoch": None, "next_batch": 0},
        },
    )
    loader = data.train_dataloader()
    assert loader.epoch == 5
    list(loader)
    assert loader.epoch == 6


def test_restoring_state_into_a_live_loader(dataset_dir: Path) -> None:
    """A checkpoint loaded after the loader exists must still take effect."""
    source = _data(dataset_dir, augment=False, seed=6)
    iterator = iter(source.train_dataloader())
    next(iterator)
    mid_epoch = source.state_dict()
    expected = [batch["media"].clone() for batch in iterator]

    data = _data(dataset_dir, augment=False, seed=6)
    live = data.train_dataloader()
    data.load_state_dict(mid_epoch)
    observed = [batch["media"] for batch in live]

    assert len(observed) == len(expected)
    assert all(
        torch.equal(got, wanted) for got, wanted in zip(observed, expected, strict=True)
    )
    next_loader = data.train_dataloader()
    assert len(list(next_loader)) == 2


def test_restoring_no_loader_rewinds_a_live_loader_to_the_epoch_count(
    dataset_dir: Path,
) -> None:
    data = _data(dataset_dir, augment=False, seed=6)
    live = data.train_dataloader()
    live.epoch = 9
    data.load_state_dict({"epoch": 0, "loader": None})
    assert live.epoch == 0


@pytest.mark.parametrize("live_first", [True, False])
def test_checkpointed_epoch_survives_a_missing_loader_cursor(
    dataset_dir: Path,
    live_first: bool,
) -> None:
    """``epoch`` alone resumes the shuffle sequence, live loader or not."""
    trained = _data(dataset_dir, augment=False, seed=6)
    for _ in range(2):
        list(trained.train_dataloader())
    expected = next(iter(trained.train_dataloader()))["media"]

    restored = _data(dataset_dir, augment=False, seed=6)
    live = restored.train_dataloader() if live_first else None
    restored.load_state_dict({"epoch": 2, "loader": None})
    loader = live or restored.train_dataloader()
    assert loader.epoch == 2
    assert restored.state_dict()["epoch"] == 2
    assert torch.equal(next(iter(loader))["media"], expected)


def test_a_path_working_dir_is_literal(tmp_path: Path) -> None:
    config = SudokuData.Config()
    config.base_dir = Path("/owner")
    config.working_dir = tmp_path
    assert config.finalize().working_dir == tmp_path
    config = SudokuData.Config()
    config.base_dir = "/owner"
    assert config.finalize().working_dir == Path("/owner/datasets/sudoku-extreme")


def test_an_unowned_config_resolves_like_every_sudoku_dataset() -> None:
    assert (
        SudokuData.Config().finalize().working_dir
        == PuzzleDataset.Config().finalize().working_dir
        == Path("/opt/scratch/datasets/sudoku-extreme")
    )


def test_epoch_order_reads_the_bounds_without_a_per_puzzle_sync(
    dataset_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``int(tensor)`` per puzzle is a host sync; a whole epoch needs at most two."""
    synced: list[int] = []
    original = Tensor.__int__

    def counted(self: Tensor) -> int:
        synced.append(1)
        return original(self)

    data = _data(dataset_dir, augment=False, seed=0)
    train, test = data.train_dataloader(), data.eval_dataloader()
    monkeypatch.setattr(Tensor, "__int__", counted)
    train._order(0)
    assert synced == []
    test._order(0)
    assert len(synced) == 2  # Only the first and last bound.


def test_missing_data_names_the_preparer(tmp_path: Path) -> None:
    data = _data(tmp_path)
    with pytest.raises(FileNotFoundError) as error:
        data.train_dataloader()

    assert str(error.value) == (
        f"no prepared sudoku data at {tmp_path / 'train'}; build it with "
        "`uv --quiet run --frozen python -m "
        "priml.baselines.sudoku.scripts.prepare_data`."
    )


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
