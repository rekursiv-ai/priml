"""Tests for puzzle dataset file and iterator validation."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import json

from torch import Tensor

import numpy as np
import pytest
import torch

from priml.baselines.sudoku import puzzle_data
from priml.baselines.sudoku.puzzle_data import (
    PuzzleDataset,
    _build_dihedral_indices,
    _get_device,
    _PuzzleBatchIterator,
    _subset_eval_iterator,
    augment_sudoku,
    load_puzzle_dataset,
    resolve_working_dir,
)
from priml.baselines.sudoku.puzzle_spec import SudokuSpec
from priml.lib.custom_json import ReadError, convert


def _write(
    root: Path,
    *,
    metadata: bool = True,
    groups: bool = True,
    split_name: str = "train",
) -> None:
    split = root / split_name
    split.mkdir(parents=True, exist_ok=True)
    if metadata:
        (split / "dataset.json").write_text(
            json.dumps({"vocab_size": 11, "seq_len": 81}),
        )
    if groups:
        np.save(split / "all__group_indices.npy", np.array([0, 2], dtype=np.int32))
    np.save(split / "all__inputs.npy", np.full((2, 81), 1, dtype=np.int32))
    np.save(split / "all__labels.npy", np.full((2, 81), 1, dtype=np.int32))


def _write_split(root: Path, split_name: str, values: list[int]) -> None:
    _write(root, split_name=split_name)
    split = root / split_name
    np.save(
        split / "all__inputs.npy",
        np.repeat(np.asarray(values, dtype=np.int32)[:, None], 81, axis=1),
    )
    np.save(
        split / "all__labels.npy",
        np.repeat(np.asarray(values, dtype=np.int32)[:, None], 81, axis=1),
    )
    np.save(
        split / "all__group_indices.npy",
        np.array([0, 2, 4, len(values)], dtype=np.int32),
    )


def _config(root: Path, **overrides: object) -> PuzzleDataset.Config:
    config = PuzzleDataset.Config(working_dir=root, device="cpu", batch_size=2)
    config.spec = SudokuSpec()
    for name, value in overrides.items():
        setattr(config, name, value)
    return config


def test_loader_rejects_missing_files_and_negative_caps(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError) as error:
        load_puzzle_dataset(tmp_path, "train")
    assert str(error.value) == (
        f"Dataset directory not found: {tmp_path / 'train'}. Build it first with"
        " scripts/prepare_data.py."
    )
    _write(tmp_path)
    (tmp_path / "train" / "dataset.json").unlink()
    with pytest.raises(FileNotFoundError, match="metadata"):
        load_puzzle_dataset(tmp_path, "train")
    _write(tmp_path, metadata=True, groups=False)
    (tmp_path / "train" / "all__group_indices.npy").unlink()
    with pytest.raises(FileNotFoundError, match="Required file"):
        load_puzzle_dataset(tmp_path, "train")
    _write(tmp_path)
    with pytest.raises(ValueError, match="non-negative"):
        load_puzzle_dataset(tmp_path, "train", max_samples=-1)


def test_dataset_validates_indices_and_spec_and_subset(tmp_path: Path) -> None:
    _write(tmp_path)
    with pytest.raises(ValueError, match="strictly ascending"):
        _config(tmp_path, eval_instance_indices=(0, 0)).make()
    with pytest.raises(ValueError, match="non-negative"):
        _config(tmp_path, eval_instance_indices=(-1,)).make()
    with pytest.raises(ValueError, match="epoch_offset"):
        _config(tmp_path).make().train_dataloader().__class__(
            tmp_path,
            "cpu",
            2,
            epoch_offset=-1,
            spec=SudokuSpec(),
        )
    wrong = _config(tmp_path)
    wrong.spec = SudokuSpec(grid_shape=(4, 4), box_shape=(2, 2), vocab_size=6)
    with pytest.raises(ValueError, match=r".") as error:
        wrong.make().train_dataloader()
    assert str(error.value) == "Prepared vocabulary does not match dataset spec."
    wrong.spec = SudokuSpec(grid_shape=(4, 4), box_shape=(2, 2), vocab_size=11)
    with pytest.raises(ValueError, match=r".") as error:
        wrong.make().train_dataloader()
    assert str(error.value) == "Prepared grid does not match dataset spec."
    assert _get_device("auto").type in {"cpu", "cuda", "mps"}
    iterator = _PuzzleBatchIterator(tmp_path, "cpu", 3, spec=SudokuSpec())
    assert next(iter(iterator))["valid_count"] == 2


def test_dataset_iterator_edges(tmp_path: Path) -> None:
    _write(tmp_path)
    loaded = load_puzzle_dataset(tmp_path, "train", max_samples=1)
    assert loaded["inputs"].shape[0] == 1
    dataset = _config(tmp_path, num_instances=1).make()
    loader = dataset.train_dataloader()
    assert len(loader) == 1
    assert next(iter(loader))["media"].shape[0] == 2
    dataset.train_dataloader()
    dataset.load_state_dict({"train_epochs": 2})
    assert dataset.state_dict().get("train_epochs") == 2
    assert dataset.eval_batch_size == 2
    assert dataset.config.device == "cpu"


def test_dataset_starts_with_empty_epoch_state(tmp_path: Path) -> None:
    _write_split(tmp_path, "train", [1, 2, 3, 4, 5])
    dataset = _config(tmp_path).make()

    assert dataset._train_epochs == 0
    assert dataset._active_train_iter is None
    assert dataset.state_dict() == {"train_epochs": 0}


def test_train_loader_forwards_config_and_tracks_active_epoch(tmp_path: Path) -> None:
    _write_split(tmp_path, "train", [1, 2, 3, 4, 5])
    dataset = _config(
        tmp_path,
        batch_size=2,
        num_instances=2,
        max_samples=3,
        seed=41,
        augment=True,
        augment_digits_only=True,
        augment_seed=73,
    ).make()

    with patch.object(torch, "Generator", wraps=torch.Generator) as generator_factory:
        loader = dataset.train_dataloader()

    assert generator_factory.call_args.kwargs["device"] == torch.device("cpu")
    assert loader.shuffle is True
    assert loader.batch_size == 2
    assert loader.n_instances == 2
    assert loader.inputs[:, 0].tolist() == [1, 2, 3]
    assert loader.labels[:, 0].tolist() == [1, 2, 3]
    assert loader.instance_bounds.tolist() == [0, 2, 3]
    assert loader.seed == 41
    assert loader.augment is True
    assert loader.augment_digits_only is True
    assert loader._augment_generator is not None
    assert dataset._active_train_iter is loader

    augmented = next(iter(loader))
    plain = _PuzzleBatchIterator(
        tmp_path,
        "cpu",
        2,
        num_instances=2,
        max_samples=3,
        seed=41,
        augment=False,
        spec=SudokuSpec(),
    )
    original = next(iter(plain))
    expected_inputs, expected_labels = augment_sudoku(
        original["media"],
        original["label"],
        spec=SudokuSpec(),
        digits_only=True,
        generator=torch.Generator().manual_seed(73),
    )
    torch.testing.assert_close(augmented["media"], expected_inputs, rtol=0, atol=0)
    torch.testing.assert_close(augmented["label"], expected_labels, rtol=0, atol=0)
    assert dataset.state_dict() == {"train_epochs": 1}
    resumed = dataset.train_dataloader()
    assert resumed._epoch == 1
    assert dataset._active_train_iter is resumed
    dataset.load_state_dict({"train_epochs": 4})
    assert resumed._epoch == 4
    assert dataset.state_dict() == {"train_epochs": 4}


def test_train_loader_applies_instance_cap_without_sample_cap(tmp_path: Path) -> None:
    _write_split(tmp_path, "train", [1, 2, 3, 4, 5, 6])

    loader = _config(tmp_path, num_instances=1).make().train_dataloader()

    assert loader.inputs[:, 0].tolist() == [1, 2]
    assert loader.labels[:, 0].tolist() == [1, 2]
    assert loader.instance_bounds.tolist() == [0, 2]
    assert loader.n_instances == 1


def test_eval_loader_forwards_limits_and_disables_augmentation(tmp_path: Path) -> None:
    _write_split(tmp_path, "train", [1, 2, 3, 4, 5])
    _write_split(tmp_path, "test", [6, 7, 8, 9, 10])
    dataset = _config(
        tmp_path,
        batch_size=2,
        eval_batch_size=3,
        eval_num_instances=2,
        max_samples=3,
        augment=True,
    ).make()

    with patch(
        "priml.baselines.sudoku.puzzle_data._PuzzleBatchIterator",
        wraps=_PuzzleBatchIterator,
    ) as iterator_factory:
        loader = dataset.eval_dataloader()

    assert iterator_factory.call_args.kwargs["train"] is False
    batches = list(loader)
    assert loader.shuffle is False
    assert loader.batch_size == 3
    assert loader.n_instances == 2
    assert loader.inputs[:, 0].tolist() == [6, 7, 8]
    assert loader.instance_bounds.tolist() == [0, 2, 3]
    assert loader.augment is False
    assert loader._augment_generator is None
    assert len(batches) == 1
    assert batches[0]["media"][:, 0].tolist() == [6, 7, 8]
    assert batches[0]["label"][:, 0].tolist() == [6, 7, 8]
    assert batches[0]["valid_count"] == 3


def test_eval_loader_applies_instance_and_sample_caps_independently(
    tmp_path: Path,
) -> None:
    _write_split(tmp_path, "train", [1, 2, 3, 4, 5, 6])
    _write_split(tmp_path, "test", [7, 8, 9, 10, 11, 12])

    instance_capped = _config(tmp_path, eval_num_instances=1).make().eval_dataloader()
    assert instance_capped.inputs[:, 0].tolist() == [7, 8]
    assert instance_capped.instance_bounds.tolist() == [0, 2]
    assert instance_capped.n_instances == 1

    sample_capped = _config(tmp_path, max_samples=3).make().eval_dataloader()
    assert sample_capped.inputs[:, 0].tolist() == [7, 8, 9]
    assert sample_capped.instance_bounds.tolist() == [0, 2, 3]
    assert sample_capped.n_instances == 2


def test_dataset_subset_bounds_and_state_load(tmp_path: Path) -> None:
    _write(tmp_path)
    _write(tmp_path, split_name="test")
    dataset = _config(tmp_path, eval_instance_indices=(0,)).make()
    assert next(iter(dataset.eval_dataloader()))["valid_count"] == 2
    with pytest.raises(ValueError, match="outside"):
        _config(tmp_path, eval_instance_indices=(1, 2)).make().eval_dataloader()
    dataset.load_state_dict({"train_epochs": 3})
    assert dataset.state_dict().get("train_epochs") == 3


def test_resolve_working_dir_keeps_logical_path_beneath_base() -> None:
    assert resolve_working_dir(None, "/datasets/sudoku") == Path(
        "/opt/scratch/datasets/sudoku",
    )
    assert resolve_working_dir("/runs", "nested/data") == Path("/runs/nested/data")
    assert resolve_working_dir("/runs", "X/nested") == Path("/runs/X/nested")


@pytest.mark.parametrize(
    ("cuda_available", "mps_available", "expected"),
    [(True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")],
)
def test_auto_device_prefers_cuda_then_mps_then_cpu(
    monkeypatch: pytest.MonkeyPatch,
    cuda_available: bool,
    mps_available: bool,
    expected: str,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps_available)

    assert _get_device("auto") == torch.device(expected)


def test_augmentation_preserves_tokens_and_pairing() -> None:
    spec = SudokuSpec(grid_shape=(2, 2), box_shape=(1, 2))
    inputs = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]], dtype=torch.int32)
    labels = torch.tensor([[3, 2, 1, 0], [0, 1, 2, 3]], dtype=torch.int32)
    generator = torch.Generator().manual_seed(13)

    augmented_inputs, augmented_labels = augment_sudoku(
        inputs,
        labels,
        spec=spec,
        digits_only=True,
        generator=generator,
    )

    assert augmented_inputs.dtype == torch.int32
    assert augmented_labels.dtype == torch.int32
    assert augmented_inputs.shape == inputs.shape
    assert augmented_labels.shape == labels.shape
    assert (augmented_inputs == 1).sum(dim=1).tolist() == [1, 1]
    for row in range(len(inputs)):
        transform = dict(
            zip(inputs[row].tolist(), augmented_inputs[row].tolist(), strict=True),
        )
        assert [transform[token] for token in labels[row].tolist()] == (
            augmented_labels[row].tolist()
        )


def test_augmentation_rejects_rectangular_grid() -> None:
    spec = SudokuSpec(grid_shape=(2, 3), box_shape=(1, 3))
    inputs = torch.zeros((2, 6), dtype=torch.int64)

    with pytest.raises(ValueError, match="Expected") as error:
        augment_sudoku(inputs, inputs, spec=spec)
    assert str(error.value) == "Expected inputs.shape[1] == n * n."


def test_augmentation_rejects_spec_with_wrong_grid_shape() -> None:
    spec = SudokuSpec(grid_shape=(2, 3), box_shape=(1, 3))
    inputs = torch.zeros((2, 9), dtype=torch.int64)

    with pytest.raises(ValueError, match="Expected") as error:
        augment_sudoku(inputs, inputs, spec=spec)
    assert str(error.value) == "Expected spec.grid_shape == (n, n)."


def test_augmentation_factories_use_input_device_and_dtype() -> None:
    device = torch.device("cpu")
    spec = SudokuSpec(grid_shape=(3, 3), box_shape=(3, 3))
    inputs = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7, 8], [8, 7, 6, 5, 4, 3, 2, 1, 0]],
        dtype=torch.float64,
    )
    generator = torch.Generator(device=device)
    with (
        patch.object(torch, "rand", wraps=torch.rand) as rand,
        patch.object(torch, "randint", wraps=torch.randint) as randint,
    ):
        augmented_inputs, augmented_labels = augment_sudoku(
            inputs,
            inputs,
            spec=spec,
            generator=generator,
        )

    assert augmented_inputs.device == inputs.device
    assert augmented_labels.device == inputs.device
    assert augmented_inputs.dtype == inputs.dtype
    assert augmented_labels.dtype == inputs.dtype
    rand.assert_called_once_with(2, 3, device=device, generator=generator)
    randint.assert_called_once_with(0, 8, (2,), device=device, generator=generator)

    meta_inputs = torch.empty((2, 9), device="meta", dtype=torch.float64)
    meta_outputs = augment_sudoku(meta_inputs, meta_inputs, spec=spec)
    assert all(output.device.type == "meta" for output in meta_outputs)
    assert all(output.dtype == torch.float64 for output in meta_outputs)


def test_legacy_augmentation_keeps_token_ten_fixed() -> None:
    spec = SudokuSpec(grid_shape=(4, 4), box_shape=(2, 2), vocab_size=12)
    inputs = torch.arange(12, dtype=torch.int32).repeat(2, 2)[:, :16]
    labels = inputs.clone()

    expected_by_seed = {
        0: [
            [1, 4, 3, 0, 7, 6, 5, 2, 11, 10, 9, 8, 1, 4, 3, 0],
            [2, 11, 7, 2, 3, 10, 6, 3, 1, 9, 5, 1, 0, 8, 4, 0],
        ],
        23: [
            [3, 4, 2, 0, 11, 10, 9, 8, 7, 6, 5, 1, 3, 4, 2, 0],
            [4, 11, 7, 4, 3, 10, 6, 3, 2, 9, 5, 2, 0, 8, 1, 0],
        ],
        91: [
            [1, 4, 3, 0, 11, 10, 9, 8, 7, 6, 5, 2, 1, 4, 3, 0],
            [0, 3, 1, 4, 8, 9, 10, 11, 2, 5, 6, 7, 0, 3, 1, 4],
        ],
    }
    for seed, expected in expected_by_seed.items():
        augmented_inputs, augmented_labels = augment_sudoku(
            inputs,
            labels,
            spec=spec,
            generator=torch.Generator().manual_seed(seed),
        )

        assert augmented_inputs.shape == inputs.shape
        assert augmented_labels.shape == labels.shape
        assert augmented_inputs.dtype == torch.int32
        assert augmented_labels.dtype == torch.int32
        assert (augmented_inputs == 10).sum(dim=1).tolist() == [1, 1]
        assert torch.equal(augmented_inputs, augmented_labels)
        assert (augmented_inputs == 0).sum(dim=1).tolist() == [2, 2]
        assert torch.all(
            (augmented_inputs >= 0) & (augmented_inputs < spec.vocab_size),
        )
        assert augmented_inputs.tolist() == expected


def test_iterator_constructor_uses_requested_meta_device(tmp_path: Path) -> None:
    _write(tmp_path)
    iterator = _PuzzleBatchIterator(
        tmp_path,
        "meta",
        2,
        train=True,
        shuffle=False,
        augment=True,
        spec=SudokuSpec(),
    )

    assert iterator.device.type == "meta"
    assert iterator.inputs.device.type == "meta"
    assert iterator.labels.device.type == "meta"
    assert iterator.instance_bounds.device.type == "meta"
    assert iterator.augment is True
    assert iterator._augment_generator is None


def test_iterator_applies_sample_cap_before_instance_cap(tmp_path: Path) -> None:
    _write(tmp_path)
    split = tmp_path / "train"
    np.save(
        split / "all__group_indices.npy",
        np.array([0, 2, 4, 6], dtype=np.int32),
    )
    np.save(
        split / "all__inputs.npy",
        np.repeat(np.arange(6, dtype=np.int32)[:, None], 81, axis=1),
    )
    np.save(
        split / "all__labels.npy",
        np.repeat((10 + np.arange(6, dtype=np.int32))[:, None], 81, axis=1),
    )

    iterator = _PuzzleBatchIterator(
        tmp_path,
        "cpu",
        2,
        num_instances=2,
        max_samples=3,
        shuffle=False,
        spec=SudokuSpec(),
    )

    assert iterator.inputs[:, 0].tolist() == [0, 1, 2]
    assert iterator.labels[:, 0].tolist() == [10, 11, 12]
    assert iterator.instance_bounds.tolist() == [0, 2, 3]
    assert iterator.n_instances == 2
    assert list(iterator)[1]["media"][:, 0].tolist() == [2, 0]


@pytest.mark.parametrize(
    ("num_instances", "expected_inputs", "expected_bounds"),
    [
        (2, [0, 1, 2, 3], [0, 2, 4]),
        (3, [0, 1, 2, 3, 4, 5], [0, 2, 4, 6]),
        (4, [0, 1, 2, 3, 4, 5], [0, 2, 4, 6]),
    ],
)
def test_iterator_instance_limit_boundaries(
    tmp_path: Path,
    num_instances: int,
    expected_inputs: list[int],
    expected_bounds: list[int],
) -> None:
    _write(tmp_path)
    split = tmp_path / "train"
    np.save(
        split / "all__group_indices.npy",
        np.array([0, 2, 4, 6], dtype=np.int32),
    )
    np.save(
        split / "all__inputs.npy",
        np.repeat(np.arange(6, dtype=np.int32)[:, None], 81, axis=1),
    )
    np.save(
        split / "all__labels.npy",
        np.repeat((10 + np.arange(6, dtype=np.int32))[:, None], 81, axis=1),
    )

    iterator = _PuzzleBatchIterator(
        tmp_path,
        "cpu",
        2,
        num_instances=num_instances,
        spec=SudokuSpec(),
    )

    assert iterator.inputs[:, 0].tolist() == expected_inputs
    assert iterator.instance_bounds.tolist() == expected_bounds
    assert iterator.n_instances == min(num_instances, 3)


def test_iterator_emits_exact_order_and_pads_final_batch(tmp_path: Path) -> None:
    _write(tmp_path)
    split = tmp_path / "train"
    np.save(
        split / "all__inputs.npy",
        np.repeat(np.arange(3, dtype=np.int32)[:, None], 81, axis=1),
    )
    np.save(
        split / "all__labels.npy",
        np.repeat((4 + np.arange(3, dtype=np.int32))[:, None], 81, axis=1),
    )
    np.save(
        split / "all__group_indices.npy",
        np.array([0, 2, 3], dtype=np.int32),
    )
    iterator = _PuzzleBatchIterator(
        tmp_path,
        "cpu",
        2,
        train=True,
        shuffle=False,
        spec=SudokuSpec(),
    )

    with (
        patch.object(torch, "arange", wraps=torch.arange) as arange,
        patch.object(torch, "zeros", wraps=torch.zeros) as zeros,
    ):
        batches = list(iterator)

    assert all(
        call.kwargs["device"] == torch.device("cpu") for call in arange.call_args_list
    )
    assert all(
        call.kwargs["device"] == torch.device("cpu") for call in zeros.call_args_list
    )
    assert len(iterator) == 2
    assert len(batches) == 2
    assert batches[0]["media"].dtype == torch.int32
    assert batches[0]["label"].dtype == torch.int32
    assert batches[0]["media"][:, 0].tolist() == [0, 1]
    assert batches[0]["label"][:, 0].tolist() == [4, 5]
    assert batches[0]["valid_count"] == 2
    assert batches[1]["media"][:, 0].tolist() == [2, 0]
    assert batches[1]["label"][:, 0].tolist() == [6, 0]
    assert batches[1]["valid_count"] == 1
    assert batches[1]["puzzle_identifiers"].dtype == torch.int32
    assert batches[1]["puzzle_identifiers"].tolist() == [0, 0]
    assert iterator.device == torch.device("cpu")
    assert iterator.vocab_size == 11
    assert iterator.seq_len == 81
    assert iterator.n_instances == 2
    assert iterator.batch_size == 2
    assert iterator.shuffle is False
    assert iterator.seed is None
    assert iterator.augment is False
    assert iterator.augment_digits_only is False
    assert iterator._augment_generator is None
    assert iterator._epoch == 1


def test_full_batches_skip_padding_allocation() -> None:
    iterator = _PuzzleBatchIterator.__new__(_PuzzleBatchIterator)
    iterator.instance_bounds = torch.tensor([0, 2, 4])
    iterator.n_instances = 2
    iterator.shuffle = False
    iterator.seed = None
    iterator._epoch = 0
    iterator.inputs = torch.arange(8).reshape(4, 2)
    iterator.labels = iterator.inputs + 100
    iterator.batch_size = 2
    iterator.augment = False
    iterator.spec = SudokuSpec()
    iterator.augment_digits_only = False
    iterator._augment_generator = None

    with patch.object(Tensor, "new_zeros", wraps=Tensor.new_zeros) as new_zeros:
        batches = list(iterator)

    assert len(batches) == 2
    assert [batch["valid_count"] for batch in batches] == [2, 2]
    assert [batch["media"][:, 0].tolist() for batch in batches] == [[0, 2], [4, 6]]
    new_zeros.assert_not_called()


def _ordered_first_epoch(root: Path, seed: int) -> list[int]:
    dataset = _config(
        root,
        batch_size=4,
        seed=seed,
        num_instances=2,
        max_samples=6,
    ).make()
    return convert(
        torch.cat(
            [
                batch["media"][: batch["valid_count"], 0]
                for batch in dataset.train_dataloader()
            ],
        ).tolist(),
        list[int],
    )


def test_train_loader_uses_seeded_instance_and_sample_limits(tmp_path: Path) -> None:
    _write(tmp_path)
    split = tmp_path / "train"
    np.save(
        split / "all__group_indices.npy",
        np.array([0, 3, 6], dtype=np.int32),
    )
    np.save(
        split / "all__inputs.npy",
        np.repeat(np.arange(6, dtype=np.int32)[:, None], 81, axis=1),
    )
    np.save(
        split / "all__labels.npy",
        np.repeat((10 + np.arange(6, dtype=np.int32))[:, None], 81, axis=1),
    )

    assert len(_ordered_first_epoch(tmp_path, 31)) == 6
    assert _ordered_first_epoch(tmp_path, 31) == _ordered_first_epoch(tmp_path, 31)
    assert _ordered_first_epoch(tmp_path, 31) != _ordered_first_epoch(tmp_path, 37)


def test_iterator_shuffle_advances_seed_by_epoch(tmp_path: Path) -> None:
    _write(tmp_path)
    split = tmp_path / "train"
    np.save(
        split / "all__group_indices.npy",
        np.array([0, 3, 6], dtype=np.int32),
    )
    np.save(
        split / "all__inputs.npy",
        np.repeat(np.arange(6, dtype=np.int32)[:, None], 81, axis=1),
    )
    np.save(
        split / "all__labels.npy",
        np.repeat((10 + np.arange(6, dtype=np.int32))[:, None], 81, axis=1),
    )
    iterator = _PuzzleBatchIterator(
        tmp_path,
        "cpu",
        6,
        seed=31,
        spec=SudokuSpec(),
    )
    with (
        patch.object(torch, "Generator", wraps=torch.Generator) as generators,
        patch.object(torch, "randperm", wraps=torch.randperm) as randperm,
    ):
        next(iter(iterator))
        assert iterator._epoch == 1
        second_epoch = next(iter(iterator))["media"][:, 0].tolist()
        assert iterator._epoch == 2
        third_epoch = next(iter(iterator))["media"][:, 0].tolist()
        assert iterator._epoch == 3

    assert all(
        call.kwargs["device"] == torch.device("cpu")
        for call in generators.call_args_list
    )
    assert all(
        call.kwargs["device"] == torch.device("cpu") for call in randperm.call_args_list
    )
    next_epoch = _PuzzleBatchIterator(
        tmp_path,
        "cpu",
        6,
        seed=32,
        spec=SudokuSpec(),
    )

    assert second_epoch == next(iter(next_epoch))["media"][:, 0].tolist()
    assert third_epoch != second_epoch


def test_loader_caps_rows_and_clips_group_boundaries(tmp_path: Path) -> None:
    _write(tmp_path)
    split = tmp_path / "train"
    np.save(
        split / "all__group_indices.npy",
        np.array([0, 2, 4], dtype=np.int32),
    )
    np.save(
        split / "all__inputs.npy",
        np.arange(4 * 81, dtype=np.int64).reshape(4, 81),
    )
    np.save(
        split / "all__labels.npy",
        (1000 + np.arange(4 * 81, dtype=np.int64)).reshape(4, 81),
    )

    loaded = load_puzzle_dataset(tmp_path, "train", max_samples=3)

    assert loaded["inputs"].dtype == torch.int32
    assert loaded["labels"].dtype == torch.int32
    assert loaded["inputs"][:, 0].tolist() == [0, 81, 162]
    assert loaded["labels"][:, 0].tolist() == [1000, 1081, 1162]
    assert loaded["group_indices"].tolist() == [0, 2, 3]
    assert loaded["vocab_size"] == 11
    assert loaded["seq_len"] == 81

    empty = load_puzzle_dataset(tmp_path, "train", max_samples=0)
    assert empty["inputs"].shape == (0, 81)
    assert empty["labels"].shape == (0, 81)
    assert empty["group_indices"].tolist() == [0]

    (split / "dataset.json").write_text("{}")
    with pytest.raises(ReadError):
        load_puzzle_dataset(tmp_path, "train")


def test_subset_eval_iterator_selects_exact_instances(tmp_path: Path) -> None:
    _write(tmp_path, split_name="test")
    split = tmp_path / "test"
    np.save(
        split / "all__group_indices.npy",
        np.array([0, 2, 5, 6], dtype=np.int32),
    )
    np.save(
        split / "all__inputs.npy",
        np.repeat(np.arange(6, dtype=np.int32)[:, None], 81, axis=1),
    )
    np.save(
        split / "all__labels.npy",
        np.repeat((10 + np.arange(6, dtype=np.int32))[:, None], 81, axis=1),
    )
    dataset = _config(tmp_path, eval_instance_indices=(1, 2)).make()

    batches = list(dataset.eval_dataloader())

    assert len(batches) == 2
    assert batches[0]["valid_count"] == 2
    assert batches[0]["media"][:, 0].tolist() == [2, 3]
    assert batches[0]["label"][:, 0].tolist() == [12, 13]
    assert batches[0]["puzzle_identifiers"].dtype == torch.int32
    assert batches[0]["puzzle_identifiers"].tolist() == [0, 0]
    assert batches[1]["valid_count"] == 2
    assert batches[1]["media"][:, 0].tolist() == [4, 5]
    assert batches[1]["label"][:, 0].tolist() == [14, 15]

    iterator = _PuzzleBatchIterator(
        tmp_path,
        "cpu",
        2,
        train=False,
        shuffle=False,
        spec=SudokuSpec(),
    )
    subset = _subset_eval_iterator(iterator, (1, 2))
    assert subset.instance_bounds.tolist() == [0, 3, 4]
    assert subset.n_instances == 2
    assert subset.inputs[:, 0].tolist() == [2, 3, 4, 5]


def test_subset_eval_iterator_pins_dtype_device_and_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    iterator = _PuzzleBatchIterator.__new__(_PuzzleBatchIterator)
    iterator.inputs = torch.arange(24, dtype=torch.int32).reshape(6, 4)
    iterator.labels = iterator.inputs + 100
    iterator.instance_bounds = torch.tensor([0, 2, 3, 6], dtype=torch.int64)
    iterator.n_instances = 3

    tensor = Mock(wraps=torch.tensor)
    zeros = Mock(wraps=torch.zeros)
    original_to = Tensor.to
    to_devices: list[torch.device] = []

    def to(tensor_value: Tensor, *args: object, **kwargs: object) -> Tensor:
        device = args[0] if args else kwargs.get("device")
        assert isinstance(device, (torch.device, str))
        to_devices.append(torch.device(device))
        return original_to(tensor_value, device)

    monkeypatch.setattr(torch, "tensor", tensor)
    monkeypatch.setattr(torch, "zeros", zeros)
    monkeypatch.setattr(Tensor, "to", to)

    selected = _subset_eval_iterator(iterator, (0, 2))

    expected_inputs = torch.tensor(
        [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [12, 13, 14, 15],
            [16, 17, 18, 19],
            [20, 21, 22, 23],
        ],
        dtype=torch.int32,
    )
    assert selected is iterator
    torch.testing.assert_close(iterator.inputs, expected_inputs, rtol=0, atol=0)
    torch.testing.assert_close(iterator.labels, expected_inputs + 100, rtol=0, atol=0)
    assert iterator.instance_bounds.dtype == torch.int64
    torch.testing.assert_close(iterator.instance_bounds, torch.tensor([0, 2, 5]))
    assert iterator.n_instances == 2
    assert tensor.call_args_list[0].kwargs["dtype"] is torch.int64
    assert zeros.call_args_list[0].kwargs["dtype"] is torch.int64
    assert to_devices == [torch.device("cpu"), torch.device("cpu")]

    out_of_range = _PuzzleBatchIterator.__new__(_PuzzleBatchIterator)
    out_of_range.inputs = torch.zeros((2, 3))
    out_of_range.labels = torch.zeros((2, 3))
    out_of_range.instance_bounds = torch.tensor([0, 1])
    out_of_range.n_instances = 1
    with pytest.raises(ValueError, match="max 1 is outside"):
        _subset_eval_iterator(out_of_range, (1,))


def test_loader_preserves_mmap_and_metadata_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    split = tmp_path / "test"
    split.mkdir()
    (split / "dataset.json").write_text('{"vocab_size":11,"seq_len":4}')
    np.save(split / "all__group_indices.npy", np.array([0, 2, 5, 6], dtype=np.int32))
    source_inputs = np.arange(24, dtype=np.int16).reshape(6, 4)
    source_labels = (source_inputs + 30).astype(np.int16)
    np.save(split / "all__inputs.npy", source_inputs)
    np.save(split / "all__labels.npy", source_labels)

    load = Mock(wraps=np.load)
    convert_metadata = Mock(wraps=convert)
    monkeypatch.setattr(np, "load", load)
    monkeypatch.setattr(puzzle_data, "convert", convert_metadata)

    data = load_puzzle_dataset(tmp_path, "test", max_samples=5)

    assert data["inputs"].dtype == torch.int32
    assert data["labels"].dtype == torch.int32
    torch.testing.assert_close(
        data["inputs"],
        torch.from_numpy(source_inputs[:5].astype(np.int32)),
    )
    torch.testing.assert_close(
        data["labels"],
        torch.from_numpy(source_labels[:5].astype(np.int32)),
    )
    assert data["group_indices"].dtype == torch.int32
    torch.testing.assert_close(
        data["group_indices"],
        torch.tensor([0, 2, 5], dtype=torch.int32),
    )
    assert data["vocab_size"] == 11
    assert data["seq_len"] == 4
    assert [call.args[0] for call in load.call_args_list] == [
        split / "all__group_indices.npy",
        split / "all__inputs.npy",
        split / "all__labels.npy",
    ]
    assert [call.kwargs.get("mmap_mode") for call in load.call_args_list] == [
        None,
        "r",
        "r",
    ]
    assert [call.args for call in convert_metadata.call_args_list[1:]] == [
        (11, int),
        (4, int),
    ]


def test_augmentation_pins_digit_only_output_and_dihedral_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = SudokuSpec(grid_shape=(2, 2), box_shape=(1, 2))
    inputs = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]], dtype=torch.int16)
    labels = torch.tensor([[3, 2, 1, 0], [0, 1, 2, 3]], dtype=torch.int64)
    arange = Mock(wraps=torch.arange)
    original_build = _build_dihedral_indices
    devices: list[torch.device | None] = []

    def build(n: int, device: torch.device) -> Tensor:
        devices.append(device)
        return original_build(n, device)

    def rand(
        batch: int,
        width: int,
        *,
        device: torch.device,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        assert (batch, width) == (2, 2)
        del generator
        return torch.tensor([[0.9, 0.1], [0.9, 0.1]], device=device)

    def randint(
        low: int,
        high: int,
        size: tuple[int, ...],
        *,
        device: torch.device,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        assert (low, high, size) == (0, 8, (2,))
        del generator
        return torch.zeros(size, dtype=torch.int64, device=device)

    monkeypatch.setattr(torch, "arange", arange)
    monkeypatch.setattr(torch, "rand", rand)
    monkeypatch.setattr(torch, "randint", randint)
    monkeypatch.setattr(puzzle_data, "_build_dihedral_indices", build)

    augmented_inputs, augmented_labels = augment_sudoku(
        inputs,
        labels,
        spec=spec,
        digits_only=True,
    )

    torch.testing.assert_close(
        augmented_inputs,
        torch.tensor([[0, 1, 3, 2], [2, 3, 1, 0]], dtype=torch.int16),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        augmented_labels,
        torch.tensor([[2, 3, 1, 0], [0, 1, 3, 2]], dtype=torch.int64),
        rtol=0,
        atol=0,
    )
    assert arange.call_args_list[0].kwargs["dtype"] is torch.long
    assert arange.call_args_list[0].kwargs["device"] == inputs.device
    assert devices == [inputs.device]


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
