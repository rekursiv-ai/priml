"""ARC2 sampling and padding, checked against source-minted records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import json

from numpy.typing import NDArray
from torch import Tensor

import numpy as np
import pytest
import torch
import torch.distributed as dist

from priml.baselines.arcagi1.augmentation import ArcSpec
from priml.baselines.arcagi2.data import Arc2Data, ArcBatches
from priml.baselines.arcagi2.record_test import assert_matches, reduce


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping
    from pathlib import Path


class Loader(Protocol):
    """The dataset surface the recorder drives."""

    def train_dataloader(self) -> Iterable[Mapping[str, object]]:
        """Return a training pass."""
        ...


@dataclass(slots=True, kw_only=True)
class _Prepared:
    inputs: Tensor | NDArray[np.generic]
    labels: Tensor | NDArray[np.generic]
    groups: NDArray[np.int64]
    puzzles: NDArray[np.int64]
    identifiers: NDArray[np.int64]
    ignore_label_id: int
    batch_size: int
    device: torch.device
    seed: int


@dataclass(slots=True, kw_only=True)
class _TestDataSource:
    prepared: _Prepared

    def train_dataloader(self) -> _Prepared:
        return self.prepared

    def eval_dataloader(self) -> _Prepared:
        return self.prepared


def _tensor(batch: Mapping[str, object], key: str) -> Tensor:
    value = batch[key]
    assert isinstance(value, Tensor)
    return value


def write_tree(root: Path) -> None:
    """Two tasks, four rows, sharing one tree between both splits."""
    for split in ("train", "test"):
        directory = root / split
        directory.mkdir()
        (directory / "dataset.json").write_text(
            json.dumps({"ignore_label_id": 0, "blank_identifier_id": 0}),
        )
        input_offset = int(split == "test")
        arrays = {
            "inputs": (np.arange(36, dtype=np.int32).reshape(4, 9) + input_offset) % 12,
            "labels": np.arange(36, dtype=np.int32).reshape(4, 9) % 12,
            "puzzle_indices": np.array([0, 2, 4], dtype=np.int64),
            "group_indices": np.array([0, 1, 2], dtype=np.int64),
            "puzzle_identifiers": np.array([1, 2], dtype=np.int32),
        }
        for name, array in arrays.items():
            np.save(directory / f"all__{name}.npy", array)


def record_passes(build: Callable[[], Loader]) -> dict[str, Tensor]:
    """Record two consecutive training passes of one loader."""
    loader = build().train_dataloader()
    out: dict[str, object] = {}
    for epoch in range(2):
        for index, batch in enumerate(loader):
            for key in ("media", "label", "puzzle_identifiers"):
                value = batch[key]
                assert isinstance(value, Tensor)
                out[f"{epoch}/{index}/{key}"] = value
    return reduce(out)


def port_data(root: Path) -> Arc2Data:
    """Build the exported loader over ``root``."""
    return Arc2Data.Config(
        working_dir=root,
        batch_size=2,
        device="cpu",
        epochs_per_iter=1,
    ).make()


def test_reference_batches(tmp_path: Path) -> None:
    """Sampled task order and batches match; a mid-pass resume continues."""
    write_tree(tmp_path)
    assert_matches("data", "passes", record_passes(lambda: port_data(tmp_path)))
    port = port_data(tmp_path)
    actual_loader = port.train_dataloader()
    assert len(actual_loader) == 2
    iterator = iter(actual_loader)
    next(iterator)
    resumed = port_data(tmp_path)
    resumed.load_state_dict(port.state_dict())
    rest = list(iterator)
    replay = list(resumed.train_dataloader())
    assert len(rest) == len(replay)
    for expected, actual in zip(rest, replay, strict=True):
        left, right = expected["media"], actual["media"]
        assert isinstance(left, Tensor)
        assert isinstance(right, Tensor)
        assert torch.equal(left, right), "mid-pass resume"


def test_reference_batches_golden_bites(tmp_path: Path) -> None:
    """One changed token in the recorded passes is reported, not absorbed."""
    write_tree(tmp_path)
    record = record_passes(lambda: port_data(tmp_path))
    record["0/0/media"] = record["0/0/media"].clone()
    record["0/0/media"].view(-1)[0] += 1
    with pytest.raises(AssertionError, match="1 mismatches"):
        assert_matches("data", "passes", record)


def test_arc2_requires_resident_data_and_checkpoint_fields(tmp_path: Path) -> None:
    write_tree(tmp_path)
    config = Arc2Data.Config(working_dir=tmp_path, batch_size=2, device="cpu")
    data = config.make()
    state = {key: value for key, value in data.state_dict().items() if key != "passes"}
    with pytest.raises(KeyError):
        data.load_state_dict(state)


def test_eval_batches_preserve_rows_and_pad_every_field(tmp_path: Path) -> None:
    write_tree(tmp_path)
    data = Arc2Data.Config(
        working_dir=tmp_path,
        batch_size=2,
        eval_batch_size=3,
        device="cpu",
    ).make()

    stream = data.eval_dataloader()
    assert stream.train is False
    assert stream.epochs_per_iter == 1
    assert len(stream) == 2
    batches = list(stream)

    assert len(batches) == 2
    assert batches[0]["valid_count"] == 3
    assert batches[1]["valid_count"] == 1
    assert torch.equal(
        _tensor(batches[0], "media"),
        (torch.arange(27).reshape(3, 9) + 1) % 12,
    )
    assert torch.equal(
        _tensor(batches[0], "puzzle_identifiers"),
        torch.tensor([1, 1, 2]),
    )
    assert torch.equal(_tensor(batches[1], "media")[0], (torch.arange(27, 36) + 1) % 12)
    assert torch.equal(
        _tensor(batches[1], "puzzle_identifiers"),
        torch.tensor([2, 0, 0]),
    )
    assert torch.equal(
        _tensor(batches[1], "media")[1:],
        torch.zeros((2, 9), dtype=torch.int32),
    )
    assert torch.equal(_tensor(batches[1], "label")[1:], torch.full((2, 9), -100))
    assert torch.equal(
        _tensor(batches[1], "puzzle_identifiers")[1:],
        torch.zeros(2, dtype=torch.int64),
    )
    for batch in batches:
        tags = batch["spatial_tags"]
        assert isinstance(tags, Tensor)
        assert torch.equal(tags, torch.tensor([[1, 0, 0]] * 3))
        assert set(batch) == {
            "media",
            "label",
            "puzzle_identifiers",
            "valid_count",
            "spatial_tags",
        }


def test_batch_stream_default_pass_and_training_source(tmp_path: Path) -> None:
    write_tree(tmp_path)
    data = port_data(tmp_path)

    stream = ArcBatches(data.prepared, train=True, epochs_per_iter=1)

    assert stream.passes == 0
    assert stream.train is True
    assert stream.active_pass is None
    assert stream.next_batch == 0
    assert len(stream) == 2
    stream.next_batch = 1
    assert len(stream) == 1
    data.passes = 7
    assert data.train_dataloader().passes == 7


def test_training_length_uses_active_pass_and_next_pass_seed() -> None:
    prepared = _Prepared(
        inputs=torch.ones((6, 9), dtype=torch.int32),
        labels=torch.ones((6, 9), dtype=torch.int32),
        groups=np.array([0, 2, 3]),
        puzzles=np.array([0, 1, 5, 6]),
        identifiers=np.array([1, 2, 3]),
        ignore_label_id=0,
        batch_size=2,
        device=torch.device("cpu"),
        seed=0,
    )
    data = _TestDataSource(prepared=prepared)
    stream = ArcBatches(data, train=True, epochs_per_iter=4, passes=0)
    counts = [sum(1 for _ in stream._plan(pass_index)) for pass_index in range(1, 20)]
    assert len(set(counts)) > 1
    active_pass = 1
    fallback_pass = next(
        index
        for index, count in enumerate(counts, start=1)
        if count != counts[active_pass - 1]
    )
    stream.active_pass = active_pass
    stream.passes = fallback_pass - 1
    assert len(stream) == counts[active_pass - 1]
    stream.active_pass = None
    next_pass = next(
        index for index in range(1, len(counts)) if counts[index - 1] != counts[index]
    )
    stream.passes = next_pass - 1
    assert len(stream) == counts[next_pass - 1]


def test_training_packs_only_available_rows_into_global_batches() -> None:
    prepared = _Prepared(
        inputs=torch.ones((6, 9), dtype=torch.int32),
        labels=torch.ones((6, 9), dtype=torch.int32),
        groups=np.array([0, 1, 2, 3]),
        puzzles=np.array([0, 2, 4, 6]),
        identifiers=np.array([1, 2, 3]),
        ignore_label_id=0,
        batch_size=3,
        device=torch.device("cpu"),
        seed=0,
    )
    data = _TestDataSource(prepared=prepared)

    batches = list(ArcBatches(data, train=True, epochs_per_iter=2))

    assert len(batches) == 3
    assert [batch["valid_count"] for batch in batches] == [3, 3, 3]


def test_training_drops_a_short_global_batch() -> None:
    prepared = _Prepared(
        inputs=torch.ones((2, 9), dtype=torch.int32),
        labels=torch.ones((2, 9), dtype=torch.int32),
        groups=np.array([0, 1, 2]),
        puzzles=np.array([0, 1, 2]),
        identifiers=np.array([1, 2]),
        ignore_label_id=0,
        batch_size=3,
        device=torch.device("cpu"),
        seed=0,
    )
    data = _TestDataSource(prepared=prepared)

    stream = ArcBatches(data, train=True, epochs_per_iter=1)

    assert len(stream) == 0
    assert list(stream) == []


def test_plan_samples_without_replacement(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[bool | None] = []

    class Generator:
        def __init__(self, bit_generator: object) -> None:
            del bit_generator

        def permutation(self, count: int) -> np.ndarray:
            return np.arange(count)

        def integers(self, low: int, high: int) -> int:
            del high
            return low

        def choice(
            self,
            count: int,
            size: int,
            *,
            replace: bool | None,
        ) -> np.ndarray:
            del count
            calls.append(replace)
            return np.arange(size)

    prepared = _Prepared(
        inputs=torch.ones((4, 9), dtype=torch.int32),
        labels=torch.ones((4, 9), dtype=torch.int32),
        groups=np.array([0, 2]),
        puzzles=np.array([0, 2, 4]),
        identifiers=np.array([1, 2]),
        ignore_label_id=0,
        batch_size=2,
        device=torch.device("cpu"),
        seed=0,
    )
    data = _TestDataSource(prepared=prepared)
    monkeypatch.setattr(np.random, "Generator", Generator)

    batches = list(ArcBatches(data, train=True, epochs_per_iter=1))

    assert len(batches) == 1
    assert calls
    assert all(replace is False for replace in calls)


def test_full_batches_do_not_allocate_padding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _Prepared(
        inputs=torch.ones((4, 9), dtype=torch.int32),
        labels=torch.ones((4, 9), dtype=torch.int32),
        groups=np.array([0, 1, 2]),
        puzzles=np.array([0, 2, 4]),
        identifiers=np.array([1, 2]),
        ignore_label_id=0,
        batch_size=2,
        device=torch.device("cpu"),
        seed=0,
    )
    data = _TestDataSource(prepared=prepared)
    concatenations: list[object] = []
    original_cat = torch.cat

    def cat(tensors: list[Tensor], dim: int = 0) -> Tensor:
        concatenations.append(tensors)
        return original_cat(tensors, dim=dim)

    monkeypatch.setattr(torch, "cat", cat)

    batches = list(ArcBatches(data, train=True, epochs_per_iter=1))

    assert [batch["valid_count"] for batch in batches] == [2, 2]
    assert concatenations == []


def test_batch_stream_stores_prepared_index_lists() -> None:
    prepared = _Prepared(
        inputs=torch.ones((2, 9), dtype=torch.int32),
        labels=torch.ones((2, 9), dtype=torch.int32),
        groups=np.array([0, 1, 2]),
        puzzles=np.array([0, 1, 2]),
        identifiers=np.array([1, 2]),
        ignore_label_id=0,
        batch_size=2,
        device=torch.device("cpu"),
        seed=0,
    )
    data = _TestDataSource(prepared=prepared)

    stream = ArcBatches(data, train=True, epochs_per_iter=1)

    assert stream.groups == [0, 1, 2]
    assert stream.puzzles == [0, 1, 2]


@pytest.mark.parametrize("nonresident", ["inputs", "labels"])
def test_batch_stream_rejects_either_nonresident_array(nonresident: str) -> None:
    inputs = torch.ones((4, 9), dtype=torch.int32)
    labels = torch.ones((4, 9), dtype=torch.int32)
    if nonresident == "inputs":
        inputs = np.ones((4, 9), dtype=np.int32)
    else:
        labels = np.ones((4, 9), dtype=np.int32)
    prepared = _Prepared(
        inputs=inputs,
        labels=labels,
        groups=np.array([0, 1, 2]),
        puzzles=np.array([0, 2, 4]),
        identifiers=np.array([1, 2]),
        ignore_label_id=0,
        batch_size=2,
        device=torch.device("cpu"),
        seed=0,
    )
    data = _TestDataSource(prepared=prepared)

    with pytest.raises(
        TypeError,
        match=r"^ARC2 requires device-resident prepared data$",
    ):
        ArcBatches(data, train=True, epochs_per_iter=1)


def test_device_factories_receive_the_resident_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _Prepared(
        inputs=torch.ones((4, 9), dtype=torch.int32),
        labels=torch.ones((4, 9), dtype=torch.int32),
        groups=np.array([0, 1, 2]),
        puzzles=np.array([0, 2, 4]),
        identifiers=np.array([1, 2]),
        ignore_label_id=0,
        batch_size=2,
        device=torch.device("cpu"),
        seed=0,
    )
    data = _TestDataSource(prepared=prepared)
    devices: list[object] = []
    original_tensor = torch.tensor
    original_to = Tensor.to

    def tensor_factory(
        values: list[int],
        *,
        device: torch.device | str | None = None,
    ) -> Tensor:
        if values == [1, 0, 0]:
            devices.append(device)
        return original_tensor(values, device=device)

    def tensor_to(value: Tensor, *args: object, **kwargs: object) -> Tensor:
        assert not kwargs
        assert len(args) == 1
        target = args[0]
        if isinstance(target, torch.dtype):
            return original_to(value, target)
        assert isinstance(target, (torch.device, str))
        devices.append(target)
        return original_to(value, device=target)

    monkeypatch.setattr(torch, "tensor", tensor_factory)
    monkeypatch.setattr(Tensor, "to", tensor_to)

    batch = next(iter(ArcBatches(data, train=False, epochs_per_iter=1)))

    tags = batch["spatial_tags"]
    assert isinstance(tags, Tensor)
    assert tags.device == torch.device("cpu")
    assert devices == [torch.device("cpu"), torch.device("cpu")]


def test_task_limits_and_spec_reach_the_prepared_reader(tmp_path: Path) -> None:
    write_tree(tmp_path)
    spec = ArcSpec(max_grid=9)
    config = Arc2Data.Config(
        working_dir=tmp_path,
        batch_size=2,
        device="meta",
        spec=spec,
        num_tasks=1,
        num_eval_tasks=1,
    )

    data = config.make()

    assert data.prepared.config.spec.max_grid == 9
    assert data.prepared.config.device == "meta"
    assert data.prepared.config.num_tasks == 1
    assert data.prepared.config.num_eval_tasks == 1


@pytest.mark.parametrize("rank", [0, 1])
def test_eval_batches_are_rank_sliced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rank: int,
) -> None:
    write_tree(tmp_path)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_rank", lambda: rank)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    data = Arc2Data.Config(
        working_dir=tmp_path,
        batch_size=2,
        device="cpu",
    ).make()

    stream = data.eval_dataloader()
    assert len(stream) == 1
    batches = list(stream)

    assert len(batches) == 1
    start = rank * 18
    assert torch.equal(
        _tensor(batches[0], "media"),
        (torch.arange(start, start + 18).reshape(2, 9) + 1) % 12,
    )
    assert batches[0]["valid_count"] == 2


def test_checkpoint_accepts_no_active_pass(tmp_path: Path) -> None:
    write_tree(tmp_path)
    data = port_data(tmp_path)
    state = data.state_dict()

    data.load_state_dict(state)

    assert data.state_dict()["active_pass"] is None


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("passes", "bad"),
        ("active_pass", "bad"),
        ("next_batch", "bad"),
        ("timer_epoch", []),
    ],
)
def test_checkpoint_rejects_malformed_fields(
    tmp_path: Path,
    key: str,
    value: object,
) -> None:
    write_tree(tmp_path)
    data = port_data(tmp_path)
    state: dict[str, object] = dict(data.state_dict())
    state[key] = value

    with pytest.raises(TypeError):
        data.load_state_dict(state)


def test_checkpoint_restores_an_existing_live_stream(tmp_path: Path) -> None:
    write_tree(tmp_path)
    source = port_data(tmp_path)
    source_stream = source.train_dataloader()
    source_iterator = iter(source_stream)
    next(source_iterator)
    checkpoint = source.state_dict()
    expected = list(source_iterator)

    restored = port_data(tmp_path)
    live_stream = restored.train_dataloader()
    restored.load_state_dict(checkpoint)

    assert live_stream.passes == 1
    assert live_stream.active_pass == 1
    assert live_stream.next_batch == 1
    assert len(live_stream) == 1
    actual = list(live_stream)
    assert len(actual) == 1
    assert torch.equal(_tensor(actual[0], "media"), _tensor(expected[0], "media"))
    assert torch.equal(_tensor(actual[0], "label"), _tensor(expected[0], "label"))
    assert torch.equal(
        _tensor(actual[0], "puzzle_identifiers"),
        _tensor(expected[0], "puzzle_identifiers"),
    )


def test_checkpoint_keeps_the_active_batch_cursor(tmp_path: Path) -> None:
    write_tree(tmp_path)
    source = port_data(tmp_path)
    stream = source.train_dataloader()
    assert len(stream) == 2
    iterator = iter(stream)
    next(iterator)
    checkpoint = source.state_dict()
    assert checkpoint["passes"] == 1
    assert checkpoint["active_pass"] == 1
    assert checkpoint["next_batch"] == 1
    source.passes = 99
    source.active_pass = 98
    source.next_batch = 97
    checkpoint = source.state_dict()
    assert checkpoint["passes"] == 1
    assert checkpoint["active_pass"] == 1
    assert checkpoint["next_batch"] == 1
    restored = port_data(tmp_path)
    restored.load_state_dict(checkpoint)
    expected = list(iterator)
    actual = list(restored.train_dataloader())
    assert len(expected) == len(actual) == 1
    assert torch.equal(_tensor(expected[0], "media"), _tensor(actual[0], "media"))
    assert restored.state_dict()["active_pass"] is None


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
