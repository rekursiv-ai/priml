"""Tests for the CHARM parquet-backed dataset protocol."""

from __future__ import annotations

from pathlib import Path
from typing import Final, TypedDict, cast

import math

from pyarrow import parquet
from treekle import from_plain

import numpy as np
import pyarrow as pa
import pytest
import torch

from priml.baselines.arcagi import parquet_data
from priml.baselines.arcagi.parquet import (
    ParquetSourceData,
    parse_offline_aug,
    read_parquet_source,
)
from priml.baselines.arcagi.parquet_data import (
    ArcInstanceMetadata,
    ArcParquetData,
    PairSource,
)
from priml.train.custom_types import TrainStepProtocol


_CWD: Final = Path(__file__).resolve().parent


def test_tiny_source_minted_golden_replays_actual_native_runtime(
    tmp_path: Path,
) -> None:
    """Replay source-minted task, instance, pair, and tensor values."""
    fixture = _write(
        tmp_path / "tiny-source-golden.parquet",
        rows=[
            _row(
                "tiny-a",
                train=[
                    _pair([[1, 0], [2, 3]], out=[[4], [5]]),
                    _pair([[6]], out=[[7, 8], [9, 0]]),
                ],
                test=[_pair([[2, 3, 4]], out=[[5, 6]])],
                augs=["", "dih|6||colorperm|0312456789"],
            ),
            _row(
                "tiny-b",
                train=[],
                test=[_pair([[8], [9]], out=[[1, 2, 3]])],
                augs=[""],
            ),
        ],
    )
    golden_path = _CWD / "testdata/charm_data.pt"
    golden = from_plain(
        cast(object, torch.load(golden_path, map_location="cpu", weights_only=True)),
        _PairSourceGolden,
    )
    cfg = ArcParquetData.Config()
    cfg.train_sources = [
        PairSource.Config(path=fixture, pair_selection="both", multiplicity=2),
    ]
    cfg.eval_source = fixture
    cfg.canvas_size = golden["canvas_size"]
    cfg.batch_size = 1
    cfg.eval_batch_size = 1
    cfg.num_workers = 0
    cfg.p_translate = golden["p_translate"]
    data = ArcParquetData(
        cfg.finalize(),
        rng_for_sample=lambda index: np.random.RandomState(
            index + golden["sample_seed_offset"],
        ),
    )

    expected_tasks = {
        item["puzzle_name"]: item["task_id"] for item in golden["task_to_id"]
    }
    expected_instances = {
        tuple(item["unique_str"].split("||", maxsplit=1)): item["instance_id"]
        for item in golden["instance_to_id"]
    }
    expected_metadata: list[ArcInstanceMetadata | None] = [
        None,
    ] * (len(expected_instances) + 1)
    for (puzzle_name, offline_aug), instance_id in expected_instances.items():
        dih_idx, colorperm_slots = parse_offline_aug(offline_aug)
        expected_metadata[instance_id] = ArcInstanceMetadata(
            puzzle_name=puzzle_name,
            task_id=expected_tasks[puzzle_name],
            dih_idx=dih_idx,
            colorperm_slots=colorperm_slots,
        )
    assert data.task_to_id == expected_tasks
    assert data.instance_to_id == expected_instances
    assert data.num_tasks == len(expected_tasks)
    assert data.num_instances == len(expected_instances)
    assert data.instance_metadata == tuple(expected_metadata)
    for index, expected in enumerate(golden["train"]):
        sample = data.train_dataset[index]
        assert sample["canonical_puzzle_name"] == expected["puzzle_name"]
        assert sample["puzzle_idxs"] == expected["task_id"]
        assert sample["puzzle_identifiers"] == expected["instance_id"]
        assert sample["input_metadata"]["pair_index"] == expected["pair_index"]
        assert sample["input_metadata"]["offline_aug"] == expected["offline_aug"]
        assert sample["input_metadata"]["pair_type"] == expected["pair_type"]
        assert torch.equal(
            torch.from_numpy(sample["media"]),
            expected["input"].to(dtype=torch.int32),
        )
        expected_output = expected["output"]
        assert expected_output is not None
        label = sample.get("label")
        assert label is not None
        assert torch.equal(
            torch.from_numpy(label),
            expected_output.to(dtype=torch.int32),
        )
    for index, expected in enumerate(golden["eval"]):
        sample = data.eval_dataset[index]
        assert sample["canonical_puzzle_name"] == expected["puzzle_name"]
        assert sample["puzzle_idxs"] == expected["task_id"]
        assert sample["puzzle_identifiers"] == expected["instance_id"]
        assert sample["input_metadata"]["pair_index"] == expected["pair_index"]
        assert sample["input_metadata"]["offline_aug"] == expected["offline_aug"]
        assert sample["input_metadata"]["pair_type"] == expected["pair_type"]
        assert torch.equal(
            torch.from_numpy(sample["media"]),
            expected["input"].to(dtype=torch.int32),
        )
        assert "label" not in sample


def test_train_eval_ids_metadata_and_demo_separation(tmp_path: Path) -> None:
    """Check paired train/eval IDs and keep eval labels out of batches."""
    path = _write(
        tmp_path / "mixed.parquet",
        rows=[
            _row(
                "task-a",
                train=[
                    _pair([[1, 2], [3, 4]], out=[[5], [6], [7]]),
                    _pair([[8]], out=[[9, 0]]),
                ],
                test=[_pair([[2, 1]], out=[[4, 3]])],
                augs=["", "dih|1", "colorperm|0312456789"],
            ),
            _row(
                "task-b",
                train=[_pair([[1]], out=[[2]])],
                test=[_pair([[3]], out=[[4]])],
                augs=[""],
            ),
        ],
    )
    data = _configured_data(path, eval_path=path)

    train_batches = list(data.train_dataloader())
    train_names = [
        name for batch in train_batches for name in batch["canonical_puzzle_names"]
    ]
    assert len(train_names) == 28
    assert set(train_names) == {"task-a", "task-b"}
    assert {
        task_id
        for batch in train_batches
        for task_id in from_plain(batch["puzzle_idxs"].tolist(), list[int])
    } == {0, 1}
    assert "label" in train_batches[0]
    assert train_batches[0]["media"].dtype == torch.int32
    assert train_batches[0]["label"].dtype == torch.int32
    assert train_batches[0]["spatial_tags"].shape[1] == 3

    eval_batches = list(data.eval_dataloader())
    eval_names = [
        name for batch in eval_batches for name in batch["canonical_puzzle_names"]
    ]
    assert eval_names == ["task-a", "task-a", "task-a", "task-b"]
    assert all("label" not in batch for batch in eval_batches)
    canonical_output = eval_batches[0]["canonical_outputs"][0]
    assert canonical_output is not None
    assert canonical_output.tolist() == [[4, 3]]
    assert from_plain(eval_batches[0]["puzzle_idxs"].tolist(), list[int]) == [0, 0]
    assert eval_batches[0]["puzzle_identifiers"].tolist()[0] == 1
    assert eval_batches[0]["dih_idxs"].tolist() == [0, 1]
    assert data.instance_metadata[1] is not None
    assert data.instance_metadata[1].puzzle_name == "task-a"
    assert data.instance_metadata[1].task_id == 0
    assert data.instance_metadata[2] is not None
    assert data.instance_metadata[2].dih_idx == 1
    assert data.instance_metadata[3] is not None
    assert data.instance_metadata[3].colorperm_slots == (2, 0, 1, 3, 4, 5, 6, 7, 8)


def test_path_order_duplicate_task_identity_and_multiplicity(tmp_path: Path) -> None:
    """Preserve source order, semantic IDs, and appended repetitions."""
    first = _write(
        tmp_path / "first.parquet",
        rows=[
            _row(
                "shared",
                train=[_pair([[1]], out=[[2]])],
                test=[],
                augs=["", "dih|1"],
            ),
        ],
    )
    second = _write(
        tmp_path / "second.parquet",
        rows=[
            _row(
                "shared",
                train=[_pair([[3]], out=[[4]])],
                test=[],
                augs=["", "dih|1"],
            ),
            _row("later", train=[_pair([[5]], out=[[6]])], test=[], augs=[""]),
        ],
    )
    cfg = ArcParquetData.Config()
    cfg.train_sources = [
        PairSource.Config(path=first, pair_selection="train"),
        PairSource.Config(path=second, pair_selection="train", multiplicity=2),
    ]
    cfg.eval_source = first
    cfg.batch_size = 2
    cfg.p_translate = 0.0
    data = cfg.make()
    samples = [data.train_dataset[index] for index in range(len(data.train_dataset))]

    assert [sample["canonical_puzzle_name"] for sample in samples] == [
        "shared",
        "shared",
        "shared",
        "shared",
        "later",
        "shared",
        "shared",
        "later",
    ]
    assert [sample["puzzle_idxs"] for sample in samples[:5]] == [0, 0, 0, 0, 1]
    assert [sample["puzzle_identifiers"] for sample in samples] == [
        1,
        2,
        1,
        2,
        3,
        1,
        2,
        3,
    ]
    assert samples[0]["canonical_input"].tolist() == [[1]]
    assert samples[2]["canonical_input"].tolist() == [[3]]


def test_mixed_source_selection_keeps_test_only_rows_and_offsets(
    tmp_path: Path,
) -> None:
    """Append selected test pairs after demonstrations across source rows."""
    demonstrations = _write(
        tmp_path / "demonstrations.parquet",
        rows=[
            _row(
                "demo-task",
                train=[_pair([[1]], out=[[2]]), _pair([[3]], out=[[4]])],
                test=[],
                augs=[""],
            ),
        ],
    )
    mixed = _write(
        tmp_path / "mixed.parquet",
        rows=[
            _row(
                "prefix-task",
                train=[_pair([[5]], out=[[6]])],
                test=[_pair([[7]], out=[[8]])],
                augs=[""],
            ),
            _row(
                "test-only-task",
                train=[],
                test=[_pair([[8]], out=[[9]]), _pair([[9]], out=[[8]])],
                augs=[""],
            ),
            _row(
                "demo-only-in-both-task",
                train=[_pair([[4]], out=[[5]])],
                test=[],
                augs=[""],
            ),
        ],
    )
    config = ArcParquetData.Config()
    config.train_sources = [
        PairSource.Config(path=demonstrations, pair_selection="train"),
        PairSource.Config(path=mixed, pair_selection="both"),
    ]
    config.eval_source = mixed
    config.p_translate = 0.0
    data = config.make()

    samples = [data.train_dataset[index] for index in range(len(data.train_dataset))]

    assert [sample["canonical_puzzle_name"] for sample in samples] == [
        "demo-task",
        "demo-task",
        "prefix-task",
        "prefix-task",
        "test-only-task",
        "test-only-task",
        "demo-only-in-both-task",
    ]
    assert [sample["canonical_input"].tolist() for sample in samples] == [
        [[1]],
        [[3]],
        [[5]],
        [[7]],
        [[8]],
        [[9]],
        [[4]],
    ]
    assert [sample["input_metadata"]["pair_index"] for sample in samples] == [
        0,
        1,
        0,
        0,
        0,
        1,
        0,
    ]
    assert data.task_to_id == {
        "demo-task": 0,
        "prefix-task": 1,
        "test-only-task": 2,
        "demo-only-in-both-task": 3,
    }


def test_empty_eval_rows_do_not_stop_later_id_registration(tmp_path: Path) -> None:
    """Continue task and view registration after rows without test pairs."""
    train_path = _write(
        tmp_path / "train.parquet",
        rows=[
            _row("train-task", train=[_pair([[1]], out=[[2]])], test=[], augs=[""]),
        ],
    )
    eval_path = _write(
        tmp_path / "eval.parquet",
        rows=[
            _row("empty-first", train=[], test=[], augs=[""]),
            _row("eval-first", train=[], test=[_pair([[3]], out=[[4]])], augs=[""]),
            _row("empty-middle", train=[], test=[], augs=[""]),
            _row("eval-last", train=[], test=[_pair([[5]], out=[[6]])], augs=[""]),
        ],
    )
    config = ArcParquetData.Config()
    config.train_sources = [PairSource.Config(path=train_path)]
    config.eval_source = eval_path
    config.num_workers = 0
    data = config.make()

    samples = [data.eval_dataset[index] for index in range(len(data.eval_dataset))]

    assert [sample["canonical_puzzle_name"] for sample in samples] == [
        "eval-first",
        "eval-last",
    ]
    assert [sample["puzzle_idxs"] for sample in samples] == [1, 2]
    assert [sample["puzzle_identifiers"] for sample in samples] == [2, 3]


def test_sources_and_eval_share_one_parquet_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reuse each parsed parquet source across source selections and evaluation."""
    path = _write(
        tmp_path / "shared.parquet",
        rows=[_row("task", train=[_pair([[1]], out=[[2]])], test=[], augs=[""])],
    )
    original_reader = read_parquet_source
    read_paths: list[Path] = []

    def counted_reader(
        source_path: Path,
        *,
        max_grid_size: int,
    ) -> ParquetSourceData:
        read_paths.append(source_path)
        return original_reader(source_path, max_grid_size=max_grid_size)

    monkeypatch.setattr(parquet_data, "read_parquet_source", counted_reader)
    config = ArcParquetData.Config()
    config.train_sources = [
        PairSource.Config(path=path),
        PairSource.Config(path=path, multiplicity=2),
    ]
    config.eval_source = path

    config.make()

    assert read_paths == [path]


def test_distinct_eval_source_uses_configured_grid_filter(tmp_path: Path) -> None:
    """Apply the configured canvas limit when loading a distinct eval file."""
    train_path = _write(
        tmp_path / "train-small-canvas.parquet",
        rows=[
            _row("train", train=[_pair([[1]], out=[[2]])], test=[], augs=[""]),
        ],
    )
    eval_path = _write(
        tmp_path / "eval-small-canvas.parquet",
        rows=[
            _row("fits", train=[], test=[_pair([[3, 4, 5]], out=[[6]])], augs=[""]),
            _row(
                "too-wide",
                train=[],
                test=[_pair([[1, 2, 3, 4]], out=[[5]])],
                augs=[""],
            ),
        ],
    )
    config = ArcParquetData.Config()
    config.train_sources = [PairSource.Config(path=train_path)]
    config.eval_source = eval_path
    config.canvas_size = 3
    config.eval_batch_size = 2
    config.num_workers = 0
    data = config.make()

    batches = list(data.eval_dataloader())

    assert [batch["valid_count"] for batch in batches] == [1]
    assert batches[0]["canonical_puzzle_names"] == ["fits"]
    assert batches[0]["media"].shape == (2, 9)


def test_registration_skips_rows_with_no_selected_pairs(tmp_path: Path) -> None:
    """Register IDs only for rows that contribute selected training pairs."""
    path = _write(
        tmp_path / "registration.parquet",
        rows=[
            _row("empty", train=[], test=[_pair([[1]], out=[[2]])], augs=[""]),
            _row("live", train=[_pair([[3]], out=[[4]])], test=[], augs=[""]),
        ],
    )
    cfg = ArcParquetData.Config()
    cfg.train_sources = [PairSource.Config(path=path, pair_selection="train")]
    cfg.eval_source = path

    data = cfg.make()

    assert data.task_to_id == {"live": 0, "empty": 1}
    assert data.instance_to_id == {("live", ""): 1, ("empty", ""): 2}


def test_raw_offline_aliases_keep_distinct_instance_ids(tmp_path: Path) -> None:
    """Keep literal offline augmentation aliases as distinct instances."""
    path = _write(
        tmp_path / "aliases.parquet",
        rows=[
            _row("task", train=[_pair([[1]], out=[[2]])], test=[], augs=["", "dih|0"]),
        ],
    )
    cfg = ArcParquetData.Config()
    cfg.train_sources = [PairSource.Config(path=path)]
    cfg.eval_source = path

    data = cfg.make()
    samples = [data.train_dataset[index] for index in range(len(data.train_dataset))]

    assert [sample["puzzle_identifiers"] for sample in samples] == [1, 2]
    assert [sample["dih_idxs"] for sample in samples] == [0, 0]
    assert data.instance_metadata[1] == data.instance_metadata[2]
    assert data.instance_to_id == {("task", ""): 1, ("task", "dih|0"): 2}


def test_selected_targetless_test_fails_with_source_context(tmp_path: Path) -> None:
    """Reject selected targetless training pairs with source context."""
    path = _write(
        tmp_path / "targetless.parquet",
        rows=[
            _row(
                "targetless-task",
                train=[_pair([[1]], out=[[2]])],
                test=[{"input": [[3]], "output": None}],
                augs=[""],
            ),
        ],
    )
    cfg = ArcParquetData.Config()
    cfg.train_sources = [PairSource.Config(path=path, pair_selection="both")]
    cfg.eval_source = path

    with pytest.raises(
        ValueError,
        match=r"targetless\.parquet.*targetless-task.*pair 0",
    ):
        cfg.make()


def test_pair_metadata_retains_task_local_index_after_filtering(
    tmp_path: Path,
) -> None:
    """Retain original task-local pair indexes after grid filtering."""
    oversized = [[1] * 31]
    path = _write(
        tmp_path / "local-index.parquet",
        rows=[
            _row(
                "task",
                train=[
                    _pair(oversized, out=[[2]]),
                    _pair([[3]], out=[[4]]),
                ],
                test=[
                    _pair(oversized, out=[[5]]),
                    _pair([[6]], out=[[7]]),
                ],
                augs=[""],
            ),
        ],
    )
    cfg = ArcParquetData.Config()
    cfg.train_sources = [PairSource.Config(path=path, pair_selection="both")]
    cfg.eval_source = path

    data = cfg.make()

    assert data.train_dataset[0]["input_metadata"]["pair_index"] == 1
    assert data.train_dataset[1]["input_metadata"]["pair_index"] == 1
    assert data.eval_dataset[0]["input_metadata"]["pair_index"] == 1


def test_evaluation_retains_canonical_target_beyond_canvas_width(
    tmp_path: Path,
) -> None:
    """Keep canonical eval targets even when wider than the model canvas."""
    output = [[2] * 256]
    path = _write(
        tmp_path / "wide-eval-output.parquet",
        rows=[_row("task", train=[], test=[_pair([[1]], out=output)], augs=[""])],
    )
    cfg = ArcParquetData.Config()
    cfg.train_sources = [PairSource.Config(path=path)]
    cfg.eval_source = path
    cfg.canvas_size = 30
    cfg.num_workers = 0
    data = cfg.make()

    sample = data.eval_dataset[0]

    assert sample["canonical_output"] is not None
    assert sample["canonical_output"].shape == (1, 256)
    assert sample["canonical_output"].tolist() == output
    assert "label" not in sample


def test_eval_distributed_sampler_keeps_duplicate_prediction_votes(
    tmp_path: Path,
) -> None:
    """Retain DistributedSampler padding duplicates as evaluation votes."""
    path = _write(
        tmp_path / "eval-padding.parquet",
        rows=[
            _row(
                str(index),
                train=[],
                test=[_pair([[index]], out=[[index]])],
                augs=[""],
            )
            for index in range(5)
        ],
    )

    left = _eval_task_ids(path, rank=0)
    right = _eval_task_ids(path, rank=1)

    assert left == [0, 2, 4]
    assert right == [1, 3, 0]
    assert left.count(0) + right.count(0) == 2


def test_rank_shards_follow_distributed_sampler_epochs(tmp_path: Path) -> None:
    """Shard expanded training indexes using seeded sampler epochs."""
    path = _write(
        tmp_path / "rank.parquet",
        rows=[
            _row(
                str(index),
                train=[_pair([[index]], out=[[index]])],
                test=[],
                augs=[""],
            )
            for index in range(5)
        ],
    )

    left_data = _make_rank_data(path, rank=0)
    right_data = _make_rank_data(path, rank=1)
    left = left_data.train_dataloader()
    right = right_data.train_dataloader()
    left_data.train_dataset.set_epoch(4)
    right_data.train_dataset.set_epoch(4)
    left_ids = [
        task_id
        for batch in left
        for task_id in from_plain(batch["puzzle_idxs"].tolist(), list[int])
    ]
    right_ids = [
        task_id
        for batch in right
        for task_id in from_plain(batch["puzzle_idxs"].tolist(), list[int])
    ]
    expected_left = torch.utils.data.DistributedSampler[int](
        left_data.train_dataset,
        num_replicas=2,
        rank=0,
        shuffle=True,
        seed=17,
    )
    expected_right = torch.utils.data.DistributedSampler[int](
        right_data.train_dataset,
        num_replicas=2,
        rank=1,
        shuffle=True,
        seed=17,
    )
    expected_left.set_epoch(4)
    expected_right.set_epoch(4)
    assert left_ids == list(expected_left)
    assert right_ids == list(expected_right)
    assert len(left_ids) == len(right_ids) == 3
    assert sorted(set(left_ids + right_ids)) == [0, 1, 2, 3, 4]
    assert len(left_ids + right_ids) == 6


def test_expected_counts_and_resume_state(tmp_path: Path) -> None:
    """Verify configured counts and epoch-start checkpoint restart behavior."""
    path = _write(
        tmp_path / "state.parquet",
        rows=[_row("one", train=[_pair([[1]], out=[[2]])], test=[], augs=[""])],
    )
    cfg = ArcParquetData.Config()
    cfg.train_sources = [PairSource.Config(path=path, pair_selection="train")]
    cfg.eval_source = path
    cfg.num_tasks = 1
    cfg.num_instances = 1
    cfg.batch_size = 1
    cfg.num_workers = 0
    data = cfg.make()
    train_loader = data.train_dataloader()
    eval_loader = data.eval_dataloader()
    assert data.train_dataset.sampler is train_loader.sampler
    assert data.eval_dataset.sampler is eval_loader.sampler
    data.eval_dataset.set_epoch(4)
    assert data.eval_dataset.sampler is not None
    data.timer_epoch.global_count = 3
    state = data.state_dict()
    resumed = cfg.make()
    resumed.load_state_dict(state)

    assert resumed.timer_epoch.global_count == 3
    assert resumed.state_dict() == state
    first_pass = next(iter(data.train_dataloader()))
    restarted_pass = next(iter(resumed.train_dataloader()))
    assert (
        first_pass["canonical_puzzle_names"] == restarted_pass["canonical_puzzle_names"]
    )
    with pytest.raises(TypeError, match="timer_epoch"):
        resumed.load_state_dict({})

    wrong = ArcParquetData.Config()
    wrong.train_sources = [PairSource.Config(path=path, pair_selection="train")]
    wrong.eval_source = path
    wrong.num_tasks = 2
    with pytest.raises(ValueError, match="num_tasks"):
        wrong.make()


def test_train_tail_eval_rank_padding_and_cpu_collation(tmp_path: Path) -> None:
    """Drop incomplete train tails and track padded eval tails on CPU."""
    path = _write(
        tmp_path / "tails.parquet",
        rows=[
            _row(
                str(index),
                train=[_pair([[index]], out=[[index]])],
                test=[_pair([[index]], out=[[index]])],
                augs=[""],
            )
            for index in range(5)
        ],
    )
    cfg = ArcParquetData.Config()
    cfg.train_sources = [PairSource.Config(path=path, pair_selection="train")]
    cfg.eval_source = path
    cfg.batch_size = 2
    cfg.eval_batch_size = 2
    cfg.num_workers = 0
    data = cfg.make()
    object.__setattr__(data, "device", torch.device("meta"))

    train_batches = list(data.train_dataloader())
    eval_batches = list(data.eval_dataloader())

    assert sum(batch["valid_count"] for batch in train_batches) == 4
    assert train_batches[0]["media"].device.type == "cpu"
    assert "label" in train_batches[0]
    assert train_batches[0]["label"].device.type == "cpu"
    assert [batch["valid_count"] for batch in eval_batches] == [2, 2, 1]
    assert eval_batches[-1]["canonical_puzzle_names"] == ["4"]


@pytest.mark.parametrize("pair_count", [0, 1])
def test_train_loader_rejects_no_complete_rank_local_batch(
    tmp_path: Path,
    pair_count: int,
) -> None:
    """Reject rank-local train indexes with no complete batch."""
    path = _write(
        tmp_path / f"short-{pair_count}.parquet",
        rows=[
            _row(
                "short",
                train=[_pair([[1]], out=[[1]])] if pair_count else [],
                test=[],
                augs=[""],
            ),
        ],
    )
    cfg = ArcParquetData.Config()
    cfg.train_sources = [PairSource.Config(path=path)]
    cfg.eval_source = path
    cfg.batch_size = 2
    cfg.num_workers = 0
    data = cfg.make()

    with pytest.raises(ValueError, match="no complete rank-local training batch"):
        data.train_dataloader()
    assert list(data.eval_dataloader()) == []


def test_bind_step_passes_id_indexed_instance_metadata(tmp_path: Path) -> None:
    """Bind semantic metadata indexed by puzzle instance ID to the step."""
    path = _write(
        tmp_path / "bind.parquet",
        rows=[_row("one", train=[_pair([[1]], out=[[2]])], test=[], augs=[""])],
    )
    cfg = ArcParquetData.Config()
    cfg.train_sources = [PairSource.Config(path=path)]
    cfg.eval_source = path
    data = cfg.make()

    class Receiver:
        metadata: object | None = None

        def bind_instance_metadata(self, metadata: object) -> None:
            self.metadata = metadata

    receiver = Receiver()
    data.bind_step(cast(TrainStepProtocol, receiver))

    assert receiver.metadata is data.instance_metadata


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        ("canvas_size", 0, "Canvas and batch sizes"),
        ("num_workers", -1, "num_workers"),
        ("p_translate", 1.1, "p_translate"),
        ("p_translate", float("nan"), "p_translate"),
        ("rank", 0, "rank and world_size"),
    ],
)
def test_finalize_rejects_invalid_runtime_controls(
    field_name: str,
    value: object,
    message: str,
) -> None:
    """Keep scalar values printable at finalize and reject them before parquet I/O."""
    cfg = ArcParquetData.Config()
    cfg.train_sources = [PairSource.Config(path="train.parquet")]
    cfg.eval_source = "eval.parquet"
    setattr(cfg, field_name, value)

    finalized = cfg.finalize()
    if field_name == "canvas_size":
        assert finalized.canvas_size == value
    elif field_name == "num_workers":
        assert finalized.num_workers == value
    elif field_name == "p_translate":
        if isinstance(value, float) and math.isnan(value):
            assert math.isnan(finalized.p_translate)
        else:
            assert finalized.p_translate == value
    else:
        assert finalized.rank == value

    with pytest.raises(ValueError, match=message):
        finalized.make()


def test_finalize_limits_compact_source_slots_before_io() -> None:
    """Reject unrepresentable source slots in runtime construction before I/O."""
    cfg = ArcParquetData.Config()
    cfg.train_sources = [
        PairSource.Config(path=f"source-{index}.parquet") for index in range(257)
    ]
    cfg.eval_source = "eval.parquet"

    finalized = cfg.finalize()
    with pytest.raises(ValueError, match="At most 256 ordered train sources"):
        finalized.make()


def test_source_and_data_configuration_validation(tmp_path: Path) -> None:
    """Validate source policies and configured runtime counts."""
    invalid_selection = PairSource.Config()
    object.__setattr__(invalid_selection, "pair_selection", "eval")
    invalid_selection.finalize()
    invalid = ArcParquetData.Config()
    invalid.train_sources = [invalid_selection]
    invalid.eval_source = "missing.parquet"
    with pytest.raises(ValueError, match="pair_selection"):
        invalid.make()

    invalid_multiplicity = PairSource.Config()
    invalid_multiplicity.multiplicity = 0
    invalid_multiplicity.finalize()
    invalid = ArcParquetData.Config()
    invalid.train_sources = [invalid_multiplicity]
    invalid.eval_source = "missing.parquet"
    with pytest.raises(ValueError, match="multiplicity"):
        invalid.make()

    empty_config = ArcParquetData.Config().finalize()
    with pytest.raises(ValueError, match="At least one"):
        empty_config.make()

    path = _write(
        tmp_path / "count.parquet",
        rows=[_row("one", train=[_pair([[1]], out=[[2]])], test=[], augs=[""])],
    )
    wrong_instances = ArcParquetData.Config()
    wrong_instances.train_sources = [PairSource.Config(path=path)]
    wrong_instances.eval_source = path
    wrong_instances.num_instances = 2
    with pytest.raises(ValueError, match="num_instances"):
        wrong_instances.make()


def test_eval_only_tasks_join_shared_semantic_id_map(tmp_path: Path) -> None:
    """Assign eval-only tasks IDs after the ordered training task map."""
    train_path = _write(
        tmp_path / "train-only.parquet",
        rows=[_row("train-task", train=[_pair([[1]], out=[[2]])], test=[], augs=[""])],
    )
    eval_path = _write(
        tmp_path / "eval-only.parquet",
        rows=[_row("eval-task", train=[], test=[_pair([[3]], out=[[4]])], augs=[""])],
    )
    cfg = ArcParquetData.Config()
    cfg.train_sources = [PairSource.Config(path=train_path)]
    cfg.eval_source = eval_path
    cfg.eval_batch_size = 1
    cfg.num_workers = 0
    data = cfg.make()

    batches = list(data.eval_dataloader())

    assert data.num_tasks == 2
    assert data.task_to_id == {"train-task": 0, "eval-task": 1}
    assert data.instance_metadata[2] == ArcInstanceMetadata(
        puzzle_name="eval-task",
        task_id=1,
        dih_idx=0,
        colorperm_slots=tuple(range(9)),
    )
    assert batches[0]["puzzle_idxs"].tolist() == [1]


def test_sample_rng_is_injected_per_access(tmp_path: Path) -> None:
    """Use the injected sample RNG factory on each dataset access."""
    path = _write(
        tmp_path / "rng.parquet",
        rows=[
            _row(
                "one",
                train=[_pair([[1]], out=[[2]])],
                test=[_pair([[3]], out=[[4]])],
                augs=[""],
            ),
        ],
    )
    cfg = ArcParquetData.Config()
    cfg.train_sources = [PairSource.Config(path=path)]
    cfg.eval_source = path
    cfg.p_translate = 0.0
    rng_indices: list[int] = []

    def rng_for_sample(index: int) -> np.random.RandomState:
        rng_indices.append(index)
        return np.random.RandomState(index)

    data = ArcParquetData(
        cfg.finalize(),
        rng_for_sample=rng_for_sample,
    )
    data.train_dataset[0]
    data.train_dataset[0]
    data.eval_dataset[0]

    assert rng_indices == [0, 0, 0]


def test_distributed_defaults_and_invalid_rank(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Read initialized process-group coordinates and reject invalid ranks."""
    path = _write(
        tmp_path / "distributed.parquet",
        rows=[_row("one", train=[_pair([[1]], out=[[2]])], test=[], augs=[""])],
    )
    distributed = torch.distributed
    monkeypatch.setattr(distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(distributed, "get_rank", lambda: 1)
    monkeypatch.setattr(distributed, "get_world_size", lambda: 2)
    cfg = ArcParquetData.Config()
    cfg.train_sources = [PairSource.Config(path=path)]
    cfg.eval_source = path
    data = cfg.make()
    assert (data.rank, data.world_size) == (1, 2)

    invalid = ArcParquetData.Config()
    invalid.train_sources = [PairSource.Config(path=path)]
    invalid.eval_source = path
    invalid.rank = 2
    invalid.world_size = 2
    with pytest.raises(ValueError, match="rank must be"):
        invalid.make()


def test_finalize_resolves_every_path_under_base_dir_without_io() -> None:
    """Resolve all source paths under base_dir during I/O-free finalize."""
    cfg = ArcParquetData.Config()
    cfg.base_dir = "/datasets/data-root"
    cfg.train_sources = [
        PairSource.Config(path="/datasets/train.parquet", pair_selection="both"),
    ]
    cfg.eval_source = "/datasets/eval.parquet"
    resolved = cfg.copy_tree().finalize()

    assert resolved.train_sources[0].path == Path(
        "/datasets/data-root/datasets/train.parquet",
    )
    assert resolved.eval_source == Path("/datasets/data-root/datasets/eval.parquet")


def _write(path: Path, *, rows: list[dict[str, object]]) -> Path:
    parquet.write_table(pa.Table.from_pylist(rows), path)
    return path


def _row(
    name: str,
    *,
    train: list[dict[str, object]],
    test: list[dict[str, object]],
    augs: list[str],
) -> dict[str, object]:
    return {
        "format_version": 2,
        "puzzle_id": name,
        "train": train,
        "test": test,
        "offline_augs": augs,
    }


def _pair(inp: list[list[int]], *, out: list[list[int]]) -> dict[str, object]:
    return {"input": inp, "output": out}


class _PairSourceGolden(TypedDict):
    source_revision: str
    canvas_size: int
    p_translate: float
    sample_seed_offset: int
    task_to_id: list[_PairSourceGoldenTask]
    instance_to_id: list[_PairSourceGoldenInstance]
    train: list[_PairSourceGoldenSample]
    eval: list[_PairSourceGoldenSample]


class _PairSourceGoldenTask(TypedDict):
    task_id: int
    puzzle_name: str


class _PairSourceGoldenInstance(TypedDict):
    instance_id: int
    unique_str: str


class _PairSourceGoldenSample(TypedDict):
    index: int
    puzzle_name: str
    task_id: int
    instance_id: int
    offline_aug: str
    pair_type: str
    pair_index: int
    input: torch.Tensor
    output: torch.Tensor | None


def _configured_data(train_path: Path, *, eval_path: Path) -> ArcParquetData:
    cfg = ArcParquetData.Config()
    cfg.train_sources = [
        PairSource.Config(path=train_path, pair_selection="train"),
        PairSource.Config(path=train_path, pair_selection="both", multiplicity=2),
    ]
    cfg.eval_source = eval_path
    cfg.batch_size = 2
    cfg.eval_batch_size = 2
    cfg.num_workers = 0
    cfg.p_translate = 0.0
    return cfg.make()


def _eval_task_ids(path: Path, *, rank: int) -> list[int]:
    """Return valid evaluation task ids from one DistributedSampler rank."""
    cfg = ArcParquetData.Config()
    cfg.train_sources = [PairSource.Config(path=path, pair_selection="train")]
    cfg.eval_source = path
    cfg.rank = rank
    cfg.world_size = 2
    cfg.eval_batch_size = 4
    cfg.num_workers = 0
    data = cfg.make()
    return [
        task_id
        for batch in data.eval_dataloader()
        for task_id in from_plain(
            batch["puzzle_idxs"].tolist()[: batch["valid_count"]],
            list[int],
        )
    ]


def _make_rank_data(path: Path, *, rank: int) -> ArcParquetData:
    """Build one seeded training sampler shard for the requested rank."""
    cfg = ArcParquetData.Config()
    cfg.train_sources = [PairSource.Config(path=path, pair_selection="train")]
    cfg.eval_source = path
    cfg.batch_size = 1
    cfg.num_workers = 0
    cfg.rank = rank
    cfg.world_size = 2
    cfg.seed = 17
    cfg.p_translate = 0.0
    return cfg.make()


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
