"""Tests for the TMax rollout data source."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import dataclasses
import functools
import json
import tempfile

from torch import Tensor

import pytest
import torch

from priml import runtime
from priml.baselines.tmax import data as data_module
from priml.baselines.tmax.data import TMaxRolloutData, _read_masks, _row_mapping
from priml.baselines.tmax.rollouts import (
    RolloutRecord,
    pack_rollouts,
    read_rollout_records,
)
from priml.baselines.tmax.train_step import TMaxDPPOTrainStep, tiny_qwen35_config
from priml.train.parallelism import FullySharded, NoParallel


if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh

    from priml.baselines.tmax.rollouts import PackedRow
    from priml.distributed.testing import WarmPoolGetter
    from priml.train.custom_types import TrainStepProtocol


ROOT = Path(__file__).parent
SMOKE_FIXTURE = ROOT / "testdata" / "smoke_rollout.jsonl"


def test_data_cadence_reads_a_real_published_rollout() -> None:
    """Read and pack one real record from the published rollout archive."""
    config = TMaxRolloutData.Config()
    config.working_dir = ROOT / "testdata" / "recorded_rollout.jsonl"
    config.records_per_update = 1
    config.num_samples_per_prompt = 1
    config.mask_tool_use = False
    # A group of one always has zero spread, so upstream forbids filtering
    # there; this test exercises the plain path, with no filtering or
    # refilling.
    config.filter_zero_std_samples = False
    config.active_sampling = False
    data = config.make()
    batch = next(iter(data.train_dataloader()))
    rows = cast(tuple[Mapping[str, object], ...], batch["rows"])
    assert len(rows) == 1
    row = rows[0]
    query_responses = cast(Tensor, row["query_responses"])
    assert query_responses.numel() == 2_854
    # The yielded batch carries the update-wide denominator the step's loss
    # divides by.
    assert batch["response_token_count"] == int(
        cast(Tensor, row["response_mask"])[1:].bool().sum(),
    )


def _shard_worker(result_dir: str, smoke_path: str, mesh: DeviceMesh) -> None:
    """Worker: verify this rank's contiguous slice, shared denominator, refusal."""
    result_path = Path(result_dir)
    rank = mesh.get_rank()
    try:
        runtime._device_mesh = mesh
        smoke = TMaxRolloutData.Config()
        smoke.working_dir = smoke_path
        smoke.records_per_update = 2
        smoke.num_samples_per_prompt = 2
        smoke.mask_tool_use = False
        smoke.pack_length = 10  # one short rollout per row: two rows total
        batch = next(iter(smoke.make().train_dataloader()))
        rows = cast(tuple[Mapping[str, object], ...], batch["rows"])
        expected_tail = (3274, 2923, 539, 2923) if rank == 0 else (9764, 728, 3418, 279)
        sliced_ok = (
            len(rows) == 1
            and batch["response_token_count"] == 8
            and tuple(cast(Tensor, rows[0]["query_responses"])[-4:].tolist())
            == expected_tail
        )
        starved = TMaxRolloutData.Config()
        starved.working_dir = ROOT / "testdata" / "recorded_rollout.jsonl"
        starved.records_per_update = 1
        starved.num_samples_per_prompt = 1
        starved.mask_tool_use = False
        starved.filter_zero_std_samples = False
        starved.active_sampling = False
        try:
            next(iter(starved.make().train_dataloader()))
            refusal = "no-refusal"
        except ValueError as error:
            refusal = "refused" if "data-parallel world" in str(error) else repr(error)
        ok = sliced_ok and refusal == "refused"
        (result_path / f"rank_{rank}").write_text(
            "ok" if ok else f"FAIL:rows={len(rows)} refusal={refusal}",
        )
    except Exception as error:  # noqa: BLE001 -- The worker must serialize every subprocess failure for the parent assertion.
        (result_path / f"rank_{rank}").write_text(f"FAIL:{error!r}")
    finally:
        runtime._device_mesh = None


@pytest.mark.cli_python_subprocess
def test_rows_are_sharded_across_the_dp_world(warm_pools: WarmPoolGetter) -> None:
    """Each rank trains a disjoint contiguous slice of the same packed update."""
    pool = warm_pools({"dp": 2})
    with tempfile.TemporaryDirectory() as tmp:
        pool(functools.partial(_shard_worker, tmp, str(SMOKE_FIXTURE)))
        results = {p.name: p.read_text() for p in Path(tmp).iterdir() if p.is_file()}
    assert results == {"rank_0": "ok", "rank_1": "ok"}, results


def _smoke() -> TMaxRolloutData.Config:
    """Two records, unmasked: the cheapest complete update."""
    config = TMaxRolloutData.Config()
    config.working_dir = SMOKE_FIXTURE
    config.records_per_update = 2
    config.num_samples_per_prompt = 2
    config.mask_tool_use = False
    return config


def test_working_dir_is_required() -> None:
    """Require a rollout file to train on."""
    with pytest.raises(ValueError, match="working_dir"):
        TMaxRolloutData.Config().make()


def test_records_per_update_must_be_positive() -> None:
    """Reject an update with no records."""
    config = _smoke()
    config.records_per_update = 0
    with pytest.raises(ValueError, match="records_per_update must be positive"):
        config.make()


@pytest.mark.parametrize("value", [0, -2])
def test_num_samples_per_prompt_must_be_positive(value: int) -> None:
    """Reject a group size that cannot describe a prompt group."""
    config = _smoke()
    config.num_samples_per_prompt = value
    with pytest.raises(ValueError, match="num_samples_per_prompt must be positive"):
        config.make()


def test_records_per_update_must_be_a_whole_number_of_groups() -> None:
    """Require each update to contain whole prompt groups."""
    config = _smoke()
    config.records_per_update = 3
    with pytest.raises(ValueError, match="divisible"):
        config.make()


def test_pack_length_must_be_positive() -> None:
    """Reject a token budget that cannot hold a rollout."""
    config = _smoke()
    config.pack_length = 0
    with pytest.raises(ValueError, match="pack_length must be positive"):
        config.make()


def test_bind_step_adopts_the_steps_pad_id() -> None:
    """Use the bound step's checkpoint-resolved padding id."""
    data = _smoke().make()
    data.bind_step(cast("TrainStepProtocol", SimpleNamespace(pad_token_id=7)))
    assert data._pad_token_id() == 7


def test_the_pad_id_uses_the_explicit_setting_without_a_bound_step() -> None:
    """Use an explicit padding id without building a training step."""
    config = _smoke()
    config.pad_token_id = 5
    assert config.make()._pad_token_id() == 5


def test_bind_step_rejects_a_different_explicit_pad_id() -> None:
    """Reject different padding ids for packing and scoring."""
    config = _smoke()
    config.pad_token_id = 5
    data = config.make()
    with pytest.raises(ValueError, match=r"must match.*resolved"):
        data.bind_step(cast("TrainStepProtocol", SimpleNamespace(pad_token_id=7)))


def test_the_same_loader_rewinds_for_the_next_epoch() -> None:
    """Replay the same loader object because TrainLoop keeps it across epochs."""
    data = _smoke().make()
    loader = data.train_dataloader()
    first = list(loader)
    second = list(loader)
    assert len(first) == len(second) == 1
    for before, after in zip(
        cast(tuple[Mapping[str, object], ...], first[0]["rows"]),
        cast(tuple[Mapping[str, object], ...], second[0]["rows"]),
        strict=True,
    ):
        assert torch.equal(
            cast(Tensor, before["query_responses"]),
            cast(Tensor, after["query_responses"]),
        )


def test_an_incomplete_final_update_is_refused() -> None:
    """Reject a trailing partial update instead of training it."""
    config = TMaxRolloutData.Config()
    config.working_dir = ROOT / "testdata" / "recorded_rollout.jsonl"
    config.records_per_update = 2
    config.num_samples_per_prompt = 1
    config.mask_tool_use = False
    config.filter_zero_std_samples = False
    config.active_sampling = False
    with pytest.raises(ValueError, match="prompt slots"):
        next(iter(config.make().train_dataloader()))


def _composed_record(index: int, reward: float) -> RolloutRecord:
    """One minimal rollout whose response token identifies it by index."""
    return RolloutRecord(
        step=1,
        sample_idx=index,
        prompt_idx=index // 2,
        prompt_tokens=(1,),
        response_tokens=(100 + index,),
        logprobs=(-0.5,),
        reward=reward,
        finish_reason="stop",
        tool_mask=(1,),
    )


def _write_rollouts(path: Path, records: list[RolloutRecord]) -> None:
    """Write a one-shard rollout file the data layer's glob picks up."""
    path.write_text(
        "".join(json.dumps(dataclasses.asdict(record)) + "\n" for record in records),
        encoding="utf-8",
    )


def _batch_records(batch: Mapping[str, object]) -> list[int]:
    """Return the response token of each one-record row, in row order.

    Requires a config with ``pack_length=1``, which gives every rollout its
    own row, so membership reads straight off the row's last token.
    """
    return [
        int(cast(Tensor, row["query_responses"])[-1].item())
        for row in cast(tuple[Mapping[str, object], ...], batch["rows"])
    ]


def test_a_zero_spread_group_is_dropped_and_its_slot_refilled(
    tmp_path: Path,
) -> None:
    """The released recipe's behavior: dropped groups never reach the update."""
    path = tmp_path / "composed_rollouts_000.jsonl"
    _write_rollouts(
        path,
        [
            _composed_record(0, 1.0),
            _composed_record(1, 1.0),  # zero spread: dropped, slot refilled
            _composed_record(2, 0.0),
            _composed_record(3, 1.0),
            _composed_record(4, 0.0),
            _composed_record(5, 1.0),
        ],
    )
    config = TMaxRolloutData.Config()
    config.working_dir = path
    config.records_per_update = 4
    config.num_samples_per_prompt = 2
    config.pack_length = 1
    data = config.make()
    updates = list(data.train_dataloader())
    assert len(updates) == 1
    assert _batch_records(updates[0]) == [102, 103, 104, 105]


def test_without_active_sampling_a_dropped_group_wastes_its_slot(
    tmp_path: Path,
) -> None:
    """Leave a filtered group's slot empty when active sampling is disabled."""
    path = tmp_path / "composed_rollouts_000.jsonl"
    _write_rollouts(
        path,
        [
            _composed_record(0, 1.0),
            _composed_record(1, 1.0),  # zero spread: the window loses a slot
            _composed_record(2, 0.0),
            _composed_record(3, 1.0),
            _composed_record(4, 0.0),
            _composed_record(5, 1.0),
            _composed_record(6, 1.0),
            _composed_record(7, 1.0),  # zero spread: window two loses a slot
        ],
    )
    config = TMaxRolloutData.Config()
    config.working_dir = path
    config.records_per_update = 4
    config.num_samples_per_prompt = 2
    config.pack_length = 1
    config.active_sampling = False
    data = config.make()
    updates = list(data.train_dataloader())
    # Two windows: the first trains on its one surviving group, the second
    # on the group the refill would otherwise have drawn early.
    rows_per_update = [
        len(cast(tuple[Mapping[str, object]], update["rows"])) for update in updates
    ]
    assert rows_per_update == [2, 2]
    assert _batch_records(updates[0]) == [102, 103]
    assert _batch_records(updates[1]) == [104, 105]


def test_a_window_of_only_zero_spread_groups_is_refused(tmp_path: Path) -> None:
    """Reject a window whose groups all have zero reward spread."""
    path = tmp_path / "composed_rollouts_000.jsonl"
    _write_rollouts(path, [_composed_record(0, 1.0), _composed_record(1, 1.0)])
    config = TMaxRolloutData.Config()
    config.working_dir = path
    config.records_per_update = 2
    config.num_samples_per_prompt = 2
    config.pack_length = 1
    config.active_sampling = False
    with pytest.raises(ValueError, match="nothing to train on"):
        next(iter(config.make().train_dataloader()))


def test_active_sampling_running_out_of_artifact_is_loud(tmp_path: Path) -> None:
    """Offline there is nothing to refill from: the file ends, and that must raise."""
    path = tmp_path / "composed_rollouts_000.jsonl"
    _write_rollouts(
        path,
        [_composed_record(0, 1.0), _composed_record(1, 1.0)],
    )
    config = TMaxRolloutData.Config()
    config.working_dir = path
    config.records_per_update = 2
    config.num_samples_per_prompt = 2
    config.pack_length = 1
    with pytest.raises(ValueError, match="prompt slots"):
        next(iter(config.make().train_dataloader()))


def test_an_artifact_ending_mid_group_is_refused(tmp_path: Path) -> None:
    """Treat a rollout file ending mid-group as corrupt."""
    path = tmp_path / "composed_rollouts_000.jsonl"
    _write_rollouts(
        path,
        [
            _composed_record(0, 0.0),
            _composed_record(1, 1.0),
            _composed_record(2, 0.0),
        ],
    )
    config = TMaxRolloutData.Config()
    config.working_dir = path
    config.records_per_update = 2
    config.num_samples_per_prompt = 2
    config.pack_length = 1
    config.filter_zero_std_samples = False
    config.active_sampling = False
    data = config.make()
    updates = iter(data.train_dataloader())
    with pytest.raises(ValueError, match=r"group has 1 records; expected 2"):
        next(updates)


def test_filtering_a_group_of_one_is_refused() -> None:
    """Upstream forbids the filter at group size one; so does this data layer."""
    config = _smoke()
    config.num_samples_per_prompt = 1
    config.records_per_update = 1
    with pytest.raises(ValueError, match="group of one"):
        config.make()


def test_active_sampling_without_filtering_is_refused() -> None:
    """Require filtering when active sampling is enabled."""
    config = _smoke()
    config.filter_zero_std_samples = False
    with pytest.raises(ValueError, match="no slot is ever refilled"):
        config.make()


def test_the_eval_slice_is_never_composed(tmp_path: Path) -> None:
    """Upstream evaluates with both filters off; eval does the same."""
    path = tmp_path / "composed_rollouts_000.jsonl"
    _write_rollouts(
        path,
        [
            _composed_record(0, 1.0),
            _composed_record(1, 1.0),  # zero spread: eval keeps it anyway
            _composed_record(2, 0.0),
            _composed_record(3, 1.0),
        ],
    )
    config = TMaxRolloutData.Config()
    config.working_dir = path
    config.records_per_update = 2
    config.num_samples_per_prompt = 2
    config.pack_length = 1
    config.eval_records = 4
    batch = next(iter(config.make().eval_dataloader()))
    assert _batch_records(batch) == [100, 101, 102, 103]


def test_eval_dataloader_is_empty_by_default() -> None:
    """Yield no evaluation batches when no records are requested."""
    assert list(_smoke().make().eval_dataloader()) == []


def test_eval_dataloader_yields_the_requested_records() -> None:
    """Pack the requested evaluation records with the training packing settings."""
    config = _smoke()
    config.eval_records = 2
    batch = next(iter(config.make().eval_dataloader()))
    assert cast(tuple[Mapping[str, object], ...], batch["rows"])
    assert cast(int, batch["response_token_count"]) > 0


def test_eval_records_must_be_whole_groups() -> None:
    """Require whole prompt groups for evaluation."""
    config = _smoke()
    config.eval_records = 1
    with pytest.raises(ValueError, match="whole prompt groups"):
        next(iter(config.make().eval_dataloader()))


def _packed_rows(count: int) -> list[PackedRow]:
    """Return ``count`` distinct one-rollout rows, in order.

    ``pack_length=1`` gives every rollout its own row, so row ``index`` carries
    response token ``10 + index`` and a slice can be read by value.
    """
    return pack_rollouts(
        [
            RolloutRecord(
                step=1,
                sample_idx=index,
                prompt_idx=0,
                prompt_tokens=(1,),
                response_tokens=(10 + index,),
                logprobs=(-0.1,),
                reward=1.0,
                finish_reason="stop",
                tool_mask=(1,),
            )
            for index in range(count)
        ],
        advantages=torch.ones(count),
        pack_length=1,
        pad_token_id=0,
        mask_tool_use=False,
    )


def test_rows_are_sliced_contiguously_across_the_data_parallel_world(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rank 1 of 3 takes its contiguous slice, including one remainder row."""
    monkeypatch.setattr(data_module, "data_parallel_shape", lambda: (1, 3))
    data = _smoke().make()
    sliced = data._shard_rows(_packed_rows(8))
    assert [row.query_responses.tolist() for row in sliced] == [
        [1, 13],
        [1, 14],
        [1, 15],
    ]


def test_row_shards_are_padded_to_equal_collective_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pad short ranks with zero-loss rows to align collectives."""
    monkeypatch.setattr(data_module, "data_parallel_shape", lambda: (2, 3))
    data = _smoke().make()
    sliced = data._shard_rows(_packed_rows(8))

    assert len(sliced) == 3
    assert [row.query_responses.tolist() for row in sliced[:2]] == [[1, 16], [1, 17]]
    assert not sliced[2].response_mask.any()
    assert not sliced[2].advantages.any()


def test_shards_make_the_same_number_of_segment_forwards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Equal row counts alone do not equalize FSDP model calls."""
    records = [
        RolloutRecord(
            step=0,
            sample_idx=index,
            prompt_idx=index // 2,
            prompt_tokens=(1,),
            response_tokens=(2,) * (length - 1),
            logprobs=(-0.2,) * (length - 1),
            reward=float(index % 2),
            finish_reason="stop",
            tool_mask=(1,) * (length - 1),
        )
        for index, length in enumerate((2, 2, 2, 6))
    ]
    rows = pack_rollouts(
        records,
        advantages=torch.tensor([1.0, -1.0, 1.0, -1.0]),
        pack_length=6,
        pad_token_id=0,
        min_num_batches=2,
    )
    assert [row.packed_seq_lens for row in rows] == [(2, 2, 2), (6,)]
    shards: list[list[PackedRow]] = []
    for rank in (0, 1):
        monkeypatch.setattr(
            data_module,
            "data_parallel_shape",
            lambda rank=rank: (rank, 2),
        )
        shards.append(_smoke().make()._shard_rows(rows))

    assert [len(shard) for shard in shards] == [1, 1]
    assert [
        sum(length >= 2 for length in shard[0].packed_seq_lens) for shard in shards
    ] == [3, 3]
    for original, shard in zip(rows, shards, strict=True):
        size = len(original.query_responses)
        torch.testing.assert_close(
            shard[0].query_responses[:size],
            original.query_responses,
        )
        torch.testing.assert_close(
            shard[0].response_mask[:size],
            original.response_mask,
        )
        assert not shard[0].response_mask[size:].any()
        assert not shard[0].advantages[size:].any()

    config = TMaxDPPOTrainStep.Config()
    config.model = tiny_qwen35_config()
    config.parallelism = NoParallel.Config(device="cpu")
    config.dtype_autocast = None
    step = config.make()
    token_count = 8.0
    original_loss = step.train_loss(
        rows=tuple(_row_mapping(row) for row in rows),
        response_token_count=token_count,
    )["loss"]
    sharded_loss = sum(
        (
            step.train_loss(
                rows=tuple(_row_mapping(row) for row in shard),
                response_token_count=token_count,
            )["loss"]
            for shard in shards
        ),
        torch.zeros_like(original_loss),
    )
    torch.testing.assert_close(sharded_loss, original_loss, rtol=1e-6, atol=1e-7)


def test_single_token_row_can_pad_an_fsdp_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rank with no scorable real segment still joins the other rank."""
    records = [
        RolloutRecord(
            step=0,
            sample_idx=index,
            prompt_idx=0,
            prompt_tokens=(1,),
            response_tokens=(2,) if index == 0 else (),
            logprobs=(-0.2,) if index == 0 else (),
            reward=float(index),
            finish_reason="stop",
            tool_mask=(1,) if index == 0 else (),
        )
        for index in range(2)
    ]
    rows = pack_rollouts(
        records,
        advantages=torch.tensor([1.0, -1.0]),
        pack_length=2,
        pad_token_id=0,
        min_num_batches=2,
    )
    assert [row.packed_seq_lens for row in rows] == [(2,), (1,)]
    monkeypatch.setattr(data_module, "data_parallel_shape", lambda: (1, 2))
    padded = _smoke().make()._shard_rows(rows)[0]
    assert padded.packed_seq_lens == (1, 2)
    assert padded.query_responses.tolist() == [1, 1, 1]
    assert not padded.response_mask.any()


def _uneven_segment_fsdp_worker(result_dir: str, mesh: DeviceMesh) -> None:
    """Run one real FSDP update with three segments on rank 0 and one on rank 1."""
    rank = mesh.get_rank()
    try:
        runtime._device_mesh = mesh
        torch.manual_seed(0)
        config = TMaxDPPOTrainStep.Config()
        config.model = tiny_qwen35_config()
        config.parallelism = FullySharded.Config()
        config.dtype_autocast = None
        step = config.make()
        data_config = TMaxRolloutData.Config()
        data_config.working_dir = SMOKE_FIXTURE
        data_config.records_per_update = 4
        data_config.num_samples_per_prompt = 2
        data_config.mask_tool_use = False
        data_config.pack_length = 6
        records = [
            RolloutRecord(
                step=0,
                sample_idx=index,
                prompt_idx=index // 2,
                prompt_tokens=(1,),
                response_tokens=(2,) * (length - 1),
                logprobs=(-0.2,) * (length - 1),
                reward=float(index % 2),
                finish_reason="stop",
                tool_mask=(1,) * (length - 1),
            )
            for index, length in enumerate((2, 2, 2, 6))
        ]
        batch = data_config.make().pack_update(records)
        result = step.train_step(**step.preprocess_batch(batch))
        ok = (
            batch["response_token_count"] == 8
            and step.global_step == 1
            and bool(torch.isfinite(result["loss"]))
        )
        Path(result_dir, f"rank_{rank}").write_text("ok" if ok else "mismatch")
    except Exception as error:  # noqa: BLE001 -- Serialize worker failures.
        Path(result_dir, f"rank_{rank}").write_text(f"FAIL:{error!r}")
    finally:
        runtime._device_mesh = None


@pytest.mark.compute_distributed
def test_uneven_segments_complete_two_rank_fsdp_update(
    warm_pools: WarmPoolGetter,
) -> None:
    """Uneven packing must not stall FSDP's per-segment collectives."""
    pool = warm_pools({"dp": 2})
    with tempfile.TemporaryDirectory() as result_dir:
        pool(functools.partial(_uneven_segment_fsdp_worker, result_dir))
        results = {
            path.name: path.read_text()
            for path in Path(result_dir).iterdir()
            if path.is_file()
        }
    assert results == {"rank_0": "ok", "rank_1": "ok"}, results


def test_shard_rows_refuses_a_world_it_cannot_cover(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject a row set that cannot provide work to every rank."""
    monkeypatch.setattr(data_module, "data_parallel_shape", lambda: (0, 3))
    data = _smoke().make()
    with pytest.raises(ValueError, match="cannot cover a data-parallel world"):
        data._shard_rows(_packed_rows(2))


class _FakeMeshDim:
    """Stand-in for one device-mesh dimension."""

    def __init__(self, *, size: int, rank: int) -> None:
        """Store the fake data-parallel size and rank."""
        self._size = size
        self._rank = rank

    def size(self) -> int:
        """Return the fake mesh size."""
        return self._size

    def get_local_rank(self) -> int:
        """Return the fake local rank."""
        return self._rank


class _FakeMesh:
    """Stand-in for a device mesh carrying one ``dp`` dimension."""

    mesh_dim_names = ("dp",)

    def __init__(self, *, size: int, rank: int) -> None:
        """Build the fake data-parallel mesh dimension."""
        self._dim = _FakeMeshDim(size=size, rank=rank)

    def __getitem__(self, name: str) -> _FakeMeshDim:
        """Return the fake data-parallel dimension."""
        assert name == "dp"
        return self._dim


def test_a_multi_rank_mesh_reports_this_ranks_dp_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Read this process's data-parallel slot from the global mesh."""
    monkeypatch.setattr(
        data_module,
        "global_device_mesh",
        lambda: _FakeMesh(size=2, rank=1),
    )
    assert data_module.data_parallel_shape() == (1, 2)


def test_a_single_rank_mesh_collapses_to_one_world(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Treat a one-rank mesh as an undistributed world."""
    monkeypatch.setattr(
        data_module,
        "global_device_mesh",
        lambda: _FakeMesh(size=1, rank=0),
    )
    assert data_module.data_parallel_shape() == (0, 1)


def test_a_directory_of_shards_is_globbed(tmp_path: Path) -> None:
    """Read every matching rollout shard from a configured directory."""
    shard = tmp_path / "smoke_rollouts_000.jsonl"
    shard.write_text(
        SMOKE_FIXTURE.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    config = _smoke()
    config.working_dir = tmp_path
    batch = next(iter(config.make().train_dataloader()))
    assert cast(tuple[Mapping[str, object], ...], batch["rows"])


def test_an_empty_rollout_directory_is_refused(tmp_path: Path) -> None:
    """Treat an empty rollout directory as missing data."""
    config = _smoke()
    config.working_dir = tmp_path
    with pytest.raises(FileNotFoundError, match="No TMax rollout shards"):
        next(iter(config.make().train_dataloader()))


def test_a_tool_mask_sidecar_supplies_the_omitted_masks(tmp_path: Path) -> None:
    """Supply masks missing from older released records through a sidecar.

    Live ``exp000`` records include ``tool_mask`` directly; the published
    fixture predates that field.
    """
    record = read_rollout_records(ROOT / "testdata" / "recorded_rollout.jsonl")[0]
    sidecar = tmp_path / "masks.jsonl"
    sidecar.write_text(
        json.dumps(
            {
                "step": record.step,
                "sample_idx": record.sample_idx,
                "tool_mask": [1] * len(record.response_tokens),
            },
        )
        + "\n",
        encoding="utf-8",
    )
    config = TMaxRolloutData.Config()
    config.working_dir = ROOT / "testdata" / "recorded_rollout.jsonl"
    config.tool_mask_path = sidecar
    config.records_per_update = 1
    config.num_samples_per_prompt = 1
    config.mask_tool_use = True
    config.filter_zero_std_samples = False
    config.active_sampling = False
    batch = next(iter(config.make().train_dataloader()))
    assert cast(tuple[Mapping[str, object], ...], batch["rows"])


def test_state_round_trips_the_cursor_and_timer() -> None:
    """Restore the consumed-record position and step timer on resume."""
    data = _smoke().make()
    data._position = 1
    state = data.state_dict()
    restored = _smoke().make()
    restored.load_state_dict(state)
    assert restored._position == 1
    assert restored.timer_epoch.state_dict() == data.timer_epoch.state_dict()


def test_a_degenerate_record_is_refused_rather_than_packing_to_nothing(
    tmp_path: Path,
) -> None:
    """A token-free rollout must not silently become an empty update."""
    path = tmp_path / "empty_rollouts_000.jsonl"
    path.write_text(
        json.dumps(
            {
                "step": 1,
                "sample_idx": 0,
                "prompt_idx": 0,
                "prompt_tokens": [],
                "response_tokens": [],
                "logprobs": [],
                "reward": 1.0,
                "finish_reason": "stop",
                "tool_mask": [],
            },
        )
        + "\n",
        encoding="utf-8",
    )
    config = TMaxRolloutData.Config()
    config.working_dir = path
    config.records_per_update = 1
    config.num_samples_per_prompt = 1
    config.mask_tool_use = False
    config.filter_zero_std_samples = False
    config.active_sampling = False
    with pytest.raises(ValueError, match="packed into zero rows"):
        next(iter(config.make().train_dataloader()))


def test_a_tool_mask_sidecar_rejects_a_non_object_line(tmp_path: Path) -> None:
    """Reject a mask-file line that is not a JSON object."""
    path = tmp_path / "masks.jsonl"
    path.write_text("[1, 2]\n", encoding="utf-8")
    with pytest.raises(TypeError, match="expected a JSON object"):
        _read_masks(path)


def test_a_tool_mask_sidecar_rejects_an_incomplete_record(tmp_path: Path) -> None:
    """Reject a mask record missing its sample index or mask."""
    path = tmp_path / "masks.jsonl"
    path.write_text('{"step": 1}\n', encoding="utf-8")
    with pytest.raises(TypeError, match="invalid tool-mask record"):
        _read_masks(path)


def test_a_tool_mask_sidecar_skips_blank_lines(tmp_path: Path) -> None:
    """Ignore blank lines in a tool-mask file."""
    path = tmp_path / "masks.jsonl"
    path.write_text(
        '\n{"step": 1, "sample_idx": 2, "tool_mask": [1, 0]}\n\n',
        encoding="utf-8",
    )
    assert _read_masks(path) == {(1, 2): (1, 0)}


@pytest.mark.parametrize("value", [True, 1.5, float("nan")])
def test_a_tool_mask_sidecar_strictly_parses_integers(
    tmp_path: Path,
    value: object,
) -> None:
    """Require every tool-mask entry to be an exact integer."""
    path = tmp_path / "masks.jsonl"
    path.write_text(
        json.dumps({"step": 1, "sample_idx": 2, "tool_mask": [value]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"masks.jsonl:1: tool_mask\[0\].*integer"):
        _read_masks(path)


def test_artifact_validation_allows_nonzero_sparse_ids(tmp_path: Path) -> None:
    """Allow ordered rollout ids that are sparse and start above zero."""
    path = tmp_path / "ordered_rollouts_000.jsonl"
    records = [
        dataclasses.replace(
            _composed_record(index, float(index % 2)),
            step=10 + 10 * (index // 2),
            prompt_idx=30 + 10 * (index // 2),
            sample_idx=100 + index * 3,
        )
        for index in range(4)
    ]
    _write_rollouts(path, records)
    config = _smoke()
    config.working_dir = path
    config.records_per_update = 4
    config.filter_zero_std_samples = False
    config.active_sampling = False
    assert next(iter(config.make().train_dataloader()))


@pytest.mark.parametrize(
    ("records", "message"),
    [
        (
            [
                _composed_record(0, 0.0),
                dataclasses.replace(_composed_record(1, 1.0), prompt_idx=1),
            ],
            "expected one group",
        ),
        (
            [
                dataclasses.replace(_composed_record(1, 0.0), sample_idx=5),
                dataclasses.replace(_composed_record(0, 1.0), sample_idx=5),
            ],
            "not strictly after",
        ),
        (
            [
                _composed_record(0, 0.0),
                _composed_record(1, 1.0),
                dataclasses.replace(_composed_record(2, 0.0), prompt_idx=0),
                dataclasses.replace(_composed_record(3, 1.0), prompt_idx=0),
            ],
            "recurs",
        ),
        (
            [
                dataclasses.replace(_composed_record(0, 0.0), prompt_idx=2),
                dataclasses.replace(_composed_record(1, 1.0), prompt_idx=2),
                dataclasses.replace(_composed_record(2, 0.0), prompt_idx=1),
                dataclasses.replace(_composed_record(3, 1.0), prompt_idx=1),
            ],
            "lexicographic",
        ),
    ],
)
def test_artifact_validation_rejects_invalid_group_layout_before_updates(
    tmp_path: Path,
    records: list[RolloutRecord],
    message: str,
) -> None:
    """Reject an invalid prompt-group layout before yielding an update."""
    # Bad group layouts fail validation before any record is consumed.
    path = tmp_path / "invalid_rollouts_000.jsonl"
    _write_rollouts(path, records)
    config = _smoke()
    config.working_dir = path
    config.records_per_update = 2
    config.filter_zero_std_samples = False
    config.active_sampling = False
    data = config.make()
    with pytest.raises(ValueError, match=message):
        next(iter(data.train_dataloader()))
    assert data._position == 0
