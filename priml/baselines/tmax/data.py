"""Read TMax rollout JSONL files and yield packed rows, one batch per optimizer update.

A packed row is several complete rollouts concatenated into one sequence
rather than padded to equal length: no compute is spent on padding, and
attention never crosses from one rollout into the next.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import replace
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, Self, cast, override, runtime_checkable

import json

from configgle import Fig

import torch

from priml.baselines.tmax.rollouts import (
    AdvantageNormalization,
    PackedRow,
    RolloutRecord,
    group_advantages,
    pack_rollouts,
    parse_json_integer,
    parse_json_integer_array,
    read_rollout_records,
)
from priml.paths import resolve_working_dir
from priml.runtime import global_device_mesh
from priml.timer import CheckpointableStepTimer


if TYPE_CHECKING:
    from priml.train.custom_types import TrainStepProtocol


@runtime_checkable
class ResolvesPadTokenId(Protocol):
    """A training step that has resolved the pad id the data layer must reuse.

    Scoring replaces this id before gathering labels, so a data layer that
    removed a different id would train on tokens the objective never sees.
    """

    pad_token_id: int


class _RolloutBatches:
    """Re-iterable view of one dataset's recorded rollout updates."""

    def __init__(
        self,
        iterator: Callable[[], Iterator[dict[str, object]]],
    ) -> None:
        self.iterator = iterator

    def __iter__(self) -> Iterator[dict[str, object]]:
        return self.iterator()


def data_parallel_shape() -> tuple[int, int]:
    """Return ``(rank, world_size)`` for this process in the ``"dp"`` mesh dimension.

    ``"dp"`` is both FullySharded's default shard dimension and the loop's
    data-seed dimension, so the row slice and the gradient reduction always
    use the same data-parallel world. A single-process run has no mesh and
    reports ``(0, 1)``.
    """
    mesh = global_device_mesh()
    if mesh is None or mesh.mesh_dim_names is None or "dp" not in mesh.mesh_dim_names:
        return 0, 1
    dp = mesh["dp"]
    world = int(dp.size())
    if world <= 1:
        return 0, 1
    return int(dp.get_local_rank()), world


class TMaxRolloutData:
    """Read TMax rollout JSONL files and yield packed rows per optimizer update.

    The boundary between collection and training is deliberately just files.
    The source can be a live directory written by TMax's rollout collector,
    or a released trace copied to cluster storage. This class does not create
    prompts, run Harbor, start Docker, or call vLLM.
    """

    class Config(Fig["TMaxRolloutData"]):
        """Rollout file, grouping, and packing configuration."""

        base_dir: Path | str | None = None
        """Root supplied by the experiment."""

        working_dir: Path | str = ""
        """One TMax ``*_rollouts_*.jsonl`` shard or a directory of shards."""

        tool_mask_path: Path | str | None = None
        """Optional separate file with ``tool_mask`` values the shards omit."""

        records_per_update: int = 256
        """Rollouts consumed by one DPPO update, 8 prompts x 32 samples."""

        num_samples_per_prompt: int = 32
        """Group size used by the centered reward baseline."""

        advantage_normalization: AdvantageNormalization = "centered"
        """Reward normalization from the released 4B recipe."""

        pack_length: int = 67_584
        """Maximum prompt-plus-response tokens per packed row."""

        pad_token_id: int | None = None
        """Id removed from packed responses; prompts holding it are rejected.

        ``None`` falls back to the id the bound training step resolved from
        its checkpoint, so packing and scoring always remove the same token.
        """

        mask_tool_use: bool = True
        """Exclude sandbox-written tokens from the policy objective."""

        filter_zero_std_samples: bool = True
        """Drop a whole prompt group whose rewards have zero spread.

        Upstream's ``accumulate_inference_batches`` refuses such a group
        (``np.std(reward_scores) == 0``): when every sample in a group
        solves the task or every one fails, all centered advantages are
        zero, so those tokens carry no learning signal. Upstream's default
        is on and the released launcher leaves it on.
        """

        active_sampling: bool = True
        """A dropped group does not use up one of the update's prompt slots.

        The released launcher passes ``--active_sampling``: the accumulation
        loop keeps drawing fresh groups until the update again holds
        ``records_per_update // num_samples_per_prompt`` accepted groups.
        Without it, upstream trains the update on whatever groups survived
        one window. Upstream's library default is off; this offline
        default follows the released recipe.
        """

        min_num_batches: int = 1
        """Minimum packed rows per update.

        At packing time, the effective minimum is raised to the data-parallel
        world size so every rank receives at least one row, matching upstream's
        use of ``dp_world_size`` in its packing pass.
        """

        eval_records: int = 0
        """Number of records exposed by the optional final evaluation loader."""

        @override
        def finalize(self) -> Self:
            """Resolve the rollout paths before building the data source."""
            if not str(self.working_dir):
                raise ValueError("TMax working_dir must name a JSONL file.")
            self.working_dir = resolve_working_dir(self.base_dir, self.working_dir)
            if self.tool_mask_path is not None:
                self.tool_mask_path = resolve_working_dir(
                    self.base_dir,
                    self.tool_mask_path,
                )
            return super().finalize()

    def __init__(self, config: Config) -> None:
        """Validate the configuration and record this rank's data-parallel slot."""
        if not config.working_dir:
            raise ValueError("TMax working_dir must name a JSONL file.")
        if config.records_per_update <= 0:
            raise ValueError("records_per_update must be positive.")
        if config.num_samples_per_prompt <= 0:
            raise ValueError("num_samples_per_prompt must be positive.")
        if config.records_per_update % config.num_samples_per_prompt:
            raise ValueError(
                "records_per_update must be divisible by num_samples_per_prompt.",
            )
        # Upstream's two validation rules, mirrored: active sampling exists
        # only to refill slots the filter empties, and a group of one always
        # has zero reward spread, so filtering it would drop every group.
        if config.active_sampling and not config.filter_zero_std_samples:
            raise ValueError(
                "active_sampling needs filter_zero_std_samples; with nothing "
                "filtered, no slot is ever refilled.",
            )
        if config.filter_zero_std_samples and config.num_samples_per_prompt == 1:
            raise ValueError(
                "filter_zero_std_samples needs num_samples_per_prompt > 1; a "
                "group of one always has zero reward spread.",
            )
        if config.pack_length <= 0:
            raise ValueError("pack_length must be positive.")
        self.config = config
        self.timer_epoch = CheckpointableStepTimer()
        self._position = 0
        self._dp_rank, self._dp_world = data_parallel_shape()
        self._step_pad_token_id: int | None = None
        self._records_cache: list[RolloutRecord] | None = None

    def bind_step(self, step: TrainStepProtocol) -> None:
        """Use the bound step's pad id so packing and scoring remove one id."""
        if isinstance(step, ResolvesPadTokenId):
            step_pad_token_id = int(step.pad_token_id)
            if self.config.pad_token_id not in (None, step_pad_token_id):
                raise ValueError(
                    "dataset.pad_token_id must match the training step's "
                    f"resolved pad_token_id ({step_pad_token_id}).",
                )
            self._step_pad_token_id = step_pad_token_id

    def train_dataloader(self) -> Iterable[dict[str, object]]:
        """Return a replayable stream of this rank's packed rollout updates."""
        return _RolloutBatches(self._iter_train_batches)

    def _iter_train_batches(self) -> Iterator[dict[str, object]]:
        """Yield one pass, preserving the read position until the pass completes."""
        records = self._records()
        while self._position < len(records):
            batch, consumed = self._compose_update(records, self._position)
            self._position = consumed
            yield self._pack_update(batch)
        self._position = 0

    def pack_update(self, records: Sequence[RolloutRecord]) -> dict[str, object]:
        """Pack one already-built update for the bound data-parallel rank.

        Live collection uses the same packing step as replaying files. It
        broadcasts records, not tensors, so every rank runs this deterministic
        conversion locally and the update's denominator stays global.
        """
        return self._pack_update(records)

    def _pack_update(self, batch: Sequence[RolloutRecord]) -> dict[str, object]:
        """Compute advantages and pack one accepted rollout update."""
        rewards = [record.reward for record in batch]
        advantages = group_advantages(
            rewards,
            num_samples_per_prompt=self.config.num_samples_per_prompt,
            normalization=self.config.advantage_normalization,
        )
        rows = pack_rollouts(
            batch,
            advantages=advantages,
            pack_length=self.config.pack_length,
            pad_token_id=self._pad_token_id(),
            mask_tool_use=self.config.mask_tool_use,
            min_num_batches=max(self.config.min_num_batches, self._dp_world),
        )
        if not rows:
            raise ValueError("A non-empty rollout update packed into zero rows.")
        return {
            "rows": tuple(_row_mapping(row) for row in self._shard_rows(rows)),
            # The update's GLOBAL denominator: every rank packs the identical
            # update before slicing, so this count needs no cross-rank
            # communication.
            "response_token_count": _response_token_count(rows),
        }

    def _compose_update(
        self,
        records: Sequence[RolloutRecord],
        position: int,
    ) -> tuple[list[RolloutRecord], int]:
        """Build one update using upstream's group filters.

        The rollout data is a stream of prompt groups. Upstream's
        accumulation loop drops a whole group whose rewards have zero
        spread; with ``active_sampling`` (the released launcher's setting)
        the emptied slot is refilled from the stream, so the update always
        holds ``records_per_update`` accepted records. Without it, the
        update trains on the survivors of exactly one window of
        ``records_per_update`` input records. Dropped groups still advance
        the consumed count, so the saved position moves past them.

        Args:
          records: The cached rollout records.
          position: First unconsumed record index.

        Returns:
          batch: The update's accepted records, whole prompt groups.
          consumed: Record index after the update's last consumed group.

        Raises:
          ValueError: The rollout file ended mid-group or before the
            update's slots were filled, or every group in the window was
            dropped.

        """
        config = self.config
        group_size = config.num_samples_per_prompt
        slots = config.records_per_update // group_size
        accepted: list[RolloutRecord] = []
        consumed = position
        slots_seen = 0
        while True:
            group = records[consumed : consumed + group_size]
            if not group:
                raise ValueError(
                    "Rollout file ended before one update's prompt slots "
                    f"were filled; accepted {len(accepted)} of "
                    f"{config.records_per_update} records.",
                )
            if len(group) != group_size:
                raise ValueError(
                    f"Rollout file ended with {len(group)} records; expected "
                    f"{group_size} to complete a prompt group.",
                )
            consumed += group_size
            slots_seen += 1
            dead = config.filter_zero_std_samples and _zero_spread(group)
            if config.active_sampling:
                if dead:
                    continue
                accepted.extend(group)
                if len(accepted) == config.records_per_update:
                    break
            else:
                if not dead:
                    accepted.extend(group)
                # The window always consumes exactly this many groups,
                # whether or not they survived; a dropped group's slot is
                # left empty, not refilled.
                if slots_seen == slots:
                    break
        if not accepted:
            raise ValueError(
                "Every prompt group in the update window had zero reward "
                "spread; the update has nothing to train on.",
            )
        return accepted, consumed

    def eval_dataloader(self) -> Iterator[dict[str, object]]:
        """Yield the whole eval slice on every rank.

        Never filtered or refilled: upstream's in-training evaluation runs
        the accumulation loop with both the zero-spread filter and active
        sampling off, so eval numbers stay the same no matter what the
        training filter is set to.
        """
        if self.config.eval_records <= 0:
            return
        records = self._records()[: self.config.eval_records]
        if len(records) % self.config.num_samples_per_prompt:
            raise ValueError("eval_records must contain whole prompt groups.")
        advantages = group_advantages(
            [record.reward for record in records],
            num_samples_per_prompt=self.config.num_samples_per_prompt,
            normalization=self.config.advantage_normalization,
        )
        rows = pack_rollouts(
            records,
            advantages=advantages,
            pack_length=self.config.pack_length,
            pad_token_id=self._pad_token_id(),
            mask_tool_use=self.config.mask_tool_use,
            min_num_batches=self.config.min_num_batches,
        )
        # Not rank-sliced: TrainLoop does not reduce evaluation outputs across
        # ranks, so every rank must see the complete slice rather than a
        # data-parallel subset.
        yield {
            "rows": tuple(_row_mapping(row) for row in rows),
            "response_token_count": _response_token_count(rows),
        }

    def _shard_rows(self, rows: list[PackedRow]) -> list[PackedRow]:
        """Slice this rank's contiguous share of the update's packed rows.

        Contiguous, like upstream's worker split
        (``data_loader.prepare_collated_data_for_workers``). Both row counts
        and scored segment counts must match: FSDP runs a collective for
        each segment's forward pass. Padding adds no loss.
        """
        if self._dp_world <= 1:
            return rows
        if len(rows) < self._dp_world:
            # Identical on every rank (deterministic packing of identical
            # files), so raising together cannot desynchronize collectives.
            raise ValueError(
                f"{len(rows)} packed rows cannot cover a data-parallel world of "
                f"{self._dp_world}; raise records_per_update or min_num_batches.",
            )
        base, extra = divmod(len(rows), self._dp_world)
        rows_per_rank = base + (1 if extra else 0)
        shards = []
        for rank in range(self._dp_world):
            start = rank * base + min(rank, extra)
            stop = start + base + (1 if rank < extra else 0)
            shard = rows[start:stop]
            while len(shard) < rows_per_rank:
                shard.append(_zero_loss_row(shard[0]))
            shards.append(shard)
        return [
            _pad_scoring_segments(
                row,
                max(_scoring_segments(shard[index]) for shard in shards),
            )
            for index, row in enumerate(shards[self._dp_rank])
        ]

    def _pad_token_id(self) -> int:
        """Resolve the packing pad id: explicit, the bound step's, then 0."""
        if self.config.pad_token_id is not None:
            return self.config.pad_token_id
        if self._step_pad_token_id is not None:
            return self._step_pad_token_id
        return 0  # A synthetic run without a bound step has no tokenizer.

    def _records(self) -> list[RolloutRecord]:
        """Read shards once per process, then apply the optional mask file.

        Cached at first read on purpose: every rank must pack byte-identical
        updates, and a live collector's growing directory would leave the
        ranks reading different data mid-run. A shard that appears later is
        picked up on restart.
        """
        if self._records_cache is not None:
            return self._records_cache
        path = Path(self.config.working_dir)
        if path.is_dir():
            paths = sorted(path.glob("*_rollouts_*.jsonl"))
        else:
            paths = [path]
        if not paths:
            raise FileNotFoundError(f"No TMax rollout shards under {path}.")
        records = [record for shard in paths for record in read_rollout_records(shard)]
        masks = _read_masks(self.config.tool_mask_path)
        if masks:
            records = [
                replace(record, tool_mask=masks.get((record.step, record.sample_idx)))
                if record.tool_mask is None
                else record
                for record in records
            ]
        _validate_artifact(
            records,
            group_size=self.config.num_samples_per_prompt,
        )
        self._records_cache = records
        return records

    def state_dict(self) -> dict[str, object]:
        """Save the read position and epoch timer."""
        return {
            "position": self._position,
            "timer_epoch": self.timer_epoch.state_dict(),
        }

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        """Restore the read position and epoch timer."""
        self._position = int(cast(int, state_dict.get("position", 0)))
        timer = state_dict.get("timer_epoch")
        if isinstance(timer, Mapping):
            self.timer_epoch.load_state_dict(cast(Mapping[str, object], timer))


def _row_mapping(row: PackedRow) -> dict[str, object]:
    """Expose the native TMax field names to ``TMaxDPPOTrainStep``."""
    return {
        "query_responses": row.query_responses,
        "attention_mask": row.attention_mask,
        "position_ids": row.position_ids,
        "response_mask": row.response_mask,
        "prompt_mask": row.prompt_mask,
        "rollout_sample_ids": row.rollout_sample_ids,
        "model_steps": row.model_steps,
        "vllm_logprobs": row.vllm_logprobs,
        "advantages": row.advantages,
    }


def _zero_loss_row(row: PackedRow) -> PackedRow:
    """Return a padding row: keeps collective counts equal and adds no loss."""
    return replace(
        row,
        response_mask=row.response_mask.new_zeros(row.response_mask.shape),
        advantages=row.advantages.new_zeros(row.advantages.shape),
        num_actions=0,
    )


def _scoring_segments(row: PackedRow) -> int:
    """Count segments long enough to call the model for next-token scoring."""
    return sum(length >= 2 for length in row.packed_seq_lens)


def _pad_scoring_segments(row: PackedRow, count: int) -> PackedRow:
    """Add two-token, no-loss segments until this row makes ``count`` forwards."""
    missing = count - _scoring_segments(row)
    if not missing:
        return row
    source = row.query_responses[:1]
    if len(source) != 1:
        raise ValueError("An empty packed row cannot pad FSDP scoring.")
    segment = int(row.attention_mask[-1])
    return replace(
        row,
        query_responses=torch.cat((row.query_responses, source.repeat(2 * missing))),
        attention_mask=torch.cat(
            (
                row.attention_mask,
                row.attention_mask.new_tensor(
                    [
                        segment + index
                        for index in range(1, missing + 1)
                        for _ in range(2)
                    ],
                ),
            ),
        ),
        position_ids=torch.cat(
            (row.position_ids, row.position_ids.new_tensor([0, 1] * missing)),
        ),
        response_mask=torch.cat(
            (row.response_mask, row.response_mask.new_zeros(2 * missing)),
        ),
        prompt_mask=torch.cat(
            (row.prompt_mask, row.prompt_mask.new_ones(2 * missing)),
        ),
        rollout_sample_ids=torch.cat(
            (
                row.rollout_sample_ids,
                row.rollout_sample_ids.new_full((2 * missing,), -1),
            ),
        ),
        model_steps=torch.cat(
            (row.model_steps, row.model_steps.new_full((2 * missing,), -1)),
        ),
        vllm_logprobs=torch.cat(
            (
                row.vllm_logprobs,
                row.vllm_logprobs.new_full((2 * missing,), float("nan")),
            ),
        ),
        advantages=torch.cat(
            (row.advantages, row.advantages.new_zeros(2 * missing)),
        ),
        dones=torch.cat((row.dones, row.dones.new_zeros(2 * missing))),
        packed_seq_lens=row.packed_seq_lens + (2,) * missing,
        original_responses=row.original_responses + ((),) * missing,
    )


def _response_token_count(rows: Sequence[PackedRow]) -> int:
    """Count shifted response tokens across all rows: the update's denominator."""
    return sum(int(row.response_mask[1:].bool().sum()) for row in rows)


def _zero_spread(group: Sequence[RolloutRecord]) -> bool:
    """Report whether every reward in a prompt group is the same.

    Upstream's check is ``np.std(reward_scores) == 0``. PriML compares the
    parsed values directly, so only literally identical rewards are dropped,
    without importing numpy for one comparison.
    """
    first = group[0].reward
    return all(record.reward == first for record in group[1:])


def _read_masks(path: Path | str | None) -> dict[tuple[int, int], tuple[int, ...]]:
    """Read a JSONL mask file keyed by ``(step, sample_idx)``."""
    if path is None:
        return {}
    masks: dict[tuple[int, int], tuple[int, ...]] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = cast(object, json.loads(line))
            if not isinstance(payload, dict):
                raise TypeError(f"{path}:{line_number}: expected a JSON object.")
            payload = cast(dict[str, object], payload)
            try:
                origin = f"{path}:{line_number}"
                key = (
                    parse_json_integer(
                        payload["step"],
                        origin=origin,
                        field="step",
                    ),
                    parse_json_integer(
                        payload["sample_idx"],
                        origin=origin,
                        field="sample_idx",
                    ),
                )
                value = tuple(
                    parse_json_integer_array(
                        payload["tool_mask"],
                        origin=origin,
                        field="tool_mask",
                    ),
                )
            except KeyError as error:
                raise TypeError(
                    f"{path}:{line_number}: invalid tool-mask record.",
                ) from error
            masks[key] = value
    return masks


def _validate_artifact(
    records: Sequence[RolloutRecord],
    *,
    group_size: int,
) -> None:
    """Validate grouping and ordering before any update is produced."""
    seen: set[tuple[int, int]] = set()
    previous_key: tuple[int, int] | None = None
    for start in range(0, len(records), group_size):
        group = records[start : start + group_size]
        location = f"records[{start}:{start + len(group)}]"
        if len(group) != group_size:
            raise ValueError(
                f"{location}: group has {len(group)} records; expected {group_size}.",
            )
        key = (group[0].step, group[0].prompt_idx)
        mismatched = [
            start + offset
            for offset, record in enumerate(group)
            if (record.step, record.prompt_idx) != key
        ]
        if mismatched:
            raise ValueError(
                f"{location}: expected one group {key}, but record "
                f"{mismatched[0]} belongs to "
                f"{(records[mismatched[0]].step, records[mismatched[0]].prompt_idx)}.",
            )
        if key in seen:
            raise ValueError(f"{location}: group {key} recurs after it ended.")
        if previous_key is not None and key <= previous_key:
            raise ValueError(
                f"{location}: group {key} is not after {previous_key} in "
                "lexicographic file order.",
            )
        sample_indices = [record.sample_idx for record in group]
        for offset, (before, after) in enumerate(
            pairwise(sample_indices),
            start=1,
        ):
            if after <= before:
                raise ValueError(
                    f"{location}: group {key} sample_idx at record "
                    f"{start + offset} is {after}, not strictly after {before}.",
                )
        seen.add(key)
        previous_key = key
