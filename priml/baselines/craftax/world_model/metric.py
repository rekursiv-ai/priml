"""Validation bits per byte for the Craftax world model, and its zstd reference.

As in nanochat's ``evaluate_bpb``, the metric sums the NLL of every validation
target and divides by the bytes those targets occupy in a canonical
uncompressed record: 1 byte per cell field, 2 per auxiliary value, 1 for the
action, 2 for the reward, and 1 for done. A frame is 894 bytes and a whole
decision 898. The denominator does not depend on tokenization, so flat and
hierarchical models, and every ablation, are directly comparable.

The metric reads one record per local job, in the canonical record's order:
the action, reward, and done of an ``act`` job, then the frame the job
generates, when it exists. ``craftax_target_nll`` fills it from the model's
own ``target_terms``, so the metric scores exactly what the loss scores.

Sums are kept per modality and per stratum, each record counted by the batch's
``weight`` at its job's position. What the ratios describe depends on the
validation set. Over ``data.EvalSpans`` the weights undo the drawing, so every
ratio, overall and per stratum, estimates the natural distribution. Over the
stratified windows every record counts 1, so the plain ratio describes the
sampled mix, and ``natural_decisions`` only reweights each stratum to its
natural share: within a stratum a long episode's decisions still count less
than a short one's, so ``*_natural`` is then a stratum-reweighted mix, not the
natural distribution. The design's primary metric reads
``nats_per_decision_natural``.

The reference line is zstd -19 over the same canonical records: a model that
does not beat a general-purpose compressor has learned little. Each
micro-batch's records are compressed alone, as the model reads only its own
window: a micro-batch is also the unit ``data.EvalSpans`` hands to ranks, so the
ranks' summed sizes are the same however many ranks there are.
"""

from collections.abc import Mapping
from dataclasses import field
from pathlib import Path
from typing import (
    Final,
    Protocol,
    Self,
    TypedDict,
    override,
    runtime_checkable,
)

import dataclasses
import math

from configgle import Fig, Makeable
from torch import Tensor, nn

import numpy as np
import torch
import torch.distributed as dist

from priml.baselines.craftax.world_model.batch import PackedBatch
from priml.baselines.craftax.world_model.index import (
    FLOORS,
    STRATA,
    Event,
)
from priml.baselines.craftax.world_model.loss import Nll
from priml.baselines.craftax.world_model.model import WorldModelLogits
from priml.baselines.craftax.world_model.schema import (
    FrameSchema,
    craftax_schema,
)
from priml.custom_types import HasNormalizedWorkingDirPattern
from priml.lib import zstd_compat
from priml.lib.codec import from_plain
from priml.paths import resolve_working_dir
from priml.train.custom_types import TrackerProtocol
from priml.train.tracker import unwrap_tracker_config


MODALITIES: Final = ("action", "reward", "done", "board", "hud")
"""Modalities in accumulator order; ``bpb/<modality>`` reports each."""

HEAD_BYTES: Final = {"action": 1, "reward": 2, "done": 1}
"""Canonical bytes of each target an ``act`` job's record starts with."""

SCALAR_BYTES: Final = 2
"""Canonical bytes of one auxiliary value (int16)."""

ZSTD_LEVEL: Final = 19
"""The reference compressor's level, as the plan fixes it."""

NAMED_SERIES: Final = ("total_training_flops", "total_training_time")
"""Training series nanochat logs without the ``train/`` prefix."""


@runtime_checkable
class ScoresTargets(Protocol):
    """A model that scores a packed batch as ``WorldModel`` does, e.g. ``FlatModel``."""

    def logits(self, batch: PackedBatch, /) -> WorldModelLogits:
        """Return action logits per global position and local logits per job."""
        ...

    def target_terms(
        self,
        batch: PackedBatch,
        logits: WorldModelLogits,
        /,
    ) -> dict[str, tuple[Nll, Tensor]]:
        """Return each modality's per-target NLL and scored mask."""
        ...


def craftax_target_nll(model: nn.Module, media: object) -> Tensor:
    """Return each job's record NLL, ``[J·(1 + local_slots)]``.

    Row ``j`` of the ``[J, 1 + local_slots]`` view is job ``j``'s record: the
    NLL of its own action, predicted from the ``obs`` position before it, then
    of each local slot. Every entry comes from the model's ``target_terms`` and
    is 0 where unscored, but for a padding job's action: it sits at position 0
    and reads that position's own action NLL, scored when its window opens
    mid-episode, so only an ``act`` job's action counts (``counted_nll``).

    Args:
      model: A ``ScoresTargets`` model.
      media: The ``PackedBatch`` it scores.

    Returns:
      nll: Flat per-target NLL in nats, the evaluation output
        ``CraftaxBitsPerByte`` consumes.

    """
    assert isinstance(model, ScoresTargets)
    assert isinstance(media, PackedBatch)
    terms = model.target_terms(media, model.logits(media))
    scored = {
        name: torch.where(mask, value.nll, 0.0) for name, (value, mask) in terms.items()
    }
    jobs = len(media.job_at)
    # A real start job's preceding position is never scored, so its action reads
    # 0; a padding job at position 0 reads position 0's, as the docstring says.
    action = scored.pop("action").flatten()[(media.job_at.long() - 1).clamp(min=0)]
    local = [value.reshape(jobs, -1) for value in scored.values()]
    return torch.cat([action[:, None], *local], dim=-1).flatten()


def canonical_records(batch: PackedBatch) -> bytes:
    """Return the canonical bytes of every scored target, in job order.

    Per ``act`` job: the action (uint8), reward (int16), done (uint8), then the
    next frame when it exists. Per ``start`` job: the first frame. A frame is
    its 99×8 cell values (uint8) then its 51 auxiliary values (int16), all
    little-endian.

    Args:
      batch: A packed Craftax micro-batch on the CPU.

    Returns:
      data: As many bytes as the metric's denominator counts for ``batch``.

    """
    following = batch.job_next.clamp(min=0).long()
    head = np.zeros((len(batch.job_at), 4), dtype=np.uint8)
    head[:, 0] = batch.action.flatten()[batch.job_at.long()].numpy()
    head[:, 1:3] = _little(batch.job_reward[:, None])
    head[:, 3] = batch.job_done.numpy()
    frame = np.concatenate(
        [batch.cells[following].flatten(1).numpy(), _little(batch.aux[following])],
        axis=1,
    )
    has_head, has_frame = (part.numpy()[:, None] for part in _record_parts(batch))
    keep = np.concatenate(
        [
            np.repeat(has_head, head.shape[1], axis=1),
            np.broadcast_to(has_frame, frame.shape),
        ],
        axis=1,
    )
    return bytes(np.concatenate([head, frame], axis=1)[keep])


def job_weights(media: PackedBatch, weight: Tensor) -> Tensor:
    """Return what each job's record counts: ``weight`` read at the job's position.

    Args:
      media: The packed micro-batch.
      weight: What every position's decision counts, float ``[B, t_g]``, on
        ``media``'s device.

    Returns:
      job_weight: One weight per job, ``[J]``.

    """
    return weight.flatten()[media.job_at.long()]


def counted_nll(nll: Tensor, *, media: PackedBatch, weight: Tensor) -> Tensor:
    """Return the NLL ``CraftaxBitsPerByte`` counts: every record at its job's weight.

    ``craftax_target_nll`` is 0 wherever a target goes unscored, but a padding
    job sits at position 0 and reads that position's action NLL, which counts
    only for an ``act`` job; so the action column counts for ``act`` jobs alone.

    Args:
      nll: ``craftax_target_nll`` of ``media``.
      media: The packed micro-batch.
      weight: What every position's decision counts, float ``[B, t_g]``, on
        ``media``'s device.

    Returns:
      nll: The weighted sum, a 0-dim tensor.

    """
    records = nll.view(len(media.job_at), -1)
    has_head, _ = _record_parts(media)
    per_job = records[:, 1:].sum(-1) + torch.where(has_head, records[:, 0], 0.0)
    return (per_job * job_weights(media, weight)).sum()


class CraftaxBitsPerByte:
    """Validation bits per byte per modality and stratum, lower being better.

    Consumes ``craftax_target_nll`` as its ``logits``, with the batch's
    ``media``, ``stratum``, and ``weight``.
    """

    class Config(Fig["CraftaxBitsPerByte"]):
        """The local slot layout of the scored jobs."""

        schema: FrameSchema = field(default_factory=craftax_schema)
        """Local slot layout of the scored jobs."""

    def __init__(self, config: Config) -> None:
        self.schema = config.schema
        self.natural_decisions = torch.zeros(0, dtype=torch.float64)
        """Decisions per stratum in the natural validation distribution.

        ``WorldModelLoop`` sets it from the stream's ``eval_sampler.counts``;
        empty skips the ``*_natural`` keys."""
        self.modality, self.width = _record_columns(config.schema)
        self.reference = torch.zeros(2, dtype=torch.float64)
        """Compressed and raw bytes of the canonical records of this process's
        first evaluation, each micro-batch compressed alone; the validation
        windows are fixed, so later evaluations repeat it."""
        self.measure_reference = True
        """Whether ``update`` compresses its records into ``reference``; on until
        the first ``compute``. zstd -19 over a large tile set's records takes
        minutes, so a caller that needs no reference turns it off."""
        self.reset()

    def reset(self) -> None:
        """Zero the sums; the zstd reference survives, as the records repeat."""
        self.nats = torch.zeros(len(MODALITIES), STRATA, dtype=torch.float64)
        self.bytes = torch.zeros(len(MODALITIES), STRATA, dtype=torch.float64)
        self.decisions = torch.zeros(STRATA, dtype=torch.float64)

    def update(self, logits: Tensor, **batch: object) -> None:
        """Accumulate one micro-batch under each job's decision stratum and weight.

        Args:
          logits: ``craftax_target_nll`` of the batch, in nats.
          **batch: ``media`` (``PackedBatch``), ``stratum`` (int64 ``[B, t_g]``),
            and optionally ``weight`` (float ``[B, t_g]``), what each job's record
            counts, read at the job's position; without it every record counts 1.

        """
        media, stratum = batch["media"], batch["stratum"]
        assert isinstance(media, PackedBatch)
        assert isinstance(stratum, Tensor)
        media = media.to(torch.device("cpu"), non_blocking=False)
        weight = batch.get("weight", torch.ones_like(stratum, dtype=torch.float64))
        assert isinstance(weight, Tensor)
        job_weight = job_weights(media, weight.cpu()).double()
        # Unscored history leaves both the sums and the zstd reference.
        counted = job_weight > 0
        nll = logits.detach().double().cpu().view(len(counted), -1)[counted]
        media, job_weight = _select_jobs(media, counted), job_weight[counted]
        has_head, has_frame = _record_parts(media)
        heads = 1 + len(self.schema.prefix_names)
        keep = torch.cat(
            [
                has_head[:, None].expand(-1, heads),
                has_frame[:, None].expand(-1, self.schema.frame_slots),
            ],
            dim=-1,
        )
        job_stratum = stratum.cpu().flatten()[media.job_at.long()]
        cell = (self.modality * STRATA + job_stratum[:, None]).flatten()
        scored = torch.where(keep, nll, 0.0) * job_weight[:, None]
        self.nats.view(-1).index_add_(0, cell, scored.flatten())
        size = torch.where(keep, self.width, 0.0) * job_weight[:, None]
        self.bytes.view(-1).index_add_(0, cell, size.flatten())
        self.decisions.index_add_(0, job_stratum, has_head.double() * job_weight)
        records = canonical_records(media) if self.measure_reference else b""
        if records:
            compressed = len(zstd_compat.compress(records, level=ZSTD_LEVEL))
            self.reference += torch.tensor([compressed, len(records)])

    def compute(self) -> dict[str, float]:
        """Return bits per byte overall, per modality, and per stratum.

        Returns:
          metrics: ``bpb``, ``bpb/<modality>``, ``bpb/floor_<k>``, ``bpb/<event>``,
            ``nats_per_decision``, ``zstd19_bpb``, and, with natural counts,
            ``bpb_natural`` and ``nats_per_decision_natural``.

        Raises:
          ValueError: Nothing was scored, which would otherwise report zero,
            the best possible value.

        """
        nats, size, decisions, reference = _all_reduce(
            self.nats,
            self.bytes,
            self.decisions,
            self.reference,
        )
        if size.sum() <= 0:
            raise ValueError("Craftax bits per byte has no scored targets.")
        self.measure_reference = False
        bits = math.log(2)
        result = {"bpb": _ratio(nats, size, scale=bits)}
        for index, name in enumerate(MODALITIES):
            result[f"bpb/{name}"] = _ratio(nats[index], size[index], scale=bits)
        grid = (FLOORS, len(Event))
        stratum_nats, stratum_bytes = nats.sum(0).view(grid), size.sum(0).view(grid)
        for floor in range(FLOORS):
            if stratum_bytes[floor].sum() > 0:
                result[f"bpb/floor_{floor}"] = _ratio(
                    stratum_nats[floor],
                    stratum_bytes[floor],
                    scale=bits,
                )
        for event in Event:
            if stratum_bytes[:, event].sum() > 0:
                result[f"bpb/{event.name.lower()}"] = _ratio(
                    stratum_nats[:, event],
                    stratum_bytes[:, event],
                    scale=bits,
                )
        result["nats_per_decision"] = _ratio(nats, decisions, scale=1.0)
        if reference[1] > 0:
            result["zstd19_bpb"] = float(8 * reference[0] / reference[1])
        if len(self.natural_decisions):
            natural = self.natural_decisions
            weight = torch.where(decisions > 0, natural / decisions, 0.0)
            weighted = nats.sum(0) * weight
            result["bpb_natural"] = _ratio(weighted, size.sum(0) * weight, scale=bits)
            result["nats_per_decision_natural"] = _ratio(
                weighted,
                decisions * weight,
                scale=1.0,
            )
        return result

    class StateDict(TypedDict):
        """The accumulated sums, flattened to lists of floats."""

        nats: list[float]
        bytes: list[float]
        decisions: list[float]

    # The zstd reference is not state: it describes the records this process
    # scores, and a restored run may score others (an evaluation-only run on
    # another window length), so each process measures its own once.
    def state_dict(self) -> StateDict:
        """Return the accumulated sums."""
        return {
            "nats": from_plain(self.nats.flatten().tolist(), list[float]),
            "bytes": from_plain(self.bytes.flatten().tolist(), list[float]),
            "decisions": from_plain(self.decisions.tolist(), list[float]),
        }

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        """Restore state produced by :meth:`state_dict`.

        Args:
          state_dict: The sums, as :meth:`state_dict` wrote them.

        """
        nats, size, decisions = (
            torch.tensor(
                from_plain(state_dict[name], list[float]),
                dtype=torch.float64,
            )
            for name in ("nats", "bytes", "decisions")
        )
        self.nats = nats.view(len(MODALITIES), STRATA)
        self.bytes = size.view(len(MODALITIES), STRATA)
        self.decisions = decisions


class NanochatSeries:
    """Rename the loop's series to the names nanochat's ``base_train.py`` logs.

    ``train/total_training_flops`` and ``train/total_training_time`` lose their
    prefix, and ``eval/val_<key>`` (the metric named ``val``) becomes
    ``val/<key>``; the same metric over the training split's spans,
    ``eval/train_split_val_<key>``, becomes ``train_split/<key>``, beside it.
    Everything else passes through unchanged.
    """

    class Config(Fig["NanochatSeries"]):
        """The tracker receiving the renamed series."""

        tracker: Makeable[TrackerProtocol] | None = None
        """Child tracker."""

        base_dir: Path | str | None = None
        """Owner directory supplied during parent finalization."""

        working_dir: Path | str = "/"
        """Logical root inherited by the child tracker."""

        @override
        def finalize(self) -> Self:
            self.working_dir = resolve_working_dir(self.base_dir, self.working_dir)
            # Beneath an asynchronous wrapper, as ``TrackerList`` reaches it.
            child = (
                None if self.tracker is None else unwrap_tracker_config(self.tracker)
            )
            if (
                isinstance(child, HasNormalizedWorkingDirPattern)
                and child.base_dir is None
            ):
                child.base_dir = self.working_dir
            return super().finalize()

    def __init__(self, config: Config) -> None:
        if config.tracker is None:
            raise ValueError("NanochatSeries requires a child tracker config.")
        self.tracker = config.tracker.make()

    def log_metrics(
        self,
        metrics: Mapping[str, object],
        step: int,
        *,
        prefix: str = "",
    ) -> None:
        """Forward ``metrics`` under nanochat's names.

        Args:
          metrics: Metric name to value.
          step: Global step.
          prefix: The loop's prefix, ``train/`` or ``eval/``.

        """
        groups: dict[str, dict[str, object]] = {}
        for key, value in metrics.items():
            renamed, name = _rename(prefix, key)
            groups.setdefault(renamed, {})[name] = value
        for renamed, group in groups.items():
            self.tracker.log_metrics(group, step, prefix=renamed)

    def log_images(self, key: str, images: list[object], step: int) -> None:
        """Forward images unchanged."""
        self.tracker.log_images(key, images, step)

    def log_notes(self, notes: str) -> None:
        """Forward run notes unchanged."""
        self.tracker.log_notes(notes)

    def close(self) -> None:
        """Close the child tracker."""
        self.tracker.close()


def _rename(prefix: str, key: str) -> tuple[str, str]:
    """Return the nanochat prefix and name of one series."""
    if prefix == "train/" and key in NAMED_SERIES:
        return "", key
    if prefix == "eval/" and key.startswith("val_"):
        return "val/", key.removeprefix("val_")
    if prefix == "eval/" and key.startswith("train_split_val_"):
        return "train_split/", key.removeprefix("train_split_val_")
    return prefix, key


def _select_jobs(batch: PackedBatch, keep: Tensor) -> PackedBatch:
    """Return ``batch`` with only the jobs ``keep`` marks."""
    return dataclasses.replace(
        batch,
        **{
            field.name: getattr(batch, field.name)[keep]
            for field in dataclasses.fields(batch)
            if field.name.startswith("job_")
        },
    )


def _record_parts(batch: PackedBatch) -> tuple[Tensor, Tensor]:
    """Return which jobs' records hold a head (``act`` jobs) and a next frame."""
    return ~batch.job_is_start, batch.job_next >= 0


def _record_columns(schema: FrameSchema) -> tuple[Tensor, Tensor]:
    """Return each record column's modality index and canonical byte width."""
    heads = ["action", *schema.prefix_names]
    board, hud = MODALITIES.index("board"), MODALITIES.index("hud")
    modality = [MODALITIES.index(name) for name in heads]
    modality += [board] * schema.cell_slots + [hud] * len(schema.scalar_ranges)
    width = [HEAD_BYTES[name] for name in heads]
    width += [len(schema.cell_fields)] * schema.cell_slots
    width += [SCALAR_BYTES] * len(schema.scalar_ranges)
    return torch.tensor(modality), torch.tensor(width, dtype=torch.float64)


def _little(values: Tensor) -> np.ndarray:
    """Return ``[N, K]`` ``values`` as little-endian int16 bytes, ``[N, 2K]``."""
    return values.numpy().astype("<i2").view(np.uint8)


def _ratio(nats: Tensor, size: Tensor, *, scale: float) -> float:
    """Return ``sum(nats) / (scale · sum(size))``."""
    return float(nats.sum() / (scale * size.sum()))


def _all_reduce(*tensors: Tensor) -> list[Tensor]:
    """Sum float64 tensors across ranks; unchanged in a single process."""
    if not (dist.is_available() and dist.is_initialized()):
        return list(tensors)
    flat = torch.cat([t.flatten() for t in tensors])
    if dist.get_backend() != "gloo":
        flat = flat.to(torch.device("cuda", torch.cuda.current_device()))
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    parts = flat.cpu().split([t.numel() for t in tensors])
    return [part.view(t.shape) for part, t in zip(parts, tensors, strict=True)]
