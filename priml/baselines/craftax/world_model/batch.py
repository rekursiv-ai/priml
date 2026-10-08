"""Packed training batches: episode segments laid out in fixed global windows.

A window holds ``t_g`` global positions. Each segment contributes an optional
``start`` position (when it begins its episode) and then an ``obs``/``act``
pair per decision. Segments follow each other until the window is full, and
the last one is cut off at ``t_g``. Any unused tail is padding. Every
``start`` and ``act`` position owns a local job that predicts the next frame.
``cu_seqlens`` gives each window ``s_max + 1`` segment slots, the last one for
the padding tail, so its shape never depends on the data.
"""

from collections.abc import Sequence
from typing import cast

import dataclasses
import enum

import torch


class Kind(enum.IntEnum):
    """Global position kinds."""

    START = 0
    OBS = 1
    ACT = 2
    PAD = 3


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Segment:
    """Consecutive decisions of one episode, ready to pack.

    Attributes:
      cells: Cell values of each frame, uint8 ``[F, 99, 8]``. ``F`` is the
        decision count, plus one when the frame after the last decision exists.
      aux: Auxiliary token values of each frame, int16 ``[F, 51]``.
      actions: Executed actions, uint8 ``[N]``.
      reward: Realized rewards, int16 ``[N]``.
      done: Terminal flags, bool ``[N]``.
      starts_episode: Whether decision 0 is the episode's first decision.

    """

    cells: torch.Tensor
    aux: torch.Tensor
    actions: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor
    starts_episode: bool


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class PackedBatch:
    """One micro-batch of ``B`` windows, ``F`` frames, and ``J`` local jobs.

    Attributes:
      kind: Position kind, uint8 ``[B, t_g]``.
      segment: Segment index within the window, int16 ``[B, t_g]``.
      pos: RoPE position within the segment, int32 ``[B, t_g]``.
      cu_seqlens: Segment boundaries of the flattened batch, int32
        ``[B·(s_max + 1) + 1]``.
      frame_of: Frame index of each ``obs`` position, else -1, int32 ``[B, t_g]``.
      action: The game's action at each ``act`` position, else 0, uint8 ``[B, t_g]``.
      cells: Cell values of every referenced frame, uint8 ``[F, 99, 8]``.
      aux: Auxiliary token values of every referenced frame, int16 ``[F, 51]``.
      job_at: Flat global position of each job, int32 ``[J]``.
      job_memory: Frame the job's decoder reads, or -1 for none, int32 ``[J]``.
      job_next: Frame the job generates, or -1 after a terminal, int32 ``[J]``.
      job_reward: Realized reward target, int16 ``[J]``.
      job_done: Terminal target, bool ``[J]``.
      job_is_start: Whether the job belongs to a ``start`` position, bool ``[J]``.

    """

    kind: torch.Tensor
    segment: torch.Tensor
    pos: torch.Tensor
    cu_seqlens: torch.Tensor
    frame_of: torch.Tensor
    action: torch.Tensor
    cells: torch.Tensor
    aux: torch.Tensor
    job_at: torch.Tensor
    job_memory: torch.Tensor
    job_next: torch.Tensor
    job_reward: torch.Tensor
    job_done: torch.Tensor
    job_is_start: torch.Tensor

    def to(self, device: torch.device, *, non_blocking: bool = True) -> "PackedBatch":
        """Return a copy with every tensor on ``device``.

        Args:
          device: Where the tensors go.
          non_blocking: Copy asynchronously where the source allows it.

        Returns:
          batch: The moved copy.

        """
        moved = {
            field.name: cast("torch.Tensor", getattr(self, field.name)).to(
                device,
                non_blocking=non_blocking,
            )
            for field in dataclasses.fields(self)
        }
        return PackedBatch(**moved)


def pack(
    windows: Sequence[Sequence[tuple[Segment, torch.Tensor]]],
    *,
    t_g: int,
    s_max: int,
) -> tuple[PackedBatch, torch.Tensor]:
    """Pack each window's segments into ``t_g`` global positions, with their strata.

    Args:
      windows: Per window, its segments in order, each with the stratum of its
        decisions (at least one per decision). Segments that do not fit, or
        beyond the first ``s_max``, are dropped.
      t_g: Global positions per window.
      s_max: Real segments allowed per window; one more slot holds padding.

    Returns:
      batch: The packed micro-batch.
      stratum: Decision stratum of every position, -1 at padding, int64
        ``[B, t_g]``.

    """
    packer = _Packer(t_g=t_g, s_max=s_max, windows=len(windows))
    for row, parts in enumerate(windows):
        packer.add_window(row, parts)
    return packer.finish()


def pack_windows(
    windows: Sequence[Sequence[Segment]],
    *,
    t_g: int,
    s_max: int,
) -> PackedBatch:
    """Pack each window's segments into ``t_g`` global positions; see ``pack``.

    Args:
      windows: Per window, the segments to lay out in order.
      t_g: Global positions per window.
      s_max: Real segments allowed per window; one more slot holds padding.

    Returns:
      batch: The packed micro-batch.

    """
    batch, _ = pack(
        [
            [(s, torch.zeros(len(s.cells), dtype=torch.int64)) for s in segments]
            for segments in windows
        ],
        t_g=t_g,
        s_max=s_max,
    )
    return batch


class _Packer:
    """Lay out packed windows, one tensor slice per segment."""

    def __init__(self, *, t_g: int, s_max: int, windows: int) -> None:
        self.t_g = t_g
        self.s_max = s_max
        shape = (windows, t_g)
        self.kind = torch.full(shape, int(Kind.PAD), dtype=torch.uint8)
        self.segment = torch.zeros(shape, dtype=torch.int16)
        self.pos = torch.zeros(shape, dtype=torch.int32)
        self.frame_of = torch.full(shape, -1, dtype=torch.int32)
        self.action = torch.zeros(shape, dtype=torch.uint8)
        self.stratum = torch.full(shape, -1, dtype=torch.int64)
        self.ends: list[int] = []
        self.cells: list[torch.Tensor] = []
        self.aux: list[torch.Tensor] = []
        self.frames = 0
        self.jobs: list[torch.Tensor] = [torch.empty(3, 0, dtype=torch.int64)]
        self.job_reward: list[torch.Tensor] = [torch.empty(0, dtype=torch.int64)]
        self.job_flags: list[torch.Tensor] = [torch.empty(2, 0, dtype=torch.bool)]

    def add_window(
        self,
        row: int,
        parts: Sequence[tuple[Segment, torch.Tensor]],
    ) -> None:
        """Lay out one window's segments, then its padding tail.

        Args:
          row: Window index in the batch.
          parts: The window's segments with their decisions' strata, in order.

        """
        cursor = 0
        used = 0
        for segment, strata in parts[: self.s_max]:
            if cursor == self.t_g:
                break
            cursor += self._add_segment(row, segment, strata, start=cursor, index=used)
            used += 1
            self.ends.append(row * self.t_g + cursor)
        self.segment[row, cursor:] = used
        self.pos[row, cursor:] = torch.arange(self.t_g - cursor, dtype=torch.int32)
        self.ends.extend([row * self.t_g + self.t_g] * (self.s_max + 1 - used))

    def finish(self) -> tuple[PackedBatch, torch.Tensor]:
        """Return the packed batch and the position strata.

        Returns:
          batch: Every window added so far.
          stratum: Decision stratum of every position, -1 at padding.

        """
        jobs = torch.cat(self.jobs, dim=1)
        flags = torch.cat(self.job_flags, dim=1)
        batch = PackedBatch(
            kind=self.kind,
            segment=self.segment,
            pos=self.pos,
            cu_seqlens=torch.tensor([0, *self.ends], dtype=torch.int32),
            frame_of=self.frame_of,
            action=self.action,
            cells=torch.cat(self.cells)
            if self.cells
            else torch.empty(0, 99, 8, dtype=torch.uint8),
            aux=torch.cat(self.aux)
            if self.aux
            else torch.empty(0, 51, dtype=torch.int16),
            job_at=jobs[0].int(),
            job_memory=jobs[1].int(),
            job_next=jobs[2].int(),
            job_reward=torch.cat(self.job_reward).short(),
            job_done=flags[0],
            job_is_start=flags[1],
        )
        return batch, self.stratum

    def _add_segment(
        self,
        row: int,
        segment: Segment,
        strata: torch.Tensor,
        *,
        start: int,
        index: int,
    ) -> int:
        """Lay out one segment from ``start``; return its length in positions."""
        room = self.t_g - start
        head = int(segment.starts_episode)
        steps = len(segment.actions)
        decisions = min(steps, (room - head + 1) // 2)
        length = min(room, head + 2 * steps)
        frames = min(len(segment.cells), decisions + 1)
        acts = min(decisions, (length - head) // 2)
        base = self.frames
        self.cells.append(segment.cells[:frames])
        self.aux.append(segment.aux[:frames])
        self.frames += frames
        self.segment[row, start : start + length] = index
        self.pos[row, start : start + length] = torch.arange(length, dtype=torch.int32)
        step = torch.arange(decisions)
        obs = start + head + 2 * step
        self.kind[row, obs] = int(Kind.OBS)
        self.frame_of[row, obs] = (base + step).int()
        self.stratum[row, obs] = strata[:decisions]
        act = obs[:acts] + 1
        self.kind[row, act] = int(Kind.ACT)
        self.action[row, act] = segment.actions[:acts]
        self.stratum[row, act] = strata[:acts]
        flat = row * self.t_g
        if head:
            self.kind[row, start] = int(Kind.START)
            self.stratum[row, start] = strata[0]
            self._add_jobs(torch.tensor([[flat + start], [-1], [base]]), start=True)
        done = segment.done[:acts]
        following = torch.where(
            ~done & (step[:acts] + 1 < frames),
            base + step[:acts] + 1,
            -1,
        )
        jobs = torch.stack([flat + act, base + step[:acts], following])
        self._add_jobs(jobs, reward=segment.reward[:acts], done=done)
        return length

    def _add_jobs(
        self,
        jobs: torch.Tensor,
        *,
        start: bool = False,
        reward: torch.Tensor | None = None,
        done: torch.Tensor | None = None,
    ) -> None:
        """Append jobs as ``[3, J]`` rows: position, memory frame, next frame."""
        count = jobs.shape[1]
        self.jobs.append(jobs)
        self.job_reward.append(
            reward.long()
            if reward is not None
            else torch.zeros(count, dtype=torch.int64),
        )
        is_start = torch.full((count,), fill_value=start)
        done = done if done is not None else torch.zeros(count, dtype=torch.bool)
        self.job_flags.append(torch.stack([done, is_start]))
