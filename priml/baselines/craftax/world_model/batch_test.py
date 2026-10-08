"""Check packing of episode segments into fixed-length global windows."""

from collections.abc import Sequence
from typing import cast

import dataclasses

import torch

from priml.baselines.craftax.world_model.batch import (
    Kind,
    PackedBatch,
    Segment,
    pack_windows,
)


KIND_START, KIND_OBS, KIND_ACT, KIND_PAD = Kind.START, Kind.OBS, Kind.ACT, Kind.PAD


def _segment(decisions: int, *, frames: int, starts: bool, tag: int) -> Segment:
    cells = torch.zeros(frames, 99, 8, dtype=torch.uint8)
    cells[:, 0, 0] = torch.arange(frames, dtype=torch.uint8) + tag
    return Segment(
        cells=cells,
        aux=torch.zeros(frames, 51, dtype=torch.int16),
        actions=torch.arange(decisions, dtype=torch.uint8) + tag,
        reward=torch.ones(decisions, dtype=torch.int16),
        done=torch.arange(decisions) == decisions - 1
        if frames == decisions
        else torch.zeros(decisions, dtype=torch.bool),
        starts_episode=starts,
    )


def test_episode_start_then_obs_act_pairs() -> None:
    segment = _segment(3, frames=3, starts=True, tag=10)
    batch = pack_windows([[segment]], t_g=8, s_max=4)
    kinds = batch.kind[0].tolist()
    assert kinds == [KIND_START, KIND_OBS, KIND_ACT] * 1 + [KIND_OBS, KIND_ACT] * 2 + [
        KIND_PAD,
    ]
    assert batch.pos[0, :7].tolist() == list(range(7))
    assert batch.action[0, 2].item() == 10
    assert batch.cells[batch.frame_of[0, 1], 0, 0].item() == 10


def test_jobs_point_at_memory_and_next_frames() -> None:
    segment = _segment(3, frames=3, starts=True, tag=10)
    batch = pack_windows([[segment]], t_g=8, s_max=4)
    assert batch.job_is_start.tolist() == [True, False, False, False]
    assert batch.job_at.tolist() == [0, 2, 4, 6]
    assert batch.job_memory[0].item() == -1
    next_first = int(batch.job_next[0])
    assert batch.cells[next_first, 0, 0].item() == 10
    assert batch.cells[batch.job_memory[1], 0, 0].item() == 10
    assert batch.cells[batch.job_next[1], 0, 0].item() == 11
    assert batch.job_next[3].item() == -1
    assert batch.job_done.tolist() == [False, False, False, True]


def test_second_episode_starts_new_segment_with_position_reset() -> None:
    first = _segment(2, frames=2, starts=False, tag=10)
    second = _segment(2, frames=3, starts=True, tag=50)
    batch = pack_windows([[first, second]], t_g=10, s_max=4)
    assert batch.segment[0].tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 1, 2]
    assert batch.pos[0].tolist() == [0, 1, 2, 3, 0, 1, 2, 3, 4, 0]
    assert batch.kind[0, 4].item() == KIND_START
    assert batch.cu_seqlens.tolist() == [0, 4, 9, 10, 10, 10]


def test_truncated_window_keeps_next_frame_of_last_act() -> None:
    segment = _segment(5, frames=6, starts=False, tag=0)
    batch = pack_windows([[segment]], t_g=5, s_max=2)
    assert batch.kind[0].tolist() == [KIND_OBS, KIND_ACT] * 2 + [KIND_OBS]
    last_act_job = batch.job_next[-1]
    assert batch.cells[last_act_job, 0, 0].item() == 2
    assert batch.cu_seqlens.tolist() == [0, 5, 5, 5]


def test_batches_flatten_windows_and_offset_frames() -> None:
    windows = [
        [_segment(2, frames=3, starts=True, tag=0)],
        [_segment(2, frames=3, starts=True, tag=100)],
    ]
    batch = pack_windows(windows, t_g=5, s_max=2)
    assert batch.kind.shape == (2, 5)
    assert batch.cu_seqlens.tolist() == [0, 5, 5, 5, 10, 10, 10]
    second_obs = batch.frame_of[1, 1]
    assert batch.cells[second_obs, 0, 0].item() == 100
    assert batch.job_at.tolist() == [0, 2, 4, 5, 7, 9]


def test_segments_beyond_s_max_are_dropped_and_padded() -> None:
    parts = [_segment(1, frames=1, starts=True, tag=i) for i in range(3)]
    batch = pack_windows([parts], t_g=12, s_max=2)
    assert batch.segment[0].max().item() == 2
    assert batch.kind[0, 6:].tolist() == [KIND_PAD] * 6
    assert batch.cu_seqlens.tolist() == [0, 3, 6, 12]


def test_an_empty_pack_keeps_the_documented_dtypes() -> None:
    batch = pack_windows([[]], t_g=4, s_max=1)
    assert (batch.cells.dtype, batch.cells.shape) == (torch.uint8, (0, 99, 8))
    assert (batch.aux.dtype, batch.aux.shape) == (torch.int16, (0, 51))
    assert batch.job_reward.dtype == torch.int16
    assert batch.job_at.dtype == batch.job_memory.dtype == torch.int32
    assert batch.job_done.dtype == batch.job_is_start.dtype == torch.bool


def test_pack_matches_the_step_by_step_reference_packer() -> None:
    windows = [
        [
            _segment(3, frames=3, starts=True, tag=1),
            _segment(4, frames=5, starts=True, tag=2),
        ],
        [_segment(9, frames=10, starts=False, tag=3)],
        [_segment(1, frames=1, starts=True, tag=i) for i in range(4)],
        [
            _segment(2, frames=2, starts=False, tag=4),
            _segment(6, frames=6, starts=True, tag=5),
        ],
        [_segment(3, frames=3, starts=False, tag=6)],
        [],
    ]
    packed = pack_windows(windows, t_g=11, s_max=3)
    reference = _Reference(t_g=11, s_max=3, windows=len(windows))
    for row, segments in enumerate(windows):
        reference.add_window(row, segments)
    for field in dataclasses.fields(PackedBatch):
        a = cast("torch.Tensor", getattr(packed, field.name))
        b = cast("torch.Tensor", getattr(reference.finish(), field.name))
        assert a.dtype == b.dtype, field.name
        assert torch.equal(a, b), field.name


# The original implementation's step-by-step packer, kept as the reference the
# vectorized ``batch.pack`` is checked against.
class _Reference:
    """Lay out packed windows one decision at a time."""

    def __init__(self, *, t_g: int, s_max: int, windows: int) -> None:
        self.t_g = t_g
        self.s_max = s_max
        shape = (windows, t_g)
        self.kind = torch.full(shape, int(Kind.PAD), dtype=torch.uint8)
        self.segment = torch.zeros(shape, dtype=torch.int16)
        self.pos = torch.zeros(shape, dtype=torch.int32)
        self.frame_of = torch.full(shape, -1, dtype=torch.int32)
        self.action = torch.zeros(shape, dtype=torch.uint8)
        self.ends: list[int] = []
        self.cells: list[torch.Tensor] = []
        self.aux: list[torch.Tensor] = []
        self.frames = 0
        self.jobs: list[tuple[int, int, int, int, bool, bool]] = []

    def add_window(self, row: int, segments: Sequence[Segment]) -> None:
        """Lay out one window's segments, then its padding tail.

        Args:
          row: Window index in the batch.
          segments: The window's segments, in order.

        """
        cursor = 0
        used = 0
        for segment in segments[: self.s_max]:
            if cursor == self.t_g:
                break
            length = self._add_segment(row, segment, start=cursor, index=used)
            cursor += length
            used += 1
            self.ends.append(row * self.t_g + cursor)
        self.segment[row, cursor:] = used
        self.pos[row, cursor:] = torch.arange(self.t_g - cursor, dtype=torch.int32)
        tail = [row * self.t_g + self.t_g] * (self.s_max + 1 - used)
        self.ends.extend(tail)

    def finish(self) -> PackedBatch:
        """Return the packed batch.

        Returns:
          batch: Every window added so far.

        """
        jobs = torch.tensor([job[:4] for job in self.jobs], dtype=torch.int32)
        flags = torch.tensor([job[4:] for job in self.jobs], dtype=torch.bool)
        empty = torch.empty(0, dtype=torch.int32)
        return PackedBatch(
            kind=self.kind,
            segment=self.segment,
            pos=self.pos,
            cu_seqlens=torch.tensor([0, *self.ends], dtype=torch.int32),
            frame_of=self.frame_of,
            action=self.action,
            cells=torch.cat(self.cells) if self.cells else torch.empty(0, 99, 8),
            aux=torch.cat(self.aux) if self.aux else torch.empty(0, 51),
            job_at=jobs[:, 0] if self.jobs else empty,
            job_memory=jobs[:, 1] if self.jobs else empty,
            job_next=jobs[:, 2] if self.jobs else empty,
            job_reward=jobs[:, 3].to(torch.int16) if self.jobs else empty,
            job_done=flags[:, 0] if self.jobs else empty.bool(),
            job_is_start=flags[:, 1] if self.jobs else empty.bool(),
        )

    def _add_segment(
        self,
        row: int,
        segment: Segment,
        *,
        start: int,
        index: int,
    ) -> int:
        """Lay out one segment from ``start``; return its length in positions."""
        room = self.t_g - start
        head = 1 if segment.starts_episode else 0
        decisions = min(len(segment.actions), (room - head + 1) // 2)
        length = min(room, head + 2 * len(segment.actions))
        frames = min(len(segment.cells), decisions + 1)
        base = self.frames
        self.cells.append(segment.cells[:frames])
        self.aux.append(segment.aux[:frames])
        self.frames += frames
        span = slice(start, start + length)
        self.segment[row, span] = index
        self.pos[row, span] = torch.arange(length, dtype=torch.int32)
        flat = row * self.t_g
        if head:
            self.kind[row, start] = Kind.START
            self.jobs.append((flat + start, -1, base, 0, False, True))
        for step in range(decisions):
            obs = start + head + 2 * step
            self.kind[row, obs] = Kind.OBS
            self.frame_of[row, obs] = base + step
            if obs + 1 >= start + length:
                break
            self.kind[row, obs + 1] = Kind.ACT
            self.action[row, obs + 1] = segment.actions[step]
            done = bool(segment.done[step])
            following = base + step + 1 if not done and step + 1 < frames else -1
            reward = int(segment.reward[step])
            self.jobs.append(
                (flat + obs + 1, base + step, following, reward, done, False),
            )
        return length


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
