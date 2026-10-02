"""Tests for the adaptive-computation-time pool."""

from __future__ import annotations

from typing import Final, NamedTuple

from torch import Tensor

import pytest
import torch

from priml.baselines.sudoku.act import (
    ActPool,
    AtomicPool,
    CellCorruption,
    FeedbackCarry,
    ForcedContinue,
    HaltTraining,
    SampledMinimum,
    SlotScramble,
    StreamingPool,
    ZeroStart,
    rollout,
)


SLOTS: Final = 3
GRID: Final = 5
SEQ: Final = 6
WIDTH: Final = 2
VOCAB: Final = 7


def init(rows: int) -> tuple[Tensor, Tensor]:
    """Distinct initial latents, so a seated slot is recognizable."""
    return torch.full((rows, SEQ, WIDTH), 7.0), torch.full((rows, SEQ, WIDTH), 9.0)


def pool(config: ActPool.Config) -> ActPool:
    """Build ``config`` at the test geometry."""
    config.batch_size = SLOTS
    config.max_steps = 3
    config.grid_len = GRID
    config.seq_len = SEQ
    config.channels_hidden = WIDTH
    return config.make()


def batch(fill: int, rows: int = SLOTS) -> tuple[Tensor, Tensor]:
    """``rows`` puzzles, row ``r`` filled with ``fill + r``."""
    media = (torch.arange(rows) + fill).unsqueeze(-1).expand(rows, GRID).clone()
    return media, media + 1


def advance(subject: ActPool, active: Tensor, *, halt: Tensor) -> None:
    """One step's carry update and halt decision, from zero latents."""
    zeros = torch.zeros(SLOTS, SEQ, WIDTH)
    subject.update_carry(z_slow=zeros, z_fast=zeros, active=active)
    subject.release(init, halt=subject.halt_mask(halt), active=active)


def test_atomic_first_refill_seats_every_slot() -> None:
    subject = pool(AtomicPool.Config())
    media, labels = batch(2)
    active = subject.refill(
        init,
        media=media,
        labels=labels,
        valid_count=SLOTS,
        puzzle_ids=None,
        ignore_label_id=-100,
    )
    assert torch.equal(subject.inputs, media)
    assert bool(active.all())
    assert bool((subject.z_slow == 7.0).all())


def test_a_zero_start_seats_slots_at_zero() -> None:
    subject = pool(AtomicPool.Config(start=ZeroStart.Config()))
    media, labels = batch(2)
    subject.refill(
        init,
        media=media,
        labels=labels,
        valid_count=SLOTS,
        puzzle_ids=None,
        ignore_label_id=-100,
    )
    assert bool((subject.z_slow == 0.0).all())
    assert bool((subject.z_fast == 0.0).all())


def test_atomic_keeps_occupied_slots_and_their_task() -> None:
    subject = pool(AtomicPool.Config())
    first, labels = batch(2)
    ids = torch.tensor([4, 5, 6])
    subject.refill(
        init,
        media=first,
        labels=labels,
        valid_count=SLOTS,
        puzzle_ids=ids,
        ignore_label_id=-100,
    )
    subject.halted = torch.tensor([True, False, True])
    second, labels2 = batch(4)
    subject.refill(
        init,
        media=second,
        labels=labels2,
        valid_count=SLOTS,
        puzzle_ids=ids + 10,
        ignore_label_id=-100,
    )
    assert torch.equal(subject.inputs[1], first[1])
    assert torch.equal(subject.inputs[2], second[2])
    assert subject.puzzle_ids.tolist() == [14, 5, 16]


def test_atomic_masks_padding_labels_and_ids() -> None:
    subject = pool(AtomicPool.Config())
    media, labels = batch(2)
    subject.refill(
        init,
        media=media,
        labels=labels,
        valid_count=2,
        puzzle_ids=torch.tensor([4, 5, 6]),
        ignore_label_id=-100,
    )
    assert bool((subject.labels[2] == -100).all())
    assert not bool((subject.labels[:2] == -100).any())
    assert subject.puzzle_ids.tolist() == [4, 5, 0]


def test_atomic_rejects_a_batch_of_the_wrong_width() -> None:
    subject = pool(AtomicPool.Config())
    media, labels = batch(2, rows=2)
    with pytest.raises(ValueError, match="batch of 3"):
        subject.refill(
            init,
            media=media,
            labels=labels,
            valid_count=2,
            puzzle_ids=None,
            ignore_label_id=-100,
        )


def test_slots_halt_at_the_step_cap() -> None:
    subject = pool(AtomicPool.Config(halting=HaltTraining.Config()))
    media, labels = batch(2)
    never = torch.full((SLOTS,), -100.0)
    for _ in range(3):
        active = subject.refill(
            init,
            media=media,
            labels=labels,
            valid_count=SLOTS,
            puzzle_ids=None,
            ignore_label_id=-100,
        )
        advance(subject, active, halt=never)
    assert bool(subject.halted.all())
    assert subject.steps.tolist() == [3, 3, 3]


def test_without_halting_only_the_cap_releases() -> None:
    subject = pool(AtomicPool.Config(halting=None))
    subject.steps = torch.tensor([0, 3, 1])
    assert subject.halt_mask(torch.full((SLOTS,), 100.0)).tolist() == [
        False,
        True,
        False,
    ]


def streaming(config: StreamingPool.Config) -> StreamingPool:
    """Build a streaming pool at the test geometry."""
    subject = pool(config)
    assert isinstance(subject, StreamingPool)
    return subject


def test_streaming_queues_only_valid_rows() -> None:
    subject = streaming(StreamingPool.Config())
    media, labels = batch(2)
    active = subject.refill(
        init,
        media=media,
        labels=labels,
        valid_count=2,
        puzzle_ids=None,
        ignore_label_id=-100,
    )
    assert active.tolist() == [True, True, False]
    assert torch.equal(subject.inputs[:2], media[:2])
    assert len(subject.pending_inputs) == 0


def test_streaming_holds_surplus_until_a_slot_frees() -> None:
    subject = streaming(StreamingPool.Config(halting=None))
    first, labels = batch(2)
    second, labels2 = batch(5)
    subject.refill(
        init,
        media=first,
        labels=labels,
        valid_count=SLOTS,
        puzzle_ids=None,
        ignore_label_id=-100,
    )
    active = subject.refill(
        init,
        media=second,
        labels=labels2,
        valid_count=SLOTS,
        puzzle_ids=None,
        ignore_label_id=-100,
    )
    assert torch.equal(subject.inputs, first)
    assert len(subject.pending_inputs) == SLOTS
    subject.steps = torch.tensor([0, 2, 0])
    advance(subject, active, halt=torch.zeros(SLOTS))
    # Slot 1 reached the cap and took the oldest queued puzzle, fresh.
    assert torch.equal(subject.inputs[1], second[0])
    assert subject.steps.tolist() == [1, 0, 1]
    assert bool((subject.z_slow[1] == 7.0).all())
    assert bool((subject.z_slow[0] == 0.0).all())
    assert len(subject.pending_inputs) == 2


def test_streaming_leaves_empty_slots_untouched() -> None:
    subject = pool(StreamingPool.Config())
    media, labels = batch(2)
    active = subject.refill(
        init,
        media=media,
        labels=labels,
        valid_count=1,
        puzzle_ids=None,
        ignore_label_id=-100,
    )
    advance(subject, active, halt=torch.full((SLOTS,), -100.0))
    assert subject.steps.tolist() == [1, 0, 0]
    assert bool((subject.z_slow[1:] == 0.0).all())


def test_streaming_marks_only_the_slots_that_halted_this_step() -> None:
    subject = streaming(StreamingPool.Config(halting=None))
    assert subject.halted.tolist() == [False, False, False]
    media, labels = batch(2)
    active = subject.refill(
        init,
        media=media,
        labels=labels,
        valid_count=2,
        puzzle_ids=None,
        ignore_label_id=-100,
    )
    subject.steps = torch.tensor([2, 0, 2])
    advance(subject, active, halt=torch.zeros(SLOTS))
    # Slot 2 is at the cap but empty, so it did not halt.
    assert subject.halted.tolist() == [True, False, False]


def test_atomic_seating_marks_every_slot_active() -> None:
    subject = pool(AtomicPool.Config())
    media, labels = batch(2)
    active = subject.refill(
        init,
        media=media,
        labels=labels,
        valid_count=SLOTS,
        puzzle_ids=None,
        ignore_label_id=-100,
    )
    assert torch.equal(subject.active, active)
    assert bool(active.all())


def test_streaming_rejects_task_ids() -> None:
    subject = pool(StreamingPool.Config())
    media, labels = batch(2)
    with pytest.raises(ValueError, match="task ids"):
        subject.refill(
            init,
            media=media,
            labels=labels,
            valid_count=SLOTS,
            puzzle_ids=torch.zeros(SLOTS),
            ignore_label_id=-100,
        )


def test_seated_slots_restart_their_feedback_from_the_puzzle() -> None:
    subject = pool(AtomicPool.Config(feedback=FeedbackCarry.Config()))
    subject.feedback = torch.full((SLOTS, GRID), 6)
    subject.halted = torch.tensor([True, False, True])
    media, labels = batch(2)
    subject.refill(
        init,
        media=media,
        labels=labels,
        valid_count=SLOTS,
        puzzle_ids=None,
        ignore_label_id=-100,
    )
    assert subject.feedback[:, 0].tolist() == [2, 6, 4]


def test_decode_restores_clues_only_when_given() -> None:
    media = torch.tensor([[1, 4, 1], [5, 1, 1]])
    logits = torch.nn.functional.one_hot(torch.full((2, 3), 3), VOCAB).float()
    plain = FeedbackCarry.Config().make()
    clamped = FeedbackCarry.Config(givens=(2, 5)).make()
    assert plain.decode(logits, media=media).tolist() == [[3, 3, 3], [3, 3, 3]]
    assert clamped.decode(logits, media=media).tolist() == [[3, 4, 3], [5, 3, 3]]


def test_cell_corruption_validates_and_is_a_noop_at_zero() -> None:
    grid = torch.full((2, 3), 4)
    corrupt = CellCorruption.Config(rate=0.0).make()
    generator = torch.Generator().manual_seed(0)
    assert torch.equal(corrupt(grid, given=grid > 0, generator=generator), grid)
    with pytest.raises(ValueError, match="rate"):
        CellCorruption.Config(rate=2.0).make()


def test_slot_scramble_spares_clue_cells() -> None:
    grid = torch.full((2, 3), 4)
    given = torch.tensor([[True, False, False], [False, False, True]])
    scramble = SlotScramble.Config(prob=1.0, cells=3, low=5, high=6).make()
    out = scramble(grid, given=given, generator=torch.Generator().manual_seed(0))
    assert out.tolist() == [[4, 5, 5], [5, 5, 4]]


def test_advance_feedback_reports_the_corrupted_fraction() -> None:
    subject = pool(
        AtomicPool.Config(
            feedback=FeedbackCarry.Config(
                corruption=CellCorruption.Config(rate=1.0, low=6, high=7),
            ),
        ),
    )
    logits = torch.nn.functional.one_hot(torch.full((SLOTS, GRID), 2), VOCAB).float()
    assert float(subject.advance_feedback(logits)) == 1.0
    assert bool((subject.feedback == 6).all())


def test_exploration_policies() -> None:
    fired = torch.tensor([True, True])
    steps = torch.tensor([1, 3])
    generator = torch.Generator().manual_seed(0)
    assert SampledMinimum.Config(prob=0).make()(
        fired,
        steps=steps,
        max_steps=3,
        generator=generator,
    ).tolist() == [True, True]
    assert ForcedContinue.Config(prob=1).make()(
        fired,
        steps=steps,
        max_steps=3,
        generator=generator,
    ).tolist() == [False, False]


def test_halting_is_independent_of_the_ambient_stream() -> None:
    torch.manual_seed(0)
    undisturbed = _halt_sequence(disturb=False)
    torch.manual_seed(0)
    assert _halt_sequence(disturb=True) == undisturbed


def test_geometry_must_be_inherited() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        AtomicPool.Config(batch_size=SLOTS).make()


def test_advance_feedback_requires_a_carry() -> None:
    subject = pool(AtomicPool.Config())
    logits = torch.zeros(SLOTS, GRID, VOCAB)
    with pytest.raises(ValueError, match="feedback carry"):
        subject.advance_feedback(logits)


def test_only_atomic_seating_scores_the_slots_that_halted() -> None:
    """Streaming metrics score every active slot, so it reports no halt mask."""
    atomic = pool(AtomicPool.Config())
    atomic.halted = torch.tensor([True, False, True])
    assert atomic.halted_this_step() is atomic.halted
    assert pool(StreamingPool.Config()).halted_this_step() is None


def test_streaming_moves_its_queue_with_the_pool() -> None:
    # No halting: its generator cannot live on the meta device.
    subject = streaming(StreamingPool.Config(halting=None))
    media, labels = batch(2, rows=SLOTS + 1)
    subject.refill(
        init,
        media=media,
        labels=labels,
        valid_count=SLOTS + 1,
        puzzle_ids=None,
        ignore_label_id=-100,
    )
    assert subject.pending_inputs.shape == (1, GRID)

    subject.to(torch.device("meta"))

    assert subject.pending_inputs.device.type == "meta"
    assert subject.pending_labels.device.type == "meta"
    assert subject.pending_inputs.shape == (1, GRID)


def test_rollout_carries_latents_feedback_and_task_ids_to_the_cap() -> None:
    solver = _Recorder()
    media = torch.tensor([[1, 4, 1], [1, 1, 1]])
    ids = torch.tensor([3, 5])
    carry = FeedbackCarry.Config(givens=(4, 4)).make()
    logits, halt = rollout(
        solver,
        media=media,
        max_steps=3,
        carry=carry,
        prefix_kwargs={"puzzle_identifiers": ids},
    )
    assert len(solver.calls) == 3
    for index, call in enumerate(solver.calls):
        assert call.tokens is media
        assert call.collect_intermediates is False
        assert call.prefix_kwargs == {"puzzle_identifiers": ids}
        assert call.z_slow.tolist() == [[float(index)]] * 2
        assert call.z_fast.tolist() == [[2.0 * index]] * 2
    assert halt.tolist() == [3.0, 3.0]
    assert logits is solver.outputs[-1].logits
    first, second, third = solver.feedback
    assert first is media
    assert second.tolist() == [[2, 4, 2], [2, 2, 2]]
    assert torch.equal(third, second)


def test_rollout_without_a_carry_feeds_nothing_back() -> None:
    solver = _Recorder()
    rollout(solver, media=torch.ones(2, 3, dtype=torch.long), max_steps=2, carry=None)
    assert len(solver.calls) == 2
    assert solver.feedback == []
    assert solver.calls[0].prefix_kwargs == {}


def test_rollout_of_one_step_runs_one_forward() -> None:
    solver = _Recorder()
    rollout(solver, media=torch.ones(2, 3, dtype=torch.long), max_steps=1, carry=None)
    assert len(solver.calls) == 1
    with pytest.raises(ValueError, match="max_steps"):
        rollout(
            solver,
            media=torch.ones(2, 3, dtype=torch.long),
            max_steps=0,
            carry=None,
        )


class _Call(NamedTuple):
    tokens: Tensor
    z_slow: Tensor
    z_fast: Tensor
    collect_intermediates: bool
    prefix_kwargs: dict[str, object]


class _Out(NamedTuple):
    logits: Tensor
    halt: Tensor
    z_slow: Tensor
    z_fast: Tensor


class _Recorder:
    """A solver that records every call and advances its latents by 1 and 2."""

    def __init__(self) -> None:
        self.calls: list[_Call] = []
        self.outputs: list[_Out] = []
        self.feedback: list[Tensor] = []

    def init_latents(self, batch_size: int, /) -> tuple[Tensor, Tensor]:
        return torch.zeros(batch_size, 1), torch.zeros(batch_size, 1)

    def set_feedback(self, grid: Tensor | None) -> None:
        assert grid is not None
        self.feedback.append(grid)

    def __call__(
        self,
        tokens: Tensor,
        z_slow: Tensor,
        z_fast: Tensor,
        *,
        collect_intermediates: bool,
        **prefix_kwargs: object,
    ) -> _Out:
        self.calls.append(
            _Call(tokens, z_slow, z_fast, collect_intermediates, dict(prefix_kwargs)),
        )
        rows = tokens.shape[0]
        logits = torch.nn.functional.one_hot(torch.full((rows, 3), 2), VOCAB).float()
        out = _Out(
            logits,
            torch.full((rows,), float(len(self.calls))),
            z_slow + 1,
            z_fast + 2,
        )
        self.outputs.append(out)
        return out


def _halt_sequence(*, disturb: bool) -> list[bool]:
    """Whether any slot halted, over three steps of a fully-exploring pool."""
    subject = pool(
        AtomicPool.Config(
            halting=HaltTraining.Config(exploration=SampledMinimum.Config(prob=1.0)),
        ),
    )
    media, labels = batch(2)
    out: list[bool] = []
    for _ in range(3):
        if disturb:
            torch.rand(17)  # Ambient draws that must not matter.
        active = subject.refill(
            init,
            media=media,
            labels=labels,
            valid_count=SLOTS,
            puzzle_ids=None,
            ignore_label_id=-100,
        )
        advance(subject, active, halt=torch.full((SLOTS,), 5.0))
        out.append(bool(subject.halted.any()))
    return out


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
