"""Tests for the adaptive-computation-time pool."""

from __future__ import annotations

from typing import Final, NamedTuple
from unittest.mock import Mock

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
    assert plain.given(media).dtype == torch.bool
    assert plain.given(media).tolist() == [[False, False, False], [False, False, False]]
    assert clamped.given(media).tolist() == [[False, True, False], [True, False, False]]
    assert plain.decode(logits, media=media).tolist() == [[3, 3, 3], [3, 3, 3]]
    assert clamped.decode(logits, media=media).tolist() == [[3, 4, 3], [5, 3, 3]]


def test_decode_converts_restored_clues_to_prediction_dtype() -> None:
    media = torch.tensor([[2.5, 0.0, 4.5]])
    logits = torch.nn.functional.one_hot(torch.tensor([[1, 2, 3]]), VOCAB).float()
    decoded = FeedbackCarry.Config(givens=(2, 5)).make().decode(logits, media=media)
    assert decoded.dtype == torch.long
    assert decoded.tolist() == [[2, 2, 4]]


def test_cell_corruption_validates_and_is_a_noop_at_zero() -> None:
    grid = torch.full((2, 3), 4)
    corrupt = CellCorruption.Config(rate=0.0).make()
    generator = torch.Generator().manual_seed(0)
    assert torch.equal(corrupt(grid, given=grid > 0, generator=generator), grid)
    for rate in (float("nan"), -0.1, 1.1):
        with pytest.raises(ValueError, match=f"got {rate}"):
            CellCorruption.Config(rate=rate).make()


def test_slot_scramble_excludes_slots_equal_to_probability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grid = torch.full((3, 2), 4)
    draws = [torch.full((3,), 0.5), torch.zeros((3, 2))]
    monkeypatch.setattr(torch, "rand", Mock(side_effect=draws))
    monkeypatch.setattr(torch, "randint", Mock(return_value=torch.full((3, 2), 5)))
    out = SlotScramble.Config(prob=0.5, cells=1, low=5, high=6).make()(
        grid,
        given=torch.zeros_like(grid, dtype=torch.bool),
        generator=torch.Generator(),
    )
    assert torch.equal(out, grid)


def test_slot_scramble_uses_fraction_of_grid_length_for_cell_rate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grid = torch.full((3, 2), 4)
    draws = [torch.zeros(3), torch.tensor([[0.1, 0.5], [0.9, 0.2], [0.3, 0.7]])]
    monkeypatch.setattr(torch, "rand", Mock(side_effect=draws))
    monkeypatch.setattr(torch, "randint", Mock(return_value=torch.full((3, 2), 5)))
    out = SlotScramble.Config(prob=1.0, cells=1, low=5, high=6).make()(
        grid,
        given=torch.zeros_like(grid, dtype=torch.bool),
        generator=torch.Generator(),
    )
    assert out.tolist() == [[5, 4], [4, 5], [5, 4]]


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


def test_halt_training_to_passes_requested_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    subject = HaltTraining.Config().make()
    generator = Mock(wraps=torch.Generator)
    monkeypatch.setattr(torch, "Generator", generator)
    subject.to(torch.device("cpu"))
    assert generator.call_args.kwargs == {"device": torch.device("cpu")}


def test_feedback_carry_to_passes_requested_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    subject = FeedbackCarry.Config().make()
    generator = Mock(wraps=torch.Generator)
    monkeypatch.setattr(torch, "Generator", generator)
    subject.to(torch.device("cpu"))
    assert generator.call_args.kwargs == {"device": torch.device("cpu")}


def test_halt_training_to_resets_seeded_generator() -> None:
    config = HaltTraining.Config(seed=23)
    subject = config.make()
    torch.rand(4, generator=subject.generator)
    subject.to(torch.device("cpu"))
    actual = torch.rand(4, generator=subject.generator)
    expected = torch.rand(4, generator=torch.Generator().manual_seed(23))
    assert torch.equal(actual, expected)


def test_feedback_carry_to_resets_seeded_generator() -> None:
    config = FeedbackCarry.Config(seed=29)
    subject = config.make()
    torch.rand(4, generator=subject.generator)
    subject.to(torch.device("cpu"))
    actual = torch.rand(4, generator=subject.generator)
    expected = torch.rand(4, generator=torch.Generator().manual_seed(29))
    assert torch.equal(actual, expected)


def test_halt_training_keeps_config_weight_exploration_and_seed() -> None:
    config = HaltTraining.Config(
        weight=0.25,
        exploration=SampledMinimum.Config(prob=0.75),
        seed=17,
    )
    subject = config.make()
    assert subject.config == config
    assert subject.weight == 0.25
    assert isinstance(subject.exploration, SampledMinimum)
    expected = torch.Generator().manual_seed(17)
    assert torch.equal(
        torch.rand(5, generator=subject.generator),
        torch.rand(5, generator=expected),
    )


def test_halting_is_independent_of_the_ambient_stream() -> None:
    torch.manual_seed(0)
    undisturbed = _halt_sequence(disturb=False)
    torch.manual_seed(0)
    assert _halt_sequence(disturb=True) == undisturbed


def test_geometry_must_be_inherited() -> None:
    for field_name in ("grid_len", "seq_len", "channels_hidden"):
        config = AtomicPool.Config(
            batch_size=SLOTS,
            grid_len=GRID,
            seq_len=SEQ,
            channels_hidden=WIDTH,
        )
        setattr(config, field_name, 0)
        with pytest.raises(
            ValueError,
            match=(
                rf"grid_len, seq_len, and channels_hidden must be positive; "
                rf"they are normally inherited from the model during finalize\. "
                rf"Got {config.grid_len}, {config.seq_len}, {config.channels_hidden}\."
            ),
        ):
            config.make()
        setattr(config, field_name, 1)
        config.make()


def test_cell_corruption_draws_on_grid_device_with_generator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rand = Mock(wraps=torch.rand)
    randint = Mock(wraps=torch.randint)
    monkeypatch.setattr(torch, "rand", rand)
    monkeypatch.setattr(torch, "randint", randint)
    grid = torch.full((2, 3), 4)
    generator = torch.Generator().manual_seed(4)
    CellCorruption.Config(rate=0.5, low=2, high=5).make()(
        grid,
        given=torch.zeros_like(grid, dtype=torch.bool),
        generator=generator,
    )
    assert rand.call_args.args == (grid.shape,)
    assert rand.call_args.kwargs == {"device": grid.device, "generator": generator}
    assert randint.call_args.args == (2, 5, grid.shape)
    assert randint.call_args.kwargs == {"device": grid.device, "generator": generator}


def test_slot_scramble_draw_order_device_and_generator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rand = Mock(wraps=torch.rand)
    randint = Mock(wraps=torch.randint)
    monkeypatch.setattr(torch, "rand", rand)
    monkeypatch.setattr(torch, "randint", randint)
    grid = torch.full((2, 3), 4)
    generator = torch.Generator().manual_seed(4)
    SlotScramble.Config(prob=0.5, cells=2, low=2, high=5).make()(
        grid,
        given=torch.zeros_like(grid, dtype=torch.bool),
        generator=generator,
    )
    assert [call.args for call in rand.call_args_list] == [(2,), (2, 3)]
    assert all(
        call.kwargs == {"device": grid.device, "generator": generator}
        for call in rand.call_args_list
    )
    assert randint.call_args.args == (2, 5, (2, 3))
    assert randint.call_args.kwargs == {"device": grid.device, "generator": generator}


def test_cell_corruption_does_not_replace_a_draw_equal_to_rate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grid = torch.full((2, 3), 4)
    monkeypatch.setattr(torch, "rand", Mock(return_value=torch.full((2, 3), 0.5)))
    monkeypatch.setattr(torch, "randint", Mock(return_value=torch.full((2, 3), 5)))
    out = CellCorruption.Config(rate=0.5, low=5, high=6).make()(
        grid,
        given=torch.zeros_like(grid, dtype=torch.bool),
        generator=torch.Generator(),
    )
    assert torch.equal(out, grid)


def test_forced_continue_keeps_a_draw_equal_to_probability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch, "rand", Mock(return_value=torch.full((3,), 0.5)))
    fired = torch.tensor([True, False, True])
    out = ForcedContinue.Config(prob=0.5).make()(
        fired,
        steps=torch.zeros(3, dtype=torch.long),
        max_steps=3,
        generator=torch.Generator(),
    )
    assert torch.equal(out, fired)


def test_exploration_draws_use_fired_device_and_dedicated_generator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rand = Mock(wraps=torch.rand)
    randint = Mock(wraps=torch.randint)
    ones = Mock(wraps=torch.ones)
    monkeypatch.setattr(torch, "rand", rand)
    monkeypatch.setattr(torch, "randint", randint)
    monkeypatch.setattr(torch, "ones", ones)
    fired = torch.tensor([True, False, True])
    generator = torch.Generator().manual_seed(4)
    SampledMinimum.Config(prob=0.5).make()(
        fired,
        steps=torch.tensor([1, 2, 3]),
        max_steps=3,
        generator=generator,
    )
    assert rand.call_args.args == (3,)
    assert rand.call_args.kwargs == {"device": fired.device, "generator": generator}
    assert randint.call_args.args == (2, 4, (3,))
    assert randint.call_args.kwargs == {"device": fired.device, "generator": generator}
    assert ones.call_count == 1
    assert ones.call_args.args == (3,)
    assert ones.call_args.kwargs == {"dtype": torch.long, "device": fired.device}
    rand.reset_mock()
    ForcedContinue.Config(prob=0.5).make()(
        fired,
        steps=torch.zeros(3, dtype=torch.long),
        max_steps=3,
        generator=generator,
    )
    assert rand.call_args.args == (3,)
    assert rand.call_args.kwargs == {"device": fired.device, "generator": generator}


def test_sampled_minimum_excludes_draw_equal_to_probability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch, "rand", Mock(return_value=torch.full((3,), 0.5)))
    monkeypatch.setattr(torch, "randint", Mock(return_value=torch.full((3,), 2)))
    fired = torch.tensor([True, True, True])
    result = SampledMinimum.Config(prob=0.5).make()(
        fired,
        steps=torch.ones(3, dtype=torch.long),
        max_steps=3,
        generator=torch.Generator(),
    )
    assert result.tolist() == [True, True, True]


def test_sampled_minimum_zero_probability_skips_random_draws(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rand = Mock(wraps=torch.rand)
    randint = Mock(wraps=torch.randint)
    ones = Mock(wraps=torch.ones)
    monkeypatch.setattr(torch, "rand", rand)
    monkeypatch.setattr(torch, "randint", randint)
    monkeypatch.setattr(torch, "ones", ones)
    fired = torch.tensor([True, False, True])
    steps = torch.tensor([0, 1, 2])
    result = SampledMinimum.Config(prob=0.0).make()(
        fired,
        steps=steps,
        max_steps=3,
        generator=torch.Generator(),
    )
    rand.assert_not_called()
    randint.assert_not_called()
    assert ones.call_count == 1
    assert ones.call_args.args == (3,)
    assert ones.call_args.kwargs == {"dtype": torch.long, "device": fired.device}
    assert result.tolist() == [False, False, True]


def test_pool_initializes_every_state_tensor_with_contract_dtype() -> None:
    subject = pool(AtomicPool.Config(dtype=torch.float64))
    assert subject.inputs.shape == (SLOTS, GRID)
    assert subject.inputs.dtype == torch.long
    assert subject.labels.dtype == torch.long
    assert subject.z_slow.shape == (SLOTS, SEQ, WIDTH)
    assert subject.z_slow.dtype == torch.float64
    assert subject.z_fast.dtype == torch.float64
    assert subject.steps.dtype == torch.long
    assert subject.active.dtype == torch.bool
    assert subject.halted.tolist() == [True, True, True]
    assert subject.puzzle_ids.dtype == torch.int32
    assert subject.feedback.shape == (SLOTS, GRID)
    assert subject.halting is not None


def test_streaming_casts_both_seated_latents_to_storage_dtype() -> None:
    subject = streaming(StreamingPool.Config(halting=None, dtype=torch.float64))
    media, labels = batch(2)
    subject.refill(
        init,
        media=media,
        labels=labels,
        valid_count=2,
        puzzle_ids=None,
        ignore_label_id=-100,
    )
    assert subject.z_slow.dtype == torch.float64
    assert subject.z_fast.dtype == torch.float64
    assert subject.z_slow[0].tolist() == [[7.0] * WIDTH] * SEQ
    assert subject.z_fast[1].tolist() == [[9.0] * WIDTH] * SEQ


def test_streaming_empty_queue_has_grid_dimension() -> None:
    subject = streaming(StreamingPool.Config())
    assert subject.pending_inputs.shape == (0, GRID)
    assert subject.pending_labels.shape == (0, GRID)


def test_advance_feedback_is_clean_without_corruption() -> None:
    subject = pool(AtomicPool.Config(feedback=FeedbackCarry.Config()))
    logits = torch.nn.functional.one_hot(torch.full((SLOTS, GRID), 3), VOCAB).float()
    changed = subject.advance_feedback(logits)
    assert changed == 0.0
    assert subject.feedback.tolist() == [[3] * GRID] * SLOTS


def test_advance_feedback_corrupts_only_nonclue_cells_and_reports_fraction() -> None:
    subject = pool(
        AtomicPool.Config(
            feedback=FeedbackCarry.Config(
                givens=(2, 2),
                corruption=SlotScramble.Config(
                    prob=1.0,
                    cells=GRID,
                    low=6,
                    high=7,
                ),
            ),
        ),
    )
    subject.inputs[:] = torch.tensor([[2, 3, 2, 3, 3]] * SLOTS)
    logits = torch.nn.functional.one_hot(torch.full((SLOTS, GRID), 3), VOCAB).float()
    changed = subject.advance_feedback(logits)
    assert isinstance(changed, Tensor)
    assert torch.equal(changed, torch.tensor(3 / 5))
    assert subject.feedback.tolist() == [[2, 6, 2, 6, 6]] * SLOTS


def test_halt_mask_uses_strictly_positive_logits_and_cap() -> None:
    subject = pool(AtomicPool.Config())
    subject.steps = torch.tensor([0, 0, 3])
    assert subject.halt_mask(torch.tensor([0.0, 1.0, -1.0])).tolist() == [
        False,
        True,
        True,
    ]


def test_pool_to_moves_state_tensors() -> None:
    subject = pool(AtomicPool.Config(halting=None))
    subject.to(torch.device("meta"))
    for tensor in (
        subject.inputs,
        subject.labels,
        subject.z_slow,
        subject.z_fast,
        subject.steps,
        subject.active,
        subject.halted,
        subject.puzzle_ids,
        subject.feedback,
    ):
        assert tensor.device.type == "meta"
    assert subject.halting is None
    assert subject.carry is None


def test_pool_to_forwards_device_to_halting_and_feedback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    subject = pool(
        AtomicPool.Config(
            feedback=FeedbackCarry.Config(),
        ),
    )
    assert subject.halting is not None
    assert subject.carry is not None
    halting_to = Mock()
    carry_to = Mock()
    monkeypatch.setattr(subject.halting, "to", halting_to)
    monkeypatch.setattr(subject.carry, "to", carry_to)
    device = torch.device("meta")

    subject.to(device)

    assert halting_to.call_args.args == (device,)
    assert carry_to.call_args.args == (device,)


def test_advance_feedback_passes_dedicated_corruption_generator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rand = Mock(wraps=torch.rand)
    monkeypatch.setattr(torch, "rand", rand)
    subject = pool(
        AtomicPool.Config(
            feedback=FeedbackCarry.Config(
                corruption=CellCorruption.Config(rate=1.0, low=6, high=7),
            ),
        ),
    )
    logits = torch.nn.functional.one_hot(torch.full((SLOTS, GRID), 2), VOCAB).float()
    subject.advance_feedback(logits)
    assert subject.carry is not None
    assert rand.call_args.kwargs == {
        "device": subject.inputs.device,
        "generator": subject.carry.generator,
    }


def test_advance_feedback_requires_a_carry() -> None:
    subject = pool(AtomicPool.Config())
    logits = torch.zeros(SLOTS, GRID, VOCAB)
    with pytest.raises(
        ValueError,
        match=r"^advance_feedback needs a pool with a feedback carry\.$",
    ):
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
