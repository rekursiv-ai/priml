"""Tests for the adaptive-computation-time pool."""

from __future__ import annotations

from torch import Tensor

import pytest
import torch

from priml.baselines.sudoku.act import ActPool


def _pool(**overrides: object) -> ActPool:
    config = ActPool.Config(batch_size=4, max_steps=3)
    config.grid_len = 81
    config.seq_len = 81
    config.hidden_size = 8
    for name, value in overrides.items():
        setattr(config, name, value)
    return config.make()


def _batch(fill: int = 5) -> tuple[Tensor, Tensor]:
    media = torch.full((4, 81), fill, dtype=torch.long)
    labels = torch.full((4, 81), fill, dtype=torch.long)
    return media, labels


def test_first_refill_seats_every_slot() -> None:
    """The pool starts empty, so the first batch fills it completely."""
    pool = _pool()
    media, labels = _batch()
    seated, _, active = pool.refill(
        media, labels=labels, valid_count=4, ignore_label_id=-100
    )
    assert torch.equal(seated, media)
    assert bool(active.all())


def test_occupied_slots_keep_their_puzzle() -> None:
    """A slot mid-solve must not be handed a new puzzle."""
    pool = _pool()
    first, labels = _batch(fill=5)
    pool.refill(first, labels=labels, valid_count=4, ignore_label_id=-100)
    # Halt only slots 0 and 2; the others are still working.
    pool.halted = torch.tensor([True, False, True, False])
    second, labels2 = _batch(fill=7)
    seated, _, _ = pool.refill(
        second, labels=labels2, valid_count=4, ignore_label_id=-100
    )
    assert torch.equal(seated[0], second[0])
    assert torch.equal(seated[2], second[2])
    assert torch.equal(seated[1], first[1])
    assert torch.equal(seated[3], first[3])


def test_padding_rows_are_masked_out_of_the_loss() -> None:
    """A short final batch must not train the model on filler."""
    pool = _pool()
    media, labels = _batch()
    _, seated_labels, _ = pool.refill(
        media, labels=labels, valid_count=2, ignore_label_id=-100
    )
    assert bool((seated_labels[2:] == -100).all())
    assert not bool((seated_labels[:2] == -100).any())


def test_wrong_batch_width_is_rejected() -> None:
    pool = _pool()
    with pytest.raises(ValueError, match="holds 4 slots"):
        pool.refill(
            torch.zeros(3, 81, dtype=torch.long),
            labels=torch.zeros(3, 81),
            valid_count=3,
            ignore_label_id=-100,
        )


def test_slots_halt_at_the_step_cap() -> None:
    """Whatever the halt head says, no puzzle exceeds the cap."""
    pool = _pool(halt_exploration_prob=0.0)
    media, labels = _batch()
    never_halt = torch.full((4,), -100.0)
    for _ in range(3):
        pool.refill(media, labels=labels, valid_count=4, ignore_label_id=-100)
        pool.advance(
            torch.zeros(4, 81, 8),
            z_fast=torch.zeros(4, 81, 8),
            logits=torch.zeros(4, 81, 11),
            halt=never_halt,
            media=media,
        )
    assert bool(pool.halted.all())
    assert int(pool.steps.max()) == 3


def test_givens_survive_the_feedback_loop() -> None:
    """The model may revise its guesses but not the puzzle's clues."""
    pool = _pool()
    media = torch.full((4, 81), 1, dtype=torch.long)  # all empty
    media[:, :10] = 7  # ten clues
    decoded = torch.full((4, 81), 3, dtype=torch.long)
    clamped = pool.clamp_givens(decoded, media=media)
    assert bool((clamped[:, :10] == 7).all())
    assert bool((clamped[:, 10:] == 3).all())


def test_halt_loss_targets_whether_the_grid_is_solved() -> None:
    """The halt head learns "am I done", so a correct grid is its target."""
    pool = _pool()
    labels = torch.full((4, 81), 5, dtype=torch.long)
    # Rows 0 and 1 predict perfectly; rows 2 and 3 are wrong everywhere.
    logits = torch.zeros(4, 81, 11)
    logits[:2, :, 5] = 10.0
    logits[2:, :, 3] = 10.0
    active = torch.ones(4, dtype=torch.bool)

    solved_confident = pool.halt_loss(
        logits,
        labels=labels,
        halt=torch.tensor([10.0, 10.0, -10.0, -10.0]),
        active=active,
        ignore_label_id=-100,
    )[0]
    solved_wrong = pool.halt_loss(
        logits,
        labels=labels,
        halt=torch.tensor([-10.0, -10.0, 10.0, 10.0]),
        active=active,
        ignore_label_id=-100,
    )[0]
    assert float(solved_confident) < float(solved_wrong)


def test_halt_rng_is_independent_of_the_ambient_stream() -> None:
    """Two runs from identical state agree regardless of other RNG use."""
    torch.manual_seed(0)
    undisturbed = _halt_sequence(disturb=False)
    torch.manual_seed(0)
    assert _halt_sequence(disturb=True) == undisturbed


def test_rng_state_round_trips() -> None:
    """Resume must not restart the exploration sequence."""
    pool = _pool()
    state = pool.state_dict()
    before = torch.rand(4, generator=pool._generator)
    pool.load_state_dict(state)
    assert torch.equal(torch.rand(4, generator=pool._generator), before)


def test_geometry_must_be_inherited() -> None:
    config = ActPool.Config(batch_size=4)
    with pytest.raises(ValueError, match="must be positive"):
        config.make()


def _halt_sequence(*, disturb: bool) -> list[bool]:
    """Whether any slot halted, over three steps of a fully-exploring pool."""
    pool = _pool(halt_exploration_prob=1.0)
    media, labels = _batch()
    out: list[bool] = []
    for _ in range(3):
        if disturb:
            torch.rand(17)  # ambient draws that must not matter
        pool.refill(media, labels=labels, valid_count=4, ignore_label_id=-100)
        pool.advance(
            torch.zeros(4, 81, 8),
            z_fast=torch.zeros(4, 81, 8),
            logits=torch.zeros(4, 81, 11),
            halt=torch.full((4,), 5.0),
            media=media,
        )
        out.append(bool(pool.halted.any()))
    return out


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
