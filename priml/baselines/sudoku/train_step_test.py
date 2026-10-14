"""Tests for the sudoku train step."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

from treekle import from_plain

import pytest
import torch

from priml.baselines.sudoku import train_step
from priml.baselines.sudoku.act import (
    AtomicPool,
    FeedbackCarry,
    ForcedContinue,
    HaltTraining,
    StreamingPool,
)
from priml.baselines.sudoku.embedding import GridEmbedding, PredictionFeedback
from priml.baselines.sudoku.model import DeepRecurrence
from priml.baselines.sudoku.prefix import SparsePuzzleEmbedding
from priml.baselines.sudoku.train_step import SudokuTrainStep
from priml.train.parallelism import NoParallel


if TYPE_CHECKING:
    from collections.abc import Mapping


class _TestPrefixKwargs(TypedDict, total=False):
    puzzle_identifiers: object


class _RolloutSpy:
    def __init__(self, logits: torch.Tensor, halt: torch.Tensor) -> None:
        self.logits = logits
        self.halt = halt
        self.call_count = 0
        self.received: (
            tuple[
                object,
                torch.Tensor,
                int,
                FeedbackCarry | None,
                Mapping[str, object],
            ]
            | None
        ) = None

    def __call__(
        self,
        model: object,
        *,
        media: torch.Tensor,
        max_steps: int,
        carry: FeedbackCarry | None,
        prefix_kwargs: Mapping[str, object] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert prefix_kwargs is not None
        self.call_count += 1
        self.received = (model, media, max_steps, carry, prefix_kwargs)
        return self.logits, self.halt


def _step(*, act: bool = False, prefix: bool = False) -> SudokuTrainStep:
    config = SudokuTrainStep.Config()
    config.parallelism = NoParallel.Config(device="cpu")
    config.compile = None
    config.dtype_autocast = None
    config.total_train_steps = 8
    config.model.channels_in = 16
    config.model.num_layers = 1
    config.model.vocab_size = 11
    config.model.embedding = GridEmbedding.Config(grid_shape=(81,))
    if prefix:
        config.model.prefix = SparsePuzzleEmbedding.Config(
            num_puzzles=11,
            num_tokens=1,
            batch_size=4,
        )
    if act:
        config.model.recurrence = DeepRecurrence.Config(slow_cycles=2, fast_cycles=1)
        config.pool = AtomicPool.Config(
            batch_size=4,
            max_steps=3,
            feedback=FeedbackCarry.Config(givens=(2, 10)),
        )
    torch.manual_seed(0)
    return config.make()


def _batch() -> dict[str, object]:
    return {
        "media": torch.randint(2, 11, (4, 81)),
        "label": torch.randint(2, 11, (4, 81)),
        "valid_count": 4,
    }


@pytest.mark.parametrize("act", [False, True])
def test_loss_decreases_over_a_few_steps(act: bool) -> None:
    """Both modes actually learn on a repeated batch."""
    step = _step(act=act)
    batch = _batch()
    losses = [float(step.train_step(**batch)["loss"]) for _ in range(3)]
    assert losses[-1] < losses[0]


def test_optimizer_partitions_the_model() -> None:
    """Muon takes the reasoning matrices; AdamW takes tables and heads.

    Each parameter belongs to exactly one member, which the composite verifies;
    this pins WHICH, since the split is the recipe.
    """
    step = _step()
    named = {id(p): n for n, p in step.model.named_parameters()}
    assigned = [
        {named[id(p)] for p in _parameters(group)}
        for group in step.optimizer.param_groups
    ]
    everything: set[str] = set()
    for names in assigned:
        everything |= names
    assert everything == set(named.values())
    # No parameter appears twice.
    assert sum(len(names) for names in assigned) == len(everything)
    # Heads and lookup tables are never orthogonalized.
    for names in assigned:
        if any("reasoning" in n for n in names):
            assert not any("head" in n or "embed" in n for n in names)


def test_ema_shadow_seeds_then_averages() -> None:
    """At warmup the shadow copies live weights, then trails them."""
    step = _step()
    batch = _batch()
    step.train_step(**batch)
    shadow = step.ema_shadow
    assert shadow is not None
    live = {n: p.detach().clone() for n, p in step.model.named_parameters()}
    for name, value in live.items():
        assert torch.equal(shadow[name], value)

    step.train_step(**batch)
    moved = {n: p.detach().clone() for n, p in step.model.named_parameters()}
    decay = step.config.ema_decay
    for name, first in live.items():
        expected = first.mul(decay).add(moved[name], alpha=1 - decay)
        assert torch.allclose(shadow[name], expected, atol=1e-6)


def test_eval_packs_halt_then_grid() -> None:
    """The metric reads the grid from the END, so the packing must match."""
    step = _step()
    out = step.eval_loss(**_batch())
    assert out["model"].shape == (4, 1 + 81)
    predictions = out["model"][:, 1:]
    assert torch.equal(predictions, predictions.round())


def test_eval_rollout_passes_the_pool_carry_and_prefix_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    step = _step(act=True)
    pool = step.pool
    assert pool is not None
    assert pool.carry is not None
    media = torch.randint(2, 11, (4, 81))
    prefix_kwargs: _TestPrefixKwargs = {
        "puzzle_identifiers": torch.arange(4),
    }
    logits = torch.zeros(4, 81, 11)
    halt = torch.zeros(4)
    spy = _RolloutSpy(logits, halt)
    monkeypatch.setattr(train_step, "rollout", spy)

    actual_logits, actual_halt = step._eval_rollout(media, prefix_kwargs)

    assert actual_logits is logits
    assert actual_halt is halt
    assert spy.call_count == 1
    assert spy.received is not None
    model, received_media, max_steps, carry, received_prefix_kwargs = spy.received
    assert model is step.net
    assert received_media is media
    assert max_steps == pool.config.max_steps
    assert carry is pool.carry
    assert received_prefix_kwargs is prefix_kwargs


def test_eval_rollout_passes_prefix_kwargs_without_an_act_pool() -> None:
    step = _step(prefix=True)
    step.net.eval()
    media = torch.randint(2, 11, (4, 81))
    puzzle_identifiers = torch.arange(1, 5, dtype=torch.int32)
    prefix_kwargs: _TestPrefixKwargs = {
        "puzzle_identifiers": puzzle_identifiers,
    }
    expected = step.net(media, **prefix_kwargs)

    logits, halt = step._eval_rollout(media, prefix_kwargs)

    assert torch.equal(logits, expected.logits)
    assert torch.equal(halt, expected.halt)


def test_act_metrics_appear_only_with_act() -> None:
    """A plain run's metrics carry nothing about halting."""
    plain = _step().train_step(**_batch()).get("metrics", {})
    recurrent = _step(act=True).train_step(**_batch()).get("metrics", {})
    assert "halt_loss" not in plain
    assert {"halt_loss", "halted_frac", "act_steps"} <= set(recurrent)


def test_state_round_trips_including_ema() -> None:
    """A restored step continues rather than restarting."""
    step = _step()
    batch = _batch()
    step.train_step(**batch)
    state = step.state_dict()

    restored = _step()
    restored.load_state_dict(state)
    assert restored.global_step == step.global_step
    for (name, a), (_, b) in zip(
        step.model.named_parameters(),
        restored.model.named_parameters(),
        strict=True,
    ):
        assert torch.equal(a, b), name
    shadow, restored_shadow = step.ema_shadow, restored.ema_shadow
    assert shadow is not None
    assert restored_shadow is not None
    for name, value in shadow.items():
        assert torch.equal(restored_shadow[name], value)


def test_act_steps_average_only_occupied_slots() -> None:
    """An empty streaming slot is not a puzzle that took zero steps."""
    config = SudokuTrainStep.Config()
    config.parallelism = NoParallel.Config(device="cpu")
    config.compile = None
    config.dtype_autocast = None
    config.model.channels_in = 16
    config.model.num_layers = 1
    config.model.vocab_size = 11
    config.model.embedding = GridEmbedding.Config(grid_shape=(81,))
    config.model.recurrence = DeepRecurrence.Config(slow_cycles=1, fast_cycles=1)
    config.pool = StreamingPool.Config(
        batch_size=4,
        max_steps=3,
        halting=HaltTraining.Config(exploration=ForcedContinue.Config(prob=1.0)),
    )
    torch.manual_seed(0)
    step = config.make()
    step.train_step(**{**_batch(), "valid_count": 2})
    metrics = step.train_step(**{**_batch(), "valid_count": 0}).get("metrics", {})
    assert float(metrics["act_steps"]) == 1.0


def test_act_pool_is_not_checkpointed() -> None:
    """In-flight puzzles are bound to a batch, so resume starts them fresh.

    Only the halting RNG persists: restarting that would replay the same
    exploration decisions after every resume.
    """
    step = _step(act=True)
    step.train_step(**_batch())
    state = step.state_dict()
    assert "halt_rng" in state
    assert "corruption_rng" in state


def test_feedback_reaches_the_channel() -> None:
    """The pool hands the decoded grid to whichever channel consumes it."""
    config = SudokuTrainStep.Config()
    config.parallelism = NoParallel.Config(device="cpu")
    config.compile = None
    config.dtype_autocast = None
    config.model.channels_in = 16
    config.model.num_layers = 1
    config.model.vocab_size = 11
    embedding = GridEmbedding.Config(grid_shape=(81,))
    embedding.channels = [PredictionFeedback.Config()]
    config.model.embedding = embedding
    config.model.recurrence = DeepRecurrence.Config(slow_cycles=1, fast_cycles=1)
    config.pool = AtomicPool.Config(
        batch_size=4,
        max_steps=2,
        feedback=FeedbackCarry.Config(givens=(2, 10)),
    )
    torch.manual_seed(0)
    step = config.make()
    step.eval_loss(**_batch())
    embedding = step.net.embedding
    channels = embedding.channels
    channel = channels[0]
    assert isinstance(channel, PredictionFeedback)
    # Consumed by the final rollout step, never left stashed.
    assert channel._feedback_ids is None


def test_schedule_and_loss_seams() -> None:
    step = _step()
    assert step.progress_learning_schedule >= 0.0
    assert step.train_loss(**_batch())["loss"].ndim == 0
    assert step.call_eval(**_batch()).shape == (4, 81, 11)
    with pytest.raises(ValueError, match="Expected not args"):
        step.call_eval("unexpected", **_batch())


def test_unclipped_step_and_no_ema() -> None:
    config = SudokuTrainStep.Config()
    config.parallelism = NoParallel.Config(device="cpu")
    config.compile = None
    config.dtype_autocast = None
    config.gradient_clip_norm = float("inf")
    config.use_ema = False
    config.model.channels_in = 16
    config.model.num_layers = 1
    config.model.vocab_size = 11
    config.model.embedding = GridEmbedding.Config(grid_shape=(81,))
    step = config.make()
    assert step.ema_shadow is None
    assert step.train_step(**_batch())["loss"].shape == (1,)


def test_a_finite_clip_reports_the_pre_clip_gradient_norm() -> None:
    config = SudokuTrainStep.Config()
    config.parallelism = NoParallel.Config(device="cpu")
    config.compile = None
    config.dtype_autocast = None
    config.gradient_clip_norm = 1e-6
    config.model.channels_in = 16
    config.model.num_layers = 1
    config.model.vocab_size = 11
    config.model.embedding = GridEmbedding.Config(grid_shape=(81,))
    torch.manual_seed(0)
    step = config.make()
    grad_norm = step.train_step(**_batch()).get("metrics", {})["grad_norm"]
    assert isinstance(grad_norm, torch.Tensor)
    assert float(grad_norm) > config.gradient_clip_norm


def test_a_pre_timer_checkpoint_is_refused() -> None:
    step = _step()
    state = {k: v for k, v in step.state_dict().items() if k != "timer_step"}
    with pytest.raises(ValueError, match="predates the shared step timer"):
        step.load_state_dict({**state, "global_step": 3})


def test_act_resume_restores_the_dedicated_generators() -> None:
    step = _step(act=True)
    step.train_step(**_batch())
    state = step.state_dict()
    pool = step.pool
    assert pool is not None
    assert pool.halting is not None
    assert pool.carry is not None
    expected = (
        torch.rand(3, generator=pool.halting.generator),
        torch.rand(3, generator=pool.carry.generator),
    )
    restored = _step(act=True)
    restored.load_state_dict(state)
    restored_pool = restored.pool
    assert restored_pool is not None
    assert restored_pool.halting is not None
    assert restored_pool.carry is not None
    assert torch.equal(
        torch.rand(3, generator=restored_pool.halting.generator),
        expected[0],
    )
    assert torch.equal(
        torch.rand(3, generator=restored_pool.carry.generator),
        expected[1],
    )


def test_act_refuses_non_tensor_puzzle_identifiers() -> None:
    step = _step(act=True)
    with pytest.raises(TypeError, match="puzzle_identifiers must be a Tensor"):
        step.train_step(**_batch(), puzzle_identifiers=[0, 1, 2, 3])


def test_act_prefix_reads_the_ids_each_slot_was_seated_with() -> None:
    """A seated slot keeps its puzzle's id, not the incoming row's."""
    step = _step(act=True, prefix=True)
    pool = step.pool
    assert pool is not None
    seated = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
    step.train_step(**_batch(), puzzle_identifiers=seated)
    kwargs = step._prefix_kwargs({"puzzle_identifiers": seated.flip(0)})
    assert kwargs.get("puzzle_identifiers") is pool.puzzle_ids


def test_eval_scores_the_ema_after_warmup_under_autocast() -> None:
    """Past warmup the eval reads the shadow, so live weights cannot move it."""
    config = SudokuTrainStep.Config()
    config.parallelism = NoParallel.Config(device="cpu")
    config.compile = None
    config.total_train_steps = 8
    config.model.channels_in = 16
    config.model.num_layers = 1
    config.model.vocab_size = 11
    config.model.embedding = GridEmbedding.Config(grid_shape=(81,))
    torch.manual_seed(0)
    step = config.make()
    assert config.dtype_autocast == torch.bfloat16
    batch = _batch()
    for _ in range(2):
        step.train_step(**batch)
    ema_logits = step.call_eval(**batch)
    assert ema_logits.dtype == torch.bfloat16
    with torch.no_grad():
        for parameter in step.model.parameters():
            parameter.add_(1.0)
    assert torch.equal(step.call_eval(**batch), ema_logits)


def test_horizon_must_be_positive() -> None:
    config = SudokuTrainStep.Config()
    config.total_train_steps = 0
    with pytest.raises(ValueError, match="total_train_steps must be positive"):
        config.make()


def _parameters(group: dict[str, object]) -> list[torch.Tensor]:
    return from_plain(group["params"], list[torch.Tensor])


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
