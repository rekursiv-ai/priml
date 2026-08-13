"""Tests for the sudoku train step."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from priml.baselines.sudoku.act import ActPool
from priml.baselines.sudoku.embedding import GridEmbedding, PredictionFeedback
from priml.baselines.sudoku.model import DeepRecurrence
from priml.baselines.sudoku.train_step import SudokuTrainStep


def _step(*, act: bool = False) -> SudokuTrainStep:
    config = SudokuTrainStep.Config()
    config.device = "cpu"
    config.dtype_autocast = None
    config.total_train_steps = 8
    config.model.hidden_size = 16
    config.model.num_layers = 1
    config.model.embedding = GridEmbedding.Config()
    if act:
        config.model.recurrence = DeepRecurrence.Config(slow_cycles=2, fast_cycles=1)
        config.act = ActPool.Config(batch_size=4, max_steps=3)
    torch.manual_seed(0)
    return config.make()


def _batch() -> dict[str, Any]:
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
    groups = step.optimizer.param_groups
    assigned = [{named[id(p)] for p in group["params"]} for group in groups]
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


def test_act_pool_is_not_checkpointed() -> None:
    """In-flight puzzles are bound to a batch, so resume starts them fresh.

    Only the halting RNG persists: restarting that would replay the same
    exploration decisions after every resume.
    """
    step = _step(act=True)
    step.train_step(**_batch())
    assert set(step.state_dict()["act"]) == {"halt_rng"}


def test_feedback_reaches_the_channel() -> None:
    """The pool hands the decoded grid to whichever channel consumes it."""
    config = SudokuTrainStep.Config()
    config.device = "cpu"
    config.dtype_autocast = None
    config.model.hidden_size = 16
    config.model.num_layers = 1
    embedding = GridEmbedding.Config()
    embedding.channels = [PredictionFeedback.Config()]
    config.model.embedding = embedding
    config.model.recurrence = DeepRecurrence.Config(slow_cycles=1, fast_cycles=1)
    config.act = ActPool.Config(batch_size=4, max_steps=2)
    torch.manual_seed(0)
    step = config.make()
    step.eval_loss(**_batch())
    channel = step.model.embedding.channels[0]
    assert isinstance(channel, PredictionFeedback)
    # Consumed by the final rollout step, never left stashed.
    assert channel._feedback_ids is None


def test_horizon_must_be_positive() -> None:
    config = SudokuTrainStep.Config()
    config.total_train_steps = 0
    with pytest.raises(ValueError, match="total_train_steps must be positive"):
        config.make()


if __name__ == "__main__":
    from priml.lib.testing import test_main

    test_main(__file__)
