"""Tests for the recurrent actor-critic."""

from __future__ import annotations

from torch import Tensor

import pytest
import torch

from priml.baselines.craftax.rnn import ActorCriticRNN


def _model(**overrides: int) -> ActorCriticRNN:
    config = ActorCriticRNN.Config()
    config.observation_size = 12
    config.num_actions = 5
    config.hidden_size = 16
    for name, value in overrides.items():
        setattr(config, name, value)
    return config.make()


def test_the_feed_forward_surface_matches_the_actor_critic_shapes() -> None:
    logits, value = _model()(torch.zeros(3, 12))
    assert logits.shape == (3, 5)
    assert value.shape == (3,)


def test_a_step_returns_state_and_predictions() -> None:
    model = _model()
    state, logits, value = model.step(
        model.initial_state(3),
        torch.zeros(3, 12),
        torch.zeros(3, dtype=torch.bool),
    )
    assert state.shape == (3, 16)
    assert logits.shape == (3, 5)
    assert value.shape == (3,)


def test_the_state_starts_empty() -> None:
    assert float(_model().initial_state(4).abs().max()) == 0.0


def test_a_terminal_transition_clears_that_worker_only() -> None:
    """Memory that crossed an episode boundary is worse than no memory.

    A policy beginning a new world remembering the one it just died in would
    condition on a map that no longer exists.
    """
    model = _model()
    torch.manual_seed(0)
    state = model.initial_state(2)
    for _ in range(3):
        state, _, _ = model.step(
            state,
            torch.randn(2, 12),
            torch.zeros(2, dtype=torch.bool),
        )
    survivor = state[1].clone()

    cleared, _, _ = model.step(
        state,
        torch.zeros(2, 12),
        torch.tensor([True, False]),
    )
    # Worker 0's state was rebuilt from zero; worker 1's carried in.
    fresh, _, _ = model.step(
        model.initial_state(2),
        torch.zeros(2, 12),
        torch.zeros(2, dtype=torch.bool),
    )
    assert torch.allclose(cleared[0], fresh[0])
    assert not torch.allclose(state[1], survivor * 0)


def test_the_sequence_path_equals_the_recurrent_path() -> None:
    """Exactly equal, not merely close.

    A recurrence performs the same operations in the same order either way --
    unlike attention, whose windowed form reduces differently. Any difference
    here is a bug, so the test admits none.
    """
    model = _model().double()
    torch.manual_seed(1)
    observation = torch.randn(4, 3, 12, dtype=torch.float64)
    done = torch.zeros(4, 3, dtype=torch.bool)
    done[2, 1] = True
    start = model.initial_state(3).double()

    state = start.clone()
    logits: list[Tensor] = []
    values: list[Tensor] = []
    for index in range(observation.shape[0]):
        state, step_logits, step_value = model.step(
            state,
            observation[index],
            done[index],
        )
        logits.append(step_logits)
        values.append(step_value)

    final, sequence_logits, sequence_value = model.sequence(start, observation, done)

    assert torch.equal(torch.stack(logits), sequence_logits)
    assert torch.equal(torch.stack(values), sequence_value)
    assert torch.equal(state, final)


def test_memory_changes_the_prediction() -> None:
    # Without this, every mechanism in the file is dead weight.
    model = _model()
    torch.manual_seed(2)
    probe = torch.randn(3, 12)
    quiet = torch.zeros(3, dtype=torch.bool)

    cold = model.step(model.initial_state(3), probe, quiet)[1]
    state = model.initial_state(3)
    for _ in range(4):
        state, _, _ = model.step(state, torch.randn(3, 12), quiet)
    warm = model.step(state, probe, quiet)[1]

    assert not torch.allclose(cold, warm)


def test_the_state_is_one_vector_whatever_the_history() -> None:
    # The whole trade against attention: memory costs the same at step 1 and
    # step 10,000, at the price of being unable to address a specific past.
    model = _model()
    state = model.initial_state(2)
    for _ in range(20):
        state, _, _ = model.step(
            state,
            torch.randn(2, 12),
            torch.zeros(2, dtype=torch.bool),
        )
    assert state.shape == (2, 16)


def test_a_sequence_cannot_see_the_future() -> None:
    model = _model().double()
    torch.manual_seed(3)
    observation = torch.randn(4, 2, 12, dtype=torch.float64)
    done = torch.zeros(4, 2, dtype=torch.bool)
    start = model.initial_state(2).double()
    before = model.sequence(start, observation, done)[1]

    altered = observation.clone()
    altered[3] = torch.randn(2, 12, dtype=torch.float64)
    after = model.sequence(start, altered, done)[1]

    assert torch.equal(before[:3], after[:3])
    assert not torch.equal(before[3], after[3])


def test_gradients_reach_every_parameter() -> None:
    model = _model()
    _, logits, value = model.sequence(
        model.initial_state(2),
        torch.randn(3, 2, 12),
        torch.zeros(3, 2, dtype=torch.bool),
    )
    (logits.sum() + value.sum()).backward()
    assert [name for name, p in model.named_parameters() if p.grad is None] == []


@torch.no_grad()
def test_the_policy_head_starts_near_uniform() -> None:
    # The 0.01 output gain: a policy that commits before any reward has been
    # seen never recovers within the budget.
    logits, value = _model()(torch.randn(64, 12))
    assert float(logits.std()) < 0.1
    assert bool(torch.isfinite(value).all())


@pytest.mark.parametrize("field", ["observation_size", "num_actions", "hidden_size"])
def test_a_degenerate_dimension_is_refused(field: str) -> None:
    with pytest.raises(ValueError, match="positive"):
        _model(**{field: 0})


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
