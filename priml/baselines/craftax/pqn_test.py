"""Tests for the recurrent Q-network and its exploration schedule."""

from __future__ import annotations

from torch import Tensor

import pytest
import torch

from priml.baselines.craftax.pqn import RecurrentQNetwork, epsilon_at


def _model(**overrides: int) -> RecurrentQNetwork:
    config = RecurrentQNetwork.Config()
    config.observation_size = 12
    config.num_actions = 5
    config.hidden_size = 16
    for name, value in overrides.items():
        setattr(config, name, value)
    return config.make()


def test_it_values_every_action() -> None:
    values = _model()(torch.zeros(3, 12))
    assert values.shape == (3, 5)


def test_a_step_returns_both_recurrent_tensors() -> None:
    # An LSTM carries two, unlike the GRU next door; a checkpoint that saved
    # one would resume with half a memory.
    model = _model()
    (hidden, cell), values = model.step(
        model.initial_state(3),
        torch.zeros(3, 12),
        torch.zeros(3, dtype=torch.int64),
        torch.zeros(3, dtype=torch.bool),
    )
    assert hidden.shape == (3, 16)
    assert cell.shape == (3, 16)
    assert values.shape == (3, 5)


def test_the_previous_action_changes_the_values() -> None:
    """A Q-learner needs it and a policy-gradient method does not.

    The value of a state depends on what the agent just tried, and
    epsilon-greedy exploration makes that unpredictable from the observation.
    """
    model = _model()
    model.eval()
    torch.manual_seed(0)
    observation = torch.randn(2, 12)
    quiet = torch.zeros(2, dtype=torch.bool)
    first = model.step(
        model.initial_state(2),
        observation,
        torch.zeros(2, dtype=torch.int64),
        quiet,
    )[1]
    second = model.step(
        model.initial_state(2),
        observation,
        torch.full((2,), 3, dtype=torch.int64),
        quiet,
    )[1]
    assert not torch.allclose(first, second)


@torch.no_grad()
def test_a_terminal_transition_clears_both_state_tensors() -> None:
    model = _model()
    model.eval()
    torch.manual_seed(1)
    state = model.initial_state(2)
    for _ in range(3):
        state, _ = model.step(
            state,
            torch.randn(2, 12),
            torch.zeros(2, dtype=torch.int64),
            torch.zeros(2, dtype=torch.bool),
        )
    assert float(state[0][0].abs().max()) > 0.0

    (hidden, cell), _ = model.step(
        state,
        torch.zeros(2, 12),
        torch.zeros(2, dtype=torch.int64),
        torch.tensor([True, False]),
    )
    fresh = model.step(
        model.initial_state(2),
        torch.zeros(2, 12),
        torch.zeros(2, dtype=torch.int64),
        torch.zeros(2, dtype=torch.bool),
    )[0]
    assert torch.allclose(hidden[0], fresh[0][0])
    assert torch.allclose(cell[0], fresh[1][0])


def test_the_sequence_path_equals_the_recurrent_path() -> None:
    # Exactly equal: a recurrence performs the same operations in the same
    # order either way.
    model = _model().double()
    model.eval()
    torch.manual_seed(2)
    observation = torch.randn(4, 3, 12, dtype=torch.float64)
    previous_action = torch.randint(0, 5, (4, 3))
    done = torch.zeros(4, 3, dtype=torch.bool)
    done[2, 1] = True
    hidden, cell = model.initial_state(3)
    start = (hidden.double(), cell.double())

    state = start
    values: list[Tensor] = []
    for index in range(observation.shape[0]):
        state, step_values = model.step(
            state,
            observation[index],
            previous_action[index],
            done[index],
        )
        values.append(step_values)

    _, sequence = model.sequence(start, observation, previous_action, done)
    assert torch.equal(torch.stack(values), sequence)


@torch.no_grad()
def test_the_observation_is_renormalized_before_encoding() -> None:
    # Batch renormalization on the raw observation is what absorbs the shift
    # as the policy changes under its own training.
    model = _model()
    model.train()
    torch.manual_seed(3)
    for _ in range(3):
        model(torch.randn(32, 12) * 5.0 + 2.0)
    assert float(model.normalize.running_mean.abs().max()) > 0.0


def test_gradients_reach_every_parameter() -> None:
    model = _model()
    _, values = model.sequence(
        model.initial_state(2),
        torch.randn(3, 2, 12),
        torch.zeros(3, 2, dtype=torch.int64),
        torch.zeros(3, 2, dtype=torch.bool),
    )
    values.sum().backward()
    assert [name for name, p in model.named_parameters() if p.grad is None] == []


@pytest.mark.parametrize("field", ["observation_size", "num_actions", "hidden_size"])
def test_a_degenerate_dimension_is_refused(field: str) -> None:
    with pytest.raises(ValueError, match="positive"):
        _model(**{field: 0})


def test_exploration_starts_certain_and_ends_rare() -> None:
    assert epsilon_at(0, total_updates=1_000) == 1.0
    assert epsilon_at(1_000, total_updates=1_000) == pytest.approx(0.005)


def test_exploration_decays_over_the_configured_fraction() -> None:
    """Front-loaded on purpose.

    A Q-learner has no entropy bonus, so this schedule is the whole of its
    exploration -- and a run decaying across its full length would still be
    acting half-randomly at the end.
    """
    # Reaches the floor at 10% of the run, not at the end.
    assert epsilon_at(100, total_updates=1_000) == pytest.approx(0.005)
    assert epsilon_at(50, total_updates=1_000) == pytest.approx(0.5025)


def test_exploration_never_falls_below_its_floor() -> None:
    # Some randomness forever: a purely greedy Q-learner stops discovering
    # anything it has not already valued.
    assert epsilon_at(10_000, total_updates=1_000) == pytest.approx(0.005)


@pytest.mark.parametrize(
    ("field", "value"),
    [("total_updates", 0), ("decay_fraction", 0.0), ("decay_fraction", 1.5)],
)
def test_an_invalid_schedule_is_refused(field: str, value: float) -> None:
    total_updates, decay_fraction = 100, 0.1
    if field == "total_updates":
        total_updates = int(value)
    else:
        decay_fraction = value
    with pytest.raises(ValueError, match="must"):
        epsilon_at(0, total_updates=total_updates, decay_fraction=decay_fraction)


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
