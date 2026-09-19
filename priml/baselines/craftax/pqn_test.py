"""Tests for the recurrent Q-network and its exploration schedule."""

from __future__ import annotations

from typing import cast

from torch import Tensor, nn

import pytest
import torch

from priml.baselines.craftax.pqn import RecurrentQNetwork, epsilon_at
from priml.cost import cost
from priml.model.norm import BatchRenorm, LayerNorm
from priml.testing.cost import assert_cost_matches_torch


def _model(**overrides: int) -> RecurrentQNetwork:
    config = RecurrentQNetwork.Config()
    config.observation_size = 12
    config.num_actions = 5
    config.channels_in = 16
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


@pytest.mark.parametrize("field", ["observation_size", "num_actions", "channels_in"])
def test_a_degenerate_dimension_is_refused(field: str) -> None:
    with pytest.raises(ValueError, match="positive"):
        _model(**{field: 0})


def _stepped(model: nn.Module, inputs: tuple[Tensor, ...]) -> Tensor:
    """Take one recurrent step from a carried state; reduce the Q-values."""
    observation, hidden, cell = inputs
    envs = observation.shape[0]
    _, q_values = cast(RecurrentQNetwork, model).step(
        (hidden, cell),
        observation,
        torch.zeros(envs, dtype=torch.int64),
        torch.zeros(envs, dtype=torch.bool),
    )
    return q_values.sum()


def test_the_cost_matches_torch_and_prices_the_lstm_step() -> None:
    """One token is one recurrent step of one worker.

    Both carried tensors have gradients, as at every step after the first of
    :meth:`sequence`, so torch counts the state-side gate matmul's full
    adjoint: measured, ``nn.LSTMCell(4, 6)`` on three rows is 4320 FLOPs
    with a differentiable state and 3456 without.
    """
    config = RecurrentQNetwork.Config()
    config.observation_size = 12
    config.num_actions = 5
    config.channels_in = 16
    analytical = assert_cost_matches_torch(
        config,
        build_input=lambda: (
            torch.randn(3, 12, requires_grad=True),
            torch.randn(3, 16, requires_grad=True),
            torch.randn(3, 16, requires_grad=True),
        ),
        seq_len=1,
        batch_size=3,
        dtype=None,
        run=_stepped,
    )
    gates = 4 * 16 * (16 + 5) + 4 * 16 * 16
    weights = 12 * 16 + gates + 16 * 5
    biases = 16 + 2 * 4 * 16 + 5
    renorm = BatchRenorm.Config()
    renorm.channels_in = 12
    norms = cost(renorm, seq_len=1, batch_size=3, dtype=None) + cost(
        LayerNorm.Config(16, elementwise_affine=True),
        seq_len=1,
        batch_size=3,
        dtype=None,
    )
    assert analytical.params == weights + biases + norms.params
    assert analytical["flops", "primal", "matmul"].sum() == 3 * 2 * weights
    # Beyond the norms: biases, the encoder's ReLU, the reset of both carried
    # tensors, and thirteen operations per LSTM unit.
    assert analytical["flops", "primal", "elementwise"].sum() - norms[
        "flops",
        "primal",
        "elementwise",
    ].sum() == 3 * (biases + 16 + 2 * 16 + 13 * 16)
    assert analytical["flops", "adjoint", "elementwise"].sum() - norms[
        "flops",
        "adjoint",
        "elementwise",
    ].sum() == 3 * (16 + 2 * 16 + 22 * 16)
    # The one-hot previous action is written, not computed.
    assert analytical["bytes", "primal", "selection"].sum() == 4 * 3 * 5
    assert analytical.bytes_state == 4 * 3 * 2 * 16


def test_cost_accounts_for_recurrent_operand_bytes_and_dtype() -> None:
    config = RecurrentQNetwork.Config()
    config.observation_size = 3
    config.channels_in = 2
    config.num_actions = 4
    narrow = config.cost(seq_len=1, batch_size=4, dtype=torch.bfloat16)
    wide = config.cost(seq_len=1, batch_size=4, dtype=None)
    assert narrow.bytes_state == 32
    assert wide.bytes_state == 64
    assert narrow["bytes", "primal", "selection"].sum() == 4 * 8
    assert (
        wide["bytes", torch.float32].sum() == 2 * narrow["bytes", torch.bfloat16].sum()
    )
    assert wide["flops"].sum() == narrow["flops"].sum()


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
