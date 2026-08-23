"""Tests for the generalized advantage estimator."""

from __future__ import annotations

import pytest
import torch

from priml.math.advantage import (
    explained_variance,
    generalized_advantage,
    q_lambda_targets,
)


def test_matches_hand_computed_recursion() -> None:
    # Two steps, one env, no terminal. Backwards:
    #   delta_1 = 2.0 + 0.5*0.1 - 0.2 = 1.85, trace_1 = 1.85
    #   delta_0 = 1.0 + 0.5*0.2 - 0.4 = 0.70
    #   trace_0 = 0.70 + 0.5*0.5*1.85 = 1.1625
    advantages, targets = generalized_advantage(
        rewards=torch.tensor([[1.0], [2.0]]),
        values=torch.tensor([[0.4], [0.2]]),
        dones=torch.tensor([[0.0], [0.0]]),
        last_value=torch.tensor([0.1]),
        discount=0.5,
        trace_decay=0.5,
    )
    assert advantages.flatten().tolist() == pytest.approx([1.1625, 1.85])
    assert targets.flatten().tolist() == pytest.approx([1.5625, 2.05])


def test_terminal_step_blocks_credit_from_the_future() -> None:
    # Step 0 is terminal, so its bootstrap and carried trace both vanish and
    # the advantage collapses to the immediate residual 1.0 - 0.4.
    advantages, _ = generalized_advantage(
        rewards=torch.tensor([[1.0], [2.0]]),
        values=torch.tensor([[0.4], [0.2]]),
        dones=torch.tensor([[1.0], [0.0]]),
        last_value=torch.tensor([0.1]),
        discount=0.5,
        trace_decay=0.5,
    )
    assert advantages.flatten().tolist() == pytest.approx([0.6, 1.85])


def test_boolean_dones_behave_as_indicators() -> None:
    rewards = torch.tensor([[1.0], [2.0]])
    values = torch.tensor([[0.4], [0.2]])
    last_value = torch.tensor([0.1])
    from_bool, _ = generalized_advantage(
        rewards=rewards,
        values=values,
        dones=torch.tensor([[True], [False]]),
        last_value=last_value,
        discount=0.5,
        trace_decay=0.5,
    )
    from_float, _ = generalized_advantage(
        rewards=rewards,
        values=values,
        dones=torch.tensor([[1.0], [0.0]]),
        last_value=last_value,
        discount=0.5,
        trace_decay=0.5,
    )
    assert torch.equal(from_bool, from_float)


def test_trace_decay_one_recovers_the_discounted_return() -> None:
    rewards = torch.tensor([[1.0], [2.0], [3.0]])
    values = torch.zeros(3, 1)
    _, targets = generalized_advantage(
        rewards=rewards,
        values=values,
        dones=torch.zeros(3, 1),
        last_value=torch.zeros(1),
        discount=0.5,
        trace_decay=1.0,
    )
    # With zero baselines and no truncation the target IS the discounted sum.
    assert targets.flatten().tolist() == pytest.approx([1 + 0.5 * 2 + 0.25 * 3, 3.5, 3])


def test_multiple_environments_are_independent() -> None:
    # The second env terminates at step 0, the first does not; the columns must
    # not influence each other.
    advantages, _ = generalized_advantage(
        rewards=torch.tensor([[1.0, 1.0], [2.0, 2.0]]),
        values=torch.tensor([[0.4, 0.4], [0.2, 0.2]]),
        dones=torch.tensor([[0.0, 1.0], [0.0, 0.0]]),
        last_value=torch.tensor([0.1, 0.1]),
        discount=0.5,
        trace_decay=0.5,
    )
    assert advantages[:, 0].tolist() == pytest.approx([1.1625, 1.85])
    assert advantages[:, 1].tolist() == pytest.approx([0.6, 1.85])


def test_explained_variance_reports_fit_and_constant_targets() -> None:
    values = torch.tensor([1.0, 2.0, 3.0])
    assert float(explained_variance(values, values)) == pytest.approx(1.0)
    assert float(explained_variance(values, torch.full((3,), 2.0))) == pytest.approx(
        0.0
    )
    # Predicting the mean explains nothing but is not negative.
    assert float(explained_variance(torch.full((3,), 2.0), values)) == pytest.approx(
        0.0
    )


def test_q_lambda_bootstraps_from_the_greedy_action() -> None:
    """The point of a Q-learner: the target needs no separate critic.

    One step, no termination, so the target is exactly the reward plus the
    discounted best Q-value available next.
    """
    targets = q_lambda_targets(
        rewards=torch.tensor([[1.0]]),
        q_values=torch.tensor([[[0.0, 0.0]], [[2.0, 7.0]]]),
        dones=torch.tensor([[0.0]]),
        discount=0.5,
        trace_decay=0.9,
    )
    assert targets.tolist() == [[1.0 + 0.5 * 7.0]]


def test_q_lambda_stops_at_a_terminal_step() -> None:
    # Not merely a zeroed bootstrap: there is no next state to be greedy in,
    # so the target is the reward and nothing else.
    targets = q_lambda_targets(
        rewards=torch.tensor([[3.0]]),
        q_values=torch.tensor([[[0.0]], [[100.0]]]),
        dones=torch.tensor([[1.0]]),
        discount=0.99,
        trace_decay=0.9,
    )
    assert targets.tolist() == [[3.0]]


def test_q_lambda_at_zero_decay_is_the_one_step_target() -> None:
    # The endpoints are what make the mixing factor meaningful, so both are
    # pinned rather than assumed.
    rewards = torch.tensor([[1.0], [2.0]])
    q_values = torch.tensor([[[1.0]], [[3.0]], [[5.0]]])
    dones = torch.zeros(2, 1)
    targets = q_lambda_targets(
        rewards=rewards,
        q_values=q_values,
        dones=dones,
        discount=0.5,
        trace_decay=0.0,
    )
    assert targets[0].tolist() == [1.0 + 0.5 * 3.0]


def test_q_lambda_credit_does_not_cross_an_episode_boundary() -> None:
    # A reward after a terminal step must not raise the target before it.
    rewards = torch.tensor([[1.0], [50.0]])
    q_values = torch.zeros(3, 1, 2)
    ended = q_lambda_targets(
        rewards=rewards,
        q_values=q_values,
        dones=torch.tensor([[1.0], [0.0]]),
        discount=0.99,
        trace_decay=1.0,
    )
    assert ended[0].tolist() == [1.0]


def test_q_lambda_refuses_a_missing_bootstrap_step() -> None:
    # Without the extra Q-value the last transition has nothing to bootstrap
    # from, and silently truncating would bias every target in the rollout.
    with pytest.raises(ValueError, match="one more Q-value"):
        q_lambda_targets(
            rewards=torch.zeros(3, 2),
            q_values=torch.zeros(3, 2, 4),
            dones=torch.zeros(3, 2),
            discount=0.99,
            trace_decay=0.9,
        )


def test_q_lambda_refuses_an_empty_sequence() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        q_lambda_targets(
            rewards=torch.zeros(0, 2),
            q_values=torch.zeros(1, 2, 4),
            dones=torch.zeros(0, 2),
            discount=0.99,
            trace_decay=0.9,
        )


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
