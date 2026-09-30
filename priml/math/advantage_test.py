"""Tests for the generalized advantage estimator."""

from __future__ import annotations

import pytest
import torch

from priml.math.advantage import (
    explained_variance,
    generalized_advantage,
    observation_aligned_advantage,
    q_lambda_targets,
)


def test_matches_hand_computed_recursion() -> None:
    # Two steps, no terminal; env k scales every input by k, so its outputs
    # scale by k too. Backwards, for the first env:
    #   delta_1 = 2.0 + 0.5*0.1 - 0.2 = 1.85, trace_1 = 1.85
    #   delta_0 = 1.0 + 0.5*0.2 - 0.4 = 0.70
    #   trace_0 = 0.70 + 0.5*0.5*1.85 = 1.1625.
    scale = torch.tensor([1.0, 2.0, 3.0])
    advantages, targets = generalized_advantage(
        rewards=torch.tensor([[1.0], [2.0]]) * scale,
        values=torch.tensor([[0.4], [0.2]]) * scale,
        dones=torch.zeros(2, 3),
        last_value=0.1 * scale,
        discount=0.5,
        trace_decay=0.5,
    )
    torch.testing.assert_close(advantages, torch.tensor([[1.1625], [1.85]]) * scale)
    torch.testing.assert_close(targets, torch.tensor([[1.5625], [2.05]]) * scale)


def test_terminal_step_blocks_credit_from_the_future() -> None:
    # Step 0 is terminal, so its bootstrap and carried trace both vanish and
    # the advantage collapses to the immediate residual 1.0 - 0.4. Env k
    # scales every input by k.
    scale = torch.tensor([1.0, 2.0, 3.0])
    advantages, _ = generalized_advantage(
        rewards=torch.tensor([[1.0], [2.0]]) * scale,
        values=torch.tensor([[0.4], [0.2]]) * scale,
        dones=torch.tensor([[1.0], [0.0]]).expand(2, 3),
        last_value=0.1 * scale,
        discount=0.5,
        trace_decay=0.5,
    )
    torch.testing.assert_close(advantages, torch.tensor([[0.6], [1.85]]) * scale)


def test_boolean_dones_behave_as_indicators() -> None:
    rewards = torch.tensor([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]])
    values = torch.tensor([[0.4, 0.3, 0.2], [0.2, 0.1, 0.5]])
    last_value = torch.tensor([0.1, 0.7, 0.9])
    dones = torch.tensor([[True, False, True], [False, True, True]])
    from_bool, _ = generalized_advantage(
        rewards=rewards,
        values=values,
        dones=dones,
        last_value=last_value,
        discount=0.5,
        trace_decay=0.5,
    )
    from_float, _ = generalized_advantage(
        rewards=rewards,
        values=values,
        dones=dones.float(),
        last_value=last_value,
        discount=0.5,
        trace_decay=0.5,
    )
    assert torch.equal(from_bool, from_float)


def test_trace_decay_one_recovers_the_discounted_return() -> None:
    rewards = torch.tensor([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
    values = torch.zeros(3, 2)
    _, targets = generalized_advantage(
        rewards=rewards,
        values=values,
        dones=torch.zeros(3, 2),
        last_value=torch.zeros(2),
        discount=0.5,
        trace_decay=1.0,
    )
    # With zero baselines and no truncation the target IS the discounted sum.
    assert targets[:, 0].tolist() == pytest.approx([1 + 0.5 * 2 + 0.25 * 3, 3.5, 3])
    assert targets[:, 1].tolist() == pytest.approx([4 + 0.5 * 5 + 0.25 * 6, 8, 6])


def test_multiple_environments_are_independent() -> None:
    # The second env terminates at step 0, the third at step 1, the first
    # never; the columns must not influence each other.
    advantages, _ = generalized_advantage(
        rewards=torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]),
        values=torch.tensor([[0.4, 0.4, 0.4], [0.2, 0.2, 0.2]]),
        dones=torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        last_value=torch.tensor([0.1, 0.1, 0.1]),
        discount=0.5,
        trace_decay=0.5,
    )
    assert advantages[:, 0].tolist() == pytest.approx([1.1625, 1.85])
    assert advantages[:, 1].tolist() == pytest.approx([0.6, 1.85])
    assert advantages[:, 2].tolist() == pytest.approx([1.15, 1.8])


def test_observation_aligned_advantage_reads_the_next_steps_reward() -> None:
    # The hand-computed recursion above, one step later: step 0's reward and
    # terminal arrived with the first observation and are never read, and the
    # last step, with nothing after it, is left at zero.
    advantages, targets = observation_aligned_advantage(
        rewards=torch.tensor([[9.0, 1.0, 2.0], [9.0, 1.0, 2.0]]),
        values=torch.tensor([[0.4, 0.2, 0.1], [0.4, 0.2, 0.1]]),
        dones=torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        discount=0.5,
        trace_decay=0.5,
    )
    assert advantages.tolist() == [pytest.approx([1.1625, 1.85, 0.0])] * 2
    assert targets.tolist() == [pytest.approx([1.5625, 2.05, 0.1])] * 2


def test_observation_aligned_terminal_blocks_the_bootstrap_before_it() -> None:
    # The transition into the last observation ended an episode: step 1 takes
    # its reward alone, and step 0 carries only that.
    advantages, _ = observation_aligned_advantage(
        rewards=torch.tensor([[9.0, 1.0, 2.0], [9.0, 1.0, 2.0]]),
        values=torch.tensor([[0.4, 0.2, 0.1], [0.4, 0.2, 0.1]]),
        dones=torch.tensor([[False, False, True], [False, False, False]]),
        discount=0.5,
        trace_decay=0.5,
    )
    assert advantages[0].tolist() == pytest.approx([1.15, 1.8, 0.0])
    assert advantages[1].tolist() == pytest.approx([1.1625, 1.85, 0.0])


def test_observation_aligned_advantage_rounds_as_a_fused_fp32_walk() -> None:
    """At PufferLib's Craftax coefficients, the bits of a kernel's fp32 recursion.

    The kernel takes both coefficients as fp32 arguments and forms the decay
    from them; at this pair that product rounds as the float64 one does.
    """
    discount, trace_decay = 0.999414682, 0.801190972
    generator = torch.Generator().manual_seed(0)
    rewards = torch.randn(256, 128, generator=generator)
    values = torch.randn(256, 128, generator=generator)
    dones = (torch.rand(256, 128, generator=generator) < 0.1).float()
    advantages, targets = observation_aligned_advantage(
        rewards=rewards,
        values=values,
        dones=dones,
        discount=discount,
        trace_decay=trace_decay,
    )
    gamma = torch.tensor(discount)
    decay = gamma * torch.tensor(trace_decay)
    expected = torch.zeros_like(values)
    trace = torch.zeros(256)
    for step in range(126, -1, -1):
        continuing = 1.0 - dones[:, step + 1]
        bootstrap = gamma * values[:, step + 1] * continuing + rewards[:, step + 1]
        trace = (bootstrap - values[:, step]) + (decay * trace) * continuing
        expected[:, step] = trace
    assert torch.equal(advantages, expected)
    assert torch.equal(targets, expected + values)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_observation_aligned_advantage_runs_in_the_inputs_dtype(
    dtype: torch.dtype,
) -> None:
    advantages, targets = observation_aligned_advantage(
        rewards=torch.ones(2, 4, dtype=dtype),
        values=torch.zeros(2, 4, dtype=dtype),
        dones=torch.zeros(2, 4, dtype=dtype),
        discount=0.99,
        trace_decay=0.95,
    )
    assert advantages.dtype == targets.dtype == dtype


def test_explained_variance_reports_fit_and_constant_targets() -> None:
    values = torch.tensor([1.0, 2.0, 3.0])
    assert float(explained_variance(values, values)) == pytest.approx(1.0)
    assert float(explained_variance(values, torch.full((3,), 2.0))) == pytest.approx(
        0.0,
    )
    # Predicting the mean explains nothing but is not negative.
    assert float(explained_variance(torch.full((3,), 2.0), values)) == pytest.approx(
        0.0,
    )


def test_q_lambda_bootstraps_from_the_greedy_action() -> None:
    """The point of a Q-learner: the target needs no separate critic.

    One step, no termination, so the target is exactly the reward plus the
    discounted best Q-value available next.
    """
    targets = q_lambda_targets(
        rewards=torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        q_values=torch.tensor(
            [
                [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
                [[2.0, 7.0, 4.0, 1.0], [5.0, 0.0, 6.0, 3.0], [1.0, 2.0, 3.0, 4.0]],
                [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
            ],
        ),
        dones=torch.zeros(2, 3),
        discount=0.5,
        trace_decay=0.0,
    )
    assert targets.tolist() == [
        [1.0 + 0.5 * 7.0, 2.0 + 0.5 * 6.0, 3.0 + 0.5 * 4.0],
        [4.0, 5.0, 6.0],
    ]


def test_q_lambda_stops_at_a_terminal_step() -> None:
    # Not merely a zeroed bootstrap: there is no next state to be greedy in,
    # so the target is the reward and nothing else.
    targets = q_lambda_targets(
        rewards=torch.tensor([[3.0, 4.0, 5.0, 6.0], [7.0, 8.0, 9.0, 10.0]]),
        q_values=torch.full((3, 4, 5), 100.0),
        dones=torch.ones(2, 4),
        discount=0.99,
        trace_decay=0.9,
    )
    assert targets.tolist() == [[3.0, 4.0, 5.0, 6.0], [7.0, 8.0, 9.0, 10.0]]


def test_q_lambda_at_zero_decay_is_the_one_step_target() -> None:
    # The endpoints are what make the mixing factor meaningful, so both are
    # pinned rather than assumed.
    rewards = torch.arange(8.0).reshape(2, 4)
    q_values = torch.arange(60.0).reshape(3, 4, 5)
    dones = torch.zeros(2, 4)
    targets = q_lambda_targets(
        rewards=rewards,
        q_values=q_values,
        dones=dones,
        discount=0.5,
        trace_decay=0.0,
    )
    # The greedy next Q-values are 24, 29, 34 and 39.
    assert targets[0].tolist() == [
        0 + 0.5 * 24,
        1 + 0.5 * 29,
        2 + 0.5 * 34,
        3 + 0.5 * 39,
    ]


def test_q_lambda_credit_does_not_cross_an_episode_boundary() -> None:
    # A reward after a terminal step must not raise the target before it.
    rewards = torch.tensor([[1.0, 2.0, 3.0, 4.0], [50.0, 60.0, 70.0, 80.0]])
    q_values = torch.zeros(3, 4, 5)
    ended = q_lambda_targets(
        rewards=rewards,
        q_values=q_values,
        dones=torch.tensor([[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]]),
        discount=0.99,
        trace_decay=1.0,
    )
    assert ended[0].tolist() == [1.0, 2.0, 3.0, 4.0]


def test_q_lambda_refuses_a_missing_bootstrap_step() -> None:
    # Without the extra Q-value the last transition has nothing to bootstrap
    # from, and silently truncating would bias every target in the rollout.
    with pytest.raises(ValueError, match="one more Q-value"):
        q_lambda_targets(
            rewards=torch.zeros(2, 3),
            q_values=torch.zeros(2, 3, 4),
            dones=torch.zeros(2, 3),
            discount=0.99,
            trace_decay=0.9,
        )


def test_q_lambda_refuses_an_empty_sequence() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        q_lambda_targets(
            rewards=torch.zeros(0, 3),
            q_values=torch.zeros(2, 3, 4),
            dones=torch.zeros(0, 3),
            discount=0.99,
            trace_decay=0.9,
        )


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
