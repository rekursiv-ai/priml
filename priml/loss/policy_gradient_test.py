"""Tests for the clipped policy-gradient objectives.

The torch rule's formulas are pinned against float64 autograd of the
objective, and its outputs and gradients against a golden on any CPU.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final, TypedDict, Unpack

import math

from torch import Tensor

import pytest
import torch

from priml.loss.policy_gradient import (
    ClippedPolicyLoss,
    TorchPPO,
    categorical_entropy,
    clipped_policy_loss,
)
from priml.math.advantage import observation_aligned_advantage
from priml.testing.bfb import host_agnostic_numerics
from priml.testing.golden import assert_tensor_golden
from priml.testing.policy_gradient import (
    portable_minibatch,
    random_minibatch,
    rule_outputs,
)


_CWD: Final = Path(__file__).resolve().parent


class _TermsOverrides(TypedDict, total=False):
    log_probs: Tensor
    behavior_log_probs: Tensor
    advantages: Tensor
    values: Tensor


def _terms(**overrides: Unpack[_TermsOverrides]) -> ClippedPolicyLoss:
    return clipped_policy_loss(
        log_probs=overrides.get(
            "log_probs",
            torch.tensor([-0.70, -0.60, -0.80, -0.50]),
        ),
        behavior_log_probs=overrides.get(
            "behavior_log_probs",
            torch.tensor([-0.70, -0.60, -0.80, -0.50]),
        ),
        advantages=overrides.get(
            "advantages",
            torch.tensor([1.0, -0.5, 0.25, 0.5]),
        ),
        values=overrides.get(
            "values",
            torch.tensor([0.1, 0.2, -0.1, 0.0]),
        ),
        behavior_values=torch.tensor([0.1, 0.2, -0.1, 0.0]),
        targets=torch.tensor([0.5, 0.1, 0.4, -0.2]),
        entropy=torch.tensor([1.0, 1.0, 1.0, 1.0]),
        clip_epsilon=0.2,
    )


def test_unchanged_policy_gives_unit_ratio_and_zero_divergence() -> None:
    terms = _terms()
    assert float(terms.approx_kl) == pytest.approx(0.0)
    assert float(terms.clip_fraction) == 0.0
    # With ratio 1 the policy term is the negated mean standardized advantage,
    # which is zero by construction.
    assert float(terms.policy) == pytest.approx(0.0, abs=1e-6)


def test_value_term_is_half_mean_squared_error_inside_the_trust_region() -> None:
    terms = _terms()
    values = torch.tensor([0.1, 0.2, -0.1, 0.0])
    targets = torch.tensor([0.5, 0.1, 0.4, -0.2])
    assert float(terms.value) == pytest.approx(
        float(0.5 * ((values - targets) ** 2).mean()),
    )


def test_clipping_bounds_the_value_term_against_a_far_prediction() -> None:
    # A value 10 away from its behavior estimate is clipped to 0.2 away, so the
    # squared error is bounded by the unclipped one.
    clipped = _terms(values=torch.tensor([10.0, 0.2, -0.1, 0.0]))
    assert float(clipped.value) < float(0.5 * (10.0 - 0.5) ** 2)


def test_ratio_leaving_the_trust_region_is_reported() -> None:
    terms = _terms(log_probs=torch.tensor([-0.70, -0.60, -0.80, 0.50]))
    assert float(terms.clip_fraction) == pytest.approx(0.25)
    assert float(terms.approx_kl) > 0.0


def test_divergence_estimate_is_non_negative_in_both_directions() -> None:
    behavior = torch.tensor([-0.7, -0.6, -0.8, -0.5])
    for shift in (-0.3, 0.3):
        terms = _terms(log_probs=behavior + shift, behavior_log_probs=behavior)
        assert float(terms.approx_kl) > 0.0


def test_constant_advantages_stay_finite() -> None:
    # Zero spread would divide by zero without the epsilon floor.
    terms = _terms(advantages=torch.ones(4))
    assert math.isfinite(float(terms.policy))


def test_gradients_reach_both_heads() -> None:
    log_probs = torch.tensor([-0.7, -0.6, -0.8, -0.5], requires_grad=True)
    values = torch.tensor([0.1, 0.2, -0.1, 0.0], requires_grad=True)
    terms = _terms(log_probs=log_probs, values=values)
    (terms.policy + terms.value).backward()
    assert log_probs.grad is not None
    assert values.grad is not None
    assert float(log_probs.grad.abs().sum()) > 0.0
    assert float(values.grad.abs().sum()) > 0.0


def test_entropy_matches_closed_form_for_a_uniform_distribution() -> None:
    log_probs = torch.full((2, 5), -math.log(5.0))
    assert categorical_entropy(log_probs).tolist() == pytest.approx(
        [math.log(5.0)] * 2,
    )


def test_entropy_of_a_deterministic_distribution_is_zero() -> None:
    logits = torch.tensor([[0.0, -math.inf, -math.inf], [-math.inf, -math.inf, 0.0]])
    entropy = categorical_entropy(torch.log_softmax(logits, dim=-1))
    assert entropy.tolist() == pytest.approx([0.0, 0.0])


def test_masked_actions_do_not_poison_entropy() -> None:
    # exp(-inf) * -inf is NaN, so a masked action must be dropped, not summed.
    logits = torch.tensor([[0.0, 0.0, -math.inf], [-math.inf, 0.0, 0.0]])
    entropy = categorical_entropy(torch.log_softmax(logits, dim=-1))
    assert entropy.tolist() == pytest.approx([math.log(2.0)] * 2)


def test_masked_entropy_has_finite_gradients() -> None:
    logits = torch.tensor(
        [[0.0, 1.0, -math.inf], [1.0, 0.0, -math.inf]],
        requires_grad=True,
    )
    entropy = categorical_entropy(torch.log_softmax(logits, dim=-1))
    entropy.sum().backward()

    unmasked_logits = torch.tensor([[0.0, 1.0], [1.0, 0.0]], requires_grad=True)
    torch.distributions.Categorical(logits=unmasked_logits).entropy().sum().backward()

    assert logits.grad is not None
    assert unmasked_logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    torch.testing.assert_close(logits.grad[:, :2], unmasked_logits.grad)
    torch.testing.assert_close(
        logits.grad,
        torch.tensor(
            [[0.19661193, -0.19661193, 0.0], [-0.19661193, 0.19661193, 0.0]],
        ),
    )


def test_fully_masked_logits_remain_nonfinite() -> None:
    logits = torch.full((2, 3), -math.inf)
    entropy = categorical_entropy(torch.log_softmax(logits, dim=-1))

    assert torch.isnan(entropy).all()


@pytest.mark.parametrize("corrupt", [float("nan"), float("inf")])
def test_corrupt_entropy_inputs_remain_nonfinite(corrupt: float) -> None:
    entropy = categorical_entropy(
        torch.tensor([[0.0, corrupt, -math.inf], [corrupt, 0.0, -math.inf]]),
    )

    assert (torch.isnan(entropy) | torch.isinf(entropy)).all()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("discount", float("nan")),
        ("trace_decay", 1.5),
        ("clip_epsilon", float("inf")),
        ("entropy_coefficient", -1e-3),
    ],
)
def test_the_rule_refuses_a_coefficient_it_would_run_wrongly(
    field: str,
    value: float,
) -> None:
    config = TorchPPO.Config()
    setattr(config, field, value)
    with pytest.raises(ValueError, match=field):
        config.make()


def test_horizon_must_be_positive() -> None:
    rule = TorchPPO.Config().make()
    rule.check_horizon(2)
    rule.check_horizon(1)
    for horizon in (0, -1):
        with pytest.raises(
            ValueError,
            match=f"horizon must be positive, not {horizon}",
        ):
            rule.check_horizon(horizon)


def test_log_probs_is_the_masked_log_softmax() -> None:
    batch = random_minibatch(rows=3, horizon=4)
    rule = TorchPPO.Config().make()
    result = rule.log_probs(
        batch["decoded"],
        batch["actions"],
        batch["action_mask"],
    )
    masked = torch.where(
        batch["action_mask"] != 0,
        batch["decoded"][..., :43].float(),
        TorchPPO.Config.MASKED_LOGIT,
    )
    torch.testing.assert_close(result.logps, masked.log_softmax(-1))
    torch.testing.assert_close(
        result.new_lp,
        result.logps.gather(-1, batch["actions"].long()[..., None])[..., 0],
    )
    assert torch.equal(result.values, batch["decoded"][..., 43])
    assert result.logps.dtype == result.new_lp.dtype == torch.float32
    # An illegal action carries essentially no probability.
    assert result.logps[batch["action_mask"] == 0].max() < -1e3


def test_advantage_is_gae_from_the_next_row_with_a_zero_last_step() -> None:
    batch = random_minibatch(rows=3, horizon=6)
    config = TorchPPO.Config()
    discount, trace_decay = config.discount, config.trace_decay
    advantages, returns = config.make().advantage(
        batch["values"],
        batch["rewards"],
        batch["terminals"],
    )
    v = batch["values"].double()
    r = batch["rewards"].double()
    d = batch["terminals"].double()
    expected = torch.zeros_like(v)
    for time in range(4, -1, -1):
        continuing = 1 - d[:, time + 1]
        delta = r[:, time + 1] + discount * v[:, time + 1] * continuing - v[:, time]
        expected[:, time] = (
            delta + discount * trace_decay * expected[:, time + 1] * continuing
        )
    assert torch.equal(advantages[:, -1], torch.zeros_like(advantages[:, -1]))
    torch.testing.assert_close(advantages.double(), expected, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(returns.double(), (v + expected), rtol=1e-2, atol=1e-2)
    assert advantages.dtype == returns.dtype == torch.bfloat16


def test_advantage_runs_the_observation_aligned_estimate_in_fp32() -> None:
    batch = random_minibatch(rows=3, horizon=6)
    config = TorchPPO.Config()
    advantages, returns = config.make().advantage(
        batch["values"],
        batch["rewards"],
        batch["terminals"],
    )
    expected, expected_returns = observation_aligned_advantage(
        rewards=batch["rewards"].float(),
        values=batch["values"].float(),
        dones=batch["terminals"].float(),
        discount=config.discount,
        trace_decay=config.trace_decay,
    )
    assert torch.equal(advantages, expected.bfloat16())
    assert torch.equal(returns, expected_returns.bfloat16())


def test_the_advantages_take_the_dtype_the_values_and_rewards_promote_to() -> None:
    """fp32 rewards keep the fp32 estimate that bf16 inputs round once."""
    batch = random_minibatch(rows=3, horizon=6)
    rule = TorchPPO.Config().make()
    values, terminals = batch["values"], batch["terminals"]
    rounded = rule.advantage(values, batch["rewards"], terminals)
    exact = rule.advantage(values, batch["rewards"].float(), terminals)
    assert all(value.dtype == torch.bfloat16 for value in rounded)
    assert all(value.dtype == torch.float32 for value in exact)
    for ours, theirs in zip(exact, rounded, strict=True):
        assert torch.equal(ours.bfloat16(), theirs)
        assert not torch.equal(ours, theirs.float())


def test_loss_gradients_agree_with_autograd_of_the_objective() -> None:
    batch = random_minibatch(rows=4, horizon=8, seed=1)
    config = TorchPPO.Config()
    rule = config.make()
    logprobs = rule.log_probs(
        batch["decoded"],
        batch["actions"],
        batch["action_mask"],
    )
    advantages, returns = rule.advantage(
        logprobs.values,
        batch["rewards"],
        batch["terminals"],
    )
    loss = rule.loss(
        logprobs,
        decoded=batch["decoded"],
        actions=batch["actions"],
        old_logprobs=batch["old_logprobs"],
        advantages=advantages,
        values=batch["values"],
        returns=returns,
    )
    logits = batch["decoded"][..., :43].double().requires_grad_()
    value_pred = batch["decoded"][..., 43].double().requires_grad_()
    # The objective in float64 for autograd: the mean per row.
    clip = config.clip_epsilon
    masked = torch.where(
        batch["action_mask"] != 0,
        logits,
        TorchPPO.Config.MASKED_LOGIT,
    )
    logps = masked.log_softmax(-1)
    new_lp = logps.gather(-1, batch["actions"].long()[..., None])[..., 0]
    ratio = torch.exp(new_lp - batch["old_logprobs"].double())
    adv = advantages.double()
    policy = torch.maximum(-adv * ratio, -adv * ratio.clamp(1 - clip, 1 + clip))
    val = batch["values"].double()
    ret = returns.double()
    value_clip = config.value_clip_epsilon
    clipped = val + (value_pred - val).clamp(-value_clip, value_clip)
    value = 0.5 * torch.maximum((value_pred - ret) ** 2, (clipped - ret) ** 2)
    entropy = -(logps.exp() * logps).sum(-1)
    (
        policy + config.value_coefficient * value - config.entropy_coefficient * entropy
    ).mean().backward()
    assert logits.grad is not None
    assert value_pred.grad is not None
    torch.testing.assert_close(
        loss.grad_logits.double(),
        logits.grad,
        rtol=1e-4,
        atol=1e-7,
    )
    torch.testing.assert_close(
        loss.grad_values.double(),
        value_pred.grad,
        rtol=1e-4,
        atol=1e-7,
    )
    assert loss.losses.shape == (len(TorchPPO.Config.LOSS_NAMES),)
    # The total is the policy, value and entropy terms combined.
    torch.testing.assert_close(
        loss.losses[3],
        loss.losses[0]
        + config.value_coefficient * loss.losses[1]
        - config.entropy_coefficient * loss.losses[2],
    )


def test_the_value_clip_zeroes_the_gradient_where_the_clipped_loss_wins() -> None:
    batch = random_minibatch(rows=2, horizon=3, seed=2)
    rule = TorchPPO.Config().make()
    # The live value sits on the return but far from the rollout's value, so
    # the clipped value loses more than the unclipped: its branch wins, and it
    # is constant in the prediction.
    decoded = batch["decoded"]
    returns = decoded[..., 43]
    values = (returns.float() - 10).bfloat16()
    logprobs = rule.log_probs(decoded, batch["actions"], batch["action_mask"])
    loss = rule.loss(
        logprobs,
        decoded=decoded,
        actions=batch["actions"],
        old_logprobs=batch["old_logprobs"],
        advantages=torch.zeros_like(returns),
        values=values,
        returns=returns,
    )
    assert torch.equal(loss.grad_values, torch.zeros_like(loss.grad_values))


def test_calling_the_rule_runs_its_stages_and_backpropagates_their_gradient() -> None:
    """The call's total differentiates to :meth:`loss`'s gradients, times the seed."""
    batch = random_minibatch(rows=3, horizon=4, seed=4)
    rule = TorchPPO.Config().make()
    logprobs = rule.log_probs(batch["decoded"], batch["actions"], batch["action_mask"])
    advantages, returns = rule.advantage(
        logprobs.values,
        batch["rewards"],
        batch["terminals"],
    )
    loss = rule.loss(
        logprobs,
        decoded=batch["decoded"],
        actions=batch["actions"],
        old_logprobs=batch["old_logprobs"],
        advantages=advantages,
        values=batch["values"],
        returns=returns,
    )
    decoded = batch["decoded"].clone().requires_grad_()
    total, losses = rule(
        decoded,
        actions=batch["actions"],
        action_mask=batch["action_mask"],
        old_logprobs=batch["old_logprobs"],
        rewards=batch["rewards"],
        terminals=batch["terminals"],
        values=batch["values"],
    )
    assert torch.equal(losses, loss.losses)
    assert torch.equal(total, loss.losses[3])
    assert not losses.requires_grad
    (2 * total).backward()
    closed_form = torch.cat((loss.grad_logits, loss.grad_values[..., None]), dim=-1)
    assert decoded.grad is not None
    assert decoded.grad.dtype == torch.bfloat16
    assert torch.equal(decoded.grad, (2 * closed_form).bfloat16())


def test_the_reference_rule_matches_its_golden() -> None:
    """The torch rule on 2 rows of 3 steps with the default coefficients, frozen.

    Portable inputs inside ``host_agnostic_numerics``: every CPU computes the
    same bits.
    """
    batch = portable_minibatch(rows=2, horizon=3)
    with host_agnostic_numerics():
        outputs = rule_outputs(TorchPPO.Config().make(), batch)
    assert_tensor_golden(_CWD / "testdata" / "torch_ppo.pt", outputs)


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
