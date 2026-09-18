"""Tests for the policy and value network."""

from __future__ import annotations

from typing import cast

from torch import Tensor, nn

import pytest
import torch

from priml.baselines.craftax.model import ActorCritic
from priml.testing.cost import assert_cost_matches_torch


def _model(**overrides: int) -> ActorCritic:
    config = ActorCritic.Config()
    config.observation_size = 32
    config.num_actions = 5
    config.channels_in = 16
    config.num_layers = 2
    for name, value in overrides.items():
        setattr(config, name, value)
    return config.make()


def test_it_scores_every_action_and_values_the_state() -> None:
    logits, value = _model()(torch.randn(3, 32))
    assert logits.shape == (3, 5)
    assert value.shape == (3,)


def test_the_towers_are_independent() -> None:
    # A shared trunk would tie the critic's features to the policy's; these
    # are deliberately separate parameters.
    model = _model()
    policy_parameters = {id(p) for p in model.policy.parameters()}
    value_parameters = {id(p) for p in model.value.parameters()}
    assert not policy_parameters & value_parameters


def test_the_initial_policy_is_nearly_uniform() -> None:
    # The output head is scaled down so early updates are not spent undoing
    # an arbitrary initial preference.
    with torch.no_grad():
        logits, _ = _model(observation_size=64, num_actions=8)(torch.randn(256, 64))
    assert float(logits.softmax(-1).max(-1).values.mean()) < 0.2


def test_the_value_head_is_not_squashed_at_initialization() -> None:
    with torch.no_grad():
        _, value = _model()(torch.randn(256, 32))
    assert float(value.std()) > 0.0


def test_gradients_reach_both_towers() -> None:
    model = _model()
    logits, value = model(torch.randn(4, 32))
    (logits.sum() + value.sum()).backward()
    assert all(p.grad is not None for p in model.parameters())
    policy_layer = model.policy[0]
    value_layer = model.value[0]
    assert isinstance(policy_layer, torch.nn.Linear)
    assert isinstance(value_layer, torch.nn.Linear)
    assert policy_layer.weight.grad is not None
    assert value_layer.weight.grad is not None
    assert bool(policy_layer.weight.grad.any())
    assert bool(value_layer.weight.grad.any())


def test_depth_and_width_follow_the_configuration() -> None:
    model = _model(channels_in=24, num_layers=3)
    policy_layer = model.policy[0]
    assert isinstance(policy_layer, torch.nn.Linear)
    assert policy_layer.out_features == 24
    # Three hidden layers, each followed by an activation, then the head.
    assert len(model.policy) == 7


def test_weights_are_orthogonally_initialized() -> None:
    # Orthogonal columns keep activations from collapsing or exploding as
    # they pass through a deep tanh stack.
    layer = _model(channels_in=32, observation_size=32).policy[0]
    assert isinstance(layer, torch.nn.Linear)
    weight = layer.weight.detach()
    product = weight @ weight.T
    identity = torch.eye(product.shape[0]) * ((2.0**0.5) ** 2)
    assert torch.allclose(product, identity, atol=1e-5)


def test_biases_start_at_zero() -> None:
    biases = [
        module.bias
        for module in _model().modules()
        if isinstance(module, torch.nn.Linear)
    ]
    assert biases
    assert all(bias is not None and not bool(bias.any()) for bias in biases)


@pytest.mark.parametrize("field", ["observation_size", "num_actions", "channels_in"])
def test_a_degenerate_geometry_is_refused(field: str) -> None:
    config = ActorCritic.Config()
    setattr(config, field, 0)
    with pytest.raises(ValueError, match="positive"):
        config.make()


def test_it_defaults_to_the_environment_geometry() -> None:
    config = ActorCritic.Config()
    assert config.observation_size == 8_268
    assert config.num_actions == 43


def _scored(model: nn.Module, observation: Tensor) -> Tensor:
    """Reduce both towers' outputs to one scalar for the backward pass."""
    logits, value = cast(ActorCritic, model)(observation)
    return logits.sum() + value.sum()


def test_the_cost_matches_torch_and_prices_the_tanh_towers() -> None:
    """One token is one observation scored by both towers.

    The observation carries a gradient so torch runs the full adjoint of
    the first layer, which is how the matmul primitive prices every input.
    """
    config = ActorCritic.Config()
    config.observation_size = 12
    config.num_actions = 5
    config.channels_in = 16
    config.num_layers = 2
    analytical = assert_cost_matches_torch(
        config,
        build_input=lambda: torch.randn(3, 12, requires_grad=True),
        seq_len=1,
        batch_size=3,
        num_tokens=1 * 3,
        dtype=None,
        run=_scored,
    )
    weights = 2 * (12 * 16 + 16 * 16) + 16 * 5 + 16 * 1
    biases = 2 * (16 + 16) + 5 + 1
    assert analytical.params == weights + biases
    assert analytical["flops", "primal", "matmul"].sum() == 2 * weights
    # Every bias add, then one tanh per hidden unit of both towers; the
    # adjoint is ``g * (1 - t**2)`` on the saved output.
    hidden_units = 2 * 2 * 16
    assert analytical["flops", "primal", "elementwise"].sum() == biases + hidden_units
    assert analytical["flops", "adjoint", "elementwise"].sum() == 3 * hidden_units
    assert analytical["flops", "adjoint", "reduction"].sum() == biases * 2 / 3
    assert analytical.bytes_state == 0


def test_cost_accounts_for_operand_bytes_and_dtype() -> None:
    config = ActorCritic.Config()
    config.observation_size = 3
    config.channels_in = 2
    config.num_actions = 4
    config.num_layers = 1
    counted = config.cost(seq_len=1, batch_size=4, dtype=torch.bfloat16)
    assert counted["bytes", "primal", "matmul"].sum() == 2 * (
        2 * (3 + 2 + 6 / 4) + (2 + 4 + 8 / 4) + (2 + 1 + 2 / 4)
    )
    wide = config.cost(seq_len=1, batch_size=4, dtype=None)
    assert (
        wide["bytes", torch.float32].sum() == 2 * counted["bytes", torch.bfloat16].sum()
    )
    assert wide["flops"].sum() == counted["flops"].sum()


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
