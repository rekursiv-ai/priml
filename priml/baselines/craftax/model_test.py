"""Tests for the policy and value network."""

from __future__ import annotations

import math

import pytest
import torch

from priml.baselines.craftax.model import ActorCritic


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
    assert bool(model.policy[0].weight.grad.any())
    assert bool(model.value[0].weight.grad.any())


def test_depth_and_width_follow_the_configuration() -> None:
    model = _model(channels_in=24, num_layers=3)
    assert model.policy[0].out_features == 24
    # Three hidden layers, each followed by an activation, then the head.
    assert len(model.policy) == 7


def test_weights_are_orthogonally_initialized() -> None:
    # Orthogonal columns keep activations from collapsing or exploding as
    # they pass through a deep tanh stack.
    weight = _model(channels_in=32, observation_size=32).policy[0].weight.detach()
    product = weight @ weight.T
    identity = torch.eye(product.shape[0]) * (math.sqrt(2.0) ** 2)
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


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
