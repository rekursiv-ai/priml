"""Unit tests for Craftax_Baselines' actor-critic at tiny sizes on the CPU."""

from __future__ import annotations

from typing import cast

from torch import Tensor, nn

import pytest
import torch

from priml.baselines.craftax.policies.actor_critic import ActorCritic
from priml.model.linear import Linear
from priml.testing.cost import assert_cost_matches_torch


def test_the_default_config_is_craftax_baselines_geometry() -> None:
    config = ActorCritic.Config().copy_tree().finalize()
    widths = [
        (linear.channels_in, linear.channels_out)
        for linear in (
            config.proj_in,
            config.proj_hidden,
            config.proj_policy,
            config.proj_value,
        )
    ]
    assert widths == [(8_268, 512), (512, 512), (512, 43), (512, 1)]
    for linear in (
        config.proj_in,
        config.proj_hidden,
        config.proj_policy,
        config.proj_value,
    ):
        assert linear.bias


def test_each_tower_is_its_hidden_layers_then_its_head() -> None:
    model = _tiny().make()
    assert [len(model.actor), len(model.critic)] == [3, 3]
    assert _widths(model.actor) == [4, 4, 3]
    assert _widths(model.critic) == [4, 4, 1]


def test_the_forward_contract() -> None:
    model = _tiny().make()
    observations = torch.randn(5, 6)
    state = model.initial_state(5)
    assert state.shape == (0, 5, 0)
    decoded, carry = model.forward_fused(observations, state, None)
    assert decoded.shape == (5, 4)
    assert carry is state
    # The empty carry ``initial_state`` returns: no layers and no width.
    given = torch.zeros(0, 5, 0)
    assert model.forward_fused(observations, state, None, carry=given)[1] is given
    # One observation over a window of 3 steps: a time axis of 1, expanded.
    window, final, _ = model.forward_sequence(
        observations.reshape(5, 1, 6).expand(5, 3, 6),
        state,
        torch.zeros(5, 3),
    )
    assert window.shape == (5, 3, 4)
    assert final is state
    # Each step of a window is scored alone, as the step itself is.
    assert torch.equal(window[:, 2], decoded)


def test_the_towers_are_tanh_mlps_whose_heads_fuse_into_one_row() -> None:
    model = _tiny().make()
    observations = torch.randn(3, 6)

    def run(tower: nn.ModuleList) -> Tensor:
        features = observations
        for layer in list(tower)[:-1]:
            assert isinstance(layer, Linear)
            features = torch.tanh(
                nn.functional.linear(features, layer.weight, layer.bias),
            )
        head = tower[-1]
        assert isinstance(head, Linear)
        return nn.functional.linear(features, head.weight, head.bias)

    expected = torch.cat((run(model.actor), run(model.critic)), dim=-1)
    assert torch.equal(model(observations), expected)


def test_a_bf16_policy_draws_its_init_in_fp32_then_casts() -> None:
    """Drawn in bf16, a layer's orthogonal weights would not be the fp32 draw rounded."""
    config = _tiny()
    config.dtype = torch.bfloat16
    torch.manual_seed(0)
    bf16 = config.make()
    torch.manual_seed(0)
    fp32 = _tiny().make()
    for low, full in zip(bf16.parameters(), fp32.parameters(), strict=True):
        assert low.dtype == torch.bfloat16
        assert torch.equal(low, full.bfloat16())
    assert bf16.initial_state(2).dtype == torch.bfloat16


def test_the_critic_shares_no_weight_with_the_actor() -> None:
    model = _tiny().make()
    model(torch.randn(3, 6))[:, -1].sum().backward()
    for parameter in model.actor.parameters():
        assert parameter.grad is not None
        assert not parameter.grad.any()
    for parameter in model.critic.parameters():
        assert parameter.grad is not None
        assert parameter.grad.any()


@pytest.mark.parametrize(
    ("tower", "index", "gain"),
    [
        ("actor", 0, 2**0.5),
        ("actor", 1, 2**0.5),
        ("actor", 2, 0.01),
        ("critic", 0, 2**0.5),
        ("critic", 2, 1.0),
    ],
)
def test_each_weight_is_orthogonal_at_its_layers_gain(
    tower: str,
    index: int,
    gain: float,
) -> None:
    """Rows orthonormal times the gain (every weight here is at least as wide as tall)."""
    torch.manual_seed(0)
    layer = cast("object", getattr(_tiny().make(), tower)[index])
    assert isinstance(layer, Linear)
    rows = layer.weight.shape[0]
    torch.testing.assert_close(
        layer.weight @ layer.weight.T,
        gain**2 * torch.eye(rows),
        atol=1e-5 * gain**2,
        rtol=0.0,
    )
    assert layer.bias is not None
    assert not layer.bias.any()


def test_the_towers_draw_different_weights() -> None:
    torch.manual_seed(0)
    model = _tiny().make()
    actor, critic = model.actor[0], model.critic[0]
    assert isinstance(actor, Linear)
    assert isinstance(critic, Linear)
    assert not torch.equal(actor.weight, critic.weight)


def test_a_tower_needs_a_layer() -> None:
    config = _tiny()
    config.num_layers = 0
    with pytest.raises(ValueError, match="num_layers"):
        config.make()


def test_the_cost_prices_both_towers_as_torch_runs_them() -> None:
    """The observations take no gradient, so each first layer forms its weight's alone."""
    assert_cost_matches_torch(
        _tiny(),
        build_input=lambda: torch.randn(2, 5, 6),
        run=_window_sum,
        seq_len=5,
        batch_size=2,
        dtype=None,
    )


def _window_sum(model: nn.Module, observations: Tensor) -> Tensor:
    """Sum a window's fused rows."""
    assert isinstance(model, ActorCritic)
    return model(observations).sum()


def _widths(tower: nn.ModuleList) -> list[int]:
    widths: list[int] = []
    for layer in tower:
        assert isinstance(layer, Linear)
        widths.append(layer.out_features)
    return widths


def _tiny() -> ActorCritic.Config:
    """Return the actor-critic at test size: 6 inputs, 2 layers of 4, 3 actions."""
    config = ActorCritic.Config()
    config.observation_size = 6
    config.channels_hidden = 4
    config.num_layers = 2
    config.num_actions = 3
    return config


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
