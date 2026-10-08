"""Unit tests for Craftax_Baselines' recurrent actor-critic at tiny sizes on the CPU."""

from __future__ import annotations

from typing import Final

from torch import Tensor, nn

import pytest
import torch

from priml.baselines.craftax.policies.rnn import ActorCriticRNN
from priml.model.linear import Linear
from priml.model.swiglu import relu
from priml.testing.bfb import host_agnostic_numerics
from priml.testing.cost import assert_cost_matches_torch


BATCH: Final = 2
TIME: Final = 6
OBSERVATION: Final = 7
WIDTH: Final = 5
ACTIONS: Final = 3


def test_the_default_config_is_craftax_baselines_geometry() -> None:
    config = ActorCriticRNN.Config().copy_tree().finalize()
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
    assert config.num_layers == 2
    assert config.activation is relu
    for linear in (
        config.proj_in,
        config.proj_hidden,
        config.proj_policy,
        config.proj_value,
    ):
        assert linear.bias


def test_the_parameters_are_the_embedding_the_cell_then_both_heads() -> None:
    model = _tiny().make()
    shapes = [(name, tuple(weight.shape)) for name, weight in model.named_parameters()]
    head = [
        ("0.weight", (WIDTH, WIDTH)),
        ("0.bias", (WIDTH,)),
        ("1.weight", (WIDTH, WIDTH)),
        ("1.bias", (WIDTH,)),
    ]
    assert shapes == [
        ("proj_in.weight", (WIDTH, OBSERVATION)),
        ("proj_in.bias", (WIDTH,)),
        ("cell.weight_ih", (3 * WIDTH, WIDTH)),
        ("cell.weight_hh", (3 * WIDTH, WIDTH)),
        ("cell.bias_ih", (3 * WIDTH,)),
        ("cell.bias_hh", (3 * WIDTH,)),
        *((f"actor.{name}", shape) for name, shape in head),
        ("actor.2.weight", (ACTIONS, WIDTH)),
        ("actor.2.bias", (ACTIONS,)),
        *((f"critic.{name}", shape) for name, shape in head),
        ("critic.2.weight", (1, WIDTH)),
        ("critic.2.bias", (1,)),
    ]


def test_the_carry_is_one_zero_state_per_environment() -> None:
    model = _tiny().make()
    state = model.initial_state(BATCH)
    # One layer: ``initial_state`` carries the cell's one state.
    assert state.shape == (1, BATCH, WIDTH)
    assert state.dtype == model.dtype == torch.float32
    assert not state.any()
    assert model.initial_state(BATCH, device="meta").device.type == "meta"


def test_a_float64_policy_reads_and_carries_in_float64() -> None:
    """The weights' dtype sets the observations' and the carry's; the auxiliary loss stays fp32."""
    model = _tiny().make().double()
    state = model.initial_state(BATCH)
    assert model.dtype == state.dtype == torch.float64
    # fp32 observations, as the env writes them.
    window, final, auxiliary = model.forward_sequence(
        torch.randn(BATCH, TIME, OBSERVATION),
        state,
        torch.zeros(BATCH, TIME),
    )
    assert window.dtype == final.dtype == torch.float64
    assert auxiliary.dtype == torch.float32


def test_a_step_resets_the_carry_where_an_episode_starts_and_nowhere_else() -> None:
    """A new world is begun with no memory of the one just left."""
    model = _tiny().make()
    observations = torch.randn(BATCH, OBSERVATION)
    state = torch.randn_like(model.initial_state(BATCH))
    with torch.no_grad():
        decoded, carry = model.forward_fused(observations, state, torch.tensor([1, 0]))
        fresh, fresh_carry = model.forward_fused(
            observations,
            model.initial_state(BATCH),
            None,
        )
        carried, carried_carry = model.forward_fused(observations, state, None)
    assert torch.equal(decoded[0], fresh[0])
    assert torch.equal(carry[0, 0], fresh_carry[0, 0])
    assert torch.equal(decoded[1], carried[1])
    assert torch.equal(carry[0, 1], carried_carry[0, 1])
    # What the carry remembers changes the prediction.
    assert not torch.equal(fresh[1], carried[1])


def test_a_given_carry_is_advanced_in_place() -> None:
    model = _tiny().make()
    observations = torch.randn(BATCH, OBSERVATION)
    state = torch.randn_like(model.initial_state(BATCH))
    with torch.no_grad():
        expected, advanced = model.forward_fused(observations, state.clone(), None)
        decoded, carry = model.forward_fused(observations, state, None, carry=state)
    assert carry is state
    assert torch.equal(state, advanced)
    assert torch.equal(decoded, expected)


def test_a_window_replays_the_steps_exactly() -> None:
    """The learner's window is the rollout's steps one after another, resets included.

    The window embeds and decodes every step at once where a step does one;
    in ``host_agnostic_numerics`` the two round alike.
    """
    model = _tiny().make()
    observations = torch.randn(BATCH, TIME, OBSERVATION)
    start = torch.zeros(BATCH, TIME)
    start[1, 3] = 1.0
    state = torch.randn_like(model.initial_state(BATCH))
    with torch.no_grad(), host_agnostic_numerics():
        window, final, auxiliary = model.forward_sequence(observations, state, start)
        carry = state
        steps: list[Tensor] = []
        for step in range(TIME):
            decoded, carry = model.forward_fused(
                observations[:, step],
                carry,
                start[:, step],
            )
            steps.append(decoded)
    assert window.shape == (BATCH, TIME, ACTIONS + 1)
    assert torch.equal(window, torch.stack(steps, dim=1))
    assert torch.equal(final, carry)
    assert auxiliary.dtype == torch.float32
    assert float(auxiliary) == 0.0


def test_a_window_cannot_see_its_future() -> None:
    model = _tiny().make()
    observations = torch.randn(BATCH, TIME, OBSERVATION)
    start = torch.zeros(BATCH, TIME)
    state = model.initial_state(BATCH)
    with torch.no_grad():
        before = model.forward_sequence(observations, state, start)[0]
        observations[:, -1] = torch.randn(BATCH, OBSERVATION)
        after = model.forward_sequence(observations, state, start)[0]
    assert torch.equal(before[:, :-1], after[:, :-1])
    assert not torch.equal(before[:, -1], after[:, -1])


def test_gradients_reach_every_parameter_through_the_window() -> None:
    model = _tiny().make()
    window = model.forward_sequence(
        torch.randn(BATCH, TIME, OBSERVATION),
        model.initial_state(BATCH),
        torch.zeros(BATCH, TIME),
    )[0]
    window.sum().backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert parameter.grad.any(), name


def test_the_policy_starts_near_uniform() -> None:
    """The 0.01 gain at the policy's output: logits near zero before any reward."""
    torch.manual_seed(0)
    model = _tiny().make()
    with torch.no_grad():
        decoded, _ = model.forward_fused(
            torch.randn(BATCH, OBSERVATION),
            model.initial_state(BATCH),
            None,
        )
    assert float(decoded[:, :ACTIONS].abs().max()) < 0.1


@pytest.mark.parametrize(
    ("path", "gain"),
    [
        ("proj_in", 2**0.5),
        ("actor.0", 2**0.5),
        ("actor.1", 2**0.5),
        ("actor.2", 0.01),
        ("critic.0", 2**0.5),
        ("critic.2", 1.0),
    ],
)
def test_each_projection_is_orthogonal_at_its_gain(path: str, gain: float) -> None:
    """Rows orthonormal times the gain (every weight here is at least as wide as tall)."""
    torch.manual_seed(0)
    layer = _tiny().make().get_submodule(path)
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


def test_a_head_needs_a_layer_and_one_is_enough() -> None:
    config = _tiny()
    config.num_layers = 1
    model = config.make()
    assert (len(model.actor), len(model.critic)) == (2, 2)
    config.num_layers = 0
    with pytest.raises(ValueError, match="num_layers"):
        config.make()


def test_a_feature_is_refused() -> None:
    model = _tiny().make()
    feature = torch.zeros(BATCH, WIDTH)
    with pytest.raises(ValueError, match=r"^ActorCriticRNN reads no feature$"):
        model.forward_fused(
            torch.zeros(BATCH, OBSERVATION),
            model.initial_state(BATCH),
            None,
            features=feature,
        )
    with pytest.raises(ValueError, match=r"^ActorCriticRNN reads no feature$"):
        model.forward_sequence(
            torch.zeros(BATCH, TIME, OBSERVATION),
            model.initial_state(BATCH),
            torch.zeros(BATCH, TIME),
            features=feature[:, None].expand(BATCH, TIME, WIDTH),
        )


def test_the_cost_prices_the_window_as_torch_runs_it() -> None:
    """The cell runs per step; its first step's state, the window's carry, has no gradient."""
    analytical = assert_cost_matches_torch(
        _tiny(),
        build_input=lambda: (
            torch.randn(BATCH, TIME, OBSERVATION),
            # One layer: ``ActorCriticRNN.initial_state`` carries the cell's one state.
            torch.randn(1, BATCH, WIDTH),
            (torch.rand(BATCH, TIME) < 0.3).float(),
        ),
        run=_window_sum,
        seq_len=TIME,
        batch_size=BATCH,
        dtype=None,
    )
    rows = BATCH * TIME
    # The embedding's weight gradient alone; the cell's two products per step,
    # the first state's weight gradient alone; two heads of two hidden layers.
    embedding = 2 * 2 * rows * OBSERVATION * WIDTH
    gates = 2 * BATCH * WIDTH * 3 * WIDTH
    cell = 3 * gates * TIME + 3 * gates * (TIME - 1) + 2 * gates
    heads = 3 * 2 * rows * WIDTH * (2 * 2 * WIDTH + ACTIONS + 1)
    assert analytical["flops", "matmul"].sum() == embedding + cell + heads
    # Biases: the embedding's, both gates' every step, each head layer's and
    # output's. Per unit and row: the ReLUs after the embedding and the four
    # hidden head layers, the reset, then eleven in the cell, seventeen back.
    biases = rows * (WIDTH + 2 * 3 * WIDTH + 2 * 2 * WIDTH + ACTIONS + 1)
    units = rows * WIDTH
    assert analytical["flops", "primal", "elementwise"].sum() == biases + units * (
        1 + 2 * 2 + 1 + 11
    )
    assert analytical["flops", "adjoint", "elementwise"].sum() == units * (
        1 + 2 * 2 + 1 + 17
    )
    assert analytical.bytes_state == 4 * BATCH * WIDTH


def _window_sum(model: nn.Module, inputs: tuple[Tensor, ...]) -> Tensor:
    """Sum a window's fused rows: the inputs are observations, carry and starts."""
    assert isinstance(model, ActorCriticRNN)
    observations, state, start = inputs
    return model.forward_sequence(observations, state, start)[0].sum()


def _tiny() -> ActorCriticRNN.Config:
    """Return the policy at test size: 7 inputs, a GRU of 5, heads of 2 layers, 3 actions."""
    config = ActorCriticRNN.Config()
    config.observation_size = OBSERVATION
    config.channels_hidden = WIDTH
    config.num_actions = ACTIONS
    return config


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
