"""Tests for the recurrent Q-network, its exploration and its previous-action feature."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import math
import sys

import numpy as np
import pytest
import torch

from priml.baselines.craftax.policies.pqn import (
    EpsilonGreedy,
    GreedySampler,
    PreviousAction,
    PreviousActionStep,
    RecurrentQNetwork,
    epsilon_at,
)
from priml.baselines.craftax.rollout import TorchPhiloxSampler
from priml.cost import cost
from priml.model.norm import BatchRenorm, LayerNorm
from priml.testing.cost import assert_cost_matches_torch


if TYPE_CHECKING:
    from torch import Tensor, nn

OBSERVATION: Final = 7
ACTIONS: Final = 5
WIDTH: Final = 6
ENVS: Final = 3
TIME: Final = 4


def _model() -> RecurrentQNetwork:
    config = RecurrentQNetwork.Config()
    config.observation_size = OBSERVATION
    config.num_actions = ACTIONS
    config.channels_hidden = WIDTH
    return config.make()


# The feature is one id per row: ``PreviousAction``'s width.
def _previous(*shape: int, action: int = 0) -> Tensor:
    return torch.full((*shape, 1), float(action))


def test_it_values_every_action_then_their_maximum() -> None:
    model = _model()
    torch.manual_seed(0)
    decoded, _ = model.forward_fused(
        torch.randn(ENVS, OBSERVATION),
        model.initial_state(ENVS),
        None,
        features=_previous(ENVS),
    )
    assert decoded.shape == (ENVS, ACTIONS + 1)
    assert torch.equal(decoded[:, -1], decoded[:, :-1].amax(dim=-1))


def test_a_step_carries_both_recurrent_tensors() -> None:
    # An LSTM carries two, unlike the GRUs elsewhere; a carry that kept one
    # would resume with half a memory.
    model = _model()
    _, state = model.forward_fused(
        torch.zeros(ENVS, OBSERVATION),
        model.initial_state(ENVS),
        None,
        features=_previous(ENVS),
    )
    assert state.shape == (2, ENVS, WIDTH)
    assert float(state[1].detach().abs().max()) > 0.0


def test_every_worker_starts_from_a_zero_carry_on_the_asked_device() -> None:
    model = _model()
    assert torch.equal(model.initial_state(ENVS), torch.zeros(2, ENVS, WIDTH))
    assert model.initial_state(ENVS, device="meta").device.type == "meta"


@torch.no_grad()
def test_the_hidden_and_the_cell_state_each_move_the_values() -> None:
    """The carry is hidden then cell, and the cell reads each in its place."""
    model = _model()
    torch.manual_seed(7)
    observations = torch.randn(ENVS, OBSERVATION)
    state = torch.randn(2, ENVS, WIDTH)
    base, _ = model.forward_fused(observations, state, None, features=_previous(ENVS))
    for layer in range(2):
        moved = state.clone()
        moved[layer] += 1.0
        values, _ = model.forward_fused(
            observations,
            moved,
            None,
            features=_previous(ENVS),
        )
        assert not torch.allclose(values, base), layer


@torch.no_grad()
def test_a_given_carry_is_advanced_in_place() -> None:
    """The actor's step graph passes its carry as both ``state`` and ``carry``."""
    model = _model()
    torch.manual_seed(1)
    observations = torch.randn(ENVS, OBSERVATION)
    start = torch.randn(2, ENVS, WIDTH)
    expected, advanced = model.forward_fused(
        observations,
        start,
        None,
        features=_previous(ENVS),
    )
    carry = start.clone()
    decoded, returned = model.forward_fused(
        observations,
        carry,
        None,
        carry=carry,
        features=_previous(ENVS),
    )
    assert returned is carry
    assert torch.equal(carry, advanced)
    assert torch.equal(decoded, expected)


@torch.no_grad()
def test_the_previous_action_changes_the_values() -> None:
    """A Q-learner needs it and a policy-gradient method does not.

    The value of a state depends on what the agent just tried, and
    epsilon-greedy exploration makes that unpredictable from the observation.
    """
    model = _model()
    torch.manual_seed(2)
    observations = torch.randn(ENVS, OBSERVATION)
    first, _ = model.forward_fused(
        observations,
        model.initial_state(ENVS),
        None,
        features=_previous(ENVS),
    )
    second, _ = model.forward_fused(
        observations,
        model.initial_state(ENVS),
        None,
        features=_previous(ENVS, action=3),
    )
    assert not torch.allclose(first, second)


@torch.no_grad()
def test_an_episode_start_clears_both_state_tensors() -> None:
    model = _model()
    torch.manual_seed(3)
    observations = torch.randn(ENVS, OBSERVATION)
    state = torch.randn(2, ENVS, WIDTH)
    _, reset = model.forward_fused(
        observations,
        state,
        torch.tensor([1.0, 0.0, 0.0]),
        features=_previous(ENVS),
    )
    _, fresh = model.forward_fused(
        observations,
        model.initial_state(ENVS),
        None,
        features=_previous(ENVS),
    )
    _, kept = model.forward_fused(observations, state, None, features=_previous(ENVS))
    assert torch.equal(reset[:, 0], fresh[:, 0])
    assert torch.equal(reset[:, 1:], kept[:, 1:])


def test_the_sequence_path_equals_the_stepped_path() -> None:
    # Exactly equal: a window performs each step's operations in the same order.
    model = _model().double()
    model.eval()
    torch.manual_seed(4)
    observations = torch.randn(ENVS, TIME, OBSERVATION, dtype=torch.float64)
    previous = torch.randint(0, ACTIONS, (ENVS, TIME, 1)).double()
    starts = torch.zeros(ENVS, TIME, dtype=torch.float64)
    starts[1, 2] = 1.0
    state = model.initial_state(ENVS).double()
    stepped: list[Tensor] = []
    for step in range(TIME):
        decoded, state = model.forward_fused(
            observations[:, step],
            state,
            starts[:, step],
            features=previous[:, step],
        )
        stepped.append(decoded)

    sequence, final, auxiliary = model.forward_sequence(
        observations,
        model.initial_state(ENVS).double(),
        starts,
        features=previous,
    )
    assert torch.equal(torch.stack(stepped, dim=1), sequence)
    assert torch.equal(state, final)
    assert float(auxiliary) == 0.0


@torch.no_grad()
def test_training_renormalizes_by_the_whole_window_and_inference_by_the_running_one() -> (
    None
):
    """Batch renormalization on the raw observation absorbs the policy's shift.

    As purejaxql's network does, a training window is one batch: the statistics
    over every environment and step move the running ones once, by the
    momentum. The actor and the evaluation read them and change nothing.
    """
    model = _model()
    torch.manual_seed(5)
    observations = torch.randn(ENVS, TIME, OBSERVATION) * 5.0 + 2.0
    arguments = (
        observations,
        model.initial_state(ENVS),
        torch.zeros(ENVS, TIME),
    )
    model.eval()
    model.forward_sequence(*arguments, features=_previous(ENVS, TIME))
    assert int(model.normalize.steps) == 0
    model.train()
    model.forward_sequence(*arguments, features=_previous(ENVS, TIME))
    rows = observations.reshape(ENVS * TIME, OBSERVATION)
    mean = rows.mean(dim=0)
    variance = ((rows - mean) ** 2).mean(dim=0)
    momentum = model.normalize.config.momentum
    assert int(model.normalize.steps) == 1
    torch.testing.assert_close(model.normalize.running_mean, (1 - momentum) * mean)
    torch.testing.assert_close(
        model.normalize.running_var,
        momentum + (1 - momentum) * variance,
    )


def test_gradients_reach_every_parameter() -> None:
    model = _model()
    decoded, _, _ = model.forward_sequence(
        torch.randn(ENVS, TIME, OBSERVATION),
        model.initial_state(ENVS),
        torch.zeros(ENVS, TIME),
        features=_previous(ENVS, TIME),
    )
    decoded.sum().backward()
    assert [name for name, p in model.named_parameters() if p.grad is None] == []


@pytest.mark.parametrize(
    "field",
    ["observation_size", "num_actions", "channels_hidden"],
)
def test_a_degenerate_dimension_is_refused(field: str) -> None:
    config = RecurrentQNetwork.Config()
    setattr(config, field, 0)
    with pytest.raises(
        ValueError,
        match=r"^RecurrentQNetwork dimensions must be positive$",
    ):
        config.make()


def test_the_least_dimensions_build() -> None:
    # One feature, one action and one unit: the least the refusal above admits.
    config = RecurrentQNetwork.Config()
    config.observation_size = config.num_actions = config.channels_hidden = 1
    assert config.make().head.out_features == 1


def test_a_missing_previous_action_is_refused() -> None:
    model = _model()
    with pytest.raises(
        ValueError,
        match=r"^RecurrentQNetwork reads the previous action as its feature$",
    ):
        model.forward_fused(
            torch.zeros(ENVS, OBSERVATION),
            model.initial_state(ENVS),
            None,
        )


def _stepped(model: nn.Module, inputs: tuple[Tensor, ...]) -> Tensor:
    """Take one recurrent step from a carried state; reduce the fused rows."""
    observations, hidden, cell = inputs
    assert isinstance(model, RecurrentQNetwork)
    decoded, _ = model.forward_fused(
        observations,
        torch.stack((hidden, cell)),
        None,
        features=_previous(observations.shape[0]),
    )
    return decoded.sum()


def test_the_cost_matches_torch_and_prices_the_lstm_step() -> None:
    """One token is one recurrent step of one worker.

    Both carried tensors have gradients, as at every step after a window's
    first, so torch counts the state-side gate matmul's full adjoint.
    """
    config = RecurrentQNetwork.Config()
    config.observation_size = OBSERVATION
    config.num_actions = ACTIONS
    config.channels_hidden = WIDTH
    analytical = assert_cost_matches_torch(
        config,
        build_input=lambda: (
            torch.randn(ENVS, OBSERVATION),
            torch.randn(ENVS, WIDTH, requires_grad=True),
            torch.randn(ENVS, WIDTH, requires_grad=True),
        ),
        seq_len=1,
        batch_size=ENVS,
        dtype=None,
        run=_stepped,
    )
    gates = 4 * WIDTH * (WIDTH + ACTIONS) + 4 * WIDTH * WIDTH
    weights = OBSERVATION * WIDTH + gates + WIDTH * ACTIONS
    biases = WIDTH + 2 * 4 * WIDTH + ACTIONS
    renorm = BatchRenorm.Config()
    renorm.channels_in = OBSERVATION
    norms = cost(renorm, seq_len=1, batch_size=ENVS, dtype=None) + cost(
        LayerNorm.Config(WIDTH, elementwise_affine=True),
        seq_len=1,
        batch_size=ENVS,
        dtype=None,
    )
    assert analytical.params == weights + biases + norms.params
    assert analytical["flops", "primal", "matmul"].sum() == ENVS * 2 * weights
    # The one-hot previous action is written, not computed.
    assert analytical["bytes", "primal", "selection"].sum() == 4 * ENVS * ACTIONS
    # The greedy value: a max over the actions, a comparison fewer than them.
    assert analytical["flops", "primal", "reduction"].sum() - norms[
        "flops",
        "primal",
        "reduction",
    ].sum() == ENVS * (ACTIONS - 1)
    assert analytical.bytes_state == 4 * ENVS * 2 * WIDTH


def test_cost_scales_its_bytes_with_the_dtype_and_not_its_flops() -> None:
    config = RecurrentQNetwork.Config()
    config.observation_size = 3
    config.channels_hidden = 2
    config.num_actions = 4
    narrow = config.cost(seq_len=1, batch_size=5, dtype=torch.bfloat16)
    wide = config.cost(seq_len=1, batch_size=5, dtype=None)
    assert narrow.bytes_state == 2 * 5 * 2 * 2
    assert wide.bytes_state == 4 * 5 * 2 * 2
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
    ("field", "value", "message"),
    [
        ("total_updates", 0, "total_updates must be positive"),
        ("decay_fraction", 0.0, "decay_fraction must be in (0, 1]"),
        ("decay_fraction", 1.5, "decay_fraction must be in (0, 1]"),
    ],
)
def test_an_invalid_schedule_is_refused(field: str, value: float, message: str) -> None:
    total_updates, decay_fraction = 100, 0.1
    if field == "total_updates":
        total_updates = int(value)
    else:
        decay_fraction = value
    with pytest.raises(ValueError, check=lambda error: str(error) == message):
        epsilon_at(0, total_updates=total_updates, decay_fraction=decay_fraction)


def test_schedule_accepts_one_update_and_full_run_decay() -> None:
    assert epsilon_at(1, total_updates=1) == pytest.approx(0.005)
    assert epsilon_at(100, total_updates=100, decay_fraction=1.0) == pytest.approx(
        0.005,
    )


def _explorer(
    *,
    start: float = 1.0,
    finish: float = 0.005,
    decay_fraction: float = 0.1,
    steps_per_update: int = 2,
    total_updates: int = 30,
) -> EpsilonGreedy:
    config = EpsilonGreedy.Config()
    config.start = start
    config.finish = finish
    config.decay_fraction = decay_fraction
    config.steps_per_update = steps_per_update
    config.total_updates = total_updates
    config.sampler = TorchPhiloxSampler.Config()
    return config.make()


def _decoded(agents: int, *, seed: int = 0) -> Tensor:
    """Return random action values and their maximum, as the network fuses them."""
    values = torch.randn(agents, ACTIONS, generator=torch.Generator().manual_seed(seed))
    return torch.cat((values, values.amax(dim=-1, keepdim=True)), dim=-1)


@pytest.mark.parametrize(
    ("start", "finish", "decay_fraction", "total_updates"),
    [(1.0, 0.005, 0.1, 30), (0.8, 0.1, 0.5, 6), (0.9, 0.2, 0.1, 5)],
    ids=["reference", "half-the-run", "within-one-update"],
)
def test_each_steps_rate_is_the_schedules_at_the_draw_counts_update(
    start: float,
    finish: float,
    decay_fraction: float,
    total_updates: int,
) -> None:
    """Two steps an update: each pair of draws reads one update's rate, as the host has it."""
    explorer = _explorer(
        start=start,
        finish=finish,
        decay_fraction=decay_fraction,
        total_updates=total_updates,
    )
    expected = [
        epsilon_at(
            update,
            total_updates=total_updates,
            start=start,
            finish=finish,
            decay_fraction=decay_fraction,
        )
        for update in range(8)
    ]
    assert [explorer.epsilon(update) for update in range(8)] == expected
    rates = explorer._epsilons(torch.arange(0, 2 * 8, dtype=torch.int64))
    assert rates.tolist() == np.repeat(np.array(expected, dtype=np.float32), 2).tolist()


def test_full_exploration_draws_every_action_alike() -> None:
    # At epsilon 1 every action is random, which is what makes the first
    # updates informative rather than a self-fulfilling greedy loop.
    explorer = _explorer(start=1.0, finish=1.0)
    agents = 64
    sampled = explorer(
        _decoded(agents),
        torch.ones(agents, ACTIONS, dtype=torch.uint8),
        explorer.draws(agents, device="cpu"),
        buffer=0,
    )
    assert len(set(sampled.actions.tolist())) == ACTIONS
    torch.testing.assert_close(
        sampled.logprobs,
        torch.full((agents,), -math.log(ACTIONS)),
    )


def test_no_exploration_takes_the_greedy_action() -> None:
    explorer = _explorer(start=0.0, finish=0.0)
    decoded = _decoded(16, seed=1)
    draws = explorer.draws(16, device="cpu")
    sampled = explorer(
        decoded,
        torch.ones(16, ACTIONS, dtype=torch.uint8),
        draws,
        buffer=0,
    )
    assert torch.equal(sampled.actions, decoded[:, :-1].argmax(dim=-1).float())
    assert torch.equal(sampled.values, decoded[:, -1])
    assert sampled.logprobs.abs().max() < 1e-6
    assert draws.tolist() == [1] * 16


def test_exploration_draws_only_legal_actions() -> None:
    explorer = _explorer(start=1.0, finish=1.0)
    agents = 64
    mask = torch.ones(agents, ACTIONS, dtype=torch.uint8)
    mask[:, 1:3] = 0
    decoded = _decoded(agents, seed=2)
    # The greedy action is illegal everywhere, so the legal maximum stands in.
    decoded[:, 1] = 100.0
    sampled = explorer(decoded, mask, explorer.draws(agents, device="cpu"), buffer=0)
    assert set(sampled.actions.tolist()) == {0.0, 3.0, 4.0}


def test_the_greedy_action_is_the_best_legal_one() -> None:
    explorer = _explorer(start=0.0, finish=0.0)
    mask = torch.ones(4, ACTIONS, dtype=torch.uint8)
    mask[:, 1] = 0
    decoded = _decoded(4, seed=4)
    decoded[:, 1] = 100.0
    sampled = explorer(decoded, mask, explorer.draws(4, device="cpu"), buffer=0)
    legal = torch.where(mask != 0, decoded[:, :-1], -math.inf)
    assert torch.equal(sampled.actions, legal.argmax(dim=-1).float())


def test_exploration_spreads_epsilon_over_the_legal_actions() -> None:
    """Each draw's log-probability is the epsilon-greedy distribution's, in the asked dtype.

    ``epsilon / legal`` on each legal action, and the rest on the greedy one.
    """
    explorer = _explorer(start=0.5, finish=0.5)
    agents = 32
    mask = torch.ones(agents, ACTIONS, dtype=torch.uint8)
    mask[:, 0] = 0
    decoded = _decoded(agents, seed=5)
    sampled = explorer(
        decoded,
        mask,
        explorer.draws(agents, device="cpu"),
        buffer=0,
        dtype=torch.float64,
    )
    greedy = torch.where(mask != 0, decoded[:, :-1], -math.inf).argmax(dim=-1)
    taken = sampled.actions.long()
    assert bool((taken == greedy).any())
    assert bool((taken != greedy).any())
    expected = torch.where(
        taken == greedy,
        math.log(0.5 + 0.5 / (ACTIONS - 1)),
        math.log(0.5 / (ACTIONS - 1)),
    ).double()
    assert sampled.logprobs.dtype == sampled.values.dtype == torch.float64
    # The sampler takes the log and the logsumexp in fp32.
    torch.testing.assert_close(sampled.logprobs, expected, rtol=0.0, atol=1e-5)
    assert torch.equal(sampled.values, decoded[:, -1].double())


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("start", 1.5, r"^start must be in \[0, 1\], not 1\.5$"),
        ("finish", -0.1, r"^finish must be in \[0, 1\], not -0\.1$"),
        (
            "steps_per_update",
            0,
            r"^steps_per_update and total_updates must be positive$",
        ),
        ("total_updates", 0, r"^steps_per_update and total_updates must be positive$"),
        ("decay_fraction", 0.0, r"^decay_fraction must be in \(0, 1\], not 0\.0$"),
        ("decay_fraction", 1.5, r"^decay_fraction must be in \(0, 1\], not 1\.5$"),
        ("decay_fraction", math.nan, r"^decay_fraction must be in \(0, 1\], not nan$"),
    ],
)
def test_an_invalid_exploration_is_refused(
    field: str,
    value: float,
    message: str,
) -> None:
    config = EpsilonGreedy.Config()
    config.steps_per_update = 2
    config.total_updates = 3
    setattr(config, field, value)
    with pytest.raises(ValueError, match=message):
        config.make()


def test_the_least_exploration_schedule_is_accepted() -> None:
    config = EpsilonGreedy.Config()
    config.steps_per_update = config.total_updates = 1
    config.decay_fraction = 1.0
    config.sampler = TorchPhiloxSampler.Config()
    assert config.make().epsilon(0) == 1.0


def test_the_streams_live_where_the_rollout_asks() -> None:
    for sampler in (_explorer(), GreedySampler.Config().make()):
        draws = sampler.draws(3, device="meta")
        assert (draws.device.type, draws.dtype, draws.shape) == (
            "meta",
            torch.int64,
            (3,),
        )


def test_the_greedy_sampler_takes_the_best_legal_action_and_draws_nothing() -> None:
    sampler = GreedySampler.Config().make()
    decoded = _decoded(4, seed=3)
    mask = torch.ones(4, ACTIONS, dtype=torch.uint8)
    best = decoded[:, :-1].argmax(dim=-1)
    mask[0, best[0]] = 0
    draws = sampler.draws(4, device="cpu")
    sampled = sampler(decoded, mask, draws, buffer=1, dtype=torch.float64)
    masked = torch.where(mask != 0, decoded[:, :-1], -math.inf)
    assert torch.equal(sampled.actions, masked.argmax(dim=-1).float())
    assert sampled.actions[0] != best[0]
    assert sampled.logprobs.dtype == sampled.values.dtype == torch.float64
    assert torch.equal(sampled.values, decoded[:, -1].double())
    assert torch.equal(sampled.logprobs, torch.zeros(4, dtype=torch.float64))
    assert draws.tolist() == [0] * 4


def test_the_greedy_values_are_a_copy_in_the_decoders_dtype() -> None:
    """The step graph reuses the decoder's rows, so the stored values must not alias them."""
    sampler = GreedySampler.Config().make()
    decoded = _decoded(4, seed=6).double()
    expected = decoded[:, -1].clone()
    sampled = sampler(
        decoded,
        torch.ones(4, ACTIONS, dtype=torch.uint8),
        sampler.draws(4, device="cpu"),
        buffer=0,
    )
    decoded.zero_()
    assert sampled.logprobs.dtype == sampled.values.dtype == torch.float64
    assert torch.equal(sampled.values, expected)


def test_the_greedy_sampler_writes_where_the_decoder_lives() -> None:
    sampler = GreedySampler.Config().make()
    sampled = sampler(
        torch.zeros(4, ACTIONS + 1, device="meta"),
        torch.ones(4, ACTIONS, dtype=torch.uint8, device="meta"),
        sampler.draws(4, device="meta"),
        buffer=0,
    )
    assert {sampled.logprobs.device.type, sampled.values.device.type} == {"meta"}


def test_the_previous_action_feature_is_the_uploaded_action_as_a_column() -> None:
    source = PreviousAction.Config().make()
    assert (source.width, source.joint, source.context_decisions) == (1, False, 0)
    # It needs no room, so a rollout steps its whole horizon between hooks.
    assert source.hook_interval == sys.maxsize
    assert source.history_archive(4, device=torch.device("cpu")) is None
    step = source.make_engine(rows=3, device=torch.device("cpu"))
    actions = torch.tensor([4.0, 0.0, 2.0])
    feature = step(torch.zeros(3, OBSERVATION), torch.tensor([1.0, 0.0, 0.0]), actions)
    # Carried over the first row's episode start, as the reference does.
    assert feature.tolist() == [[4.0], [0.0], [2.0]]


def test_the_previous_action_needs_no_upkeep() -> None:
    step = PreviousActionStep()
    assert step.plan == 0
    assert step.capture_state() == []
    assert step.ensure_room(5) == {}
    step.reset()
    step.begin_window(torch.ones(3, dtype=torch.bool))
    assert step.restore_history(None, torch.full((3,), -1)) is None
    step.release()


def test_the_previous_action_refuses_what_only_a_joint_feature_keeps() -> None:
    step = PreviousActionStep()
    with pytest.raises(ValueError, match=r"^the previous action keeps no context$"):
        step.context()
    with pytest.raises(ValueError, match=r"^the previous action keeps no decisions$"):
        step.last_decisions(2)
    with pytest.raises(
        ValueError,
        match=r"^the previous action keeps no history to rebuild$",
    ):
        step.rebuild()
    with pytest.raises(
        ValueError,
        match=r"^the previous action keeps no history to save$",
    ):
        step.save_history(
            {},
            torch.zeros(2, dtype=torch.int64),
            slice(0, 2),
            previous_action=torch.zeros(2),
            fresh=torch.zeros(2),
        )


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
