"""Tests for the gated Transformer-XL actor-critic at tiny sizes on the CPU."""

from __future__ import annotations

from pathlib import Path
from typing import Final

from torch import Tensor, nn

import pytest
import torch

from priml.baselines.craftax.policies.gtrxl import GTrXLPolicy
from priml.testing.bfb import assert_bfb_against_golden
from priml.testing.cost import assert_cost_matches_torch


_CWD: Final = Path(__file__).resolve().parent

OBSERVATION_SIZE: Final = 9
NUM_ACTIONS: Final = 4
WIDTH: Final = 6
MEMORY: Final = 7
QKV: Final = 2 * 4
"""Two heads of 4."""
CARRY: Final = 2 * WIDTH + 1
"""Two layers' inputs, then the written flag."""


def test_the_config_propagates_its_widths() -> None:
    config = _config().copy_tree().finalize()
    assert (config.proj_in.channels_in, config.proj_in.channels_out) == (9, 6)
    assert config.block.channels_hidden == WIDTH
    assert (config.decoder.observation_size, config.decoder.num_actions) == (6, 4)
    assert config.decoder.dtype == config.dtype


def test_the_memory_starts_empty_and_fills_a_row_per_step_then_slides() -> None:
    model = _config().make()
    state = model.initial_state(3)
    assert state.shape == (MEMORY, 3, CARRY)
    assert not state.any()
    written: list[int] = []
    rows: list[Tensor] = []
    for _ in range(MEMORY + 2):
        decoded, state = model.forward_fused(torch.randn(3, 9), state, None)
        assert decoded.shape == (3, NUM_ACTIONS + 1)
        written.append(int(state[:, 0, -1].sum()))
        rows.append(state[-1].clone())
    assert written == [1, 2, 3, 4, 5, 6, 7, 7, 7]
    # The newest row is last; each step pushes the row before it one slot back.
    assert torch.equal(state[-2], rows[-2])
    assert torch.equal(state[0], rows[-MEMORY])


def test_an_episode_start_empties_that_environment_only() -> None:
    # Memory that crossed an episode boundary would let a policy condition on a
    # world it is no longer in.
    model = _config().make()
    state = model.initial_state(2)
    for _ in range(3):
        _, state = model.forward_fused(torch.randn(2, 9), state, None)
    _, state = model.forward_fused(torch.randn(2, 9), state, torch.tensor([1.0, 0.0]))
    assert state[..., -1].sum(dim=0).tolist() == [1.0, 4.0]
    assert not state[:-1, 0].any()


def test_a_given_carry_is_advanced_in_place() -> None:
    """The rollout hands its carry as ``carry``, which its captured graph addresses."""
    model = _config().make()
    state = model.initial_state(2)
    _, state = model.forward_fused(torch.randn(2, 9), state, None)
    observations = torch.randn(2, 9)
    expected = model.forward_fused(observations, state, None)
    decoded, carry = model.forward_fused(observations, state, None, carry=state)
    assert carry is state
    assert torch.equal(decoded, expected[0])
    assert torch.equal(carry, expected[1])


@pytest.mark.parametrize("warmup", [0, 3], ids=["cold", "warm"])
def test_a_window_equals_its_steps_taken_one_at_a_time(warmup: int) -> None:
    """The whole reason both exist: one trains, one acts, and they must agree.

    Exact up to rounding while the memory holds every step either path reads
    (warmup plus the window within ``memory_length``): one attention per step
    and one over the window reduce in different orders, so the paths agree in
    float64 to 1e-12, where float32 would blur a real disagreement into a
    plausible tolerance. A start mid-window checks the episode mask.
    """
    model = _config(dtype=torch.float64).make()
    torch.manual_seed(0)
    state = model.initial_state(3)
    for _ in range(warmup):
        _, state = model.forward_fused(torch.randn(3, 9), state, None)
    observations = torch.randn(3, 4, 9, dtype=torch.float64)
    starts = torch.zeros(3, 4)
    starts[1, 2] = 1.0
    stepped, stepped_state = _steps(model, observations, state, starts)
    decoded, final, loss = model.forward_sequence(observations, state, starts)
    torch.testing.assert_close(decoded, stepped, rtol=0, atol=1e-12)
    torch.testing.assert_close(final, stepped_state, rtol=0, atol=1e-12)
    assert float(loss) == 0.0


def test_the_memory_changes_the_prediction() -> None:
    # If it did not, every mechanism in the module would be dead weight.
    model = _config().make()
    torch.manual_seed(2)
    observations = torch.randn(3, 9)
    state = model.initial_state(3)
    cold, _ = model.forward_fused(observations, state, None)
    for _ in range(4):
        _, state = model.forward_fused(torch.randn(3, 9), state, None)
    warm, _ = model.forward_fused(observations, state, None)
    assert not torch.allclose(cold, warm)


def test_the_gates_start_closed() -> None:
    """An untrained layer is the identity, which is what makes it trainable.

    With the gate open at initialization, the first high-variance policy
    gradients pass through a randomly initialized transformer and destroy the
    representation before it carries anything.
    """
    config = _config()
    config.block.gating_bias = 20.0
    block = config.make().blocks[0]
    hidden = torch.randn(2, 3, WIDTH)
    keys = torch.randn(2, 5, WIDTH)
    # The 1 is the heads axis: ``GTrXLPolicy.forward`` builds the mask as
    # ``[batch, 1, time, keys]``, one mask every head broadcasts.
    mask = torch.ones(2, 1, 3, 5, dtype=torch.bool)
    positional = torch.randn(5, WIDTH)
    torch.testing.assert_close(
        block(keys, hidden, positional, mask),
        hidden,
        rtol=0,
        atol=1e-6,
    )


def test_a_lag_is_encoded_the_same_wherever_the_window_sits() -> None:
    """The property that makes a sliding memory attendable at all.

    One layer and a memory of one step, so the remembered row is the encoder's
    output for the same context in both runs and only the absolute time
    differs. With more layers a deeper row also depends on what its step
    attended, a real difference in history rather than in position.
    """
    config = _config(dtype=torch.float64)
    config.num_layers = 1
    config.memory_length = 1
    model = config.make()
    torch.manual_seed(5)
    steps = torch.randn(9, 2, OBSERVATION_SIZE, dtype=torch.float64)
    # The same two context steps and probe, after no warmup and after six steps.
    cold = _probe(model, steps[6:])
    warm = _probe(model, steps)
    torch.testing.assert_close(cold, warm, rtol=0, atol=1e-12)


def test_a_step_cannot_see_the_future() -> None:
    # Changing a later observation must not move an earlier prediction, or the
    # value target would be fit against information the policy lacked.
    model = _config(dtype=torch.float64).make()
    torch.manual_seed(3)
    observations = torch.randn(2, 4, 9, dtype=torch.float64)
    state, starts = model.initial_state(2), torch.zeros(2, 4)
    before = model(observations, state, starts)[0]
    altered = observations.clone()
    altered[:, 3] = torch.randn(2, 9, dtype=torch.float64)
    after = model(altered, state, starts)[0]
    torch.testing.assert_close(before[:, :3], after[:, :3], rtol=0, atol=1e-12)
    assert not torch.allclose(before[:, 3], after[:, 3])


def test_a_window_does_not_read_across_an_episode_start() -> None:
    model = _config(dtype=torch.float64).make()
    torch.manual_seed(4)
    state = model.initial_state(2)
    _, state = model.forward_fused(torch.randn(2, 9), state, None)
    observations = torch.randn(2, 4, 9, dtype=torch.float64)
    starts = torch.zeros(2, 4)
    starts[0, 2] = 1.0
    before = model(observations, state, starts)[0]
    altered = observations.clone()
    altered[0, 0] = torch.randn(9, dtype=torch.float64)
    after = model(altered, state, starts)[0]
    # Worker 0's steps from its new episode are sealed off from the ones before.
    torch.testing.assert_close(before[0, 2:], after[0, 2:], rtol=0, atol=1e-12)
    assert not torch.allclose(before[0, 1], after[0, 1])
    # And from the memory: emptying it moves nothing after the start.
    emptied = model(observations, torch.zeros_like(state), starts)[0]
    torch.testing.assert_close(before[0, 2:], emptied[0, 2:], rtol=0, atol=1e-12)


def test_gradients_reach_every_parameter() -> None:
    model = _config().make()
    state = model.initial_state(2)
    _, state = model.forward_fused(torch.randn(2, 9), state, None)
    decoded, _, _ = model.forward_sequence(
        torch.randn(2, 3, 9),
        state,
        torch.zeros(2, 3),
    )
    decoded.sum().backward()
    missing = [name for name, p in model.named_parameters() if p.grad is None]
    assert missing == []


@torch.no_grad()
def test_the_policy_head_starts_near_uniform() -> None:
    # The 0.01 output gain: a policy that commits to an action before any
    # reward has been seen never recovers within the budget.
    model = _config().make()
    decoded, _ = model.forward_fused(torch.randn(64, 9), model.initial_state(64), None)
    assert float(decoded[:, :NUM_ACTIONS].std()) < 0.1
    assert bool(decoded.isfinite().all())


@pytest.mark.parametrize("name", ["num_layers", "memory_length", "channels_hidden"])
def test_a_policy_without_layers_memory_or_width_is_refused(name: str) -> None:
    config = _config()
    setattr(config, name, 0)
    with pytest.raises(ValueError, match="positive"):
        config.make()


@pytest.mark.parametrize("name", ["heads", "channels_head"])
def test_a_block_without_heads_is_refused(name: str) -> None:
    config = _config()
    setattr(config.block, name, 0)
    with pytest.raises(ValueError, match="positive"):
        config.make()


def test_a_feature_is_refused() -> None:
    model = _config().make()
    state = model.initial_state(2)
    with pytest.raises(ValueError, match="reads no feature"):
        model.forward_fused(torch.randn(2, 9), state, None, features=torch.ones(2, 3))
    with pytest.raises(ValueError, match="reads no feature"):
        model.forward_sequence(
            torch.randn(2, 3, 9),
            state,
            torch.zeros(2, 3),
            features=torch.ones(2, 3, 5),
        )


def test_the_cost_matches_torch_on_a_step() -> None:
    """One step of each environment attends over its whole memory.

    Every key row -- seven remembered and the step's own -- is normalized and
    projected once. The relative-position projection reads one constant table
    the three environments share: its forward and weight-gradient rows are
    counted, and no input gradient.
    """
    analytical = assert_cost_matches_torch(
        _config(),
        build_input=lambda: (
            torch.randn(3, OBSERVATION_SIZE, requires_grad=True),
            torch.randn(MEMORY, 3, CARRY, requires_grad=True),
        ),
        seq_len=1,
        batch_size=3,
        dtype=None,
        run=_stepped,
    )
    keys = MEMORY + 1
    # Per layer: the table projection's forward rows, which have no input gradient.
    table = 2 * WIDTH * QKV * keys
    assert analytical["flops", "adjoint", "matmul"].sum() == (
        2 * analytical["flops", "primal", "matmul"].sum() - 2 * table
    )
    # The relative scores are gathered into place: elements moved forward, one
    # scatter-add per element back. Per layer, environment and head: the score
    # bytes and the int64 index.
    assert analytical["flops", "primal", "selection"].sum() == 0
    selected = 2 * 3 * 2 * keys
    assert analytical["bytes", "primal", "selection", torch.float32] == 4 * selected
    assert analytical["bytes", "primal", "selection", torch.int64] == 8 * selected
    assert analytical["flops", "adjoint", "selection"].sum() == selected
    # One memory row per environment per step.
    assert analytical.bytes_state == 4 * 3 * CARRY


def test_the_cost_matches_torch_on_a_window() -> None:
    """A window's queries share its keys, so each key row is counted once.

    Four steps over a seven-row memory: each query sees eleven keys, and the
    eleven-row table is shared by the three windows of the invocation.
    """
    assert_cost_matches_torch(
        _config(),
        build_input=lambda: (
            torch.randn(3, 4, OBSERVATION_SIZE, requires_grad=True),
            torch.randn(MEMORY, 3, CARRY, requires_grad=True),
        ),
        seq_len=4,
        batch_size=3,
        dtype=None,
        run=_windowed,
    )


def test_the_cost_prices_every_stage_at_the_policys_dtype() -> None:
    narrow_config = _config()
    narrow_config.dtype = torch.bfloat16
    narrow = (
        narrow_config.copy_tree()
        .finalize()
        .cost(
            seq_len=4,
            batch_size=2,
            dtype=None,
        )
    )
    # The bus's dtype is unread: the policy runs at its own.
    wide = (
        _config()
        .copy_tree()
        .finalize()
        .cost(
            seq_len=4,
            batch_size=2,
            dtype=torch.bfloat16,
        )
    )
    assert (narrow.bytes_state, wide.bytes_state) == (2 * 2 * CARRY, 4 * 2 * CARRY)
    assert (
        wide["bytes", torch.float32].sum() == 2 * narrow["bytes", torch.bfloat16].sum()
    )
    assert wide["flops"].sum() == narrow["flops"].sum()
    selected = 2 * 4 * 2 * (MEMORY + 4)
    assert narrow["bytes", "primal", "selection", torch.bfloat16] == 2 * 2 * selected
    assert narrow["bytes", "primal", "selection", torch.int64] == 8 * 2 * selected
    assert narrow["bytes", "adjoint", "selection", torch.bfloat16] == (
        2 * 2 * 2 * selected
    )
    assert narrow["bytes", "primal", "reduction"].sum() > 0


def test_the_tiny_policy_matches_its_bfb_golden() -> None:
    """A window over a part-filled memory with starts, its gradients, then a step."""
    assert_bfb_against_golden(
        golden_dir=_CWD / "testdata",
        golden_name="gtrxl_policy_tiny",
        build_module=_config().make,
        build_input=_bfb_inputs,
        run=_score,
    )


def _config(*, dtype: torch.dtype = torch.float32) -> GTrXLPolicy.Config:
    """Return the policy at test size: 2 layers of 6, 2 heads of 4, 7 rows, towers of 10."""
    config = GTrXLPolicy.Config()
    config.observation_size = OBSERVATION_SIZE
    config.num_actions = NUM_ACTIONS
    config.channels_hidden = WIDTH
    config.memory_length = MEMORY
    config.block.heads = 2
    config.block.channels_head = QKV // 2
    config.decoder.channels_hidden = 10
    config.dtype = dtype
    return config


def _steps(
    model: GTrXLPolicy,
    observations: Tensor,
    state: Tensor,
    starts: Tensor,
) -> tuple[Tensor, Tensor]:
    """Drive the rollout's path one step at a time; return the rows and the memory."""
    decoded: list[Tensor] = []
    for index in range(observations.shape[1]):
        row, state = model.forward_fused(
            observations[:, index],
            state,
            starts[:, index],
        )
        decoded.append(row)
    return torch.stack(decoded, dim=1), state


def _probe(model: GTrXLPolicy, steps: Tensor) -> Tensor:
    """Step the policy through all but the last of ``steps``; score the last."""
    state = model.initial_state(steps.shape[1])
    for observations in steps[:-1]:
        _, state = model.forward_fused(observations, state, None)
    return model.forward_fused(steps[-1], state, None)[0]


def _stepped(module: nn.Module, inputs: tuple[Tensor, ...]) -> Tensor:
    """Take one step over a full memory; reduce the fused row."""
    assert isinstance(module, GTrXLPolicy)
    observations, state = inputs
    return module.forward_fused(observations, state, None)[0].sum()


def _windowed(module: nn.Module, inputs: tuple[Tensor, ...]) -> Tensor:
    """Score a window over a full memory; reduce the fused rows."""
    assert isinstance(module, GTrXLPolicy)
    observations, state = inputs
    starts = torch.zeros(observations.shape[:2])
    return module.forward_sequence(observations, state, starts)[0].sum()


def _bfb_inputs() -> dict[str, Tensor]:
    """Draw 3 windows of 4 steps, a memory written 0, 3 and 7 rows deep, two starts."""
    state = torch.randn(MEMORY, 3, CARRY)
    written = torch.arange(MEMORY)[:, None] >= torch.tensor([MEMORY, MEMORY - 3, 0])
    state = torch.where(written[..., None], state, 0.0)
    state[..., -1] = written.float()
    starts = torch.zeros(3, 4)
    starts[1, 2] = starts[2, 0] = 1.0
    return {"observations": torch.randn(3, 4, 9), "state": state, "starts": starts}


def _score(module: nn.Module, inputs: dict[str, Tensor]) -> Tensor:
    """Return the window, its final memory and a step, then every gradient."""
    assert isinstance(module, GTrXLPolicy)
    observations, state, starts = (
        inputs["observations"],
        inputs["state"],
        inputs["starts"],
    )
    decoded, final, _ = module.forward_sequence(observations, state, starts)
    weights = torch.linspace(-1.0, 1.0, decoded.numel()).reshape(decoded.shape)
    (decoded * weights).sum().backward()
    with torch.no_grad():
        step, following = module.forward_fused(observations[:, 0], state, starts[:, 0])
    gradients: list[Tensor] = []
    for name, parameter in module.named_parameters():
        assert parameter.grad is not None, name
        gradients.append(parameter.grad.flatten())
    outputs = (decoded, final, step, following)
    return torch.cat([output.flatten() for output in outputs] + gradients)


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
