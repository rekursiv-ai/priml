"""Tests for the gated Transformer-XL actor-critic."""

from __future__ import annotations

from torch import Tensor

import pytest
import torch

from priml.baselines.craftax.gtrxl import ActorCriticGTrXL


def _model(**overrides: float) -> ActorCriticGTrXL:
    config = ActorCriticGTrXL.Config()
    config.observation_size = 12
    config.num_actions = 5
    config.embed_dim = 16
    config.num_heads = 2
    config.num_layers = 2
    config.qkv_dim = 16
    config.hidden_size = 8
    config.memory_length = 8
    for name, value in overrides.items():
        setattr(config, name, value)
    return config.make()


def _rollout(
    model: ActorCriticGTrXL,
    observation: Tensor,
    done: Tensor,
    memory: Tensor,
    valid_length: Tensor,
) -> tuple[Tensor, Tensor]:
    """Drive the recurrent path one step at a time."""
    logits: list[Tensor] = []
    values: list[Tensor] = []
    for index in range(observation.shape[0]):
        memory, valid_length, step_logits, step_value = model.step(
            memory,
            valid_length,
            observation[index],
            done[index],
        )
        logits.append(step_logits)
        values.append(step_value)
    return torch.stack(logits), torch.stack(values)


def test_the_feed_forward_surface_matches_the_actor_critic_shapes() -> None:
    logits, value = _model()(torch.zeros(3, 12))
    assert logits.shape == (3, 5)
    assert value.shape == (3,)


def test_a_step_returns_memory_and_predictions() -> None:
    model = _model()
    memory, valid_length = model.initial_state(3)
    memory, valid_length, logits, value = model.step(
        memory,
        valid_length,
        torch.zeros(3, 12),
        torch.zeros(3, dtype=torch.bool),
    )
    assert memory.shape == (3, 8, 2, 16)
    assert valid_length.tolist() == [1, 1, 1]
    assert logits.shape == (3, 5)
    assert value.shape == (3,)


def test_memory_fills_and_then_saturates() -> None:
    model = _model(memory_length=3)
    memory, valid_length = model.initial_state(2)
    lengths: list[int] = []
    for _ in range(5):
        memory, valid_length, _, _ = model.step(
            memory,
            valid_length,
            torch.zeros(2, 12),
            torch.zeros(2, dtype=torch.bool),
        )
        lengths.append(int(valid_length[0]))
    assert lengths == [1, 2, 3, 3, 3]


def test_a_terminal_transition_clears_that_worker_only() -> None:
    # Memory that crossed an episode boundary would let a policy condition on
    # a world it is no longer in.
    model = _model()
    memory, valid_length = model.initial_state(2)
    for _ in range(3):
        memory, valid_length, _, _ = model.step(
            memory,
            valid_length,
            torch.randn(2, 12),
            torch.zeros(2, dtype=torch.bool),
        )
    memory, valid_length, _, _ = model.step(
        memory,
        valid_length,
        torch.randn(2, 12),
        torch.tensor([True, False]),
    )
    assert valid_length.tolist() == [1, 4]


def test_the_sequence_path_equals_the_recurrent_path() -> None:
    """The whole reason both exist: one trains, one acts, they must agree.

    Computed in float64 because the two paths issue different reductions --
    one attention per step versus one over the window -- so float32 rounding
    would hide a real disagreement behind a plausible tolerance.
    """
    model = _model().double()
    torch.manual_seed(0)
    observation = torch.randn(4, 3, 12, dtype=torch.float64)
    done = torch.zeros(4, 3, dtype=torch.bool)
    done[2, 1] = True
    memory, valid_length = model.initial_state(3)

    logits, value = _rollout(model, observation, done, memory.double(), valid_length)
    sequence_logits, sequence_value = model.sequence(
        memory.double(),
        valid_length,
        observation,
        done,
    )

    assert torch.allclose(logits, sequence_logits, atol=1e-12)
    assert torch.allclose(value, sequence_value, atol=1e-12)


def test_the_two_paths_agree_on_a_warm_memory() -> None:
    # A gradient window in the middle of a rollout starts from a filled
    # cache, which is the case the mask arithmetic actually has to get right.
    model = _model().double()
    torch.manual_seed(1)
    warmup = torch.randn(4, 3, 12, dtype=torch.float64)
    memory, valid_length = model.initial_state(3)
    memory = memory.double()
    for index in range(warmup.shape[0]):
        memory, valid_length, _, _ = model.step(
            memory,
            valid_length,
            warmup[index],
            torch.zeros(3, dtype=torch.bool),
        )

    observation = torch.randn(4, 3, 12, dtype=torch.float64)
    done = torch.zeros(4, 3, dtype=torch.bool)
    logits, value = _rollout(model, observation, done, memory, valid_length)
    sequence_logits, sequence_value = model.sequence(
        memory,
        valid_length,
        observation,
        done,
    )
    assert torch.allclose(logits, sequence_logits, atol=1e-12)
    assert torch.allclose(value, sequence_value, atol=1e-12)


def test_memory_changes_the_prediction() -> None:
    # If it did not, every mechanism in this file would be dead weight.
    model = _model()
    torch.manual_seed(2)
    observation = torch.randn(3, 12)
    memory, valid_length = model.initial_state(3)
    cold, _ = model.step(
        memory,
        valid_length,
        observation,
        torch.zeros(3, dtype=torch.bool),
    )[2:]
    for _ in range(4):
        memory, valid_length, _, _ = model.step(
            memory,
            valid_length,
            torch.randn(3, 12),
            torch.zeros(3, dtype=torch.bool),
        )
    warm, _ = model.step(
        memory,
        valid_length,
        observation,
        torch.zeros(3, dtype=torch.bool),
    )[2:]
    assert not torch.allclose(cold, warm)


def test_the_gate_starts_closed() -> None:
    """An untrained layer is the identity, which is what makes it trainable.

    With the gate open at initialization, the first high-variance policy
    gradients pass through a randomly-initialized transformer and destroy the
    representation before it carries anything.
    """
    model = _model(gating_bias=20.0)
    hidden = torch.randn(2, 3, 16)
    layer = model.layers[0]
    assert isinstance(layer, torch.nn.Module)
    gated = layer.gate_attention(hidden, torch.randn(2, 3, 16))
    assert torch.allclose(gated, hidden, atol=1e-6)


def test_a_lag_is_encoded_the_same_wherever_the_window_sits() -> None:
    """The property that makes a sliding memory attendable at all.

    Two identical steps taken at different absolute times must produce the
    same prediction given the same recent history, which only holds because
    attention scores a key by its LAG rather than its position.
    """
    # One layer and a one-step memory, so the remembered row is exactly
    # ``encoder(context)`` in both cases and only the absolute time differs.
    # With more layers the deeper rows also depend on what was attended
    # before them, which is a real difference in history, not in position.
    model = _model(num_layers=1, memory_length=1).double()
    torch.manual_seed(5)
    context = torch.randn(1, 2, 12, dtype=torch.float64)
    probe = torch.randn(2, 12, dtype=torch.float64)
    quiet = torch.zeros(2, dtype=torch.bool)

    def predict(*, warmup: int) -> Tensor:
        memory, valid_length = model.initial_state(2)
        memory = memory.double()
        for _ in range(warmup):
            memory, valid_length, _, _ = model.step(
                memory,
                valid_length,
                torch.randn(2, 12, dtype=torch.float64),
                quiet,
            )
        for index in range(context.shape[0]):
            memory, valid_length, _, _ = model.step(
                memory,
                valid_length,
                context[index],
                quiet,
            )
        return model.step(memory, valid_length, probe, quiet)[2]

    # The two-step memory holds exactly ``context`` in both cases; only the
    # absolute step count differs.
    assert torch.allclose(predict(warmup=0), predict(warmup=6), atol=1e-12)


def test_a_masked_step_cannot_see_the_future() -> None:
    # Changing a later observation must not move an earlier prediction, or
    # the value target would be fit against information the policy lacked.
    model = _model().double()
    torch.manual_seed(3)
    observation = torch.randn(4, 2, 12, dtype=torch.float64)
    done = torch.zeros(4, 2, dtype=torch.bool)
    memory, valid_length = model.initial_state(2)
    before = model.sequence(memory.double(), valid_length, observation, done)[0]

    altered = observation.clone()
    altered[3] = torch.randn(2, 12, dtype=torch.float64)
    after = model.sequence(memory.double(), valid_length, altered, done)[0]

    assert torch.allclose(before[:3], after[:3], atol=1e-12)
    assert not torch.allclose(before[3], after[3])


def test_the_sequence_path_does_not_look_across_an_episode() -> None:
    model = _model().double()
    torch.manual_seed(4)
    observation = torch.randn(4, 2, 12, dtype=torch.float64)
    done = torch.zeros(4, 2, dtype=torch.bool)
    done[2, 0] = True
    memory, valid_length = model.initial_state(2)
    before = model.sequence(memory.double(), valid_length, observation, done)[0]

    altered = observation.clone()
    altered[0, 0] = torch.randn(12, dtype=torch.float64)
    after = model.sequence(memory.double(), valid_length, altered, done)[0]

    # Worker 0's post-terminal steps are sealed off from its own prior ones.
    assert torch.allclose(before[2:, 0], after[2:, 0], atol=1e-12)
    assert not torch.allclose(before[1, 0], after[1, 0])


def test_gradients_reach_every_parameter() -> None:
    model = _model()
    memory, valid_length = model.initial_state(2)
    logits, value = model.sequence(
        memory,
        valid_length,
        torch.randn(3, 2, 12),
        torch.zeros(3, 2, dtype=torch.bool),
    )
    (logits.sum() + value.sum()).backward()
    missing = [name for name, p in model.named_parameters() if p.grad is None]
    assert missing == []


@torch.no_grad()
def test_the_policy_head_starts_near_uniform() -> None:
    # The 0.01 output gain: an initial policy that commits to an action before
    # any reward has been seen never recovers within the budget.
    logits, value = _model()(torch.randn(64, 12))
    assert float(logits.std()) < 0.1
    assert bool(torch.isfinite(value).all())


@pytest.mark.parametrize(
    "field",
    ["embed_dim", "num_heads", "num_layers", "qkv_dim", "hidden_size", "memory_length"],
)
def test_a_degenerate_dimension_is_refused(field: str) -> None:
    with pytest.raises(ValueError, match="positive"):
        _model(**{field: 0})


def test_heads_that_do_not_divide_the_projection_are_refused() -> None:
    with pytest.raises(ValueError, match="divide"):
        _model(num_heads=3, qkv_dim=16)


if __name__ == "__main__":
    from priml.lib.testing import test_main

    test_main(__file__)
