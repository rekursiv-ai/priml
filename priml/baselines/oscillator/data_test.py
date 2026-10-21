"""Check the simulator equations and independent forecasting splits."""

from unittest import mock

import pytest
import torch

from priml.baselines.oscillator.data import (
    ForecastData,
    InitialStates,
    Oscillator,
    Windows,
)


def test_integrator_matches_one_explicit_update() -> None:
    cfg = Oscillator.Config()
    cfg.substeps = 1
    cfg.cubic = 0.4
    # The simulator's state is always a position/velocity pair.
    initial = torch.tensor([[0.2, -0.1], [0.4, 0.3], [-0.8, 0.2]])
    original = initial.clone()
    position, velocity = original.unbind(dim=-1)
    expected_velocity = velocity + cfg.step_size * (
        -cfg.stiffness * position - cfg.damping * velocity - cfg.cubic * position.pow(3)
    )
    expected_position = position + cfg.step_size * expected_velocity
    # One output frame isolates a single integration update.
    actual = cfg.make().forecast(initial, steps=1)
    assert torch.equal(
        actual[:, 0],
        torch.stack((expected_position, expected_velocity), dim=-1),
    )
    assert torch.equal(initial, original)


def test_rollouts_continue_from_last_observation() -> None:
    simulator = Oscillator.Config().make()
    initial = InitialStates.Config(count=4).make().sample()
    full = simulator.forecast(initial, steps=8)
    prefix = simulator.forecast(initial, steps=5)
    suffix = simulator.forecast(prefix[:, -1], steps=3)
    assert torch.equal(full, torch.cat((prefix, suffix), dim=1))


def test_substep_override_uses_the_same_kernel_without_mutating_config() -> None:
    cfg = Oscillator.Config()
    simulator = cfg.make()
    before = simulator.config.copy_tree()
    states = torch.tensor([[0.2, -0.1], [0.4, 0.3], [-0.8, 0.2]])
    ordinary = simulator.forecast(states, steps=5)
    assert torch.equal(ordinary, simulator.forecast(states, steps=5, substeps=None))
    assert torch.equal(ordinary, simulator.forecast(states, steps=5, substeps=4))
    refined = cfg.copy_tree()
    refined.substeps = 16
    expected = refined.make().forecast(states, steps=5)
    assert torch.equal(expected, simulator.forecast(states, steps=5, substeps=16))
    assert not torch.equal(ordinary, expected)
    history = simulator.forecast(states, steps=4)
    assert torch.equal(
        simulator.forecast_history(history, steps=5, substeps=16),
        refined.make().forecast(history[:, -1], steps=5),
    )
    assert simulator.config == before


@pytest.mark.parametrize("substeps", [0, -1])
def test_substep_override_requires_a_positive_count(substeps: int) -> None:
    simulator = Oscillator.Config().make()
    states = torch.zeros(3, 2)
    with pytest.raises(ValueError, match="substeps"):
        simulator.forecast(states, steps=5, substeps=substeps)
    with pytest.raises(ValueError, match="substeps"):
        simulator.forecast_history(states[:, None], steps=5, substeps=substeps)


def test_sampling_is_repeatable_and_does_not_consume_global_rng() -> None:
    cfg = InitialStates.Config()
    cfg.count = 4
    before = torch.get_rng_state().clone()
    first = cfg.make().sample()
    assert torch.equal(first, cfg.make().sample())
    assert torch.equal(before, torch.get_rng_state())
    cfg.seed += 1
    assert not torch.equal(first, cfg.make().sample())


def test_sampling_maps_uniform_draws_to_signed_bounded_states() -> None:
    # Row 0 sits on the sign threshold; row 1 draws the amplitude's upper edge.
    draws = torch.tensor([[0.25, 0.5, 0.0], [1.0, 0.25, 0.75]])
    cfg = InitialStates.Config(count=2, amplitude=(0.2, 1.0), velocity=0.4)
    with mock.patch.object(torch, "rand", return_value=draws) as rand:
        states = cfg.make().sample()
    assert rand.call_args.args == (2, 3)
    torch.testing.assert_close(states, torch.tensor([[0.4, -0.4], [-1.0, 0.2]]))


def test_forecasting_windows_share_no_trajectory_between_splits() -> None:
    cfg = ForecastData.Config()
    cfg.seq_len = 5
    cfg.pred_len = 3
    cfg.train.count = 4
    cfg.validation.count = 6
    data = cfg.make()
    train = data.windows(cfg.train, simulator=data.reference)
    validation = data.windows(cfg.validation, simulator=data.reference)
    assert train.media.shape == (4, 5, 2)
    assert train.label.shape == (4, 3, 2)
    assert (
        not (train.media[:, None, 0] == validation.media[None, :, 0]).all(dim=-1).any()
    )
    expected = data.reference.forecast(train.media[:, -1], steps=cfg.pred_len)
    assert torch.equal(train.label, expected)


def test_identical_simulators_agree_and_missing_force_changes_forecasts() -> None:
    cfg = ForecastData.Config()
    cfg.reference = cfg.cheap.copy_tree()
    data = cfg.make()
    initial = InitialStates.Config(count=4).make().sample()
    reference = data.reference.forecast(initial, steps=3)
    assert torch.equal(reference, data.cheap.forecast(initial, steps=3))
    cfg.reference.cubic = 0.4
    assert not torch.equal(reference, cfg.reference.make().forecast(initial, steps=3))


@pytest.mark.parametrize(
    "cfg",
    [
        Oscillator.Config(step_size=0.0),
        Oscillator.Config(damping=-0.1),
        Oscillator.Config(cubic=float("nan")),
        Oscillator.Config(substeps=0),
    ],
)
def test_bad_simulator_settings_fail(cfg: Oscillator.Config) -> None:
    with pytest.raises(ValueError, match=r"finite|positive|nonnegative"):
        cfg.make()


def test_duplicate_split_seeds_and_mismatched_time_grids_fail() -> None:
    cfg = ForecastData.Config()
    cfg.test.seed = cfg.train.seed
    with pytest.raises(ValueError, match="separate seed"):
        cfg.make()
    cfg = ForecastData.Config()
    cfg.cheap.step_size *= 2
    with pytest.raises(ValueError, match="observation grid"):
        cfg.make()


def test_bad_initial_conditions_and_empty_history_fail() -> None:
    with pytest.raises(ValueError, match="finite"):
        InitialStates.Config(velocity=float("nan")).make()
    with pytest.raises(ValueError, match="Invalid"):
        InitialStates.Config(count=0).make()
    with pytest.raises(ValueError, match="positive"):
        ForecastData.Config(seq_len=0).make()
    simulator = Oscillator.Config().make()
    with pytest.raises(ValueError, match="pairs"):
        simulator.forecast(torch.zeros(4, 3), steps=3)
    with pytest.raises(ValueError, match="floating-point"):
        simulator.forecast(torch.zeros(4, 2, dtype=torch.int64), steps=3)


def test_history_forecast_continues_from_the_last_observed_state() -> None:
    simulator = Oscillator.Config().make()
    history = InitialStates.Config(count=3).make().sample()[:, None, :].repeat(1, 4, 1)
    history[:, :-1] += 1.0
    forecast = simulator.forecast_history(history, steps=2)
    assert forecast.shape == (3, 2, 2)
    assert torch.equal(forecast, simulator.forecast(history[:, -1], steps=2))


def test_divergence_propagates_as_nonfinite_values() -> None:
    values = Oscillator.Config().make().forecast(torch.full((4, 2), 1e38), steps=3)
    assert values.shape == (4, 3, 2)
    assert not torch.isfinite(values).all()


def test_amplitude_bounds_must_be_ordered_and_nonnegative() -> None:
    with pytest.raises(ValueError, match="Invalid"):
        InitialStates.Config(amplitude=(-0.1, 1.0)).make()
    with pytest.raises(ValueError, match="Invalid"):
        InitialStates.Config(amplitude=(1.0, 1.0)).make()


@pytest.mark.parametrize("bad", ["empty", "channels", "count", "rank", "integer"])
def test_windows_must_pair_floating_state_sequences(bad: str) -> None:
    media, label = torch.zeros(3, 4, 2), torch.zeros(3, 5, 2)
    if bad == "empty":
        media, label = media[:0], label[:0]
    elif bad == "channels":
        label = label[..., :1]
    elif bad == "count":
        label = label[:2]
    elif bad == "rank":
        media = media[0]
    else:
        media = media.long()
    with pytest.raises(ValueError, match=r"windows|trajectory"):
        Windows(media=media, label=label)


def test_split_seeds_stay_within_the_generator_seed_range() -> None:
    # The CPU generator keeps 32 seed bits: these two seeds draw identical states.
    same = InitialStates.Config(seed=101 + 2**32, count=4)
    with pytest.raises(ValueError, match="seeds"):
        same.make()
    with pytest.raises(ValueError, match="seeds"):
        InitialStates.Config(seed=-1).make()
    assert InitialStates.Config(seed=2**32 - 1, count=4).make().sample().shape == (4, 2)


def test_channel_stats_use_histories_and_targets() -> None:
    # Two channels: the simulator's position/velocity pair (STATE_CHANNELS).
    media = torch.arange(3 * 4 * 2, dtype=torch.float32).reshape(3, 4, 2)
    label = -torch.arange(3 * 5 * 2, dtype=torch.float32).reshape(3, 5, 2)
    mean, scale = Windows(media=media, label=label).channel_stats()
    values = torch.cat((media, label), dim=1)
    assert torch.equal(mean, values.mean(dim=(0, 1), keepdim=True))
    assert torch.equal(scale, values.std(dim=(0, 1), correction=0, keepdim=True))
    constant = Windows(media=torch.ones(3, 4, 2), label=torch.ones(3, 5, 2))
    _, floored = constant.channel_stats()
    # channel_stats keeps singleton axes so the scale broadcasts over windows.
    assert torch.equal(floored, torch.full((1, 1, 2), 1e-6))


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
