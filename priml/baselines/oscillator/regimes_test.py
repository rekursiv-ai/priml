"""Check paired regimes, observed information, and reference continuation."""

from dataclasses import replace

import math

import pytest
import torch

from priml.baselines.oscillator.regimes import (
    DynamicsDomain,
    ObservedChange,
    ObservedWindows,
    PositionObservation,
    RegimeData,
    RegimeDynamics,
)


def test_position_mask_preserves_motion_without_hidden_storage() -> None:
    full = RegimeData.Config(seq_len=11, pred_len=5)
    masked = full.copy_tree()
    masked.observation = PositionObservation.Config()
    before = torch.get_rng_state().clone()
    state = full.make().sample(count=3, seed=17)
    position = masked.make().sample(count=3, seed=17)
    assert torch.equal(before, torch.get_rng_state())
    assert torch.equal(position.states, state.states)
    assert torch.equal(position.windows.media, state.windows.media[..., :1])
    assert torch.equal(position.windows.label, state.windows.label[..., :1])
    assert position.windows.media.untyped_storage().nbytes() == 3 * 11 * 8
    assert position.windows.label.untyped_storage().nbytes() == 3 * 5 * 8
    saved = position.states.clone()
    position.windows.media.fill_(123)
    position.windows.label.fill_(-123)
    assert torch.equal(saved, position.states)


def test_episode_count_is_nested_and_independent_streams_are_distinct() -> None:
    data = RegimeData.Config(seq_len=11, pred_len=5).make()
    small = data.sample(count=3, seed=17)
    large = data.sample(count=7, seed=17)
    other = data.sample(count=3, seed=18)
    assert torch.equal(small.states, large.states[:3])
    assert torch.equal(small.after, large.after[:3])
    assert not torch.equal(small.states, other.states)
    assert not torch.equal(small.before, small.after)
    assert small.after.unique(dim=0).shape[0] == 3


@pytest.mark.parametrize("position_only", [False, True])
def test_observed_change_leaves_evidence_and_no_unannounced_future_switch(
    position_only: bool,
) -> None:
    cfg = RegimeData.Config(seq_len=19, pred_len=5)
    cfg.schedule = ObservedChange.Config(recent_steps=(7, 11))
    if position_only:
        cfg.observation = PositionObservation.Config()
    data = cfg.make()
    episode = data.sample(count=3, seed=19)
    recent = cfg.seq_len - episode.change_at
    assert ((recent >= 7) & (recent <= 11)).all()
    assert (episode.change_at > 0).all()
    for index in range(1, cfg.seq_len + cfg.pred_len):
        coefficients = torch.where(
            (index >= episode.change_at)[:, None],
            episode.after,
            episode.before,
        )
        assert torch.equal(
            episode.states[:, index],
            data.dynamics.advance(episode.states[:, index - 1], coefficients),
        )
    oracle = data.dynamics.forecast(
        episode.states[:, cfg.seq_len - 1],
        coefficients=episode.after,
        steps=cfg.pred_len,
    )
    assert torch.equal(oracle, episode.states[:, cfg.seq_len :])


def test_stationary_control_matches_future_coefficients_and_initial_states() -> None:
    cfg = RegimeData.Config(seq_len=19, pred_len=5)
    stationary = cfg.make().sample(count=3, seed=23)
    cfg.schedule = ObservedChange.Config(recent_steps=(7, 11))
    changed = cfg.make().sample(count=3, seed=23)
    assert torch.equal(stationary.after, changed.after)
    assert torch.equal(stationary.states[:, 0], changed.states[:, 0])
    assert not stationary.change_at.any()
    assert not torch.equal(stationary.states[:, 1:], changed.states[:, 1:])


def test_domain_changes_only_future_draw_mapping() -> None:
    data = RegimeData.Config(seq_len=11, pred_len=5).make()
    ordinary = data.sample(count=3, seed=29)
    domain = DynamicsDomain.Config(stiffness=(2.2, 3.2))
    shifted = data.sample(count=3, seed=29, domain=domain)
    assert torch.equal(ordinary.before, shifted.before)
    assert torch.equal(ordinary.states[:, 0], shifted.states[:, 0])
    assert torch.equal(ordinary.after[:, 1:], shifted.after[:, 1:])
    assert ((shifted.after[:, 0] >= 2.2) & (shifted.after[:, 0] <= 3.2)).all()


def test_integrator_matches_explicit_batched_update_and_refines() -> None:
    cfg = RegimeDynamics.Config(substeps=1)
    # Each physical state has exactly two coordinates and three coefficients.
    state = torch.tensor([[0.2, -0.1], [0.4, 0.3]], dtype=torch.float64)
    coefficients = torch.tensor([[1.1, 0.2, 0.4], [0.9, 0.1, 0.3]])
    position, velocity = state.unbind(-1)
    k, c, b = coefficients.unbind(-1)
    velocity = velocity + cfg.step_size * (
        -k * position - c * velocity - b * position**3
    )
    expected = torch.stack((position + cfg.step_size * velocity, velocity), -1)
    assert torch.equal(cfg.make().advance(state, coefficients), expected)
    forecasts: list[torch.Tensor] = []
    for substeps in (2, 8, 32):
        cfg.substeps = substeps
        forecasts.append(cfg.make().forecast(state, coefficients=coefficients, steps=7))
    assert (forecasts[1] - forecasts[2]).abs().max() < (
        forecasts[0] - forecasts[2]
    ).abs().max()
    with pytest.raises(ValueError, match="positive"):
        cfg.make().forecast(state, coefficients=coefficients, steps=0)


def test_training_scales_include_only_the_supplied_windows() -> None:
    media = torch.arange(3 * 7 * 2).reshape(3, 7, 2).double()
    label = torch.arange(3 * 5 * 2).reshape(3, 5, 2).double() + 2
    windows = ObservedWindows(media=media, label=label)
    mean, scale = windows.channel_stats()
    values = torch.cat((media, label), 1)
    assert torch.equal(mean, values.mean((0, 1), keepdim=True))
    assert torch.equal(scale, values.std((0, 1), correction=0, keepdim=True))
    zero = ObservedWindows(media=torch.zeros_like(media), label=torch.zeros_like(label))
    assert (zero.channel_stats()[1] == 1e-6).all()


@pytest.mark.parametrize(
    "which",
    ["shape", "empty", "channel", "batch", "integer", "nan", "inf"],
)
def test_observed_windows_reject_invalid_cells_and_geometry(which: str) -> None:
    media = torch.zeros(3, 7, 2)
    label = torch.zeros(3, 5, 2)
    if which == "shape":
        media = media[0]
    elif which == "empty":
        label = label[:, :0]
    elif which == "channel":
        label = label[..., :1]
    elif which == "batch":
        label = label[:2]
    elif which == "integer":
        media = media.long()
    else:
        label[0, 0, 0] = float(which)
    with pytest.raises(ValueError, match="finite paired"):
        ObservedWindows(media=media, label=label)


@pytest.mark.parametrize("field", ["stiffness", "damping", "cubic"])
@pytest.mark.parametrize(
    "bounds",
    [(-0.1, 1.0), (2.0, 1.0), (0.5, math.inf), (math.nan, 1.0)],
)
def test_domains_reject_invalid_ranges(field: str, bounds: tuple[float, float]) -> None:
    cfg = DynamicsDomain.Config()
    setattr(cfg, field, bounds)
    with pytest.raises(ValueError, match="ranges"):
        cfg.make()


def test_zero_stiffness_and_invalid_schedule_are_rejected() -> None:
    with pytest.raises(ValueError, match="positive"):
        DynamicsDomain.Config(stiffness=(0, 1)).make()
    for recent in ((4, 7), (8, 7)):
        with pytest.raises(ValueError, match="at least five"):
            ObservedChange.Config(recent_steps=recent).make()
    with pytest.raises(ValueError, match="earlier regime"):
        ObservedChange.Config(recent_steps=(7, 11)).make()(torch.zeros(3), seq_len=11)


@pytest.mark.parametrize(("count", "seed"), [(0, 1), (3, -1), (3, 2**32)])
def test_sampling_rejects_invalid_streams(count: int, seed: int) -> None:
    with pytest.raises(ValueError, match="count and random seed"):
        RegimeData.Config().make().sample(count=count, seed=seed)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("seq_len", 6),
        ("pred_len", 0),
        ("velocity", -1.0),
        ("velocity", math.inf),
        ("amplitude", (1.0, 0.5)),
        ("amplitude", (-1.0, 1.0)),
    ],
)
def test_generator_rejects_invalid_geometry_and_bounds(
    field: str,
    value: object,
) -> None:
    cfg = RegimeData.Config()
    setattr(cfg, field, value)
    with pytest.raises(ValueError, match="geometry"):
        cfg.make()


@pytest.mark.parametrize(("step_size", "substeps"), [(0.0, 2), (math.inf, 2), (0.1, 0)])
def test_integrator_rejects_invalid_time_geometry(
    step_size: float,
    substeps: int,
) -> None:
    with pytest.raises(ValueError, match="positive"):
        RegimeDynamics.Config(step_size=step_size, substeps=substeps).make()


def test_dataclass_replacement_still_validates_windows() -> None:
    windows = ObservedWindows(media=torch.zeros(3, 7, 2), label=torch.zeros(3, 5, 2))
    with pytest.raises(ValueError, match="finite paired"):
        replace(windows, label=torch.full_like(windows.label, math.nan))


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
