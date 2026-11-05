"""Check the driven generator, its glitchy sensor, schedule, and shooting reference."""

import pytest
import torch

from priml.baselines.oscillator.forced import (
    ForcedData,
    ForcedDomain,
    ForcedDynamics,
    ForcedShooting,
    GlitchySensor,
)
from priml.baselines.oscillator.hidden import HiddenDynamics


def small(**changes: object) -> ForcedData.Config:
    cfg = ForcedData.Config(
        seq_len=96,
        pred_len=8,
        recent_steps=(16, 24),
        segment_steps=(16, 24),
    )
    for key, value in changes.items():
        setattr(cfg, key, value)
    return cfg


def test_pool_is_nested_and_streams_are_distinct() -> None:
    data = small().make()
    before = torch.get_rng_state().clone()
    pool = data.sample(count=3, seed=17)
    larger = data.sample(count=9, seed=17)
    assert torch.equal(before, torch.get_rng_state())
    assert torch.equal(pool.states, larger.states[:3])
    assert torch.equal(pool.windows.media, larger.windows.media[:3])
    assert not torch.equal(pool.states, data.sample(count=3, seed=18).states)


def test_sensor_glitches_and_quantizes_but_never_touches_the_state() -> None:
    cfg = small()
    noisy = cfg.make().sample(count=64, seed=5)
    cfg.observation = GlitchySensor.Config(noise=(0, 0), spike_rate=0, quantum=0)
    clean = cfg.make().sample(count=64, seed=5)
    assert torch.equal(noisy.states, clean.states)
    assert torch.equal(clean.windows.media[..., 0], clean.states[:, :96, 0])
    media = noisy.windows.media[..., 0]
    assert torch.allclose(media, (media / 0.01).round() * 0.01)
    spikes = (media - noisy.states[:, :96, 0]).abs() > 0.4
    assert 0.005 < float(spikes.double().mean()) < 0.04


def test_schedule_places_every_change_in_the_history() -> None:
    episode = small().make().sample(count=400, seed=11)
    changed = episode.change_at > 0
    assert 0.08 < float((~changed).double().mean()) < 0.25
    recent = 96 - episode.change_at[changed]
    assert int(recent.min()) >= 16
    assert int(recent.max()) <= 24
    # The partner and drive persist; only the observed mass changes.
    assert torch.equal(episode.before[:, 6:], episode.after[:, 6:])
    assert not torch.equal(episode.before[changed, :6], episode.after[changed, :6])
    assert torch.equal(episode.before[~changed], episode.after[~changed])


def test_reserved_domain_moves_only_the_final_regime() -> None:
    domain = ForcedDomain.Config(quadratic=(0.3, 0.8))
    episode = small().make().sample(count=50, seed=3, domain=domain)
    changed = episode.change_at > 0
    assert float(episode.after[:, 4].min()) >= 0.3
    assert float(episode.before[changed, 4].abs().max()) == 0


def test_oracle_continuation_reproduces_targets_exactly() -> None:
    data = small().make()
    episode = data.sample(count=5, seed=2)
    oracle = data.dynamics.forecast(
        episode.states[:, 95],
        coefficients=episode.after,
        steps=8,
    )
    assert torch.equal(oracle, episode.states[:, 96:])


def test_hidden_drive_changes_the_observed_motion() -> None:
    data = small().make()
    episode = data.sample(count=8, seed=8)
    undriven = episode.after.clone()
    undriven[:, 9] = 0
    path = data.dynamics.forecast(episode.states[:, 95], coefficients=undriven, steps=8)
    assert not torch.allclose(path[..., 0], episode.states[:, 96:, 0])


def test_shooting_recovers_clean_stationary_motion_and_rejects_other_physics() -> None:
    cfg = small(seq_len=48, pred_len=16, changes=(1.0,), recent_steps=(40, 40))
    cfg.observation = GlitchySensor.Config(noise=(0, 0), spike_rate=0, quantum=0)
    data = cfg.make()
    episode = data.sample(count=4, seed=21)
    method = ForcedShooting.Config(visible_steps=40, iterations=800).make()
    method.fit(episode.windows, validation=episode.windows, dynamics=data.dynamics)
    prediction = method.predict(episode.windows.media)
    assert prediction.shape == episode.windows.label.shape
    error = (prediction[..., 0] - episode.states[:, 48:, 0]).square().mean()
    persistence = episode.windows.media[:, -1:, 0] - episode.states[:, 48:, 0]
    assert float(error) < 0.05 * float(persistence.square().mean())
    with pytest.raises(TypeError):
        method.fit(
            episode.windows,
            validation=episode.windows,
            dynamics=HiddenDynamics(HiddenDynamics.Config()),
        )


@pytest.mark.parametrize(
    "change",
    [
        {"recent_steps": (4, 10)},
        {"segment_steps": (40, 60)},
        {"changes": (0.5, 0.4)},
    ],
)
def test_invalid_geometry_is_rejected(change: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="must be valid"):
        small(**change).make()


def test_invalid_components_and_unbounded_draws_are_rejected() -> None:
    with pytest.raises(ValueError, match="physical"):
        ForcedDomain.Config(cubic=(-1.0, 0.5)).make()
    with pytest.raises(ValueError, match="positive"):
        ForcedDynamics.Config(substeps=0).make()
    with pytest.raises(ValueError, match="nonnegative"):
        GlitchySensor.Config(spike_rate=2.0).make()
    data = small().make()
    with pytest.raises(ValueError, match="positive"):
        data.dynamics.forecast(
            torch.zeros(1, 5),
            coefficients=torch.ones(1, 11),
            steps=0,
        )
    with pytest.raises(ValueError, match="must be valid"):
        data.sample(count=0, seed=1)
    # An inverted spring with no cubic restoring escapes to infinity.
    runaway = ForcedDomain.Config(stiffness=(-200.0, -200.0), cubic=(0.0, 0.0))
    with pytest.raises(ValueError, match="unbounded"):
        data.sample(count=2, seed=1, domain=runaway)
