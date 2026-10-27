"""Check the hidden-partner generator, its latent information, and the reference."""

import pytest
import torch

from priml.baselines.oscillator.adaptation import RecentPhysics
from priml.baselines.oscillator.hidden import (
    HiddenData,
    HiddenDomain,
    LibraryShooting,
    NoisyPosition,
)


def test_pool_is_nested_and_streams_are_distinct() -> None:
    data = HiddenData.Config(seq_len=60, pred_len=8, recent_steps=(20, 40)).make()
    before = torch.get_rng_state().clone()
    small = data.sample(count=3, seed=17)
    large = data.sample(count=9, seed=17)
    other = data.sample(count=3, seed=18)
    assert torch.equal(before, torch.get_rng_state())
    assert torch.equal(small.states, large.states[:3])
    assert torch.equal(small.windows.media, large.windows.media[:3])
    assert not torch.equal(small.states, other.states)


def test_noise_is_observed_but_never_part_of_the_state() -> None:
    cfg = HiddenData.Config(seq_len=60, pred_len=8, recent_steps=(20, 40))
    noisy = cfg.make().sample(count=4, seed=5)
    cfg.observation = NoisyPosition.Config(noise=(0.0, 0.0))
    clean = cfg.make().sample(count=4, seed=5)
    assert torch.equal(noisy.states, clean.states)
    assert torch.equal(clean.windows.media[..., 0], clean.states[:, :60, 0])
    residual = noisy.windows.media[..., 0] - noisy.states[:, :60, 0]
    assert 0.005 < float(residual.std()) < 0.05
    assert noisy.windows.media.shape[-1] == 1
    saved = noisy.states.clone()
    noisy.windows.media.fill_(9)
    noisy.windows.label.fill_(9)
    assert torch.equal(saved, noisy.states)


def test_schedule_mixes_stationary_and_changed_without_future_switches() -> None:
    cfg = HiddenData.Config(seq_len=60, pred_len=8, recent_steps=(20, 40))
    episode = cfg.make().sample(count=200, seed=11)
    changed = episode.change_at > 0
    assert 0.1 < float((~changed).double().mean()) < 0.4
    recent = 60 - episode.change_at[changed]
    assert int(recent.min()) >= 20
    assert int(recent.max()) <= 40
    # The hidden partner and coupling persist; only the observed mass changes.
    assert torch.equal(episode.before[:, 4:], episode.after[:, 4:])
    assert not torch.equal(episode.before[:, :4], episode.after[:, :4])


def test_reserved_domain_moves_only_the_after_regime() -> None:
    cfg = HiddenData.Config(seq_len=60, pred_len=8, recent_steps=(20, 40))
    domain = HiddenDomain.Config(friction=(0.06, 0.12))
    episode = cfg.make().sample(count=50, seed=3, domain=domain)
    assert float(episode.after[:, 3].min()) >= 0.06
    assert float(episode.before[:, 3].max()) <= 0.02


def test_oracle_continuation_reproduces_targets_exactly() -> None:
    data = HiddenData.Config(seq_len=60, pred_len=8, recent_steps=(20, 40)).make()
    episode = data.sample(count=5, seed=2)
    oracle = data.dynamics.forecast(
        episode.states[:, 59],
        coefficients=episode.after,
        steps=8,
    )
    assert torch.equal(oracle, episode.states[:, 60:])


def test_partner_is_hidden_but_changes_the_observed_motion() -> None:
    data = HiddenData.Config(seq_len=60, pred_len=8, recent_steps=(20, 40)).make()
    episode = data.sample(count=4, seed=8)
    state = episode.states[:, 59].clone()
    state[:, 2:] = 0
    decoupled = data.dynamics.forecast(state, coefficients=episode.after, steps=8)
    assert not torch.allclose(decoupled[..., 0], episode.states[:, 60:, 0])


def test_shooting_recovers_noiseless_motion_and_rejects_other_physics() -> None:
    cfg = HiddenData.Config(seq_len=60, pred_len=16, recent_steps=(40, 50))
    cfg.observation = NoisyPosition.Config(noise=(0.0, 0.0))
    cfg.stationary = 1.0
    data = cfg.make()
    episode = data.sample(count=4, seed=21)
    method = LibraryShooting.Config(visible_steps=40, iterations=400).make()
    method.fit(episode.windows, validation=episode.windows, dynamics=data.dynamics)
    prediction = method.predict(episode.windows.media)
    assert prediction.shape == episode.windows.label.shape
    error = (prediction[..., 0] - episode.states[:, 60:, 0]).square().mean()
    persistence = episode.windows.media[:, -1:, 0] - episode.states[:, 60:, 0]
    assert float(error) < 0.05 * float(persistence.square().mean())
    with pytest.raises(TypeError):
        method.fit(
            episode.windows,
            validation=episode.windows,
            dynamics=RecentPhysics,  # pyright: ignore[reportArgumentType]
        )


def test_part_two_physics_prior_still_runs_on_the_coupled_data() -> None:
    data = HiddenData.Config(seq_len=60, pred_len=8, recent_steps=(20, 40)).make()
    episode = data.sample(count=3, seed=4)
    method = RecentPhysics.Config().make()
    method.fit(episode.windows, validation=episode.windows, dynamics=data.dynamics)
    prediction = method.predict(episode.windows.media)
    assert prediction.shape == episode.windows.label.shape
    assert torch.isfinite(prediction).all()


@pytest.mark.parametrize(
    "change",
    [
        {"recent_steps": (4, 10)},
        {"recent_steps": (20, 60)},
        {"stationary": 1.5},
    ],
)
def test_invalid_geometry_is_rejected(change: dict[str, object]) -> None:
    cfg = HiddenData.Config(seq_len=60, pred_len=8, recent_steps=(20, 40))
    for key, value in change.items():
        setattr(cfg, key, value)
    with pytest.raises(ValueError, match="must be valid"):
        cfg.make()
