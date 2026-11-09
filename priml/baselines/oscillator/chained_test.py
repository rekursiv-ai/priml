"""Check the chained generator, its stuck sensor, schedule, and shooting reference."""

import pytest
import torch

from priml.baselines.oscillator.chained import (
    ChainedData,
    ChainedDomain,
    ChainedDynamics,
    ChainedShooting,
    StuckSensor,
)
from priml.baselines.oscillator.forced import ForcedDynamics


def small(**changes: object) -> ChainedData.Config:
    cfg = ChainedData.Config(
        seq_len=96,
        pred_len=8,
        recent_steps=(16, 20),
        segment_steps=(16, 18),
        outage=(0, 0),
    )
    for key, value in changes.items():
        setattr(cfg, key, value)
    return cfg


def test_pool_is_nested_and_oracle_continuation_is_exact() -> None:
    data = small().make()
    pool = data.sample(count=3, seed=17)
    larger = data.sample(count=9, seed=17)
    assert pool.states.shape == (3, 104, 8)
    assert torch.equal(pool.states, larger.states[:3])
    oracle = data.dynamics.forecast(
        larger.states[:, 95],
        coefficients=larger.after,
        steps=8,
    )
    assert torch.equal(oracle, larger.states[:, 96:])


def test_far_mass_and_second_tone_change_the_observed_motion() -> None:
    data = small().make()
    episode = data.sample(count=8, seed=8)
    start = episode.states[:, 95]
    for column in (11, 14):
        removed = episode.after.clone()
        removed[:, column] = 0
        path = data.dynamics.forecast(start, coefficients=removed, steps=8)
        assert not torch.allclose(path[..., 0], episode.states[:, 96:, 0])


def test_up_to_four_changes_keep_the_hidden_chain_and_drive() -> None:
    episode = small().make().sample(count=400, seed=11)
    changed = episode.change_at > 0
    assert 0.05 < float((~changed).double().mean()) < 0.16
    assert int((96 - episode.change_at[changed]).max()) <= 20
    assert torch.equal(episode.before[:, 6:], episode.after[:, 6:])
    assert not torch.equal(episode.before[changed, :6], episode.after[changed, :6])


def test_stuck_sensor_repeats_readings_in_bursts() -> None:
    cfg = small()
    cfg.observation = StuckSensor.Config(
        noise=(0, 0),
        spike_rate=0,
        quantum=0,
        stuck_rate=0.05,
    )
    episode = cfg.make().sample(count=64, seed=5)
    media = episode.windows.media[..., 0]
    frozen = media[:, 1:] == media[:, :-1]
    assert 0.1 < float(frozen.double().mean()) < 0.4
    assert torch.equal(media[~frozen.any(1)], episode.states[~frozen.any(1), :96, 0])
    cfg.observation.stuck_rate = 0
    clean = cfg.make().sample(count=64, seed=5)
    assert torch.equal(clean.windows.media[..., 0], episode.states[:, :96, 0])


def test_outage_freezes_only_the_newest_readings() -> None:
    cfg = small(outage=(3, 6))
    data = cfg.make()
    episode = data.sample(count=32, seed=4)
    live = small().make().sample(count=32, seed=4)
    assert torch.equal(episode.windows.label, live.windows.label)
    media, before = episode.windows.media[..., 0], live.windows.media[..., 0]
    held = (media != before).cumsum(1) > 0
    assert torch.equal(media[:, -7:-6], before[:, -7:-6])
    assert torch.equal(media[:, -3:], media[:, -1:].expand(-1, 3))
    assert int(held.sum(1).max()) <= 6
    with pytest.raises(ValueError, match="must be valid"):
        small(outage=(8, 4)).make()


def test_shooting_needs_the_chained_form_and_fits_clean_motion() -> None:
    cfg = small(seq_len=48, pred_len=8, changes=(1.0,), recent_steps=(40, 40))
    cfg.observation = StuckSensor.Config(
        noise=(0, 0),
        spike_rate=0,
        quantum=0,
        stuck_rate=0,
    )
    cfg.dynamics.substeps = 2
    data = cfg.make()
    episode = data.sample(count=2, seed=21)
    method = ChainedShooting.Config(visible_steps=24, iterations=200, restarts=4).make()
    method.fit(episode.windows, validation=episode.windows, dynamics=data.dynamics)
    prediction = method.predict(episode.windows.media)
    assert prediction.shape == episode.windows.label.shape
    assert torch.isfinite(prediction).all()
    with pytest.raises(TypeError):
        method.fit(
            episode.windows,
            validation=episode.windows,
            dynamics=ForcedDynamics.Config().make(),
        )


def test_invalid_ranges_are_rejected() -> None:
    with pytest.raises(ValueError, match="physical"):
        ChainedDomain.Config(far_stiffness=(0.0, 1.0)).make()
    with pytest.raises(ValueError, match="physical"):
        ChainedDomain.Config(far_damping=(-0.1, 0.1)).make()
    with pytest.raises(ValueError, match="nonnegative"):
        StuckSensor.Config(stuck_length=0.5).make()
    assert isinstance(ChainedDynamics.Config().make(), ForcedDynamics)


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
