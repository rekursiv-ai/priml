"""Tests for the ghost-overlay captures: the tiers, the pilots and their settings.

A budget of one decision recording one episode per environment, each on its
world, is the capture env's doing, tested in ``capture/env_test.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from priml.baselines.craftax.ghosts import experiments
from priml.baselines.craftax.lib.compat import ExactScan
from priml.baselines.craftax.model import MinGRUPolicy
from priml.baselines.craftax.world_model.capture.source import (
    PolicySource,
)


if TYPE_CHECKING:
    from collections.abc import Callable

    from priml.baselines.craftax.world_model.capture.worker import (
        CaptureWorker,
    )


PILOTS = (experiments.pilot_early, experiments.pilot_medium, experiments.pilot_high)
CAPTURES = (
    experiments.capture_early,
    experiments.capture_medium,
    experiments.capture_high,
)


def test_tiers_are_arms_0_to_2_reading_their_runs_final_checkpoints() -> None:
    configs = [_finalized(factory) for factory in PILOTS]
    assert [c.arm for c in configs] == [0, 1, 2]
    runs = Path("/opt/scratch/runs/craftax")
    assert [_source(c).checkpoint for c in configs] == [
        runs / "exp001/checkpoints/step_00000476.pt",
        runs / "exp102/checkpoints/step_00038146.pt",
        runs / "exp103/checkpoints/step_00038146.pt",
    ]
    assert {(c.decisions, c.worker) for c in configs} == {(1, 0)}


def test_pilots_play_64_episodes_on_each_of_pool_worlds_0_to_15() -> None:
    configs = [_finalized(factory) for factory in PILOTS]
    envs = [_source(c).env for c in configs]
    assert {e.world_seeds for e in envs} == {tuple(range(16))}
    assert {(e.num_envs, e.num_buffers) for e in envs} == {(1_024, 4)}
    assert {(c.root, c.generation) for c in configs} == {
        (
            Path("/opt/scratch/artifacts/craftax/ghosts/capture/pilot"),
            0,
        ),
    }


def test_the_pilot_verifier_replays_every_pilot_episode() -> None:
    config = experiments.pilot_verifier()
    assert config.root == experiments.pilot_high().root
    assert config.fraction == 1.0


def test_captures_play_the_pilots_policies_1_000_times_on_world_15() -> None:
    configs = [_finalized(factory) for factory in CAPTURES]
    for capture, pilot in zip(configs, PILOTS, strict=True):
        source, piloted = _source(capture), _source(_finalized(pilot))
        assert (capture.arm, capture.decisions) == (pilot().arm, 1)
        assert source.checkpoint == piloted.checkpoint
        assert (source.env.num_envs, source.env.num_buffers) == (1_000, 4)
        assert source.env.world_seeds == (15,)
    assert {(c.root.name, c.generation) for c in configs} == {("w15", 1)}
    verifier = experiments.capture_verifier().copy_tree().finalize()
    assert verifier.root == configs[0].root


def test_extras_play_2_048_more_episodes_of_each_tier_in_generation_2() -> None:
    captures = [*CAPTURES, experiments.capture_boss]
    extras = [
        experiments.extra_early,
        experiments.extra_medium,
        experiments.extra_high,
        experiments.extra_boss,
    ]
    for extra, capture in zip(extras, captures, strict=True):
        config, main = _finalized(extra), _finalized(capture)
        source = _source(config)
        assert (config.arm, config.generation) == (main.arm, 2)
        assert source.checkpoint == _source(main).checkpoint
        assert (source.env.num_envs, source.env.num_buffers) == (2_048, 4)
        assert source.env.world_seeds == (15,)
        assert config.root == main.root.with_name("w15-extra")
    assert experiments.extra_verifier().root.name == "w15-extra"


def test_the_boss_tier_is_arm_3_of_the_captures_with_the_fine_tunes_numerics() -> None:
    config = _finalized(experiments.capture_boss)
    assert (config.arm, config.generation, config.decisions) == (3, 1, 1)
    assert config.root == _finalized(experiments.capture_high).root
    source = _source(config)
    assert source.checkpoint == Path(
        "/opt/scratch/artifacts/craftax/ghosts/policies/boss-s73-2999975936.pt",
    )
    assert (source.env.world_seeds, source.env.num_envs) == ((15,), 1_000)
    model = source.policy.model
    assert isinstance(model, MinGRUPolicy.Config)
    assert model.state_dtype == model.output_dtype == model.dtype
    assert isinstance(model.block.scan, ExactScan.Config)


def _finalized(factory: Callable[[], CaptureWorker.Config]) -> CaptureWorker.Config:
    """Return a factory's config with its paths resolved."""
    return factory().copy_tree().finalize()


def _source(config: CaptureWorker.Config) -> PolicySource.Config:
    """Return a worker's policy source."""
    source = config.source
    assert isinstance(source, PolicySource.Config)
    return source


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
