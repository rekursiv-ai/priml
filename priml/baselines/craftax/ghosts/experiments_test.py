"""Tests for the ghost-overlay captures: the tiers, the pilots and one episode per row."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from priml.baselines.craftax.ghosts import experiments
from priml.baselines.craftax.lib.compat import ExactScan
from priml.baselines.craftax.model import MinGRUPolicy
from priml.baselines.craftax.world_model.archive import (
    read_manifest,
    read_summaries,
)
from priml.baselines.craftax.world_model.capture.source import (
    PolicySource,
    RandomSource,
)
from priml.baselines.craftax.world_model.capture.verify import (
    verify_shard,
)
from priml.lib.codec import from_plain


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


def test_tiers_are_arms_0_to_2_of_their_policies() -> None:
    configs = [factory() for factory in PILOTS]
    assert [c.arm for c in configs] == [0, 1, 2]
    assert [_source(c).checkpoint_sha256[:8] for c in configs] == [
        "20a89a85",
        "4034e882",
        "38dfe59f",
    ]
    assert {(c.decisions, c.worker) for c in configs} == {(1, 0)}


@pytest.mark.parametrize("factory", [*PILOTS, experiments.capture_boss])
def test_every_tier_refuses_weights_of_another_sha256(
    factory: Callable[[], CaptureWorker.Config],
    tmp_path: Path,
) -> None:
    source = _source(factory())
    source.checkpoint = tmp_path / "policy.pt"
    source.checkpoint.write_bytes(b"not the tier's weights")
    with pytest.raises(ValueError, match="wrong SHA-256"):
        source.make()


def test_pilots_play_64_episodes_on_each_of_pool_worlds_0_to_15() -> None:
    configs = [factory().copy_tree().finalize() for factory in PILOTS]
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
    configs = [factory().copy_tree().finalize() for factory in CAPTURES]
    for capture, pilot in zip(configs, PILOTS, strict=True):
        source, piloted = _source(capture), _source(pilot())
        assert (capture.arm, capture.decisions) == (pilot().arm, 1)
        assert source.checkpoint == piloted.checkpoint
        assert source.checkpoint_sha256 == piloted.checkpoint_sha256
        assert (source.env.num_envs, source.env.num_buffers) == (1_000, 4)
        assert source.env.world_seeds == (15,)
    assert {(c.root.name, c.generation) for c in configs} == {("w15", 1)}
    assert experiments.capture_verifier().root == configs[0].root


def test_extras_play_2_048_more_episodes_of_each_tier_in_generation_2() -> None:
    captures = [*CAPTURES, experiments.capture_boss]
    extras = [
        experiments.extra_early,
        experiments.extra_medium,
        experiments.extra_high,
        experiments.extra_boss,
    ]
    for extra, capture in zip(extras, captures, strict=True):
        config, main = extra().copy_tree().finalize(), capture()
        source = _source(config)
        assert (config.arm, config.generation) == (main.arm, 2)
        assert source.checkpoint_sha256 == _source(main).checkpoint_sha256
        assert (source.env.num_envs, source.env.num_buffers) == (2_048, 4)
        assert source.env.world_seeds == (15,)
        assert config.root == main.root.with_name("w15-extra")
    assert experiments.extra_verifier().root.name == "w15-extra"


def test_the_boss_tier_is_arm_3_of_the_captures_with_the_fine_tunes_numerics() -> None:
    config = experiments.capture_boss().copy_tree().finalize()
    assert (config.arm, config.generation, config.decisions) == (3, 1, 1)
    assert config.root == experiments.capture_high().root
    source = _source(config)
    assert source.checkpoint_sha256[:8] == "6d81fc29"
    assert (source.env.world_seeds, source.env.num_envs) == ((15,), 1_000)
    model = source.policy.model
    assert isinstance(model, MinGRUPolicy.Config)
    assert model.state_dtype == model.output_dtype == model.dtype
    assert isinstance(model.block.scan, ExactScan.Config)


# A pilot's worker with random play on four of its rows: four random-play
# episodes captured, encoded, published and replayed again.
def test_a_tier_records_one_episode_per_environment_on_its_world(
    tmp_path: Path,
) -> None:
    config = experiments.pilot_high()
    config.root = tmp_path
    config.run_root = tmp_path / "runs"
    config.poll_seconds = 0.0
    env = _source(config).env
    env.num_envs = 4
    env.num_buffers = 2
    random = config.source = RandomSource.Config()
    random.env = env
    report = config.make().capture()
    assert report.episodes == 4
    rows: list[tuple[int, int, int]] = []
    for manifest in tmp_path.glob("*/arm2/w0/MANIFEST.jsonl"):
        for line in read_manifest(manifest.parent):
            verdict = verify_shard(
                manifest.parent,
                line,
                fraction=1.0,
                seed=0,
                name=line.shard,
            )
            assert verdict.mismatch == ""
            rows += [
                (
                    from_plain(s.summary["episode"], int),
                    from_plain(s.summary["environment"], int),
                    s.receipt.world_seed,
                )
                for s in read_summaries(manifest.parent, line)
            ]
    assert sorted(rows) == [(i, i, i) for i in range(4)]


def _source(config: CaptureWorker.Config) -> PolicySource.Config:
    """Return a worker's policy source."""
    source = config.source
    assert isinstance(source, PolicySource.Config)
    return source


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
