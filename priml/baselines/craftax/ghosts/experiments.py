"""Capture jobs of the ghost overlay: one episode per environment, all of one world.

Each model tier is one port policy, sampled as trained, as one capture arm:

| Tier | Arm | Policy | Transitions | Eval perf | Deepest floor reached |
| --- | ---: | --- | ---: | ---: | --- |
| early | 0 | exp001's final checkpoint (epoch 476) | 250M | 15.22% | dungeon 72.3% |
| medium | 1 | exp102 at 20B, seed 73 | 20.0B | 56.24% | ice realm 6.4% |
| high | 2 | exp103 at 20B, seed 74 | 20.0B | 77.66% | graveyard 60.5% |
| boss | 3 | the boss-fight fine-tune of our recipe, step 2,999,975,936 | -- | 80.06% | graveyard 65.8% |

A tier's run records exactly one episode per environment. Its budget is one
decision, so once the first recorded episode ends the env records no further
one and drains those in flight (``capture/env.py``). Every environment resets
into its episode at construction, in row order, so environment ``i`` records
ordinal ``i`` from the first draw of its own action stream, and every episode
starts at the same moment. Rows whose episode ended keep playing unrecorded
worlds until the longest one ends. Every twentieth ordinal is a validation
episode, as in every capture, so a tier's episodes lie under ``train/`` and
``val/`` of ``arm{arm}/w0``; their summaries' ``"episode"`` orders them.

- ``pilot_<tier>`` plays 1,024 environments, ordinal ``i`` on pool world
  ``i % 16``: 64 episodes on each of worlds 0-15, the worlds every tier
  trained and was evaluated on. ``select_world.py`` picks from the pilots the
  one world all tiers share.
- ``capture_<tier>`` plays 1,000 environments, every one on that world, 15, in
  seed generation 1: action streams other than the pilot's, so the pilot's
  64 episodes on world 15 are a sample independent of these 1,000.
- ``extra_<tier>`` plays 2,048 more environments on world 15, in generation 2,
  into ``w15-extra``: the page's short set takes the episodes that die or win
  within 10,000 decisions, and wins shortest first, from both captures. Of
  the 1,000 main episodes, 998, 659 and 373 die within 10,000 decisions in
  the early, medium and high tiers, which never win, and 562 in the boss
  tier, besides its 406 wins within them. The high tier's rate set the size;
  with the extra episodes 3,039, 1,990, 1,151 and 2,954 end within 10,000
  decisions, the boss tier's including 1,231 wins (``endings.py``). These
  play uncapped as well: a cap would cut every win longer than it (53 of the
  boss tier's 1,284), and the tiers that never win only need more rows.
- ``pilot_verifier``, ``capture_verifier`` and ``extra_verifier`` replay
  every episode of every shard under their root; a job sets ``launch`` and
  ``workers`` to those of the workers it follows.

World 15 lies nearest every tier's evaluation: a summed distance of 0.308,
against 0.452 for the next world, 13 (``select_world.py`` on the pilots).
Its mean returns are 34.80, 128.92 and 176.67 against evaluations' 34.40,
127.11 and 175.52.

The boss tier defeats the necromancer in 43.5% of its evaluation episodes
(4,377 of 10,053). It has no pilot, since the other three chose the world.
Capture plays on past the necromancer's fall, as replay does, so a boss
episode that wins goes on to its death or the timeout.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from priml.baselines.craftax.experiments import exp001, exp102, exp103
from priml.baselines.craftax.lib.compat import ExactScan
from priml.baselines.craftax.model import MinGRUPolicy
from priml.baselines.craftax.world_model.capture.source import (
    PolicySource,
)
from priml.baselines.craftax.world_model.capture.verify import (
    ReplayVerifier,
)
from priml.baselines.craftax.world_model.capture.worker import (
    CaptureWorker,
)


if TYPE_CHECKING:
    from priml.baselines.craftax.train_step import CraftaxTrainStep


def pilot_early() -> CaptureWorker.Config:
    """Pilot the early tier: 64 episodes on each of pool worlds 0-15."""
    return _pilot(_early())


def pilot_medium() -> CaptureWorker.Config:
    """Pilot the medium tier: 64 episodes on each of pool worlds 0-15."""
    return _pilot(_medium())


def pilot_high() -> CaptureWorker.Config:
    """Pilot the high tier: 64 episodes on each of pool worlds 0-15."""
    return _pilot(_high())


def pilot_verifier() -> ReplayVerifier.Config:
    """Replay every episode the pilots publish."""
    config = ReplayVerifier.Config()
    config.root = pilot_early().root
    config.log_dir = Path(
        "/opt/scratch/artifacts/craftax/ghosts/capture/verifier",
    )
    config.fraction = 1.0
    return config


def capture_early() -> CaptureWorker.Config:
    """Capture the early tier: 1,000 episodes of world 15."""
    return _capture(_early())


def capture_medium() -> CaptureWorker.Config:
    """Capture the medium tier: 1,000 episodes of world 15."""
    return _capture(_medium())


def capture_high() -> CaptureWorker.Config:
    """Capture the high tier: 1,000 episodes of world 15."""
    return _capture(_high())


def capture_boss() -> CaptureWorker.Config:
    """Capture the boss tier: 1,000 episodes of world 15."""
    return _capture(_boss())


def capture_verifier() -> ReplayVerifier.Config:
    """Replay every episode the captures publish."""
    config = pilot_verifier()
    config.root = capture_early().root
    return config


def extra_early() -> CaptureWorker.Config:
    """Capture 2,048 more early-tier episodes of world 15 for the short set."""
    return _extra(_early())


def extra_medium() -> CaptureWorker.Config:
    """Capture 2,048 more medium-tier episodes of world 15 for the short set."""
    return _extra(_medium())


def extra_high() -> CaptureWorker.Config:
    """Capture 2,048 more high-tier episodes of world 15 for the short set."""
    return _extra(_high())


def extra_boss() -> CaptureWorker.Config:
    """Capture 2,048 more boss-tier episodes of world 15 for the short set."""
    return _extra(_boss())


def extra_verifier() -> ReplayVerifier.Config:
    """Replay every episode the extra captures publish."""
    config = pilot_verifier()
    config.root = extra_early().root
    return config


def _early() -> CaptureWorker.Config:
    """Return the early tier's worker: exp001's final policy as arm 0."""
    return _tier(
        arm=0,
        policy=exp001().step,
        checkpoint=Path(
            "/opt/scratch/runs/craftax/exp001/checkpoints/step_00000476.pt",
        ),
        sha256="20a89a85eb8f9d48d307bdbf6b1b178979ab69e1c09032321892193616c9259f",
    )


def _medium() -> CaptureWorker.Config:
    """Return the medium tier's worker: exp102's seed-73 policy as arm 1."""
    return _tier(
        arm=1,
        policy=exp102().step,
        checkpoint=Path(
            "/opt/scratch/artifacts/craftax/world-model/behaviour-policies/"
            "exp006-s73-00038146.pt",
        ),
        sha256="4034e88238769309b80d4ed121832c9fe18e4479e9a7c6c4950e9c79d2d852b0",
    )


def _high() -> CaptureWorker.Config:
    """Return the high tier's worker: exp103's seed-74 policy as arm 2."""
    return _tier(
        arm=2,
        policy=exp103().step,
        checkpoint=Path(
            "/opt/scratch/artifacts/craftax/world-model/behaviour-policies/"
            "exp007-s74-00038146.pt",
        ),
        sha256="38dfe59f233640b571abea4efe4b71476437c0ed501e12ad8fe4e36d69039cef",
    )


def _boss() -> CaptureWorker.Config:
    """Return the boss tier's worker: the boss-fight fine-tune as arm 3."""
    policy = exp103().step
    model = policy.model
    assert isinstance(model, MinGRUPolicy.Config)
    # The fine-tune's numerics: bf16 carry and decoder output, PufferLib's scan.
    model.state_dtype = model.output_dtype = model.dtype
    model.block.scan = ExactScan.Config()
    return _tier(
        arm=3,
        policy=policy,
        checkpoint=Path(
            "/opt/scratch/artifacts/craftax/ghosts/policies/boss-s73-2999975936.pt",
        ),
        sha256="6d81fc29e14e3360a983a14041de988621b3608e4d3c26da0f0e8201f21c7d49",
    )


def _tier(
    *,
    arm: int,
    policy: CraftaxTrainStep.Config,
    checkpoint: Path,
    sha256: str,
) -> CaptureWorker.Config:
    """Return worker 0 of a tier: its policy as an arm, one episode per environment."""
    config = CaptureWorker.Config()
    config.arm = arm
    config.decisions = 1
    source = _source(config)
    source.policy = policy
    source.checkpoint = checkpoint
    source.checkpoint_sha256 = sha256
    source.env.num_buffers = 4
    return config


def _pilot(config: CaptureWorker.Config) -> CaptureWorker.Config:
    """Make a tier's worker its pilot: 64 episodes on each of pool worlds 0-15."""
    config.root = Path(
        "/opt/scratch/artifacts/craftax/ghosts/capture/pilot",
    )
    env = _source(config).env
    env.world_seeds = tuple(range(16))
    env.num_envs = 64 * len(env.world_seeds)
    return config


def _capture(config: CaptureWorker.Config) -> CaptureWorker.Config:
    """Make a tier's worker its capture: 1,000 episodes of world 15, generation 1."""
    env = _source(config).env
    env.world_seeds = (15,)
    env.num_envs = 1_000
    config.generation = 1
    config.root = (
        Path(
            "/opt/scratch/artifacts/craftax/ghosts/capture",
        )
        / f"w{env.world_seeds[0]}"
    )
    return config


def _extra(config: CaptureWorker.Config) -> CaptureWorker.Config:
    """Make a tier's worker its extra capture: 2,048 more of world 15, generation 2."""
    config = _capture(config)
    _source(config).env.num_envs = 2_048
    config.generation = 2
    config.root = config.root.with_name(f"{config.root.name}-extra")
    return config


def _source(config: CaptureWorker.Config) -> PolicySource.Config:
    """Return a worker's policy source."""
    source = config.source
    assert isinstance(source, PolicySource.Config)
    return source
