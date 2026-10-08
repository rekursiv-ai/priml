"""Capture jobs of the behaviour mixture: one factory per arm, and the verifier.

The mixture has four arms and shares, filled by the port's own trained
policies:

| Arm | Factory | Policy | Share |
| --- | --- | --- | ---: |
| 0 | ``exp103_s74`` | exp103 at 20B, seed 74 (77.66%), sampled | 70% |
| 1 | ``exp103_s74_epsilon`` | the same, epsilon 0.05 over legal actions | 10% |
| 2 | ``exp102_s73`` | exp102 at 20B, seed 73 (56.24%), sampled | 10% |
| 3 | ``exp000_s73`` | exp000 at 3.5B (48.45%), sampled | 10% |

Arm 0 is the strongest policy, the one that reaches the last floor in 61% of
its evaluation episodes; arm 2 is exp102's recipe (a board encoder and
feasibility loss); arm 3 is the base recipe, a weaker player of a different
architecture. Shares are of the 3.5B-decision target, and each arm's four
workers, one per seed range, capture a quarter of it. The shares, and v2's
decisions below, are budgets: a worker finishes the episodes in flight when
its budget is met (``worker.py``), so an archive holds more, most for the arms
of the longest episodes, and its own mix differs; ``freeze_corpus.py
--decisions`` draws the shares back out of it. Every factory captures
worker 0; a launch picks the worker with ``--override worker=N`` and names
itself with ``--override launch=ID``, an id no other launch uses. Each arm
plays its policy exactly as trained, with learning, practice and auxiliary
objectives off, on the game's default rules: exp103's episodes, which ended at
the necromancer's fall in training, play on.

The policies are their runs' final weights, copied out of the runs' last
``TrainLoop`` checkpoints by ``{"step": {"model": ...}}`` alone, into
``behaviour-policies/``.

The second dataset version adds, per arm, fresh episodes with training
episodes cut after a stall without reward (``*_v2``, generation 1) and
branches of Troll, Fire and Ice states of the fresh training episodes
(``*_branch``, generation 2):

| Arm | Stall cap | Epsilon | Fresh | Branches |
| --- | ---: | --- | ---: | ---: |
| 0 | 2,000 | 0 | 3.406B | 1.460B |
| 1 | 2,000 | 0.04-0.06 per episode | 0.257B | 0.257B |
| 2 | 10,000 | 0 | 0.369B | 0.369B |
| 3 | 10,000 | 0 | 0.272B | 0.272B |

Validation episodes are never cut, and branches are training episodes only,
so the validation split is natural play.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from priml.baselines.craftax.experiments import exp000, exp102, exp103
from priml.baselines.craftax.world_model.capture.branches import (
    BranchFeeder,
)
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


def exp103_s74() -> CaptureWorker.Config:
    """Capture arm 0: exp103's seed-74 run at 20B, sampled."""
    return _arm(
        arm=0,
        share=70,
        policy=exp103().step,
        checkpoint="exp007-s74-00038146.pt",
        sha256="38dfe59f233640b571abea4efe4b71476437c0ed501e12ad8fe4e36d69039cef",
    )


def exp103_s74_epsilon() -> CaptureWorker.Config:
    """Capture arm 1: arm 0's policy with 5% of actions replaced by uniform legal ones."""
    config = exp103_s74()
    config.arm = 1
    config.decisions = 3_500_000_000 * 10 // 100 // 4
    _source(config).env.epsilon = 0.05
    return config


def exp102_s73() -> CaptureWorker.Config:
    """Capture arm 2: exp102's seed-73 run at 20B, sampled."""
    return _arm(
        arm=2,
        share=10,
        policy=exp102().step,
        checkpoint="exp006-s73-00038146.pt",
        sha256="4034e88238769309b80d4ed121832c9fe18e4479e9a7c6c4950e9c79d2d852b0",
    )


def exp000_s73() -> CaptureWorker.Config:
    """Capture arm 3: exp000's run at 3.5B, sampled."""
    return _arm(
        arm=3,
        share=10,
        policy=exp000().step,
        checkpoint="exp000-s73-00006662.pt",
        sha256="e4f2adcd0751fbd40186262bd1fab1b553b2046c30f4ad2ef5c31d743660ac10",
    )


def verifier() -> ReplayVerifier.Config:
    """Verify every closed shard; a launch sets ``launch`` and ``workers``."""
    config = ReplayVerifier.Config()
    config.root = Path("/opt/scratch/datasets/craftax/world-model/archive-v1")
    return config


def exp103_s74_v2() -> CaptureWorker.Config:
    """Capture v2's fresh arm 0: training episodes cut at 2,000 stalls."""
    return _fresh(exp103_s74(), decisions=851_508_459, stall_limit=2_000)


def exp103_s74_epsilon_v2() -> CaptureWorker.Config:
    """Capture v2's fresh arm 1: each episode's epsilon drawn from 0.04-0.06."""
    config = _fresh(exp103_s74_epsilon(), decisions=64_152_549, stall_limit=2_000)
    env = _source(config).env
    env.epsilon = 0.04
    env.epsilon_high = 0.06
    return config


def exp102_s73_v2() -> CaptureWorker.Config:
    """Capture v2's fresh arm 2: training episodes cut at 10,000 stalls."""
    return _fresh(exp102_s73(), decisions=92_152_350, stall_limit=10_000)


def exp000_s73_v2() -> CaptureWorker.Config:
    """Capture v2's fresh arm 3: training episodes cut at 10,000 stalls."""
    return _fresh(exp000_s73(), decisions=67_974_772, stall_limit=10_000)


def exp103_s74_branch() -> CaptureWorker.Config:
    """Capture v2's arm 0 branches of its Troll, Fire and Ice states."""
    return _branch(exp103_s74_v2(), decisions=364_932_197)


def exp103_s74_epsilon_branch() -> CaptureWorker.Config:
    """Capture v2's arm 1 branches of its Troll, Fire and Ice states."""
    return _branch(exp103_s74_epsilon_v2(), decisions=64_152_548)


def exp102_s73_branch() -> CaptureWorker.Config:
    """Capture v2's arm 2 branches of its Troll, Fire and Ice states."""
    return _branch(exp102_s73_v2(), decisions=92_152_350)


def exp000_s73_branch() -> CaptureWorker.Config:
    """Capture v2's arm 3 branches of its Troll, Fire and Ice states."""
    return _branch(exp000_s73_v2(), decisions=67_974_773)


def e2e_exp103_s74() -> CaptureWorker.Config:
    """Capture arm 0 at the end-to-end check's size: 64 environments, 700k decisions.

    The whole loop at small scale: each ``e2e_`` arm
    captures into one root with every in-flight episode drained, and
    ``e2e_verifier`` verifies a tenth of each shard.
    """
    return _e2e(exp103_s74(), decisions=700_000)


def e2e_exp103_s74_epsilon() -> CaptureWorker.Config:
    """Capture arm 1 at the end-to-end check's size: 100k decisions."""
    return _e2e(exp103_s74_epsilon(), decisions=100_000)


def e2e_exp102_s73() -> CaptureWorker.Config:
    """Capture arm 2 at the end-to-end check's size: 100k decisions."""
    return _e2e(exp102_s73(), decisions=100_000)


def e2e_exp000_s73() -> CaptureWorker.Config:
    """Capture arm 3 at the end-to-end check's size: 100k decisions."""
    return _e2e(exp000_s73(), decisions=100_000)


def e2e_verifier() -> ReplayVerifier.Config:
    """Verify a tenth of every shard the ``e2e_`` arms publish."""
    config = verifier()
    config.root = Path("/opt/scratch/datasets/craftax/world-model/e2e")
    config.fraction = 0.1
    return config


def _e2e(config: CaptureWorker.Config, *, decisions: int) -> CaptureWorker.Config:
    """Shrink an arm to the end-to-end check: its root, 64 environments, small shards."""
    config.root = Path("/opt/scratch/datasets/craftax/world-model/e2e")
    config.decisions = decisions
    config.shard_decisions = 100_000
    env = _source(config).env
    env.num_envs = 64
    env.num_buffers = 2
    return config


def _arm(
    *,
    arm: int,
    share: int,
    policy: CraftaxTrainStep.Config,
    checkpoint: str,
    sha256: str,
) -> CaptureWorker.Config:
    """Return worker 0 of one arm of the first dataset version."""
    config = CaptureWorker.Config()
    config.root = Path("/opt/scratch/datasets/craftax/world-model/archive-v1")
    config.arm = arm
    config.decisions = 3_500_000_000 * share // 100 // 4
    source = _source(config)
    source.policy = policy
    source.checkpoint = (
        Path("/opt/scratch/artifacts/craftax/world-model/behaviour-policies")
        / checkpoint
    )
    source.checkpoint_sha256 = sha256
    source.env.num_envs = 1_024
    source.env.num_buffers = 4
    return config


def _source(config: CaptureWorker.Config) -> PolicySource.Config:
    """Return a worker's policy source."""
    source = config.source
    assert isinstance(source, PolicySource.Config)
    return source


def _fresh(
    config: CaptureWorker.Config,
    *,
    decisions: int,
    stall_limit: int,
) -> CaptureWorker.Config:
    """Make a first-version arm the second's: generation 1, training cut at stalls."""
    config.root = Path("/opt/scratch/datasets/craftax/world-model/archive-v2")
    config.generation = 1
    config.decisions = decisions
    _source(config).env.stall_limit = stall_limit
    return config


def _branch(config: CaptureWorker.Config, *, decisions: int) -> CaptureWorker.Config:
    """Make a second-version arm branch from its node's pools, as generation 2."""
    root = Path("/opt/scratch/datasets/craftax/world-model/archive-v2-branch")
    config.root = root
    config.generation = 2
    config.decisions = decisions
    feeder = _source(config).branches = BranchFeeder.Config()
    feeder.pools = root / "pools"
    return config
