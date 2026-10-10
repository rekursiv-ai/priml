"""Capture jobs of the behaviour mixture: one factory per arm, and the verifier.

The mixture has four arms and shares, filled by the port's own trained
policies:

| Arm | Factory | Policy | Share |
| --- | --- | --- | ---: |
| 0 | ``arm0`` | exp103 at 20B, sampled | 70% |
| 1 | ``arm1`` | the same, epsilon 0.05 over legal actions | 10% |
| 2 | ``arm2`` | exp102 at 20B, sampled | 10% |
| 3 | ``arm3`` | exp000 at 3.5B, sampled | 10% |

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

Each arm reads its policy experiment's final ``TrainLoop`` checkpoint, at the
path that experiment's run writes beneath ``base_dir``, so the experiment
trains first. The published archive's arms played exp103's seed-74 run
(77.66%), exp102's seed-73 run (56.24%) and exp000's run (48.45%).

The second dataset version adds, per arm, fresh episodes with training
episodes cut after a stall without reward (``arm*_v2``, generation 1) and
branches of Troll, Fire and Ice states of the fresh training episodes
(``arm*_branch``, generation 2):

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

from priml.baselines.craftax.experiments import (
    exp000,
    exp102,
    exp103,
    exp_smoke,
)
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


def arm0() -> CaptureWorker.Config:
    """Capture arm 0: exp103's run at 20B, sampled; seed 74's in the published archive."""
    config = CaptureWorker.Config()
    config.root = Path("/datasets/craftax/world-model/archive-v1")
    config.arm = 0
    config.decisions = 3_500_000_000 * 70 // 100 // 4
    source = config.source
    assert isinstance(source, PolicySource.Config)
    policy = exp103()
    source.policy = policy.step
    source.checkpoint = Path(
        f"/runs/{policy.study_name}/{policy.experiment_name}/checkpoints/"
        f"step_{int(policy.max_steps):08d}.pt",
    )
    source.env.num_envs = 1_024
    source.env.num_buffers = 4
    return config


def arm1() -> CaptureWorker.Config:
    """Capture arm 1: arm 0's policy with 5% of actions replaced by uniform legal ones."""
    config = arm0()
    config.arm = 1
    config.decisions = 3_500_000_000 * 10 // 100 // 4
    source = config.source
    assert isinstance(source, PolicySource.Config)
    source.env.epsilon = 0.05
    return config


def arm2() -> CaptureWorker.Config:
    """Capture arm 2: exp102's run at 20B, sampled; seed 73's in the published archive."""
    config = arm0()
    config.arm = 2
    config.decisions = 3_500_000_000 * 10 // 100 // 4
    source = config.source
    assert isinstance(source, PolicySource.Config)
    policy = exp102()
    source.policy = policy.step
    source.checkpoint = Path(
        f"/runs/{policy.study_name}/{policy.experiment_name}/checkpoints/"
        f"step_{int(policy.max_steps):08d}.pt",
    )
    return config


def arm3() -> CaptureWorker.Config:
    """Capture arm 3: exp000's run at 3.5B, sampled; seed 73, its config's own."""
    config = arm2()
    config.arm = 3
    source = config.source
    assert isinstance(source, PolicySource.Config)
    policy = exp000()
    source.policy = policy.step
    source.checkpoint = Path(
        f"/runs/{policy.study_name}/{policy.experiment_name}/checkpoints/"
        f"step_{int(policy.max_steps):08d}.pt",
    )
    return config


def verifier() -> ReplayVerifier.Config:
    """Verify every closed shard; a launch sets ``launch`` and ``workers``."""
    config = ReplayVerifier.Config()
    config.root = Path("/datasets/craftax/world-model/archive-v1")
    return config


def arm0_v2() -> CaptureWorker.Config:
    """Capture v2's fresh arm 0: training episodes cut at 2,000 stalls."""
    config = arm0()
    config.root = Path("/datasets/craftax/world-model/archive-v2")
    config.generation = 1
    config.decisions = 851_508_459
    source = config.source
    assert isinstance(source, PolicySource.Config)
    source.env.stall_limit = 2_000
    return config


def arm1_v2() -> CaptureWorker.Config:
    """Capture v2's fresh arm 1: each episode's epsilon drawn from 0.04-0.06."""
    config = arm1()
    config.root = Path("/datasets/craftax/world-model/archive-v2")
    config.generation = 1
    config.decisions = 64_152_549
    source = config.source
    assert isinstance(source, PolicySource.Config)
    source.env.stall_limit = 2_000
    source.env.epsilon = 0.04
    source.env.epsilon_high = 0.06
    return config


def arm2_v2() -> CaptureWorker.Config:
    """Capture v2's fresh arm 2: training episodes cut at 10,000 stalls."""
    config = arm2()
    config.root = Path("/datasets/craftax/world-model/archive-v2")
    config.generation = 1
    config.decisions = 92_152_350
    source = config.source
    assert isinstance(source, PolicySource.Config)
    source.env.stall_limit = 10_000
    return config


def arm3_v2() -> CaptureWorker.Config:
    """Capture v2's fresh arm 3: training episodes cut at 10,000 stalls."""
    config = arm3()
    config.root = Path("/datasets/craftax/world-model/archive-v2")
    config.generation = 1
    config.decisions = 67_974_772
    source = config.source
    assert isinstance(source, PolicySource.Config)
    source.env.stall_limit = 10_000
    return config


def arm0_branch() -> CaptureWorker.Config:
    """Capture v2's arm 0 branches of its Troll, Fire and Ice states."""
    config = arm0_v2()
    config.root = Path("/datasets/craftax/world-model/archive-v2-branch")
    config.generation = 2
    config.decisions = 364_932_197
    source = config.source
    assert isinstance(source, PolicySource.Config)
    feeder = source.branches = BranchFeeder.Config()
    feeder.pools = config.root / "pools"
    return config


def arm1_branch() -> CaptureWorker.Config:
    """Capture v2's arm 1 branches of its Troll, Fire and Ice states."""
    config = arm1_v2()
    config.root = Path("/datasets/craftax/world-model/archive-v2-branch")
    config.generation = 2
    config.decisions = 64_152_548
    source = config.source
    assert isinstance(source, PolicySource.Config)
    feeder = source.branches = BranchFeeder.Config()
    feeder.pools = config.root / "pools"
    return config


def arm2_branch() -> CaptureWorker.Config:
    """Capture v2's arm 2 branches of its Troll, Fire and Ice states."""
    config = arm2_v2()
    config.root = Path("/datasets/craftax/world-model/archive-v2-branch")
    config.generation = 2
    config.decisions = 92_152_350
    source = config.source
    assert isinstance(source, PolicySource.Config)
    feeder = source.branches = BranchFeeder.Config()
    feeder.pools = config.root / "pools"
    return config


def arm3_branch() -> CaptureWorker.Config:
    """Capture v2's arm 3 branches of its Troll, Fire and Ice states."""
    config = arm3_v2()
    config.root = Path("/datasets/craftax/world-model/archive-v2-branch")
    config.generation = 2
    config.decisions = 67_974_773
    source = config.source
    assert isinstance(source, PolicySource.Config)
    feeder = source.branches = BranchFeeder.Config()
    feeder.pools = config.root / "pools"
    return config


def e2e_arm0() -> CaptureWorker.Config:
    """Capture arm 0 at the end-to-end check's size: 64 environments, 700k decisions.

    The whole loop at small scale: each ``e2e_`` arm
    captures into one root with every in-flight episode drained, and
    ``e2e_verifier`` verifies a tenth of each shard.
    """
    config = arm0()
    config.root = Path("/datasets/craftax/world-model/e2e")
    config.decisions = 700_000
    config.shard_decisions = 100_000
    source = config.source
    assert isinstance(source, PolicySource.Config)
    source.env.num_envs = 64
    source.env.num_buffers = 2
    return config


def e2e_arm1() -> CaptureWorker.Config:
    """Capture arm 1 at the end-to-end check's size: 100k decisions."""
    config = arm1()
    config.root = Path("/datasets/craftax/world-model/e2e")
    config.decisions = 100_000
    config.shard_decisions = 100_000
    source = config.source
    assert isinstance(source, PolicySource.Config)
    source.env.num_envs = 64
    source.env.num_buffers = 2
    return config


def e2e_arm2() -> CaptureWorker.Config:
    """Capture arm 2 at the end-to-end check's size: 100k decisions."""
    config = arm2()
    config.root = Path("/datasets/craftax/world-model/e2e")
    config.decisions = 100_000
    config.shard_decisions = 100_000
    source = config.source
    assert isinstance(source, PolicySource.Config)
    source.env.num_envs = 64
    source.env.num_buffers = 2
    return config


def e2e_arm3() -> CaptureWorker.Config:
    """Capture arm 3 at the end-to-end check's size: 100k decisions."""
    config = arm3()
    config.root = Path("/datasets/craftax/world-model/e2e")
    config.decisions = 100_000
    config.shard_decisions = 100_000
    source = config.source
    assert isinstance(source, PolicySource.Config)
    source.env.num_envs = 64
    source.env.num_buffers = 2
    return config


def e2e_verifier() -> ReplayVerifier.Config:
    """Verify a tenth of every shard the ``e2e_`` arms publish."""
    config = verifier()
    config.root = Path("/datasets/craftax/world-model/e2e")
    config.fraction = 0.1
    return config


def smoke() -> CaptureWorker.Config:
    """Capture arm 3 at minimum size from exp_smoke's policy, to check the chain.

    Not a dataset: it answers whether a policy run's checkpoint captures into
    shards that freeze into a corpus the world model's ``exp_smoke`` reads.
    exp_smoke is exp000 at minimum size, so the arm is exp000's; its 16
    environments record 20,000 decisions into ``world-model/smoke``.
    """
    config = e2e_arm3()
    config.root = Path("/datasets/craftax/world-model/smoke")
    config.decisions = 20_000
    config.shard_decisions = 10_000
    source = config.source
    assert isinstance(source, PolicySource.Config)
    policy = exp_smoke()
    source.policy = policy.step
    source.checkpoint = Path(
        f"/runs/{policy.study_name}/{policy.experiment_name}/checkpoints/"
        f"step_{int(policy.max_steps):08d}.pt",
    )
    source.env.num_envs = 16
    return config
