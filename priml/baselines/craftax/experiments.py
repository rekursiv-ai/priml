r"""Craftax experiments: PufferLib's trainer, ported bit for bit.

``exp000`` is PufferLib's own Craftax recipe (``config/craftax.ini`` at pin
``6ffa5b10``) at its one-GPU budget, its policy drawing its own init from
PufferLib's distributions under PufferLib's seed, 73. The class defaults are
PufferLib's ``config/default.ini``; exp000 states what ``craftax.ini``
changes. Every later experiment forks a named parent and applies ONE change.
exp000-exp004 are the main line, ending in our recipe (exp003) and its
boss-fight fine-tune (exp004); exp005-exp008 are side runs off it;
exp009-exp015 are baselines, published recipes restated here. The
world-model experiments, an experimental alternative to the main line, are
``experiments_world_model``'s.

    exp000     PufferLib's recipe, 3,492,806,656 transitions, 6,662 epochs
      +-- exp001     the same recipe on the port's defaults, not PufferLib's bits
      |     +-- exp002     board encoder, injection, feasibility loss, 20B
      |           +-- exp003     frontier practice, self-imitation, rewards / 8
      |           |     +-- exp004     fine-tuned for the boss fight, 3B more
      |           |     +-- exp008     a newly generated world at every reset
      |           +-- exp006     width 2,048 on 1,024 environments x 512 steps
      |                 +-- exp007     a ConvNeXt trunk on the board
      +-- exp005     the same recipe at 250M transitions, the qualification run
      +-- exp009     the same recipe on original Craftax as published
      |     +-- exp010     Craftax_Baselines' 1B PPO recipe on that setup
      |           +-- exp011     its 1M-interaction geometry, 256 envs x 16 steps
      |           +-- exp012     PPO-RNN: a reset-aware GRU, whole trajectories
      |           +-- exp013     PQN: Q-learning with an LSTM in place of PPO
      |           +-- exp014     its 1B geometry at 100M, a screening budget
      |           +-- exp015     GTrXL: PPO with gated Transformer-XL memory
      +-- exp_smoke  the same recipe at minimum size, to check a machine

Every random stream is seeded in the config, so a config replays its run: the
environments' ``rand_r`` (``env.seed``), the action streams (``sampler.seed``),
and torch's generator (``seed``), which draws the policy's init. exp000 sets
PufferLib's seed, 73, and every fork keeps it, except exp010 and its forks,
which take their reference's seed, 42, for every stream they draw.

Launch::

    uv --quiet run --frozen python -m priml priml.baselines.craftax.experiments.exp000

References:
    https://github.com/PufferAI/PufferLib
        Suarez. PufferLib (MIT license), ``config/craftax.ini``, pin ``6ffa5b10``.

"""

from __future__ import annotations

from dataclasses import field
from typing import TYPE_CHECKING, Self, override

import math

from configgle import Makeable, Makes, PartialConfig

from priml.baselines.craftax.data import CraftaxRollouts
from priml.baselines.craftax.env import (
    BossFightReward,
    FreshWorlds,
    StallCap,
    WorldPool,
)
from priml.baselines.craftax.learners.gtrxl_train_step import TrajectoryWindows
from priml.baselines.craftax.learners.imitation import BranchImitation
from priml.baselines.craftax.learners.pqn_train_step import CraftaxPQNTrainLoop
from priml.baselines.craftax.learners.practice import FrontierPractice
from priml.baselines.craftax.learners.rnn_update import ShuffledTrajectories
from priml.baselines.craftax.learners.update import ShuffledTransitions
from priml.baselines.craftax.lib.adam import ClippedAdam
from priml.baselines.craftax.lib.compat import (
    ExactMuon,
    ExactPhiloxSampler,
    ExactPPO,
    ExactScan,
)
from priml.baselines.craftax.metric import CraftaxScore
from priml.baselines.craftax.model import (
    DenseObservation,
    FeasibilityLoss,
    MinGRUPolicy,
)
from priml.baselines.craftax.policies.actor_critic import ActorCritic
from priml.baselines.craftax.policies.encoder import (
    MLP,
    BoardEncoder,
    ConvNextTrunk,
)
from priml.baselines.craftax.policies.gtrxl import GTrXLPolicy
from priml.baselines.craftax.policies.pqn import EpsilonGreedy
from priml.baselines.craftax.policies.rnn import ActorCriticRNN
from priml.baselines.craftax.rollout import PhiloxSampler
from priml.baselines.craftax.train_step import (
    AgentWindows,
    CraftaxTrainStep,
    ProgressSchedule,
    cosine_annealing_fp32,
)
from priml.loss.policy_gradient_kernel import TritonPPO
from priml.math.schedules import constant, linear, one_cycle
from priml.model.embedding import MultiHotEmbedding
from priml.optimizers.fused_muon import FusedMuon
from priml.runtime import SingleProcess
from priml.train.checkpointer import Checkpointer
from priml.train.custom_types import CheckpointerProtocol
from priml.train.tracker import (
    AsyncTracker,
    FileTracker,
    TrackerList,
    WandbTracker,
    unwrap_tracker_config,
)
from priml.train.train_loop import TrainLoop


if TYPE_CHECKING:
    from torch import nn
else:
    from wrapt import lazy_import

    # ~1050 ms; only exp002's zero init reads it.
    nn = lazy_import("torch.nn")


class CraftaxTrainLoop(
    Makes["TrainLoop"],
    TrainLoop.Config[CraftaxTrainStep.Config, CraftaxRollouts.Config],
):
    """A training loop with the Craftax step, the epoch cadence and checkpoints in place.

    A checkpoint holds the whole pipeline, rollout slot and environments
    included, so a resumed run continues as if never stopped: about 1.3 GB for
    exp000 and 9.1 GB for exp009's symbolic observations. One is written every
    200 epochs and the newest two are kept (2.6 and 18 GB), where keeping all 34
    of a run would fill 44 and 310 GB. A run compared checkpoint by checkpoint
    with PufferLib's, whose masters it hashes, keeps them all:
    ``keep_last_n=-1``.
    """

    step: CraftaxTrainStep.Config = field(default_factory=CraftaxTrainStep.Config)
    """The actor, the learner and the learning rule."""

    dataset: CraftaxRollouts.Config = field(default_factory=CraftaxRollouts.Config)
    """One tick per epoch; the data lives in the step's environments."""

    checkpointer: Makeable[CheckpointerProtocol] | None = field(
        default_factory=lambda: Checkpointer.Config(save_every=200, keep_last_n=2),
    )
    """Every 200 epochs, the newest two kept."""

    @override
    def finalize(self) -> Self:
        # Unnamed, W&B makes a name up, and a board of seeds cannot be told apart.
        dashboard = _dashboard(self.tracker)
        if dashboard is not None and not dashboard.name:
            dashboard.name = self.experiment_name
        if self.step.base_dir is None:
            self.step.base_dir = self.base_dir
        return super().finalize()


def exp000() -> CraftaxTrainLoop:
    """PufferLib's Craftax trainer at its one-GPU budget, from the policy's own init.

    Frozen: it is the reference every fork is measured against.

    The policy draws its weights from PufferLib's distributions by PufferLib's
    algorithm -- the table from N(0, 1), every projection from
    U(+-1/sqrt(fan_in)), in fp32, rounded once to bf16 -- from torch's
    generator under PufferLib's seed, 73.

    Hypothesis:
      The port runs PufferLib's training exactly -- the same masters after
      every epoch, the same final score -- at PufferLib's speed or better.
      Training depends on the init's distribution, not on its draws, so from
      its own init it scores roughly PufferLib's 48.45%, inside the ~9pp that
      same-seed reruns spread at 20B transitions.

    References:
      https://github.com/PufferAI/PufferLib (pin 6ffa5b10, config/craftax.ini)

    Results:
      H200 + 16 CPUs, 3.49B transitions: 47.20% perf (seed 73).

      From its own init: perf 47.20% (score 106.68, achievement rate 68.85%)
      over 10,011 episodes. Training took 1,871 s for 3.49B transitions
      (1.87M transitions/s), and the eval 24 s.

      The measurements below were made from PufferLib's seed-73 init, which
      this recipe loaded before it drew its own: perf 48.452%, 1.25 points
      above its own init's, inside the same-seed spread.
      Bit-identical to PufferLib's run: all 34
      masters checkpoints and the final masters (9b8eeb39...) equal
      PufferLib's files, and the final evaluation is PufferLib's to the last
      digit (perf 48.452%, 10,016 episodes).

      Speed, one H200 and 16 CPUs, interleaved in one allocation:

        | | PufferLib | port | port / PufferLib |
        |---|---|---|---|
        | training, transitions/s | 1,322,762 | 1,936,186 | 1.46x |
        | eval, 10,000 episodes, s | 30.37 | 23.00 | 0.76x |

      The final refactored tree trains at 1.928M transitions/s, 0.998x the
      pre-refactor tree in one allocation, with byte-equal masters after 40
      epochs.

    Returns:
      cfg: Configured CraftaxTrainLoop.

    """
    cfg = CraftaxTrainLoop()
    cfg.study_name = "craftax"
    cfg.experiment_name = "exp000"
    cfg.seed = 73

    # PufferLib's kernels round as nvcc compiled them; the compat classes carry that
    # arithmetic, which the bits of every epoch follow. Its build also keeps the carry,
    # the decoder's output and every stored rollout value in its one precision, the
    # weights' bf16 (``precision_t``).
    model = cfg.step.model
    assert isinstance(model, MinGRUPolicy.Config)
    model.block.scan = ExactScan.Config()
    model.state_dtype = model.output_dtype = cfg.step.rollout.dtype = model.dtype
    cfg.step.sampler = ExactPhiloxSampler.Config()

    muon = cfg.step.optimizer = ExactMuon.Config()
    muon.lr = 0.00472887093
    muon.momentum = 0.930526733
    muon.max_grad_norm = 0.818616688

    windows = cfg.step.learner
    assert isinstance(windows, AgentWindows.Config)
    ppo = windows.objective = ExactPPO.Config()
    ppo.discount = 0.999414682
    ppo.trace_decay = 0.801190972
    ppo.clip_epsilon = 0.226401091
    ppo.value_clip_epsilon = 1.14975154
    ppo.value_coefficient = 0.78808409
    ppo.entropy_coefficient = 0.000213571053
    windows.minibatch_size = 32_768
    windows.replay_ratio = 1.17263889

    # PufferLib anneals in fp32, which the bits of every epoch follow.
    cfg.step.schedule = PartialConfig(cosine_annealing_fp32, minimum=0.0)

    # PufferLib's budget is in transitions and runs whole epochs of one rollout.
    epochs = 3_492_806_656 // (cfg.step.env.num_envs * cfg.step.rollout.horizon)
    cfg.max_steps = cfg.step.train_budget_steps = epochs

    # PufferLib evaluates once, after training (``pufferl.cu:3255-3271``).
    cfg.num_steps_eval = -1
    cfg.metrics_eval["craftax"] = CraftaxScore.Config()

    dashboard = WandbTracker.Config()
    dashboard.project = "craftax"
    wrapper = AsyncTracker.Config()
    wrapper.tracker = dashboard
    cfg.tracker = TrackerList.Config()
    cfg.tracker.trackers = {"metrics": FileTracker.Config(), "wandb": wrapper}
    return cfg


def exp001() -> CraftaxTrainLoop:
    """exp000 on the port's defaults: standard kernel arithmetic and fp32 precision.

    exp000 pins everything PufferLib's bits depend on; exp001 keeps its recipe
    (every coefficient, the budget, the evaluation) and its own init, and
    swaps each pin for the class default: the scan, PPO, Muon and sampler
    kernels in standard Triton and torch arithmetic instead of the compat
    classes, priml's cosine instead of the fp32 one, and an fp32 carry,
    decoder output and rollout storage instead of bf16. It is the
    configuration the port recommends outside bit-for-bit work.

    Hypothesis:
      The defaults trade PufferLib's bits for precision and simplicity at no
      cost: exp001 scores roughly exp000 (47.20%), inside the same-seed spread,
      within 2% of its speed.

    References:
      ``lib/compat.py``: the kernel classes that carry PufferLib's arithmetic.

    Results:
      H200 + 16 CPUs, 3.49B transitions: 50.27% perf (seed 73).

      Confirmed: perf 50.27% over 10,025 episodes (on the final code),
      against exp000's 47.20% and PufferLib's 48.45% from PufferLib's init:
      inside the same-seed spread, so the defaults cost no reward. Training took
      1,875 s for 3.49B transitions (1.86M transitions/s), and the eval 30 s;
      in one allocation the defaults train 1.03x exp000's throughput.

    Returns:
      cfg: Configured CraftaxTrainLoop.

    """
    cfg = exp000()
    cfg.experiment_name = "exp001"
    defaults = CraftaxTrainLoop().step
    model, default_model = cfg.step.model, defaults.model
    assert isinstance(model, MinGRUPolicy.Config)
    assert isinstance(default_model, MinGRUPolicy.Config)
    model.block.scan = default_model.block.scan
    model.state_dtype = default_model.state_dtype
    model.output_dtype = default_model.output_dtype
    cfg.step.rollout.dtype = defaults.rollout.dtype
    cfg.step.schedule = defaults.schedule
    # Each compat class only swaps arithmetic and adds no field, so its parent
    # takes its fields as they are.
    sampler, optimizer, windows = cfg.step.sampler, cfg.step.optimizer, cfg.step.learner
    assert isinstance(sampler, ExactPhiloxSampler.Config)
    assert isinstance(optimizer, ExactMuon.Config)
    assert isinstance(windows, AgentWindows.Config)
    assert isinstance(windows.objective, ExactPPO.Config)
    cfg.step.sampler = PhiloxSampler.Config().update(sampler)
    cfg.step.optimizer = FusedMuon.Config().update(optimizer)
    windows.objective = TritonPPO.Config().update(windows.objective)
    return cfg


def exp002() -> CraftaxTrainLoop:
    """exp001 with a board encoder, layer injection and a feasibility loss, for 20B.

    The board-encoder recipe of the original runs, the strongest scratch run
    among them, restated as one fork as exp010 restates its reference. The
    policy is exp001's four-layer MinGRU with three additions:

    - a first stage (:class:`~priml.baselines.craftax.policies.encoder.BoardEncoder`)
      that reads the view as a 9x11 board through a residual depthwise CNN, a
      128-wide spatial branch FiLM-conditioned on the status scalars and a
      multiscale pooling mix, the scalars themselves refined by two residual
      MLPs conditioned on the previous action, which the observation now
      carries;
    - a 1024 -> 128 -> 1024 injection from the encoder's projection into
      every layer's input;
    - a 43-way head whose loss, weighted 0.1, predicts whether the action
      taken changes the observation.

    The learner takes the recipe's value coefficient 0.5 and value clip 0.2,
    and Muon at a quarter of exp000's rate, warmed up over the first 250M
    transitions and cosine-annealed to zero over 20B. Everything else is
    exp001's, and the evaluation runs every 1B transitions.

    What differs from the reference run, besides the random streams: the
    encoder drops an input that the reference fed zeros by design and a
    gradient-scale wrapper at scale 1; the carry, the decoder's output and the
    rollout's log-probabilities, values and rewards are fp32 (exp001's
    defaults) where it kept bf16, the observations bf16 in both; the warmup is
    ``one_cycle``'s cosine ramp where it was linear, the decay identical; the
    first observation's previous action reads "none" where it read action 0.

    Hypothesis:
      The encoder, injection and feasibility loss are one treatment, the
      representation of the strongest earlier scratch recipe, with the
      coefficients it was tuned with. At 20B it scores well above PufferLib's
      recipe at that budget (48.61%), in the range its reference measured:
      66.83% at seed 73, 57.62% on a same-seed rerun, 52.2% at seed 74 after
      17.8B. The reference trained at 573,653 transitions/s (its median) on
      one H200; the port matches or beats that.

    References:
      https://arxiv.org/abs/1709.07871
        Perez et al. 2018. FiLM: visual reasoning with a general conditioning
        layer.

    Results:
      H200 + 16 CPUs, 20B transitions: 52.76% mean perf (4 seeds: 73-76).

      Within the spread of the reference's own runs of the recipe, at 0.89x
      its speed. Four seeds at 20B, each on one H200:

        | | seed 73 | seed 74 | seed 75 | seed 76 | mean | transitions/s |
        |---|---|---|---|---|---|---|
        | exp002 | 56.24% | 54.77% | 50.01% | 50.02% | 52.76% | ~512k |
        | reference runs | 66.83% | 49.90% | - | - | - | 573,653 |

      Final evaluations over about 10,000 episodes. The reference's seed-73
      run is the best of its 15 runs of the recipe's family: a full rerun at
      the same seed finished at 57.62%, another read 61.4% at 17B before it
      crashed, and its seed-74 run finished at 49.90%. Compared field by
      field, with their code paths and their training curves, the two
      recipes differ in nothing beyond the list above; the gap to the
      reference's seed-73 run is that run's draw. With a bf16 carry
      instead, the final checkpoints score -0.3 to +0.8 points from their
      own. Seed 75 shared its node and ran at ~430k transitions/s.

    Returns:
      cfg: Configured CraftaxTrainLoop.

    """
    cfg = exp001()
    cfg.experiment_name = "exp002"
    # Every seed of a recipe files under one W&B group, so they share a board.
    trackers = cfg.tracker
    assert isinstance(trackers, TrackerList.Config)
    dashboard = unwrap_tracker_config(trackers.trackers["wandb"])
    assert isinstance(dashboard, WandbTracker.Config)
    dashboard.group = "exp002"
    cfg.step.env.rules.previous_action = True

    model = cfg.step.model
    assert isinstance(model, MinGRUPolicy.Config)
    model.embedding = BoardEncoder.Config()
    injection = model.injection = MLP.Config()
    injection.channels_hidden = 128
    injection.proj_out.init_weight = nn.init.zeros_
    feasibility = model.auxiliary = FeasibilityLoss.Config()
    feasibility.coefficient = 0.1

    muon = cfg.step.optimizer
    assert isinstance(muon, FusedMuon.Config)
    # fp32(0.00472887093) / 4, exactly: the reference's quarter of exp000's rate.
    muon.lr = 0.001182217733003199
    windows = cfg.step.learner
    assert isinstance(windows, AgentWindows.Config)
    ppo = windows.objective
    assert isinstance(ppo, TritonPPO.Config)
    ppo.value_coefficient = 0.5
    ppo.value_clip_epsilon = 0.2

    transitions = cfg.step.env.num_envs * cfg.step.rollout.horizon
    epochs = 20_000_000_000 // transitions
    cfg.max_steps = cfg.step.train_budget_steps = epochs
    schedule = cfg.step.schedule = ProgressSchedule.Config()
    schedule.curve = PartialConfig(
        one_cycle,
        warmup_fraction=250_000_000 / 20_000_000_000,
        initial=0.0,
    )
    cfg.num_steps_eval = 1_000_000_000 // transitions
    return cfg


def exp003() -> CraftaxTrainLoop:
    """exp002 with frontier practice, stored carry, self-imitation, rewards over 8.

    Our recipe as its best-scoring original run trained it (frontier practice
    with the donor's carry, and reward scaling), restated as one fork:

    - 20% of transitions are practice. The last 64 environments are donors;
      each saves its world, with its live recurrent carry, when its
      achievement return crosses into a new 8-point level. The archive keeps
      32 saves per level, at most 4 from one world, and each rollout restores
      rows from it at step 0, choosing a level by one over its decayed reach
      (:class:`~priml.baselines.craftax.learners.practice.FrontierPractice`).
    - The branches of the first 64 practice rows feed a self-imitation loss,
      weighted 0.01, on one archived branch per epoch, replayed from its stored
      carry (:class:`~priml.baselines.craftax.learners.imitation.BranchImitation`).
    - A training episode ends 10,000 decisions after its last achievement
      reward, save 0.5% of episodes left uncapped.
    - The learner divides rewards by 8 instead of clamping them to [-1, 1], so
      the 8-point necromancer achievements outweigh the 1-point ones.
    - Episodes end when the necromancer falls, in training and evaluation.

    The evaluation plays uncapped and without practice. The reference's
    archive draws interleaved across its buffer threads; the port's are
    serialized, so a config gives one run.

    Hypothesis:
      Practice from self-reached frontier states, with the donor's memory, is
      what lets a scratch run reach and clear the late floors. The six
      changes are one treatment, the reference's own recipe. Its reference
      scored 75.30% at 20B (seed 73, the only seed run), against 66.83% for
      exp002's reference at that seed.

    References:
      https://arxiv.org/abs/1812.03381
        Salimans and Chen 2018. Learning Montezuma's Revenge from a single
        demonstration.
      https://arxiv.org/abs/1806.05635
        Oh et al. 2018. Self-imitation learning.

    Results:
      H200 + 16 CPUs, 20B transitions: 72.48% mean perf (3 seeds: 73, 74, 75).

      Brackets its reference, 19 points above exp002 on the mean of the same
      three seeds. Three seeds at 20B, each on one H200:

        | | seed 73 | seed 74 | seed 75 | mean | transitions/s |
        |---|---|---|---|---|---|
        | exp003 | 67.51% | 77.66% | 72.27% | 72.48% | ~509k |
        | reference runs | 72.54% | 77.04% | 72.71% | 73.25% | - |

      Final evaluations over about 10,000 episodes, ending on the
      necromancer's fall. The reference's mean is over its eight later
      seeds, 73-80 (70.54% to 77.04%); its first run, at seed 73, scored
      75.30%. Seed 74 reaches the last floor in 61% of its evaluation
      episodes, seeds 73 and 75 in 1% and 0%. At 0.99x exp002's speed,
      practice and imitation cost about 1%.

    Returns:
      cfg: Configured CraftaxTrainLoop.

    """
    cfg = exp002()
    cfg.experiment_name = "exp003"
    trackers = cfg.tracker
    assert isinstance(trackers, TrackerList.Config)
    dashboard = unwrap_tracker_config(trackers.trackers["wandb"])
    assert isinstance(dashboard, WandbTracker.Config)
    dashboard.group = "exp003"
    env = cfg.step.env
    env.rules.end_on_boss_defeat = True
    env.stall_cap = StallCap.Config()
    env.practice = FrontierPractice.Config()
    cfg.step.reward_scale = 0.125
    cfg.step.reward_clip = math.inf
    windows = cfg.step.learner
    assert isinstance(windows, AgentWindows.Config)
    windows.auxiliary = BranchImitation.Config()
    return cfg


def exp004() -> CraftaxTrainLoop:
    """exp003's final policy fine-tuned for the boss fight, 3B more transitions.

    The best arm of the original runs' boss-fight screen, restated as one
    fork. It starts from exp003's final checkpoint, the run's fp32 masters,
    with a fresh optimizer, a new world pool and an empty practice archive,
    and trains 3B more transitions (5,722 epochs) with four changes, one
    treatment, the arm's recipe:

    - Training pays +1 per necromancer hit and +1.5 per kill on the final
      floor, on top of the game's reward and before the learner's division
      by 8 (:class:`~priml.baselines.craftax.env.BossFightReward`). The
      game rewards only the first hit and the eighth, its defeat, so a policy
      that has learned the floors hits once and waits out the clock. The
      evaluation pays the game's reward alone.
    - The discount rises from 0.9994 to 0.9998, a horizon of about 5,000
      decisions instead of 1,700, the length of a fight.
    - The rate is a tenth of exp003's peak, held constant: the parent ended
      its schedule at zero, and a decay to zero over the fine-tune scored
      lower.
    - The budget is 3B transitions from the parent's 20B.

    Hypothesis:
      A policy that reaches the final floor but rarely wins learns the fight
      once every hit and every wave's kill pays and its horizon spans a
      fight: it defeats the necromancer in a large share of evaluation
      episodes, which pay no bonus, where exp003 rarely does.

    References:
      https://rekursiv.ai/blog/craftax/
        The boss-fight reward and its screen of six arms.

    Results:
      Reference (the original runs, one H200 each): from exp003's seed-78 run
      at 20B, fine-tuned at seed 73, the arm's final weights score 80.06% perf
      over 10,053 episodes, 4,377 of them (43.5%) defeating the necromancer,
      evaluated on the port in the original runs' numerics (bf16 carry and
      PufferLib's scan). Their own evaluation counted 4,415 defeats in 10,052
      episodes (43.92%), a second fine-tuning seed 43.7%. The other arms:
      with a cosine to zero, 42.8%; with the discount kept as well, 34.9%;
      with the discount kept and the rate constant, 2 in 10,109 after 5B. Not
      yet run on the port.

    Returns:
      cfg: Configured CraftaxTrainLoop.

    """
    cfg = exp003()
    # The parent's last checkpoint, read before this fork renames the run and
    # sets its own budget.
    cfg.step.checkpoint = (
        f"/runs/{cfg.study_name}/{cfg.experiment_name}/checkpoints/"
        f"step_{int(cfg.max_steps):08d}.pt"
    )
    cfg.experiment_name = "exp004"
    trackers = cfg.tracker
    assert isinstance(trackers, TrackerList.Config)
    dashboard = unwrap_tracker_config(trackers.trackers["wandb"])
    assert isinstance(dashboard, WandbTracker.Config)
    dashboard.group = "exp004"
    cfg.step.env.boss_fight_reward = BossFightReward.Config()
    windows = cfg.step.learner
    assert isinstance(windows, AgentWindows.Config)
    ppo = windows.objective
    assert isinstance(ppo, TritonPPO.Config)
    ppo.discount = 0.9998
    muon = cfg.step.optimizer
    assert isinstance(muon, FusedMuon.Config)
    # A tenth of exp003's peak as the arm's command wrote it; peak / 10 lands
    # one double ulp above.
    muon.lr = 0.0001182217733003199
    schedule = cfg.step.schedule
    assert isinstance(schedule, ProgressSchedule.Config)
    schedule.curve = PartialConfig(constant)
    transitions = cfg.step.env.num_envs * cfg.step.rollout.horizon
    cfg.max_steps = cfg.step.train_budget_steps = 3_000_000_000 // transitions
    return cfg


def exp005() -> CraftaxTrainLoop:
    """exp000 at 250M transitions: PufferLib's qualification budget.

    Only the budget changes. The cosine then spans 476 epochs, as it did in
    PufferLib's own 250M run.

    Hypothesis:
      The port reproduces PufferLib's 250M run. Its fp32 masters match
      PufferLib's checkpoints after epochs 200, 400 and 476 (138b2dd1...,
      696a5d5d... and 0265d1d8...) unless an undefined spawn read occurs
      first, about one per 4.1e8 transitions. Its final evaluation matches
      PufferLib's 15.2224% over 10,019 episodes.

    References:
      https://github.com/PufferAI/PufferLib (pin 6ffa5b10, config/craftax.ini)

    Results:
      H200 + 16 CPUs, 250M transitions: 15.2224% perf (seed 73).

      Measured from PufferLib's seed-73 init, which exp000 loaded before it
      drew its own. Bit-identical to PufferLib's run: the
      masters after epochs 200, 400 and 476 equal PufferLib's, and the final
      evaluation is PufferLib's 15.2224%. Its speed is exp000's (the same
      code): 1.46x PufferLib's training throughput.

    Returns:
      cfg: Configured CraftaxTrainLoop.

    """
    cfg = exp000()
    cfg.experiment_name = "exp005"
    epochs = 250_000_000 // (cfg.step.env.num_envs * cfg.step.rollout.horizon)
    cfg.max_steps = cfg.step.train_budget_steps = epochs
    return cfg


def exp006() -> CraftaxTrainLoop:
    """exp002 at width 2,048, on 1,024 environments of 512 steps.

    The recipe of the original runs' capacity arm, restated as one fork:
    exp002's policy twice as wide
    -- ``proj_in`` 1,635 -> 2,048, four MinGRU layers of 2,048, each injection
    2,048 -> 128 -> 2,048 -- learning from windows twice as long, its rollout
    regrouped from 2,048 environments of 256 steps into 1,024 of 512. The
    rollout keeps its 524,288 transitions, so the minibatches of 32,768 (64
    environments' whole windows), the 18 updates per rollout, the budget, the
    schedule and the evaluation are exp002's.

    Hypothesis:
      Width and window length are one treatment, the reference's own arm. At
      20B it scores in its reference's range, above exp002: the reference's
      seed-73 run read 16.10% at 250M, 27.46% at 1B and 65.10% at 20B.

    References:
      exp002's.

    Results:
      Reference (the original runs, seed 73, 20B): 65.10% perf. Not yet run
      on the port.

    Returns:
      cfg: Configured CraftaxTrainLoop.

    """
    cfg = exp002()
    cfg.experiment_name = "exp006"
    trackers = cfg.tracker
    assert isinstance(trackers, TrackerList.Config)
    dashboard = unwrap_tracker_config(trackers.trackers["wandb"])
    assert isinstance(dashboard, WandbTracker.Config)
    dashboard.group = "exp006"
    model = cfg.step.model
    assert isinstance(model, MinGRUPolicy.Config)
    model.channels_hidden = 2_048
    # 1,024 x 512 = 2,048 x 256 transitions: exp002's epoch counts stand.
    cfg.step.env.num_envs = 1_024
    cfg.step.rollout.horizon = 512
    return cfg


def exp007() -> CraftaxTrainLoop:
    """exp006 with a ConvNeXt trunk in place of the board's residual blocks.

    The original runs' ConvNeXt-trunk encoder arm, restated as one fork. The
    board's two 16-channel depthwise residual blocks
    become one :class:`~priml.baselines.craftax.policies.encoder.ConvNextTrunk`: a
    1 x 1 lift to 64 channels; two ConvNeXt blocks, each a 5 x 5 depthwise
    window, a per-cell LayerNorm, a 256-wide MLP with exact GELU and a layer
    scale starting at 1e-6; and a 1 x 1 projection back onto the board, zero
    at init. The FiLM branch, the multiscale mix and everything after them are
    exp006's, and both start as the multi-hot embedding.

    What differs from the reference, besides the random streams, is exp002's
    list; its D2 now covers the trunk too. The reference wraps the trunk in
    gradient-scale round trips at scale 1, ``t + (b - t)`` and ``x +
    scale(proj_out(b))``: identities in real arithmetic that round the board
    in bf16. The parity suite holds the policy to the
    reference's bytes with D1 and D2 applied, and with D1 alone once its trunk
    reproduces the round trips.

    Hypothesis:
      A wider trunk of ConvNeXt blocks reads the board better early. Its
      reference's seed-73 run read 18.99% at 250M and 38.27% at 1B, against
      exp006's reference's 16.10% and 27.46%, and 62.01% at 20B against its
      65.10%.

    References:
      https://arxiv.org/abs/2201.03545
        Liu et al. 2022. A ConvNet for the 2020s.

    Results:
      Reference (the original runs, seed 73, 20B): 62.01% perf. Not yet run
      on the port.

    Returns:
      cfg: Configured CraftaxTrainLoop.

    """
    cfg = exp006()
    cfg.experiment_name = "exp007"
    trackers = cfg.tracker
    assert isinstance(trackers, TrackerList.Config)
    dashboard = unwrap_tracker_config(trackers.trackers["wandb"])
    assert isinstance(dashboard, WandbTracker.Config)
    dashboard.group = "exp007"
    model = cfg.step.model
    assert isinstance(model, MinGRUPolicy.Config)
    encoder = model.embedding
    assert isinstance(encoder, BoardEncoder.Config)
    encoder.board.block = ConvNextTrunk.Config()
    encoder.board.num_layers = 1
    return cfg


def exp008() -> CraftaxTrainLoop:
    """exp003 starting each episode in a newly generated world, evaluated on its pool.

    The world-source arm of the recipe runs' original-Craftax ablations,
    restated as one fork: every reset generates a new world from the
    environment's own
    stream, as original Craftax does (D6), so training never revisits a
    world; practice still restores the states its donors saved. The
    evaluation is exp003's, uncapped and without practice on the 8,192-world
    pool, so the two score on the same worlds.

    Hypothesis:
      Worlds never seen twice teach what holds across worlds rather than
      what holds in the pool's, and exp008 scores above exp003 on exp003's
      own pool. Its reference, scored on the pool at 6B against seed-matched
      controls: 60.80% and 60.51% against 58.42% and 56.28% (seeds 73 and 74,
      +3.3 points on the mean), reaching the Ice floor in 21% and 27% of
      episodes against under 0.4%. The lead is not settled: seed 73's
      training curve fell behind its control's at 7.5B, and at 20B the arm
      read 73.63% and 74.54% on fresh worlds against its controls' 74.62% and
      71.22% on the pool, two populations. Generating a world at every reset
      costs CPU time: one live sample of the reference read 380k
      transitions/s against its control's 519k.

    References:
      https://arxiv.org/abs/2402.16801
        Matthews et al. 2024. Craftax: a lightning-fast benchmark for
        open-ended reinforcement learning (a new world at every reset).

    Results:
      Reference (the original runs, seeds 73 and 74, scored on the pool at
      6B): 60.80% and 60.51% perf, against their controls' 58.42% and
      56.28%. At 20B, scored on fresh worlds: 73.63% and 74.54%. Not yet
      run on the port.

    Returns:
      cfg: Configured CraftaxTrainLoop.

    """
    cfg = exp003()
    cfg.experiment_name = "exp008"
    trackers = cfg.tracker
    assert isinstance(trackers, TrackerList.Config)
    dashboard = unwrap_tracker_config(trackers.trackers["wandb"])
    assert isinstance(dashboard, WandbTracker.Config)
    dashboard.group = "exp008"
    # exp003's evaluation env, taken before the restart changes: left unset, the
    # evaluation copies training's and would score on fresh worlds, not exp003's.
    evaluation = cfg.step.evaluation.env = cfg.step.env.copy_tree()
    evaluation.stall_cap = None
    evaluation.practice = None
    cfg.step.env.restart = FreshWorlds.Config()
    return cfg


def exp009() -> CraftaxTrainLoop:
    """exp000 on original Craftax as published (``docs/differences.md``).

    Every environment option that PufferLib changed goes back to original
    Craftax: its reward, 0.1 per health point changed in place of the armour
    term and the -1 on death (D1); an episode that ends when the necromancer
    falls (D2); one tick per decision while sleeping or resting (D4); no
    illegal-action mask (D5); a fresh world at every reset instead of the
    8,192-world pool (D6); and its 8,268-float symbolic observation (D8).
    That observation is already a feature vector, so ``proj_in`` reads it in
    place of the embedding bag. The rest of the learner, its recipe and the
    budget are exp000's, and the evaluation plays by the same rules.

    Hypothesis:
      The PufferLib recipe trains on original Craftax too, scoring below
      exp000: the mask and the collapsed sleep make exploration cheaper, and
      fresh worlds make every episode new. The six options are one treatment,
      not six: the question is the benchmark as published, and no subset of
      them is that benchmark. Each option's own effect on the environment is
      a unit test of ``game/step_test.py``.

    References:
      https://github.com/MichaelTMatthews/Craftax (Craftax-Symbolic-v1)

    Results:
      H200 + 16 CPUs, 3.49B transitions: 12.48% perf (seed 73).

      Confirmed: perf 12.48% over 10,008 episodes against exp000's 47.20%
      (on the final code). Training took 3,842 s for
      3.49B transitions (0.91M transitions/s; fresh worlds, one tick per
      decision and the 8,268-float observation cost exp000's pool and packed
      view), and the eval 308 s. A run on earlier code, before the init fix,
      scored 15.53%, inside the same-seed spread.

    Returns:
      cfg: Configured CraftaxTrainLoop.

    """
    cfg = exp000()
    cfg.experiment_name = "exp009"
    rules = cfg.step.env.rules
    rules.original_reward = True
    rules.end_on_boss_defeat = True
    rules.collapse_sleep = False
    rules.action_mask = False
    rules.symbolic_observation = True
    cfg.step.env.restart = FreshWorlds.Config()
    model = cfg.step.model
    assert isinstance(model, MinGRUPolicy.Config)
    model.embedding = DenseObservation.Config()
    return cfg


def exp010() -> CraftaxTrainLoop:
    """exp009 learning by Craftax_Baselines' 1B PPO recipe, seed 42.

    The environment and its evaluation are exp009's: original Craftax's rules,
    fresh worlds and 8,268-float symbolic observation. The learner is the
    official baseline's (``ppo.py`` and ``models/actor_critic.py`` at commit
    ``7ce36fa``, Craftax-Symbolic-v1): separate actor and critic MLPs of three
    tanh layers of 512 with orthogonal init, in fp32 with TF32 matmuls (JAX's
    default on this GPU, measured); 1,024 environments x 64 steps, each rollout
    collected with the current weights, then learned from (one slot);
    generalized advantages over the whole rollout, gamma 0.99 and lambda 0.8,
    bootstrapped from the observation after it, rewards unclipped; four passes
    of eight shuffled minibatches of 8,192 transitions, each standardizing its
    advantages, with clip 0.2 on the ratio and the value, value weight 0.5 and
    entropy weight 0.01; Adam at 2e-4 (eps 1e-5) annealed linearly to zero
    over the run, behind a global gradient-norm clip of 1.0; 15,258 updates,
    999,948,288 transitions. The recipe's parts are one treatment: the
    question is whether the port's environment reproduces the baseline's
    score under the baseline's own learner.

    What still differs from the reference, besides the random streams (init,
    sampling, shuffles and worlds draw from other generators):

    - The environment is the port's Numba game on CPU threads, not JAX Craftax
      1.6.1 on the GPU; ``docs/differences.md`` says where and how it was
      checked against JAX.
    - An ended episode restarts in a newly generated world. The reference's
      optimistic reset draws 64 fresh worlds per step for its 1,024
      environments, so when more than 64 episodes end in one step some share
      a world.
    - Actions are drawn by the port's inverse-CDF Philox sampler, not by
      distrax's categorical sampler: the same distribution, other draws.
    - The global clip divides by ``norm + 1e-6`` (torch) instead of ``norm``
      (optax), and the advantages are standardized by ``std + 1.19e-7`` (fp32's
      epsilon, ``clipped_policy_loss``) instead of ``std + 1e-8``.
    - The score is PufferLib's evaluation: 10,000 fresh episodes after
      training (``metric.CraftaxScore``), where the reference reports the
      episodes that finish during its training updates.

    Hypothesis:
      With the baseline's learner the port's environment scores as the
      baseline does: within seed noise of the official JAX runs' seeds 42-44
      on the same hardware, and near the 11.867% that the torch port of JAX
      Craftax this baseline replaced measured with the recipe.

    References:
      https://github.com/MichaelTMatthews/Craftax_Baselines (commit 7ce36fa)
      https://arxiv.org/abs/2402.16801
        Matthews et al. 2024. Craftax: a lightning-fast benchmark for
        open-ended reinforcement learning.

    Results:
      H200 + 16 CPUs, 1B transitions: 12.37% mean perf (3 seeds: 42, 43, 44).

      Matches the official baseline's reward at 2.1x its speed. Both ran 1B
      transitions on one H200 of the same kind, seeds 42, 43 and 44:

        | | seed 42 | seed 43 | seed 44 | mean | transitions/s |
        |---|---|---|---|---|---|
        | exp010 | 12.79% | 11.99% | 12.34% | 12.37% | ~332k |
        | Craftax_Baselines (JAX) | 11.74% | 12.58% | 12.16% | 12.16% | ~155k |

      exp010's figures are its final evaluation (normalized return over about
      10,000 episodes; training about 50 minutes per seed). The official run's are its training return over the last 20
      updates, which ``ppo.py`` reports, with its speed end to end including
      compilation (about 1.9 hours per seed). The paper
      reports about 11.9%.

    Returns:
      cfg: Configured CraftaxTrainLoop.

    """
    cfg = exp009()
    cfg.experiment_name = "exp010"
    cfg.seed = 42
    runtime = cfg.runtime
    assert isinstance(runtime, SingleProcess.Config)
    runtime.float32_matmul_precision = "high"

    model = cfg.step.model = ActorCritic.Config()
    # The baseline stores its rollout in its policy's fp32, where exp000 stores
    # PufferLib's bf16.
    cfg.step.rollout.dtype = model.dtype
    cfg.step.optimizer = PartialConfig(
        ClippedAdam,
        lr=2e-4,
        eps=1e-5,
        max_grad_norm=1.0,
    )
    schedule = cfg.step.schedule = ProgressSchedule.Config()
    schedule.curve = PartialConfig(linear)
    # Its defaults are the reference's four passes of eight minibatches.
    learner = cfg.step.learner = ShuffledTransitions.Config()
    learner.seed = 42
    cfg.step.reward_clip = math.inf

    cfg.step.env.num_envs = 1_024
    # Eight buffers of 128: the synchronous rollout is the epoch's critical
    # path, and more buffers overlap its policy steps with the game's (H200,
    # 16 CPUs: 316k transitions/s against 262k with four).
    cfg.step.env.num_buffers = 8
    cfg.step.env.seed = 42
    sampler = cfg.step.sampler
    assert isinstance(sampler, PhiloxSampler.Config)
    sampler.seed = 42
    cfg.step.rollout.horizon = 64
    cfg.step.rollout.num_slots = 1
    cfg.step.rollout.bootstrap = True

    # The reference counts updates as ``1e9 // 64 // 1024``, whole rollouts.
    epochs = 1_000_000_000 // (cfg.step.env.num_envs * cfg.step.rollout.horizon)
    cfg.max_steps = cfg.step.train_budget_steps = epochs
    return cfg


def exp011() -> CraftaxTrainLoop:
    """exp010 at Craftax_Baselines' 1M-interaction geometry: 256 x 16 steps, lr 3e-4.

    The published one-million-interaction reproduction, exp000 of the earlier
    torch port of Craftax_Baselines, restated as one fork: 256 environments x
    16 steps in place of 1,024 x 64, Adam at 3e-4 in place of 2e-4, and 244
    updates, 999,424 transitions. The geometry, the rate and the budget are
    one treatment, the inverse of the step from that port's exp000 to its 1B
    recipe. The rest is exp010's: the actor-critic, four passes of eight
    shuffled minibatches (512 transitions each here), every coefficient, the
    linear anneal over the run, seed 42 for every stream, and the evaluation.

    Not carried over from the earlier port's exp000, each because this
    package's environment and score are exp010's:

    - Its optimistic reset, 16 environments to a freshly generated world: an
      ended episode restarts in a world of its own.
    - Its score, 64 environments x 10,000 steps from seed 42, averaging the
      episodes that finished, their return counting original Craftax's 0.1
      per point of health. The score here is exp010's: 10,000 episodes after
      training, scored by their achievement return (``metric.CraftaxScore``).

    What differs from the reference itself is exp010's list, unchanged.

    Hypothesis:
      A feed-forward actor-critic trained with clipped PPO reaches roughly
      2.2% at one million interactions, as the earlier JAX port of this
      recipe measured (2.212% at 999,424 steps): the bar any added mechanism
      must clear to earn its complexity.

    References:
      https://github.com/MichaelTMatthews/Craftax_Baselines (commit 7ce36fa)
      https://arxiv.org/abs/2402.16801
        Matthews et al. 2024. Craftax: a lightning-fast benchmark for
        open-ended reinforcement learning.

    Results:
      H200 + 16 CPUs, 999,424 transitions: 2.45% perf (seed 42).

      Confirmed at seed 42: perf 2.45% over 10,011 episodes (achievement rate
      8.26%), against the JAX port's 2.212%, whose normalized return also
      counted the 0.1 per point of health. Training took 14 s for 999,424
      transitions on one H200 and 16 CPUs (about 72k transitions/s: 256
      environments leave the GPU mostly idle), and the evaluation 12 s.

    Returns:
      cfg: Configured CraftaxTrainLoop.

    """
    cfg = exp010()
    cfg.experiment_name = "exp011"
    cfg.step.env.num_envs = 256
    cfg.step.rollout.horizon = 16
    adam = cfg.step.optimizer
    assert isinstance(adam, PartialConfig)
    adam.lr = 3e-4
    # The reference counts updates as ``1e6 // 16 // 256``, whole rollouts.
    epochs = 1_000_000 // (cfg.step.env.num_envs * cfg.step.rollout.horizon)
    cfg.max_steps = cfg.step.train_budget_steps = epochs
    return cfg


def exp012() -> CraftaxTrainLoop:
    """exp010 with Craftax_Baselines' PPO-RNN policy: a reset-aware GRU, whole trajectories.

    The recipe of exp002 of the earlier torch port of Craftax_Baselines, which
    this package replaced: ``ppo_rnn.py``, forked from exp010 as that exp002
    forked its 1B PPO recipe. Two changes, inseparable, since a recurrent
    policy cannot learn from transitions shuffled apart:

    - The policy (:class:`~priml.baselines.craftax.policies.rnn.ActorCriticRNN`):
      a 512-wide ReLU embedding of the observation feeds one GRU cell of 512,
      whose state is zeroed wherever an episode starts; separate actor and
      critic heads of two ReLU layers of 512 read it. The projections are
      orthogonal, sqrt(2) in the embedding and hidden layers, 0.01 and 1 at
      the policy's and the value's outputs; the cell keeps torch's init.
    - The minibatches
      (:class:`~priml.baselines.craftax.learners.rnn_update.ShuffledTrajectories`):
      each of the four passes shuffles the 1,024 agents into eight minibatches
      of 128 whole 64-step trajectories, and replays the GRU over each from
      the carry its agents started the rollout with.

    Everything else is exp010's: the environment and its 1,024 x 64 rollout;
    the advantages over the whole rollout, bootstrapped from the value of the
    observation after it, read from the state the last step left, reset where
    that step ended an episode; the objective and its coefficients; Adam at
    2e-4 annealed linearly to zero over 15,258 updates behind a global clip of
    1.0; TF32 matmuls; seed 42 for every stream; and the evaluation.

    Not carried over from the earlier port's exp002, besides exp010's list:

    - Its optimistic reset, 64 fresh worlds a step shared by the episodes that
      end in it: the port's environment generates a new world at every reset
      (``FreshWorlds``), as exp010 plays.
    - Its full-precision fp32 matmuls: exp010's TF32, the reference's JAX
      default on this GPU.
    - Its evaluation, 64 environments for 10,000 steps: exp010's 10,000 fresh
      episodes (``metric.CraftaxScore``).
    - Its shuffles from torch's generator and its ``multinomial`` actions:
      exp010's NumPy shuffles keyed by the seed and the epoch, and its Philox
      sampler. The same distributions, other draws.

    Hypothesis:
      Craftax's achievements are sequential, so a policy that remembers
      anything should beat one that remembers nothing, and a GRU is the
      cheapest memory: its state is one vector at step 1 and step 10,000.
      exp012 scores above exp010's 12.37% (mean of seeds 42-44), near the
      16.099% the JAX port of this recipe measured at seed 42.

    References:
      https://github.com/MichaelTMatthews/Craftax_Baselines (``ppo_rnn.py``)
      https://arxiv.org/abs/1406.1078
        Cho et al. 2014. Learning phrase representations using RNN
        encoder-decoder for statistical machine translation.

    Results:
      H200 + 16 CPUs, 1B transitions: 16.69% perf (seed 42).

      Confirmed: 16.69% normalized return at seed 42 (final evaluation over
      10,022 episodes; achievement rate 36.93%, the dungeon reached in 90.6%
      of them), against the JAX port's 16.099% at that seed and exp010's
      12.79% (12.37% the mean of seeds 42-44): the memory is worth about four
      points. Training took 5,930 s for 999,948,288 transitions on one H200
      (~169k transitions/s, half exp010's: the learner replaying the GRU step
      by step takes 0.26 s of each 0.37 s epoch), and the evaluation 9 s.

    Returns:
      cfg: Configured CraftaxTrainLoop.

    """
    cfg = exp010()
    cfg.experiment_name = "exp012"
    cfg.step.model = ActorCriticRNN.Config()
    learner = cfg.step.learner
    assert isinstance(learner, ShuffledTransitions.Config)
    # ShuffledTrajectories adds no field, so it takes exp010's as they are.
    cfg.step.learner = ShuffledTrajectories.Config().update(learner)
    return cfg


def exp013() -> CraftaxPQNTrainLoop:
    """exp010 learned by PQN: an LSTM Q-learner, no replay buffer, no target network.

    The setup is exp010's: original Craftax's rules, fresh worlds and 8,268-float
    symbolic observation, 1,024 environments in eight buffers, seed 42 for
    every stream, and the 10,000-episode evaluation after training, here played
    by the greedy action. The learner is the PQN recipe of the earlier torch
    port of Craftax_Baselines, its exp003, which this one replaces
    (``pqn_train_step.CraftaxPQNTrainStep``'s defaults), with its batch
    renormalization and loss as purejaxql has them: batch renormalization of
    the observation, a biased 512-wide projection, LayerNorm and ReLU, an
    LSTM of 512 over that encoding and the previous action one-hot, and a
    biased 43-way value head, in fp32; 128-step rollouts collected
    epsilon-greedily with the current weights, the rate falling linearly from
    1 to 0.005 over the first tenth of the updates; Q(lambda) targets, gamma
    0.99 and lambda 0.5, from the collecting weights' own values, built once
    per rollout and bootstrapped from the observation after it; four passes of
    four minibatches of 256 whole trajectories, each half the mean squared
    error, its window renormalized as one batch, and RAdam at 3e-4 annealed
    linearly to zero over the run behind a global gradient-norm clip of 0.5;
    7,629 updates, 999,948,288 transitions. The recipe's parts are one
    treatment: the question is the recipe's score on exp010's setup.

    Not carried over from the earlier port's exp003:

    - Batch renormalization takes a minibatch's whole window as one batch, and
      the loss is half the mean squared error, as purejaxql has them. That port
      renormalized each step's environments apart, moving the running
      statistics 128 times a minibatch, and regressed the whole squared
      error.
    - The environment is the port's Numba game on original Craftax's rules
      (exp009), not that port's torch reimplementation of JAX Craftax; an
      ended episode restarts in a newly generated world, where its optimistic
      reset drew 64 fresh worlds per step.
    - Each action is one draw of the port's Philox inverse-CDF sampler from
      the epsilon-greedy distribution, not a coin and a uniform action from
      torch's generator: the same distribution, other draws. The trajectory
      shuffles come from NumPy's seed sequence of seed 42 and the update.
    - The targets read every action's value rescored after the rollout with
      the weights and running normalization that collected it, not the
      actor's own outputs: the same values but for the products' batch
      shapes.
    - Matmuls run in TF32, exp010's precision, where that port ran full fp32.
    - The score is exp010's: 10,000 fresh episodes after training
      (``metric.CraftaxScore``), where that port scored 64 environments for
      10,000 steps.

    Where the recipe differs from purejaxql's, that port's choices are kept:

    - The targets are built once per rollout from the collecting weights'
      values under the running normalization; purejaxql rebuilds them in every
      minibatch from the training network's own values, stopped.
    - A rollout of 128 steps regresses 128 targets, the last bootstrapped from
      the observation after it; purejaxql regresses the first 127 and
      bootstraps from the last.
    - Batch renormalization's mean correction scales by ``sqrt(var + eps)``
      (priml's ``BatchRenorm``), purejaxql's by ``sqrt(var)``.
    - The rate anneals once per update, to zero; purejaxql's every gradient
      step, to 1e-20. The clip divides by ``norm + 1e-6`` (torch), not by
      ``norm`` (optax).
    - The weights start from torch's defaults, the LayerNorm's epsilon is
      1e-5; purejaxql's are flax's (LeCun normal, an orthogonal recurrent
      kernel, zero biases) and 1e-6.
    - Training starts at once; purejaxql first plays 128 random steps.

    As in purejaxql, the previous action carries over an episode's end, and
    it is action 0 before an environment's first step.

    Hypothesis:
      At a thousand parallel workers the buffer is redundant, because the batch
      is already decorrelated; and batch renormalization plus a multi-step
      Q(lambda) target keeps the regression stable enough to drop the target
      network. If that holds, a value method trained on exp010's budget beats
      exp010's 12.37% and reaches about the 16.0% normalized return the
      reference reports.

    References:
      https://arxiv.org/abs/2407.04811
        Gallici et al. 2024. Simplifying deep temporal difference learning.
      https://github.com/mttga/purejaxql
        The reference, ``pqn_rnn_craftax.py`` and its Craftax config.

    Results:
      H200 + 16 CPUs, 1B transitions: 16.32% perf (seed 42).

      Confirmed at seed 42: 16.32% normalized return (final evaluation over
      10,118 episodes; achievement rate 34.93%, the dungeon reached in 89.4%
      of them), above exp010's 12.79% at that seed and about the reference's
      reported 16.0%. Training took 10,740 s for 999,948,288 transitions on
      one H200 (~93k transitions/s: the learner takes about 1.1 s of each
      1.4 s update), and the evaluation 9 s.

      Before the BatchRenorm and loss fixes: 7.15% at seed 42.

    Returns:
      cfg: Configured CraftaxPQNTrainLoop.

    """
    parent = exp010()
    cfg = CraftaxPQNTrainLoop()
    # The run around the learner is exp010's: its seed, precision, tracker,
    # checkpoints, evaluation cadence and score.
    cfg.update(parent, step=cfg.step)
    cfg.experiment_name = "exp013"
    cfg.step.env = parent.step.env
    sampler = cfg.step.sampler
    assert isinstance(sampler, EpsilonGreedy.Config)
    sampler.sampler = parent.step.sampler
    cfg.step.seed = 42
    # The reference counts updates as ``1e9 // 128 // 1024``, whole rollouts.
    updates = 1_000_000_000 // (cfg.step.env.num_envs * cfg.step.rollout.horizon)
    cfg.max_steps = cfg.step.train_budget_steps = updates
    return cfg


def exp014() -> CraftaxTrainLoop:
    """exp010's 1B geometry at 100M interactions, a screening budget.

    exp011 of the earlier torch port of Craftax_Baselines, restated as one
    fork: only the budget moves, 1,525 updates of 1,024 x 64, 99,942,400
    transitions, and the learning-rate horizon with it. The anneal is
    budget-relative, so a run stopped early on exp010's horizon would end
    while the rate was still high and measure something that is not a
    smaller version of its parent.

    Not carried over from the earlier port's exp011, each because this
    package's environment and score are exp010's:

    - Its parent's optimistic reset, which it carried: an ended episode
      restarts in a world of its own.
    - Its score: 64 environments x 10,000 steps, as our exp011 states.
    - The scanned evaluator and adaptive reset pool of its JAX counterpart,
      which that port had already dropped: both worked around XLA's static
      shapes.

    Hypothesis:
      The 1B geometry keeps its learning signal at 100M, a screen short
      enough to run several treatments at once. Its JAX counterpart scored
      8.976% here against its parent's 11.867%, so the screen ranks recipes
      without reproducing their scores.

    References:
      https://github.com/MichaelTMatthews/Craftax_Baselines (commit 7ce36fa)
      https://arxiv.org/abs/2402.16801
        Matthews et al. 2024. Craftax: a lightning-fast benchmark for
        open-ended reinforcement learning.

    Results:
      H200 + 16 CPUs, 100M transitions: 9.56% perf (seed 42).

      Confirmed at seed 42: perf 9.56% over 10,028 episodes (achievement rate
      28.07%; 35% of episodes reach the dungeon), against its JAX
      counterpart's 8.976% and below exp010's 12.37% at 1B, the gap the
      screen predicts. Training took 306 s for 99,942,400 transitions on one
      H200 and 16 CPUs (about 327k transitions/s, exp010's speed), and the
      evaluation 11 s.

    Returns:
      cfg: Configured CraftaxTrainLoop.

    """
    cfg = exp010()
    cfg.experiment_name = "exp014"
    epochs = 100_000_000 // (cfg.step.env.num_envs * cfg.step.rollout.horizon)
    cfg.max_steps = cfg.step.train_budget_steps = epochs
    return cfg


def exp015() -> CraftaxTrainLoop:
    """exp010 with a policy that remembers: a gated Transformer-XL, at 1B.

    transformerXL_PPO_JAX's recipe on exp010's setup, restated as one fork.
    The policy is a gated Transformer-XL
    (:class:`~priml.baselines.craftax.policies.gtrxl.GTrXLPolicy`): the
    observation projected to 256, two blocks of relative attention (eight
    heads of 32) and an MLP, each behind a GRU gate initialized closed (bias
    2), over a memory of the last 128 steps' layer inputs, each step
    addressable on its own; then exp010's two towers, two ReLU layers of 256
    each. The learner learns from whole trajectories
    (:class:`~priml.baselines.craftax.learners.gtrxl_train_step.TrajectoryWindows`):
    four passes of eight shuffled groups of 128 environments, each trajectory
    cut into two windows of 64 steps that start from the memory the actor
    held there. Rollouts grow to 128 steps, so the memory has a rollout to
    hold; the discount rises to 0.999 and the entropy weight falls to 0.002,
    because a policy that can remember is worth pointing at rewards further
    away. Adam, its rate and annealing, the clips, the value weight and the
    1,024 environments are exp010's; 7,629 updates of 1,024 x 128 steps,
    999,948,288 transitions, exp010's budget.

    Not carried over from the earlier torch port's exp013, besides the random
    streams:

    - exp010's environment and evaluation: an ended episode restarts in a
      newly generated world, where that port's exp013 dealt one world to 16 workers
      (optimistic reset); the score is PufferLib's 10,000 fresh episodes, where
      that port's exp013 scored the episodes 64 workers finished in 10,000 steps.
    - Each window's starting memory is replayed, not stored: the rollout keeps
      the memory only at its first step, so before the update the learner
      reruns the policy over the first 64 steps with the collection weights.
      The values are the actor's, up to the rounding of a differently shaped
      GEMM; the cost is one more policy pass over half the rollout per update.
    - Matmuls run in TF32 (exp010's ``float32_matmul_precision``, JAX's
      default on the reference's GPU), where that port's exp013 ran full fp32.
    - exp010's sampler and clip arithmetic, as its docstring lists them.

    Hypothesis:
      Craftax's achievements are sequential -- wood, then a table, then a
      pickaxe, then stone -- and a memoryless policy has to rediscover its
      progress from the view in front of it. Memory should be worth more here
      than architecture usually is: roughly the 18.159% that the JAX port of
      this recipe measured at seed 42 (the reference reports 18.3%), against
      exp010's 12.37%.

    References:
      https://github.com/Reytuag/transformerXL_PPO_JAX
      https://arxiv.org/abs/1910.06764
        Parisotto et al. 2020. Stabilizing transformers for reinforcement
        learning.

    Results:
      H200 + 16 CPUs, 1B transitions: 17.44% perf (seed 42).

      17.44% normalized return at seed 42 (final evaluation over 10,042
      episodes; achievements 39.5%), against the JAX port's 18.159% at the
      same seed and exp010's 12.37%: memory is worth about 5 points here.
      Trained in 2.6 hours on one H200 at ~107k transitions/s.

    Returns:
      cfg: Configured CraftaxTrainLoop.

    """
    cfg = exp010()
    cfg.experiment_name = "exp015"
    cfg.step.model = GTrXLPolicy.Config()
    # Its defaults are the reference's passes, windows and coefficients.
    learner = cfg.step.learner = TrajectoryWindows.Config()
    learner.seed = 42
    cfg.step.rollout.horizon = 128
    epochs = 1_000_000_000 // (cfg.step.env.num_envs * cfg.step.rollout.horizon)
    cfg.max_steps = cfg.step.train_budget_steps = epochs
    return cfg


def exp_smoke() -> CraftaxTrainLoop:
    """exp000 at minimum size, for checking a machine end to end.

    Not a result: it answers whether the loop runs, so every axis that costs
    time without bearing on that answer is cut, down to a network narrow
    enough to train in seconds.

    Returns:
      cfg: A four-epoch CraftaxTrainLoop.

    """
    cfg = exp000()
    cfg.experiment_name = "exp_smoke"
    model = cfg.step.model
    assert isinstance(model, MinGRUPolicy.Config)
    model.channels_hidden = 8
    model.num_layers = 1
    embedding = model.embedding
    assert isinstance(embedding, MultiHotEmbedding.Config)
    embedding.channels_out = 2
    cfg.step.env.num_envs = 16
    cfg.step.env.num_buffers = 2
    pool = cfg.step.env.restart
    assert isinstance(pool, WorldPool.Config)
    pool.num_worlds = 64
    cfg.step.rollout.horizon = 16
    windows = cfg.step.learner
    assert isinstance(windows, AgentWindows.Config)
    windows.minibatch_size = 128
    cfg.max_steps = cfg.step.train_budget_steps = 4
    cfg.step.evaluation.num_episodes = 1
    cfg.tracker = FileTracker.Config()
    return cfg


def _dashboard(tracker: object) -> WandbTracker.Config | None:
    """Return the W&B tracker in a tree of tracker lists and async wrappers, if any."""
    if isinstance(tracker, WandbTracker.Config):
        return tracker
    if isinstance(tracker, AsyncTracker.Config):
        return _dashboard(tracker.tracker)
    if isinstance(tracker, TrackerList.Config):
        for member in tracker.trackers.values():
            found = _dashboard(member)
            if found is not None:
                return found
    return None
