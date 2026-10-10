r"""Craftax experiments: PufferLib's trainer, ported bit for bit.

``exp000`` is PufferLib's own Craftax recipe (``config/craftax.ini`` at pin
``6ffa5b10``) at its one-GPU budget, its policy drawing its own init from
PufferLib's distributions under PufferLib's seed, 73. The class defaults are
PufferLib's ``config/default.ini``; exp000 states what ``craftax.ini``
changes. Every later experiment forks a named parent and applies ONE change.
exp000-exp008 are baselines, published recipes restated here; exp101 on are
the port's own.

    exp000     PufferLib's recipe, 3,492,806,656 transitions, 6,662 epochs
      +-- exp001     the same recipe at 250M transitions, the qualification run
      +-- exp002     the same recipe on original Craftax as published
      |     +-- exp003     Craftax_Baselines' 1B PPO recipe on that setup
      |           +-- exp004     its 1M-interaction geometry, 256 envs x 16 steps
      |           +-- exp005     PPO-RNN: a reset-aware GRU, whole trajectories
      |           +-- exp006     PQN: Q-learning with an LSTM in place of PPO
      |           +-- exp007     its 1B geometry at 100M, a screening budget
      |           +-- exp008     GTrXL: PPO with gated Transformer-XL memory
      +-- exp101     the same recipe on the port's defaults, not PufferLib's bits
      |     +-- exp102     board encoder, injection, feasibility loss, 20B
      |           +-- exp103     frontier practice, self-imitation, rewards / 8
      |           |     +-- exp109     a newly generated world at every reset
      |           |     +-- exp114     fine-tuned for the boss fight, 3B more
      |           |     +-- exp110     an early world model, the sole encoder
      |           |           +-- exp111     a mature world model
      |           |           |     +-- exp113     trained with the policy
      |           |           +-- exp112     trained with the policy
      |           |           |     +-- exp116     stopped after 1B
      |           |           +-- exp115     stopped after 1B
      |           +-- exp104     width 2,048 on 1,024 environments x 512 steps
      |                 +-- exp105     a ConvNeXt trunk on the board
      |                       +-- exp106     250M, inside the 20B warmup
      |                             +-- exp107     a frozen world model's feature
      |                                   +-- exp108     its random-init weights
      +-- exp_smoke  the same recipe at minimum size, to check a machine

Not yet reproducible: exp110 to exp113, exp115 and exp116 read a world model
the original runs fitted, which no experiment here trains. Their factories
raise ``NotImplementedError``; each docstring keeps the record and names the
missing producer (TODO).

Every random stream is seeded in the config, so a config replays its run: the
environments' ``rand_r`` (``env.seed``), the action streams (``sampler.seed``),
and torch's generator (``seed``), which draws the policy's init. exp000 sets
PufferLib's seed, 73, and every fork keeps it, except exp003 and its forks,
which take their reference's seed, 42, for every stream they draw.

Launch::

    uv --quiet run --frozen python -m priml priml.baselines.craftax.experiments.exp000

References:
    https://github.com/PufferAI/PufferLib
        Suarez. PufferLib (MIT license), ``config/craftax.ini``, pin ``6ffa5b10``.

"""

from __future__ import annotations

from dataclasses import field
from pathlib import Path
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
from priml.baselines.craftax.world_model.experiments import (
    exp001 as world_model_exp001,
)
from priml.baselines.craftax.world_model.feature import (
    Flash4CacheAttention,
    InitialWeights,
    TrainedWeights,
    WorldModelFeature,
)
from priml.loss.policy_gradient_kernel import TritonPPO
from priml.math.schedules import constant, linear, one_cycle, warmup
from priml.model.embedding import MultiHotEmbedding
from priml.model.linear import Linear
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

    import torch
else:
    from wrapt import lazy_import

    # ~1050 ms; only exp102's zero init and the features' compiles read them.
    nn = lazy_import("torch.nn")
    torch = lazy_import("torch")


class CraftaxTrainLoop(
    Makes["TrainLoop"],
    TrainLoop.Config[CraftaxTrainStep.Config, CraftaxRollouts.Config],
):
    """A training loop with the Craftax step, the epoch cadence and checkpoints in place.

    A checkpoint holds the whole pipeline, rollout slot and environments
    included, so a resumed run continues as if never stopped: about 1.3 GB for
    exp000 and 9.1 GB for exp002's symbolic observations. One is written every
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
    cfg.experiment_name = "exp001"
    epochs = 250_000_000 // (cfg.step.env.num_envs * cfg.step.rollout.horizon)
    cfg.max_steps = cfg.step.train_budget_steps = epochs
    return cfg


def exp002() -> CraftaxTrainLoop:
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
    cfg.experiment_name = "exp002"
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


def exp101() -> CraftaxTrainLoop:
    """exp000 on the port's defaults: standard kernel arithmetic and fp32 precision.

    exp000 pins everything PufferLib's bits depend on; exp101 keeps its recipe
    (every coefficient, the budget, the evaluation) and its own init, and
    swaps each pin for the class default: the scan, PPO, Muon and sampler
    kernels in standard Triton and torch arithmetic instead of the compat
    classes, priml's cosine instead of the fp32 one, and an fp32 carry,
    decoder output and rollout storage instead of bf16. It is the
    configuration the port recommends outside bit-for-bit work.

    Hypothesis:
      The defaults trade PufferLib's bits for precision and simplicity at no
      cost: exp101 scores roughly exp000 (47.20%), inside the same-seed spread,
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
    cfg.experiment_name = "exp101"
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


def exp003() -> CraftaxTrainLoop:
    """exp002 learning by Craftax_Baselines' 1B PPO recipe, seed 42.

    The environment and its evaluation are exp002's: original Craftax's rules,
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
        | exp003 | 12.79% | 11.99% | 12.34% | 12.37% | ~332k |
        | Craftax_Baselines (JAX) | 11.74% | 12.58% | 12.16% | 12.16% | ~155k |

      exp003's figures are its final evaluation (normalized return over about
      10,000 episodes; training about 50 minutes per seed). The official run's are its training return over the last 20
      updates, which ``ppo.py`` reports, with its speed end to end including
      compilation (about 1.9 hours per seed). The paper
      reports about 11.9%.

    Returns:
      cfg: Configured CraftaxTrainLoop.

    """
    cfg = exp002()
    cfg.experiment_name = "exp003"
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


def exp004() -> CraftaxTrainLoop:
    """exp003 at Craftax_Baselines' 1M-interaction geometry: 256 x 16 steps, lr 3e-4.

    The published one-million-interaction reproduction, exp000 of the earlier
    torch port of Craftax_Baselines, restated as one fork: 256 environments x
    16 steps in place of 1,024 x 64, Adam at 3e-4 in place of 2e-4, and 244
    updates, 999,424 transitions. The geometry, the rate and the budget are
    one treatment, the inverse of the step from that port's exp000 to its 1B
    recipe. The rest is exp003's: the actor-critic, four passes of eight
    shuffled minibatches (512 transitions each here), every coefficient, the
    linear anneal over the run, seed 42 for every stream, and the evaluation.

    Not carried over from the earlier port's exp000, each because this
    package's environment and score are exp003's:

    - Its optimistic reset, 16 environments to a freshly generated world: an
      ended episode restarts in a world of its own.
    - Its score, 64 environments x 10,000 steps from seed 42, averaging the
      episodes that finished, their return counting original Craftax's 0.1
      per point of health. The score here is exp003's: 10,000 episodes after
      training, scored by their achievement return (``metric.CraftaxScore``).

    What differs from the reference itself is exp003's list, unchanged.

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
    cfg = exp003()
    cfg.experiment_name = "exp004"
    cfg.step.env.num_envs = 256
    cfg.step.rollout.horizon = 16
    adam = cfg.step.optimizer
    assert isinstance(adam, PartialConfig)
    adam.lr = 3e-4
    # The reference counts updates as ``1e6 // 16 // 256``, whole rollouts.
    epochs = 1_000_000 // (cfg.step.env.num_envs * cfg.step.rollout.horizon)
    cfg.max_steps = cfg.step.train_budget_steps = epochs
    return cfg


def exp005() -> CraftaxTrainLoop:
    """exp003 with Craftax_Baselines' PPO-RNN policy: a reset-aware GRU, whole trajectories.

    The recipe of exp002 of the earlier torch port of Craftax_Baselines, which
    this package replaced: ``ppo_rnn.py``, forked from exp003 as that exp002
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

    Everything else is exp003's: the environment and its 1,024 x 64 rollout;
    the advantages over the whole rollout, bootstrapped from the value of the
    observation after it, read from the state the last step left, reset where
    that step ended an episode; the objective and its coefficients; Adam at
    2e-4 annealed linearly to zero over 15,258 updates behind a global clip of
    1.0; TF32 matmuls; seed 42 for every stream; and the evaluation.

    Not carried over from the earlier port's exp002, besides exp003's list:

    - Its optimistic reset, 64 fresh worlds a step shared by the episodes that
      end in it: the port's environment generates a new world at every reset
      (``FreshWorlds``), as exp003 plays.
    - Its full-precision fp32 matmuls: exp003's TF32, the reference's JAX
      default on this GPU.
    - Its evaluation, 64 environments for 10,000 steps: exp003's 10,000 fresh
      episodes (``metric.CraftaxScore``).
    - Its shuffles from torch's generator and its ``multinomial`` actions:
      exp003's NumPy shuffles keyed by the seed and the epoch, and its Philox
      sampler. The same distributions, other draws.

    Hypothesis:
      Craftax's achievements are sequential, so a policy that remembers
      anything should beat one that remembers nothing, and a GRU is the
      cheapest memory: its state is one vector at step 1 and step 10,000.
      exp005 scores above exp003's 12.37% (mean of seeds 42-44), near the
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
      of them), against the JAX port's 16.099% at that seed and exp003's
      12.79% (12.37% the mean of seeds 42-44): the memory is worth about four
      points. Training took 5,930 s for 999,948,288 transitions on one H200
      (~169k transitions/s, half exp003's: the learner replaying the GRU step
      by step takes 0.26 s of each 0.37 s epoch), and the evaluation 9 s.

    Returns:
      cfg: Configured CraftaxTrainLoop.

    """
    cfg = exp003()
    cfg.experiment_name = "exp005"
    cfg.step.model = ActorCriticRNN.Config()
    learner = cfg.step.learner
    assert isinstance(learner, ShuffledTransitions.Config)
    # ShuffledTrajectories adds no field, so it takes exp003's as they are.
    cfg.step.learner = ShuffledTrajectories.Config().update(learner)
    return cfg


def exp006() -> CraftaxPQNTrainLoop:
    """exp003 learned by PQN: an LSTM Q-learner, no replay buffer, no target network.

    The setup is exp003's: original Craftax's rules, fresh worlds and 8,268-float
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
    treatment: the question is the recipe's score on exp003's setup.

    Not carried over from the earlier port's exp003:

    - Batch renormalization takes a minibatch's whole window as one batch, and
      the loss is half the mean squared error, as purejaxql has them. That port
      renormalized each step's environments apart, moving the running
      statistics 128 times a minibatch, and regressed the whole squared
      error.
    - The environment is the port's Numba game on original Craftax's rules
      (exp002), not that port's torch reimplementation of JAX Craftax; an
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
    - Matmuls run in TF32, exp003's precision, where that port ran full fp32.
    - The score is exp003's: 10,000 fresh episodes after training
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
      network. If that holds, a value method trained on exp003's budget beats
      exp003's 12.37% and reaches about the 16.0% normalized return the
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
      of them), above exp003's 12.79% at that seed and about the reference's
      reported 16.0%. Training took 10,740 s for 999,948,288 transitions on
      one H200 (~93k transitions/s: the learner takes about 1.1 s of each
      1.4 s update), and the evaluation 9 s.

      Before the BatchRenorm and loss fixes: 7.15% at seed 42.

    Returns:
      cfg: Configured CraftaxPQNTrainLoop.

    """
    parent = exp003()
    cfg = CraftaxPQNTrainLoop()
    # The run around the learner is exp003's: its seed, precision, tracker,
    # checkpoints, evaluation cadence and score.
    cfg.update(parent, step=cfg.step)
    cfg.experiment_name = "exp006"
    cfg.step.env = parent.step.env
    sampler = cfg.step.sampler
    assert isinstance(sampler, EpsilonGreedy.Config)
    sampler.sampler = parent.step.sampler
    cfg.step.seed = 42
    # The reference counts updates as ``1e9 // 128 // 1024``, whole rollouts.
    updates = 1_000_000_000 // (cfg.step.env.num_envs * cfg.step.rollout.horizon)
    cfg.max_steps = cfg.step.train_budget_steps = updates
    return cfg


def exp007() -> CraftaxTrainLoop:
    """exp003's 1B geometry at 100M interactions, a screening budget.

    exp011 of the earlier torch port of Craftax_Baselines, restated as one
    fork: only the budget moves, 1,525 updates of 1,024 x 64, 99,942,400
    transitions, and the learning-rate horizon with it. The anneal is
    budget-relative, so a run stopped early on exp003's horizon would end
    while the rate was still high and measure something that is not a
    smaller version of its parent.

    Not carried over from the earlier port's exp011, each because this
    package's environment and score are exp003's:

    - Its parent's optimistic reset, which it carried: an ended episode
      restarts in a world of its own.
    - Its score: 64 environments x 10,000 steps, as exp004 states.
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
      counterpart's 8.976% and below exp003's 12.37% at 1B, the gap the
      screen predicts. Training took 306 s for 99,942,400 transitions on one
      H200 and 16 CPUs (about 327k transitions/s, exp003's speed), and the
      evaluation 11 s.

    Returns:
      cfg: Configured CraftaxTrainLoop.

    """
    cfg = exp003()
    cfg.experiment_name = "exp007"
    epochs = 100_000_000 // (cfg.step.env.num_envs * cfg.step.rollout.horizon)
    cfg.max_steps = cfg.step.train_budget_steps = epochs
    return cfg


def exp008() -> CraftaxTrainLoop:
    """exp003 with a policy that remembers: a gated Transformer-XL, at 1B.

    transformerXL_PPO_JAX's recipe on exp003's setup, restated as one fork.
    The policy is a gated Transformer-XL
    (:class:`~priml.baselines.craftax.policies.gtrxl.GTrXLPolicy`): the
    observation projected to 256, two blocks of relative attention (eight
    heads of 32) and an MLP, each behind a GRU gate initialized closed (bias
    2), over a memory of the last 128 steps' layer inputs, each step
    addressable on its own; then exp003's two towers, two ReLU layers of 256
    each. The learner learns from whole trajectories
    (:class:`~priml.baselines.craftax.learners.gtrxl_train_step.TrajectoryWindows`):
    four passes of eight shuffled groups of 128 environments, each trajectory
    cut into two windows of 64 steps that start from the memory the actor
    held there. Rollouts grow to 128 steps, so the memory has a rollout to
    hold; the discount rises to 0.999 and the entropy weight falls to 0.002,
    because a policy that can remember is worth pointing at rewards further
    away. Adam, its rate and annealing, the clips, the value weight and the
    1,024 environments are exp003's; 7,629 updates of 1,024 x 128 steps,
    999,948,288 transitions, exp003's budget.

    Not carried over from the earlier torch port's exp013, besides the random
    streams:

    - exp003's environment and evaluation: an ended episode restarts in a
      newly generated world, where exp013 dealt one world to 16 workers
      (optimistic reset); the score is PufferLib's 10,000 fresh episodes, where
      exp013 scored the episodes 64 workers finished in 10,000 steps.
    - Each window's starting memory is replayed, not stored: the rollout keeps
      the memory only at its first step, so before the update the learner
      reruns the policy over the first 64 steps with the collection weights.
      The values are the actor's, up to the rounding of a differently shaped
      GEMM; the cost is one more policy pass over half the rollout per update.
    - Matmuls run in TF32 (exp003's ``float32_matmul_precision``, JAX's
      default on the reference's GPU), where exp013 ran full fp32.
    - exp003's sampler and clip arithmetic, as its docstring lists them.

    Hypothesis:
      Craftax's achievements are sequential -- wood, then a table, then a
      pickaxe, then stone -- and a memoryless policy has to rediscover its
      progress from the view in front of it. Memory should be worth more here
      than architecture usually is: roughly the 18.159% that the JAX port of
      this recipe measured at seed 42 (the reference reports 18.3%), against
      exp003's 12.37%.

    References:
      https://github.com/Reytuag/transformerXL_PPO_JAX
      https://arxiv.org/abs/1910.06764
        Parisotto et al. 2020. Stabilizing transformers for reinforcement
        learning.

    Results:
      H200 + 16 CPUs, 1B transitions: 17.44% perf (seed 42).

      17.44% normalized return at seed 42 (final evaluation over 10,042
      episodes; achievements 39.5%), against the JAX port's 18.159% at the
      same seed and exp003's 12.37%: memory is worth about 5 points here.
      Trained in 2.6 hours on one H200 at ~107k transitions/s.

    Returns:
      cfg: Configured CraftaxTrainLoop.

    """
    cfg = exp003()
    cfg.experiment_name = "exp008"
    cfg.step.model = GTrXLPolicy.Config()
    # Its defaults are the reference's passes, windows and coefficients.
    learner = cfg.step.learner = TrajectoryWindows.Config()
    learner.seed = 42
    cfg.step.rollout.horizon = 128
    epochs = 1_000_000_000 // (cfg.step.env.num_envs * cfg.step.rollout.horizon)
    cfg.max_steps = cfg.step.train_budget_steps = epochs
    return cfg


def exp102() -> CraftaxTrainLoop:
    """exp101 with a board encoder, layer injection and a feasibility loss, for 20B.

    The board-encoder recipe of the original runs, the strongest scratch run
    among them, restated as one fork as exp003 restates its reference. The
    policy is exp101's four-layer MinGRU with three additions:

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
    exp101's, and the evaluation runs every 1B transitions.

    What differs from the reference run, besides the random streams: the
    encoder drops an input that the reference fed zeros by design and a
    gradient-scale wrapper at scale 1; the carry, the decoder's output and the
    rollout's log-probabilities, values and rewards are fp32 (exp101's
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
        | exp102 | 56.24% | 54.77% | 50.01% | 50.02% | 52.76% | ~512k |
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
    cfg = exp101()
    cfg.experiment_name = "exp102"
    # Every seed of a recipe files under one W&B group, so they share a board.
    trackers = cfg.tracker
    assert isinstance(trackers, TrackerList.Config)
    dashboard = unwrap_tracker_config(trackers.trackers["wandb"])
    assert isinstance(dashboard, WandbTracker.Config)
    dashboard.group = "exp102"
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


def exp103() -> CraftaxTrainLoop:
    """exp102 with frontier practice, stored carry, self-imitation, rewards over 8.

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
      exp102's reference at that seed.

    References:
      https://arxiv.org/abs/1812.03381
        Salimans and Chen 2018. Learning Montezuma's Revenge from a single
        demonstration.
      https://arxiv.org/abs/1806.05635
        Oh et al. 2018. Self-imitation learning.

    Results:
      H200 + 16 CPUs, 20B transitions: 72.48% mean perf (3 seeds: 73, 74, 75).

      Brackets its reference, 19 points above exp102 on the mean of the same
      three seeds. Three seeds at 20B, each on one H200:

        | | seed 73 | seed 74 | seed 75 | mean | transitions/s |
        |---|---|---|---|---|---|
        | exp103 | 67.51% | 77.66% | 72.27% | 72.48% | ~509k |
        | reference runs | 72.54% | 77.04% | 72.71% | 73.25% | - |

      Final evaluations over about 10,000 episodes, ending on the
      necromancer's fall. The reference's mean is over its eight later
      seeds, 73-80 (70.54% to 77.04%); its first run, at seed 73, scored
      75.30%. Seed 74 reaches the last floor in 61% of its evaluation
      episodes, seeds 73 and 75 in 1% and 0%. At 0.99x exp102's speed,
      practice and imitation cost about 1%.

    Returns:
      cfg: Configured CraftaxTrainLoop.

    """
    cfg = exp102()
    cfg.experiment_name = "exp103"
    trackers = cfg.tracker
    assert isinstance(trackers, TrackerList.Config)
    dashboard = unwrap_tracker_config(trackers.trackers["wandb"])
    assert isinstance(dashboard, WandbTracker.Config)
    dashboard.group = "exp103"
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


def exp109() -> CraftaxTrainLoop:
    """exp103 starting each episode in a newly generated world, evaluated on its pool.

    The world-source arm of the recipe runs' original-Craftax ablations,
    restated as one fork: every reset generates a new world from the
    environment's own
    stream, as original Craftax does (D6), so training never revisits a
    world; practice still restores the states its donors saved. The
    evaluation is exp103's, uncapped and without practice on the 8,192-world
    pool, so the two score on the same worlds.

    Hypothesis:
      Worlds never seen twice teach what holds across worlds rather than
      what holds in the pool's, and exp109 scores above exp103 on exp103's
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
    cfg = exp103()
    cfg.experiment_name = "exp109"
    trackers = cfg.tracker
    assert isinstance(trackers, TrackerList.Config)
    dashboard = unwrap_tracker_config(trackers.trackers["wandb"])
    assert isinstance(dashboard, WandbTracker.Config)
    dashboard.group = "exp109"
    # exp103's evaluation env, taken before the restart changes: left unset, the
    # evaluation copies training's and would score on fresh worlds, not exp103's.
    evaluation = cfg.step.evaluation.env = cfg.step.env.copy_tree()
    evaluation.stall_cap = None
    evaluation.practice = None
    cfg.step.env.restart = FreshWorlds.Config()
    return cfg


def exp110() -> CraftaxTrainLoop:
    """exp103 with a frozen world model, fitted to early play, as its only encoder.

    Not yet reproducible: its world model is the original runs' fit to 100M
    decisions of early play, which no experiment here trains. TODO: add the
    capture of an early-training exp103 policy, its corpus, and the world-model
    experiment that fits it, then read that run's final checkpoint; or drop
    exp110 to exp113 and exp115 and exp116.

    The early frozen arm of the sole world-model experiment, on the original
    implementation's engine, restated as one fork. The policy keeps exp103's four
    MinGRU layers of 1,024, its heads, its feasibility loss and its whole
    recipe, and loses its encoder: no board encoder, no ``proj_in`` and no
    injection. Its trunk reads only a frozen world model's state of the
    episode: each actor step, a model of exp001's geometry
    (``world_model.experiments``), fitted to 100M decisions of early play,
    reads the episode so far and taps its final-normed hidden state at the current
    observation, 1,152 floats. A new projection without bias, 1,152 -> 1,024,
    drawn as every projection is, maps it into the first layer. The
    observation reaches only the feasibility loss's targets. The rollout
    stores each step's feature; the learner, and the self-imitation replaying
    a branch, read the stored ones.

    The engine is exp107's, the original implementation's: at most 1,024 positions per
    row, two per decision; a row that would overflow restarts from its last
    256 decisions, as a mid-episode training window; a practice restore begins
    a mid-episode window at the restored observation. It runs in bf16 with
    FlashAttention 4 over the cache and its per-block functions compiled.

    The evaluation is the reference arms': 10,000 episodes every 250M
    transitions -- every 477 epochs, 250,085,376 transitions, the first epoch
    past 250M -- and at the end. It plays 1,024 environments, 4 buffers of
    256, uncapped and without practice: an evaluation of 2,048 has no room
    beside training's caches. Of the H200's 150.1 GB, this config's training
    reserved 95.7 GB over its first three epochs, re-prefills included, and
    an evaluation of 1,024 beside it 130.7 GB. Beside training, an evaluation
    of 2,048 ran out of memory (a random-init world model, practice off). With
    512 training environments an evaluation of 1,024 played 59.6k
    transitions/s against 62.5k for 2,048. Its 10,000 episodes come from 1,024
    environments' streams, not exp103's 2,048.

    The budget is exp103's 20B; exp115 is its 1B arm.

    Hypothesis:
      A world model's state, fitted only to predict early play, is by itself
      an observation a policy can learn from: the arm learns, below exp103,
      whose encoder trains with the policy. On the original engine it trains
      at the original implementation's actor speed, 37-45k transitions/s once
      contexts fill (its reference runs: 37.0k median at a context of 384-448
      decisions; 52-61k from this config's init), where the reference's exact
      512-decision windows, recomputed at every decision, ran at 483.

    References:
      ``world_model/README.md``, "The frozen feature for RL": the engine and
      its ``Refill`` history.

    Results:
      H200 + 16 CPUs, 20B transitions: still training at seed 73, 52.35%
      perf at its 9.5B evaluation, reading the original runs' early fit. Its
      1B arm is exp115's.

      Reference: none. The original runs ended before any evaluation.

    Returns:
      cfg: Configured CraftaxTrainLoop.

    """
    # The early fit has no producer in this chain; the record stays in the
    # docstring above.
    msg = "TODO: exp110 needs a world model fitted to early play; see its docstring."
    raise NotImplementedError(msg)


def exp111() -> CraftaxTrainLoop:
    """exp110 with the world model fitted to mature play.

    Not yet reproducible: its world model is the original runs' fit to 100M
    decisions of mature play, which no experiment here trains. TODO: add the
    capture of a 20B exp103 policy's play, its corpus, and the world-model
    experiment that fits it, then read that run's final checkpoint; or drop
    exp111 and exp113.

    The mature frozen arm: the same model, engine and policy, the weights
    fitted to 100M decisions of the strongest mature control's play, whose
    episodes reach the late floors.

    Hypothesis:
      A world model that has seen the late floors gives the policy a state of
      them: exp111 scores above exp110.

    References:
      exp110's.

    Results:
      H200 + 16 CPUs, 20B transitions: still training at seed 73, 67.33%
      perf at its 7.0B evaluation, reading the original runs' mature fit.

      Reference: none. The original runs ended before any evaluation.

    Returns:
      cfg: Configured CraftaxTrainLoop.

    """
    # The mature fit has no producer in this chain; the record stays in the
    # docstring above.
    msg = "TODO: exp111 needs a world model fitted to mature play; see its docstring."
    raise NotImplementedError(msg)


def exp112() -> CraftaxTrainLoop:
    """exp110 training its world model with the policy.

    Not yet reproducible: it builds once exp110 does (TODO there).

    The early joint arm, on exp110's engine, restated as one fork. The learner
    trains a copy of the world model's feature weights -- its frame table and
    encoder, ``obs_proj``, the action table, ``start``, the 20 global blocks
    and the final norm -- with the policy, as the reference's joint arms
    registered them: one optimizer group after the policy's, under exp103's
    Muon, schedule and global clip, each fused QKV projection orthogonalized
    as one matrix. Each window recomputes its features from the contexts the
    actor read, with the copy's current weights
    (:mod:`~priml.baselines.craftax.world_model.context`): the steps whose
    contexts begin at the same decision share one causal pass, and the
    features' gradient reaches the weights in a second pass. The feature
    turns joint with it: each rollout stores every step's frame tokens, previous
    action and context, and each row's last 511 decisions before it; before
    each rollout the copy is published into the actor's model and every row's
    history rebuilt under it. The replay runs FlashAttention 4, its per-block
    and encoder functions compiled. The self-imitation still reads the stored
    features, so its loss trains the policy alone.

    On an H200, the learner's epoch at exp103's geometry took 88 s on
    contexts of 257 to 512 decisions, where the reference's, recomputing a
    512-decision window for every transition, took 6,214 s. This config's
    training, from its init, ran 6.9k-10.8k transitions/s over its first
    three epochs, its contexts averaging 115-120 decisions and each rebuild
    taking 13-15 s; it peaked at 89.1 GiB allocated and 113.3 GiB reserved,
    and an evaluation of 1,024 environments beside it at 108.7 and 135.8 GiB,
    of the H200's 140.4. At the same
    weights, after a rebuild, the learner's features are within 0.3%
    (relative RMS) of those the actor read.

    Hypothesis:
      Training the encoder with the policy adapts its state to what the
      policy needs: exp112 scores above exp110, its frozen counterpart.

    References:
      ``world_model/README.md``, "The frozen feature for RL" (``joint``).

    Results:
      H200 + 16 CPUs, 20B transitions: still training at seed 73, 54.16%
      perf at its 1.25B evaluation. Its 1B arm is exp116's.

      Reference: none. The original runs ended before any evaluation.

    Returns:
      cfg: Configured CraftaxTrainLoop.

    """
    # Its parent raises until its producer lands; the record stays in the
    # docstring above.
    msg = "TODO: exp112 builds once exp110 does; see exp110's docstring."
    raise NotImplementedError(msg)


def exp113() -> CraftaxTrainLoop:
    """exp111 training its world model with the policy.

    Not yet reproducible: it builds once exp111 does (TODO there).

    The mature joint arm, on exp111's engine: exp112's change, applied to the
    mature world model.

    Hypothesis:
      As exp112's: exp113 scores above exp111, its frozen counterpart.

    References:
      exp112's.

    Results:
      H200 + 16 CPUs, 20B transitions: still training at seed 73, 60.31%
      perf at its 1.25B evaluation.

      Reference: none. The original runs ended before any evaluation.

    Returns:
      cfg: Configured CraftaxTrainLoop.

    """
    # Its parent raises until its producer lands; the record stays in the
    # docstring above.
    msg = "TODO: exp113 builds once exp111 does; see exp111's docstring."
    raise NotImplementedError(msg)


def exp114() -> CraftaxTrainLoop:
    """exp103's final policy fine-tuned for the boss fight, 3B more transitions.

    The best arm of the original runs' boss-fight screen, restated as one
    fork. It starts from exp103's final checkpoint, the run's fp32 masters,
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
    - The rate is a tenth of exp103's peak, held constant: the parent ended
      its schedule at zero, and a decay to zero over the fine-tune scored
      lower.
    - The budget is 3B transitions from the parent's 20B.

    Hypothesis:
      A policy that reaches the final floor but rarely wins learns the fight
      once every hit and every wave's kill pays and its horizon spans a
      fight: it defeats the necromancer in a large share of evaluation
      episodes, which pay no bonus, where exp103 rarely does.

    References:
      https://rekursiv.ai/blog/craftax/
        The boss-fight reward and its screen of six arms.

    Results:
      Reference (the original runs, one H200 each): from exp103's seed-78 run
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
    cfg = exp103()
    # The parent's last checkpoint, read before this fork renames the run and
    # sets its own budget.
    cfg.step.checkpoint = (
        f"/runs/{cfg.study_name}/{cfg.experiment_name}/checkpoints/"
        f"step_{int(cfg.max_steps):08d}.pt"
    )
    cfg.experiment_name = "exp114"
    trackers = cfg.tracker
    assert isinstance(trackers, TrackerList.Config)
    dashboard = unwrap_tracker_config(trackers.trackers["wandb"])
    assert isinstance(dashboard, WandbTracker.Config)
    dashboard.group = "exp114"
    cfg.step.env.boss_fight_reward = BossFightReward.Config()
    windows = cfg.step.learner
    assert isinstance(windows, AgentWindows.Config)
    ppo = windows.objective
    assert isinstance(ppo, TritonPPO.Config)
    ppo.discount = 0.9998
    muon = cfg.step.optimizer
    assert isinstance(muon, FusedMuon.Config)
    # A tenth of exp103's peak as the arm's command wrote it; peak / 10 lands
    # one double ulp above.
    muon.lr = 0.0001182217733003199
    schedule = cfg.step.schedule
    assert isinstance(schedule, ProgressSchedule.Config)
    schedule.curve = PartialConfig(constant)
    transitions = cfg.step.env.num_envs * cfg.step.rollout.horizon
    cfg.max_steps = cfg.step.train_budget_steps = 3_000_000_000 // transitions
    return cfg


def exp115() -> CraftaxTrainLoop:
    """exp110 stopped after 1B transitions, its schedule keeping the 20B horizon.

    Not yet reproducible: it builds once exp110 does (TODO there).

    The reference's 1B arm of the early frozen world model: the run stops
    after epoch 1,907, 999,817,216 transitions, while the schedule keeps
    exp110's 20B horizon, as the reference's did, so the rate is still near
    its peak; it evaluates just past 250M, 500M and 750M and at the end.

    Hypothesis:
      A frozen world model's state of early play is an observation a policy
      learns from within 1B transitions.

    References:
      exp110's.

    Results:
      H200 + 16 CPUs, 999,817,216 transitions: 37.51% perf (seed 73), reading
      the original runs' early fit.

      Reference: none. The original runs ended before any evaluation.

    Returns:
      cfg: Configured CraftaxTrainLoop.

    """
    # Its parent raises until its producer lands; the record stays in the
    # docstring above.
    msg = "TODO: exp115 builds once exp110 does; see exp110's docstring."
    raise NotImplementedError(msg)


def exp116() -> CraftaxTrainLoop:
    """exp112 stopped after 1B transitions, its schedule keeping the 20B horizon.

    Not yet reproducible: it builds once exp110 does (TODO there).

    The reference's 1B arm of the early joint world model: exp115's budget
    applied to exp112.

    Hypothesis:
      Training the world model with the policy adapts its state to what the
      policy needs: exp116 scores above exp115, its frozen counterpart.

    References:
      exp112's.

    Results:
      H200 + 16 CPUs, 999,817,216 transitions: 52.10% perf (seed 73),
      against exp115's 37.51%, from the original runs' early fit.

      Reference: none. The original runs ended before any evaluation.

    Returns:
      cfg: Configured CraftaxTrainLoop.

    """
    # Its parent raises until its producer lands; the record stays in the
    # docstring above.
    msg = "TODO: exp116 builds once exp110 does; see exp110's docstring."
    raise NotImplementedError(msg)


def exp104() -> CraftaxTrainLoop:
    """exp102 at width 2,048, on 1,024 environments of 512 steps.

    The recipe of the original runs' capacity arm, restated as one fork:
    exp102's policy twice as wide
    -- ``proj_in`` 1,635 -> 2,048, four MinGRU layers of 2,048, each injection
    2,048 -> 128 -> 2,048 -- learning from windows twice as long, its rollout
    regrouped from 2,048 environments of 256 steps into 1,024 of 512. The
    rollout keeps its 524,288 transitions, so the minibatches of 32,768 (64
    environments' whole windows), the 18 updates per rollout, the budget, the
    schedule and the evaluation are exp102's.

    Hypothesis:
      Width and window length are one treatment, the reference's own arm. At
      20B it scores in its reference's range, above exp102: the reference's
      seed-73 run read 16.10% at 250M, 27.46% at 1B and 65.10% at 20B.

    References:
      exp102's.

    Results:
      Reference (the original runs, seed 73, 20B): 65.10% perf. Not yet run
      on the port.

    Returns:
      cfg: Configured CraftaxTrainLoop.

    """
    cfg = exp102()
    cfg.experiment_name = "exp104"
    trackers = cfg.tracker
    assert isinstance(trackers, TrackerList.Config)
    dashboard = unwrap_tracker_config(trackers.trackers["wandb"])
    assert isinstance(dashboard, WandbTracker.Config)
    dashboard.group = "exp104"
    model = cfg.step.model
    assert isinstance(model, MinGRUPolicy.Config)
    model.channels_hidden = 2_048
    # 1,024 x 512 = 2,048 x 256 transitions: exp102's epoch counts stand.
    cfg.step.env.num_envs = 1_024
    cfg.step.rollout.horizon = 512
    return cfg


def exp105() -> CraftaxTrainLoop:
    """exp104 with a ConvNeXt trunk in place of the board's residual blocks.

    The original runs' ConvNeXt-trunk encoder arm, restated as one fork. The
    board's two 16-channel depthwise residual blocks
    become one :class:`~priml.baselines.craftax.policies.encoder.ConvNextTrunk`: a
    1 x 1 lift to 64 channels; two ConvNeXt blocks, each a 5 x 5 depthwise
    window, a per-cell LayerNorm, a 256-wide MLP with exact GELU and a layer
    scale starting at 1e-6; and a 1 x 1 projection back onto the board, zero
    at init. The FiLM branch, the multiscale mix and everything after them are
    exp104's, and both start as the multi-hot embedding.

    What differs from the reference, besides the random streams, is exp102's
    list; its D2 now covers the trunk too. The reference wraps the trunk in
    gradient-scale round trips at scale 1, ``t + (b - t)`` and ``x +
    scale(proj_out(b))``: identities in real arithmetic that round the board
    in bf16. The parity suite holds the policy to the
    reference's bytes with D1 and D2 applied, and with D1 alone once its trunk
    reproduces the round trips.

    Hypothesis:
      A wider trunk of ConvNeXt blocks reads the board better early. Its
      reference's seed-73 run read 18.99% at 250M and 38.27% at 1B, against
      exp104's reference's 16.10% and 27.46%, and 62.01% at 20B against its
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
    cfg = exp104()
    cfg.experiment_name = "exp105"
    trackers = cfg.tracker
    assert isinstance(trackers, TrackerList.Config)
    dashboard = unwrap_tracker_config(trackers.trackers["wandb"])
    assert isinstance(dashboard, WandbTracker.Config)
    dashboard.group = "exp105"
    model = cfg.step.model
    assert isinstance(model, MinGRUPolicy.Config)
    encoder = model.embedding
    assert isinstance(encoder, BoardEncoder.Config)
    encoder.board.block = ConvNextTrunk.Config()
    encoder.board.num_layers = 1
    return cfg


def exp106() -> CraftaxTrainLoop:
    """exp105 for the first 250M transitions of its 20B schedule, evaluated every 50M.

    The control arm of the original runs' frozen world-model experiment:
    exp105 stopped after 476 epochs, 249,561,088 transitions, while its schedule
    keeps the 20B horizon, so the whole run is the rate's warmup. That warmup
    is the reference's linear ramp from zero, where exp102's ``one_cycle``
    ramps by a cosine: immaterial to a 20B run, 1.25% of which it spans, but
    the whole of this one, where the cosine would give under half the linear
    rate through the first fifth and 17% less summed rate by 150M. Each
    evaluation, 10,000 episodes, runs at the first epoch past a multiple of
    50M transitions; the last is the final one.

    Hypothesis:
      The budget, the linear ramp and the evaluation cadence are one
      treatment, the reference arm's. It scores as the reference's three
      seeds did: 11.9% averaged over the evaluations at 150M, 200M and 250M.

    References:
      ``world_model/README.md``, "The frozen feature for RL".

    Results:
      H200 + 16 CPUs, 250M transitions: 12.17% perf, the mean of the
      evaluations at 150M, 200M and 250M (seed 73); 18.17% at 250M.

      Reference (the original runs, seeds 73-75): 11.92% on the same mean.

    Returns:
      cfg: Configured CraftaxTrainLoop.

    """
    cfg = exp105()
    cfg.experiment_name = "exp106"
    trackers = cfg.tracker
    assert isinstance(trackers, TrackerList.Config)
    dashboard = unwrap_tracker_config(trackers.trackers["wandb"])
    assert isinstance(dashboard, WandbTracker.Config)
    dashboard.group = "exp106"
    transitions = cfg.step.env.num_envs * cfg.step.rollout.horizon
    cfg.max_steps = 250_000_000 // transitions
    schedule = cfg.step.schedule
    assert isinstance(schedule, ProgressSchedule.Config)
    schedule.curve = PartialConfig(warmup, fraction=250_000_000 / 20_000_000_000)
    cfg.num_steps_eval = math.ceil(50_000_000 / transitions)
    return cfg


def exp107() -> CraftaxTrainLoop:
    """exp106 reading a frozen world model's state of the episode at every step.

    The treatment arm of that experiment, restated as one fork. Each actor
    step, the world model exp001 of ``world_model.experiments``, from its run's
    final checkpoint (step 1,525, trained on the base corpus), reads the
    episode so far -- each observation's frame tokens, each action -- and its
    final-normed hidden state at the current observation, 1,152 floats, is the
    feature
    (:class:`~priml.baselines.craftax.world_model.feature.WorldModelFeature`).
    The recorded run read the original implementation's exp001 at seed 0, in
    the port's layout. Its weights stay frozen. The policy adds a projection
    of the feature, 1,152 -> 2,048 and zero at init, to ``proj_in``'s output,
    so the run starts as exp106's. The rollout stores each step's feature and
    the learner reads the stored one.

    The engine holds at most 1,024 positions of each row's history, two per
    decision; before a row would overflow, it restarts from its last 256
    decisions, as a mid-episode training window. It runs in bf16 with
    FlashAttention 4 over the cache and its per-block functions compiled, as
    its reference's did.

    Hypothesis:
      A world model's state of the episode carries what the recurrent policy
      has not yet learned to keep. Its reference's three seeds averaged 47.1%
      over the evaluations at 150M, 200M and 250M, against 11.9% for exp106's
      reference (54.0, 55.2 and 54.9% at 250M).

    References:
      ``world_model/README.md``, "The frozen feature for RL".

    Results:
      H200 + 16 CPUs, 250M transitions: 42.92% perf, the mean of the
      evaluations at 150M, 200M and 250M (seed 73); 53.34% at 250M.

      Reference (the original runs, seeds 73-75): 47.08% on the same mean.

    Returns:
      cfg: Configured CraftaxTrainLoop.

    """
    cfg = exp106()
    cfg.experiment_name = "exp107"
    trackers = cfg.tracker
    assert isinstance(trackers, TrackerList.Config)
    dashboard = unwrap_tracker_config(trackers.trackers["wandb"])
    assert isinstance(dashboard, WandbTracker.Config)
    dashboard.group = "exp107"
    feature = cfg.step.feature = WorldModelFeature.Config()
    assert isinstance(feature.weights, TrainedWeights.Config)
    world_model = world_model_exp001()
    feature.weights.checkpoint = Path(
        f"/runs/{world_model.study_name}/{world_model.experiment_name}/checkpoints/"
        f"step_{int(world_model.max_steps):08d}.pt",
    )
    feature.attention = Flash4CacheAttention.Config()
    feature.compile = PartialConfig(
        torch.compile,
        fullgraph=True,
        dynamic=False,
        mode="max-autotune-no-cudagraphs",
    )
    model = cfg.step.model
    assert isinstance(model, MinGRUPolicy.Config)
    proj = model.proj_feature = Linear.Config()
    proj.channels_in = 1_152  # The world model's width.
    proj.init_weight = nn.init.zeros_
    return cfg


def exp108() -> CraftaxTrainLoop:
    """exp107 with the world model's weights at their random init.

    The random-weight arm of that experiment: the same engine, tap and
    projection over exp001's untrained model, drawn from seed 0, which is its
    reference's random-weight extract byte for byte. It separates what training put in the feature from what
    any fixed function of the history gives.

    Hypothesis:
      Random weights give nothing a policy can use: its reference's two seeds
      averaged 11.7%, against exp106's reference at 11.9%.

    References:
      ``world_model/README.md``, "The frozen feature for RL".

    Results:
      H200 + 16 CPUs, 250M transitions: 11.52% perf, the mean of the
      evaluations at 150M, 200M and 250M (seed 73); 15.76% at 250M.

      Reference (the original runs, seeds 73 and 74): 11.67% on the same
      mean.

    Returns:
      cfg: Configured CraftaxTrainLoop.

    """
    cfg = exp107()
    cfg.experiment_name = "exp108"
    trackers = cfg.tracker
    assert isinstance(trackers, TrackerList.Config)
    dashboard = unwrap_tracker_config(trackers.trackers["wandb"])
    assert isinstance(dashboard, WandbTracker.Config)
    dashboard.group = "exp108"
    feature = cfg.step.feature
    assert isinstance(feature, WorldModelFeature.Config)
    feature.weights = InitialWeights.Config()
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
