r"""Craftax experiments.

``exp000`` is the published one-million-interaction PPO reproduction. Every
later experiment forks a named parent and applies ONE change, stating its
hypothesis and source, so the chain reads as an argument rather than a pile of
settings.

    exp000  PPO, 1M interactions, 256 envs x 16 steps
      +-- exp001  the published 1B recipe: 1024 envs x 64 steps, lr 2e-4
            +-- exp002  a policy with memory, cheaply: a GRU at 1B
            +-- exp003  no policy at all: Q-learning with an LSTM at 1B
            +-- exp011  the 1B geometry at 100M, as a screening budget
                  +-- exp013  a policy with memory, expensively: GTrXL at 1B

The names are the JAX study's, kept so a torch result can be read against the
number its JAX counterpart measured. What the names do NOT carry over is the
systems work: the JAX ``exp011``/``exp013`` also selected a scanned evaluator
and a 16-state adaptive reset pool, both of which are XLA-specific and have no
torch analogue. Each factory's docstring names what it dropped, because a
receipt that omits it would read as a comparison it is not.

Launch::

    uv --quiet run --frozen python -m priml priml.baselines.craftax.experiments.exp000

``--override PATH=VALUE`` adapts a run to the machine it lands on. A
hyperparameter belongs in an experiment, not on the command line: overriding
one produces a result whose config exists nowhere in the code, so it cannot be
rerun or compared later. Write a fork instead.
"""

from __future__ import annotations

from dataclasses import field

from configgle import Makes

from priml.baselines.craftax.data import CraftaxRollouts
from priml.baselines.craftax.gtrxl_train_step import CraftaxGTrXLTrainStep
from priml.baselines.craftax.metric import CraftaxScore
from priml.baselines.craftax.pqn_train_step import CraftaxPQNTrainStep
from priml.baselines.craftax.rnn_train_step import CraftaxRNNTrainStep
from priml.baselines.craftax.train_step import CraftaxTrainStep
from priml.runtime import SingleProcess
from priml.train.train_loop import TrainLoop


class CraftaxTrainLoop(Makes["TrainLoop"], TrainLoop.Config):
    """A training loop with the Craftax step and rollout cadence in place.

    Narrowing the two slots here rather than at each call site is what lets a
    factory read ``cfg.step.model`` directly, with no ``isinstance`` narrow to
    reach a field it is about to set.
    """

    step: CraftaxTrainStep.Config = field(default_factory=CraftaxTrainStep.Config)
    """Model, environment, and PPO settings."""

    dataset: CraftaxRollouts.Config = field(default_factory=CraftaxRollouts.Config)
    """The loop's cadence; the data lives in the step's environment."""


class CraftaxRNNTrainLoop(Makes["TrainLoop"], TrainLoop.Config):
    """The same loop with the recurrent step, whose config is its own type."""

    step: CraftaxRNNTrainStep.Config = field(
        default_factory=CraftaxRNNTrainStep.Config,
    )
    """Recurrent model, environment, and PPO settings."""

    dataset: CraftaxRollouts.Config = field(default_factory=CraftaxRollouts.Config)
    """The loop's cadence; the data lives in the step's environment."""


class CraftaxPQNTrainLoop(Makes["TrainLoop"], TrainLoop.Config):
    """The same loop with the Q-learning step, whose config is its own type."""

    step: CraftaxPQNTrainStep.Config = field(
        default_factory=CraftaxPQNTrainStep.Config,
    )
    """Recurrent Q-network, environment, and Q-learning settings."""

    dataset: CraftaxRollouts.Config = field(default_factory=CraftaxRollouts.Config)
    """The loop's cadence; the data lives in the step's environment."""


class CraftaxGTrXLTrainLoop(Makes["TrainLoop"], TrainLoop.Config):
    """The same loop with the recurrent step, whose config is a different type.

    A separate class rather than a union: the recurrent step has fields the
    feed-forward one does not (``gradient_window``), and a factory that sets
    them should not have to narrow to reach them.
    """

    step: CraftaxGTrXLTrainStep.Config = field(
        default_factory=CraftaxGTrXLTrainStep.Config,
    )
    """Recurrent model, environment, and PPO settings."""

    dataset: CraftaxRollouts.Config = field(default_factory=CraftaxRollouts.Config)
    """The loop's cadence; the data lives in the step's environment."""


def exp000() -> CraftaxTrainLoop:
    """Published Craftax PPO at one million interactions, seed 42.

    The baseline every other experiment forks, and the only one that states a
    recipe rather than a change. Frozen: improvements belong in a fork, never
    in an edit here, so a result measured against it stays comparable.

    Hypothesis:
      A feed-forward actor-critic trained with clipped PPO reaches roughly
      2.2% normalized episodic return at one million interactions -- the bar
      any additional mechanism must clear to earn its complexity.

    References:
      https://github.com/MichaelTMatthews/Craftax_Baselines
      Matthews et al. 2024. Craftax: a lightning-fast benchmark for
      open-ended reinforcement learning.

    Results:
      TBD. The JAX port of this recipe measured 2.212% at 999,424 steps.

    """
    cfg = CraftaxTrainLoop()
    cfg.study_name = "craftax"
    cfg.experiment_name = "exp000"

    cfg.step.env.num_envs = 256
    cfg.step.env.seed = 42
    cfg.step.env.optimistic_reset_ratio = 16
    cfg.step.rollout_steps = 16
    cfg.step.num_epochs = 4
    cfg.step.num_minibatches = 8
    cfg.step.learning_rate = 3e-4
    cfg.step.anneal_learning_rate = True
    cfg.step.discount = 0.99
    cfg.step.trace_decay = 0.8
    cfg.step.clip_epsilon = 0.2
    cfg.step.entropy_coefficient = 0.01
    cfg.step.value_coefficient = 0.5
    cfg.step.max_grad_norm = 1.0
    cfg.step.seed = 42

    cfg.step.model.hidden_size = 512
    cfg.step.model.num_layers = 3

    # The budget is stated in INTERACTIONS, which is what the benchmark
    # compares, and converted here because the loop counts updates.
    cfg.max_steps = cfg.step.total_train_steps = _updates(
        interactions=1_000_000,
        num_envs=cfg.step.env.num_envs,
        rollout_steps=cfg.step.rollout_steps,
    )
    # One epoch spans the run: nothing here is epoch-driven, and an epoch
    # boundary would only interrupt it.
    cfg.dataset.updates_per_epoch = int(cfg.max_steps)
    cfg.num_steps_eval = cfg.max_steps

    score = cfg.metrics["craftax"] = CraftaxScore.Config()
    score.num_envs = 64
    score.steps = 10_000
    score.seed = 42

    cfg.runtime = SingleProcess.Config()
    return cfg


def exp001() -> CraftaxTrainLoop:
    """exp000 at the published billion-interaction geometry.

    Four times the workers, four times the rollout, a thousand times the
    budget, and a lower learning rate to match. Model and objective are
    untouched.

    Hypothesis:
      The same recipe reaches roughly 11.9% normalized return given a
      thousand times the experience -- the benchmark's headline PPO number,
      and evidence that exp000's score is budget-limited rather than
      recipe-limited.

    References:
      https://github.com/MichaelTMatthews/Craftax_Baselines
      Matthews et al. 2024. Craftax: a lightning-fast benchmark for
      open-ended reinforcement learning.

    Results:
      TBD. The JAX port of this recipe measured 11.867% at seed 42.

    """
    cfg = exp000()
    cfg.experiment_name = "exp001"
    cfg.step.env.num_envs = 1_024
    cfg.step.rollout_steps = 64
    cfg.step.learning_rate = 2e-4
    cfg.max_steps = cfg.step.total_train_steps = _updates(
        interactions=1_000_000_000,
        num_envs=cfg.step.env.num_envs,
        rollout_steps=cfg.step.rollout_steps,
    )
    cfg.dataset.updates_per_epoch = int(cfg.max_steps)
    cfg.num_steps_eval = cfg.max_steps
    return cfg


def exp002() -> CraftaxRNNTrainLoop:
    """exp001 with the cheapest possible memory: a reset-aware GRU.

    Replaces the feed-forward network with a single recurrent vector carried
    step to step, and the optimization with the trajectory-major form any
    recurrent policy requires. Nothing else moves -- same budget, same
    workers, same rollout, same coefficients.

    Hypothesis:
      Craftax's achievements are sequential, so a policy that remembers
      anything at all should beat one that remembers nothing. A GRU is the
      cheapest way to test that: its state is one vector, so memory costs the
      same at step 1 and step 10,000. If it captures most of exp013's gain,
      the transformer's attention is not paying for itself.

    Not carried over from the JAX exp002:
      Nothing. This is the whole treatment.

    References:
      https://github.com/MichaelTMatthews/Craftax_Baselines
      Matthews et al. 2024. Craftax: a lightning-fast benchmark for
      open-ended reinforcement learning.

    Results:
      TBD. The JAX port of this recipe measured 16.099% at seed 42.

    """
    parent = exp001()
    cfg = CraftaxRNNTrainLoop()
    cfg.study_name = parent.study_name
    cfg.experiment_name = "exp002"

    cfg.step.env.num_envs = 1_024
    cfg.step.env.seed = 42
    cfg.step.env.optimistic_reset_ratio = 16
    cfg.step.rollout_steps = 64
    cfg.step.num_epochs = 4
    cfg.step.num_minibatches = 8
    cfg.step.learning_rate = 2e-4
    cfg.step.anneal_learning_rate = True
    cfg.step.discount = 0.99
    cfg.step.trace_decay = 0.8
    cfg.step.clip_epsilon = 0.2
    cfg.step.entropy_coefficient = 0.01
    cfg.step.value_coefficient = 0.5
    cfg.step.max_grad_norm = 1.0
    cfg.step.seed = 42
    cfg.step.model.hidden_size = 512

    cfg.max_steps = cfg.step.total_train_steps = _updates(
        interactions=1_000_000_000,
        num_envs=cfg.step.env.num_envs,
        rollout_steps=cfg.step.rollout_steps,
    )
    cfg.dataset.updates_per_epoch = int(cfg.max_steps)
    cfg.num_steps_eval = cfg.max_steps
    cfg.metrics = dict(parent.metrics)
    cfg.runtime = parent.runtime
    return cfg


def exp003() -> CraftaxPQNTrainLoop:
    """exp001 with no policy network at all: recurrent Q-learning.

    The only experiment here that is not PPO. It learns action VALUES and
    acts greedily on them, exploring by a decaying random-action rate rather
    than by an entropy bonus -- and it does so without the two structures deep
    Q-learning normally requires, a replay buffer and a target network.

    Hypothesis:
      At a thousand parallel workers the buffer is redundant, because the
      batch is already decorrelated; and batch renormalization plus a
      multi-step Q(lambda) target keeps the regression stable enough to drop
      the target network. If that holds, a value method reaches PPO-RNN's
      score, which would say the policy gradient was never the essential part.

    Not carried over from the JAX exp003:
      Nothing architectural. Its 128-step rollouts, 4 minibatches, RAdam, and
      epsilon schedule are all here.

    References:
      https://arxiv.org/abs/2407.04811
      Gallici et al. 2024. Simplifying deep temporal difference learning.
      https://github.com/mttga/purejaxql

    Results:
      TBD. The JAX study never ran this at 1B: its canary projected 36
      H100-hours. The public baseline reports about 16.0 normalized return.

    """
    parent = exp001()
    cfg = CraftaxPQNTrainLoop()
    cfg.study_name = parent.study_name
    cfg.experiment_name = "exp003"

    cfg.step.env.num_envs = 1_024
    cfg.step.env.seed = 42
    cfg.step.env.optimistic_reset_ratio = 16
    cfg.step.rollout_steps = 128
    cfg.step.num_epochs = 4
    cfg.step.num_minibatches = 4
    cfg.step.learning_rate = 3e-4
    cfg.step.anneal_learning_rate = True
    cfg.step.discount = 0.99
    cfg.step.trace_decay = 0.5
    cfg.step.epsilon_start = 1.0
    cfg.step.epsilon_finish = 0.005
    cfg.step.epsilon_decay_fraction = 0.1
    cfg.step.max_grad_norm = 0.5
    cfg.step.seed = 42
    cfg.step.model.hidden_size = 512

    cfg.max_steps = cfg.step.total_train_steps = _updates(
        interactions=1_000_000_000,
        num_envs=cfg.step.env.num_envs,
        rollout_steps=cfg.step.rollout_steps,
    )
    cfg.dataset.updates_per_epoch = int(cfg.max_steps)
    cfg.num_steps_eval = cfg.max_steps
    cfg.metrics = dict(parent.metrics)
    cfg.runtime = parent.runtime
    return cfg


def exp011() -> CraftaxTrainLoop:
    """exp001 at a tenth the budget, as a screening workload.

    Only the interaction budget moves, and the learning-rate horizon moves
    with it -- annealing is budget-relative, so a shortened run with exp001's
    horizon would stop while the rate was still high and measure something
    that is not a smaller version of its parent.

    Hypothesis:
      The 1B geometry retains its learning signal and hardware behavior at
      100M, giving a screen short enough to run several treatments in
      parallel. The JAX counterpart scored 8.976% here against its parent's
      11.867%, so the screen ranks recipes without reproducing their scores.

    Not carried over from the JAX exp011:
      Its parent selected a scanned evaluator and a 16-state adaptive reset
      pool. Both were ways to get around XLA's static shapes, and neither has
      anything to work around here: ``lax.scan`` exists to keep an eval loop
      off the host, which a torch loop never leaves, and the fixed reset pool
      approximated a count this environment simply takes (see
      ``CraftaxEnv._restart``). Optimistic reset, the third treatment, is
      carried directly as ``optimistic_reset_ratio``.

    References:
      exp001.

    Results:
      TBD.

    """
    cfg = exp001()
    cfg.experiment_name = "exp011"
    cfg.max_steps = cfg.step.total_train_steps = _updates(
        interactions=100_000_000,
        num_envs=cfg.step.env.num_envs,
        rollout_steps=cfg.step.rollout_steps,
    )
    cfg.dataset.updates_per_epoch = int(cfg.max_steps)
    cfg.num_steps_eval = cfg.max_steps
    return cfg


def exp013() -> CraftaxGTrXLTrainLoop:
    """exp001 with a policy that remembers: gated Transformer-XL at 1B.

    Replaces the feed-forward network with a GTrXL carrying 128 steps of
    attention memory, and the optimization with the windowed, trajectory-major
    form a recurrent policy requires. Rollouts double to 128 steps so the
    memory has something to hold; the discount rises to 0.999 and entropy
    falls to 0.002, both because a policy that can remember is worth pointing
    at rewards further away.

    Hypothesis:
      Craftax's achievements are sequential -- wood, then a table, then a
      pickaxe, then stone -- and a memoryless policy has to rediscover its own
      progress from the tile in front of it. Memory should therefore be worth
      more here than architecture usually is, roughly 18% normalized return
      against exp001's 11.9%.

    Not carried over from the JAX exp013:
      The scanned evaluator and adaptive reset pool, for the reasons exp011
      records: both worked around XLA static shapes that do not exist here.
      Optimistic reset is carried. The architecture change is the whole of
      this fork.

    References:
      https://github.com/Reytuag/transformerXL_PPO_JAX
      Parisotto et al. 2020. Stabilizing transformers for reinforcement
      learning. https://arxiv.org/abs/1910.06764

    Results:
      TBD. The JAX port of this recipe measured 18.159% at seed 42.

    """
    parent = exp001()
    cfg = CraftaxGTrXLTrainLoop()
    cfg.study_name = parent.study_name
    cfg.experiment_name = "exp013"

    cfg.step.env.num_envs = 1_024
    cfg.step.env.seed = 42
    cfg.step.env.optimistic_reset_ratio = 16
    cfg.step.rollout_steps = 128
    cfg.step.gradient_window = 64
    cfg.step.num_epochs = 4
    cfg.step.num_minibatches = 8
    cfg.step.learning_rate = 2e-4
    cfg.step.anneal_learning_rate = True
    cfg.step.discount = 0.999
    cfg.step.trace_decay = 0.8
    cfg.step.clip_epsilon = 0.2
    cfg.step.entropy_coefficient = 0.002
    cfg.step.value_coefficient = 0.5
    cfg.step.max_grad_norm = 1.0
    cfg.step.seed = 42

    cfg.step.model.embed_dim = 256
    cfg.step.model.num_heads = 8
    cfg.step.model.num_layers = 2
    cfg.step.model.qkv_dim = 256
    cfg.step.model.hidden_size = 256
    cfg.step.model.memory_length = 128
    cfg.step.model.gating_bias = 2.0

    cfg.max_steps = cfg.step.total_train_steps = _updates(
        interactions=1_000_000_000,
        num_envs=cfg.step.env.num_envs,
        rollout_steps=cfg.step.rollout_steps,
    )
    cfg.dataset.updates_per_epoch = int(cfg.max_steps)
    cfg.num_steps_eval = cfg.max_steps
    cfg.metrics = dict(parent.metrics)
    cfg.runtime = parent.runtime
    return cfg


def exp_smoke() -> CraftaxTrainLoop:
    """exp000 at minimum size, for verifying an installation end to end.

    Not a result. It answers one question -- does the loop run -- so it is
    cut on every axis that costs time without bearing on that answer. The
    score will be near zero, which is expected.
    """
    cfg = exp000()
    cfg.experiment_name = "exp_smoke"
    cfg.step.env.num_envs = 8
    cfg.step.rollout_steps = 4
    cfg.step.num_minibatches = 2
    cfg.step.model.hidden_size = 32
    cfg.step.model.num_layers = 1
    cfg.max_steps = cfg.step.total_train_steps = 4
    cfg.dataset.updates_per_epoch = 4
    cfg.num_steps_eval = 4
    score = cfg.metrics["craftax"]
    assert isinstance(score, CraftaxScore.Config)
    score.num_envs = 4
    score.steps = 64
    return cfg


def _updates(*, interactions: int, num_envs: int, rollout_steps: int) -> int:
    """Convert an interaction budget into the update count that spends it.

    Floors, because a partial update is not an update: the run stops one
    rollout short of the budget rather than overshooting it.

    Args:
      interactions: Environment steps the run is allowed.
      num_envs: Parallel workers.
      rollout_steps: Steps each worker takes per update.

    Returns:
      updates: Optimizer steps in the run.

    """
    return interactions // (num_envs * rollout_steps)
