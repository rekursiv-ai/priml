r"""Craftax world-model experiments: experimental, not yet officially published.

An alternative approach, not the baseline's result: the main line and its
results are ``experiments``'s. Each policy here reads a world model fitted to
play that ``experiments``' policies captured (``world_model/``): its hidden
state at the current observation, the input to its behaviour-cloning head.
Nothing trains in imagination; every policy gradient comes from real
environment steps. Every experiment forks a named parent and applies ONE
change.

    experiments.exp007    the main line's ConvNeXt-trunk policy, 20B
      +-- exp000     250M, inside the 20B warmup
            +-- exp001     a frozen world model's feature
                  +-- exp002     its random-init weights
    experiments.exp003    the main line's recipe, 20B
      +-- exp003     an early world model, the sole encoder
            +-- exp004     a mature world model
            |     +-- exp006     trained with the policy
            +-- exp005     trained with the policy
            |     +-- exp008     stopped after 1B
            +-- exp007     stopped after 1B

Not yet reproducible: exp003 to exp008 read a world model the original runs
fitted, which no experiment here trains. Their factories raise
``NotImplementedError``; each docstring keeps the record and names the
missing producer (TODO).

Launch::

    uv --quiet run --frozen python -m priml priml.baselines.craftax.experiments_world_model.exp000

"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import math

from configgle import PartialConfig

from priml.baselines.craftax.experiments import (
    CraftaxTrainLoop,
    exp007 as experiments_exp007,
)
from priml.baselines.craftax.model import MinGRUPolicy
from priml.baselines.craftax.train_step import ProgressSchedule
from priml.baselines.craftax.world_model.experiments import (
    exp001 as world_model_exp001,
)
from priml.baselines.craftax.world_model.feature import (
    Flash4CacheAttention,
    InitialWeights,
    TrainedWeights,
    WorldModelFeature,
)
from priml.math.schedules import warmup
from priml.model.linear import Linear
from priml.train.tracker import (
    TrackerList,
    WandbTracker,
    unwrap_tracker_config,
)


if TYPE_CHECKING:
    from torch import nn

    import torch
else:
    from wrapt import lazy_import

    # ~1050 ms; only exp001's zero init and its feature's compile read them.
    nn = lazy_import("torch.nn")
    torch = lazy_import("torch")


def exp000() -> CraftaxTrainLoop:
    """experiments.exp007 for the first 250M of its 20B schedule, evaluated every 50M.

    The control arm of the original runs' frozen world-model experiment:
    experiments.exp007 stopped after 476 epochs, 249,561,088 transitions, while
    its schedule keeps the 20B horizon, so the whole run is the rate's warmup.
    That warmup is the reference's linear ramp from zero, where
    experiments.exp002's ``one_cycle`` ramps by a cosine: immaterial to a 20B
    run, 1.25% of which it spans, but the whole of this one, where the cosine
    would give under half the linear rate through the first fifth and 17% less
    summed rate by 150M. Each evaluation, 10,000 episodes, runs at the first
    epoch past a multiple of 50M transitions; the last is the final one.

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
    cfg = experiments_exp007()
    # A study of its own: these names repeat the recipes', and a run directory or
    # W&B board under one name would mix two experiments' checkpoints and curves.
    cfg.study_name = "craftax-world-model-rl"
    cfg.experiment_name = "exp000"
    trackers = cfg.tracker
    assert isinstance(trackers, TrackerList.Config)
    dashboard = unwrap_tracker_config(trackers.trackers["wandb"])
    assert isinstance(dashboard, WandbTracker.Config)
    dashboard.project = "craftax-world-model-rl"
    dashboard.group = "exp000"
    transitions = cfg.step.env.num_envs * cfg.step.rollout.horizon
    cfg.max_steps = 250_000_000 // transitions
    schedule = cfg.step.schedule
    assert isinstance(schedule, ProgressSchedule.Config)
    schedule.curve = PartialConfig(warmup, fraction=250_000_000 / 20_000_000_000)
    cfg.num_steps_eval = math.ceil(50_000_000 / transitions)
    return cfg


def exp001() -> CraftaxTrainLoop:
    """exp000 reading a frozen world model's state of the episode at every step.

    The treatment arm of that experiment, restated as one fork. Each actor
    step, the world model ``world_model.experiments.exp001``, from its run's
    final checkpoint (step 1,525, trained on the base corpus), reads the
    episode so far -- each observation's frame tokens, each action -- and its
    final-normed hidden state at the current observation, 1,152 floats, is the
    feature
    (:class:`~priml.baselines.craftax.world_model.feature.WorldModelFeature`).
    The recorded run read the original implementation's exp001 at seed 0, in
    the port's layout. Its weights stay frozen. The policy adds a projection
    of the feature, 1,152 -> 2,048 and zero at init, to ``proj_in``'s output,
    so the run starts as exp000's. The rollout stores each step's feature and
    the learner reads the stored one.

    The engine holds at most 1,024 positions of each row's history, two per
    decision; before a row would overflow, it restarts from its last 256
    decisions, as a mid-episode training window. It runs in bf16 with
    FlashAttention 4 over the cache and its per-block functions compiled, as
    its reference's did.

    Hypothesis:
      A world model's state of the episode carries what the recurrent policy
      has not yet learned to keep. Its reference's three seeds averaged 47.1%
      over the evaluations at 150M, 200M and 250M, against 11.9% for exp000's
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
    cfg = exp000()
    cfg.experiment_name = "exp001"
    trackers = cfg.tracker
    assert isinstance(trackers, TrackerList.Config)
    dashboard = unwrap_tracker_config(trackers.trackers["wandb"])
    assert isinstance(dashboard, WandbTracker.Config)
    dashboard.group = "exp001"
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


def exp002() -> CraftaxTrainLoop:
    """exp001 with the world model's weights at their random init.

    The random-weight arm of that experiment: the same engine, tap and
    projection over exp001's untrained model, drawn from seed 0, which is its
    reference's random-weight extract byte for byte. It separates what training put in the feature from what
    any fixed function of the history gives.

    Hypothesis:
      Random weights give nothing a policy can use: its reference's two seeds
      averaged 11.7%, against exp000's reference at 11.9%.

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
    cfg = exp001()
    cfg.experiment_name = "exp002"
    trackers = cfg.tracker
    assert isinstance(trackers, TrackerList.Config)
    dashboard = unwrap_tracker_config(trackers.trackers["wandb"])
    assert isinstance(dashboard, WandbTracker.Config)
    dashboard.group = "exp002"
    feature = cfg.step.feature
    assert isinstance(feature, WorldModelFeature.Config)
    feature.weights = InitialWeights.Config()
    return cfg


def exp003() -> CraftaxTrainLoop:
    """experiments.exp003 with a frozen world model of early play as its only encoder.

    Not yet reproducible: its world model is the original runs' fit to 100M
    decisions of early play, which no experiment here trains. TODO: add the
    capture of an early-training experiments.exp003 policy, its corpus, and
    the world-model experiment that fits it, then read that run's final
    checkpoint; or drop exp003 to exp008.

    The early frozen arm of the sole world-model experiment, on the original
    implementation's engine, restated as one fork. The policy keeps
    experiments.exp003's four MinGRU layers of 1,024, its heads, its
    feasibility loss and its whole recipe, and loses its encoder: no board
    encoder, no ``proj_in`` and no injection. Its trunk reads only a frozen
    world model's state of the episode: each actor step, a model of
    ``world_model.experiments.exp001``'s geometry, fitted to 100M decisions of
    early play, reads the episode so far and taps its final-normed hidden state
    at the current observation, 1,152 floats. A new projection without bias,
    1,152 -> 1,024, drawn as every projection is, maps it into the first
    layer. The observation reaches only the feasibility loss's targets. The
    rollout stores each step's feature; the learner, and the self-imitation
    replaying a branch, read the stored ones.

    The engine is exp001's, the original implementation's: at most 1,024 positions per
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
    environments' streams, not experiments.exp003's 2,048.

    The budget is experiments.exp003's 20B; exp007 is its 1B arm.

    Hypothesis:
      A world model's state, fitted only to predict early play, is by itself
      an observation a policy can learn from: the arm learns, below
      experiments.exp003, whose encoder trains with the policy. On the original
      engine it trains at the original implementation's actor speed, 37-45k
      transitions/s once contexts fill (its reference runs: 37.0k median at a
      context of 384-448 decisions; 52-61k from this config's init), where the
      reference's exact 512-decision windows, recomputed at every decision, ran
      at 483.

    References:
      ``world_model/README.md``, "The frozen feature for RL": the engine and
      its ``Refill`` history.

    Results:
      H200 + 16 CPUs, 20B transitions: 55.32% perf (seed 73), finished, reading
      the original runs' early fit. It reaches the Ice Realm in at most 0.01%
      of episodes at any evaluation. Its 1B arm is exp007's.

      Reference: none. The original runs ended before any evaluation.

    Returns:
      cfg: Configured CraftaxTrainLoop.

    """
    # The early fit has no producer in this chain; the record stays in the
    # docstring above.
    msg = "TODO: exp003 needs a world model fitted to early play; see its docstring."
    raise NotImplementedError(msg)


def exp004() -> CraftaxTrainLoop:
    """exp003 with the world model fitted to mature play.

    Not yet reproducible: its world model is the original runs' fit to 100M
    decisions of mature play, which no experiment here trains. TODO: add the
    capture of a 20B experiments.exp003 policy's play, its corpus, and the
    world-model experiment that fits it, then read that run's final
    checkpoint; or drop exp004 and exp006.

    The mature frozen arm: the same model, engine and policy, the weights
    fitted to 100M decisions of the strongest mature control's play, whose
    episodes reach the late floors.

    Hypothesis:
      A world model that has seen the late floors gives the policy a state of
      them: exp004 scores above exp003.

    References:
      exp003's.

    Results:
      H200 + 16 CPUs, 20B transitions: still training at seed 73; 69.91% perf at
      its latest evaluation, 15.5B, and 63.16% at 2B, reading the original runs'
      mature fit. It first scores 50%, 60% and 65% at 0.75B, 1.5B and 2.75B,
      where the eight original runs of experiments.exp003's recipe take a
      median 3.5B, 7.5B and 9.9B. Each step costs about 11 times
      experiments.exp003's.

      Reference: none. The original runs ended before any evaluation.

    Returns:
      cfg: Configured CraftaxTrainLoop.

    """
    # The mature fit has no producer in this chain; the record stays in the
    # docstring above.
    msg = "TODO: exp004 needs a world model fitted to mature play; see its docstring."
    raise NotImplementedError(msg)


def exp005() -> CraftaxTrainLoop:
    """exp003 training its world model with the policy.

    Not yet reproducible: it builds once exp003 does (TODO there).

    The early joint arm, on exp003's engine, restated as one fork. The learner
    trains a copy of the world model's feature weights -- its frame table and
    encoder, ``obs_proj``, the action table, ``start``, the 20 global blocks
    and the final norm -- with the policy, as the reference's joint arms
    registered them: one optimizer group after the policy's, under
    experiments.exp003's Muon, schedule and global clip, each fused QKV
    projection orthogonalized as one matrix. Each window recomputes its
    features from the contexts the actor read, with the copy's current weights
    (:mod:`~priml.baselines.craftax.world_model.context`): the steps whose
    contexts begin at the same decision share one causal pass, and the
    features' gradient reaches the weights in a second pass. The feature
    turns joint with it: each rollout stores every step's frame tokens, previous
    action and context, and each row's last 511 decisions before it; before
    each rollout the copy is published into the actor's model and every row's
    history rebuilt under it. The replay runs FlashAttention 4, its per-block
    and encoder functions compiled. The self-imitation still reads the stored
    features, so its loss trains the policy alone.

    On an H200, the learner's epoch at experiments.exp003's geometry took 88 s
    on contexts of 257 to 512 decisions, where the reference's, recomputing a
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
      policy needs: exp005 scores above exp003, its frozen counterpart.

    References:
      ``world_model/README.md``, "The frozen feature for RL" (``joint``).

    Results:
      H200 + 16 CPUs, 20B transitions: still training at seed 73; 67.31% perf at
      its latest evaluation, 3.25B, and 54.16% at 1.25B. Its 1B arm is exp008's.

      Reference: none. The original runs ended before any evaluation.

    Returns:
      cfg: Configured CraftaxTrainLoop.

    """
    # Its parent raises until its producer lands; the record stays in the
    # docstring above.
    msg = "TODO: exp005 builds once exp003 does; see exp003's docstring."
    raise NotImplementedError(msg)


def exp006() -> CraftaxTrainLoop:
    """exp004 training its world model with the policy.

    Not yet reproducible: it builds once exp004 does (TODO there).

    The mature joint arm, on exp004's engine: exp005's change, applied to the
    mature world model.

    Hypothesis:
      As exp005's: exp006 scores above exp004, its frozen counterpart.

    References:
      exp005's.

    Results:
      H200 + 16 CPUs, 20B transitions: still training at seed 73; 65.20% perf at
      its latest evaluation, 3.25B, and 60.31% at 1.25B.

      Reference: none. The original runs ended before any evaluation.

    Returns:
      cfg: Configured CraftaxTrainLoop.

    """
    # Its parent raises until its producer lands; the record stays in the
    # docstring above.
    msg = "TODO: exp006 builds once exp004 does; see exp004's docstring."
    raise NotImplementedError(msg)


def exp007() -> CraftaxTrainLoop:
    """exp003 stopped after 1B transitions, its schedule keeping the 20B horizon.

    Not yet reproducible: it builds once exp003 does (TODO there).

    The reference's 1B arm of the early frozen world model: the run stops
    after epoch 1,907, 999,817,216 transitions, while the schedule keeps
    exp003's 20B horizon, as the reference's did, so the rate is still near
    its peak; it evaluates just past 250M, 500M and 750M and at the end.

    Hypothesis:
      A frozen world model's state of early play is an observation a policy
      learns from within 1B transitions.

    References:
      exp003's.

    Results:
      H200 + 16 CPUs, 999,817,216 transitions: 37.51% perf (seed 73), reading
      the original runs' early fit.

      Reference: none. The original runs ended before any evaluation.

    Returns:
      cfg: Configured CraftaxTrainLoop.

    """
    # Its parent raises until its producer lands; the record stays in the
    # docstring above.
    msg = "TODO: exp007 builds once exp003 does; see exp003's docstring."
    raise NotImplementedError(msg)


def exp008() -> CraftaxTrainLoop:
    """exp005 stopped after 1B transitions, its schedule keeping the 20B horizon.

    Not yet reproducible: it builds once exp003 does (TODO there).

    The reference's 1B arm of the early joint world model: exp007's budget
    applied to exp005.

    Hypothesis:
      Training the world model with the policy adapts its state to what the
      policy needs: exp008 scores above exp007, its frozen counterpart.

    References:
      exp005's.

    Results:
      H200 + 16 CPUs, 999,817,216 transitions: 52.10% perf (seed 73),
      against exp007's 37.51%, from the original runs' early fit.

      Reference: none. The original runs ended before any evaluation.

    Returns:
      cfg: Configured CraftaxTrainLoop.

    """
    # Its parent raises until its producer lands; the record stays in the
    # docstring above.
    msg = "TODO: exp008 builds once exp003 does; see exp003's docstring."
    raise NotImplementedError(msg)
