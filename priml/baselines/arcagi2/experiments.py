"""ARC2 reference TRM recipe assembled from injectable priml components."""

from __future__ import annotations

from dataclasses import field

from configgle import Makes

from priml.baselines.arcagi2.data import Arc2Data
from priml.baselines.arcagi2.metric import PassK
from priml.baselines.arcagi2.model import PuzzleEmbedding, RotaryBlock
from priml.baselines.arcagi2.train_step import ArcDataParallel, ArcTrainStep
from priml.baselines.sudoku.act import ActPool
from priml.baselines.sudoku.model import DeepRecurrence
from priml.model.attention.self_attention import SelfAttention
from priml.model.swiglu import SwiGLU
from priml.runtime import MultiProcess, SingleProcess
from priml.train.checkpointer import Checkpointer
from priml.train.parallelism import NoParallel
from priml.train.train_loop import TrainLoop


class ArcTrainLoop(
    Makes["TrainLoop"],
    TrainLoop.Config[ArcTrainStep.Config, Arc2Data.Config],
):
    """ARC2 model, source sampling, and training objective."""

    step: ArcTrainStep.Config = field(default_factory=ArcTrainStep.Config)
    """Atomic ACT, stablemax, AdamATan2 and sparse SignSGD."""

    dataset: Arc2Data.Config = field(default_factory=Arc2Data.Config)
    """Prepared ARC2 tasks, four shuffled passes per loader iteration."""


def exp000() -> ArcTrainLoop:
    """Return the full ARC2 reference recipe, corresponding to experimental exp001.

    Hypothesis:
      The shared priml building blocks reproduce the reference TRM exactly.

    References:
      https://arxiv.org/abs/2510.04871
        Jolicoeur-Martineau. Less is More: Recursive Reasoning with Tiny Networks.

    Results:
      No benchmark run yet.

    Returns:
      config: Full-width ARC2 TRM recipe.

    """
    config = ArcTrainLoop()
    config.study_name = "arcagi2"
    config.experiment_name = "exp000"
    config.seed = 0
    config.metrics_eval[""] = PassK.Config()
    config.max_steps = config.step.total_train_steps
    config.num_steps_eval = 10_000
    config.num_steps_log = 100
    config.eval_warmup_batches = 1
    config.eval_every_epoch = False
    config.checkpointer = Checkpointer.Config(
        save_every=5_000,
        keep_last_n=8,
        keep_every=40_000,
    )
    config.runtime = MultiProcess.Config()
    config.runtime.mesh_topology = {"dp": -1, "pp": 1, "tp": 1}
    config.step.parallelism = ArcDataParallel.Config()
    config.step.parallelism.gradient_as_bucket_view = True
    return config


def exp_smoke() -> ArcTrainLoop:
    """Return a small prepared-data installation check, not a benchmark result."""
    config = exp000()
    config.experiment_name = "exp_smoke"
    config.runtime = SingleProcess.Config()
    config.step.parallelism = NoParallel.Config()
    model = config.step.model
    assert isinstance(model.block, RotaryBlock.Config)
    assert isinstance(model.block.attn, SelfAttention.Config)
    assert isinstance(model.block.ffn, SwiGLU.Config)
    assert isinstance(model.recurrence, DeepRecurrence.Config)
    assert isinstance(model.prefix, PuzzleEmbedding.Config)
    model.channels_in = 32
    model.block.attn.num_heads = 2
    model.block.attn.channels_head = 16
    model.block.rope.channels_head = 16
    model.block.ffn.round_to = 32
    model.recurrence.slow_cycles = 1
    model.recurrence.fast_cycles = 1
    model.prefix.batch_size = 2
    assert isinstance(config.step.act, ActPool.Config)
    config.step.act.batch_size = 2
    config.step.act.max_steps = 4
    config.step.total_train_steps = config.max_steps = 4
    config.step.warmup_steps = 0
    config.step.use_ema = False
    config.step.compile = None
    config.dataset.batch_size = 2
    config.dataset.eval_batch_size = 2
    config.dataset.num_tasks = 4
    config.dataset.num_eval_tasks = 4
    config.num_steps_eval = 2
    config.checkpointer = None
    return config
