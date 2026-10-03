"""ARC2 reference TRM recipe assembled from injectable priml components."""

from __future__ import annotations

from dataclasses import field
from typing import Final, Self, override

from configgle import Makes

from priml.baselines.arcagi1 import experiments
from priml.baselines.arcagi1.metric import (
    CanonicalPassK,
    PerOutputPass,
    SignalDumpTracker,
    StrictPass,
)
from priml.baselines.arcagi1.model import from_reference_name
from priml.baselines.arcagi1.train_step import TrmTrainStep
from priml.baselines.arcagi2.data import Arc2Data
from priml.baselines.arcagi2.metric import PassK
from priml.baselines.arcagi2.model import PuzzleEmbedding, RotaryBlock
from priml.baselines.arcagi2.puzzle_data import Arc2PuzzleDataset
from priml.baselines.arcagi2.scripts.build_dataset import (
    arc2_aug_policy_template,
    arc2_spatial_eval_template,
)
from priml.baselines.arcagi2.train_step import ArcDataParallel, ArcTrainStep
from priml.baselines.arcagi2.warm_start import WarmStart
from priml.baselines.sudoku.act import AtomicPool
from priml.baselines.sudoku.embedding import GridEmbedding
from priml.baselines.sudoku.model import DeepRecurrence
from priml.baselines.sudoku.prefix import SparsePuzzleEmbedding
from priml.model.attention.attention import Attention
from priml.model.attention.rope import RoPE
from priml.model.swiglu import SwiGLU
from priml.runtime import MultiProcess, SingleProcess
from priml.train.checkpointer import Checkpointer
from priml.train.parallelism import NoParallel
from priml.train.tracker import TrackerList
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

    @override
    def finalize(self) -> Self:
        self.dataset.spec.finalize()
        spec = self.dataset.spec
        model = self.step.model
        model.vocab_size = spec.vocab_size
        embedding = model.embedding
        assert isinstance(embedding, GridEmbedding.Config)
        embedding.grid_shape = spec.grid_shape
        if model.rope is not None:
            model.rope_grid_shape = spec.grid_shape
        return super().finalize()


def exp000() -> ArcTrainLoop:
    """Return the full ARC2 reference TRM recipe.

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
    assert isinstance(model.block.attn, Attention.Config)
    assert isinstance(model.block.ffn, SwiGLU.Config)
    assert isinstance(model.recurrence, DeepRecurrence.Config)
    assert isinstance(model.prefix, PuzzleEmbedding.Config)
    assert isinstance(model.block.rope, RoPE.Config)
    model.channels_in = 32
    model.block.attn.num_heads = 2
    model.block.attn.channels_head = 16
    model.block.rope.channels_head = 16
    model.block.ffn.round_to = 32
    model.recurrence.slow_cycles = 1
    model.recurrence.fast_cycles = 1
    model.prefix.batch_size = 2
    assert isinstance(config.step.pool, AtomicPool.Config)
    config.step.pool.batch_size = 2
    config.step.pool.max_steps = 4
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


NUM_PUZZLE_IDENTIFIERS: Final = 1_191_727
"""Distinct puzzle ids in ``arc2concept-aug-1000``: 1,191,726 puzzles plus blank.

A property of the prepared dataset, not a tunable: the per-task table needs a
row for each id the build assigned."""

TOTAL_TRAIN_STEPS: Final = 541_580
"""The reference sample scale -- 200,000 epochs of 1,280 groups at 4.33 examples
each over a global batch of 2,048 -- rounded up."""

LEAKED_TASKS: Final = (
    "0934a4d8",
    "136b0064",
    "16b78196",
    "981571dc",
    "aa4ec2a5",
    "da515329",
)
"""``evaluation2`` tasks that re-partition ARC-AGI-1 evaluation tasks.

A model initialized from ARC-AGI-1 training has seen their pairs, so its score
on them measures memory, not reasoning."""


class Arc2TrmTrainLoop(
    Makes["TrainLoop"],
    TrainLoop.Config[TrmTrainStep.Config, Arc2PuzzleDataset.Config],
):
    """The reference TRM step over the ARC-AGI-2 reference-plan loader."""

    step: TrmTrainStep.Config = field(default_factory=TrmTrainStep.Config)
    """Pooled ACT, halt training, and the sparse task table."""

    dataset: Arc2PuzzleDataset.Config = field(
        default_factory=Arc2PuzzleDataset.Config,
    )
    """Rank-sharded Philox sampling of the prepared ARC-AGI-2 tree."""

    @override
    def finalize(self) -> Self:
        self.dataset.spec.finalize()
        spec = self.dataset.spec
        model = self.step.model
        model.vocab_size = spec.vocab_size
        embedding = model.embedding
        assert isinstance(embedding, GridEmbedding.Config)
        embedding.grid_shape = spec.grid_shape
        if model.rope is not None:
            model.rope_grid_shape = spec.grid_shape
        return super().finalize()


def exp001() -> Arc2TrmTrainLoop:
    """Train the ARC-AGI-1 reference TRM recipe on ARC-AGI-2.

    exp000's numerics on the pooled-ACT step every later rung shares; that step
    is what the URM, signal, and warm-start rungs need. The recipe is
    ``arcagi1`` exp004 with the task table sized for this build, the horizon
    at the same sample scale, and one fewer shuffled pass per loader
    iteration (1,280 groups fill a 2,048 batch in two, plus margin).

    Hypothesis:
      The reference recipe transfers to the harder benchmark unchanged.

    References:
      https://arxiv.org/abs/2510.04871
        Jolicoeur-Martineau. Less is More: Recursive Reasoning with Tiny
        Networks.

    Results:
      TBD. A published reproduction reached pass@2 0.0333.

    Returns:
      cfg: The reference recipe on ARC-AGI-2.

    """
    return _on_arc2(experiments.exp004(), "exp001")


def exp002() -> Arc2TrmTrainLoop:
    """exp001 + the modified SwiGLU and a Muon body (``arcagi1`` exp005).

    Hypothesis:
      The faster ARC-AGI-1 convergence carries over to ARC-AGI-2.

    References:
      https://arxiv.org/abs/2601.19085
        Dillon. Speed is Confidence.

    Results:
      TBD.

    Returns:
      cfg: exp001 with the Muon recipe.

    """
    return _on_arc2(experiments.exp005(), "exp002")


def exp003() -> Arc2TrmTrainLoop:
    """exp002 with the Muon rate doubled to 0.01 (``arcagi1`` exp006).

    Hypothesis:
      The gate norm leaves headroom above 5e-3 here as on ARC-AGI-1.

    References:
      https://kellerjordan.github.io/posts/muon/

    Results:
      TBD.

    Returns:
      cfg: exp002 at Muon rate 0.01.

    """
    return _on_arc2(experiments.exp006(), "exp003")


def exp004() -> Arc2TrmTrainLoop:
    """exp002 with the URM model: one latent, ConvSwiGLU blocks, batch 96.

    ``arcagi1`` exp007's model, optimizer, and batch, on the plain tree and
    without evaluation signals; exp005 adds those.

    Hypothesis:
      The URM's gain on ARC-AGI-1 transfers in direction.

    References:
      https://arxiv.org/abs/2512.14693
        Gao et al. Universal Reasoning Model.

    Results:
      TBD.

    Returns:
      cfg: exp002 with the URM model.

    """
    urm = experiments.exp007()
    cfg = exp002()
    cfg.experiment_name = "exp004"
    cfg.step.model = urm.step.model
    cfg.step.optimizer = urm.step.optimizer
    cfg.step.pool = urm.step.pool
    cfg.dataset.batch_size = cfg.dataset.eval_batch_size = urm.dataset.batch_size
    _size_table(cfg)
    return cfg


def exp005() -> Arc2TrmTrainLoop:
    """exp004 + 0.2/0.2 spatial training aug, spatial TTA, and signal dumps.

    ``arcagi1`` exp007 on ARC-AGI-2: the translation and scale aug-policy tree,
    its two-view spatial evaluation expansion, three pass@K slices over one
    evaluation pass, and label-free signals dumped per evaluation.

    Hypothesis:
      Spatial test-time augmentation helps more here, where 40.8% of tasks ask
      for several outputs and the strict rule punishes each miss.

    References:
      https://arxiv.org/abs/2512.14693
        Gao et al. Universal Reasoning Model.

    Results:
      TBD.

    Returns:
      cfg: exp004 with the augmentation and evaluation recipe.

    """
    cfg = _on_arc2(experiments.exp007(), "exp005")
    cfg.dataset.source_dataset_dir = arc2_aug_policy_template(
        translation_prob=0.2,
        scale_prob=0.2,
    )
    cfg.dataset.working_dir = arc2_spatial_eval_template(
        spatial_views=2,
        translation_prob=0.2,
        scale_prob=0.2,
    )
    cfg.dataset.spatial_eval_views = 2
    cfg.dataset.num_puzzle_identifiers = NUM_PUZZLE_IDENTIFIERS
    for metric in cfg.metrics_eval.values():
        assert isinstance(metric, CanonicalPassK.Config)
        metric.working_dir = cfg.dataset.working_dir
    assert isinstance(cfg.checkpointer, Checkpointer.Config)
    assert isinstance(cfg.tracker, TrackerList.Config)
    signals = cfg.tracker.trackers["signals"]
    assert isinstance(signals, SignalDumpTracker.Config)
    signals.keep_last_n = cfg.checkpointer.keep_last_n
    signals.keep_every = cfg.checkpointer.keep_every
    return cfg


def exp006() -> Arc2TrmTrainLoop:
    """exp005 initialized from an ARC-AGI-1 run's body; leakage-labeled.

    The body loads from ``arcagi1`` exp007's final checkpoint; the per-task
    table, sized for a different task vocabulary, stays fresh. Six evaluation
    tasks re-partition ARC-AGI-1 tasks that run trained on, so its full-split
    scores overstate; the ``leakfree`` slice omits them.

    Hypothesis:
      ARC-AGI-1 transformation priors beat exp005's from-scratch curve.

    References:
      https://arxiv.org/abs/2512.14693
        Gao et al. Universal Reasoning Model.

    Results:
      TBD.

    Returns:
      cfg: exp005 with a warm-started body and a leak-free metric slice.

    """
    cfg = exp005()
    cfg.experiment_name = "exp006"
    cfg.step.warm_start = WarmStart.Config(
        path="/opt/scratch/runs/arcagi1/exp028_aug_eval/checkpoints/step_00388670.pt",
        rename=from_reference_name,
    )
    leakfree = cfg.metrics_eval["spatial_big"].copy_tree()
    assert isinstance(leakfree, CanonicalPassK.Config)
    leakfree.exclude_tasks = list(LEAKED_TASKS)
    cfg.metrics_eval["leakfree"] = leakfree
    return cfg


def exp007() -> Arc2TrmTrainLoop:
    """exp006 from a second ARC-AGI-1 run, to ensemble with it; leakage-labeled.

    The source trained with three slow cycles, so the recurrence matches it;
    every tensor then transfers except the per-task table.

    Hypothesis:
      Two bodies with decorrelated errors vote better than either alone.

    References:
      https://arxiv.org/abs/2512.14693
        Gao et al. Universal Reasoning Model.

    Results:
      TBD.

    Returns:
      cfg: exp006 from the second source, at its recurrence.

    """
    cfg = exp006()
    cfg.experiment_name = "exp007"
    assert isinstance(cfg.step.warm_start, WarmStart.Config)
    cfg.step.warm_start.path = (
        "/opt/scratch/runs/arcagi1/exp029/checkpoints/step_00370000.pt"
    )
    assert isinstance(cfg.step.model.recurrence, DeepRecurrence.Config)
    cfg.step.model.recurrence.slow_cycles = 3
    return cfg


# Everything the ARC-AGI-1 recipe states about the model, step, and optimizer is
# kept. Only what the dataset determines moves: its tree, the task-table size, the
# sample-scale horizon, and the scoring rules ARC-AGI-2 reports.
def _on_arc2(
    source: experiments.TrmTrainLoop,
    name: str,
) -> Arc2TrmTrainLoop:
    """Return an ``arcagi1`` TRM recipe retargeted to ARC-AGI-2."""
    cfg = Arc2TrmTrainLoop()
    cfg.update(source, skip_missing=True)
    cfg.step = source.step
    cfg.dataset = Arc2PuzzleDataset.Config().update(source.dataset, skip_missing=True)
    cfg.dataset.working_dir = "/datasets/arc2concept-aug-1000"
    cfg.dataset.epochs_per_iter = 4
    cfg.dataset.num_puzzle_identifiers = NUM_PUZZLE_IDENTIFIERS
    cfg.study_name = "arcagi2"
    cfg.experiment_name = name
    # Every evaluation forwards its signal payload, so a dumping rung writes one
    # per evaluation and any checkpoint can join an ensemble without a re-run.
    cfg.eval_extras_every_eval = True
    cfg.max_steps = cfg.step.total_train_steps = TOTAL_TRAIN_STEPS
    assert isinstance(cfg.checkpointer, Checkpointer.Config)
    cfg.checkpointer.save_every = 5_000
    for metric in cfg.metrics_eval.values():
        assert isinstance(metric, CanonicalPassK.Config)
        metric.working_dir = cfg.dataset.working_dir
        metric.rules = [StrictPass.Config(), PerOutputPass.Config()]
    _size_table(cfg)
    return cfg


def _size_table(cfg: Arc2TrmTrainLoop) -> None:
    """Size the per-task table for ARC-AGI-2 and its row batch for the pool."""
    prefix = cfg.step.model.prefix
    assert isinstance(prefix, SparsePuzzleEmbedding.Config)
    prefix.num_puzzles = NUM_PUZZLE_IDENTIFIERS
    prefix.batch_size = cfg.dataset.batch_size
