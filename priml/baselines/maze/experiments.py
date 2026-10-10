"""The Maze-Hard ladder.

Each maze is a 30x30 grid with a start and a goal; the model marks one
shortest path between them. Scoring is whole-grid exact match on the 1,000 test
mazes, so a valid shortest path other than the stored one counts as wrong (see
:mod:`priml.baselines.maze.data`). Training sees only the 1,000 training
mazes, which is what makes the benchmark hard.

The solver and step are the TRM pieces the ARC and sudoku baselines share; this
baseline adds only the maze's data, its boundary loss, and the recipes. Every
rung trains on one H100 against a fixed time budget, so a change that buys
steps and a change that improves a step are measured in the same number::

    exp000  the reference TRM recipe, one hour
      +-- exp001  canonical labels, the URM recurrence, Muon, boundary loss
            +-- exp002  one deeper pass, five with gradient (the one-hour best)
                  +-- exp003  three with gradient, four hours

Prepare the data once, then launch::

    uv --quiet run --frozen python -m priml.baselines.maze.scripts.prepare_data
    uv --quiet run --frozen python -m priml priml.baselines.maze.experiments.exp002

``exp_smoke`` checks an installation on any device in seconds; it is not scored.
"""

from __future__ import annotations

from dataclasses import field
from typing import TYPE_CHECKING, Self, override

import math

from configgle import Makes, PartialConfig

from priml.baselines.arcagi.loss import MeanOverBatch, StablemaxTokens
from priml.baselines.arcagi.model import (
    ConvSwiGLU,
    SingleLatentRecurrence,
    depthwise_shift,
)
from priml.baselines.arcagi.optimizer import with_ndim
from priml.baselines.maze.data import (
    CANONICAL,
    ORIGINAL,
    VOCAB_SIZE,
    MazeData,
)
from priml.baselines.maze.loss import CanonicalBoundary
from priml.baselines.maze.train_step import BoundaryLossTrainStep
from priml.baselines.sudoku.act import (
    AtomicPool,
    HaltTraining,
    SampledMinimum,
)
from priml.baselines.sudoku.embedding import GridEmbedding
from priml.baselines.sudoku.model import CoreCompile, DeepRecurrence
from priml.baselines.sudoku.prefix import SparsePuzzleEmbedding
from priml.baselines.sudoku.puzzle_spec import SudokuSpec
from priml.model.attention.attention import Attention
from priml.model.attention.rope import HuggingFaceFrequencies, RoPE, RoPEMixed
from priml.model.init import corrected_fan_in_normal
from priml.model.norm import RMSNorm
from priml.model.swiglu import SwiGLU
from priml.model.transformer.block import TransformerBlock
from priml.optimizers import AdamATan2, SignSGD
from priml.optimizers.composite import CompositeOptimizer
from priml.optimizers.muon import Muon
from priml.optimizers.parameter_filter import complement, excluding
from priml.runtime import SingleProcess
from priml.train.checkpointer import Checkpointer
from priml.train.ema import EMA
from priml.train.parallelism import NoParallel
from priml.train.tracker import FileTracker, TrackerList, WandbTracker
from priml.train.train_loop import TrainLoop


if TYPE_CHECKING:
    import torch
else:
    from wrapt import lazy_import

    torch = lazy_import("torch")


class MazeTrainLoop(
    Makes["TrainLoop"],
    TrainLoop.Config[BoundaryLossTrainStep.Config, MazeData.Config],
):
    """A training loop with the maze step and dataset already in place.

    Narrowing both slots here lets a factory reach ``cfg.step.model`` with no
    ``isinstance`` before a field it is about to set.
    """

    step: BoundaryLossTrainStep.Config = field(
        default_factory=BoundaryLossTrainStep.Config,
    )
    """The reference TRM step, with the optional boundary loss."""

    dataset: MazeData.Config = field(default_factory=MazeData.Config)
    """Prepared maze grids, held on the device."""

    @override
    def finalize(self) -> Self:
        # The dataset owns the geometry; the model, the boundary mask, and the
        # rotary lattice are laid out on it.
        spec = self.dataset.spec
        model = self.step.model
        model.vocab_size = spec.vocab_size
        embedding = model.embedding
        assert isinstance(embedding, GridEmbedding.Config)
        embedding.grid_shape = spec.grid_shape
        if model.rope is not None and not model.rope_grid_shape:
            model.rope_grid_shape = spec.grid_shape
        boundary = self.step.boundary
        if boundary is not None and boundary.grid_shape == (0, 0):
            boundary.grid_shape = spec.grid_shape
        # One batch size, three buffers: a pool seats a whole loader batch, and
        # the prefix's gradient buffer holds one row per pool slot.
        pool = self.step.pool
        self.dataset.batch_size = pool.batch_size
        if isinstance(model.prefix, SparsePuzzleEmbedding.Config):
            model.prefix.batch_size = pool.batch_size
        return super().finalize()


def exp000() -> MazeTrainLoop:
    """Return the reference recipe.

    - Two post-norm transformer blocks, width 512, eight heads; 2-D rotary
      positions with per-head mixed frequencies over the 30x30 grid.
    - Three slow cycles of four fast passes per forward; atomic ACT up to 16
      steps, the halt head trained at weight 0.5 with 10% sampled-minimum
      exploration.
    - A 16-token prefix from one shared, zero-initialized row, trained by
      sparse SignSGD at 1e-4 with weight decay 1.0.
    - AdamATan2 at 1e-4, betas (0.9, 0.95), weight decay 1.0; 2,000 warmup
      steps, then flat.
    - Stablemax token loss, summed per maze over the pool of 128.
    - EMA 0.999 from the first step; float32 weights under bfloat16 autocast.
    - Eight dihedral views of each training maze, the stored label moved along.
    - 16,000 steps or one training hour, whichever binds first; all 1,000 test
      mazes evaluated every 500 steps.

    Hypothesis:
      The published Maze-Hard TRM recipe, with positions laid out on the 2-D
      grid, is the bar a one-hour recipe must clear.

    References:
      https://arxiv.org/abs/2510.04871
        Jolicoeur-Martineau. Less is More: Recursive Reasoning with Tiny
        Networks.

    Results:
      H100, 1 h: 0.007 exact (7/1,000) at its 16,000-step ceiling, 2,646.7
      training seconds.
      H200, 1 h: 0.000 final at the 16,000-step ceiling; best 0.003 at step
      11,500.

    """
    cfg = MazeTrainLoop()
    cfg.study_name = "maze"
    cfg.experiment_name = "exp000"
    cfg.seed = 0

    model = cfg.step.model
    block = model.block = TransformerBlock.Config()
    block.prenorm = False
    norm1 = block.norm1 = RMSNorm.Config()
    norm1.eps = 1e-5
    norm2 = block.norm2 = RMSNorm.Config()
    norm2.eps = 1e-5
    attn = block.attn = Attention.Config()
    attn.channels_head = 64
    attn.init_weight = corrected_fan_in_normal
    ffn = block.ffn = SwiGLU.Config()
    ffn.expansion = 8 / 3
    ffn.init_weight = corrected_fan_in_normal
    ffn.init_weight_out = corrected_fan_in_normal
    recurrence = model.recurrence = DeepRecurrence.Config()
    recurrence.slow_cycles = 3
    recurrence.fast_cycles = 4
    rope = model.rope = RoPEMixed.Config()
    rope.channels_head = 64
    rope.num_heads = 8
    rope.frequencies = [
        HuggingFaceFrequencies.Config(),
        HuggingFaceFrequencies.Config(),
    ]
    rope.reduction_mode = "sum"
    # Rounded to bfloat16 on lookup and scaled in float32, as the token
    # embedding it is concatenated with is.
    prefix = model.prefix = SparsePuzzleEmbedding.Config()
    prefix.dtype = torch.bfloat16
    prefix.dtype_scale = torch.float32
    model.compile_core = CoreCompile.Config()

    step = cfg.step
    adam = step.optimizer = AdamATan2.Config()
    adam.lr = 1e-4
    adam.betas = (0.9, 0.95)
    adam.weight_decay = 1.0
    sign_sgd = step.sparse_optimizer = SignSGD.Config()
    sign_sgd.lr = 1e-4
    sign_sgd.weight_decay = 1.0
    step.token_loss = StablemaxTokens.Config()
    step.reduction = MeanOverBatch.Config()
    step.ignore_label_id = 0
    pool = step.pool = AtomicPool.Config()
    pool.batch_size = 128
    halting = pool.halting = HaltTraining.Config()
    halting.weight = 0.5
    halting.exploration = SampledMinimum.Config()
    ema = step.ema = EMA.Config()
    ema.decay = 0.999
    ema.warmup_seed = True
    ema.track_buffers = False
    ema.shadow_kind = "param_dict"
    step.warmup_steps = 2_000
    step.lr_min_ratio = 1.0
    parallelism = step.parallelism = NoParallel.Config()
    parallelism.device = "cuda"

    spec = cfg.dataset.spec = SudokuSpec()
    spec.grid_shape = (30, 30)
    spec.vocab_size = VOCAB_SIZE
    cfg.dataset.working_dir = f"/datasets/maze/{ORIGINAL}"
    cfg.dataset.seed = 0
    cfg.dataset.device = "cuda"

    cfg.max_steps = step.total_train_steps = 16_000
    cfg.max_time = 3_600.0
    cfg.max_time_kind = "train"
    cfg.max_eval_time = 3_600.0
    cfg.num_steps_eval = 500
    cfg.eval_every_epoch = False
    cfg.eval_warmup_batches = 1
    checkpointer = cfg.checkpointer
    assert isinstance(checkpointer, Checkpointer.Config)
    checkpointer.save_every = 500
    wandb = WandbTracker.Config()
    wandb.project = "maze"
    cfg.tracker = TrackerList.Config()
    cfg.tracker.trackers = {"wandb": wandb, "metrics": FileTracker.Config()}
    runtime = cfg.runtime = SingleProcess.Config()
    runtime.device = "cuda"
    return cfg


def exp001() -> MazeTrainLoop:
    """Fork exp000 with the following changes.

    - Relabel every training view in its own frame by the stored labels' rule.
    - Replace the two-latent recursion with the Universal Reasoning Model's one
      latent: four blocks, three passes per core application, two applications
      per forward, the first without gradient.
    - Add a kernel-2 depthwise convolution to each block's gated feed-forward.
    - Number positions along the flattened sequence (1-D rotary).
    - Shrink the prefix to one token.
    - Train with AdamW at 1e-4 on tables, heads, and vectors and Muon at 5e-3 on
      block matrices, weight decay 1e-4 / rate; SignSGD keeps the prefix.
    - Add the boundary loss at weight 0.1.
    - Raise the horizon and step ceiling to 24,000.

    The record reached this rung through two record-raising rungs on a
    hierarchical model this baseline does not carry; the bullets are the exact
    configuration difference.

    Hypothesis:
      One label policy removes the conflicting targets augmentation creates, a
      single deeply refined latent with local mixing suits path tracing, and
      Muon trains it faster within the hour.

    References:
      https://arxiv.org/abs/2512.14693
        Gao et al. Universal Reasoning Model.
      https://kellerjordan.github.io/posts/muon/
        Jordan et al. 2024. Muon: an optimizer for hidden layers.

    Results:
      H100, 1 h: 0.940 exact (940/1,000), its best and final, 3,600 training
      seconds.
      H200, 1 h: 0.932 final at step 15,575; best 0.941 at step 12,500.

    """
    cfg = exp000()
    cfg.experiment_name = "exp001"
    cfg.dataset.working_dir = f"/datasets/maze/{CANONICAL}"

    model = cfg.step.model
    model.num_layers = 4
    recurrence = model.recurrence = SingleLatentRecurrence.Config()
    recurrence.slow_cycles = 2
    recurrence.fast_cycles = 3
    block = model.block
    assert isinstance(block, TransformerBlock.Config)
    ffn = block.ffn = ConvSwiGLU.Config()
    ffn.init_weight = corrected_fan_in_normal
    rope = model.rope = RoPE.Config()
    rope.channels_head = 64
    model.rope_grid_shape = (math.prod(cfg.dataset.spec.grid_shape),)
    prefix = model.prefix
    assert isinstance(prefix, SparsePuzzleEmbedding.Config)
    prefix.num_tokens = 1

    # Every member decays at 1e-4 / lr: the decoupled decay per step,
    # lr * weight_decay, is then 1e-4 at any rate.
    adamw = PartialConfig(torch.optim.AdamW)
    adamw.lr = 1e-4
    adamw.betas = (0.9, 0.95)
    adamw.weight_decay = 1.0
    muon = Muon.Config()
    muon.lr = 5e-3
    muon.weight_decay = 1e-4 / muon.lr
    muon.momentum = 0.6
    muon.ns_steps = 3
    # Muon folds a rank-3 weight's leading axis into an ensemble of matrices.
    muon_ensemble = muon.copy_tree()
    muon_ensemble.ensemble_dims = 1
    optimizer = cfg.step.optimizer = CompositeOptimizer.Config()
    optimizer.optimizers = [adamw, muon, muon_ensemble]
    on_muon = excluding(Muon.eligible_tensor, "embed", "head", "register_tokens")
    optimizer.select = [
        complement(on_muon),
        with_ndim(on_muon, 2),
        with_ndim(on_muon, 3),
    ]
    optimizer.drop_empty = True
    cfg.step.boundary = CanonicalBoundary.Config()
    cfg.max_steps = cfg.step.total_train_steps = 24_000
    return cfg


def exp002() -> MazeTrainLoop:
    """Fork exp001 with the following changes.

    - Run one core application of eight passes per forward; the first three
      without gradient, the last five with it.
    - Evaluate the convolution as shifted elementwise taps.

    Hypothesis:
      One long refinement with a five-pass credit path learns more per second
      than two short ones, and the fused taps buy steps.

    References:
      https://arxiv.org/abs/2512.14693
        Gao et al. Universal Reasoning Model.

    Results:
      H100, 1 h: 0.949 exact (949/1,000), its best and final, 3,600.1 training
      seconds. All seeds at 1: best 0.939, final 0.937.
      H200, 1 h: 0.945 final at step 13,678; best 0.952 at step 12,500.

    """
    cfg = exp001()
    cfg.experiment_name = "exp002"
    model = cfg.step.model
    recurrence = model.recurrence
    assert isinstance(recurrence, SingleLatentRecurrence.Config)
    recurrence.slow_cycles = 1
    recurrence.fast_cycles = 8
    recurrence.inner_grad_loops = 5
    block = model.block
    assert isinstance(block, TransformerBlock.Config)
    assert isinstance(block.ffn, ConvSwiGLU.Config)
    block.ffn.short_conv = depthwise_shift
    return cfg


def exp003() -> MazeTrainLoop:
    """Fork exp002 with the following changes.

    - Carry gradient through the last three of the eight passes.
    - Train for four hours: 14,400 seconds, horizon and step ceiling 80,000.

    Hypothesis:
      A shorter credit path buys more steps, and four hours of them at a flat
      rate keep improving the final checkpoint.

    References:
      https://arxiv.org/abs/2512.14693
        Gao et al. Universal Reasoning Model.

    Results:
      H100, 4 h: 0.952 exact (952/1,000) final, 14,400.1 training seconds;
      best 0.957 at step 58,500.

    """
    cfg = exp002()
    cfg.experiment_name = "exp003"
    recurrence = cfg.step.model.recurrence
    assert isinstance(recurrence, SingleLatentRecurrence.Config)
    recurrence.inner_grad_loops = 3
    cfg.max_time = 14_400.0
    cfg.max_steps = cfg.step.total_train_steps = 80_000
    return cfg


def exp_smoke() -> MazeTrainLoop:
    """Fork exp002 for an unscored installation check.

    - Automatically select the available device; compile nothing.
    - Reduce width to 32 with two heads, one block, and one cycle of two passes.
    - Seat four mazes per step; run two ACT steps.
    - Stop after four steps; evaluate eight test mazes every two.
    - Write metrics locally; save no checkpoint.
    """
    cfg = exp002()
    cfg.experiment_name = "exp_smoke"
    parallelism = cfg.step.parallelism
    assert isinstance(parallelism, NoParallel.Config)
    parallelism.device = None
    cfg.dataset.device = None
    cfg.runtime = SingleProcess.Config()
    model = cfg.step.model
    model.channels_in = 32
    model.num_layers = 1
    model.compile_core = None
    recurrence = model.recurrence
    assert isinstance(recurrence, SingleLatentRecurrence.Config)
    recurrence.fast_cycles = 2
    recurrence.inner_grad_loops = 1
    rope = model.rope
    assert isinstance(rope, RoPE.Config)
    rope.channels_head = 16
    block = model.block
    assert isinstance(block, TransformerBlock.Config)
    assert isinstance(block.attn, Attention.Config)
    block.attn.channels_head = 16
    block.attn.num_heads = 2
    cfg.step.pool.batch_size = 4
    cfg.step.pool.max_steps = 2
    cfg.dataset.eval_num_instances = 8
    cfg.max_steps = 4
    cfg.num_steps_eval = 2
    cfg.checkpointer = None
    cfg.tracker = FileTracker.Config()
    return cfg
