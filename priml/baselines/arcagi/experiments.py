"""The ARC-AGI experiment ladder.

Each ARC task shows a few input/output grid pairs demonstrating a rule, then
asks for the rule applied to a held-out input. Nothing about the rule is
labelled, so the model must infer it from the examples -- which is why this is
a reasoning benchmark rather than a perception one.

The solver is the sudoku baseline's, with different values in its slots: a
30x30 grid instead of 9x9, twelve colors instead of ten digits, and a learned
per-task table sudoku has no use for. Two properties drive the rest:

* **Whole-grid scoring.** One wrong cell fails the puzzle.
* **Heavy augmentation.** Each task is stored many times under recolorings and
  dihedral transforms, so the answer is the consensus across those views
  (pass@K) rather than any single pass.

``exp000`` reproduces the reference TRM recipe and is never edited: it is the
number every other rung is measured against. The ladder through ``exp006``
follows the best recipe at each step. CHARM's coupled source recipes branch
from ``exp003``::

    exp000  reference TRM                                       46.0
      +-- exp001  Muon body + gate-normalized SwiGLU            58.125
            +-- exp002  Muon rate 0.01                          59.5
                  +-- exp003  URM: one latent, conv FFN         67.625
                        +-- exp004  shift conv, compiled eval   68.375
                        |     +-- exp005  spatial aug + TTA     70.125
                        |           +-- exp006  QK-norm, repair 71.375
                        +-- exp009  CHARM source recipe, ARC1   TBD
                              +-- exp010  CHARM source recipe, ARC2 TBD

``exp009`` extends the URM-based ``exp003`` configuration with CHARM's gated
ARC-AGI-1 source recipe. ``exp010`` extends ``exp009`` with its ARC-AGI-2 data,
memory, and horizon changes. The source settings are coupled recipes, not
one-axis ablations.

ARC-AGI-1 numbers are pass@2 under rule B (tasks solved of the 400 public
evaluation tasks), each on one 8xH100 node. ``exp007`` and ``exp008`` train
``exp000``'s and ``exp005``'s recipes on ARC-AGI-2 and report its official
strict rule. ``exp_smoke`` checks an installation.

Prepare the data once, then launch (8 GPUs)::

    uv --quiet run --frozen python -m priml.baselines.arcagi.scripts.prepare_data --experiment exp006
    uv --quiet run --frozen python -m torch.distributed.run --standalone --nproc_per_node=8 -m priml priml.baselines.arcagi.experiments.exp006

The native CHARM replications read the released parquet files beneath
``/opt/scratch/datasets/charm-release-937316c8/data/augmented/v2``. Launch
``exp009`` and ``exp010`` on eight GPUs with::

    uv --quiet run --frozen python -m torch.distributed.run --standalone --nproc_per_node=8 -m priml priml.baselines.arcagi.experiments.exp009
    uv --quiet run --frozen python -m torch.distributed.run --standalone --nproc_per_node=8 -m priml priml.baselines.arcagi.experiments.exp010
"""

from __future__ import annotations

from dataclasses import field
from functools import partial
from typing import TYPE_CHECKING, Final, Self, cast, override

from configgle import Makes, PartialConfig

from priml.baselines.arcagi.checkpoint import CheckpointActTrainStep
from priml.baselines.arcagi.data import ArcParquetData, PairSource, PuzzleData
from priml.baselines.arcagi.metric import (
    CanonicalPassK,
    ParquetPassK,
    PerOutputPass,
    StrictPass,
)
from priml.baselines.arcagi.model import (
    ConvSwiGLU,
    PuzzleEmbedding,
    RotaryBlock,
    depthwise_shift,
)
from priml.baselines.arcagi.puzzle_data import Arc2PuzzleDataset
from priml.baselines.arcagi.signal_dump import (
    SignalDumpTracker,
)
from priml.baselines.arcagi.train_step import (
    ArcDataParallel,
    BeforeBackwardPoolTransition,
    EvalSignals,
    PooledActTrainStep,
    apply_direct_schedule_to_body_and_sparse_optimizer_lrs,
)
from priml.baselines.sudoku.act import (
    AtomicPool,
    CellCorruption,
    FeedbackCarry,
    GlobalSampledMinimum,
    HaltTraining,
    SampledMinimum,
)
from priml.loss.act import ACTLoss
from priml.loss.token import MeanOverBatch, StablemaxTokens
from priml.math.linear import cast_parameters_to_input
from priml.math.loss import log_stablemax_via_probabilities
from priml.math.numeric import matrix_signum_via_coefficients
from priml.model.attention.attention import Attention
from priml.model.attention.rope import RoPE
from priml.model.conditioning import (
    CategorySlotFiLM,
    SigmoidResidualGate,
)
from priml.model.conv import SequenceConv1d
from priml.model.embedding import Embedding
from priml.model.grid_embedding import (
    GridEmbedding,
    PredictionFeedback,
)
from priml.model.grid_solver import (
    CoreCompile,
    GridSolver,
    SingleLatentGridSolver,
)
from priml.model.init import corrected_fan_in_normal
from priml.model.linear import Linear
from priml.model.norm import RMSNorm
from priml.model.prefix import ComposedSparsePrefix, SparsePuzzleEmbedding
from priml.model.recurrence import DeepRecurrence, SingleLatentRecurrence
from priml.model.swiglu import SwiGLU
from priml.model.transformer.block import TransformerBlock
from priml.optimizers import AdamATan2, AuxAdam
from priml.optimizers.composite import CompositeOptimizer
from priml.optimizers.muon import (
    Muon,
    adjust_lr_match_rms_adamw,
    momentum_add_update,
)
from priml.optimizers.parameter_filter import complement, excluding, with_ndim
from priml.runtime import MultiProcess, SingleProcess
from priml.train.checkpointer import Checkpointer
from priml.train.ema import EMA
from priml.train.parallelism import ManualDataParallel, NoParallel, SumGradients
from priml.train.tracker import FileTracker, TrackerList, WandbTracker
from priml.train.train_loop import TrainLoop


if TYPE_CHECKING:
    from collections.abc import Mapping

    from torch import Tensor

    import torch
else:
    from wrapt import lazy_import

    torch = lazy_import("torch")


NUM_PUZZLE_IDENTIFIERS: Final = 876_403
"""Distinct puzzle ids in ``arc1concept-aug-1000``: 876,402 puzzles plus blank.

A property of the prepared dataset, not a tunable. The build assigns ids once
across every split, so the largest is the puzzle count and the per-task table
needs a row for each -- a shorter table indexes off the end on the first batch
rather than training a smaller model. The dataset cannot supply this: a config
must build with no data on disk, so ``finalize`` may not read the tree."""

ARC2_NUM_PUZZLE_IDENTIFIERS: Final = 1_191_727
"""Distinct puzzle ids in ``arc2concept-aug-1000``: 1,191,726 puzzles plus blank."""

_SOURCE_DEPTHWISE_KERNEL_2_BIAS_INIT: Final = partial(
    torch.nn.init.uniform_,
    a=-(2**-0.5),
    b=2**-0.5,
)
"""Initialize the source's kernel-two depthwise bias in FP32 before casting."""

_SOURCE_MATRIX_SIGNUM_FN: Final = partial(
    matrix_signum_via_coefficients,
    coefficients=(
        (7.2086, -15.5131, 9.0178),
        (3.9623, -2.5813, 0.4542),
        (3.9466, -2.5765, 0.4544),
        (3.8991, -2.5671, 0.4566),
        (3.7186, -2.5308, 0.4653),
        (3.1390, -2.3073, 0.4733),
        (2.1715, -1.5246, 0.3885),
        (1.8648, -1.2224, 0.3577),
    ),
    steps=5,
    eps=1e-7,
    compute_dtype=torch.float32,
)
"""The fixed source polynomial matrix sign for Muon."""

ARC2_TOTAL_TRAIN_STEPS: Final = 541_580
"""The reference sample scale on ARC-AGI-2 -- 200,000 epochs of 1,280 groups at
4.33 examples each over a global batch of 2,048 -- rounded up."""


class Arc1TrainLoop(
    Makes["TrainLoop"],
    TrainLoop.Config[PooledActTrainStep.Config, PuzzleData.Config],
):
    """The reference TRM step over the ARC-AGI-1 loader."""

    step: PooledActTrainStep.Config = field(default_factory=PooledActTrainStep.Config)
    """Pooled ACT, halt training, and the sparse task table."""

    dataset: PuzzleData.Config = field(default_factory=PuzzleData.Config)
    """Rank-sharded Philox sampling of the prepared ARC tree."""

    @override
    def finalize(self) -> Self:
        self.dataset.spec.finalize()
        spec = self.dataset.spec
        model = self.step.model
        model.vocab_size = spec.vocab_size
        embedding = model.embedding
        assert isinstance(embedding, GridEmbedding.Config)
        embedding.grid_shape = spec.grid_shape
        # A recipe may lay the grid out in 2D for its rotary (e.g. ``(30, 30)``
        # for a 900-token grid); only an unset lattice takes the flat default.
        if model.rope is not None and not model.rope_grid_shape:
            model.rope_grid_shape = spec.grid_shape
        return super().finalize()


class Arc2TrainLoop(
    Makes["TrainLoop"],
    TrainLoop.Config[PooledActTrainStep.Config, Arc2PuzzleDataset.Config],
):
    """The reference TRM step over the ARC-AGI-2 loader."""

    step: PooledActTrainStep.Config = field(default_factory=PooledActTrainStep.Config)
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


class CharmTrainLoop(TrainLoop):
    """Training loop for the released ARC parquet recipes."""

    class Config(
        Makes["CharmTrainLoop"],
        TrainLoop.Config[PooledActTrainStep.Config, ArcParquetData.Config],
    ):
        """Native CHARM step, source parquet mixture, and global batch."""

        step: PooledActTrainStep.Config = field(
            default_factory=PooledActTrainStep.Config,
        )
        """Pooled ACT step, source loss, optimizer, and EMA configuration."""

        dataset: ArcParquetData.Config = field(default_factory=ArcParquetData.Config)
        """Ordered official, re-ARC, concept, and evaluation parquet sources."""

        global_batch_size: int = 768
        """Source batch across data-parallel ranks; local batch is derived."""

        mesh_dim_data_seed: str | None = None
        """Keep CHARM's source post-model RNG stream without a data reseed."""

        @override
        def finalize(self) -> Self:
            if self.global_batch_size > 0:
                self.step.backward_loss_scale = 1.0 / self.global_batch_size
                reporter = self.step.metric_reporter
                if isinstance(reporter, PartialConfig):
                    function, _, _ = reporter.deferred_call()
                    if function is _report_source_act_metrics:
                        reporter.global_batch_size = self.global_batch_size
            runtime = self.runtime
            if isinstance(runtime, MultiProcess.Config):
                dp_size = runtime.mesh_topology.get("dp", 0)
            elif isinstance(runtime, SingleProcess.Config):
                dp_size = 1
            else:
                dp_size = 0
            if (
                dp_size > 0
                and self.global_batch_size > 0
                and self.global_batch_size % dp_size == 0
            ):
                local_batch = self.global_batch_size // dp_size
                self.dataset.batch_size = local_batch
                self.dataset.eval_batch_size = local_batch
                self.step.pool.batch_size = local_batch
                prefix = self.step.model.prefix
                if isinstance(prefix, ComposedSparsePrefix.Config):
                    prefix.residual.batch_size = local_batch
            cast(GridEmbedding.Config, self.step.model.embedding).grid_shape = (
                self.dataset.canvas_size,
                self.dataset.canvas_size,
            )
            for metric in self.metrics_eval.values():
                if isinstance(metric, ParquetPassK.Config):
                    metric.canvas_size = self.dataset.canvas_size
            if self.dataset.base_dir is None:
                self.dataset.base_dir = self.base_dir
            for metric in self.metrics_eval.values():
                if isinstance(metric, ParquetPassK.Config):
                    metric.base_dir = self.dataset.base_dir
                    metric.eval_source = self.dataset.eval_source
            return super().finalize()

    def __init__(self, config: Config) -> None:
        """Reject invalid scalar batch/runtime settings before runtime startup."""
        if config.global_batch_size < 1:
            raise ValueError("CHARM global_batch_size must be positive.")
        runtime = config.runtime
        if isinstance(runtime, MultiProcess.Config):
            if any(
                dimension != "dp" and size != 1
                for dimension, size in runtime.mesh_topology.items()
            ):
                raise ValueError(
                    "CHARM requires pp and tp mesh dimensions to equal 1.",
                )
            dp_size = runtime.mesh_topology.get("dp", 0)
            if dp_size < 1:
                raise ValueError(
                    "CHARM requires an explicit positive data-parallel size.",
                )
        elif isinstance(runtime, SingleProcess.Config):
            dp_size = 1
        else:
            raise TypeError("CHARM requires MultiProcess or SingleProcess runtime.")
        if config.global_batch_size % dp_size:
            raise ValueError(
                "CHARM global_batch_size must be divisible by data-parallel size.",
            )
        super().__init__(config)


def exp000() -> Arc1TrainLoop:
    """Train the reference TRM recipe.

    The baseline every other rung forks, and the only one stating a recipe
    rather than a change: AdamATan2 body plus SignSGD task table,
    unnormalized SwiGLU, float64 stablemax, atomic ACT with sampled-minimum
    exploration, EMA 0.999, global batch 2,048 for 388,670 steps.
    ``train_step_test.py`` pins its trajectory bit for bit against the
    implementation it was ported from.

    Hypothesis:
      The shared priml pieces reproduce the reference TRM exactly.

    References:
      https://arxiv.org/abs/2510.04871
        Jolicoeur-Martineau. Less is More: Recursive Reasoning with Tiny
        Networks. Published at 44.6%.
      https://github.com/SamsungSAILMontreal/TinyRecursiveModels
        License: ``LICENSES/TinyRecursiveModels``.

    Results:
      8xH100: 46.0% pass@2 at step 370,000; the run finished.

    """
    cfg = Arc1TrainLoop()
    cfg.study_name = "arcagi"
    cfg.experiment_name = "exp000"
    cfg.seed = 0

    batch_size = 256
    # Master weights stay float32; the step autocasts forwards to bfloat16.
    model = cfg.step.model = GridSolver.Config()
    model.channels_in = 512
    model.num_layers = 2
    model.embedding = GridEmbedding.Config()
    model.block = RotaryBlock.Config()
    recurrence = model.recurrence = DeepRecurrence.Config()
    recurrence.slow_cycles = 3
    recurrence.fast_cycles = 4
    prefix = model.prefix = PuzzleEmbedding.Config()
    prefix.num_puzzles = NUM_PUZZLE_IDENTIFIERS
    prefix.batch_size = batch_size
    model.compile_core = CoreCompile.Config()
    adam = cfg.step.optimizer = AdamATan2.Config()
    adam.lr = 1e-4
    adam.betas = (0.9, 0.95)
    adam.weight_decay = 0.1
    cfg.step.token_loss = StablemaxTokens.Config()
    cfg.step.reduction = MeanOverBatch.Config()
    pool = cfg.step.pool = AtomicPool.Config()
    pool.batch_size = batch_size
    halting = pool.halting = HaltTraining.Config()
    halting.weight = 0.5
    halting.exploration = SampledMinimum.Config()
    ema = cfg.step.ema = EMA.Config()
    ema.decay = 0.999
    ema.update_after_step = 2_000
    ema.warmup_seed = True
    ema.track_buffers = False
    ema.shadow_kind = "param_dict"
    cfg.step.total_train_steps = 388_670
    cfg.step.warmup_steps = 2_000
    cfg.step.lr_min_ratio = 1.0
    cfg.step.norm_log_interval = 100
    cfg.step.parallelism = ArcDataParallel.Config(gradient_as_bucket_view=True)

    cfg.dataset.num_puzzle_identifiers = NUM_PUZZLE_IDENTIFIERS
    cfg.dataset.batch_size = batch_size
    cfg.dataset.eval_batch_size = batch_size
    cfg.dataset.epochs_per_iter = 5

    cfg.metrics_eval[""] = CanonicalPassK.Config()
    cfg.max_steps = cfg.step.total_train_steps
    cfg.num_steps_eval = 10_000
    cfg.num_steps_log = 100
    cfg.early_train_log_steps = 100
    cfg.eval_warmup_batches = 1
    cfg.eval_every_epoch = False
    checkpointer = cfg.checkpointer = Checkpointer.Config()
    checkpointer.save_every = 4_000
    checkpointer.keep_last_n = 8
    checkpointer.keep_every = 40_000
    wandb = WandbTracker.Config()
    wandb.project = "trm"
    cfg.tracker = TrackerList.Config()
    cfg.tracker.trackers = {"wandb": wandb, "metrics": FileTracker.Config()}
    cfg.runtime = MultiProcess.Config()
    return cfg


def exp001() -> Arc1TrainLoop:
    """Fork exp000 with the following changes.

    - Normalize the SwiGLU gate: ``sigmoid(gate) * RMSNorm(gate * up)``.
    - Train the body with Muon at 5e-3, AdamW at 1e-4 for the rest.

    The two are one change: plain Muon diverges on the unnormalized network,
    and the gate norm is what keeps a high-rate Muon body stable.

    Hypothesis:
      Orthogonalized updates on the reasoning matrices train faster than
      AdamATan2 once the gate norm bounds their scale.

    References:
      https://arxiv.org/abs/2601.19085
        Dillon. Speed is Confidence.

    Results:
      8xH100: 58.125% pass@2 at step 170,000; the run crashed at 230,000.
      A 50,000-step run of the same recipe: 44.875% at step 50,000.

    """
    cfg = exp000()
    cfg.experiment_name = "exp001"
    block = cfg.step.model.block
    assert isinstance(block, RotaryBlock.Config)
    assert isinstance(block.ffn, SwiGLU.Config)
    block.ffn.norm = RMSNorm.Config()
    # Every member decays at 1e-4 / lr, the reference's coupling: the decoupled
    # decay per step, lr * weight_decay, is then 1e-4 at any rate.
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
    return cfg


def exp002() -> Arc1TrainLoop:
    """Fork exp001 with the following changes.

    - Raise the Muon rate from 5e-3 to 0.01.

    Hypothesis:
      The gate norm leaves headroom above 5e-3; a higher rate converges faster.

    References:
      https://kellerjordan.github.io/posts/muon/

    Results:
      8xH100: 59.5% pass@2 at step 370,000; the run finished.

    """
    cfg = exp001()
    cfg.experiment_name = "exp002"
    assert isinstance(cfg.step.optimizer, CompositeOptimizer.Config)
    _, muon, muon_ensemble = cfg.step.optimizer.optimizers
    assert isinstance(muon, Muon.Config)
    assert isinstance(muon_ensemble, Muon.Config)
    muon.lr = muon_ensemble.lr = 0.01
    muon.weight_decay = muon_ensemble.weight_decay = 1e-4 / muon.lr
    return cfg


def exp003() -> Arc1TrainLoop:
    """Fork exp002 with the following changes.

    - Carry ONE latent instead of TRM's slow/fast pair, refined by four blocks
      six times per core application, two applications per forward.
    - Replace each block's SwiGLU with a ConvSwiGLU: a depthwise kernel-2
      convolution over neighbouring tokens on the gated output.
    - Train at per-GPU batch 96 (global 768), the URM's batch.
    - Lower the Muon rate back to 5e-3.

    These are the Universal Reasoning Model's architecture and batch, moved
    together as the URM reports them.

    Hypothesis:
      A short convolution in the feed-forward supplies the local mixing ARC's
      grids reward, and one latent at greater depth beats two shallower ones.

    References:
      https://arxiv.org/abs/2512.14693
        Gao et al. Universal Reasoning Model.

    Results:
      8xH100: 67.625% pass@2 at step 350,000; the run finished.

    """
    cfg = exp002()
    cfg.experiment_name = "exp003"
    batch_size = 96
    model = cfg.step.model
    model.num_layers = 4
    recurrence = model.recurrence = SingleLatentRecurrence.Config()
    recurrence.slow_cycles = 2
    recurrence.fast_cycles = 6
    assert isinstance(model.block, RotaryBlock.Config)
    ffn = model.block.ffn = ConvSwiGLU.Config()
    ffn.init_weight = corrected_fan_in_normal
    ffn.norm = RMSNorm.Config()
    assert isinstance(model.prefix, PuzzleEmbedding.Config)
    model.prefix.batch_size = batch_size
    assert isinstance(cfg.step.optimizer, CompositeOptimizer.Config)
    _, muon, muon_ensemble = cfg.step.optimizer.optimizers
    assert isinstance(muon, Muon.Config)
    assert isinstance(muon_ensemble, Muon.Config)
    muon.lr = muon_ensemble.lr = 5e-3
    muon.weight_decay = muon_ensemble.weight_decay = 1e-4 / muon.lr
    assert isinstance(cfg.step.pool, AtomicPool.Config)
    cfg.step.pool.batch_size = batch_size
    cfg.dataset.batch_size = batch_size
    cfg.dataset.eval_batch_size = batch_size
    return cfg


def exp004() -> Arc1TrainLoop:
    """Fork exp003 with the following changes.

    - Evaluate the short convolution as shifted elementwise taps
      (:func:`~priml.baselines.arcagi.model.depthwise_shift`), which stay
      fused inside the compiled core. Bit-equal in bfloat16 at kernel 2.
    - Keep the compiled core at evaluation, with Inductor emulating the
      precision casts; evaluate at batch 256.
    - Pack label-free signals and the per-step trajectory ahead of each
      predicted grid, and dump them beside the checkpoints they describe.
    - Checkpoint every 5,000 steps instead of 4,000.

    Hypothesis:
      None for accuracy: the same model trains and evaluates faster, and
      every evaluation's views are saved for offline analysis.

    References:
      https://arxiv.org/abs/2512.14693
        Gao et al. Universal Reasoning Model.

    Results:
      8xH100: 68.375% pass@2 at step 340,000; the run finished.

    """
    cfg = exp003()
    cfg.experiment_name = "exp004"
    block = cfg.step.model.block
    assert isinstance(block, RotaryBlock.Config)
    assert isinstance(block.ffn, ConvSwiGLU.Config)
    block.ffn.short_conv = depthwise_shift
    cfg.step.emulate_precision_casts = True
    cfg.step.signals = EvalSignals.Config(per_step=True)
    cfg.dataset.eval_batch_size = 256
    assert isinstance(cfg.step.pool, AtomicPool.Config)
    metric = cfg.metrics_eval[""]
    assert isinstance(metric, CanonicalPassK.Config)
    metric.per_step_acts = cfg.step.pool.max_steps
    assert isinstance(cfg.checkpointer, Checkpointer.Config)
    cfg.checkpointer.save_every = 5_000
    # Dumps rotate with the checkpoints they describe, so disk stays bounded.
    assert isinstance(cfg.tracker, TrackerList.Config)
    cfg.tracker.trackers["signals"] = SignalDumpTracker.Config(
        keep_last_n=cfg.checkpointer.keep_last_n,
        keep_every=cfg.checkpointer.keep_every,
    )
    return cfg


def exp005() -> Arc1TrainLoop:
    """Fork exp004 with the following changes.

    - Train on a tree whose views are also translated (p=0.2) and rescaled
      (p=0.2) inside the 30x30 canvas.
    - Evaluate two spatial views of every color/dihedral view, mapped back to
      the canonical frame before voting.
    - Report three slices of one evaluation pass: the non-spatial views, every
      view at the non-spatial view budget, and every view.

    Hypothesis:
      ARC rules are mostly invariant to where a pattern sits on the canvas, so
      spatial views add training signal and decorrelate the test-time votes.

    References:
      https://arxiv.org/abs/2512.14693
        Gao et al. Universal Reasoning Model.

    Results:
      8xH100: 70.125% pass@2 at step 380,000; 69.6% at the final step
      388,670, where the official strict rule scores 69.0%.

    """
    cfg = exp004()
    cfg.experiment_name = "exp005"
    # The tree's name spells its recipe; the build refuses a tree stamped by
    # another one.
    cfg.dataset.working_dir = (
        "/datasets/arc1concept-aug-1000-tr0p2-sc0p2-2w1p0-n1000-s42-spatialeval-v2"
    )
    cfg.dataset.augmentation.spatial.translation_prob = 0.2
    cfg.dataset.augmentation.spatial.scale_prob = 0.2
    cfg.dataset.augmentation.spatial.train_scale_weights = {2: 1.0}
    # The directory names a spatial-eval tree; this is the flag that builds one.
    cfg.dataset.augmentation.spatial_eval_views = True
    # Staged by scripts/prepare_data.py; zero keeps the loader from building it.
    cfg.dataset.num_puzzle_identifiers = 0

    base = cfg.metrics_eval[""]
    assert isinstance(base, CanonicalPassK.Config)
    base.working_dir = cfg.dataset.working_dir
    non_spatial = cfg.metrics_eval[""] = base.copy_tree()
    non_spatial.spatial_views = "non_spatial"
    non_spatial.max_views_per_input = 0
    spatial_eq = cfg.metrics_eval["spatial_eq"] = base.copy_tree()
    spatial_eq.spatial_views = "all"
    # The non-spatial per-input budget: one canonical view plus 1,000.
    spatial_eq.max_views_per_input = 1_001
    spatial_big = cfg.metrics_eval["spatial_big"] = base.copy_tree()
    spatial_big.spatial_views = "all"
    spatial_big.max_views_per_input = 0
    return cfg


def exp006() -> Arc1TrainLoop:
    """Fork exp005 with the following changes.

    - Normalize queries and keys per head with a parameter-free RMSNorm.
    - Feed the previous step's argmax grid back in through a zero-initialized
      table.
    - Replace 7.5% of the fed-back cells with random colors in training.

    The three were measured as one bundle, so they are one change here.

    Hypothesis:
      Conditioning on its own current answer lets the recurrence refine rather
      than re-derive, and corruption teaches it to repair rather than copy.

    References:
      https://arxiv.org/abs/2510.04871
        Jolicoeur-Martineau. Less is More: Recursive Reasoning with Tiny
        Networks.

    Results:
      8xH100: 71.375% pass@2 at step 280,000 (71.0% at 210,000).

    """
    cfg = exp005()
    cfg.experiment_name = "exp006"
    model = cfg.step.model
    assert isinstance(model.block, RotaryBlock.Config)
    assert isinstance(model.block.attn, Attention.Config)
    norm_qk = model.block.attn.norm_qk = RMSNorm.Config()
    norm_qk.channels_in = model.block.attn.channels_head
    assert isinstance(model.embedding, GridEmbedding.Config)
    model.embedding.channels = [PredictionFeedback.Config()]
    assert isinstance(cfg.step.pool, AtomicPool.Config)
    feedback = cfg.step.pool.feedback = FeedbackCarry.Config()
    corruption = feedback.corruption = CellCorruption.Config()
    corruption.rate = 0.075
    return cfg


def exp007() -> Arc2TrainLoop:
    """Fork exp000 with the following changes.

    - Train and evaluate on ARC-AGI-2: 1,000 ``training2`` tasks plus 160
      ConceptARC tasks, scored on the 120 ``evaluation2`` tasks.
    - Size the task table for that build, and the horizon (541,580 steps) for
      the reference's sample scale on it.
    - Draw four shuffled passes per loader iteration instead of five.
    - Score the strict rule (every test output of a task right) and the
      per-output rule beside pass@K.

    Hypothesis:
      The reference recipe transfers to the harder benchmark unchanged.

    References:
      https://arxiv.org/abs/2510.04871
        Jolicoeur-Martineau. Less is More: Recursive Reasoning with Tiny
        Networks.

    Results:
      8xH100, mid-training: best 6.67% strict@2 (7.36% pass@2) at step
      170,000; the record ends at step 330,000 of 541,580.

    """
    cfg = Arc2TrainLoop().update(exp000())
    cfg.experiment_name = "exp007"
    # Everything the recipe states about the model, step, and optimizer is kept;
    # only what the dataset determines moves.
    cfg.dataset = Arc2PuzzleDataset.Config().update(cfg.dataset)
    cfg.dataset.working_dir = "/datasets/arc2concept-aug-1000"
    cfg.dataset.epochs_per_iter = 4
    cfg.dataset.num_puzzle_identifiers = ARC2_NUM_PUZZLE_IDENTIFIERS
    assert isinstance(cfg.step.model.prefix, SparsePuzzleEmbedding.Config)
    cfg.step.model.prefix.num_puzzles = ARC2_NUM_PUZZLE_IDENTIFIERS
    cfg.max_steps = cfg.step.total_train_steps = ARC2_TOTAL_TRAIN_STEPS
    # Every evaluation forwards its signal payload, so a dumping rung writes one
    # per evaluation.
    cfg.eval_extras_every_eval = True
    assert isinstance(cfg.checkpointer, Checkpointer.Config)
    cfg.checkpointer.save_every = 5_000
    metric = cfg.metrics_eval[""]
    assert isinstance(metric, CanonicalPassK.Config)
    metric.working_dir = cfg.dataset.working_dir
    metric.rules = [StrictPass.Config(), PerOutputPass.Config()]
    return cfg


def exp008() -> Arc2TrainLoop:
    """Fork exp007 with the following changes.

    - Take exp005's recipe: its step, batch sizes, augmentation, metric
      slices, and signal dumps.
    - Build the 0.2/0.2 spatial training tree and its two-view evaluation
      expansion from the ARC-AGI-2 source.

    Hypothesis:
      Spatial test-time augmentation helps more here, where 40.8% of tasks ask
      for several outputs and the strict rule punishes each miss.

    References:
      https://arxiv.org/abs/2512.14693
        Gao et al. Universal Reasoning Model.

    Results:
      8xH100, mid-training: 17.50% strict@2 (20.556% pass@2) over every view
      at step 300,000 of 541,580.

    """
    cfg = exp007()
    cfg.experiment_name = "exp008"
    recipe = exp005()
    cfg.step = recipe.step
    cfg.step.total_train_steps = ARC2_TOTAL_TRAIN_STEPS
    assert isinstance(cfg.step.model.prefix, SparsePuzzleEmbedding.Config)
    cfg.step.model.prefix.num_puzzles = ARC2_NUM_PUZZLE_IDENTIFIERS
    cfg.tracker = recipe.tracker
    cfg.dataset.batch_size = recipe.dataset.batch_size
    cfg.dataset.eval_batch_size = recipe.dataset.eval_batch_size
    cfg.dataset.augmentation = recipe.dataset.augmentation
    cfg.dataset.source_dataset_dir = (
        "/datasets/arc2concept-aug-1000-tr0p2-sc0p2-2w1p0-n1000-s42"
    )
    cfg.dataset.working_dir = (
        "/datasets/arc2concept-aug-1000-tr0p2-sc0p2-2w1p0-n1000-s42-spatialeval-v2"
    )
    cfg.dataset.spatial_eval_views = 2
    cfg.metrics_eval = recipe.metrics_eval
    for metric in cfg.metrics_eval.values():
        assert isinstance(metric, CanonicalPassK.Config)
        metric.working_dir = cfg.dataset.working_dir
        metric.rules = [StrictPass.Config(), PerOutputPass.Config()]
    return cfg


def exp009() -> CharmTrainLoop.Config:
    """Train the complete gated CHARM recipe on ARC-AGI-1.

    Fork exp003, retaining its shared URM tree and applying the CHARM source
    changes to architecture, memory, optimizer, data mixture, and horizon.
    Those settings form one published recipe rather than separate ablations.

    Hypothesis:
      Task-conditioned gated instance residuals improve ARC-AGI-1 reasoning
      when trained with the released 100x re-ARC data mixture.

    References:
      https://arxiv.org/html/2609.33270
      Source revision 728bf26903684ef5e8bcb476cb8e55e42cf35b9d.

    Results:
      TBD; this native recipe has not completed its replicated run.

    """
    cfg = CharmTrainLoop.Config().update(exp003(), skip_missing=True)
    cfg.experiment_name = "exp009"
    cfg.model_torch_seed = torch.manual_seed
    cfg.mesh_dim_data_seed = None
    cfg.dataset = ArcParquetData.Config()
    cfg.global_batch_size = 768
    runtime = cast(MultiProcess.Config, cfg.runtime)
    runtime.mesh_topology = {"dp": 8, "pp": 1, "tp": 1}

    tasks = 960
    rows = 876_705
    source_root = "/datasets/charm-release-937316c8/data/augmented/v2"
    cfg.dataset.train_sources = [
        PairSource.Config(
            path=f"{source_root}/arc1_training_aug1000.parquet",
            pair_selection="both",
            multiplicity=1,
        ),
        PairSource.Config(
            path=f"{source_root}/rearc_aug100.parquet",
            pair_selection="both",
            multiplicity=1,
        ),
        PairSource.Config(
            path=f"{source_root}/concept_aug1000.parquet",
            pair_selection="both",
            multiplicity=2,
        ),
        PairSource.Config(
            path=f"{source_root}/arc1_eval_aug1000.parquet",
            pair_selection="train",
            multiplicity=2,
        ),
    ]
    cfg.dataset.eval_source = f"{source_root}/arc1_eval_aug1000.parquet"
    cfg.dataset.canvas_size = 30
    cfg.dataset.num_workers = 4
    cfg.dataset.seed = 0
    cfg.dataset.p_translate = 0.8
    cfg.dataset.num_tasks = tasks
    cfg.dataset.num_instances = rows - 1

    step = cfg.step
    step.total_train_steps = 1_600_000
    step.lr_step_offset = 1
    step.optimizer_lr_policy = apply_direct_schedule_to_body_and_sparse_optimizer_lrs
    step.dtype_autocast = None
    step.parallelism = ManualDataParallel.Config()
    step.gradient_reduction = SumGradients.Config()
    step.metric_reporter = PartialConfig(_report_source_act_metrics)
    urm_model = step.model
    model = SingleLatentGridSolver.Config()
    source_rope = model.rope
    source_block = cast(TransformerBlock.Config, model.block)
    source_ffn = cast(SwiGLU.Config, source_block.ffn)
    model.update(urm_model, skip_missing=True)
    step.model = model
    model.vocab_size = 12
    model.halt_outputs = 1
    model.head.linear_fn = cast_parameters_to_input
    model.rope = source_rope
    assert isinstance(model.compile_core, CoreCompile.Config)
    model.compile_core.fullgraph = False

    embedding = cast(GridEmbedding.Config, model.embedding)
    embedding.grid_shape = (30, 30)
    embedding.embed_scale = 1.0
    embedding.token_table = Embedding.Config(
        init_weight=corrected_fan_in_normal,
        lookup_dtype=torch.bfloat16,
    )
    urm_block = cast(RotaryBlock.Config, model.block)
    block = TransformerBlock.Config().update(urm_block, skip_missing=True)
    model.block = block
    attention = cast(Attention.Config, block.attn)
    attention.num_heads_kv = 8
    attention.proj_qkv = Linear.Config(
        bias=False,
        init_weight=corrected_fan_in_normal,
        linear_fn=cast_parameters_to_input,
    )
    attention.proj_out = Linear.Config(
        bias=False,
        init_weight=corrected_fan_in_normal,
        linear_fn=cast_parameters_to_input,
    )
    block.attn = attention
    ffn = source_ffn
    ffn.channels_hidden = 1_536
    ffn.expansion = 4.0
    assert isinstance(ffn.middle, SequenceConv1d.Config)
    ffn.middle.conv.groups = 1_536
    ffn.middle.conv.init_bias = _SOURCE_DEPTHWISE_KERNEL_2_BIAS_INIT
    block.ffn = ffn
    recurrence = cast(SingleLatentRecurrence.Config, model.recurrence)
    recurrence.slow_cycles = 1
    recurrence.fast_cycles = 12
    recurrence.inner_grad_loops = 6
    prefix = model.prefix = ComposedSparsePrefix.Config()
    prefix.channels_out = 512
    prefix.num_tokens = 16
    prefix.composition = CategorySlotFiLM.Config(
        num_categories=tasks,
        num_augmentations=8,
        num_slots=9,
        channels_out=512,
    )
    prefix.residual.num_puzzles = rows
    prefix.residual.channels_out = 32
    prefix.residual.num_tokens = 1
    prefix.residual.embed_scale = 1.0
    prefix.residual_gate = SigmoidResidualGate.Config()

    pool = cast(AtomicPool.Config, step.pool)
    pool.steps_dtype = torch.int32
    pool.dtype = torch.bfloat16
    assert isinstance(pool.halting, HaltTraining.Config)
    pool.halting.exploration = GlobalSampledMinimum.Config(prob=0.1)
    step.pool_transition = BeforeBackwardPoolTransition.Config()
    step.objective = ACTLoss.Config(
        token_loss=StablemaxTokens.Config(
            log_normalizer=log_stablemax_via_probabilities,
        ),
        halt_weight=0.5,
    )
    ema = cast(EMA.Config, step.ema)
    ema.update_after_step = 0
    ema.warmup_seed = False
    ema.update_fn = _source_order_ema_update
    step.ema_initialization = EMA.initialize
    rank_two = with_ndim(Muon.eligible_tensor, 2)
    optimizer = cast(CompositeOptimizer.Config, step.optimizer)
    muon = cast(Muon.Config, optimizer.optimizers[1])
    muon.lr = 1e-4
    muon.weight_decay = 0.1
    muon.momentum = 0.95
    muon.ns_steps = 5
    muon.adjust_lr_fn = adjust_lr_match_rms_adamw
    muon.momentum_update_fn = momentum_add_update
    muon.matrix_signum_fn = _SOURCE_MATRIX_SIGNUM_FN
    optimizer.optimizers = [
        muon,
        AuxAdam.Config(
            lr=1e-4,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.1,
        ),
    ]
    optimizer.select = [rank_two, complement(rank_two)]

    cfg.max_steps = step.total_train_steps
    cfg.eval_warmup_batches = 0
    checkpointer = cast(Checkpointer.Config, cfg.checkpointer)
    checkpointer.save_every = 10_000
    checkpointer.keep_last_n = -1
    checkpointer.keep_every = 0
    cfg.metrics_eval = {
        "": ParquetPassK.Config(
            eval_source=cfg.dataset.eval_source,
            canvas_size=30,
            pass_ks=(1, 2, 5, 10, 100, 1000),
            aggregated_voting=True,
        ),
    }
    tracker = cast(TrackerList.Config, cfg.tracker)
    wandb = cast(WandbTracker.Config, tracker.trackers["wandb"])
    wandb.project = "charm"
    wandb.name = "arc-agi-1-charm-rearc100"
    return cfg


def exp010() -> CharmTrainLoop.Config:
    """Train the complete ungated CHARM recipe on ARC-AGI-2.

    Fork exp009's coupled source bundle, then apply the published ARC-AGI-2
    differences: ungated residual, ARC2 data/task counts, and the 2.38M-step
    horizon. These settings form one published recipe, not separate ablations.

    Hypothesis:
      The released ungated instance residual and 100x re-ARC mixture transfer
      the CHARM recipe to ARC-AGI-2's larger task set.

    References:
      https://arxiv.org/html/2609.33270
      Source revision 728bf26903684ef5e8bcb476cb8e55e42cf35b9d.

    Results:
      TBD; this native recipe has not completed its replicated run.

    """
    cfg = exp009()
    cfg.experiment_name = "exp010"
    source_root = "/datasets/charm-release-937316c8/data/augmented/v2"
    cfg.dataset.train_sources = [
        PairSource.Config(
            path=f"{source_root}/arc2_training_aug1000.parquet",
            pair_selection="both",
            multiplicity=1,
        ),
        PairSource.Config(
            path=f"{source_root}/rearc2_aug100.parquet",
            pair_selection="both",
            multiplicity=1,
        ),
        PairSource.Config(
            path=f"{source_root}/concept_aug1000.parquet",
            pair_selection="both",
            multiplicity=2,
        ),
        PairSource.Config(
            path=f"{source_root}/arc2_eval_aug1000.parquet",
            pair_selection="train",
            multiplicity=2,
        ),
    ]
    cfg.dataset.eval_source = f"{source_root}/arc2_eval_aug1000.parquet"
    cfg.dataset.num_tasks = 1_280
    cfg.dataset.num_instances = 1_289_151 - 1
    cfg.step.total_train_steps = 2_380_000
    prefix = cast(ComposedSparsePrefix.Config, cfg.step.model.prefix)
    cast(CategorySlotFiLM.Config, prefix.composition).num_categories = 1_280
    prefix.residual.num_puzzles = 1_289_151
    prefix.residual_gate = None
    cfg.max_steps = cfg.step.total_train_steps
    tracker = cast(TrackerList.Config, cfg.tracker)
    wandb = cast(WandbTracker.Config, tracker.trackers["wandb"])
    wandb.name = "arc-agi-2-charm-rearc2-100"
    return cfg


def eval006() -> Arc1TrainLoop:
    """Evaluate saved exp006 weights through the native ARC loop.

    Returns:
      config: The exp006 recipe with checkpoint evaluation enabled.

    """
    cfg = exp006()
    cfg.experiment_name = "eval006"
    cfg.step = CheckpointActTrainStep.Config().update(cfg.step)
    cfg.eval_only = True
    cfg.eval_warmup_batches = 0
    cfg.checkpointer = None
    cfg.tracker = FileTracker.Config()
    return cfg


def eval009() -> CharmTrainLoop.Config:
    """Evaluate saved exp009 weights through the native ARC loop.

    Returns:
      config: The exp009 recipe with checkpoint evaluation enabled.

    """
    cfg = exp009()
    cfg.experiment_name = "eval009"
    cfg.step = step = CheckpointActTrainStep.Config().update(cfg.step)
    step.ema_initialization = None
    cfg.eval_only = True
    cfg.checkpointer = None
    cfg.tracker = FileTracker.Config()
    return cfg


def eval010() -> CharmTrainLoop.Config:
    """Evaluate saved exp010 weights through the native ARC loop.

    Returns:
      config: The exp010 recipe with checkpoint evaluation enabled.

    """
    cfg = exp010()
    cfg.experiment_name = "eval010"
    cfg.step = step = CheckpointActTrainStep.Config().update(cfg.step)
    step.ema_initialization = None
    cfg.eval_only = True
    cfg.checkpointer = None
    cfg.tracker = FileTracker.Config()
    return cfg


def exp_smoke() -> Arc1TrainLoop:
    """Fork exp000 for an unscored installation check.

    - Run one uncompiled process on the best available device.
    - Reduce width/heads/head width to 32/2/16 and the recurrence to one
      slow and one fast cycle.
    - Seat two rows, at most four ACT steps each; read the first 64 rows of
      each split, already prepared.
    - Stop after four steps and two evaluation batches; save no checkpoint
      and log nowhere.

    The per-task table keeps its full height. Capping rows does not cap the
    ids they carry -- the build numbers puzzles once across every split -- so
    a shorter table would index off the end.

    """
    cfg = exp000()
    cfg.experiment_name = "exp_smoke"
    cfg.runtime = SingleProcess.Config()
    cfg.step.parallelism = NoParallel.Config()
    model = cfg.step.model
    model.compile_core = None
    assert isinstance(model.block, RotaryBlock.Config)
    assert isinstance(model.block.attn, Attention.Config)
    assert isinstance(model.block.ffn, SwiGLU.Config)
    assert isinstance(model.block.rope, RoPE.Config)
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
    assert isinstance(cfg.step.pool, AtomicPool.Config)
    cfg.step.pool.batch_size = 2
    cfg.step.pool.max_steps = 4
    cfg.max_steps = cfg.step.total_train_steps = 4
    cfg.step.warmup_steps = 0
    cfg.dataset.batch_size = 2
    cfg.dataset.eval_batch_size = 2
    cfg.dataset.max_samples = 64
    cfg.dataset.num_puzzle_identifiers = 0
    cfg.num_steps_eval = 2
    cfg.checkpointer = None
    cfg.tracker = None
    return cfg


def _source_order_ema_update(
    shadow: torch.Tensor,
    live: torch.Tensor,
    decay: float,
) -> None:
    """Apply the released EMA expression without changing its operation order."""
    shadow.copy_((1 - decay) * live + decay * shadow)


def _report_source_act_metrics(
    metrics: Mapping[str, Tensor | float],
    *,
    global_batch_size: int,
) -> dict[str, float]:
    """SUM source ACT metrics, then apply their source denominators."""
    names = (
        "accuracy",
        "count",
        "exact_accuracy",
        "lm_loss",
        "q_halt_accuracy",
        "q_halt_loss",
        "steps",
    )
    local = torch.stack([torch.as_tensor(metrics[name]) for name in names])
    values = local.clone()
    world_size = (
        torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    )
    if world_size > 1:
        torch.distributed.reduce(values, dst=0, op=torch.distributed.ReduceOp.SUM)
        if torch.distributed.get_rank() != 0:
            values = local
    count = max(float(values[names.index("count")].item()), 1.0)
    return {
        name: float(
            values[index].item()
            / (global_batch_size if name.endswith("loss") else count),
        )
        for index, name in enumerate(names)
    }
