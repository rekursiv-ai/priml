"""The ARC-AGI experiment ladder.

Each ARC task shows a few input/output grid pairs demonstrating a rule, then
asks for the rule applied to a held-out input. Nothing about the rule is
labelled, so the model must infer it from the examples -- which is why this is
a reasoning benchmark rather than a perception one.

The solver is the sudoku baseline's, with different values in its slots: a
30x30 grid instead of 9x9, twelve colors instead of ten digits, and a learned
per-task prefix sudoku has no use for. Two properties drive the rest:

* **Whole-grid scoring.** One wrong cell fails the puzzle.
* **Heavy augmentation.** Each task is stored many times under recolorings and
  dihedral transforms, so the answer is the consensus across those views
  (pass@K) rather than any single pass.

The ladder mirrors sudoku's: architecture and recurrence are independent slots,
so the same four-corner comparison holds on a harder benchmark.

Launch (8 GPUs)::

    uv --quiet run --frozen python -m torch.distributed.run \
      --standalone --nproc_per_node=8 -m priml \
      priml.baselines.arcagi1.experiments.exp000
"""

from __future__ import annotations

from dataclasses import field
from pathlib import Path
from typing import Final, Literal, Self, override

from configgle import Makes

from priml.baselines.arcagi1.data import ArcData, PuzzleData
from priml.baselines.arcagi1.loss import MeanOverBatch, StablemaxTokens
from priml.baselines.arcagi1.metric import (
    CanonicalPassK,
    PassK,
    SignalDumpTracker,
)
from priml.baselines.arcagi1.model import (
    ConvSwiGLU,
    UrmRecurrence,
    depthwise_shift,
)
from priml.baselines.arcagi1.optimizer import adamw_muon
from priml.baselines.arcagi1.scripts.build_dataset import (
    DEFAULT_SCALE_WEIGHTS,
    aug_policy_template,
)
from priml.baselines.arcagi1.scripts.build_spatial_eval import (
    spatial_eval_dataset_dir,
)
from priml.baselines.arcagi1.train_step import EvalSignals, TrmTrainStep
from priml.baselines.arcagi2.model import PuzzleEmbedding, RotaryBlock
from priml.baselines.arcagi2.train_step import ArcDataParallel
from priml.baselines.sudoku.act import (
    AtomicPool,
    CellCorruption,
    FeedbackCarry,
    HaltTraining,
    SampledMinimum,
    ZeroStart,
)
from priml.baselines.sudoku.embedding import GridEmbedding, PredictionFeedback
from priml.baselines.sudoku.model import (
    CoreCompile,
    DeepRecurrence,
    SudokuNet,
    corrected_fan_in_normal,
)
from priml.baselines.sudoku.prefix import (
    PrefixStack,
    RegisterTokens,
    SparsePuzzleEmbedding,
)
from priml.baselines.sudoku.train_step import SudokuTrainStep
from priml.model.attention.attention import Attention
from priml.model.init import kaiming_uniform
from priml.model.mlpmixer import MLPMixerBlock
from priml.model.norm import RMSNorm
from priml.model.swiglu import SwiGLU
from priml.model.transformer.block import TransformerBlock
from priml.optimizers import AdamATan2
from priml.runtime import MultiProcess, SingleProcess
from priml.train.checkpointer import Checkpointer
from priml.train.ema import EMA
from priml.train.tracker import TrackerList, WandbTracker
from priml.train.train_loop import TrainLoop


NUM_PUZZLE_IDENTIFIERS: Final = 876_403
"""Distinct puzzle ids in ``arc1concept-aug-1000``: 876,402 puzzles plus blank.

A property of the prepared dataset, not a tunable. The build assigns ids once
across every split, so the largest is the puzzle count and the per-task table
needs a row for each -- a shorter table indexes off the end on the first batch
rather than training a smaller model. The dataset cannot supply this: a config
must build with no data on disk, so ``finalize`` may not read the tree."""


class ArcTrainLoop(
    Makes["TrainLoop"],
    TrainLoop.Config[SudokuTrainStep.Config, ArcData.Config],
):
    """A training loop with the ARC step and dataset already in place.

    Narrowing both slots here rather than at each call site lets a factory
    reach ``cfg.step.model`` directly, with no ``isinstance`` before a field it
    is about to set.
    """

    step: SudokuTrainStep.Config = field(default_factory=SudokuTrainStep.Config)
    """The shared puzzle step: solver, optimization, optional recurrence."""

    dataset: ArcData.Config = field(default_factory=ArcData.Config)
    """Augmented ARC tasks, grouped so a batch draws whole tasks."""

    @override
    def finalize(self) -> Self:
        self.dataset.spec.finalize()
        spec = self.dataset.spec
        model = self.step.model
        model.vocab_size = spec.vocab_size
        embedding = model.embedding
        assert isinstance(embedding, GridEmbedding.Config)
        embedding.grid_shape = spec.grid_shape
        if isinstance(model.block, MLPMixerBlock.Config):
            model.block.seq_len = model.total_seq_len
        pool = self.step.pool
        if pool is not None and pool.feedback is not None:
            pool.feedback.givens = (2, spec.vocab_size - 1)
        return super().finalize()


def exp000() -> ArcTrainLoop:
    """Post-norm transformer over the grid, one forward per puzzle.

    The baseline every other experiment forks, and the only one stating a
    recipe rather than a change. Frozen: improvements belong in a fork, so a
    result measured against it stays comparable. The reference band is the
    ``arc1concept-aug-1000`` build from the pinned ARC preparer: 1,000
    color/dihedral augmentations per puzzle, with train-only spatial translation.

    Hypothesis:
      A plain transformer with a learned per-task vector is the strongest
      recipe that uses nothing exotic -- the bar recurrence must clear on a
      benchmark whose tasks are genuinely novel at test time.

    Returns:
      cfg: Training config with transformer block, grid embedding, and sparse
        per-task prefix over 30x30 ARC grids.

    References:
      https://arxiv.org/abs/1911.01547
        Chollet. On the Measure of Intelligence.

    Results:
      TBD.

    """
    cfg = ArcTrainLoop()
    cfg.study_name = "arcagi1"
    cfg.experiment_name = "exp000"
    cfg.seed = 0

    batch_size = 256
    model = cfg.step.model
    model.channels_in = 512
    model.num_layers = 2
    assert isinstance(model.block, TransformerBlock.Config)
    model.block.ffn = SwiGLU.Config(
        init_weight=kaiming_uniform,
        init_weight_out=kaiming_uniform,
    )
    model.embedding = GridEmbedding.Config()

    # A per-task vector plus one register token. The task embedding comes
    # first, so it owns position 0 -- where the halt head reads.
    prefix = PrefixStack.Config()
    prefix.parts = [
        SparsePuzzleEmbedding.Config(
            num_puzzles=NUM_PUZZLE_IDENTIFIERS,
            batch_size=batch_size,
        ),
        RegisterTokens.Config(num_tokens=1),
    ]
    model.prefix = prefix

    cfg.step.total_train_steps = 388_670
    cfg.step.warmup_steps = 2_000
    cfg.step.ema_decay = 0.999
    cfg.step.ema_warmup_steps = 2_000

    cfg.dataset.batch_size = batch_size
    cfg.dataset.eval_batch_size = batch_size

    cfg.metrics_eval["pass"] = PassK.Config()
    cfg.max_steps = cfg.step.total_train_steps
    cfg.num_steps_eval = 10_000
    cfg.num_steps_log = 100
    cfg.eval_warmup_batches = 1
    cfg.eval_every_epoch = False

    cfg.checkpointer = Checkpointer.Config()
    cfg.checkpointer.save_every = 4_000
    cfg.checkpointer.keep_last_n = 8
    cfg.checkpointer.keep_every = 40_000

    cfg.runtime = SingleProcess.Config()
    return cfg


def exp001() -> ArcTrainLoop:
    """exp000 with an MLP-mixer block instead of attention.

    Hypothesis:
      An ARC grid is a fixed 900 cells in a fixed arrangement, so the content
      addressing attention buys may be unnecessary: a learned mixing over
      positions can express the same spatial routing at lower cost.

    Returns:
      cfg: exp000 config with MLP-mixer replacing the transformer attention
        block.

    References:
      https://arxiv.org/abs/2105.01601
        Tolstikhin et al. MLP-Mixer: An all-MLP Architecture for Vision.

    Results:
      TBD.

    """
    cfg = exp000()
    cfg.experiment_name = "exp001"
    cfg.step.model.block = _mixer_block(-1)
    return cfg


def exp002() -> ArcTrainLoop:
    """exp000 plus deep recurrence with adaptive computation time.

    Hypothesis:
      An ARC rule is applied in steps -- find the shape, recolor it, place it
      -- so a fixed-depth network must learn in one pass what a recurrence can
      unroll. Letting each task choose its own depth should beat the same
      parameters spent in a single forward.

    Returns:
      cfg: exp000 config with deep recurrence (3 slow + 4 fast cycles per
        task) and halting policy; the feedback channel is present but
        untrained.

    References:
      https://arxiv.org/abs/2510.04871
        Jolicoeur-Martineau. Less is More: Recursive Reasoning with Tiny
        Networks.

    Results:
      TBD.

    """
    cfg = exp000()
    cfg.experiment_name = "exp002"
    cfg.step.model.recurrence = DeepRecurrence.Config(slow_cycles=3, fast_cycles=4)
    # As trained: the feedback channel is built, but the pool never fed the
    # decoded grid back in training, so its zero-initialized table stayed zero;
    # slots were seated from zero latents rather than the learned ones
    # evaluation starts from.
    embedding = cfg.step.model.embedding
    assert isinstance(embedding, GridEmbedding.Config)
    embedding.channels = [PredictionFeedback.Config()]
    cfg.step.pool = AtomicPool.Config(
        batch_size=cfg.dataset.batch_size,
        halting=HaltTraining.Config(
            weight=0.5,
            exploration=SampledMinimum.Config(),
        ),
        start=ZeroStart.Config(),
    )
    return cfg


def exp003() -> ArcTrainLoop:
    """exp002 with an MLP-mixer block: the fourth corner of the 2x2.

    Hypothesis:
      If recurrence is what makes the task work (exp002) and attention is not
      (exp001), the two gains are independent and the mixer keeps its recurrent
      gain. A drop here instead would mean the recurrence relies on attention
      specifically, not on depth.

    Results:
      TBD.

    Returns:
      cfg: exp002 config with MLP-mixer replacing the transformer attention
        block.

    """
    cfg = exp002()
    cfg.experiment_name = "exp003"
    cfg.step.model.block = _mixer_block(-1)
    return cfg


def exp_smoke() -> ArcTrainLoop:
    """exp000 at minimum size, for verifying an installation end to end.

    Not a result. It answers one question -- is the data staged and does the
    loop run -- so every axis that costs time without bearing on that answer is
    cut. Accuracy will be poor, which is expected.

    The per-task table keeps its full height. Capping tasks does not cap the
    ids they carry -- the build numbers puzzles once across every split -- so a
    table sized to ``num_tasks`` would index off the end. At this width it
    costs 112 MB and 35 ms, which does not bear on the question.

    Returns:
      cfg: exp000 config at 1/16 width (32 channels, 1 layer, 4 steps, 4 tasks)
        for installation verification.

    """
    cfg = exp000()
    cfg.experiment_name = "exp_smoke"
    cfg.step.model.channels_in = 32
    cfg.step.model.num_layers = 1
    cfg.dataset.batch_size = 8
    cfg.dataset.eval_batch_size = 8
    cfg.dataset.num_tasks = 4
    cfg.max_steps = cfg.step.total_train_steps = 4
    cfg.num_steps_eval = 2
    cfg.checkpointer = None

    prefix = cfg.step.model.prefix
    assert isinstance(prefix, PrefixStack.Config)
    table = prefix.parts[0]
    assert isinstance(table, SparsePuzzleEmbedding.Config)
    table.batch_size = cfg.dataset.batch_size
    return cfg


class TrmTrainLoop(
    Makes["TrainLoop"],
    TrainLoop.Config[TrmTrainStep.Config, PuzzleData.Config],
):
    """The reference TRM step over the reference-plan loader."""

    step: TrmTrainStep.Config = field(default_factory=TrmTrainStep.Config)
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


def exp004() -> TrmTrainLoop:
    """Train the reference TRM recipe.

    A new root rather than a fork of exp000..exp003: those use the shared
    sudoku step, which has no atomic pool, sparse optimizer, or stablemax. The
    recipe is the paper's: AdamATan2 body plus SignSGD task table, unnormalized
    SwiGLU, float64 stablemax, atomic ACT with sampled-minimum exploration, EMA
    0.999, global batch 2048 for 388,670 steps. ``train_step_test.py`` pins its
    trajectory bit for bit against the implementation it was ported from.

    Hypothesis:
      The shared priml pieces reproduce the reference TRM exactly.

    References:
      https://arxiv.org/abs/2510.04871
        Jolicoeur-Martineau. Less is More: Recursive Reasoning with Tiny
        Networks.

    Results:
      TBD. The implementation it was ported from reached eval pass@1 0.40375
      at step 270,000.

    Returns:
      cfg: The reference-parity recipe.

    """
    cfg = TrmTrainLoop()
    cfg.study_name = "arcagi1"
    cfg.experiment_name = "exp004"
    cfg.seed = 0

    batch_size = 256
    cfg.step.model = _reference_model(batch_size)
    cfg.step.optimizer = AdamATan2.Config(lr=1e-4, betas=(0.9, 0.95), weight_decay=0.1)
    cfg.step.token_loss = StablemaxTokens.Config()
    cfg.step.reduction = MeanOverBatch.Config()
    cfg.step.pool = AtomicPool.Config(
        batch_size=batch_size,
        halting=HaltTraining.Config(
            weight=0.5,
            exploration=SampledMinimum.Config(),
        ),
    )
    cfg.step.ema = EMA.Config(
        decay=0.999,
        update_after_step=2_000,
        warmup_seed=True,
        track_buffers=False,
        shadow_kind="param_dict",
    )
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
    cfg.checkpointer = Checkpointer.Config(
        save_every=4_000,
        keep_last_n=8,
        keep_every=40_000,
    )
    cfg.tracker = WandbTracker.Config(project="trm")
    cfg.runtime = MultiProcess.Config(mesh_topology={"dp": -1, "pp": 1, "tp": 1})
    return cfg


def exp005() -> TrmTrainLoop:
    """exp004 + the modified SwiGLU and a Muon body.

    Two changes, inseparable: the gate norm is what keeps a high-rate Muon body
    stable. Label smoothing is not set: the stablemax loss has no such term.

    Hypothesis:
      Orthogonalized updates on the reasoning matrices train faster than
      AdamATan2 once the gate norm bounds their scale.

    References:
      https://arxiv.org/abs/2601.19085
        Dillon. Speed is Confidence.

    Results:
      TBD. The implementation it was ported from reached eval pass@1 0.44625
      at step 70,000.

    Returns:
      cfg: exp004 with the Muon recipe.

    """
    cfg = exp004()
    cfg.experiment_name = "exp005"
    block = cfg.step.model.block
    assert isinstance(block, TransformerBlock.Config)
    assert isinstance(block.ffn, SwiGLU.Config)
    block.ffn.norm = RMSNorm.Config()
    cfg.step.optimizer = adamw_muon(adamw_lr=1e-4, muon_lr=5e-3)
    return cfg


def exp006() -> TrmTrainLoop:
    """exp005 with the Muon rate doubled to 0.01.

    Hypothesis:
      The gate norm leaves headroom above 5e-3; a higher rate converges faster.

    References:
      https://kellerjordan.github.io/posts/muon/

    Results:
      TBD.

    Returns:
      cfg: exp005 at Muon rate 0.01.

    """
    cfg = exp005()
    cfg.experiment_name = "exp006"
    cfg.step.optimizer = adamw_muon(adamw_lr=1e-4, muon_lr=0.01)
    return cfg


def exp007() -> TrmTrainLoop:
    """exp006 + the URM model, 0.2/0.2 spatial training aug, and signal dumps.

    A single latent refined by four ConvSwiGLU blocks six times per core
    application, two applications per forward; Muon back at 5e-3; batch 96; the
    translation and scale aug-policy tree; and evaluation packing label-free
    signals.

    Hypothesis:
      A short convolution in the feed-forward supplies the local mixing ARC's
      grids reward, and one latent at greater depth beats two shallower ones.

    References:
      https://arxiv.org/abs/2512.14693
        Gao et al. Universal Reasoning Model.

    Results:
      TBD. The implementation it was ported from reached eval pass@2 0.70125
      at step 380,000.

    Returns:
      cfg: exp006 with the URM recipe.

    """
    cfg = exp006()
    cfg.experiment_name = "exp007"
    batch_size = 96
    model = cfg.step.model = _reference_model(batch_size)
    model.num_layers = 4
    model.recurrence = UrmRecurrence.Config(slow_cycles=2, fast_cycles=6)
    assert isinstance(model.block, TransformerBlock.Config)
    model.block.ffn = ConvSwiGLU.Config(
        short_conv=depthwise_shift,
        init_weight=corrected_fan_in_normal,
        norm=RMSNorm.Config(),
    )
    cfg.step.optimizer = adamw_muon(adamw_lr=1e-4, muon_lr=5e-3)
    assert isinstance(cfg.step.pool, AtomicPool.Config)
    cfg.step.pool.batch_size = batch_size
    cfg.step.emulate_precision_casts = True
    cfg.step.signals = EvalSignals.Config(per_step=True)

    cfg.dataset.working_dir = spatial_eval_dataset_dir(
        spatial_views=2,
        source_name=Path(
            aug_policy_template(translation_prob=0.2, scale_prob=0.2),
        ).name,
    )
    cfg.dataset.augmentation.spatial.translation_prob = 0.2
    cfg.dataset.augmentation.spatial.scale_prob = 0.2
    cfg.dataset.augmentation.spatial.train_scale_weights = dict(DEFAULT_SCALE_WEIGHTS)
    cfg.dataset.batch_size = batch_size
    cfg.dataset.eval_batch_size = 256
    # The spatial expansion is built by scripts/build_spatial_eval.py, not the
    # loader; a positive count would make the loader rebuild the plain tree.
    cfg.dataset.num_puzzle_identifiers = 0

    base = CanonicalPassK.Config(per_step_acts=cfg.step.pool.max_steps)
    base.working_dir = cfg.dataset.working_dir
    cfg.metrics_eval = {
        "": _sliced(base, spatial_views="non_spatial", max_views=0),
        # The non-spatial per-input budget: one canonical view plus 1,000.
        "spatial_eq": _sliced(base, spatial_views="all", max_views=1_001),
        "spatial_big": _sliced(base, spatial_views="all", max_views=0),
    }
    assert isinstance(cfg.checkpointer, Checkpointer.Config)
    cfg.checkpointer.save_every = 5_000
    cfg.tracker = TrackerList.Config(
        trackers={
            "wandb": WandbTracker.Config(project="trm"),
            "signals": SignalDumpTracker.Config(),
        },
    )
    return cfg


def exp008() -> TrmTrainLoop:
    """exp007 + QK-norm, prediction feedback, and corrupted-feedback repair.

    The three mechanisms were measured as one bundle, so they are one change
    here: parameter-free per-head RMSNorm on Q and K; the
    previous step's argmax grid fed back through a zero-initialized table; and
    7.5% of fed-back cells replaced by random colors in training.

    Hypothesis:
      Conditioning on its own current answer lets the recurrence refine rather
      than re-derive, and corruption teaches it to repair rather than copy.

    References:
      https://arxiv.org/abs/2510.04871
        Jolicoeur-Martineau. Less is More: Recursive Reasoning with Tiny
        Networks.

    Results:
      TBD. The implementation it was ported from reached eval pass@2 0.71375
      at step 280,000; ``scripts/reproduce_blog_post.py`` reruns it.

    Returns:
      cfg: exp007 with the bundle mechanisms.

    """
    cfg = exp007()
    cfg.experiment_name = "exp008"
    model = cfg.step.model
    assert isinstance(model.block, TransformerBlock.Config)
    assert isinstance(model.block.attn, Attention.Config)
    model.block.attn.norm_qk = RMSNorm.Config(model.block.attn.channels_head)
    assert isinstance(model.embedding, GridEmbedding.Config)
    model.embedding.channels = [PredictionFeedback.Config()]
    assert isinstance(cfg.step.pool, AtomicPool.Config)
    cfg.step.pool.feedback = FeedbackCarry.Config(
        corruption=CellCorruption.Config(rate=0.075),
    )
    return cfg


# The model half of exp004: every value the URM fork keeps, so both build from one
# place. Master weights stay float32; the step autocasts forwards to bfloat16.
def _reference_model(batch_size: int) -> SudokuNet.Config:
    """Return the reference TRM solver sized for ``batch_size`` slots."""
    model = SudokuNet.Config()
    model.channels_in = 512
    model.num_layers = 2
    model.embedding = GridEmbedding.Config()
    model.block = RotaryBlock.Config()
    model.recurrence = DeepRecurrence.Config(slow_cycles=3, fast_cycles=4)
    model.prefix = PuzzleEmbedding.Config(
        num_puzzles=NUM_PUZZLE_IDENTIFIERS,
        batch_size=batch_size,
    )
    model.compile_core = CoreCompile.Config()
    return model


def _sliced(
    base: CanonicalPassK.Config,
    *,
    spatial_views: Literal["all", "non_spatial"],
    max_views: int,
) -> CanonicalPassK.Config:
    """Clone ``base`` with one spatial-view slice and per-input budget."""
    metric = base.copy_tree()
    metric.spatial_views = spatial_views
    metric.max_views_per_input = max_views
    return metric


# Post-norm, matching the transformer default: a recurrence feeds a block its own
# output, and an unnormalized residual stream compounds when it does.
def _mixer_block(seq_len: int) -> MLPMixerBlock.Config:
    """Return an MLP-mixer block shaped for the padded grid plus its prefix."""
    return MLPMixerBlock.Config(
        seq_len=seq_len,
        prenorm=False,
        token_mixer=SwiGLU.Config(
            norm=RMSNorm.Config(),
            init_weight=kaiming_uniform,
            init_weight_out=kaiming_uniform,
        ),
        channel_mixer=SwiGLU.Config(
            norm=RMSNorm.Config(),
            init_weight=kaiming_uniform,
            init_weight_out=kaiming_uniform,
        ),
    )
