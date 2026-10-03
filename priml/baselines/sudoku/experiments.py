"""The sudoku experiment ladder.

Two mechanisms vary, independently, so the ladder is a 2x2 rather than a chain:

* which block mixes the grid tokens -- transformer or MLP-mixer,
* whether the solver runs a recurrence with adaptive computation time.

::

              transformer   MLP-mixer
    plain       exp000        exp001
    recurrent   exp002        exp003

Both axes are config VALUES -- a slot filled differently -- so the four share
one model class, one train step, and one dataset. That is what makes the
comparison meaningful: nothing differs except the thing named.

``exp000`` is the naive recipe and is never edited; improvements are forks, so
a number measured against it stays comparable. Run any of them::

    uv --quiet run --frozen python -m priml \
        priml.baselines.sudoku.experiments.exp000
"""

from __future__ import annotations

from dataclasses import field
from typing import Self, override

import math

from configgle import Makes

from priml.baselines.sudoku.act import (
    AtomicPool,
    FeedbackCarry,
    HaltTraining,
    LearnedStart,
    SampledMinimum,
    ZeroStart,
)
from priml.baselines.sudoku.data import SudokuData
from priml.baselines.sudoku.embedding import (
    FactoredPositions,
    GridEmbedding,
    PredictionFeedback,
)
from priml.baselines.sudoku.eval import HpsEval, Reproduction, SieveEval
from priml.baselines.sudoku.metric import GridAccuracy
from priml.baselines.sudoku.model import DeepRecurrence
from priml.baselines.sudoku.train_step import SudokuTrainStep
from priml.baselines.sudoku.trainer import Trainer
from priml.baselines.sudoku.trm import recipe_block
from priml.model.attention.attention import Attention
from priml.model.init import kaiming_uniform
from priml.model.mlpmixer import MLPMixerBlock
from priml.model.norm import RMSNorm
from priml.model.swiglu import SwiGLU
from priml.model.transformer.block import TransformerBlock
from priml.runtime import SingleProcess
from priml.train.train_loop import TrainLoop


class SudokuTrainLoop(
    Makes["TrainLoop"],
    TrainLoop.Config[SudokuTrainStep.Config, SudokuData.Config],
):
    """A training loop with the sudoku step and dataset already in place.

    Narrowing the two slots here rather than at each call site is what lets a
    factory reach ``cfg.step.model`` directly, with no ``isinstance`` narrow
    before a field it is about to set.
    """

    step: SudokuTrainStep.Config = field(default_factory=SudokuTrainStep.Config)
    """Model, optimization, and the optional recurrence."""

    dataset: SudokuData.Config = field(default_factory=SudokuData.Config)
    """Prepared sudoku puzzles, served from device memory."""

    @override
    def finalize(self) -> Self:
        self.dataset.spec.finalize()
        spec = self.dataset.spec
        model = self.step.model
        model.vocab_size = spec.vocab_size
        embedding = model.embedding
        assert isinstance(embedding, GridEmbedding.Config)
        embedding.grid_shape = (math.prod(spec.grid_shape),)
        for channel in embedding.channels:
            if isinstance(channel, FactoredPositions.Config):
                channel.grid_shape = spec.grid_shape
                channel.box_shape = spec.box_shape
        if isinstance(model.block, MLPMixerBlock.Config):
            model.block.seq_len = model.total_seq_len
        pool = self.step.pool
        if pool is not None and pool.feedback is not None:
            pool.feedback.givens = (2, spec.vocab_size - 1)
        return super().finalize()


def exp000() -> SudokuTrainLoop:
    """Post-norm transformer over the grid, one forward per puzzle.

    The baseline every other experiment forks, and the only one stating a
    recipe rather than a change. Frozen: improvements belong in a fork, so a
    result measured against it stays comparable.

    Hypothesis:
      A plain transformer with learned row/column/box positions, AdamW on the
      lookup tables and Muon on the reasoning matrices, is the strongest recipe
      that uses nothing exotic -- the bar recurrence must clear to earn its
      cost.

    Returns:
      config: SudokuTrainLoop configuration with 2 layers, 512 hidden dim,
        384 batch size, 19.5k steps.

    References:
      https://arxiv.org/abs/2510.04871
        Jolicoeur-Martineau. Less is More: Recursive Reasoning with Tiny
        Networks.

    Results:
      TBD.

    """
    cfg = SudokuTrainLoop()
    cfg.study_name = "sudoku"
    cfg.experiment_name = "exp000"

    embedding = GridEmbedding.Config()
    # Sudoku's constraints are row, column, and box, so positions are described
    # by which of each a cell belongs to rather than by its index in a flattened
    # sequence.
    embedding.channels = [FactoredPositions.Config()]
    cfg.step.model.embedding = embedding
    cfg.step.model.channels_in = 512
    cfg.step.model.num_layers = 2
    assert isinstance(cfg.step.model.block, TransformerBlock.Config)
    cfg.step.model.block.ffn = SwiGLU.Config(
        init_weight=kaiming_uniform,
        init_weight_out=kaiming_uniform,
    )

    cfg.dataset.batch_size = 384
    cfg.dataset.seed = 0
    # Mid-training evaluation reads a fixed prefix of the test split as a proxy;
    # the reported number is measured on the whole thing, and the two
    # populations are not comparable.
    cfg.dataset.num_eval_puzzles = 2_000

    cfg.metrics_eval["accuracy"] = GridAccuracy.Config()
    cfg.max_steps = cfg.step.total_train_steps = 19_500
    cfg.num_steps_eval = 1_000
    cfg.runtime = SingleProcess.Config()
    return cfg


def exp001() -> SudokuTrainLoop:
    """exp000 with an MLP-mixer block instead of attention.

    Hypothesis:
      A sudoku grid is a fixed 81 cells in a fixed arrangement, so the content
      addressing attention buys may be unnecessary: a learned mixing over
      positions can express the same row/column/box routing at lower cost. If
      so, the mixer matches the transformer, and attention is not what makes
      this task work.

    Returns:
      config: SudokuTrainLoop with mixer block in place of attention.

    References:
      https://arxiv.org/abs/2105.01601
        Tolstikhin et al. MLP-Mixer: An all-MLP Architecture for Vision.

    Results:
      TBD.

    """
    cfg = exp000()
    cfg.experiment_name = "exp001"
    cfg.step.model.block = _mixer_block()
    return cfg


def exp002() -> SudokuTrainLoop:
    """exp000 plus deep recurrence with adaptive computation time.

    Hypothesis:
      Constraint propagation is iterative -- filling one cell licenses filling
      the next -- so a fixed-depth network must learn in one pass what a
      recurrence can unroll. Re-applying a small stack over a carried latent,
      and letting each puzzle choose its own depth, should beat the same
      parameters spent in a single forward.

    Returns:
      config: SudokuTrainLoop with deep recurrence; the feedback channel is
        present but untrained.

    References:
      https://arxiv.org/abs/2510.04871
        Jolicoeur-Martineau. Less is More: Recursive Reasoning with Tiny
        Networks.

    Results:
      TBD.

    """
    cfg = exp000()
    cfg.experiment_name = "exp002"
    cfg.step.model.recurrence = DeepRecurrence.Config()
    # As trained: the feedback channel is built, but the pool never fed the
    # decoded grid back in training, so its zero-initialized table stayed zero
    # and the recurrence carried belief only in the latent; slots were seated
    # from zero latents rather than the learned ones evaluation starts from.
    embedding = cfg.step.model.embedding
    assert isinstance(embedding, GridEmbedding.Config)
    embedding.channels = [FactoredPositions.Config(), PredictionFeedback.Config()]
    cfg.step.pool = AtomicPool.Config(
        batch_size=cfg.dataset.batch_size,
        max_steps=32,
        halting=HaltTraining.Config(exploration=SampledMinimum.Config()),
        start=ZeroStart.Config(),
    )
    return cfg


def exp003() -> SudokuTrainLoop:
    """exp002 with an MLP-mixer block: the fourth corner of the 2x2.

    Hypothesis:
      If recurrence is what makes the task work (exp002) and attention is not
      (exp001), the two gains are independent and the mixer keeps its recurrent
      gain. A drop here instead would mean the recurrence relies on attention
      specifically, not on depth.

    Results:
      TBD.

    Returns:
      config: SudokuTrainLoop with mixer block and deep recurrence.

    """
    cfg = exp002()
    cfg.experiment_name = "exp003"
    cfg.step.model.block = _mixer_block()
    return cfg


def exp004() -> Trainer.Config:
    """Attention TRM with every later mechanism off: the blog's ladder root.

    The rekursiv.ai sudoku ladder (``exp004``-``exp010``) runs on the TRM
    trainer, whose defaults ARE the ``exp010`` recipe; this rung switches every
    later mechanism off explicitly and each fork turns one back on. Mid-training
    eval scores the first 2,000 test rows (the 2K proxy), which underestimates
    the full 422,786-puzzle set.

    Hypothesis:
      The reference TRM recipe -- post-norm attention (hidden 512, 8 heads, 2
      layers, slow 3 / fast 4), label smoothing 0.2, Muon 0.02 + AdamW 1e-4,
      EMA 0.9 from step 5,000, ACT 16, batch 384 -- with cosine annealing to
      zero over exactly the 62,000 trained steps is a reproducible control.
      The horizon is load-bearing: a 100k horizon truncated by a wall-clock
      cap spreads 35pp across seeds.

    References:
      https://arxiv.org/abs/2510.04871
        Jolicoeur-Martineau. Less is More: Recursive Reasoning with Tiny
        Networks.

    Results:
      0.61 exact (2K proxy, final; n=1).

    Returns:
      config: Baseline trainer configuration.

    """
    cfg = Trainer.Config()
    cfg.study_name = "sudoku"
    cfg.experiment_name = "exp004"
    cfg.seed = 0

    cfg.model.slow_cycles = 3
    cfg.model.fast_cycles = 4
    cfg.model.pos2d_grid_shape = None
    block = recipe_block()
    attn = block.attn
    assert isinstance(attn, Attention.Config)
    attn.norm_qk = None
    cfg.model.block = block

    cfg.max_steps = 62_000
    cfg.total_train_steps = 62_000
    cfg.max_act_steps = 16
    cfg.label_smoothing = 0.2
    cfg.ema_decay = 0.9
    cfg.ema_warmup_steps = 5_000
    cfg.csp_loss_weight = 0.0
    cfg.feedback = False

    cfg.dataset.batch_size = 384
    cfg.dataset.seed = 0
    cfg.dataset.eval_num_instances = 2_000
    cfg.dataset.augment = True
    return cfg


def exp005() -> Trainer.Config:
    """exp004 plus learned row, column, and box position tables.

    Hypothesis:
      Attention spends capacity discovering that cell (r, c) shares
      constraints with row r, column c, and its 3x3 box. Adding learned
      row + column + box embeddings to the grid tokens (RoPE kept, so the
      delta is strictly additive; +13,824 params) hands it that structure.

    References:
      https://arxiv.org/abs/2510.04871
        Jolicoeur-Martineau. Less is More: Recursive Reasoning with Tiny
        Networks.

    Results:
      0.70 exact (2K proxy, final at 62k; +9pp over exp004; n=1).

    Returns:
      config: Trainer configuration with structural position tables.

    """
    cfg = exp004()
    cfg.experiment_name = "exp005"
    cfg.model.pos2d_grid_shape = (0, 0)
    return cfg


def exp006() -> Trainer.Config:
    """exp005 plus parameter-free QK-norm at the 12k convergence horizon.

    Hypothesis:
      A shared affine-free RMSNorm over the per-head queries and keys bounds
      the attention logits, so training converges several times sooner. The
      horizon moves with it (62k -> 12k, eval every 500): QK-norm converges
      near 10k steps, and post-mastery training at high LR collapses the ACT
      depth, so the cosine must anneal inside that window. One change,
      because the norm without the shorter anneal trains past convergence.

    References:
      https://arxiv.org/abs/2010.04245
        Henry et al. Query-Key Normalization for Transformers.

    Results:
      ~5x faster convergence (0.71@10k vs 0.15@10k, 2K proxy; n=1). QK-norm
      is also what keeps the deep recurrence of exp009 trainable.

    Returns:
      config: Trainer configuration with parameter-free QK normalization.

    """
    cfg = exp005()
    cfg.experiment_name = "exp006"
    assert isinstance(cfg.model.block, TransformerBlock.Config)
    attn = cfg.model.block.attn
    assert isinstance(attn, Attention.Config)
    attn.norm_qk = RMSNorm.Config()
    cfg.max_steps = 12_000
    cfg.total_train_steps = 12_000
    cfg.num_steps_eval = 500
    return cfg


def exp007() -> Trainer.Config:
    """exp006 plus the consolidated training stack, including the CSP loss.

    Hypothesis:
      The reference TRM's own training choices -- ACT 32, no label smoothing,
      EMA 0.999 from step 0, and digit-only augmentation (the legacy branch
      mixes the blank marker into the permutation) -- plus a differentiable
      row/column/box cardinality loss at weight 0.3 each help, and help
      together. Consolidated into one rung because each was measured in
      isolation before and the blog reports them as one stack.

    References:
      https://arxiv.org/abs/2307.04895
        Yang, Lee & Park. Learning to Solve Constraint Satisfaction Problems
        with Recurrent Transformer.

    Results:
      0.78 exact (2K proxy, last-5 mean; config-family mean ~0.755-0.763,
      seed spread ~2.4pp).

    Returns:
      config: Trainer configuration with the consolidated training stack.

    """
    cfg = exp006()
    cfg.experiment_name = "exp007"
    cfg.max_act_steps = 32
    cfg.label_smoothing = 0.0
    cfg.ema_decay = 0.999
    cfg.ema_warmup_steps = 0
    cfg.csp_loss_weight = 0.3
    cfg.dataset.augment_digits_only = True
    return cfg


def exp008() -> Trainer.Config:
    """exp007 plus corrupted-feedback repair training.

    Hypothesis:
      Feeding each ACT step its previous decoded grid, and corrupting that
      grid in training (probability 0.5 per slot, ~12 non-given cells),
      teaches an explicit repair skill the recurrence otherwise lacks. The
      feedback table is zero-initialized, so the step-0 forward is
      bit-identical to exp007's.

    References:
      https://arxiv.org/abs/2510.04871
        Jolicoeur-Martineau. Less is More: Recursive Reasoning with Tiny
        Networks.

    Results:
      0.82 exact (2K proxy, last-5 mean; paired gain +4.3..+5.9pp over exp007
      on 3/3 seeds).

    Returns:
      config: Trainer configuration with corrupted-feedback repair training.

    """
    cfg = exp007()
    cfg.experiment_name = "exp008"
    cfg.feedback = True
    return cfg


def exp009() -> Trainer.Config:
    """exp008 at the deep recurrence shape, slow 6 / fast 9.

    Hypothesis:
      Depth and repair are complementary rather than sub-additive: more
      cycles per ACT step give the repair skill more room to act. QK-norm
      (exp006) is what keeps the deeper stack trainable. Also moves the train
      seed to 44, the seed the measured lineage carries forward to exp010.

    References:
      https://arxiv.org/abs/2510.04871
        Jolicoeur-Martineau. Less is More: Recursive Reasoning with Tiny
        Networks.

    Results:
      0.86-0.90 exact at 12k (2K proxy, last-5 on two paired seeds: 0.8634 /
      0.8972; depth x repair +5.9..+7.5pp paired). Still climbing at 12k: the
      anneal is too short for this class.

    Returns:
      config: Trainer configuration with deep recurrence.

    """
    cfg = exp008()
    cfg.experiment_name = "exp009"
    cfg.model.slow_cycles = 6
    cfg.model.fast_cycles = 9
    cfg.seed = 44
    return cfg


def exp010() -> Trainer.Config:
    """exp009 at the 19,500-step anneal: the generator recipe.

    Hypothesis:
      Mastery arrives later on the deep + repair class, so a longer cosine
      (12k -> 19,500, the longest that fits one H100-hour) converts exp009's
      un-annealed climb into a higher plateau.

    References:
      https://arxiv.org/abs/2510.04871
        Jolicoeur-Martineau. Less is More: Recursive Reasoning with Tiny
        Networks.

    Results:
      0.93 exact (2K proxy: final 0.9295, last-5 0.9300); full-set
      deterministic pass 0.9624 (406,892/422,786). Trained with the ambient
      global-RNG augmentation stream (``dataset.augment_seed = None``).

    Returns:
      config: Trainer configuration for the generator recipe.

    """
    cfg = exp009()
    cfg.experiment_name = "exp010"
    cfg.max_steps = 19_500
    cfg.total_train_steps = 19_500
    return cfg


def exp011() -> HpsEval.Config:
    """exp010 plus hypothesis-pinning search at evaluation.

    Scores exp010's final checkpoint, so run exp010 first.

    Hypothesis:
      The halt head's confidence is a usable acceptance signal. Accept a grid
      only when q >= 7.875 at ACT steps 24, 28 and 32; otherwise pin a digit
      into the most uncertain cell as an extra given and re-run, as a small
      tree search hps(5, 3, 2, 512). Most of exp010's residual 3.8% should be
      recoverable without retraining.

    References:
      https://arxiv.org/abs/2510.04871
        Jolicoeur-Martineau. Less is More: Recursive Reasoning with Tiny
        Networks.

    Results:
      0.99920 exact on the full set (422,447/422,786; ~2h on one H100), against
      exp010's 0.9624. 321 accepted grids are wrong: confidence alone is not a
      sound acceptance gate, which exp012 and exp013 address.

    Returns:
      config: HPS evaluation of exp010's final checkpoint.

    """
    parent = exp010()
    cfg = HpsEval.Config()
    cfg.experiment_name = "exp011"
    cfg.checkpoint_path = _final_checkpoint(parent)
    cfg.model = parent.copy_tree().finalize().model
    return cfg


def exp012() -> Reproduction.Config:
    """exp010 plus a nine-view agreement lock at evaluation.

    Trains exp010, then evaluates it: each puzzle is solved (with exp011's
    search) on the identity view and on a symmetric copy; two agreeing grids
    lock, and a disagreement takes the modal grid over all nine views.

    Hypothesis:
      Two exact symmetries of one puzzle rarely produce the same WRONG grid,
      so agreement is a sound, verifier-free acceptance gate. Closing exp011's
      321 wrong accepts should need about two solves per puzzle.

    References:
      https://arxiv.org/abs/2510.04871
        Jolicoeur-Martineau. Less is More: Recursive Reasoning with Tiny
        Networks.

    Results:
      422,786/422,786 on the full set measured, zero wrong locks (~26-28h on
      one H100 including training). The measured run resumed from a
      5,000-step checkpoint of an earlier launch of the same pinned-seed
      pipeline; steps 5,000-19,500 and every evaluation ran fresh.

    Returns:
      config: Train-then-lock pipeline over the exp010 recipe.

    """
    cfg = Reproduction.Config()
    cfg.study_name = "sudoku"
    cfg.experiment_name = "exp012"
    cfg.generator = exp010()
    cfg.generator_names = ("exp012_generator",)
    return cfg


def exp013() -> Reproduction.Config:
    """exp010 plus a verifier-committee sieve at evaluation.

    Harvests candidate grids from exp010's final checkpoint, so run exp010
    first, and fits three verifiers on them (seeds 0/1/2). Then trains a fresh
    exp010 and evaluates it with the sieve: one cheap pass, symmetry-view
    searches on the survivors, and an escalated search on the last few, every
    round offered to the committee lock.

    Hypothesis:
      A small network that sees only (puzzle, candidate) and attends along
      rows, columns and boxes can learn to verify solutions. Three seeds err
      on different grids, so unanimity is a sound lock -- and it makes
      exp010's cheap single pass usable, retiring ~96% of puzzles at once.

    References:
      https://arxiv.org/abs/2307.04895
        Yang, Lee & Park. Learning to Solve Constraint Satisfaction Problems
        with Recurrent Transformer.

    Results:
      422,786/422,786 on the full set measured on exp010's frozen
      checkpoint, zero false accepts over ~440K live accepts, 0.91 GPU-h
      (the cheap pass retires ~96.2% in ~666s). This train-then-sieve
      pipeline read 422,786/422,786 at two checkpoints before the run
      crashed for unrelated infrastructure reasons.

    Returns:
      config: Train, harvest, fit, then sieve over the exp010 recipe.

    """
    cfg = Reproduction.Config()
    cfg.study_name = "sudoku"
    cfg.experiment_name = "exp013"
    cfg.screen = "committee"
    cfg.harvest_source_checkpoint = _final_checkpoint(exp010())
    cfg.generator = exp010()
    cfg.generator_names = ("exp013_generator",)
    cfg.verifier_names = tuple(f"exp013_verifier_s{seed}" for seed in range(3))
    cfg.full_eval = SieveEval.Config()
    return cfg


def exp014() -> Reproduction.Config:
    """exp012 with three training seeds as the committee instead of nine views.

    Hypothesis:
      Independently trained models disagree on errors at least as often as
      views of one model do, so a seed ensemble (44/45/46, plus six views of
      seed 44) locks as soundly as exp012 -- a check that the lock's soundness
      is not an artifact of one checkpoint.

    References:
      https://arxiv.org/abs/2510.04871
        Jolicoeur-Martineau. Less is More: Recursive Reasoning with Tiny
        Networks.

    Results:
      TBD. Predicted before launch: 422,786/422,786, mean ~2.01 passes; the
      (44, 45) first pair agreed on 0.998526 of puzzles with zero
      identical-wrong grids.

    Returns:
      config: Three-seed train-then-lock pipeline.

    """
    cfg = exp012()
    cfg.experiment_name = "exp014"
    cfg.screen = "single_view"
    cfg.trigger = "final_only"
    cfg.generator_seeds = (44, 45, 46)
    cfg.generator_names = tuple(f"exp014_s{seed}" for seed in cfg.generator_seeds)
    return cfg


def exp015() -> SudokuTrainLoop:
    """exp002 with the fed-back grid trained and slots seated from learned latents.

    exp002 builds the feedback channel but trained without feeding the decoded
    grid back, and seated slots from zero latents while evaluation starts from
    the learned ones. Both are the pool's settings, changed together: they are
    the two halves of training the recurrence the way it is evaluated. No run
    has isolated feedback without the repair curriculum exp008 bundles it with.

    Hypothesis:
      A recurrence that sees its own decoded answer refines it instead of
      re-deriving it from the puzzle each step, so exact accuracy rises over
      exp002 at equal steps.

    References:
      https://arxiv.org/abs/2510.04871
        Jolicoeur-Martineau. Less is More: Recursive Reasoning with Tiny
        Networks.

    Results:
      TBD. Compare to exp002 at the same seed.

    Returns:
      config: exp002 with feedback and the learned start.

    """
    cfg = exp002()
    cfg.experiment_name = "exp015"
    pool = cfg.step.pool
    if not isinstance(pool, AtomicPool.Config):
        raise TypeError(f"exp002's pool is atomic; got {type(pool)}.")
    pool.feedback = FeedbackCarry.Config()
    pool.start = LearnedStart.Config()
    return cfg


def exp_smoke() -> SudokuTrainLoop:
    """exp000 at minimum size, for verifying an installation end to end.

    Not a result. It answers one question -- is the data prepared and does the
    loop run -- so every axis that costs time without bearing on that answer is
    cut: a few steps, and a network narrow enough to finish in seconds.
    Accuracy will be poor, which is expected.

    Returns:
      config: SudokuTrainLoop with minimal hidden dim and steps.

    """
    cfg = exp000()
    cfg.experiment_name = "exp_smoke"
    cfg.step.model.channels_in = 32
    cfg.step.model.num_layers = 1
    cfg.dataset.batch_size = 8
    cfg.dataset.num_train_puzzles = 4
    cfg.dataset.num_eval_puzzles = 4
    cfg.max_steps = cfg.step.total_train_steps = 4
    cfg.num_steps_eval = 2
    return cfg


def _final_checkpoint(parent: Trainer.Config) -> str:
    """Return the logical path of ``parent``'s last checkpoint."""
    return (
        f"/runs/{parent.experiment_name}/checkpoints/"
        f"step_{int(parent.max_steps):08d}.pt"
    )


# Post-norm, matching the transformer default: a recurrence feeds a block its own
# output, and an unnormalized residual stream compounds when it does.
def _mixer_block() -> MLPMixerBlock.Config:
    """Return an MLP-mixer block shaped for the sudoku grid."""
    return MLPMixerBlock.Config(
        seq_len=-1,
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
