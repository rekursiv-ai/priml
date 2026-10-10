"""The sudoku experiment chain.

Every rung solves Sudoku-Extreme: read a 9x9 grid of tokens, predict the whole
completed grid. A puzzle counts only if every cell is right. The 2K proxy is
the first 2,000 test puzzles, scored during training; the full set is all
422,786. The proxy underestimates the full set.

``exp000`` is the attention TRM recipe the chain starts from and is never
edited after it lands. Each later rung forks a named parent and changes one
thing. The chain keeps the baseline, every new best, the cheapest 100%, and
three ways to lock an answer -- a learned verifier committee, agreement
between symmetric views, and a committee with no Sudoku structure at all::

    exp000  attention TRM baseline
      +-- exp001 .. exp006  one training mechanism per rung
            +-- exp007  hypothesis-pinning search at evaluation
            +-- exp008  verifier-committee sieve (harvest, fit, train, sieve)
            +-- exp009  winner-take-all latent hypotheses
            |     +-- exp010  winner recompute, three hypotheses, batch 1,152
            |           +-- exp011  staircase sieve: the cheapest 100%
            |                 +-- exp013  prior-free committee lock
            +-- exp012  nine-view agreement lock (train, then lock)

``exp007`` and ``exp011`` evaluate frozen checkpoints, so their parents run
first: ``exp011`` reads ``exp006``'s and ``exp010``'s generators and
``exp008``'s committee. ``exp013`` fits its own committee on ``exp008``'s
harvest corpus, then runs ``exp011``'s sieve. ``exp008`` and ``exp012`` are
single-GPU pipelines that train their own generator. Run any of them::

    uv --quiet run --frozen python -m priml priml.baselines.sudoku.experiments.exp006
"""

from __future__ import annotations

from priml.baselines.sudoku.agreement_lock import AgreementLockEval
from priml.baselines.sudoku.hps_eval import HpsEval
from priml.baselines.sudoku.hypotheses import (
    HypothesisRecursion,
    WinnerRecompute,
    WinnerTakeAll,
)
from priml.baselines.sudoku.latent_recursion import recipe_block
from priml.baselines.sudoku.reproduction import Reproduction
from priml.baselines.sudoku.sieve import FittedSieve, SieveEval, StaircaseSieveEval
from priml.baselines.sudoku.trainer import Trainer
from priml.baselines.sudoku.verifier import (
    DenoisingVerifier,
    VerifierAcceptor,
    row_column_positions,
)
from priml.baselines.sudoku.verifier_fit import DenoisingLoss, VerifierFit
from priml.model.attention.attention import Attention
from priml.model.norm import RMSNorm
from priml.model.transformer.block import TransformerBlock


def exp000() -> Trainer.Config:
    """Return the attention TRM baseline.

    - Post-norm attention TRM: hidden 512, 8 heads, 2 layers, slow 3 / fast 4.
    - No position tables, no QK-norm, no feedback, no CSP loss.
    - Muon 0.02 on the reasoning matrices, AdamW 1e-4 on tables and heads.
    - Label smoothing 0.2; EMA 0.9 from step 5,000; ACT depth 16.
    - Batch 384, cosine annealing to zero over exactly 62,000 steps; seed 0.
    - Evaluate the 2K proxy every 1,000 steps.

    The trainer's defaults are the ``exp006`` recipe, so this rung switches
    every later mechanism off explicitly and each fork turns one back on.

    Hypothesis:
      The reference TRM recipe with a cosine that anneals over exactly the
      trained steps is a reproducible control. The horizon is load-bearing:
      a 100k horizon cut short by a wall-clock cap spreads 35 pp across seeds.

    References:
      https://arxiv.org/abs/2510.04871
        Jolicoeur-Martineau. Less is More: Recursive Reasoning with Tiny
        Networks.

    Results:
      1x H100, 62k steps: 0.61 exact on the 2K proxy (final, 1 seed).
      Unstable: the 0.61 is one clean seed. Reproduced on 1x H200 (about
      3,120 s each), both seeds diverged and only partly recovered:
        seed 0: at 18k steps (0.247 to 0.012 exact), 0.173 final.
        seed 1: at 8k steps (0.128 to 0.0 exact), 0.1385 final.
      exp002's QK-norm removes the instability.

    """
    cfg = Trainer.Config()
    cfg.study_name = "sudoku"
    cfg.experiment_name = "exp000"
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
    # Mid-training evaluation reads a fixed prefix of the test split as a
    # proxy; reported full-set numbers come from the evaluation rungs.
    cfg.dataset.eval_num_instances = 2_000
    cfg.dataset.augment = True
    return cfg


def exp001() -> Trainer.Config:
    """Fork exp000 with the following changes.

    - Add learned row, column, and box position tables to the grid tokens.

    Hypothesis:
      Attention spends capacity discovering that cell (r, c) shares
      constraints with row r, column c, and its 3x3 box. Learned tables for
      each (RoPE kept, so the change is strictly additive; +13,824
      parameters) hand it that structure.

    References:
      https://arxiv.org/abs/2510.04871
        Jolicoeur-Martineau. Less is More: Recursive Reasoning with Tiny
        Networks.

    Results:
      1x H100, 62k steps: 0.70 exact on the 2K proxy (final, 1 seed), +9 pp
      over exp000.
      Reproduced on 1x H200 (3,372 s): 0.7105 final,
      0.708 last-5 mean.

    """
    cfg = exp000()
    cfg.experiment_name = "exp001"
    cfg.model.pos2d_grid_shape = (0, 0)
    return cfg


def exp002() -> Trainer.Config:
    """Fork exp001 with the following changes.

    - Add parameter-free QK-norm to attention.
    - Shorten the cosine horizon from 62,000 to 12,000 steps; evaluate every 500.

    The two travel together: QK-norm converges near 10k steps, and training
    past mastery at high learning rate collapses the ACT depth, so the cosine
    must anneal inside that window.

    Hypothesis:
      A shared affine-free RMSNorm over per-head queries and keys bounds the
      attention logits, so training converges several times sooner.

    References:
      https://arxiv.org/abs/2010.04245
        Henry et al. Query-Key Normalization for Transformers.

    Results:
      1x H100: about 5x faster convergence, 0.71 exact on the 2K proxy at 10k
      steps against 0.15 at 10k without QK-norm (1 seed). QK-norm is also
      what keeps the deep recurrence of exp005 trainable.
      Reproduced on 1x H200 (643 s): 0.698 at 10k,
      0.7175 final at 12k.

    """
    cfg = exp001()
    cfg.experiment_name = "exp002"
    assert isinstance(cfg.model.block, TransformerBlock.Config)
    attn = cfg.model.block.attn
    assert isinstance(attn, Attention.Config)
    attn.norm_qk = RMSNorm.Config()
    cfg.max_steps = 12_000
    cfg.total_train_steps = 12_000
    cfg.num_steps_eval = 500
    return cfg


def exp003() -> Trainer.Config:
    """Fork exp002 with the following changes.

    - ACT depth 32 (was 16).
    - No label smoothing (was 0.2).
    - EMA 0.999 from step 0 (was 0.9 from step 5,000).
    - CSP row/column/box cardinality loss at weight 0.3.
    - Augmentation permutes digit tokens only, never the blank marker.

    One rung because each was measured alone first and the blog reports
    them as one training stack.

    Hypothesis:
      The reference TRM's own training choices, plus a differentiable
      cardinality loss, help and help together.

    References:
      https://arxiv.org/abs/2307.04895
        Yang, Lee & Park. Learning to Solve Constraint Satisfaction Problems
        with Recurrent Transformer.

    Results:
      1x H100, 12k steps: 0.78 exact on the 2K proxy (last-5 mean); the
      config family's mean is 0.755-0.763 with a seed spread of about 2.4 pp.
      Reproduced on 1x H200: seed 0 (694 s) 0.7185 final, 0.727 last-5
      mean, below the family band; seed 1 (834 s) 0.770 final, 0.781
      last-5 mean.

    """
    cfg = exp002()
    cfg.experiment_name = "exp003"
    cfg.max_act_steps = 32
    cfg.label_smoothing = 0.0
    cfg.ema_decay = 0.999
    cfg.ema_warmup_steps = 0
    cfg.csp_loss_weight = 0.3
    cfg.dataset.augment_digits_only = True
    return cfg


def exp004() -> Trainer.Config:
    """Fork exp003 with the following changes.

    - Feed each ACT step its previous decoded grid, and corrupt that grid in
      training: with probability 0.5 per slot, about 12 non-given cells.

    The feedback table is zero-initialized, so the step-0 forward is
    bit-identical to exp003's.

    Hypothesis:
      Repairing a corrupted copy of its own answer teaches the recurrence an
      explicit repair skill it otherwise lacks, as in discrete diffusion.

    References:
      https://arxiv.org/abs/2107.03006
        Austin et al. Structured Denoising Diffusion Models in Discrete
        State-Spaces.

    Results:
      1x H100, 12k steps: 0.82 exact on the 2K proxy (last-5 mean); +4.3 to
      +5.9 pp over exp003, paired on 3 seeds.
      Reproduced on 1x H200: seed 0 (906 s) 0.7905 final, 0.793 last-5
      mean; seed 1 (733 s) 0.7955 final, 0.803 last-5 mean. +6.6 and
      +2.2 pp over exp003, paired by seed.

    """
    cfg = exp003()
    cfg.experiment_name = "exp004"
    cfg.feedback = True
    return cfg


def exp005() -> Trainer.Config:
    """Fork exp004 with the following changes.

    - Deep recurrence: slow 6 / fast 9 cycles (was 3 / 4).
    - Seed 44, the seed the measured lineage carries forward to exp006.

    Hypothesis:
      Depth and repair compound: more cycles per ACT step give the repair
      skill more room to act. QK-norm (exp002) keeps the deeper stack
      trainable.

    References:
      https://arxiv.org/abs/2510.04871
        Jolicoeur-Martineau. Less is More: Recursive Reasoning with Tiny
        Networks.

    Results:
      1x H100, 12k steps: 0.8634 and 0.8972 exact on the 2K proxy (last-5
      mean, two paired seeds); depth with repair adds +5.9 to +7.5 pp paired.
      Still climbing at 12k: the anneal is too short for this depth.
      Reproduced on 1x H200 (2,452 s): 0.8915 final,
      0.887 last-5 mean.

    """
    cfg = exp004()
    cfg.experiment_name = "exp005"
    cfg.model.slow_cycles = 6
    cfg.model.fast_cycles = 9
    cfg.seed = 44
    return cfg


def exp006() -> Trainer.Config:
    """Fork exp005 with the following changes.

    - Anneal over 19,500 steps (was 12,000), the longest that fits one
      H100-hour.

    Hypothesis:
      Mastery arrives later on the deep repair recipe, so a longer cosine
      turns exp005's un-annealed climb into a higher plateau.

    References:
      https://arxiv.org/abs/2510.04871
        Jolicoeur-Martineau. Less is More: Recursive Reasoning with Tiny
        Networks.

    Results:
      1x H100, 19.5k steps: 0.9300 exact on the 2K proxy (last-5 mean; final
      0.9295). Full set, deterministic pass: 0.9624 (406,892/422,786).
      Augmentation drew from the ambient global RNG
      (``dataset.augment_seed = None``).
      Reproduced on 1x H200 (3,825 s): 0.923 final,
      0.9245 last-5 mean.

    """
    cfg = exp005()
    cfg.experiment_name = "exp006"
    cfg.max_steps = 19_500
    cfg.total_train_steps = 19_500
    return cfg


def exp007() -> HpsEval.Config:
    """Fork exp006 with the following changes.

    - Evaluate with hypothesis-pinning search: accept a grid only when the
      halt logit q >= 7.875 at ACT steps 24, 28 and 32; otherwise pin a digit
      into the most uncertain cell as an extra given and re-run, as a small
      tree search hps(5, 3, 2, 512).

    Scores exp006's final checkpoint, so run exp006 first.

    Hypothesis:
      The halt head's confidence is a usable acceptance signal, so most of
      exp006's residual 3.8% is recoverable without retraining.

    References:
      https://arxiv.org/abs/2510.04871
        Jolicoeur-Martineau. Less is More: Recursive Reasoning with Tiny
        Networks.

    Results:
      1x H100, about 2 h: 0.99920 exact on the full set (422,447/422,786),
      against exp006's 0.9624. 321 accepted grids are wrong: confidence alone
      is not a sound acceptance gate, which exp008 and exp012 address.
      Reproduced on 1x H200 over this chain's exp006
      (8,337 s): 0.99957 (422,604/422,786), 142 accepted grids wrong.

    """
    parent = exp006()
    cfg = HpsEval.Config()
    cfg.experiment_name = "exp007"
    cfg.checkpoint_path = (
        f"/runs/{parent.experiment_name}/checkpoints/"
        f"step_{int(parent.max_steps):08d}.pt"
    )
    cfg.model = parent.copy_tree().finalize().model
    return cfg


def exp008() -> Reproduction.Config:
    """Fork exp006 with the following changes.

    - Lock answers with a committee of three learned verifiers: a grid
      retires only when all three accept it (minimum grid logit above 0).
    - Evaluate as a progressive sieve: one deterministic pass, then the
      nine-view HPS ladder on survivors only, then an escalated search
      hps(7, 4, 3, 8,400) and a modal fallback on the residue.

    A single-GPU pipeline: harvest candidate grids from exp006's final
    checkpoint (run exp006 first), fit the three verifiers (seeds 0/1/2),
    then train a fresh exp006 generator, screening each 1,000-step checkpoint
    under the committee and sieving the full set from the first perfect
    screen on.

    Hypothesis:
      A small network that sees only (puzzle, candidate) and attends along
      rows, columns and boxes can learn to verify solutions. Three seeds err
      on different grids, so unanimity is a sound lock, and it makes the
      cheap deterministic pass usable: it retires about 96% of puzzles.

    References:
      https://arxiv.org/abs/2307.04895
        Yang, Lee & Park. Learning to Solve Constraint Satisfaction Problems
        with Recurrent Transformer.

    Results:
      1x H100, sieve on exp006's frozen checkpoint: 422,786/422,786, zero
      false accepts over about 440K live accepts, 3,274 GPU-s (the
      deterministic pass retires about 96.2% in about 666 s). This
      train-then-sieve pipeline read 422,786/422,786 at two checkpoints
      before the run crashed for unrelated infrastructure reasons.

    """
    generator = exp006()
    model = generator.copy_tree().finalize().model
    cfg = Reproduction.Config()
    cfg.study_name = "sudoku"
    cfg.experiment_name = "exp008"
    cfg.screen = "committee"
    cfg.harvest_source_checkpoint = (
        f"/runs/{generator.experiment_name}/checkpoints/"
        f"step_{int(generator.max_steps):08d}.pt"
    )
    cfg.harvest.model = model
    cfg.generator = generator
    cfg.generator_name = "exp008_generator"
    cfg.verifier_names = tuple(f"exp008_verifier_s{seed}" for seed in range(3))
    cfg.full_eval = SieveEval.Config()
    cfg.full_eval.model = model
    return cfg


def exp009() -> Trainer.Config:
    """Fork exp006 with the following changes.

    - Train two fast-latent hypotheses per puzzle from one shared slow latent;
      only the hypothesis with the lowest cross-entropy trains (winner take
      all), with gradient through both. Each seated puzzle draws fresh
      hypotheses aligned to the first FFN's top subspace.
    - Anneal over 22,000 steps (was 19,500).

    The two travel together: the campaign gave this mechanism two H100-hours,
    and 22,000 steps is what two hypotheses complete in them.

    Hypothesis:
      Competing latent hypotheses specialize, so the deployed hypothesis 0
      solves more puzzles in a single pass.

    References:
      https://arxiv.org/abs/2601.19085
        Speed is Confidence.

    Results:
      1x H100, 22k steps (8,121 s): 0.9650 exact on the 2K proxy; full set,
      deterministic pass: 0.9876297 (417,556/422,786), against exp006's
      0.9624.
      Reproduced on 1x H200 (7,282 s): 0.971 final,
      0.967 last-5 mean on the 2K proxy.

    """
    cfg = exp006()
    cfg.experiment_name = "exp009"
    model = cfg.model = HypothesisRecursion.Config().update(cfg.model)
    if cfg.seed is None:
        raise ValueError("Expected cfg.seed is not None.")
    model.hypothesis_seed = cfg.seed
    cfg.routing = WinnerTakeAll.Config()
    cfg.max_steps = 22_000
    cfg.total_train_steps = 22_000
    return cfg


def exp010() -> Trainer.Config:
    """Fork exp009 with the following changes.

    - Route the hypotheses without gradient, then recompute only the winner
      with it.
    - Three hypotheses (was two).
    - Batch 1,152 (was 384) and a 10,000-step anneal (was 22,000).

    One change: recomputing the winner is what frees the activation memory
    the third hypothesis and the larger batch spend.

    Hypothesis:
      Winner recomputation keeps one H100 within memory while the third
      hypothesis and 11.52M puzzle slots raise single-pass accuracy beyond
      exp009's.

    References:
      https://arxiv.org/abs/2601.19085
        Speed is Confidence.

    Results:
      1x H100, 10k steps (14,456 s): 0.9770 exact on the 2K proxy; full set,
      deterministic pass: 0.9902007 (418,643/422,786).
      Reproduced on 1x H200 (13,729 s): 0.968 final,
      0.963 last-5 mean on the 2K proxy.

    """
    cfg = exp009()
    cfg.experiment_name = "exp010"
    cfg.routing = WinnerRecompute.Config()
    assert isinstance(cfg.model, HypothesisRecursion.Config)
    cfg.model.num_hypotheses = 3
    cfg.dataset.batch_size = 1_152
    cfg.max_steps = 10_000
    cfg.total_train_steps = 10_000
    return cfg


def exp011() -> StaircaseSieveEval.Config:
    """Fork exp010 with the following changes.

    - Evaluate as a staircase sieve under exp008's committee lock: exp010's
      deterministic pass over every puzzle; on survivors, the highest-q grid
      of six latent hypotheses (hypothesis 0 plus five fresh draws); then
      exp010 under exp007's search; then exp008's nine-view ladder and
      escalated tail on exp006's checkpoint.

    Scores frozen checkpoints, so run exp006, exp008 and exp010 first.

    Hypothesis:
      exp010's single pass leaves about a quarter of exp006's failures, at
      the same cost, so the sieve's expensive view rounds see a quarter of
      the puzzles. Every round only proposes grids; the committee alone
      accepts, so the 100% stays sound.

    References:
      https://arxiv.org/abs/2601.19085
        Speed is Confidence.

    Results:
      1x H100: 422,786/422,786 on the full set, zero wrong locks, 1,598.6
      GPU-s, 2.05x cheaper than exp008's sieve.

    """
    parent = exp010()
    hypothesis_model = parent.copy_tree().finalize().model
    assert isinstance(hypothesis_model, HypothesisRecursion.Config)
    cfg = StaircaseSieveEval.Config()
    cfg.experiment_name = "exp011"
    cfg.hypothesis_model = hypothesis_model
    cfg.hypothesis_checkpoint_path = (
        f"/runs/{parent.experiment_name}/checkpoints/"
        f"step_{int(parent.max_steps):08d}.pt"
    )
    ladder = exp006()
    cfg.model = ladder.copy_tree().finalize().model
    cfg.checkpoint_path = (
        f"/runs/{ladder.experiment_name}/checkpoints/"
        f"step_{int(ladder.max_steps):08d}.pt"
    )
    committee = exp008()
    fit = committee.verifier
    cfg.verifier_checkpoints = tuple(
        str(fit.run_dir).format(experiment_name=name)
        + f"/checkpoints/step_{fit.max_steps:08d}.pt"
        for name in committee.verifier_names
    )
    return cfg


def exp012() -> Reproduction.Config:
    """Fork exp006 with the following changes.

    - Lock answers by agreement: solve each puzzle with exp007's search on
      the identity view and on a symmetric copy; two agreeing grids lock, and
      a disagreement takes the modal grid over all nine views.
    - Train the generator in 1,000-step resumed segments with augmentation
      drawn from a dedicated stream seeded 44 (exp006 draws from the ambient
      global RNG), so a crashed run resumes to the same weights.

    A single-GPU pipeline: train, screen each 1,000-step checkpoint on the
    first 5,000 test puzzles once a cheap deterministic gate opens, then lock
    the full set at the first perfect screen and at the final checkpoint.

    Hypothesis:
      Two exact symmetries of one puzzle rarely produce the same WRONG grid,
      so agreement is a sound, verifier-free acceptance gate. Closing
      exp007's 321 wrong accepts should need about two solves per puzzle.

    References:
      https://arxiv.org/abs/2510.04871
        Jolicoeur-Martineau. Less is More: Recursive Reasoning with Tiny
        Networks.

    Results:
      1x H100: 422,786/422,786 on the full set at the final checkpoint, zero
      wrong locks, 2.005 solves per puzzle. 25,080.9 GPU-s (7.0 GPU-h) in
      all: training 2,985.8 GPU-s, screens 1,121.8 GPU-s, the full-set lock
      20,973.4 GPU-s. The gate never opened before step 19,500, so the first
      perfect screen and the first perfect full set are both the final
      checkpoint.

    """
    generator = exp006()
    cfg = Reproduction.Config()
    cfg.study_name = "sudoku"
    cfg.experiment_name = "exp012"
    cfg.generator = generator
    cfg.generator_name = "exp012_generator"
    lock = cfg.full_eval = AgreementLockEval.Config()
    lock.model = generator.copy_tree().finalize().model
    return cfg


def exp013() -> FittedSieve.Config:
    """Fork exp011 with the following changes.

    - Lock with a committee that encodes no Sudoku structure: three plain
      full-attention verifiers with row and column coordinates only (no
      group mask, no boxes), each also trained to fill masked digits of its
      candidate. Fit them (seeds 0/1/2, 16,000 steps) on exp008's harvest
      corpus when their checkpoints are missing.

    Run exp008 first for its corpus, and exp011's parents for the sieve.

    Hypothesis:
      Predicting a masked digit requires reading exactly the cell's row,
      column and box peers, so the denoising task makes the verifier
      discover the constraint graph that exp008 hands its committee. Such a
      committee locks soundly, so the 100% needs no structural prior.

    References:
      https://arxiv.org/abs/2107.03006
        Austin et al. Structured Denoising Diffusion Models in Discrete
        State-Spaces.

    Results:
      1x H100, sieve: 422,786/422,786 on the full set, zero wrong locks
      (422,782 locked, 4 modal), 1,613.6 GPU-s. The recorded committee was
      fit on an earlier harvest corpus; this rung fits on exp008's.

    """
    cfg = FittedSieve.Config()
    cfg.experiment_name = "exp013"
    harvest = exp008().harvest
    corpus = str(harvest.run_dir).format(experiment_name=harvest.experiment_name)
    for seed in range(3):
        member = VerifierFit.Config()
        member.experiment_name = f"exp013_verifier_s{seed}"
        member.seed = seed
        member.max_steps = 16_000
        model = member.model = DenoisingVerifier.Config()
        model.attention_mask = None
        model.position_index = row_column_positions
        member.loss = DenoisingLoss.Config()
        member.dataset.harvest_dir = corpus
        cfg.members.append(member)
    cfg.sieve = exp011()
    acceptor = cfg.sieve.acceptor
    assert isinstance(acceptor, VerifierAcceptor.Config)
    acceptor.model = cfg.members[0].model.copy_tree()
    return cfg


def exp_smoke() -> Trainer.Config:
    """Fork exp000 for an unscored installation check.

    - Automatically select the available device; no compilation.
    - Width 32, one layer, slow 1 / fast 1, ACT depth 2.
    - Batch 8, four train steps, evaluate four puzzles twice.
    - No checkpoints.
    """
    cfg = exp000()
    cfg.experiment_name = "exp_smoke"
    # A smoke run validates the installation it runs on. Scored rungs pin
    # CUDA so a scheduler mistake cannot silently produce a CPU result; this
    # unscored rung keeps the runtime's best-device fallback.
    cfg.runtime.device = None
    cfg.model.compile = False
    cfg.model.channels_in = 32
    cfg.model.num_layers = 1
    cfg.model.slow_cycles = 1
    cfg.model.fast_cycles = 1
    cfg.max_act_steps = 2
    cfg.dataset.batch_size = 8
    cfg.dataset.eval_num_instances = 4
    cfg.max_steps = 4
    cfg.total_train_steps = 4
    cfg.num_steps_eval = 2
    cfg.eval_warmup_batches = 0
    cfg.checkpointer = None
    return cfg
