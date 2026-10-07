"""ConvexTok fitting experiments: one measured vocabulary fit per run.

Each experiment is a ``MeasuredFit``. It fits a vocabulary with
``ConvexTokPreparation`` and writes ``tokenizer.json`` and ``metrics.json`` (stage
seconds, peak memory, objective, infeasibility, iterations, vocabulary diff) to
``/opt/scratch/runs/convextok/<experiment>``. The README's "Experiments" section
gives the commands that fetch the raw shards and launch a run.

A fork changes one slot of ``cfg.preparation``: ``solver`` (any callable from a
``LinearProgram`` to a solution with a ``primal`` tensor), ``presolve``,
``rounding``, ``shard_indices`` for a smaller program, or ``vocab_size``. Set
``cfg.reference`` to exp000's ``tokenizer.json`` to count the pieces that moved.
"""

from priml.baselines.convextok.measure import MeasuredFit


def exp000() -> MeasuredFit.Config:
    """ConvexTok's 16K vocabulary on ClimbMix shards 0-6, fitted as upstream fits it.

    PSLP presolve, cuOpt's default PDLP (Stable3) to tolerance 1e-4, postsolve, and
    ``det`` rounding of 16,118 learned pieces; the port's own defaults.

    Hypothesis:
      The published recipe is the bar. A solver, presolve, or rounding change earns
      its place by fitting the same vocabulary at lower time or memory, or, when its
      vocabulary differs, by a lower NanoChat BPB (exp023's recipe with the same
      preparation).

    References:
      https://arxiv.org/abs/2605.22821
        Tempus, Whittington, Schmidt, Komm, Pimentel. Tokenisation via Convex
        Relaxations.
      https://arxiv.org/abs/2507.14051
        Lu, Peng, Yang. cuPDLPx: A further enhanced GPU-based first-order solver for
        linear programming.

    Results:
      One H200: 332 s fitting -- pretokens 46 s, candidates 71 s, program 57 s,
      presolve 75 s (Numba compile included), PDLP 78 s, optimal at 5,600
      iterations. Peak memory 55.8 GiB GPU, 45.1 GiB host. 16,374 pieces, the same
      vocabulary the port fitted for nanochat's exp023 (missing 0, extra 0).

    """
    cfg = MeasuredFit.Config()
    cfg.study_name = "convextok"
    cfg.experiment_name = "exp000"
    return cfg
