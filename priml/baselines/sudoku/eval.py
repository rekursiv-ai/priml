"""The complete evaluation/verification protocol for the trained generator.

One module, six stages -- symmetry views, hypothesis-pinning search, the
verifier committee, the verifier-corpus harvest, the full-set eval engines,
and the end-to-end reproduction pipelines -- in pipeline order:

- Symmetry views (TTA): exact-symmetry transforms + the frozen view tables.
- Hypothesis-pinning search (HPS): learned-q acceptance search around a
  frozen TRM.
- Verifier: net, training data stream, dedicated fit, committee acceptor.
- Harvest: the verifier's on-policy candidate training corpus.
- Eval engines: the single-checkpoint HPS eval, the agreement-lock
  committee, and the verifier sieve.
- Reproduction pipelines: the staged from-scratch runner behind the three
  end-to-end reproductions (:class:`Reproduction`).


Symmetry views (TTA)
====================

Exact-symmetry test-time-augmentation (TTA) views for sudoku token grids.

Sudoku's exact symmetry group lets ONE frozen checkpoint act as several vote
members: evaluate a symmetry-transformed copy of each puzzle, then map the
prediction back through the inverse transform. A recorded view composes
(applied in this order):

- optional transpose of the 9x9 grid positions,
- band-preserving row and column permutations (each 3-block of lines must map
  onto a 3-block, so bands map to bands and stacks to stacks),
- digit relabeling: a permutation of digits 1-9 applied to the token values
  (tokens 2-10 encode digits 1-9; pad 0 and blank 1 are fixed points).

Every component is an exact Sudoku symmetry, so a transformed puzzle is a
legal Sudoku with the transformed unique solution: givens map to givens and
solutions to solutions. Search runs entirely in transformed space; root and
final grids are mapped back through :func:`invert_symmetry` before any
metric or vote sees them, so exact flags, dumps, and modal voting all
operate in original space against the original labels.

:data:`NINE_VIEWS` and :data:`SEED_ENSEMBLE_VIEWS` are FROZEN evaluation
tables -- the recorded view orders are the committee tie-break orders and
must not be edited.


Hypothesis-pinning search (HPS)
===============================

Hypothesis-pinning search (HPS) with learned-q acceptance.

HPS is a test-time search protocol wrapped around a FROZEN trained TRM. One
network plays three roles: proposer (its logits pick branches), propagator
(it is re-rolled per hypothesis), and pruner (its trained q-halt head prices
branches). The protocol signature is ``hps(C, K, A, budget)``:

1. **Deterministic root rollout** (ACT <= ``max_act_steps``, native 32). The
   root grid is accepted iff its learned-q score clears the acceptance
   policy; accepted puzzles never search.
2. **On rejection, branch**: pick the model's highest-entropy undecided
   (non-given) cell, take its top-``C`` logit-ordered digits, and PIN each
   digit into the INPUT tokens as an extra given (the trained model never
   overrides a given, so the givens channel is a trustworthy
   hypothesis-injection API). Re-run the full recurrence from fresh
   ``init_z``. Expand candidate-parallel breadth-first to depth ``K``; on
   tree exhaustion restart from the next root entropy cell, up to ``A``
   attempts; at most ``budget`` node rollouts per puzzle.
3. **Learned-q acceptance** (the recipe policy): a grid is accepted iff its
   q-halt logit is ``>= acceptance_threshold`` (7.875) at EVERY ACT step in
   ``acceptance_checkpoints`` ((24, 28, 32)); candidates run the full ACT
   depth (``early_exit_at_q8=False``). No rule predicate, no labels.

Ratified defaults ``hps(C=5, K=3, A=2, budget=512)``; escalation preset
``hps(7, 4, 3, 8400)`` (:func:`escalated_search_config`). Sibling hypotheses
batch together (the fast engine's win) and every forward is capped at
``search_max_rows`` (2,048) rows.

An alternative grid-predicate acceptance path (``accept_fn``) drives the same
tree with the sound label-free validity rule (:func:`accepted_grids`) or a
verifier-committee predicate, with the q-halt@8 early exit pricing false
branches -- the committee dev-screen engine.

Seam contract: :func:`run_search` / :meth:`HpsSearch.run` consume a bare
:class:`~priml.baselines.sudoku.trm.TRM` (eval weights -- the EMA
swap is the CALLER's job, e.g. via the trainer's EMA ``apply_to``) and one
dataset batch dict (``media`` / ``valid_count`` / ``puzzle_identifiers``;
``label`` is never read). :func:`det_pass` wires to
:meth:`~priml.baselines.sudoku.trainer.Trainer.eval_loss`, the
trainer's conditional-depth eval seam (per-row release at q >= +2).

Packed row layout (the dump schema shared with the lock engines, the
harvest, and the repro pipelines; width ``learned_hps_output_width(G) = 7 + 2 * G`` float32
columns)::

    col 0        : score       final learned-q of the emitted grid
    col 1        : accepted    1.0 accepted / 0.0 not
    col 2        : nodes       candidate rollouts spent
    col 3        : depth       pins at acceptance (0 root, -1 unaccepted)
    col 4        : candidates  scored root (1) + nodes
    col 5        : solution_visited   label-joined diagnostic (0 if unjoined)
    col 6        : algorithmic_decision_calls   always 0 (no rule predicate)
    cols 7..7+G  : root grid tokens (deterministic rollout argmax)
    cols 7+G..   : final grid tokens (accepted grid, else the root grid)

:func:`write_member_dump` / :func:`read_member_dump` persist these rows as an
npz member dump alongside ``media`` (and optionally ``label``).


Verifier
========

Learned Sudoku solution verifier: net, data stream, dedicated fit, acceptor.

The verifier classifies "is this candidate grid THE solution to this puzzle?"
from the (puzzle, candidate) token rows alone -- no solver internals (no
q-halt, no latents, no generator identity) and no programmatic Sudoku-rule
oracle at inference. Labels come from ground-truth solutions offline:
``label = (candidate == solution)``.

Four pieces, one file:

* :class:`SudokuVerifier` -- a tiny transformer (width 128, depth 4, heads 4,
  ~0.8M params) whose attention is masked to the 27-group Sudoku constraint
  graph: each cell attends only to the 20 peers sharing its row, column, or
  box. The structural bias -- not a rule oracle; the consistency predicate
  itself is learned -- is what buys perfect dev separation at threshold 0.
* :class:`VerifierData` -- the on-device training stream (generated positives
  / swap negatives / Hamming-corruption negatives + real solver candidates
  from the harvest corpus) and the frozen eval strata.
* :class:`VerifierFit` -- the DEDICATED training run: plain AdamW + cosine
  annealing, bf16 autocast, gradient clip 1.0, no EMA / Muon / lr_scale.
  Deliberately not the generator ``Trainer`` (its schedule cannot express
  this recipe; the committee's zero-false-accept property depends on it).
* :class:`VerifierAcceptor` -- the frozen committee lock: min over member
  grid logits must STRICTLY exceed the threshold (default 0, unanimity).

Checkpoint layout (written by :meth:`VerifierFit.run`, consumed by
:class:`VerifierAcceptor`): ``{"step": {"model": <state_dict>,
"global_step": <int>}}`` at
``<scratch>/runs/<experiment_name>/checkpoints/step_<max_steps:08d>.pt``.

Harvest shard contract (produced by :class:`Harvest`): files ``shard-*.npz``
under ``harvest_dir``, each with fields ``original`` / ``candidate``
(``[N, 81]`` token rows), ``flat_view_id`` (``[N]`` row index into the train
split; invariant ``original == train_inputs[flat_view_id]``), and
``base_group_id`` (``[N]`` in ``[0, 8)``; 8 base puzzle groups per shard,
globalized as ``base_group_id + 8 * shard_index`` in sorted-filename order).
The real corpus carries >= 800 global groups; groups ``[0, 700)`` train,
``[700, 750)`` calibrate, ``[750, 800)`` are internal holdout (the defaults;
scale the ends down for smaller corpora).


Harvest
========

Verifier-corpus harvest: on-policy candidate rows from a frozen generator.

The verifier's "real solver candidate" training stratum is harvested by
rolling a FROZEN trained generator (any recipe-class checkpoint; generator
transfer is proven) over deterministic augmentation views of the source train
split. Three label-free row generators run per selected view:

* **Root rollouts** (1 row/view): the plain deterministic full-ACT rollout of
  the view board (``source_kind`` 0).
* **Exhaustive fixed-HPS nodes** (60 rows/view at the C=5, K=2, A=2 defaults):
  every node of the complete depth-``search_depth`` pin tree -- top-C
  logit-ordered digits pinned at the max-entropy undecided cell, A root-cell
  restarts -- WITHOUT any acceptance pruning, so the node census is fixed and
  deployment-distributed (``source_kind`` 1).
* **Random-unstuck starts** (8 rows/view at the (4, 12, 24, 40) x 2 defaults):
  the root candidate grid is corrupted at ``strength`` random blank cells and
  re-rolled as the initial feedback state of the ORIGINAL board -- the
  "recover from a wrong basin" distribution (``source_kind`` 2).

Shard contract (consumed by the verifier's harvest loader, which owns the
authoritative statement): files ``shard-<i>.npz`` under ``out_dir``, fields
``original`` / ``candidate`` (``[N, 81]`` uint8 token rows), ``flat_view_id``
(``[N]`` row index into the train split; invariant ``original ==
train_inputs[flat_view_id]``) and ``base_group_id`` (``[N]`` in ``[0, 8)``; 8
consecutive base puzzle groups per shard, globalized as ``base_group_id + 8 *
shard_index`` in sorted-filename order). Extra provenance fields (``view_id``,
``candidate_index``, ``source_kind``, ``start_seed``, ``corruption_strength``,
``q_halt``, ``current_state``, ``checkpoints``, ``exact_solved``) ride along
for replay and diagnostics; the loader ignores them. At the defaults (800
groups x 2 views x 69 rows) the corpus is 110,400 rows in 100 shards.

RNG discipline (dedicated generators only; row generation never consumes
the ambient torch RNG -- construction's ``config.model.make()`` does draw
global-RNG init weights, but the strict checkpoint load overwrites every
parameter, so shard bytes stay checkpoint-determined):

* View selection: per base group ``g`` (ascending), a fresh CPU generator
  seeded ``seed + g * 1_000_003`` draws ONE ``randperm`` over the group's
  augmentation rows; the first ``views_per_group`` entries are kept in draw
  order (:func:`select_harvest_views`).
* Unstuck starts: per (puzzle row, strength, repeat) -- rows ascending,
  strengths in declared order, repeats innermost -- a fresh CPU generator
  seeded ``seed + group * 10_000_019 + view * 10_007 + start_index`` draws
  ``randperm(n_blanks)`` then ``randint(1, 9, (cells,))``
  (:func:`make_random_unstuck_starts`).

Mapping from the original harvest: that job ran through the train-loop eval
protocol with a hardcoded cluster ``output_dir`` and a named cluster
checkpoint; here ``out_dir`` and ``harvest_source_checkpoint`` are config
fields and :meth:`Harvest.run` drives the model directly. The internal
per-range ``partition`` column is dropped: the verifier splits
train/calibration/holdout by GLOBAL group-id ends (700/750/800 on the full
corpus) instead. One shard = 8 consecutive base groups, matching the source
job's sharding (one 16-view eval batch = 8 groups x 2 views).

The dev-5K search-node dump (fields ``media`` / ``final_prediction`` /
``global_index``) that verifier eval stratum 2 consumes is OPTIONAL and never
trains; :meth:`Harvest.regenerate_node_corpus` rebuilds it from the same
frozen generator.


Eval engines
============

The three full-set eval engines: the single-checkpoint HPS eval, the
agreement-lock committee, and the verifier sieve.

:class:`HpsEval` -- the canonical single-model search protocol over one
frozen checkpoint: a deterministic root rollout per puzzle, learned-q
acceptance, and the fast pin search on rejected rows -- label-free end to
end; labels join post hoc for the metrics (incl. the ``solution_visited``
false-reject diagnostic).

The two lock engines evaluate a frozen generator checkpoint (or several)
over the ordered test prefix and decide, label-free, which decoded grids to
LOCK as final answers. Labels join only in the post-hoc metrics.

:class:`AgreementLockEval` -- the lazy sequential committee (V3 rule):
evaluate the members in the frozen order; if the FIRST TWO final grids agree
on a puzzle, lock that grid (cost exactly 2 HPS passes); on any disagreement,
run the remaining members on the disagreeing puzzles only (survivor-targeted
via the dataset's ``eval_instance_indices``) and take the plain modal grid
(ties to the first-listed member). Structural soundness: after a first-pair
disagreement unanimity cannot resume, so the only wrong-lock channel is the
first pair's identical-wrong count (reported as ``eval/wrong_locks``; the
frozen member orders choose first pairs measured at zero). There is NO
deterministic pass and no verifier in this engine: every member is a
learned-q HPS pass. Members are ``(checkpoint, view)`` pairs, so one config
expresses both the nine-view committee (one checkpoint under
:data:`NINE_VIEWS`) and the seed-ensemble
committee (three checkpoints plus recorded views of the first).

:class:`SieveEval` -- the verifier-locked progressive sieve:

- R0: conditional-halt deterministic pass (per-row release at q >=
  ``det_halt_threshold``) over the whole population; every decoded grid is
  offered to the frozen verifier-committee lock (unanimity above threshold
  0 -- the only soundness-bearing component). Locked puzzles retire.
- R1..Rk: the recorded views (canonical first), each a learned-q HPS pass
  over the SURVIVORS only, each output offered to the same lock. False
  rejections are safe: they ride to the next round.
- TAIL: an escalated search pass (``tail_search``, proven (7, 4, 3, 8400))
  on the residuals, locked; anything still unlocked takes the modal grid
  over every collected candidate (first-round-listed tie-break).

The committee lock is constructed and released around every round's offer so
no resident committee coexists with the next round's compiled generator (the
internal run died with a SIGSEGV when it did).

Checkpoint loading (:func:`load_eval_weights`, shared by every eval-side
loader) accepts BOTH schemas: full trainer checkpoints
(``state["step"]["model"]`` plus the flat EMA shadow ``state["step"]["ema"]``)
and HF model-only files (top-level ``state["model"]``). Eval weights are
ALWAYS the EMA weights on the recipe (EMA warmup 0), so a full checkpoint's
shadow overlays the live parameters exactly like the trainer's eval-time EMA
swap; model-only HF files are exported EMA-applied already and load as-is.

Outputs follow the run-dir convention: metrics JSON and npz dumps land under
``<scratch>/runs/{experiment_name}/dumps/`` (the agreement lock also
writes one packed member dump per member -- the packed search row layout).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from enum import IntEnum
from functools import partial
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Final,
    Literal,
    NamedTuple,
    NotRequired,
    Protocol,
    Self,
    TypedDict,
    cast,
    override,
)

import hashlib
import json
import logging
import math
import os
import time
import zipfile

from configgle import DataclassLike, Fig, Makeable
from torch import Tensor, nn
from torch._inductor import config as inductor_config
from torch.nn import functional
from tqdm import tqdm

import numpy as np
import torch

from priml.baselines.sudoku.puzzle_data import (
    PuzzleBatch,
    PuzzleDataset,
    load_puzzle_dataset,
)
from priml.baselines.sudoku.puzzle_spec import SudokuSpec
from priml.baselines.sudoku.trainer import (
    EvalTimeLimitError,
    Trainer,
    resolve_run_dir,
    sudoku_group_indices,
)
from priml.baselines.sudoku.trm import TRM
from priml.cost import Cost, cost, elementwise_cost, reduction_cost
from priml.lib.codec import ReadError, from_plain, loads
from priml.model.attention.kernel import attention_kernel_cost
from priml.model.embedding import Embedding
from priml.model.linear import Linear
from priml.model.norm import LayerNorm
from priml.paths import resolve_working_dir
from priml.runtime import SingleProcess, get_device
from priml.train.tracker import WandbTracker


if TYPE_CHECKING:
    from numpy.lib.npyio import NpzFile
    from numpy.typing import NDArray


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Symmetry views (TTA).
# ---------------------------------------------------------------------------


IDENTITY_DIGITS: Final = (1, 2, 3, 4, 5, 6, 7, 8, 9)
"""Identity digit relabeling (digit d stays d)."""

IDENTITY_LINES: Final = (0, 1, 2, 3, 4, 5, 6, 7, 8)
"""Identity row/column permutation (line i stays i)."""

REVERSED_LINES: Final = (8, 7, 6, 5, 4, 3, 2, 1, 0)
"""Reversed row/column permutation (bands map onto bands in reverse)."""


class View(NamedTuple):
    """One recorded exact Sudoku symmetry.

    Attributes:
      name: Stable view label (member name in committee tables).
      digit_permutation: Relabeled digit for each of 1..9.
      transpose: Whether to transpose the 9x9 grid positions.
      row_permutation: Source row per output row; maps bands onto bands.
      col_permutation: Source column per output column; maps stacks onto
        stacks.

    """

    name: str
    digit_permutation: tuple[int, ...]
    transpose: bool
    row_permutation: tuple[int, ...] = IDENTITY_LINES
    col_permutation: tuple[int, ...] = IDENTITY_LINES

    def apply(self, grid: Tensor) -> Tensor:
        """Apply this view's recorded symmetry (see :func:`apply_symmetry`).

        Args:
            grid: Token grid to transform.

        Returns:
            transformed: Transformed token grid.

        """
        return apply_symmetry(
            grid,
            digit_permutation=self.digit_permutation,
            transpose_grid=self.transpose,
            row_permutation=self.row_permutation,
            col_permutation=self.col_permutation,
        )

    def invert(self, grid: Tensor) -> Tensor:
        """Invert this view's recorded symmetry (see :func:`invert_symmetry`).

        Args:
            grid: Transformed token grid.

        Returns:
            original: Original-space token grid.

        """
        return invert_symmetry(
            grid,
            digit_permutation=self.digit_permutation,
            transpose_grid=self.transpose,
            row_permutation=self.row_permutation,
            col_permutation=self.col_permutation,
        )


# The frozen nine-view committee table (tie-break order). "Trained class" =
# transform families sampled by train-time augmentation (digit permutations +
# dihedral); stack_rot_digit_shift2 and full_mix carry band/stack/line
# components that training never saw.
#
#   view                   transform                          class
#   canonical              identity (the plain pass)          trained
#   shift1                 digits cyclic +1                   trained
#   shift4                 digits cyclic +4                   trained
#   reversal               digits 1<->9, 2<->8, ...           trained
#   anti_transpose         reflect across the anti-diagonal   trained
#                          (transpose + reverse rows/cols)
#   stack_rot_digit_shift2 rotate the three column-stacks     untrained
#                          cyclically + digits cyclic +2      (stack perm)
#   full_mix               row + column perms + transpose     untrained
#                          + digit scramble, composed         (deep comp.)
#   dg_shift1_rot180       digits cyclic +1; rotate 180       trained
#                          (reverse rows + columns)
#   dg_shift3_pure         digits cyclic +3                   trained.
NINE_VIEWS: tuple[View, ...] = (
    View("canonical", IDENTITY_DIGITS, False),
    View("shift1", (2, 3, 4, 5, 6, 7, 8, 9, 1), False),
    View("shift4", (5, 6, 7, 8, 9, 1, 2, 3, 4), False),
    View("reversal", (9, 8, 7, 6, 5, 4, 3, 2, 1), False),
    View("anti_transpose", IDENTITY_DIGITS, True, REVERSED_LINES, REVERSED_LINES),
    View(
        "stack_rot_digit_shift2",
        (3, 4, 5, 6, 7, 8, 9, 1, 2),
        False,
        IDENTITY_LINES,
        (6, 7, 8, 0, 1, 2, 3, 4, 5),
    ),
    View(
        "full_mix",
        (7, 3, 9, 1, 8, 2, 4, 6, 5),
        True,
        (4, 5, 3, 8, 6, 7, 1, 2, 0),
        (5, 3, 4, 0, 1, 2, 7, 8, 6),
    ),
    View(
        "dg_shift1_rot180",
        (2, 3, 4, 5, 6, 7, 8, 9, 1),
        False,
        REVERSED_LINES,
        REVERSED_LINES,
    ),
    View("dg_shift3_pure", (4, 5, 6, 7, 8, 9, 1, 2, 3), False),
)
"""The frozen nine-view agreement-lock / sieve ladder, in tie-break order."""

# The seed-ensemble committee re-presents its anchor generator under six
# recorded views after the three plain seed members; these are members 3-8 of
# that frozen table, in member (tie-break) order.
SEED_ENSEMBLE_VIEWS: tuple[View, ...] = (
    View("shift1", (2, 3, 4, 5, 6, 7, 8, 9, 1), False),
    View("shift4", (5, 6, 7, 8, 9, 1, 2, 3, 4), False),
    View("reversal", (9, 8, 7, 6, 5, 4, 3, 2, 1), False),
    View("anti_transpose", IDENTITY_DIGITS, True, REVERSED_LINES, REVERSED_LINES),
    View(
        "dg_shift1_rot180",
        (2, 3, 4, 5, 6, 7, 8, 9, 1),
        False,
        REVERSED_LINES,
        REVERSED_LINES,
    ),
    View("dg_shift3_pure", (4, 5, 6, 7, 8, 9, 1, 2, 3), False),
)
"""The six view-members of the frozen seed-ensemble committee table."""


def apply_symmetry(
    grid: Tensor,
    *,
    digit_permutation: tuple[int, ...],
    transpose_grid: bool,
    row_permutation: tuple[int, ...] = IDENTITY_LINES,
    col_permutation: tuple[int, ...] = IDENTITY_LINES,
) -> Tensor:
    """Apply a recorded exact Sudoku symmetry to token grids.

    Position components apply as transpose first, then row/column
    permutations; digit relabeling is elementwise and commutes with all
    position components.

    Args:
      grid: ``[..., 81]`` token grid (0 pad, 1 blank, 2-10 digits 1-9); any
        dtype holding integral values, negative ignore ids pass through.
      digit_permutation: Relabeled digit for each of 1..9.
      transpose_grid: Whether to transpose the 9x9 positions.
      row_permutation: Source row per output row; must map bands onto bands.
      col_permutation: Source column per output column; must map stacks onto
        stacks.

    Returns:
      transformed: Same shape and dtype, transformed tokens.

    """
    lut = _token_lut(digit_permutation, device=grid.device)
    tokens = grid.long()
    mapped = torch.where(
        (tokens >= 0) & (tokens < lut.shape[0]),
        lut[tokens.clamp(min=0, max=lut.shape[0] - 1)],
        tokens,
    )
    if _moves_positions(row_permutation, col_permutation, transpose_grid):
        index = _position_index(
            row_permutation,
            col_permutation,
            transpose_grid=transpose_grid,
            device=grid.device,
        )
        mapped = mapped[..., index]
    return mapped.to(grid.dtype)


def invert_symmetry(
    grid: Tensor,
    *,
    digit_permutation: tuple[int, ...],
    transpose_grid: bool,
    row_permutation: tuple[int, ...] = IDENTITY_LINES,
    col_permutation: tuple[int, ...] = IDENTITY_LINES,
) -> Tensor:
    """Invert :func:`apply_symmetry` exactly.

    Args:
        grid: Transformed token grid.
        digit_permutation: Digit mapping used for the forward transform.
        transpose_grid: Whether the forward transform transposed positions.
        row_permutation: Source row per output row in the forward transform.
        col_permutation: Source column per output column in the forward transform.

    Returns:
        original: Original-space token grid.

    """
    inverse = [0] * 9
    for source, target in enumerate(digit_permutation):
        inverse[target - 1] = source + 1
    lut = _token_lut(tuple(inverse), device=grid.device)
    positioned = grid.long()
    if _moves_positions(row_permutation, col_permutation, transpose_grid):
        index = _position_index(
            row_permutation,
            col_permutation,
            transpose_grid=transpose_grid,
            device=grid.device,
        )
        positioned = positioned[..., torch.argsort(index)]
    restored = torch.where(
        (positioned >= 0) & (positioned < lut.shape[0]),
        lut[positioned.clamp(min=0, max=lut.shape[0] - 1)],
        positioned,
    )
    return restored.to(grid.dtype)


def validate_grid_permutation(name: str, permutation: tuple[int, ...]) -> None:
    """Require a 0..8 permutation whose 3-blocks map onto 3-blocks.

    Args:
      name: Field name used in the error message.
      permutation: Source line per output line.

    Raises:
      ValueError: If the permutation is not band-structure preserving.

    """
    if tuple(sorted(permutation)) != tuple(range(9)):
        raise ValueError(f"{name} must be a permutation of 0..8, got {permutation}.")
    for block in range(3):
        sources = {permutation[block * 3 + offset] // 3 for offset in range(3)}
        if len(sources) != 1:
            raise ValueError(
                f"{name} must map each band/stack onto one band/stack, got "
                f"{permutation}.",
            )


def _moves_positions(
    row_permutation: tuple[int, ...],
    col_permutation: tuple[int, ...],
    transpose_grid: bool,
) -> bool:
    """Return whether any position component is non-identity."""
    identity = tuple(range(9))
    return (
        transpose_grid
        or tuple(row_permutation) != identity
        or tuple(col_permutation) != identity
    )


def _token_lut(digit_permutation: tuple[int, ...], device: torch.device) -> Tensor:
    """Token lookup: pad 0 and blank 1 fixed; token 1+d relabels digit d."""
    table = [0, 1] + [digit_permutation[digit] + 1 for digit in range(9)]
    return torch.tensor(table, device=device, dtype=torch.long)


def _position_index(
    row_permutation: tuple[int, ...],
    col_permutation: tuple[int, ...],
    *,
    transpose_grid: bool,
    device: torch.device,
) -> Tensor:
    """Flat source index per output cell for the position components."""
    base = torch.arange(81, device=device).reshape(9, 9)
    if transpose_grid:
        base = base.T
    rows = torch.tensor(row_permutation, device=device, dtype=torch.long)
    cols = torch.tensor(col_permutation, device=device, dtype=torch.long)
    return base[rows][:, cols].reshape(-1)


# ---------------------------------------------------------------------------
# Hypothesis-pinning search (HPS).
# ---------------------------------------------------------------------------


type SearchRollout = Callable[[Tensor, Tensor], tuple[Tensor, Tensor]]
"""Candidate rollout ``(boards, rows) -> (logits, scores)`` on gathered rows;
``rows`` are original batch indices so the binding can gather per-row kwargs."""

type GridAcceptor = Callable[[Tensor, Tensor], Tensor]
"""Grid-predicate acceptance ``(preds, media_rows) -> [N] bool mask``."""


@dataclass(slots=True, frozen=True, kw_only=True)
class SearchResult:
    """Per-puzzle outputs of one :meth:`HpsSearch.run` batch (full batch B).

    Attributes:
      accepted: ``[B]`` bool; the root or a search node was accepted.
      root_predictions: ``[B, G]`` deterministic root rollout argmax grids.
      final_predictions: ``[B, G]`` protocol grids (accepted grid, else the
        root grid).
      scores: ``[B]`` learned-q of the emitted grid (root score where the
        search did not accept).
      root_scores: ``[B]`` root acceptance scores (min-over-checkpoints q).
      nodes: ``[B]`` candidate rollouts spent (0 without search).
      depth: ``[B]`` pins committed at acceptance; 0 for root acceptance,
        -1 where nothing was accepted.
      scored: ``[B]`` real (non-pad) rows; consumers slice by this mask or
        by the batch's ``valid_count``.
      visited_predictions: Per search level, ``[N, G]`` decoded candidate
        grids (the label-free trace behind :func:`solution_visited_flags`).
      visited_puzzles: Per search level, ``[N]`` owning batch row per
        candidate.

    """

    accepted: Tensor
    root_predictions: Tensor
    final_predictions: Tensor
    scores: Tensor
    root_scores: Tensor
    nodes: Tensor
    depth: Tensor
    scored: Tensor
    visited_predictions: list[Tensor]
    visited_puzzles: list[Tensor]


@dataclass(slots=True, frozen=True, kw_only=True)
class LearnedPinSearchResult:
    """Results and label-free candidate trace from learned-q pin search."""

    accepted: Tensor
    grids: Tensor
    scores: Tensor
    nodes: Tensor
    depth: Tensor
    visited_predictions: list[Tensor]
    visited_puzzles: list[Tensor]


@dataclass(slots=True, frozen=True, kw_only=True)
class LearnedCheckpointRollout:
    """Final logits plus learned-q and decoded grids at fixed ACT steps."""

    logits: Tensor
    q_scores: Tensor
    predictions: Tensor


@dataclass(slots=True, frozen=True, kw_only=True)
class DetPassResult:
    """Per-puzzle outputs of the conditional-depth deterministic pass.

    Attributes:
      grids: ``[N, G]`` decoded grids at each row's release step (CPU).
      q_halt: ``[N]`` q-halt logit at the release step (CPU).
      media: ``[N, G]`` the input boards, aligned by arrival order (CPU).
      label: ``[N, G]`` solution tokens for post-hoc scoring only (CPU).

    """

    grids: Tensor
    q_halt: Tensor
    media: Tensor
    label: Tensor


@dataclass(slots=True, frozen=True, kw_only=True)
class MemberDump:
    """Decoded member dump (see the module docstring's packed row layout).

    Attributes:
      rows: ``[N, 7 + 2G]`` float32 packed rows, verbatim from disk.
      score: ``[N]`` float32 emitted-grid learned-q (col 0).
      accepted: ``[N]`` bool acceptance flags (col 1 > 0.5).
      nodes: ``[N]`` int64 candidate rollouts spent (col 2).
      depth: ``[N]`` int64 pins at acceptance (col 3).
      candidate_count: ``[N]`` int64 scored root + nodes (col 4).
      solution_visited: ``[N]`` bool label-joined trace flag (col 5 > 0.5).
      root_predictions: ``[N, G]`` uint8 root grids (cols 7..7+G).
      final_predictions: ``[N, G]`` uint8 protocol grids (last G cols).
      media: ``[N, G]`` uint8 input boards.
      label: ``[N, G]`` uint8 solution tokens, or None when not dumped.

    """

    rows: Tensor
    score: Tensor
    accepted: Tensor
    nodes: Tensor
    depth: Tensor
    candidate_count: Tensor
    solution_visited: Tensor
    root_predictions: Tensor
    final_predictions: Tensor
    media: Tensor
    label: Tensor | None


class HpsSearch:
    """The hps(C, K, A, budget) engine with learned-q acceptance.

    ``Config()`` defaults are the recipe policy: ``hps(5, 3, 2, 512)`` with
    q >= 7.875 required at ACT steps 24, 28 AND 32 and full-depth candidate
    rollouts. See the module docstring for the algorithm and the seam
    contract; :func:`escalated_search_config` is the ``hps(7, 4, 3, 8400)``
    escalation preset.
    """

    class Config(Fig["HpsSearch"]):
        max_act_steps: int = 32
        """Full ACT rollout depth for the root and every candidate."""

        search_depth: int = 3
        """Max pins per hypothesis path (``K``)."""

        search_candidates: int = 5
        """Confidence-ordered digit candidates per selected cell (``C``)."""

        search_cell_attempts: int = 2
        """Root entropy-cell restarts after a tree is exhausted (``A``)."""

        search_budget: int = 512
        """Max candidate rollouts per puzzle; exhaustion emits the root grid
        (the protocol has no abstain)."""

        search_max_rows: int = 2_048
        """Max candidate rows per model call (memory guard for the C^K-wide
        level expansions; sibling hypotheses batch together up to this)."""

        acceptance_threshold: float = 7.875
        """Inclusive learned-q acceptance threshold (``>=``, so a score of
        exactly 7.875 is accepted)."""

        root_acceptance_threshold: float | None = None
        """Optional root-only threshold; None reuses acceptance_threshold."""

        continue_threshold: float = 0.0
        """Inclusive q@8 threshold for continuing remaining ACT compute
        (used only when ``early_exit_at_q8`` is enabled)."""

        early_exit_at_q8: bool = False
        """Use q@8 to skip remaining ACT compute for low-scoring candidates.
        The recipe keeps this OFF: checkpoint persistence needs full-depth
        candidate rollouts. Must be False when acceptance_checkpoints is
        set. (The grid-predicate path always early-exits regardless -- the
        fast-engine pruning leg.)"""

        acceptance_checkpoints: tuple[int, ...] = (24, 28, 32)
        """ACT checkpoints whose MINIMUM q defines persistent acceptance;
        () scores the final step's q only."""

        require_prediction_stability: bool = False
        """Reject unless the decoded grid is identical at every acceptance
        checkpoint."""

        dtype_autocast: torch.dtype | None = torch.bfloat16
        """Autocast compute dtype for the rollouts (mirrors the trainer);
        None disables autocast."""

    def __init__(self, config: Config) -> None:
        for name in (
            "max_act_steps",
            "search_depth",
            "search_candidates",
            "search_cell_attempts",
            "search_budget",
            "search_max_rows",
        ):
            if getattr(config, name) < 1:
                raise ValueError(f"{name} must be >= 1, got {getattr(config, name)}.")
        if config.search_budget < config.search_candidates:
            raise ValueError(
                "search_budget must cover one candidate breadth; got "
                f"{config.search_budget} < {config.search_candidates}.",
            )
        for name, value in (
            ("acceptance_threshold", config.acceptance_threshold),
            ("continue_threshold", config.continue_threshold),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value}.")
        if config.root_acceptance_threshold is not None and not math.isfinite(
            config.root_acceptance_threshold,
        ):
            raise ValueError(
                "root_acceptance_threshold must be finite when set, got "
                f"{config.root_acceptance_threshold}.",
            )
        checkpoints = config.acceptance_checkpoints
        if checkpoints and tuple(sorted(set(checkpoints))) != checkpoints:
            raise ValueError(
                "acceptance_checkpoints must be strictly increasing; got "
                f"{checkpoints}.",
            )
        if checkpoints and (
            checkpoints[0] < 1 or checkpoints[-1] > config.max_act_steps
        ):
            raise ValueError(
                "acceptance_checkpoints must lie within max_act_steps; got "
                f"{checkpoints} for {config.max_act_steps}.",
            )
        if checkpoints and config.early_exit_at_q8:
            raise ValueError(
                "early_exit_at_q8 must be false for checkpoint persistence.",
            )
        if config.require_prediction_stability and not checkpoints:
            raise ValueError(
                "require_prediction_stability needs acceptance_checkpoints.",
            )
        self.config = config

    def run(
        self,
        model: TRM,
        batch: Mapping[str, object],
        *,
        accept_fn: GridAcceptor | None = None,
    ) -> SearchResult:
        """Search one batch: deterministic root, then pin search on the rest.

        Args:
          model: Frozen TRM carrying the EVAL weights (the caller applies the
            EMA swap); flipped to eval mode here. Rollouts run under
            ``inference_mode`` + autocast (``config.dtype_autocast``).
          batch: Dataset batch contract -- ``media`` ``[B, G]`` input tokens,
            ``valid_count`` int, ``puzzle_identifiers`` ``[B]`` (required
            when the model carries a puzzle embedding). ``label`` is never
            read: the whole search path is label-free.
          accept_fn: None (default) runs the learned-q policy. A predicate
            switches BOTH the root and every node to grid acceptance
            (verifier committee / :func:`accepted_grids`) with the q-halt@8
            early-exit rollout -- the fast predicate engine. ``scores`` then
            carry the root q-halt only.

        Returns:
          result: The per-puzzle :class:`SearchResult` (full batch size; pad
            rows carry ``scored=False``).

        """
        media = batch["media"]
        assert isinstance(media, Tensor)
        valid_count = batch.get("valid_count", media.shape[0])
        assert isinstance(valid_count, int)
        step_kwargs: dict[str, Tensor] = {}
        if model.puzzle_emb is not None:
            puzzle_identifiers = batch["puzzle_identifiers"]
            assert isinstance(puzzle_identifiers, Tensor)
            step_kwargs["puzzle_identifiers"] = puzzle_identifiers
        if valid_count == 0:
            return _empty_result(media)

        model.eval()
        with torch.inference_mode(), self._autocast(model):
            if accept_fn is None:
                return self._run_learned(model, media, valid_count, step_kwargs)
            return self._run_predicate(
                model,
                media,
                valid_count,
                step_kwargs,
                accept_fn,
            )

    def _run_learned(
        self,
        model: TRM,
        media: Tensor,
        valid_count: int,
        step_kwargs: dict[str, Tensor],
    ) -> SearchResult:
        """Learned-q path: persistent root scores, then learned pin search."""
        cfg = self.config
        batch_size = media.shape[0]
        device = media.device
        if cfg.acceptance_checkpoints:
            root_checkpoint = learned_checkpoint_rollout_rows(
                model,
                step_kwargs,
                cfg.max_act_steps,
                media,
                torch.arange(batch_size, device=device),
                checkpoints=cfg.acceptance_checkpoints,
            )
            root_logits = root_checkpoint.logits
            root_scores = learned_persistence_scores(
                root_checkpoint,
                require_prediction_stability=cfg.require_prediction_stability,
            )
        else:
            root, _ = family_rollout(
                model,
                media,
                step_kwargs=step_kwargs,
                max_steps=cfg.max_act_steps,
            )
            root_logits = root["logits"]
            root_scores = root["q_halt"].float()
        root_predictions = root_logits.argmax(dim=-1)
        scored = torch.arange(batch_size, device=device) < valid_count
        root_threshold = (
            cfg.acceptance_threshold
            if cfg.root_acceptance_threshold is None
            else cfg.root_acceptance_threshold
        )
        accepted = (root_scores >= root_threshold) & scored
        final_predictions = root_predictions.clone()
        final_scores = root_scores.clone()
        nodes = torch.zeros(batch_size, dtype=torch.int64, device=device)
        depth = torch.full((batch_size,), -1, dtype=torch.int64, device=device)
        depth[accepted] = 0
        active = ~accepted & scored
        visited_predictions: list[Tensor] = []
        visited_puzzles: list[Tensor] = []
        if bool(active.any()):
            if cfg.acceptance_checkpoints:
                candidate_rollout: SearchRollout = partial(
                    learned_persistent_rollout_rows,
                    model,
                    step_kwargs,
                    cfg.max_act_steps,
                    checkpoints=cfg.acceptance_checkpoints,
                    require_prediction_stability=cfg.require_prediction_stability,
                )
            else:
                candidate_rollout = partial(
                    segmented_rollout_rows,
                    model,
                    step_kwargs,
                    cfg.max_act_steps,
                    continue_threshold=cfg.continue_threshold,
                    early_exit_at_q8=cfg.early_exit_at_q8,
                )
            search = run_learned_pin_search_fast(
                candidate_rollout,
                media=media,
                base_logits=root_logits,
                active=active,
                acceptance_threshold=cfg.acceptance_threshold,
                depth=cfg.search_depth,
                candidates=cfg.search_candidates,
                cell_attempts=cfg.search_cell_attempts,
                budget=cfg.search_budget,
                max_rows=cfg.search_max_rows,
            )
            final_predictions = torch.where(
                search.accepted.unsqueeze(-1),
                search.grids.to(final_predictions.dtype),
                final_predictions,
            )
            final_scores = torch.where(search.accepted, search.scores, final_scores)
            accepted = accepted | search.accepted
            nodes = search.nodes
            depth = torch.where(search.accepted, search.depth, depth)
            visited_predictions = search.visited_predictions
            visited_puzzles = search.visited_puzzles
        return SearchResult(
            accepted=accepted,
            root_predictions=root_predictions,
            final_predictions=final_predictions,
            scores=final_scores,
            root_scores=root_scores,
            nodes=nodes,
            depth=depth,
            scored=scored,
            visited_predictions=visited_predictions,
            visited_puzzles=visited_puzzles,
        )

    def _run_predicate(
        self,
        model: TRM,
        media: Tensor,
        valid_count: int,
        step_kwargs: dict[str, Tensor],
        accept_fn: GridAcceptor,
    ) -> SearchResult:
        """Grid-predicate path: fast engine with the q-halt@8 early exit."""
        cfg = self.config
        batch_size = media.shape[0]
        device = media.device
        root, _ = family_rollout(
            model,
            media,
            step_kwargs=step_kwargs,
            max_steps=cfg.max_act_steps,
        )
        root_logits = root["logits"]
        root_scores = root["q_halt"].float()
        root_predictions = root_logits.argmax(dim=-1)
        scored = torch.arange(batch_size, device=device) < valid_count
        accepted = accept_fn(root_predictions, media) & scored
        final_predictions = root_predictions.clone()
        nodes = torch.zeros(batch_size, dtype=torch.int64, device=device)
        depth = torch.full((batch_size,), -1, dtype=torch.int64, device=device)
        depth[accepted] = 0
        active = ~accepted & scored
        if bool(active.any()):
            # The predicate engine's q-halt@8 early exit is part of its
            # definition (false branches cost ~8 steps): hardwired here,
            # independent of the learned-path early_exit_at_q8 knob.
            found, grids, nodes, depth_search = run_pin_search_fast(
                partial(
                    segmented_rollout_rows,
                    model,
                    step_kwargs,
                    cfg.max_act_steps,
                    continue_threshold=0.0,
                    early_exit_at_q8=True,
                ),
                media=media,
                base_logits=root_logits,
                active=active,
                groups=sudoku_group_indices(SudokuSpec()).to(device),
                depth=cfg.search_depth,
                candidates=cfg.search_candidates,
                cell_attempts=cfg.search_cell_attempts,
                budget=cfg.search_budget,
                max_rows=cfg.search_max_rows,
                accept_fn=accept_fn,
            )
            final_predictions = torch.where(
                found.unsqueeze(-1),
                grids.to(final_predictions.dtype),
                final_predictions,
            )
            accepted = accepted | found
            depth = torch.where(found, depth_search, depth)
        return SearchResult(
            accepted=accepted,
            root_predictions=root_predictions,
            final_predictions=final_predictions,
            scores=root_scores.clone(),
            root_scores=root_scores,
            nodes=nodes,
            depth=depth,
            scored=scored,
            visited_predictions=[],
            visited_puzzles=[],
        )

    def _autocast(self, model: TRM) -> torch.amp.autocast:
        return torch.amp.autocast(
            device_type=model.device.type,
            dtype=self.config.dtype_autocast,
            enabled=self.config.dtype_autocast is not None,
            cache_enabled=False,
        )


def run_search(
    model: TRM,
    batch: Mapping[str, object],
    config: HpsSearch.Config,
    *,
    accept_fn: GridAcceptor | None = None,
) -> SearchResult:
    """Run the hps(...) protocol on one batch (see :meth:`HpsSearch.run`).

    Args:
      model: Frozen TRM carrying the eval weights (EMA swap caller-owned).
      batch: Dataset batch dict (``media`` / ``valid_count`` /
        ``puzzle_identifiers``); labels never read.
      config: Search policy; defaults are the recipe hps(5, 3, 2, 512) with
        q >= 7.875 at ACT (24, 28, 32).
      accept_fn: Optional grid predicate switching to the predicate engine.

    Returns:
      result: The per-puzzle :class:`SearchResult`.

    """
    return config.make().run(model, batch, accept_fn=accept_fn)


def escalated_search_config() -> HpsSearch.Config:
    """Configure the escalation preset ``hps(7, 4, 3, 8400)``.

    Raising C to 7 (not K) converted every residual coverage-exhaustion miss
    of the ratified default; the acceptance policy is unchanged.

    Returns:
      config: A fresh recipe config with C=7, K=4, A=3, budget=8,400.

    """
    config = HpsSearch.Config()
    config.search_candidates = 7
    config.search_depth = 4
    config.search_cell_attempts = 3
    config.search_budget = 8_400
    return config


def det_pass(trainer: Trainer, *, halt_threshold: float = 2.0) -> DetPassResult:
    """Conditional-depth deterministic pass over the trainer's eval split.

    The det gate / sieve round-0 pass: every row rolls until its q-halt logit
    clears ``halt_threshold`` (default +2, the registered release bar) or the
    ACT depth cap. Wires to the trainer's own conditional-depth eval seam
    (:meth:`Trainer.eval_loss` -- EMA swap, compile guard, and the per-row
    release rollout all live there); this function only flips the release
    config on for the pass and collects the per-puzzle outputs. Batch pooling
    (the internal pass used one pooled 5,000-row eval batch) is the dataset
    config's ``eval_batch_size``.

    Args:
      trainer: Trainer whose (possibly resumed) weights are scored; its
        ``eval_halt_exit`` / ``eval_halt_threshold`` are overridden for the
        duration of the pass and restored after.
      halt_threshold: Q-halt logit release threshold.

    Returns:
      result: :class:`DetPassResult` over all valid eval rows, in arrival
        order (CPU tensors; labels are carried for post-hoc scoring only).

    """
    config = trainer.config
    saved = (config.eval_halt_exit, config.eval_halt_threshold)
    config.eval_halt_exit = True
    config.eval_halt_threshold = halt_threshold
    grid_parts: list[Tensor] = []
    q_parts: list[Tensor] = []
    media_parts: list[Tensor] = []
    label_parts: list[Tensor] = []
    try:
        for raw_batch in trainer.dataset.eval_dataloader():
            batch = trainer.preprocess_batch(raw_batch)
            media = batch["media"]
            valid_count = batch["valid_count"]
            if valid_count == 0:
                continue
            packed = trainer.eval_loss(**batch)["model"]
            q_parts.append(packed[:valid_count, 0].float().cpu())
            grid_parts.append(packed[:valid_count, 1:].to(torch.int64).cpu())
            media_parts.append(media[:valid_count].to(torch.int64).cpu())
            label_parts.append(batch["label"][:valid_count].to(torch.int64).cpu())
    finally:
        config.eval_halt_exit, config.eval_halt_threshold = saved
    if not grid_parts:
        raise ValueError("det_pass found no valid eval rows.")
    return DetPassResult(
        grids=torch.cat(grid_parts),
        q_halt=torch.cat(q_parts),
        media=torch.cat(media_parts),
        label=torch.cat(label_parts),
    )


def family_rollout(
    model: TRM,
    boards: Tensor,
    *,
    step_kwargs: dict[str, Tensor],
    max_steps: int,
) -> tuple[dict[str, Tensor], Tensor]:
    """Full-ACT eval rollout on ``boards`` (the deterministic root unit).

    Fresh ``init_z``, decoded-grid feedback threaded from ``boards`` with
    given cells clamped -- exactly the trainer's fixed-depth eval recurrence,
    so a call on the original media is bit-identical to the plain eval.

    Args:
      model: The TRM (eval mode, weights already swapped).
      boards: ``[B, G]`` input token boards (givens = tokens 2-10).
      step_kwargs: Extra ``act_step`` kwargs (``puzzle_identifiers`` when the
        model carries a puzzle embedding), pre-gathered to ``boards``' rows.
      max_steps: Eval ACT rollout depth.

    Returns:
      out: The final ``act_step`` output dict.
      q_halt8: ``[B]`` q-halt logits at step ``min(8, max_steps)`` (the
        UNSAT-probe read point).

    """
    z_slow, z_fast = model.init_z(boards.shape[0])
    given = (boards >= 2) & (boards <= 10)
    feedback = boards
    q_halt8: Tensor | None = None
    out: dict[str, Tensor] = {}
    for t in range(1, max_steps + 1):
        out = model.act_step(
            boards,
            z_slow,
            z_fast,
            feedback_ids=feedback,
            **step_kwargs,
        )
        z_slow = out["z_slow"]
        z_fast = out["z_fast"]
        feedback = torch.where(given, boards, out["logits"].argmax(dim=-1))
        if t == min(8, max_steps):
            q_halt8 = out["q_halt"]
    if q_halt8 is None:
        raise ValueError("Expected q_halt8 is not None.")
    return out, q_halt8


def segmented_rollout_rows(
    model: TRM,
    step_kwargs: dict[str, Tensor],
    max_steps: int,
    boards: Tensor,
    rows: Tensor,
    *,
    continue_threshold: float,
    early_exit_at_q8: bool,
) -> tuple[Tensor, Tensor]:
    """Roll candidates to q@8, then continue selected rows to final q.

    The fast engine's pruning leg: with ``early_exit_at_q8`` only rows whose
    q@8 clears ``continue_threshold`` (inclusive) get the remaining ACT
    compute -- the trained q-halt head at step 8 is a near-zero-FP UNSAT
    probe, so false pins cost ~8 steps instead of a full rollout. Early-
    exited rows return their step-8 logits: their grids face acceptance
    as-is and their children are selected from step-8 logits (the accepted
    conversion-only risk; validity/learned-q still gates, never soundness).

    Args:
      model: The TRM (eval mode, weights already swapped).
      step_kwargs: Full-root model kwargs, gathered here by ``rows``.
      max_steps: Maximum ACT steps.
      boards: ``[N, G]`` candidate boards.
      rows: ``[N]`` original root-batch row per candidate board.
      continue_threshold: Inclusive q@8 threshold for remaining ACT compute.
      early_exit_at_q8: Continue every row when False.

    Returns:
      logits: ``[N, G, V]`` final logits for continued rows and q@8-step
        logits for early-exited rows.
      scores: ``[N]`` final q for continued rows and q@8 for early rows.

    """
    if max_steps < 1:
        raise ValueError(f"max_steps must be >= 1, got {max_steps}.")
    puzzle_identifiers = step_kwargs.get("puzzle_identifiers")
    if puzzle_identifiers is not None:
        puzzle_identifiers = puzzle_identifiers[rows]
    split = min(8, max_steps)
    z_slow, z_fast = model.init_z(boards.shape[0])
    given = (boards >= 2) & (boards <= 10)
    feedback = boards
    out = model.act_step(
        boards,
        z_slow,
        z_fast,
        puzzle_identifiers=puzzle_identifiers,
        feedback_ids=feedback,
    )
    z_slow = out["z_slow"]
    z_fast = out["z_fast"]
    feedback = torch.where(given, boards, out["logits"].argmax(dim=-1))
    for _ in range(1, split):
        out = model.act_step(
            boards,
            z_slow,
            z_fast,
            puzzle_identifiers=puzzle_identifiers,
            feedback_ids=feedback,
        )
        z_slow = out["z_slow"]
        z_fast = out["z_fast"]
        feedback = torch.where(given, boards, out["logits"].argmax(dim=-1))

    scores = out["q_halt"].float().clone()
    logits = out["logits"].clone()
    if early_exit_at_q8:
        continued = (scores >= continue_threshold).nonzero(as_tuple=True)[0]
    else:
        continued = torch.arange(boards.shape[0], device=boards.device)
    if split == max_steps:
        return logits, scores
    if not continued.shape[0]:
        return logits, scores

    boards_live = boards[continued]
    given_live = given[continued]
    z_slow_live = z_slow[continued]
    z_fast_live = z_fast[continued]
    feedback_live = feedback[continued]
    puzzle_ids_live = (
        puzzle_identifiers[continued] if puzzle_identifiers is not None else None
    )
    for _ in range(split, max_steps):
        out = model.act_step(
            boards_live,
            z_slow_live,
            z_fast_live,
            puzzle_identifiers=puzzle_ids_live,
            feedback_ids=feedback_live,
        )
        z_slow_live = out["z_slow"]
        z_fast_live = out["z_fast"]
        feedback_live = torch.where(
            given_live,
            boards_live,
            out["logits"].argmax(dim=-1),
        )
    logits[continued] = out["logits"]
    scores[continued] = out["q_halt"].float()
    return logits, scores


def learned_checkpoint_rollout_rows(
    model: TRM,
    step_kwargs: dict[str, Tensor],
    max_steps: int,
    boards: Tensor,
    rows: Tensor,
    *,
    checkpoints: tuple[int, ...],
) -> LearnedCheckpointRollout:
    """Run full ACT and retain q and decoded grids at fixed checkpoints.

    Args:
      model: The TRM (eval mode, weights already swapped).
      step_kwargs: Full-root model kwargs, gathered here by ``rows``.
      max_steps: Number of ACT steps to run for every row.
      boards: ``[N, G]`` candidate boards.
      rows: ``[N]`` original root-batch row per candidate board.
      checkpoints: Strictly increasing one-indexed ACT steps to retain.

    Returns:
      rollout: Final logits and checkpoint-aligned q scores and grids.

    Raises:
      ValueError: Checkpoints are empty, unordered, or outside ``max_steps``.

    """
    if not checkpoints:
        raise ValueError("checkpoints must contain at least one ACT step.")
    if tuple(sorted(set(checkpoints))) != checkpoints:
        raise ValueError(
            f"checkpoints must be strictly increasing, got {checkpoints}.",
        )
    if checkpoints[0] < 1 or checkpoints[-1] > max_steps:
        raise ValueError(
            f"checkpoints {checkpoints} must lie within 1..{max_steps}.",
        )
    puzzle_identifiers = step_kwargs.get("puzzle_identifiers")
    if puzzle_identifiers is not None:
        puzzle_identifiers = puzzle_identifiers[rows]
    z_slow, z_fast = model.init_z(boards.shape[0])
    given = (boards >= 2) & (boards <= 10)
    feedback = boards
    q_scores: list[Tensor] = []
    predictions: list[Tensor] = []
    out: dict[str, Tensor] = {}
    checkpoint_index = 0
    for step in range(1, max_steps + 1):
        out = model.act_step(
            boards,
            z_slow,
            z_fast,
            puzzle_identifiers=puzzle_identifiers,
            feedback_ids=feedback,
        )
        z_slow = out["z_slow"]
        z_fast = out["z_fast"]
        decoded = out["logits"].argmax(dim=-1)
        feedback = torch.where(given, boards, decoded)
        if step == checkpoints[checkpoint_index]:
            q_scores.append(out["q_halt"].float().clone())
            predictions.append(decoded.clone())
            checkpoint_index += 1
            if checkpoint_index == len(checkpoints):
                checkpoint_index = -1
    return LearnedCheckpointRollout(
        logits=out["logits"],
        q_scores=torch.stack(q_scores, dim=1),
        predictions=torch.stack(predictions, dim=1),
    )


def learned_persistence_scores(
    rollout: LearnedCheckpointRollout,
    *,
    require_prediction_stability: bool,
) -> Tensor:
    """Return weakest checkpoint q, rejecting optional decoded-grid drift.

    The recipe acceptance reduction: a grid's score is its MINIMUM q over the
    acceptance checkpoints, so acceptance requires the threshold at EVERY
    checkpoint.

    Args:
      rollout: Checkpoint-aligned model outputs.
      require_prediction_stability: Reject (score ``-inf``) unless every
        checkpoint grid matches the final checkpoint's grid.

    Returns:
      scores: ``[N]`` per-row minimum q, or ``-inf`` for unstable rows.

    """
    if rollout.q_scores.ndim != 2 or rollout.predictions.ndim != 3:
        raise ValueError("checkpoint q and predictions must have ranks 2 and 3.")
    if rollout.q_scores.shape[:2] != rollout.predictions.shape[:2]:
        raise ValueError("checkpoint q and predictions must align by row and step.")
    scores = rollout.q_scores.min(dim=1).values
    if not require_prediction_stability:
        return scores
    stable = (rollout.predictions == rollout.predictions[:, -1:, :]).all(dim=(1, 2))
    return torch.where(stable, scores, torch.full_like(scores, float("-inf")))


def learned_persistent_rollout_rows(
    model: TRM,
    step_kwargs: dict[str, Tensor],
    max_steps: int,
    boards: Tensor,
    rows: Tensor,
    *,
    checkpoints: tuple[int, ...],
    require_prediction_stability: bool,
) -> tuple[Tensor, Tensor]:
    """Run full ACT and score persistence without changing final logits.

    Args:
      model: The TRM (eval mode, weights already swapped).
      step_kwargs: Full-root model kwargs, gathered by ``rows``.
      max_steps: Number of ACT steps to run for every row.
      boards: ``[N, G]`` candidate boards.
      rows: ``[N]`` original root-batch row per candidate board.
      checkpoints: Strictly increasing one-indexed ACT steps to score.
      require_prediction_stability: Reject decoded-grid drift (``-inf``).

    Returns:
      logits: ``[N, G, V]`` final-step logits.
      scores: ``[N]`` minimum checkpoint q (the persistence score).

    """
    rollout = learned_checkpoint_rollout_rows(
        model,
        step_kwargs,
        max_steps,
        boards,
        rows,
        checkpoints=checkpoints,
    )
    return rollout.logits, learned_persistence_scores(
        rollout,
        require_prediction_stability=require_prediction_stability,
    )


def select_pin_candidates(
    logits: Tensor,
    boards: Tensor,
    *,
    n_cells: int,
    n_digits: int,
) -> tuple[Tensor, Tensor]:
    """Label-free pin candidates: max-entropy cells + confidence-ordered digits.

    Args:
      logits: ``[B, G, V]`` final-step rollout logits.
      boards: ``[B, G]`` the CURRENT boards (original givens + committed
        pins); given cells are excluded from cell selection.
      n_cells: Cells to return, by descending token-distribution entropy.
      n_digits: Digit candidates per cell, by descending logit.

    Returns:
      cells: ``[B, n_cells]`` selected cell indices.
      digits: ``[B, n_cells, n_digits]`` digit tokens (2-10), best first.

    """
    probs = logits.float().softmax(dim=-1)
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
    given = (boards >= 2) & (boards <= 10)
    score = torch.where(given, torch.full_like(entropy, -1.0), entropy)
    # Stable descending sort, not topk: on ties (e.g. uniform logits) it keeps
    # ascending index order deterministically; torch.topk's tie order is
    # unspecified and shifted between torch releases.
    cells = score.sort(dim=-1, descending=True, stable=True).indices[..., :n_cells]
    cell_logits = logits.float().gather(
        1,
        cells.unsqueeze(-1).expand(-1, -1, logits.shape[-1]),
    )
    digits = (
        cell_logits[..., 2:11]
        .sort(dim=-1, descending=True, stable=True)
        .indices[..., :n_digits]
        + 2
    )
    return cells, digits


def accepted_grids(preds: Tensor, media: Tensor, groups: Tensor) -> Tensor:
    """``[B]`` LABEL-FREE validity acceptance of decoded grids.

    A grid is accepted iff every cell is a digit token (2-10), it violates
    zero of the 27 uniqueness groups, AND it matches the ORIGINAL givens. By
    solution uniqueness an accepted grid IS the puzzle's solution: any
    complete valid grid extending the givens would otherwise be a second
    solution. The explicit completeness conjunct is load-bearing:
    violated-group counting clamps tokens above 10, so an out-of-range token
    would otherwise alias to digit 9.

    Args:
      preds: ``[B, G]`` decoded (argmax) grids.
      media: ``[B, G]`` the ORIGINAL puzzle boards (never pinned boards --
        pins are hypotheses and must not gate acceptance).
      groups: ``[27, 9]`` sudoku group indices
        (:func:`~priml.baselines.sudoku.trainer.sudoku_group_indices`).

    Returns:
      accepted: ``[B]`` acceptance mask.

    """
    complete = ((preds >= 2) & (preds <= 10)).all(dim=1)
    given = (media >= 2) & (media <= 10)
    consistent = ((preds == media) | ~given).all(dim=1)
    return complete & (_violated_group_counts(preds, groups) == 0) & consistent


def run_pin_search_fast(
    rollout: SearchRollout,
    *,
    media: Tensor,
    base_logits: Tensor,
    active: Tensor,
    groups: Tensor,
    depth: int,
    candidates: int,
    cell_attempts: int,
    budget: int,
    max_rows: int,
    accept_fn: GridAcceptor | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Candidate-parallel pin search with grid-predicate acceptance.

    The fast predicate engine: per attempt, the full candidate tree is
    expanded level by level -- every surviving node's ``candidates`` child
    boards roll out in one chunked batched pass. Cell choice per node depends
    only on that node's board and its own rollout logits. Acceptance is a
    sound grid predicate, so any accepting node emits THE unique solution;
    the only rollout-side risk is the binding's q-halt@8 early exit (lost
    conversion, never wrong acceptance).

    Args:
      rollout: ``(boards, rows) -> (logits, scores)`` on gathered rows;
        ``scores`` are ignored here (the q@8 prune lives inside the binding).
      media: ``[B, G]`` ORIGINAL puzzle boards.
      base_logits: ``[B, G, V]`` root rollout logits (level-1 cell source).
      active: ``[B]`` rows to search (root non-accepted, scored).
      groups: ``[27, 9]`` sudoku group indices.
      depth: Max pins per hypothesis path (``K``).
      candidates: Digit candidates per pin cell (``C``).
      cell_attempts: Level-1 cell restarts (``A``).
      budget: Max node rollouts per puzzle (a level is expanded only where
        ``nodes + C^level`` stays within it).
      max_rows: Rollout chunk width (memory guard).
      accept_fn: Node-acceptance predicate ``(preds, media_rows) -> [N]``
        bool mask; None uses :func:`accepted_grids` with ``groups``.

    Returns:
      found: ``[B]`` puzzles whose search accepted a grid.
      grids: ``[B, G]`` accepted grids on found rows (media elsewhere).
      nodes: ``[B]`` node rollouts spent.
      depth_used: ``[B]`` pins committed at acceptance; -1 where not found.

    """
    accept = accept_fn or partial(accepted_grids, groups=groups)
    b = media.shape[0]
    device = media.device
    cells_l1, digits_l1 = select_pin_candidates(
        base_logits,
        media,
        n_cells=cell_attempts,
        n_digits=candidates,
    )
    nodes = torch.zeros(b, dtype=torch.int64, device=device)
    depth_used = torch.full((b,), -1, dtype=torch.int64, device=device)
    found = torch.zeros(b, dtype=torch.bool, device=device)
    grids = media.clone()

    for attempt in range(cell_attempts):
        eligible = active & ~found & (nodes + candidates <= budget)
        puzzles = eligible.nonzero(as_tuple=True)[0]
        if not puzzles.shape[0]:
            break
        # Level-1 frontier: C candidate boards per puzzle at this attempt's
        # root entropy cell.
        frontier_puzzle = puzzles.repeat_interleave(candidates)
        boards = media[puzzles].clone().repeat_interleave(candidates, dim=0)
        cell0 = cells_l1[puzzles, attempt].repeat_interleave(candidates)
        digit0 = digits_l1[puzzles, attempt].reshape(-1)
        rows_local = torch.arange(boards.shape[0], device=device)
        boards[rows_local, cell0] = digit0.to(boards.dtype)

        for level in range(1, depth + 1):
            logits_list: list[Tensor] = []
            for lo in range(0, boards.shape[0], max_rows):
                chunk_logits, _ = rollout(
                    boards[lo : lo + max_rows],
                    frontier_puzzle[lo : lo + max_rows],
                )
                logits_list.append(chunk_logits)
            logits = torch.cat(logits_list)
            preds = logits.argmax(dim=-1)
            nodes.scatter_add_(0, frontier_puzzle, torch.ones_like(frontier_puzzle))
            acc = accept(preds, media[frontier_puzzle])
            win_rows = acc.nonzero(as_tuple=True)[0]
            if win_rows.shape[0]:
                # Any accepting node's grid IS the unique solution;
                # same-puzzle duplicates overwrite with identical content.
                win_puzzles = frontier_puzzle[win_rows]
                grids[win_puzzles] = preds[win_rows].to(grids.dtype)
                depth_used[win_puzzles] = level
                found[win_puzzles] = True
            if level < depth:
                # Expand survivors whose puzzle is unresolved and within budget.
                child_count = candidates ** (level + 1)
                expandable = ~found[frontier_puzzle] & (
                    nodes[frontier_puzzle] + child_count <= budget
                )
                keep = expandable.nonzero(as_tuple=True)[0]
                if not keep.shape[0]:
                    break
                parent_boards = boards[keep]
                cell_d, dig_d = select_pin_candidates(
                    logits[keep],
                    parent_boards,
                    n_cells=1,
                    n_digits=candidates,
                )
                boards = parent_boards.repeat_interleave(candidates, dim=0)
                frontier_puzzle = frontier_puzzle[keep].repeat_interleave(candidates)
                cell_child = cell_d[:, 0].repeat_interleave(candidates)
                digit_child = dig_d[:, 0].reshape(-1)
                rows_local = torch.arange(boards.shape[0], device=device)
                boards[rows_local, cell_child] = digit_child.to(boards.dtype)
    return found, grids, nodes, depth_used


def run_learned_pin_search_fast(
    rollout: SearchRollout,
    *,
    media: Tensor,
    base_logits: Tensor,
    active: Tensor,
    acceptance_threshold: float,
    depth: int,
    candidates: int,
    cell_attempts: int,
    budget: int,
    max_rows: int,
) -> LearnedPinSearchResult:
    """Candidate-parallel pin search with learned-q acceptance only.

    Args:
      rollout: Candidate-board rollout returning decoded logits and the
        learned acceptance score per row (:func:`segmented_rollout_rows` or
        :func:`learned_persistent_rollout_rows` bound to the model).
      media: ``[B, G]`` ORIGINAL puzzle boards.
      base_logits: ``[B, G, V]`` root rollout logits (level-1 cell source).
      active: ``[B]`` rows that root learned acceptance rejected.
      acceptance_threshold: Inclusive learned-score acceptance threshold.
      depth: Max pins per hypothesis path (``K``).
      candidates: Digit candidates per pin cell (``C``).
      cell_attempts: Level-1 cell restarts (``A``).
      budget: Max candidate rollouts per puzzle.
      max_rows: Rollout chunk width (memory guard).

    Returns:
      result: Earliest-positive grids, costs, and a label-free candidate
        trace.

    """
    batch_size = media.shape[0]
    device = media.device
    cells_root, digits_root = select_pin_candidates(
        base_logits,
        media,
        n_cells=cell_attempts,
        n_digits=candidates,
    )
    nodes = torch.zeros(batch_size, dtype=torch.int64, device=device)
    depth_used = torch.full((batch_size,), -1, dtype=torch.int64, device=device)
    accepted = torch.zeros(batch_size, dtype=torch.bool, device=device)
    grids = media.clone()
    scores = torch.full(
        (batch_size,),
        float("-inf"),
        dtype=torch.float32,
        device=device,
    )
    visited_predictions: list[Tensor] = []
    visited_puzzles: list[Tensor] = []

    for attempt in range(cell_attempts):
        eligible = active & ~accepted & (nodes + candidates <= budget)
        puzzles = eligible.nonzero(as_tuple=True)[0]
        if not puzzles.shape[0]:
            break
        frontier_puzzle = puzzles.repeat_interleave(candidates)
        boards = media[puzzles].clone().repeat_interleave(candidates, dim=0)
        root_cell = cells_root[puzzles, attempt].repeat_interleave(candidates)
        root_digit = digits_root[puzzles, attempt].reshape(-1)
        local_rows = torch.arange(boards.shape[0], device=device)
        boards[local_rows, root_cell] = root_digit.to(boards.dtype)

        for level in range(1, depth + 1):
            logits_parts: list[Tensor] = []
            score_parts: list[Tensor] = []
            for lo in range(0, boards.shape[0], max_rows):
                logits_part, score_part = rollout(
                    boards[lo : lo + max_rows],
                    frontier_puzzle[lo : lo + max_rows],
                )
                logits_parts.append(logits_part)
                score_parts.append(score_part)
            logits = torch.cat(logits_parts)
            verifier_scores = torch.cat(score_parts).float()
            predictions = logits.argmax(dim=-1)
            visited_predictions.append(predictions)
            visited_puzzles.append(frontier_puzzle)
            nodes.scatter_add_(0, frontier_puzzle, torch.ones_like(frontier_puzzle))

            positive_rows = (verifier_scores >= acceptance_threshold).nonzero(
                as_tuple=True,
            )[0]
            if positive_rows.shape[0]:
                positive_puzzles = frontier_puzzle[positive_rows]
                first_positive = torch.full(
                    (batch_size,),
                    boards.shape[0],
                    dtype=torch.int64,
                    device=device,
                )
                first_positive.scatter_reduce_(
                    0,
                    positive_puzzles,
                    positive_rows,
                    reduce="amin",
                    include_self=True,
                )
                winning_puzzles = (
                    (first_positive < boards.shape[0]) & ~accepted
                ).nonzero(as_tuple=True)[0]
                winning_rows = first_positive[winning_puzzles]
                grids[winning_puzzles] = predictions[winning_rows].to(grids.dtype)
                scores[winning_puzzles] = verifier_scores[winning_rows]
                depth_used[winning_puzzles] = level
                accepted[winning_puzzles] = True

            if level == depth:
                break
            child_count = candidates ** (level + 1)
            expandable = ~accepted[frontier_puzzle] & (
                nodes[frontier_puzzle] + child_count <= budget
            )
            keep = expandable.nonzero(as_tuple=True)[0]
            if not keep.shape[0]:
                break
            parent_boards = boards[keep]
            child_cell, child_digits = select_pin_candidates(
                logits[keep],
                parent_boards,
                n_cells=1,
                n_digits=candidates,
            )
            boards = parent_boards.repeat_interleave(candidates, dim=0)
            frontier_puzzle = frontier_puzzle[keep].repeat_interleave(candidates)
            pinned_cell = child_cell[:, 0].repeat_interleave(candidates)
            pinned_digit = child_digits[:, 0].reshape(-1)
            local_rows = torch.arange(boards.shape[0], device=device)
            boards[local_rows, pinned_cell] = pinned_digit.to(boards.dtype)

    return LearnedPinSearchResult(
        accepted=accepted,
        grids=grids,
        scores=scores,
        nodes=nodes,
        depth=depth_used,
        visited_predictions=visited_predictions,
        visited_puzzles=visited_puzzles,
    )


def learned_hps_output_width(grid_len: int) -> int:
    """Packed row width for ``grid_len`` cells (module-docstring layout)."""
    return 7 + 2 * grid_len


def pack_search_rows(
    result: SearchResult,
    *,
    solution_visited: Tensor | None = None,
) -> Tensor:
    """Pack a :class:`SearchResult` into the shared dump row layout.

    Args:
      result: One batch's search result.
      solution_visited: Optional ``[B]`` label-joined trace flags
        (:func:`solution_visited_flags`); None packs zeros.

    Returns:
      rows: ``[B, 7 + 2G]`` float32 packed rows (module-docstring layout).

    """
    b = result.final_predictions.shape[0]
    if solution_visited is None:
        solution_visited = torch.zeros(
            b,
            dtype=torch.bool,
            device=result.scores.device,
        )
    candidate_count = result.scored.to(torch.int64) + result.nodes
    return torch.cat(
        [
            result.scores.reshape(b, 1).float(),
            result.accepted.reshape(b, 1).float(),
            result.nodes.reshape(b, 1).float(),
            result.depth.reshape(b, 1).float(),
            candidate_count.reshape(b, 1).float(),
            solution_visited.reshape(b, 1).float(),
            torch.zeros(b, 1, dtype=torch.float32, device=result.scores.device),
            result.root_predictions.float(),
            result.final_predictions.float(),
        ],
        dim=-1,
    )


def solution_visited_flags(
    result: SearchResult,
    label: Tensor,
    *,
    ignore_label_id: int = -100,
) -> Tensor:
    """``[B]`` whether the true solution appeared at the root or any node.

    Post-hoc diagnostic ONLY: labels join after every search decision and
    output grid is frozen, so this flag can never route the search. A row
    with ``solution_visited`` and not ``accepted`` is a false reject.

    Args:
      result: One batch's search result (its label-free candidate trace).
      label: ``[B, G]`` solution tokens.
      ignore_label_id: Label value excluded from correctness.

    Returns:
      visited: ``[B]`` bool flags.

    """
    valid = label != ignore_label_id
    visited = _exact_rows(result.root_predictions, label, valid)
    for predictions, puzzles in zip(
        result.visited_predictions,
        result.visited_puzzles,
        strict=True,
    ):
        candidate_correct = _exact_rows(predictions, label[puzzles], valid[puzzles])
        visited[puzzles[candidate_correct]] = True
    return visited


def summarize_search(
    rows: Tensor,
    label: Tensor,
    *,
    ignore_label_id: int = -100,
) -> dict[str, float]:
    """Label-joined accuracy / error / coverage / cost scalars for dump rows.

    The learned-HPS metric semantics: exact and cell accuracy of the final
    grids, acceptance counts, ``false_accepts`` (accepted-but-wrong; the
    soundness read, must be 0), ``false_rejects`` (solution visited but not
    accepted), and node-cost totals.

    Args:
      rows: ``[N, 7 + 2G]`` packed rows (valid rows only, any device).
      label: ``[N, G]`` solution tokens aligned with ``rows``.
      ignore_label_id: Label value excluded from correctness.

    Returns:
      metrics: Scalar summary dict.

    """
    grid_len = label.shape[-1]
    if rows.shape[-1] != learned_hps_output_width(grid_len):
        raise ValueError(
            f"packed width {rows.shape[-1]} does not match learned-HPS width "
            f"{learned_hps_output_width(grid_len)}.",
        )
    rows = rows.detach().float().cpu()
    label = label.detach().to(torch.int64).cpu()
    valid = label != ignore_label_id
    final_predictions = rows[:, -grid_len:].to(torch.int64)
    cell_correct = (final_predictions == label) & valid
    loss_counts = valid.sum(dim=-1)
    exact = (cell_correct.sum(dim=-1) == loss_counts) & (loss_counts > 0)
    accepted = rows[:, 1] > 0.5
    solution_visited = rows[:, 5] > 0.5
    n_puzzles = int((loss_counts > 0).sum())
    n_accepted = int(accepted.sum())
    return {
        "n_puzzles": float(n_puzzles),
        "exact_accuracy": int(exact.sum()) / max(1, n_puzzles),
        "cell_accuracy": int(cell_correct.sum()) / max(1, int(loss_counts.sum())),
        "accepted": float(n_accepted),
        "false_accepts": float((accepted & ~exact).sum()),
        "false_rejects": float((solution_visited & ~accepted).sum()),
        "solution_visited": float(solution_visited.sum()),
        "candidate_count": float(rows[:, 4].sum()),
        "nodes": float(rows[:, 2].sum()),
        "depth": float(rows[accepted, 3].sum()) / max(1, n_accepted),
        "algorithmic_decision_calls": float(rows[:, 6].sum()),
    }


def write_member_dump(
    path: str | Path,
    rows: Tensor,
    *,
    media: Tensor,
    label: Tensor | None = None,
) -> None:
    """Write one member's packed search rows as an atomic npz dump.

    npz keys: ``rows`` float32 ``[N, 7 + 2G]`` (module-docstring layout),
    ``media`` uint8 ``[N, G]``, and ``label`` uint8 ``[N, G]`` when given.
    Written via temp file + atomic rename, so a present file is complete.

    Args:
      path: Destination ``.npz`` path (parent directories are created).
      rows: ``[N, 7 + 2G]`` packed rows (:func:`pack_search_rows`), valid
        rows only, in eval arrival order.
      media: ``[N, G]`` input boards aligned with ``rows``.
      label: Optional ``[N, G]`` solution tokens for post-hoc scoring.

    """
    grid_len = media.shape[-1]
    if rows.shape[-1] != learned_hps_output_width(grid_len):
        raise ValueError(
            f"packed width {rows.shape[-1]} does not match learned-HPS width "
            f"{learned_hps_output_width(grid_len)}.",
        )
    if rows.shape[0] != media.shape[0]:
        raise ValueError(
            f"rows and media disagree on N: {rows.shape[0]} != {media.shape[0]}.",
        )
    if label is not None and label.shape != media.shape:
        raise ValueError(
            f"label and media disagree on shape: {tuple(label.shape)} != "
            f"{tuple(media.shape)}.",
        )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.tmp.{os.getpid()}.npz")
    rows_array = rows.detach().cpu().numpy().astype(np.float32)
    media_array = media.detach().cpu().numpy().astype(np.uint8)
    if label is None:
        np.savez_compressed(temporary, rows=rows_array, media=media_array)
    else:
        np.savez_compressed(
            temporary,
            rows=rows_array,
            media=media_array,
            label=label.detach().cpu().numpy().astype(np.uint8),
        )
    temporary.replace(destination)


def read_member_dump(path: str | Path) -> MemberDump:
    """Read and decode a member dump written by :func:`write_member_dump`.

    Args:
      path: Source ``.npz`` path.

    Returns:
      dump: The decoded :class:`MemberDump` (CPU tensors).

    """
    with _npz(Path(path)) as archive:
        rows = torch.from_numpy(archive["rows"].astype(np.float32))
        media = torch.from_numpy(archive["media"].astype(np.uint8))
        label = (
            torch.from_numpy(archive["label"].astype(np.uint8))
            if "label" in archive
            else None
        )
    grid_len = media.shape[1]
    if rows.shape[1] != learned_hps_output_width(grid_len):
        raise ValueError(
            f"packed width {rows.shape[1]} does not match learned-HPS width "
            f"{learned_hps_output_width(grid_len)}.",
        )
    return MemberDump(
        rows=rows,
        score=rows[:, 0],
        accepted=rows[:, 1] > 0.5,
        nodes=rows[:, 2].to(torch.int64),
        depth=rows[:, 3].to(torch.int64),
        candidate_count=rows[:, 4].to(torch.int64),
        solution_visited=rows[:, 5] > 0.5,
        root_predictions=rows[:, 7 : 7 + grid_len].to(torch.uint8),
        final_predictions=rows[:, -grid_len:].to(torch.uint8),
        media=media,
        label=label,
    )


def _empty_result(media: Tensor) -> SearchResult:
    """All-pad batch (``valid_count == 0``): nothing scored, no rollouts."""
    b = media.shape[0]
    device = media.device
    false = torch.zeros(b, dtype=torch.bool, device=device)
    return SearchResult(
        accepted=false,
        root_predictions=media.long().clone(),
        final_predictions=media.long().clone(),
        scores=torch.zeros(b, dtype=torch.float32, device=device),
        root_scores=torch.zeros(b, dtype=torch.float32, device=device),
        nodes=torch.zeros(b, dtype=torch.int64, device=device),
        depth=torch.full((b,), -1, device=device),
        scored=false.clone(),
        visited_predictions=[],
        visited_puzzles=[],
    )


def _exact_rows(preds: Tensor, label: Tensor, valid: Tensor) -> Tensor:
    """``[B]`` exact-correctness over label-valid cells (pad rows False)."""
    counts = valid.sum(dim=-1)
    return (((preds == label) & valid).sum(dim=-1) == counts) & (counts > 0)


# A group is violated iff any digit token (2-10) appears more than once or any non-digit
# token (pad 0 / blank 1) appears at all.
def _violated_group_counts(preds: Tensor, groups: Tensor) -> Tensor:
    """``[B]`` violated-27-group count at ``preds`` argmax (label-free)."""
    group_tokens = preds[:, groups]
    group_tokens = torch.maximum(
        group_tokens,
        torch.zeros_like(group_tokens),
    ).clamp(max=10)
    sorted_tokens = group_tokens.sort(dim=2).values
    duplicate_digits = (
        (sorted_tokens[..., 1:] == sorted_tokens[..., :-1])
        & (sorted_tokens[..., 1:] >= 2)
    ).any(dim=2)
    invalid_tokens = (group_tokens < 2).any(dim=2)
    return (duplicate_digits | invalid_tokens).sum(dim=1).float()


# ---------------------------------------------------------------------------
# Verifier: net, data stream, dedicated fit, acceptor.
# ---------------------------------------------------------------------------


def sudoku_groups() -> Tensor:
    """Cell indices of the 27 Sudoku groups.

    Returns:
      groups: ``[27, 9]`` long tensor; rows 0-8 are the 9 grid rows, 9-17 the
        columns, 18-26 the 3x3 boxes.

    """
    cells = torch.arange(81).reshape(9, 9)
    boxes = cells.reshape(3, 3, 3, 3).permute(0, 2, 1, 3).reshape(9, 9)
    return torch.cat([cells, cells.T, boxes]).contiguous()


def group_attention_mask() -> Tensor:
    """Boolean ``[81, 81]`` mask; True where two cells share a group."""
    member = torch.zeros(27, 81, dtype=torch.bool)
    member.scatter_(1, sudoku_groups(), True)
    return (member.float().T @ member.float()) > 0


class SudokuVerifier(nn.Module):
    """Tiny transformer verifier over 81 (puzzle, candidate) cell tokens.

    Output is ``[B, 82]``: column 0 is the grid logit (positive = candidate
    IS the solution), columns 1: are per-cell wrongness logits (positive =
    this cell differs from the solution), used as dense auxiliary
    supervision.
    """

    class Config(Fig["SudokuVerifier"]):
        """Architecture of the verifier."""

        width: int = 128
        """Embedding and attention width."""

        depth: int = 4
        """Number of transformer blocks."""

        heads: int = 4
        """Attention heads per block."""

        attention_scope: Literal["full", "group"] = "group"
        """"group" (the recipe) restricts attention to cells sharing a row,
        column, or box (the 27-group incidence graph) -- a structural bias,
        not a rule oracle. "full" lets every cell attend everywhere (the
        measured control: AUROC > 0.97 but no usable zero-false-accept
        operating point)."""

        vocab_size: int = 11
        """Token vocabulary: 0=pad, 1=blank, 2-10 = digits 1-9."""

        def cost(self, *, batch_size: int, dtype: torch.dtype | None) -> Cost:
            """Cost one scoring forward over 81-cell rows, and its backward.

            Args:
              batch_size: (puzzle, candidate) rows in this invocation.
              dtype: Activation dtype; ``None`` is torch's default.

            Returns:
              cost: Whole-invocation FLOPs, logical bytes, and ownership.

            """
            width = self.width
            cells = 81
            rows = cells * batch_size
            over_cells = partial(
                cost,
                seq_len=cells,
                batch_size=batch_size,
                dtype=dtype,
            )
            over_rows = partial(cost, seq_len=1, batch_size=batch_size, dtype=dtype)
            norm = LayerNorm.Config(width, elementwise_affine=True)
            linear = partial(Linear.Config, bias=True)
            add = elementwise_cost(
                primal=width * rows,
                adjoint=width * rows,
                channels=width,
                inputs=2,
                rows=rows,
                dtype=dtype,
            )
            block = (
                over_cells(norm).tile(2, copies=2)
                + over_cells(linear(width, 3 * width))
                + attention_kernel_cost(
                    seq_len=cells,
                    batch_size=batch_size,
                    dtype=dtype,
                    num_heads=self.heads,
                    channels_head=width // self.heads,
                )
                + over_cells(linear(width, width))
                + over_cells(linear(width, 4 * width))
                + _gelu_cost(channels=4 * width, rows=rows, dtype=dtype)
                + over_cells(linear(4 * width, width))
                + add.tile(2)
            )
            tables = (
                over_cells(Embedding.Config(self.vocab_size, width)).tile(2, copies=2)
                + over_cells(Embedding.Config(cells, width))
                + add.tile(2)
            )
            heads = (
                over_cells(norm)
                + over_cells(linear(width, 1))
                + reduction_cost(
                    input_elements=rows * width,
                    output_groups=batch_size * width,
                    dtype=dtype,
                )
                + over_rows(linear(width, width))
                + _gelu_cost(channels=width, rows=batch_size, dtype=dtype)
                + over_rows(linear(width, 1))
            )
            return tables + block.tile(self.depth, copies=self.depth) + heads

    def __init__(self, config: Config) -> None:
        super().__init__()
        if config.width % config.heads != 0:
            raise ValueError(
                f"width {config.width} must divide by heads {config.heads}.",
            )
        self.config = config
        width = config.width
        self.candidate_embedding = nn.Embedding(config.vocab_size, width)
        self.puzzle_embedding = nn.Embedding(config.vocab_size, width)
        self.position_embedding = nn.Embedding(81, width)
        self.blocks = nn.ModuleList(
            _VerifierBlock(width=width, heads=config.heads) for _ in range(config.depth)
        )
        self.final_norm = nn.LayerNorm(width)
        self.cell_head = nn.Linear(width, 1)
        self.grid_head = nn.Sequential(
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, 1),
        )
        mask = (
            group_attention_mask()
            if config.attention_scope == "group"
            else torch.ones(81, 81, dtype=torch.bool)
        )
        self.attention_mask = nn.Buffer(mask, persistent=False)

    @override
    def forward(self, puzzle: Tensor, candidate: Tensor) -> Tensor:
        """Score a batch of (puzzle, candidate) rows.

        Args:
          puzzle: ``[B, 81]`` original puzzle tokens (1=blank, 2-10 digits).
          candidate: ``[B, 81]`` complete candidate grid tokens.

        Returns:
          logits: ``[B, 82]``; column 0 the grid logit, columns 1: per-cell
            wrongness logits.

        """
        positions = torch.arange(81, device=candidate.device)
        x = (
            self.candidate_embedding(candidate.long())
            + self.position_embedding(positions)
            + self.puzzle_embedding(puzzle.long())
        )
        for block in self.blocks:
            x = block(x, self.attention_mask)
        x = self.final_norm(x)
        cell_logits = self.cell_head(x).squeeze(-1)
        grid_logit = self.grid_head(x.mean(dim=-2)).squeeze(-1)
        return torch.cat([grid_logit.unsqueeze(-1).float(), cell_logits.float()], -1)


class VerifierLoss:
    """Grid BCE plus masked per-cell auxiliary BCE, per-sample (unreduced)."""

    class Config(Fig["VerifierLoss"]):
        """Loss weights of the verifier objective."""

        cell_weight: float = 0.5
        """Weight of the per-cell wrongness auxiliary BCE term."""

        negative_weight: float = 1.0
        """Extra BCE weight on wrong candidates (false-accept aversion)."""

    def __init__(self, config: Config) -> None:
        self.config = config

    def __call__(
        self,
        output: Tensor,
        *,
        label: Tensor,
        cell_label: Tensor,
        cell_mask: Tensor,
    ) -> Tensor:
        """Compute the per-sample loss vector.

        Args:
          output: ``[B, 82]`` verifier logits.
          label: ``[B]`` grid labels (1 = candidate is the solution).
          cell_label: ``[B, 81]`` per-cell wrongness labels.
          cell_mask: ``[B, 81]`` mask over cells entering the auxiliary term.

        Returns:
          loss: ``[B]`` per-sample loss (grid + weighted cell auxiliary).

        """
        grid_logit = output[:, 0].float()
        cell_logits = output[:, 1:].float()
        label = label.float()
        cell_label = cell_label.float()
        cell_mask = cell_mask.float()
        weight = torch.where(label > 0.5, 1.0, self.config.negative_weight)
        grid_loss = functional.binary_cross_entropy_with_logits(
            grid_logit,
            label,
            weight=weight,
            reduction="none",
        )
        cell_loss = functional.binary_cross_entropy_with_logits(
            cell_logits,
            cell_label,
            reduction="none",
        )
        cell_loss = (cell_loss * cell_mask).sum(-1) / cell_mask.sum(-1).clamp(min=1)
        return grid_loss + self.config.cell_weight * cell_loss


class VerifierData:
    """On-device verifier data: generated train stream + frozen eval strata.

    Train batches mix, on the fly and on device:

    - generated positives: train-split solutions under fresh
      validity-preserving augmentation (digit permutation + dihedral
      symmetry);
    - transposition-swap negatives: digit-histogram-preserving cell swaps
      (kills count-based shortcuts);
    - Hamming-k corruption negatives: k non-given cells of a solution
      replaced with different uniform digits (any single-cell change to a
      complete solution breaks a row, column, AND box);
    - optional clue-contradiction negatives: a rule-valid solution from a
      sibling row presented under this row's puzzle;
    - real solver candidates from the harvest corpus (see the module
      docstring for the shard contract), restricted to the train groups.

    Labels are exactly ``candidate == solution`` -- no Sudoku rule predicate
    anywhere. Every batch dict carries ``puzzle`` / ``candidate`` /
    ``label`` / ``cell_label`` / ``cell_mask`` / ``stratum`` tensors.

    The eval mix is FROZEN (fixed ``eval_seed``, fixed strata): 0 = harvest
    calibration groups, 1 = harvest holdout groups, 2 = search-node
    candidates from the OPTIONAL node corpus (skipped with a warning when
    the file is missing; the later strata are unaffected because the node
    block consumes no RNG draws), 3 = dev positives, 4/5 = dev Hamming-1/2
    corruptions, 6 = dev swap negatives.
    """

    class Config(Fig["VerifierData"]):
        """Data-axis knobs of the verifier candidate batch."""

        base_dir: Path | str | None = None
        """Resource root; ``None`` resolves at construction beneath
        ``/opt/scratch``."""

        working_dir: Path | str = "/datasets/sudoku-extreme"
        """Sudoku dataset root with train/ and test/ splits; resolved beneath
        ``base_dir``."""

        harvest_dir: str | Path = "/runs/harvest/harvest"
        """Directory of ``shard-*.npz`` real solver-candidate shards (the
        harvest corpus; module docstring documents the contract). A relative
        logical path resolves beneath ``base_dir``; the default is where
        :class:`~Harvest` writes at its default ``experiment_name="harvest"``."""

        node_corpus: str | Path = "/runs/harvest/harvest/node_corpus.npz"
        """OPTIONAL search-node candidate dump (fields ``media`` /
        ``final_prediction`` / ``global_index``). Feeds eval stratum 2 only,
        never training; a missing file skips the stratum with a warning."""

        batch_size: int = 512
        """Training rows per batch."""

        eval_batch_size: int = 4096
        """Eval rows per batch (ragged tail allowed)."""

        steps_per_epoch: int = 250
        """Generated train batches per epoch (stream is infinite)."""

        real_fraction: float = 0.3
        """Batch fraction drawn from real harvest solver candidates."""

        positive_fraction: float = 0.3
        """Batch fraction of clean solutions (generated positives)."""

        swap_fraction: float = 0.15
        """Batch fraction of transposition-swap negatives."""

        contradiction_fraction: float = 0.0
        """Batch fraction of sibling-row clue-contradiction negatives,
        taken out of the Hamming-corruption remainder."""

        hamming_choices: tuple[int, ...] = (1, 1, 1, 2, 2, 3, 4, 8, 16)
        """Corruption sizes sampled uniformly (repeats skew the mix)."""

        swap_choices: tuple[int, ...] = (1, 1, 2, 3)
        """Swap-pair counts sampled uniformly."""

        train_group_end: int = 700
        """Harvest base groups [0, end) train the verifier."""

        calibration_group_end: int = 750
        """Harvest groups [train_group_end, end) freeze the threshold."""

        holdout_group_end: int = 800
        """Harvest groups [calibration_group_end, end) are internal holdout.
        The 700/750/800 defaults assume the full >= 800-group harvest; scale
        all three ends down together for smaller corpora."""

        dev_puzzles: int = 5000
        """Dev slice: first N test puzzles in file order."""

        device: torch.device | str | None = None
        """Device for cached data and generation."""

        seed: int = 0
        """Train-stream seed (folded with the epoch index)."""

        eval_seed: int = 20_260_710
        """Frozen eval-mix seed. Do not change: the dev corruption strata
        must stay byte-identical across runs."""

        @override
        def finalize(self) -> Self:
            base = self.base_dir if self.base_dir is not None else Path("/opt/scratch")
            if isinstance(self.working_dir, str):
                self.working_dir = resolve_working_dir(base, self.working_dir)
            if isinstance(self.harvest_dir, str):
                self.harvest_dir = resolve_working_dir(base, self.harvest_dir)
            if isinstance(self.node_corpus, str):
                self.node_corpus = resolve_working_dir(base, self.node_corpus)
            return super().finalize()

    def __init__(self, config: Config) -> None:
        self.config = config
        self.device = get_device(config.device)
        if config.steps_per_epoch < 1:
            raise ValueError(
                f"steps_per_epoch must be >= 1, got {config.steps_per_epoch}.",
            )
        fractions = (
            config.real_fraction,
            config.positive_fraction,
            config.swap_fraction,
            config.contradiction_fraction,
        )
        if min(fractions) < 0 or sum(fractions) > 1:
            raise ValueError(f"batch fractions {fractions} must be in [0, 1].")
        if (
            config.train_group_end < 0
            or config.train_group_end > config.calibration_group_end
            or config.calibration_group_end > config.holdout_group_end
        ):
            raise ValueError(
                "group ends must satisfy 0 <= train <= calibration <= holdout, "
                f"got {config.train_group_end}/{config.calibration_group_end}/"
                f"{config.holdout_group_end}.",
            )

        # ``finalize`` already resolved every logical path beneath ``base_dir``;
        # read the finalized values verbatim. Re-resolving here would rebase an
        # already-absolute path a second time (``base / base / ...``).
        self.dataset_dir = Path(config.working_dir)
        self.harvest_dir = Path(config.harvest_dir)
        self.node_corpus = Path(config.node_corpus)

        train = load_puzzle_dataset(self.dataset_dir, "train")
        self.train_inputs = train["inputs"].to(self.device)
        self.train_labels = train["labels"].to(self.device)

        harvest = _load_harvest(self.harvest_dir, device=self.device)
        self.harvest_puzzle, self.harvest_candidate, flat_view, groups = harvest
        if int(flat_view.max()) >= len(self.train_labels):
            raise ValueError("harvest flat_view_id exceeds the train file.")
        if not torch.equal(self.harvest_puzzle, self.train_inputs[flat_view].long()):
            raise ValueError(
                "harvest original rows must equal train inputs[flat_view_id]; "
                "the harvest/train-file pairing contract is broken.",
            )
        self.harvest_solution = self.train_labels[flat_view].long()
        n_groups = int(groups.max()) + 1 if len(groups) else 0
        if config.holdout_group_end > n_groups:
            raise ValueError(
                f"holdout_group_end {config.holdout_group_end} exceeds the "
                f"harvest corpus ({n_groups} base groups): the calibration/"
                "holdout strata would be silently empty. Scale the train/"
                "calibration/holdout ends down together for smaller corpora.",
            )
        self._harvest_train_rows = (groups < config.train_group_end).nonzero()[:, 0]
        if config.real_fraction > 0 and not len(self._harvest_train_rows):
            raise ValueError(
                "real_fraction > 0 but no harvest rows fall in the train "
                f"groups [0, {config.train_group_end}).",
            )

        self._epoch = 0
        self._eval_blocks = self._build_eval_blocks(groups)

    def train_dataloader(self) -> _VerifierTrainIterator:
        """Seeded generated-batch stream; epoch index folds into the seed."""
        iterator = _VerifierTrainIterator(self, epoch=self._epoch)
        self._epoch += 1
        return iterator

    def eval_dataloader(self) -> _VerifierEvalIterator:
        """Frozen eval mix in eval_batch_size chunks."""
        return _VerifierEvalIterator(
            self._eval_blocks,
            batch_size=self.config.eval_batch_size,
        )

    class StateDict(TypedDict):
        """The train-stream epoch counter; tolerated absent on load."""

        train_epochs: NotRequired[int]

    def state_dict(self) -> StateDict:
        """Persist the train-stream epoch counter."""
        return {"train_epochs": self._epoch}

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        """Restore the train-stream epoch counter."""
        state = cast(VerifierData.StateDict, state_dict)
        if "train_epochs" in state:
            self._epoch = state["train_epochs"]

    def generate_batch(self, generator: torch.Generator) -> dict[str, Tensor]:
        """Generate one mixed training batch on device.

        Args:
          generator: Dedicated RNG stream for every draw in the batch.

        Returns:
          batch: Dict of ``puzzle`` / ``candidate`` / ``label`` /
            ``cell_label`` / ``cell_mask`` / ``stratum`` tensors; rows are
            positives, then swaps, then Hamming negatives, then real harvest
            candidates.

        """
        cfg = self.config
        n_real, n_positive, n_swap, n_contradiction = _batch_partition_counts(
            cfg.batch_size,
            (
                cfg.real_fraction,
                cfg.positive_fraction,
                cfg.swap_fraction,
                cfg.contradiction_fraction,
            ),
        )
        n_hamming = cfg.batch_size - n_real - n_positive - n_swap - n_contradiction

        rows = torch.randint(
            len(self.train_inputs),
            (n_positive + n_swap + n_hamming,),
            device=self.device,
            generator=generator,
        )
        puzzle = self.train_inputs[rows].long()
        solution = self.train_labels[rows].long()
        candidate = solution.clone()
        given = puzzle >= 2

        swap_slice = slice(n_positive, n_positive + n_swap)
        pairs = _choices(cfg.swap_choices, n_swap, generator=generator)
        candidate[swap_slice] = _swap_cells(
            candidate[swap_slice],
            given[swap_slice],
            pairs=pairs,
            generator=generator,
        )
        hamming_slice = slice(n_positive + n_swap, None)
        ks = _choices(cfg.hamming_choices, n_hamming, generator=generator)
        candidate[hamming_slice] = _corrupt_hamming(
            candidate[hamming_slice],
            given[hamming_slice],
            k=ks,
            generator=generator,
        )

        parts_puzzle = [puzzle]
        parts_candidate = [candidate]
        parts_solution = [solution]
        if n_contradiction:
            contradiction = self._contradiction_rows(
                n_contradiction,
                generator=generator,
            )
            parts_puzzle.append(contradiction[0])
            parts_candidate.append(contradiction[1])
            parts_solution.append(contradiction[2])
        if n_real:
            real = self._harvest_train_rows[
                torch.randint(
                    len(self._harvest_train_rows),
                    (n_real,),
                    device=self.device,
                    generator=generator,
                )
            ]
            parts_puzzle.append(self.harvest_puzzle[real])
            parts_candidate.append(self.harvest_candidate[real])
            parts_solution.append(self.harvest_solution[real])

        puzzle = torch.cat(parts_puzzle)
        candidate = torch.cat(parts_candidate)
        solution = torch.cat(parts_solution)
        puzzle, candidate, solution = _augment_consistent(
            puzzle,
            candidate,
            solution,
            generator=generator,
        )
        return _finish_batch(puzzle, candidate, solution)

    def _contradiction_rows(
        self,
        count: int,
        *,
        generator: torch.Generator,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return sibling solutions under unrelated puzzle givens."""
        rows = torch.randint(
            len(self.train_inputs),
            (count,),
            device=self.device,
            generator=generator,
        )
        siblings = torch.randint(
            len(self.train_inputs),
            (count,),
            device=self.device,
            generator=generator,
        )
        return (
            self.train_inputs[rows].long(),
            self.train_labels[siblings].long(),
            self.train_labels[rows].long(),
        )

    def _build_eval_blocks(self, harvest_groups: Tensor) -> list[dict[str, Tensor]]:
        """Materialize the frozen per-stratum eval blocks (fixed eval seed)."""
        cfg = self.config
        generator = torch.Generator(device=self.device)
        generator.manual_seed(cfg.eval_seed)
        blocks: list[dict[str, Tensor]] = []

        calibration = (
            (harvest_groups >= cfg.train_group_end)
            & (harvest_groups < cfg.calibration_group_end)
        ).nonzero()[:, 0]
        holdout = (
            (harvest_groups >= cfg.calibration_group_end)
            & (harvest_groups < cfg.holdout_group_end)
        ).nonzero()[:, 0]
        for stratum, rows in ((0, calibration), (1, holdout)):
            blocks.append(
                _finish_batch(
                    self.harvest_puzzle[rows],
                    self.harvest_candidate[rows],
                    self.harvest_solution[rows],
                    stratum=stratum,
                ),
            )

        test = load_puzzle_dataset(
            self.dataset_dir,
            "test",
            max_samples=cfg.dev_puzzles,
        )
        test_inputs = test["inputs"].to(self.device).long()
        test_labels = test["labels"].to(self.device).long()

        # Stratum 2 (node-corpus candidates) is optional and consumes no
        # generator draws, so skipping it leaves strata 3-6 byte-identical.
        if self.node_corpus.exists():
            with _npz(self.node_corpus) as nodes:
                node_puzzle = torch.from_numpy(np.array(nodes["media"])).to(self.device)
                node_candidate = torch.from_numpy(
                    np.array(nodes["final_prediction"]),
                ).to(self.device)
                node_index = torch.from_numpy(np.array(nodes["global_index"])).to(
                    self.device,
                )
            if not node_index.numel():
                raise ValueError(
                    f"node corpus {self.node_corpus} holds no rows; delete it "
                    "to skip eval stratum 2 or regenerate it "
                    "(Harvest.regenerate_node_corpus).",
                )
            if int(node_index.min()) < 0 or int(node_index.max()) >= len(test_labels):
                raise ValueError("node corpus global_index exceeds the dev slice.")
            if not (
                len(node_puzzle) == len(node_candidate) == len(node_index)
                and torch.equal(node_puzzle.long(), test_inputs[node_index.long()])
            ):
                raise ValueError(
                    "node corpus media must equal test inputs at global_index; "
                    "the node-corpus/dataset binding is broken.",
                )
            blocks.append(
                _finish_batch(
                    node_puzzle.long(),
                    node_candidate.long(),
                    test_labels[node_index.long()],
                    stratum=2,
                ),
            )
        else:
            logger.warning(
                "node corpus not found at %s; eval stratum 2 (search-node "
                "candidates) skipped.",
                self.node_corpus,
            )

        given = test_inputs >= 2
        blocks.append(_finish_batch(test_inputs, test_labels, test_labels, stratum=3))
        for stratum, k in ((4, 1), (5, 2)):
            corrupted = _corrupt_hamming(
                test_labels.clone(),
                given,
                k=torch.full((len(test_labels),), k, device=self.device),
                generator=generator,
            )
            blocks.append(
                _finish_batch(test_inputs, corrupted, test_labels, stratum=stratum),
            )
        swapped = _swap_cells(
            test_labels.clone(),
            given,
            pairs=torch.full((len(test_labels),), 2, device=self.device),
            generator=generator,
        )
        blocks.append(_finish_batch(test_inputs, swapped, test_labels, stratum=6))
        return blocks


class VerifierFit:
    """Dedicated verifier training run (NOT the generator trainer).

    Recipe (ported verbatim): plain ``torch.optim.AdamW`` with lr 3e-4,
    betas (0.9, 0.95) [hardcoded], weight decay 0.01;
    ``CosineAnnealingLR(T_max=max_steps, eta_min=3e-5)`` stepped once per
    optimizer step; per-batch mean gradient (sum-backward then one division);
    global-norm gradient clip 1.0; bf16 autocast around the forward only; no
    EMA, no Muon, no custom lr_scale. Three seeds (0/1/2) of this fit form
    the zero-false-accept committee.

    One deliberate divergence from the reference run: its training loop
    replayed the first epoch's generated stream for the entire fit (a
    dataloader-lifecycle accident that froze the epoch counter at zero),
    whereas this fit requests a fresh dataloader each epoch, so every epoch
    draws a freshly seeded stream (``seed + epoch``) by design.

    :meth:`run` trains ``max_steps`` steps, writes the final checkpoint
    (layout in the module docstring) plus a small ``metrics.json`` into the
    run dir, and returns the checkpoint path.
    """

    class Config(Fig["VerifierFit"]):
        """One verifier committee member's training recipe."""

        model: SudokuVerifier.Config = field(default_factory=SudokuVerifier.Config)
        """Verifier architecture (default: the group-attention recipe)."""

        loss: VerifierLoss.Config = field(default_factory=VerifierLoss.Config)
        """Grid + per-cell auxiliary BCE objective."""

        dataset: VerifierData.Config = field(default_factory=VerifierData.Config)
        """Generated stream + harvest corpus feeding the fit."""

        experiment_name: str = "verifier_s0"
        """Run identity; names the run dir (committee members use fresh
        names, e.g. verifier_s0 / verifier_s1 / verifier_s2)."""

        base_dir: Path | str | None = None
        """Scratch root; ``None`` resolves beneath ``/opt/scratch`` at
        construction (see :class:`VerifierData.Config`)."""

        run_dir: str | Path = "/runs/{experiment_name}"
        """Run output root resolved beneath ``base_dir``; checkpoints land in
        ``<run_dir>/checkpoints/``. ``{experiment_name}`` is filled at
        construction."""

        seed: int = 0
        """Weight-init seed (``torch.manual_seed`` before construction).
        The data stream is seeded separately via ``dataset.seed``."""

        max_steps: int = 4_000
        """Optimizer steps; also the cosine horizon (``T_max``)."""

        learning_rate: float = 3e-4
        """AdamW peak learning rate."""

        learning_rate_min: float = 3e-5
        """Cosine floor (``eta_min``)."""

        weight_decay: float = 0.01
        """AdamW weight decay."""

        gradient_clip_norm: float = 1.0
        """Global gradient-norm clip applied before every optimizer step."""

        dtype: torch.dtype | None = torch.bfloat16
        """Autocast compute dtype for the forward pass (weights stay fp32;
        the loss runs outside autocast). None disables autocast."""

        device: torch.device | str | None = None
        """Training device."""

        @override
        def finalize(self) -> Self:
            if self.dataset.device is None:
                self.dataset.device = self.device
            if self.dataset.base_dir is None:
                self.dataset.base_dir = self.base_dir
            return super().finalize()

    def __init__(self, config: Config) -> None:
        if config.max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {config.max_steps}.")
        if config.gradient_clip_norm <= 0:
            raise ValueError(
                f"gradient_clip_norm must be positive, got "
                f"{config.gradient_clip_norm}.",
            )
        self.config = config
        self.device = get_device(config.device)
        base = config.base_dir if config.base_dir is not None else Path("/opt/scratch")
        self.run_dir = resolve_working_dir(
            base,
            str(config.run_dir).format(experiment_name=config.experiment_name),
        )
        self.checkpoint_path = (
            self.run_dir / "checkpoints" / f"step_{config.max_steps:08d}.pt"
        )
        # Seed BEFORE construction: the member's weight init is the seed's
        # only consumer (the data stream runs on its own dedicated RNG).
        torch.manual_seed(config.seed)
        self.model = config.model.make().to(self.device)
        self.loss = config.loss.make()
        self.dataset = config.dataset.make()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.95),  # Recipe constant; deliberately not a knob.
            weight_decay=config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.max_steps,
            eta_min=config.learning_rate_min,
        )

    def run(self) -> Path:
        """Train ``max_steps`` steps and write the final checkpoint.

        Returns:
          checkpoint_path: The written
            ``<run_dir>/checkpoints/step_<max_steps:08d>.pt`` file.

        """
        cfg = self.config
        step = 0
        first_loss = math.nan
        last_loss = math.nan
        progress = tqdm(total=cfg.max_steps, desc=cfg.experiment_name, unit="step")
        while step < cfg.max_steps:
            for batch in self.dataset.train_dataloader():
                last_loss = self._train_step(batch)
                if step == 0:
                    first_loss = last_loss
                step += 1
                progress.update(1)
                if step >= cfg.max_steps:
                    break
        progress.close()
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_torch_save(
            {"step": {"model": self.model.state_dict(), "global_step": step}},
            self.checkpoint_path,
        )
        _atomic_write_json(
            self.run_dir / "metrics.json",
            {"step": step, "train/loss_first": first_loss, "train/loss": last_loss},
        )
        logger.info(
            "verifier fit %s: %d steps, loss %.4f -> %.4f, checkpoint %s",
            cfg.experiment_name,
            step,
            first_loss,
            last_loss,
            self.checkpoint_path,
        )
        return self.checkpoint_path

    def _train_step(self, batch: dict[str, Tensor]) -> float:
        """One optimizer step; returns the batch-mean loss."""
        cfg = self.config
        self.model.train()
        if cfg.dtype is not None:
            # cache_enabled=False preserves exact numerics across forwards.
            with torch.amp.autocast(
                device_type=self.device.type,
                dtype=cfg.dtype,
                cache_enabled=False,
            ):
                output = self.model(batch["puzzle"], batch["candidate"])
        else:
            output = self.model(batch["puzzle"], batch["candidate"])
        # The loss runs OUTSIDE autocast on the fp32 logits (ported step
        # semantics: autocast wraps the forward only).
        loss = self.loss(
            output,
            label=batch["label"],
            cell_label=batch["cell_label"],
            cell_mask=batch["cell_mask"],
        )
        # Ported gradient order: sum-backward, then ONE division of the
        # accumulated grads by the element count (== batch-mean gradient).
        loss.sum().backward()
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.div_(loss.numel())
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            cfg.gradient_clip_norm,
            foreach=True,
        )
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()
        return float(loss.detach().mean())


class CommitteeLock(Protocol):
    """What the sieve needs from the committee lock.

    :class:`VerifierAcceptor` inherits this protocol explicitly (nominal
    conformance) so its ``Config`` satisfies ``Makeable[CommitteeLock]``
    under checkers without structural ``type[cls]`` -> ``type[Protocol]``
    support; test stubs conform structurally.
    """

    def __call__(self, preds: Tensor, media: Tensor) -> Tensor:
        """``[N]`` bool acceptance of decoded grids against original boards."""
        ...


class VerifierAcceptor(CommitteeLock):
    """Frozen verifier committee scoring (puzzle, candidate) rows."""

    class Config(Fig["VerifierAcceptor"]):
        """Committee membership and the frozen operating rule."""

        model: SudokuVerifier.Config = field(default_factory=SudokuVerifier.Config)
        """Shared architecture of every committee member."""

        base_dir: Path | str | None = None
        """Scratch root; ``None`` resolves beneath ``/opt/scratch`` at
        construction."""

        checkpoint_paths: tuple[str | Path, ...] = (
            "/runs/verifier_s0/checkpoints/step_00004000.pt",
            "/runs/verifier_s1/checkpoints/step_00004000.pt",
            "/runs/verifier_s2/checkpoints/step_00004000.pt",
        )
        """Member checkpoints in the :class:`VerifierFit` layout
        (``state["step"]["model"]``) or the HF model-only layout (top-level
        ``state["model"]``). A relative logical path resolves beneath
        ``base_dir``. Acceptance is UNANIMOUS: the minimum member grid logit
        must STRICTLY exceed ``threshold``."""

        threshold: float = 0.0
        """Frozen accept threshold on the minimum member grid logit. The
        comparison is strict: a logit exactly at the threshold is rejected."""

        max_rows: int = 8_192
        """Scoring chunk width (memory guard)."""

        device: torch.device | str | None = None
        """Device for the committee."""

    def __init__(self, config: Config) -> None:
        if not config.checkpoint_paths:
            raise ValueError("the verifier committee needs >= 1 checkpoint.")
        if config.max_rows < 1:
            raise ValueError(f"max_rows must be >= 1, got {config.max_rows}.")
        self.config = config
        device = get_device(config.device)
        base = config.base_dir if config.base_dir is not None else Path("/opt/scratch")
        self.members: list[SudokuVerifier] = []
        for path in config.checkpoint_paths:
            member = config.model.make().to(device)
            # A ``Path`` is a literal override; a ``str`` is a logical path
            # resolved beneath the scratch base.
            resolved = (
                path if isinstance(path, Path) else resolve_working_dir(base, path)
            )
            # HF model-only files carry {"model": ...} unnested; the
            # VerifierFit layout nests it under "step".
            member.load_state_dict(_checkpoint_payload(resolved, device)["model"])
            member.eval()
            member.requires_grad_(False)
            self.members.append(member)

    @override
    def __call__(self, preds: Tensor, media: Tensor) -> Tensor:
        """Unanimous learned acceptance of decoded grids.

        Args:
          preds: ``[N, G]`` decoded candidate grids.
          media: ``[N, G]`` the ORIGINAL puzzle boards (never pinned
            boards -- pins are hypotheses, exactly like the rule path).

        Returns:
          accepted: ``[N]`` bool mask, True iff EVERY member's grid logit
            exceeds the frozen threshold.

        """
        return self.scores(preds, media) > self.config.threshold

    def scores(self, preds: Tensor, media: Tensor) -> Tensor:
        """Minimum member grid logit per row (the committee score).

        Args:
          preds: ``[N, G]`` decoded candidate grids.
          media: ``[N, G]`` the original puzzle boards.

        Returns:
          scores: ``[N]`` fp32 minimum member grid logits.

        """
        chunks: list[Tensor] = []
        with torch.inference_mode():
            for lo in range(0, preds.shape[0], self.config.max_rows):
                puzzle = media[lo : lo + self.config.max_rows]
                candidate = preds[lo : lo + self.config.max_rows]
                member_scores = torch.stack(
                    [
                        member(puzzle, candidate)[:, 0].float()
                        for member in self.members
                    ],
                )
                chunks.append(member_scores.min(dim=0).values)
        return torch.cat(chunks) if chunks else preds.new_zeros(0, dtype=torch.float32)


def _gelu_cost(*, channels: int, rows: int, dtype: torch.dtype | None) -> Cost:
    """Cost exact GELU over ``rows`` rows of ``channels``: erf forward, pdf back."""
    return elementwise_cost(
        primal=8 * channels * rows,
        adjoint=8 * channels * rows,
        channels=channels,
        rows=rows,
        dtype=dtype,
    )


class _VerifierBlock(nn.Module):
    """Pre-norm attention + MLP block with a cell-incidence attention mask."""

    def __init__(self, *, width: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.norm_attention = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)
        self.norm_mlp = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, 4 * width),
            nn.GELU(),
            nn.Linear(4 * width, width),
        )

    @override
    def forward(self, x: Tensor, attention_mask: Tensor) -> Tensor:
        batch, cells, width = x.shape
        qkv = self.qkv(self.norm_attention(x))
        q, k, v = qkv.reshape(batch, cells, 3, self.heads, -1).permute(2, 0, 3, 1, 4)
        attended = functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
        )
        attended = attended.transpose(1, 2).reshape(batch, cells, width)
        x = x + self.proj(attended)
        return x + self.mlp(self.norm_mlp(x))


class _VerifierTrainIterator:
    """One epoch of generated batches under a seeded generator."""

    def __init__(self, dataset: VerifierData, *, epoch: int) -> None:
        self._dataset = dataset
        self._epoch = epoch

    def __len__(self) -> int:
        return self._dataset.config.steps_per_epoch

    def __iter__(self) -> Iterator[dict[str, Tensor]]:
        generator = torch.Generator(device=self._dataset.device)
        generator.manual_seed(self._dataset.config.seed + self._epoch)
        for _ in range(self._dataset.config.steps_per_epoch):
            yield self._dataset.generate_batch(generator)


class _VerifierEvalIterator:
    """Chunk the frozen eval blocks into batches (ragged tail allowed)."""

    def __init__(self, blocks: list[dict[str, Tensor]], *, batch_size: int) -> None:
        self._blocks = blocks
        self._batch_size = batch_size

    def __len__(self) -> int:
        return sum(
            math.ceil(len(block["label"]) / self._batch_size) for block in self._blocks
        )

    def __iter__(self) -> Iterator[dict[str, Tensor]]:
        for block in self._blocks:
            total = len(block["label"])
            for start in range(0, total, self._batch_size):
                yield {
                    key: value[start : start + self._batch_size]
                    for key, value in block.items()
                }


def _finish_batch(
    puzzle: Tensor,
    candidate: Tensor,
    solution: Tensor,
    *,
    stratum: int = -1,
) -> dict[str, Tensor]:
    """Attach exact labels: ``label = (candidate == solution)`` per row."""
    cell_label = candidate != solution
    return {
        "puzzle": puzzle,
        "candidate": candidate,
        "label": (~cell_label.any(dim=-1)).float(),
        "cell_label": cell_label.float(),
        "cell_mask": torch.ones_like(cell_label),
        "stratum": torch.full(
            (len(puzzle),),
            stratum,
            dtype=torch.long,
            device=puzzle.device,
        ),
    }


class _HarvestShardEntry(TypedDict):
    """One shard's line in the harvest manifest."""

    file: str
    rows: int
    sha256: str


class _HarvestManifest(TypedDict):
    """The fields of ``manifest.json`` the loader binds shards to."""

    schema_version: Literal[1]
    groups_per_shard: int
    shards: list[_HarvestShardEntry]
    shard_count: int
    row_count: int


def _load_harvest(
    harvest_dir: Path,
    *,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Load the manifest-bound harvest shards and globalize their groups."""
    manifest_path = harvest_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"harvest manifest not found: {manifest_path}.")
    try:
        manifest = from_plain(loads(manifest_path.read_text()), _HarvestManifest)
    except (json.JSONDecodeError, ReadError) as error:
        raise ValueError(
            f"invalid harvest manifest {manifest_path}: {error}",
        ) from error
    groups_per_shard = manifest["groups_per_shard"]
    entries = manifest["shards"]
    declared_names = [entry["file"] for entry in entries]
    if any(Path(name).name != name for name in declared_names):
        raise ValueError(
            f"harvest manifest {manifest_path} has an invalid shard name.",
        )
    actual_names = sorted(path.name for path in harvest_dir.glob("shard-*.npz"))
    if actual_names != sorted(declared_names):
        raise ValueError(
            f"harvest manifest shard set mismatch: declared={declared_names}, "
            f"actual={actual_names}.",
        )
    if declared_names != actual_names:
        raise ValueError(
            "harvest manifest shards are not in canonical shard order: "
            f"declared={declared_names}, canonical={actual_names}.",
        )
    originals: list[Tensor] = []
    candidates: list[Tensor] = []
    flat_views: list[Tensor] = []
    groups: list[Tensor] = []
    row_count = 0
    for index, entry in enumerate(entries):
        shard = harvest_dir / entry["file"]
        if _sha256(shard) != entry["sha256"]:
            raise ValueError(f"harvest shard digest mismatch: {shard}.")
        with _npz(shard) as data:
            base_groups = np.array(data["base_group_id"], dtype=np.int64)
            if base_groups.size and (
                int(base_groups.min()) < 0 or int(base_groups.max()) >= groups_per_shard
            ):
                raise ValueError(
                    f"shard {shard.name}: base_group_id must lie in "
                    f"[0, {groups_per_shard}).",
                )
            originals.append(torch.from_numpy(np.array(data["original"])))
            candidates.append(torch.from_numpy(np.array(data["candidate"])))
            flat_views.append(torch.from_numpy(np.array(data["flat_view_id"])))
            groups.append(torch.from_numpy(base_groups) + groups_per_shard * index)
            rows = len(data["original"])
        if entry["rows"] != rows:
            raise ValueError(f"harvest shard row count mismatch: {shard}.")
        row_count += rows
    if manifest["shard_count"] != len(entries) or manifest["row_count"] != row_count:
        raise ValueError(f"harvest manifest aggregate count mismatch: {manifest_path}.")
    return (
        torch.cat(originals).to(device).long(),
        torch.cat(candidates).to(device).long(),
        torch.cat(flat_views).to(device).long(),
        torch.cat(groups).to(device),
    )


def _batch_partition_counts(
    batch_size: int,
    fractions: tuple[float, ...],
) -> tuple[int, ...]:
    """Round mixture sizes, trimming only pathological small-batch overflow."""
    counts = [round(batch_size * fraction) for fraction in fractions]
    overflow = max(0, sum(counts) - batch_size)
    for index in range(len(counts) - 1, -1, -1):
        removed = min(counts[index], overflow)
        counts[index] -= removed
        overflow -= removed
        if not overflow:
            break
    return tuple(counts)


def _choices(
    options: tuple[int, ...],
    count: int,
    *,
    generator: torch.Generator,
) -> Tensor:
    """Sample ``count`` values uniformly from ``options``."""
    table = torch.tensor(options, device=generator.device)
    idx = torch.randint(
        len(options),
        (count,),
        device=generator.device,
        generator=generator,
    )
    return table[idx]


# The replacement is a uniform nonzero offset in digit space, so every corrupted cell is
# guaranteed to change (a Hamming-k corruption has exactly k wrong cells, keeping the
# negative strata pure).
def _corrupt_hamming(
    candidate: Tensor,
    given: Tensor,
    *,
    k: Tensor,
    generator: torch.Generator,
) -> Tensor:
    """Replace k random non-given cells per row with DIFFERENT digit tokens."""
    if not len(candidate):
        return candidate
    b, n = candidate.shape
    noise = (
        torch.rand(b, n, device=candidate.device, generator=generator)
        + given.float() * 10
    )
    order = noise.argsort(dim=1)
    take = torch.arange(n, device=candidate.device).expand(b, n) < k.unsqueeze(1)
    chosen = torch.zeros_like(given)
    chosen.scatter_(1, order, take)
    offsets = torch.randint(
        1,
        9,
        (b, n),
        device=candidate.device,
        generator=generator,
        dtype=candidate.dtype,
    )
    digits = ((candidate - 2 + offsets) % 9) + 2
    return torch.where(chosen & ~given, digits, candidate)


def _swap_cells(
    candidate: Tensor,
    given: Tensor,
    *,
    pairs: Tensor,
    generator: torch.Generator,
) -> Tensor:
    """Swap the digits of random non-given cell pairs (histogram-preserving)."""
    if not len(candidate):
        return candidate
    b, n = candidate.shape
    max_pairs = int(pairs.max()) if len(pairs) else 0
    if max_pairs == 0:
        return candidate
    noise = (
        torch.rand(b, n, device=candidate.device, generator=generator)
        + given.float() * 10
    )
    order = noise.argsort(dim=1)
    first = order[:, 0 : 2 * max_pairs : 2]
    second = order[:, 1 : 2 * max_pairs : 2]
    active = torch.arange(max_pairs, device=candidate.device).expand(
        b,
        max_pairs,
    ) < pairs.unsqueeze(1)
    out = candidate.clone()
    a = out.gather(1, first)
    c = out.gather(1, second)
    out.scatter_(1, first, torch.where(active, c, a))
    out.scatter_(1, second, torch.where(active, a, c))
    return out


# Both transforms preserve Sudoku validity and clue consistency, so every label -- grid
# and per-cell -- is invariant.
def _augment_consistent(
    puzzle: Tensor,
    candidate: Tensor,
    solution: Tensor,
    *,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor, Tensor]:
    """One shared digit permutation + dihedral symmetry across the triple."""
    b = puzzle.shape[0]
    device = puzzle.device
    perms = torch.arange(11, device=device).expand(b, -1).clone()
    perms[:, 2:11] = (
        torch.rand(b, 9, device=device, generator=generator).argsort(dim=1) + 2
    )
    dihedral = _dihedral_indices(device)
    choice = torch.randint(0, 8, (b,), device=device, generator=generator)
    selected = dihedral[choice]

    def apply(tokens: Tensor) -> Tensor:
        permuted = torch.gather(perms, 1, tokens.long())
        return torch.gather(permuted, 1, selected)

    return apply(puzzle), apply(candidate), apply(solution)


def _dihedral_indices(device: torch.device) -> Tensor:
    """Return the 8 dihedral symmetries of the 9x9 grid as index permutations."""
    base = torch.arange(81, device=device).reshape(9, 9)
    symmetries: list[Tensor] = []
    for quarter_turns in range(4):
        rotated = torch.rot90(base, quarter_turns)
        symmetries.append(rotated.reshape(-1))
        symmetries.append(rotated.flip(1).reshape(-1))
    return torch.stack(symmetries)


def _atomic_torch_save(state: Mapping[str, object], path: Path) -> None:
    """Serialize with torch.save via tmp + rename (crash-safe)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    """Write JSON via tmp + rename (crash-safe)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Verifier-corpus harvest.
# ---------------------------------------------------------------------------


class HarvestSource(IntEnum):
    """Label-free candidate-state generation mechanisms (``source_kind``)."""

    ROOT = 0
    FIXED_HPS = 1
    RANDOM_UNSTUCK = 2


@dataclass(slots=True, frozen=True, kw_only=True)
class SelectedViews:
    """Flattened augmentation rows selected group-first (CPU int64 tensors).

    Attributes:
      flat_view_id: ``[V]`` row index of each selected view in the train file.
      base_group_id: ``[V]`` owning base puzzle group per view.
      view_id: ``[V]`` within-group augmentation offset per view.

    """

    flat_view_id: Tensor
    base_group_id: Tensor
    view_id: Tensor


@dataclass(slots=True, frozen=True, kw_only=True)
class RandomUnstuckStarts:
    """Deterministic label-free feedback starts and stable identifiers.

    Attributes:
      current_state: ``[N, G]`` corrupted feedback boards.
      puzzle_rows: ``[N]`` owning batch row per start.
      start_index: ``[N]`` start counter within its puzzle row (0-based,
        strengths-major, repeats innermost).
      start_seed: ``[N]`` exact generator seed of each start (replayable).
      corruption_strength: ``[N]`` requested blank-cell corruption count.

    """

    current_state: Tensor
    puzzle_rows: Tensor
    start_index: Tensor
    start_seed: Tensor
    corruption_strength: Tensor


def fixed_hps_node_count(*, depth: int, candidates: int, cell_attempts: int) -> int:
    """Return the complete fixed HPS tree size per puzzle.

    Args:
      depth: Maximum committed pins per path (``K``).
      candidates: Digit branches per selected cell (``C``).
      cell_attempts: Root entropy-cell restarts (``A``).

    Returns:
      node_count: ``A * sum(C**level for level in 1..K)``.

    Raises:
      ValueError: Any dimension is below 1.

    """
    if depth < 1 or candidates < 1 or cell_attempts < 1:
        raise ValueError("fixed HPS dimensions must all be positive.")
    # Accumulated rather than ``sum(candidates**level ...)``: ``int ** int`` is
    # typed ``Any | float`` by typeshed, and this stays exact integer math.
    total = 0
    level_nodes = 1
    for _ in range(depth):
        level_nodes *= candidates
        total += level_nodes
    return cell_attempts * total


def rows_per_view(
    *,
    search_depth: int,
    search_candidates: int,
    search_cell_attempts: int,
    random_corruption_strengths: tuple[int, ...],
    random_starts_per_strength: int,
) -> int:
    """Return the exact root, fixed-HPS, and random-start rows per view.

    Args:
      search_depth: Maximum committed pins per HPS path.
      search_candidates: Digit branches per selected cell.
      search_cell_attempts: Root entropy-cell restarts.
      random_corruption_strengths: Blank-cell corruption strengths.
      random_starts_per_strength: Independent starts per strength.

    Returns:
      row_count: ``1 + fixed_hps_node_count(...) + strengths * starts``
        (69 at the recipe defaults).

    Raises:
      ValueError: The random-start plan is empty.

    """
    if random_starts_per_strength < 1 or not random_corruption_strengths:
        raise ValueError("random-start strengths and repetitions must be nonempty.")
    return (
        1
        + fixed_hps_node_count(
            depth=search_depth,
            candidates=search_candidates,
            cell_attempts=search_cell_attempts,
        )
        + len(random_corruption_strengths) * random_starts_per_strength
    )


def select_harvest_views(
    group_bounds: Tensor,
    *,
    group_count: int,
    views_per_group: int,
    seed: int,
) -> SelectedViews:
    """Select deterministic augmentation views for base groups ``[0, count)``.

    Per base group ``g`` a dedicated CPU generator seeded
    ``seed + g * 1_000_003`` draws one ``randperm`` over the group's flattened
    augmentation rows; the first ``views_per_group`` entries are kept in draw
    order. The selection is group-major: group 0's views come first.

    Args:
      group_bounds: ``[n_groups + 1]`` instance boundary vector of the train
        file (``all__group_indices``).
      group_count: Base groups ``[0, group_count)`` to harvest.
      views_per_group: Views selected per base group.
      seed: Dedicated seed controlling only within-group view selection.

    Returns:
      selected: The :class:`SelectedViews` (``group_count * views_per_group``
        rows).

    Raises:
      ValueError: Bounds are malformed, the range exceeds the file, or a
        group holds fewer than ``views_per_group`` views.

    """
    if group_bounds.ndim != 1 or group_bounds.numel() < 2:
        raise ValueError("group_bounds must be a nonempty boundary vector.")
    if views_per_group < 1 or seed < 0:
        raise ValueError("views_per_group must be positive and seed non-negative.")
    total_groups = group_bounds.numel() - 1
    if group_count <= 0 or group_count > total_groups:
        raise ValueError(
            f"group_count {group_count} must lie in [1, {total_groups}] "
            "(the base groups on disk).",
        )
    flat_ids: list[int] = []
    group_ids: list[int] = []
    view_ids: list[int] = []
    for group in range(group_count):
        group_start = int(group_bounds[group])
        view_count = int(group_bounds[group + 1]) - group_start
        if view_count < views_per_group:
            raise ValueError(
                f"base group {group} has {view_count} views, fewer than "
                f"views_per_group={views_per_group}.",
            )
        generator = torch.Generator().manual_seed(seed + group * 1_000_003)
        selected = torch.randperm(view_count, generator=generator)[:views_per_group]
        for view in from_plain(selected.tolist(), list[int]):
            flat_ids.append(group_start + view)
            group_ids.append(group)
            view_ids.append(view)
    return SelectedViews(
        flat_view_id=torch.tensor(flat_ids, dtype=torch.int64),
        base_group_id=torch.tensor(group_ids, dtype=torch.int64),
        view_id=torch.tensor(view_ids, dtype=torch.int64),
    )


def make_random_unstuck_starts(
    originals: Tensor,
    root_candidates: Tensor,
    *,
    base_group_ids: Tensor,
    view_ids: Tensor,
    seed: int,
    corruption_strengths: tuple[int, ...],
    starts_per_strength: int,
) -> RandomUnstuckStarts:
    """Corrupt root feedback deterministically without consulting solutions.

    Per start, a fresh CPU generator seeded ``seed + group * 10_000_019 +
    view * 10_007 + start_index`` draws (in this order) ``randperm`` over the
    board's BLANK cells (original token 1) and ``randint(1, 9)`` digit
    offsets. The start state is the root candidate with the original givens
    re-clamped and ``min(strength, n_blanks)`` selected blank cells rotated to
    a guaranteed-different digit token. Strength therefore maps to "wrong
    filled-in cells", never to removed or added GIVENS -- givens are inviolate.

    Args:
      originals: ``[B, G]`` original puzzle boards (givens 2-10, blank 1).
      root_candidates: ``[B, G]`` root-rollout decoded grids.
      base_group_ids: ``[B]`` global base-group id per row (seed fold).
      view_ids: ``[B]`` within-group view offset per row (seed fold).
      seed: Dedicated random-start seed.
      corruption_strengths: Requested blank-cell corruption counts, in the
        declared (outer-loop) order.
      starts_per_strength: Independent starts per strength (inner loop).

    Returns:
      starts: The :class:`RandomUnstuckStarts`
        (``B * len(strengths) * starts_per_strength`` rows, row-major).

    Raises:
      ValueError: Shapes disagree, the plan is empty, or a row has no blanks.

    """
    if originals.shape != root_candidates.shape or originals.ndim != 2:
        raise ValueError("originals and root_candidates must be aligned grids.")
    if base_group_ids.shape != (originals.shape[0],) or view_ids.shape != (
        originals.shape[0],
    ):
        raise ValueError("base-group and view ids must align with puzzle rows.")
    if seed < 0 or starts_per_strength < 1:
        raise ValueError("random-start seed must be non-negative and count positive.")
    if not corruption_strengths or any(value < 1 for value in corruption_strengths):
        raise ValueError("corruption strengths must be positive and nonempty.")

    original_cpu = originals.detach().cpu().long()
    candidate_cpu = root_candidates.detach().cpu().long()
    group_cpu = base_group_ids.detach().cpu().long()
    view_cpu = view_ids.detach().cpu().long()
    states: list[Tensor] = []
    puzzle_rows: list[int] = []
    start_indices: list[int] = []
    start_seeds: list[int] = []
    strengths: list[int] = []
    for puzzle_row in range(original_cpu.shape[0]):
        original = original_cpu[puzzle_row]
        blank_cells = (original == 1).nonzero(as_tuple=True)[0]
        if not blank_cells.numel():
            raise ValueError(f"puzzle row {puzzle_row} has no blank feedback cells.")
        start_index = 0
        for strength in corruption_strengths:
            for _ in range(starts_per_strength):
                start_seed = (
                    seed
                    + int(group_cpu[puzzle_row]) * 10_000_019
                    + int(view_cpu[puzzle_row]) * 10_007
                    + start_index
                )
                generator = torch.Generator().manual_seed(start_seed)
                cell_count = min(strength, blank_cells.numel())
                cells = blank_cells[
                    torch.randperm(blank_cells.numel(), generator=generator)[
                        :cell_count
                    ]
                ]
                state = candidate_cpu[puzzle_row].clone()
                state[original >= 2] = original[original >= 2]
                old_tokens = state[cells]
                base_tokens = torch.where(
                    (old_tokens >= 2) & (old_tokens <= 10),
                    old_tokens,
                    torch.full_like(old_tokens, 2),
                )
                offsets = torch.randint(1, 9, (cell_count,), generator=generator)
                state[cells] = (base_tokens - 2 + offsets) % 9 + 2
                states.append(state)
                puzzle_rows.append(puzzle_row)
                start_indices.append(start_index)
                start_seeds.append(start_seed)
                strengths.append(strength)
                start_index += 1
    device = originals.device
    return RandomUnstuckStarts(
        current_state=torch.stack(states).to(device=device, dtype=originals.dtype),
        puzzle_rows=torch.tensor(puzzle_rows, device=device, dtype=torch.int64),
        start_index=torch.tensor(start_indices, device=device, dtype=torch.int64),
        start_seed=torch.tensor(start_seeds, device=device, dtype=torch.int64),
        corruption_strength=torch.tensor(strengths, device=device, dtype=torch.int64),
    )


def checkpoint_rollout_from_state(  # noqa: PLR0917 -- The public rollout protocol requires these positional state components.
    model: TRM,
    step_kwargs: dict[str, Tensor],
    max_steps: int,
    input_ids: Tensor,
    initial_feedback: Tensor,
    rows: Tensor,
    *,
    checkpoints: tuple[int, ...],
) -> LearnedCheckpointRollout:
    """Run fixed ACT checkpoints from an explicit input and feedback state.

    Generalizes
    :func:`~learned_checkpoint_rollout_rows`
    with a separate initial feedback grid -- the replay unit of every harvest
    row: roots and HPS nodes pass their board as both arguments; unstuck
    starts pass the ORIGINAL board with the corrupted state as feedback.
    Given cells always override the feedback.

    Args:
      model: The TRM (eval mode, weights already swapped).
      step_kwargs: Full-batch model kwargs, gathered here by ``rows``.
      max_steps: Number of ACT steps to run for every row.
      input_ids: ``[N, G]`` input token boards.
      initial_feedback: ``[N, G]`` first-step feedback grids.
      rows: ``[N]`` original batch row per rollout row.
      checkpoints: Strictly increasing one-indexed ACT steps to retain.

    Returns:
      rollout: Final logits and checkpoint-aligned q scores and grids.

    Raises:
      ValueError: Shapes disagree or checkpoints are invalid.

    """
    if input_ids.shape != initial_feedback.shape or input_ids.ndim != 2:
        raise ValueError("input ids and initial feedback must be aligned grids.")
    if rows.shape != (input_ids.shape[0],):
        raise ValueError("rows must name one source puzzle per rollout row.")
    if (
        not checkpoints
        or tuple(sorted(set(checkpoints))) != checkpoints
        or checkpoints[0] < 1
        or checkpoints[-1] > max_steps
    ):
        raise ValueError(f"checkpoints {checkpoints} must lie in 1..{max_steps}.")
    puzzle_identifiers = step_kwargs.get("puzzle_identifiers")
    if puzzle_identifiers is not None:
        puzzle_identifiers = puzzle_identifiers[rows]
    z_slow, z_fast = model.init_z(input_ids.shape[0])
    given = (input_ids >= 2) & (input_ids <= 10)
    feedback = torch.where(given, input_ids, initial_feedback)
    q_scores: list[Tensor] = []
    predictions: list[Tensor] = []
    out: dict[str, Tensor] = {}
    checkpoint_index = 0
    for step in range(1, max_steps + 1):
        out = model.act_step(
            input_ids,
            z_slow,
            z_fast,
            puzzle_identifiers=puzzle_identifiers,
            feedback_ids=feedback,
        )
        z_slow = out["z_slow"]
        z_fast = out["z_fast"]
        decoded = out["logits"].argmax(dim=-1)
        feedback = torch.where(given, input_ids, decoded)
        if step == checkpoints[checkpoint_index]:
            q_scores.append(out["q_halt"].float().clone())
            predictions.append(decoded.clone())
            checkpoint_index += 1
            if checkpoint_index == len(checkpoints):
                checkpoint_index = -1
    return LearnedCheckpointRollout(
        logits=out["logits"],
        q_scores=torch.stack(q_scores, dim=1),
        predictions=torch.stack(predictions, dim=1),
    )


class Harvest:
    """The verifier-corpus harvest job (shards + manifest + node-dump regen).

    Construction resolves paths, loads the train split, selects the
    deterministic views, and loads the frozen generator; :meth:`run` writes
    the shards and the manifest. Shard bytes are deterministic: same config
    and checkpoint produce byte-identical files (fixed zip timestamps, no
    ambient RNG), so the manifest's sha256 provenance is reproducible.
    """

    class Config(Fig["Harvest"]):
        """One harvest corpus: source checkpoint, view plan, row generators."""

        harvest_source_checkpoint: str | Path = ""
        """Frozen generator checkpoint the harvest rolls out. Both package
        schemas load: the full trainer checkpoint (``state["step"]["model"]``
        plus the flat ``state["step"]["ema"]`` shadow -- the EMA weights are
        the eval weights and take precedence when present) and the model-only
        flavor (``{"model": ...}`` at the top level). A relative logical path
        resolves beneath ``base_dir`` at construction."""

        model: TRM.Config = field(default_factory=TRM.Config)
        """Generator architecture (default: the recipe TRM); must match the
        checkpoint."""

        base_dir: Path | str | None = None
        """Scratch root; ``None`` resolves at construction beneath
        ``/opt/scratch``."""

        working_dir: Path | str = "/datasets/sudoku-extreme"
        """Sudoku dataset root with train/ and test/ splits; resolved beneath
        ``base_dir``."""

        experiment_name: str = "harvest"
        """Run identity; fills ``{experiment_name}`` in ``run_dir``."""

        run_dir: str | Path = "/runs/{experiment_name}/harvest"
        """Shard/manifest output directory (also the default node-corpus
        home) resolved beneath ``base_dir``. Existing shards or manifest are
        never overwritten."""

        group_count: int = 800
        """Base puzzle groups ``[0, group_count)`` harvested; shards hold 8
        consecutive groups each (the loader contract; the last shard may hold
        fewer when ``group_count`` is not a multiple of 8)."""

        views_per_group: int = 2
        """Deterministic augmentation views selected per base group."""

        seed: int = 0
        """Dedicated seed folded into view selection and unstuck starts (the
        module docstring documents the exact draw orders)."""

        max_act_steps: int = 32
        """Full ACT rollout depth for every harvest row."""

        checkpoints: tuple[int, ...] = (1, 2, 4, 8, 16, 24, 32)
        """One-indexed ACT steps whose q values are retained per row; the
        candidate grid is the final checkpoint's decoded grid."""

        search_depth: int = 2
        """Maximum committed pins in each exhaustive HPS path."""

        search_candidates: int = 5
        """Confidence-ordered token branches expanded per HPS node."""

        search_cell_attempts: int = 2
        """Independent root entropy-cell HPS trees per puzzle."""

        search_budget: int = 60
        """Per-puzzle node cap; must cover the complete declared HPS tree
        (the harvest census is exhaustive, never truncated)."""

        search_max_rows: int = 2_048
        """Maximum candidate rows in one frozen-model rollout call."""

        random_corruption_strengths: tuple[int, ...] = (4, 12, 24, 40)
        """Requested blank-cell corruptions for random feedback starts."""

        random_starts_per_strength: int = 2
        """Independent deterministic starts drawn at each corruption
        strength."""

        dtype_autocast: torch.dtype | None = torch.bfloat16
        """Autocast compute dtype for the rollouts (mirrors the trainer);
        None disables autocast."""

        emulate_precision_casts: bool = True
        """Set TorchInductor's precision-cast emulation (process-global)
        before generator model construction -- required for sound compiled
        bf16 rollouts (the trainer contract)."""

        device: torch.device | str | None = None
        """Rollout device."""

    def __init__(self, config: Config) -> None:
        if not str(config.harvest_source_checkpoint):
            raise ValueError(
                "harvest_source_checkpoint must name a generator checkpoint.",
            )
        if config.group_count < 1 or config.views_per_group < 1 or config.seed < 0:
            raise ValueError(
                "group_count and views_per_group must be positive and seed "
                "non-negative.",
            )
        if config.search_max_rows < 1:
            raise ValueError("search_max_rows must be positive.")
        if (
            not config.checkpoints
            or tuple(sorted(set(config.checkpoints))) != config.checkpoints
            or config.checkpoints[0] < 1
            or config.checkpoints[-1] > config.max_act_steps
        ):
            raise ValueError(
                f"checkpoints {config.checkpoints} must be strictly increasing "
                f"within 1..{config.max_act_steps}.",
            )
        expected_hps = fixed_hps_node_count(
            depth=config.search_depth,
            candidates=config.search_candidates,
            cell_attempts=config.search_cell_attempts,
        )
        if config.search_budget < expected_hps:
            raise ValueError(
                f"search_budget={config.search_budget} cannot exhaust "
                f"{expected_hps} HPS nodes.",
            )
        self.config = config
        # Validates the random-start plan too.
        self._rows_per_view = rows_per_view(
            search_depth=config.search_depth,
            search_candidates=config.search_candidates,
            search_cell_attempts=config.search_cell_attempts,
            random_corruption_strengths=config.random_corruption_strengths,
            random_starts_per_strength=config.random_starts_per_strength,
        )
        self.device = get_device(config.device)

        base = config.base_dir if config.base_dir is not None else Path("/opt/scratch")
        self.dataset_dir = Path(resolve_working_dir(base, config.working_dir))
        self.run_dir = resolve_working_dir(
            base,
            str(config.run_dir).format(experiment_name=config.experiment_name),
        )
        self.checkpoint_path = (
            config.harvest_source_checkpoint
            if isinstance(config.harvest_source_checkpoint, Path)
            else resolve_working_dir(base, config.harvest_source_checkpoint)
        )

        train = load_puzzle_dataset(self.dataset_dir, "train")
        self._train_inputs: Tensor = train["inputs"]
        self._train_labels: Tensor = train["labels"]
        self._selected = select_harvest_views(
            train["group_indices"].long(),
            group_count=config.group_count,
            views_per_group=config.views_per_group,
            seed=config.seed,
        )

        # Process-global TorchInductor flag; must be set before model
        # construction binds torch.compile (the trainer contract; compiled
        # bf16 rollouts are numerically unsound without it).
        inductor_config.emulate_precision_casts = config.emulate_precision_casts
        self.model = config.model.make().to(self.device)
        self.model.load_state_dict(
            load_eval_weights(self.checkpoint_path, device=self.device),
        )
        self.model.eval()
        self.model.requires_grad_(False)

    def run(self) -> Path:
        """Harvest every selected view and publish the corpus.

        Writes ``shard-<i>.npz`` (8 base groups each, atomic) and then
        ``manifest.json`` with exact row/shard accounting. A shard left by an
        interrupted run is reused once it matches the deterministic output.

        Returns:
          out_dir: The harvest directory holding the shards and manifest
            (what ``VerifierData.Config.harvest_dir`` consumes).

        Raises:
          FileExistsError: The manifest already exists.
          ValueError: An existing shard differs from the deterministic output.

        """
        cfg = self.config
        self.run_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = self.run_dir / "manifest.json"
        if manifest_path.exists():
            raise FileExistsError(
                f"refusing to overwrite harvest manifest {manifest_path}.",
            )
        shard_summaries: list[dict[str, object]] = []
        row_count = 0
        solved_count = 0
        high_q_wrong_count = 0
        source_counts: dict[int, int] = {}
        group_starts: list[int] = list(range(0, cfg.group_count, 8))
        for shard_index, group_lo in enumerate(
            cast("Iterable[int]", tqdm(group_starts, desc="harvest", unit="shard")),
        ):
            view_lo = group_lo * cfg.views_per_group
            view_hi = min(group_lo + 8, cfg.group_count) * cfg.views_per_group
            arrays = self._shard_arrays(
                flat_ids=self._selected.flat_view_id[view_lo:view_hi],
                group_ids=self._selected.base_group_id[view_lo:view_hi],
                view_ids=self._selected.view_id[view_lo:view_hi],
                first_group=group_lo,
            )
            path = self.run_dir / f"shard-{shard_index:05d}.npz"
            if path.exists():
                with _npz(path) as existing_file:
                    existing = {
                        name: np.array(existing_file[name])
                        for name in existing_file.files
                    }
                if set(existing) != set(arrays) or any(
                    not np.array_equal(existing[name], value)
                    for name, value in arrays.items()
                ):
                    raise ValueError(
                        f"existing harvest shard {path} does not match the "
                        "deterministic resumed output.",
                    )
                arrays = existing
            else:
                _write_npz(path, arrays)
            rows = len(arrays["original"])
            shard_summaries.append(
                {"file": path.name, "rows": rows, "sha256": _sha256(path)},
            )
            row_count += rows
            exact = cast("NDArray[np.bool_]", arrays["exact_solved"])
            q_halt = cast("NDArray[np.float32]", arrays["q_halt"])
            solved_count += int(np.count_nonzero(exact))
            high_q_wrong_count += int(np.count_nonzero((q_halt[:, -1] >= 0) & ~exact))
            kinds, counts = cast(
                "tuple[NDArray[np.int64], NDArray[np.int64]]",
                np.unique(arrays["source_kind"], return_counts=True),
            )
            for kind, count in zip(
                from_plain(cast(object, kinds.tolist()), list[int]),
                from_plain(cast(object, counts.tolist()), list[int]),
                strict=True,
            ):
                source_counts[kind] = source_counts.get(kind, 0) + count

        manifest = {
            "schema_version": 1,
            "source_checkpoint": str(self.checkpoint_path),
            "checkpoints": list(cfg.checkpoints),
            "group_count": cfg.group_count,
            "views_per_group": cfg.views_per_group,
            "groups_per_shard": 8,
            "rows_per_view": self._rows_per_view,
            "seed": cfg.seed,
            "row_count": row_count,
            "shard_count": len(shard_summaries),
            "solved_count": solved_count,
            "high_q_wrong_count": high_q_wrong_count,
            "source_counts": {
                str(key): value for key, value in sorted(source_counts.items())
            },
            "replay_semantics": {
                str(int(HarvestSource.ROOT)): (
                    "input=original; initial_feedback=original"
                ),
                str(int(HarvestSource.FIXED_HPS)): (
                    "input=current_state; initial_feedback=current_state"
                ),
                str(int(HarvestSource.RANDOM_UNSTUCK)): (
                    "input=original; initial_feedback=current_state"
                ),
            },
            "shards": shard_summaries,
        }
        temporary = manifest_path.with_name(f".{manifest_path.name}.tmp.{os.getpid()}")
        temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        temporary.replace(manifest_path)
        logger.info(
            "harvest %s: %d rows in %d shards -> %s",
            cfg.experiment_name,
            row_count,
            len(shard_summaries),
            self.run_dir,
        )
        return self.run_dir

    def regenerate_node_corpus(
        self,
        *,
        dev_puzzles: int = 5_000,
        batch_size: int = 512,
        root_reject_threshold: float = 0.0,
        search_depth: int = 3,
        search_candidates: int = 5,
        search_cell_attempts: int = 2,
        search_budget: int = 512,
        out_path: str | Path | None = None,
    ) -> Path:
        """Rebuild the OPTIONAL dev search-node dump for verifier eval stratum 2.

        Runs the deterministic root rollout over the first ``dev_puzzles``
        test puzzles, then exhausts the fixed HPS frontier (no acceptance
        pruning) for every root whose FINAL checkpoint q falls below
        ``root_reject_threshold``, and dumps one row per node. The defaults
        reproduce the internal census parameters (dev-5K, C=5/K=3/A=2,
        budget 512, threshold 0). Rows sort by (``global_index``,
        ``node_index``); an existing file is replaced (the dump is a derived
        artifact, unlike the shards).

        Args:
          dev_puzzles: Test-split prefix size (file order).
          batch_size: Puzzles per rollout batch.
          root_reject_threshold: Roots with final-checkpoint q below this
            enter the node frontier.
          search_depth: Maximum pins per traced path.
          search_candidates: Digit branches per selected cell.
          search_cell_attempts: Root entropy-cell restarts.
          search_budget: Maximum traced nodes per puzzle (may truncate).
          out_path: Destination ``.npz``; None writes
            ``out_dir/node_corpus.npz`` (where
            ``VerifierData.Config.node_corpus`` points by convention).

        Returns:
          path: The written dump with fields ``media`` / ``final_prediction``
            (``[N, G]`` uint8), ``global_index`` (``[N]`` int64 dev puzzle
            index), ``node_index`` (``[N]`` int32) and ``checkpoints``.

        """
        cfg = self.config
        test = load_puzzle_dataset(self.dataset_dir, "test", max_samples=dev_puzzles)
        inputs = test["inputs"]
        global_parts: list[Tensor] = []
        node_parts: list[Tensor] = []
        media_parts: list[Tensor] = []
        prediction_parts: list[Tensor] = []
        with torch.inference_mode(), self._autocast():
            for lo in cast(
                "Iterable[int]",
                tqdm(
                    range(0, inputs.shape[0], batch_size),
                    desc="node corpus",
                    unit="batch",
                ),
            ):
                originals = inputs[lo : lo + batch_size].to(self.device)
                batch = originals.shape[0]
                step_kwargs: dict[str, Tensor] = {}
                if self.model.puzzle_emb is not None:
                    step_kwargs["puzzle_identifiers"] = torch.zeros(
                        batch,
                        dtype=torch.int32,
                        device=self.device,
                    )
                rollout = partial(
                    checkpoint_rollout_from_state,
                    self.model,
                    step_kwargs,
                    cfg.max_act_steps,
                    checkpoints=cfg.checkpoints,
                )
                root = rollout(
                    originals,
                    originals,
                    torch.arange(batch, device=self.device),
                )
                active = root.q_scores[:, -1] < root_reject_threshold
                trace = _exhaustive_pin_trace(
                    partial(_board_rollout, rollout),
                    media=originals,
                    base_logits=root.logits,
                    active=active,
                    depth=search_depth,
                    candidates=search_candidates,
                    cell_attempts=search_cell_attempts,
                    budget=search_budget,
                    max_rows=cfg.search_max_rows,
                    checkpoint_count=len(cfg.checkpoints),
                )
                global_parts.append((trace.puzzle_rows + lo).cpu())
                node_parts.append(trace.node_index.cpu())
                media_parts.append(originals[trace.puzzle_rows].cpu())
                prediction_parts.append(trace.predictions[:, -1].cpu())
        global_index = torch.cat(global_parts).numpy().astype(np.int64)
        node_index = torch.cat(node_parts).numpy().astype(np.int32)
        media = torch.cat(media_parts).numpy().astype(np.uint8)
        final_prediction = torch.cat(prediction_parts).numpy().astype(np.uint8)
        order = np.lexsort((node_index, global_index))
        path = (
            Path(out_path) if out_path is not None else self.run_dir / "node_corpus.npz"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_npz(
            path,
            {
                "global_index": global_index[order],
                "node_index": node_index[order],
                "media": media[order],
                "final_prediction": final_prediction[order],
                "checkpoints": np.asarray(self.config.checkpoints, dtype=np.int16),
            },
        )
        logger.info("node corpus: %d rows -> %s", len(global_index), path)
        return path

    def _shard_arrays(
        self,
        *,
        flat_ids: Tensor,
        group_ids: Tensor,
        view_ids: Tensor,
        first_group: int,
    ) -> dict[str, np.ndarray]:
        """Generate one shard's rows (root + fixed HPS + unstuck starts)."""
        cfg = self.config
        device = self.device
        originals = self._train_inputs[flat_ids].to(device)
        batch = originals.shape[0]
        step_kwargs: dict[str, Tensor] = {}
        if self.model.puzzle_emb is not None:
            step_kwargs["puzzle_identifiers"] = torch.zeros(
                batch,
                dtype=torch.int32,
                device=device,
            )
        hps_rows = fixed_hps_node_count(
            depth=cfg.search_depth,
            candidates=cfg.search_candidates,
            cell_attempts=cfg.search_cell_attempts,
        )
        with torch.inference_mode(), self._autocast():
            rollout = partial(
                checkpoint_rollout_from_state,
                self.model,
                step_kwargs,
                cfg.max_act_steps,
                checkpoints=cfg.checkpoints,
            )
            root = rollout(originals, originals, torch.arange(batch, device=device))
            trace = _exhaustive_pin_trace(
                partial(_board_rollout, rollout),
                media=originals,
                base_logits=root.logits,
                active=torch.ones(batch, dtype=torch.bool, device=device),
                depth=cfg.search_depth,
                candidates=cfg.search_candidates,
                cell_attempts=cfg.search_cell_attempts,
                budget=cfg.search_budget,
                max_rows=cfg.search_max_rows,
                checkpoint_count=len(cfg.checkpoints),
            )
            starts = make_random_unstuck_starts(
                originals,
                root.predictions[:, -1],
                base_group_ids=group_ids.to(device),
                view_ids=view_ids.to(device),
                seed=cfg.seed,
                corruption_strengths=cfg.random_corruption_strengths,
                starts_per_strength=cfg.random_starts_per_strength,
            )
            random_parts = [
                rollout(
                    originals[starts.puzzle_rows[lo : lo + cfg.search_max_rows]],
                    starts.current_state[lo : lo + cfg.search_max_rows],
                    starts.puzzle_rows[lo : lo + cfg.search_max_rows],
                )
                for lo in range(0, starts.puzzle_rows.shape[0], cfg.search_max_rows)
            ]

        root_rows = torch.arange(batch, device=device)
        puzzle_rows = torch.cat([root_rows, trace.puzzle_rows, starts.puzzle_rows])
        candidate_index = torch.cat(
            [
                torch.zeros_like(root_rows),
                trace.node_index + 1,
                starts.start_index + hps_rows + 1,
            ],
        )
        source_kind = torch.cat(
            [
                torch.full_like(root_rows, int(HarvestSource.ROOT)),
                torch.full_like(trace.puzzle_rows, int(HarvestSource.FIXED_HPS)),
                torch.full_like(
                    starts.puzzle_rows,
                    int(HarvestSource.RANDOM_UNSTUCK),
                ),
            ],
        )
        start_seed = torch.cat(
            [
                torch.zeros_like(root_rows),
                torch.zeros_like(trace.puzzle_rows),
                starts.start_seed,
            ],
        )
        corruption_strength = torch.cat(
            [
                torch.zeros_like(root_rows),
                torch.zeros_like(trace.puzzle_rows),
                starts.corruption_strength,
            ],
        )
        q_halt = torch.cat(
            [root.q_scores, trace.q_scores] + [part.q_scores for part in random_parts],
        ).float()
        current_state = torch.cat([originals, trace.boards, starts.current_state])
        candidate = torch.cat(
            [root.predictions[:, -1], trace.predictions[:, -1]]
            + [part.predictions[:, -1] for part in random_parts],
        )
        if puzzle_rows.numel() != batch * self._rows_per_view:
            raise RuntimeError("harvest did not emit the declared rows per view.")
        order = torch.argsort(
            puzzle_rows * self._rows_per_view + candidate_index,
            stable=True,
        ).cpu()

        puzzle_cpu = puzzle_rows.cpu()[order]
        original = originals.cpu().long()[puzzle_cpu]
        current = current_state.cpu().long()[order]
        candidate_cpu = candidate.cpu().long()[order]
        labels = self._train_labels[flat_ids].long()[puzzle_cpu]
        candidate_index = candidate_index.cpu()[order]
        q_halt = q_halt.cpu()[order]
        valid = labels != -100
        valid_cells = valid.sum(dim=-1)
        exact = (((candidate_cpu == labels) & valid).sum(dim=-1) == valid_cells) & (
            valid_cells > 0
        )
        return {
            "original": original.numpy().astype(np.uint8),
            "candidate": candidate_cpu.numpy().astype(np.uint8),
            "flat_view_id": flat_ids[puzzle_cpu].numpy().astype(np.int32),
            "base_group_id": (group_ids[puzzle_cpu] - first_group)
            .numpy()
            .astype(np.int16),
            "view_id": view_ids[puzzle_cpu].numpy().astype(np.int16),
            "candidate_index": candidate_index.numpy().astype(np.int16),
            "source_kind": source_kind.cpu()[order].numpy().astype(np.uint8),
            "start_seed": start_seed.cpu()[order].numpy().astype(np.int64),
            "corruption_strength": corruption_strength.cpu()[order]
            .numpy()
            .astype(np.uint8),
            "q_halt": q_halt.numpy().astype(np.float32),
            "current_state": current.numpy().astype(np.uint8),
            "checkpoints": np.asarray(cfg.checkpoints, dtype=np.int16),
            "exact_solved": exact.numpy().astype(np.bool_),
        }

    def _autocast(self) -> torch.amp.autocast:
        return torch.amp.autocast(
            device_type=self.device.type,
            dtype=self.config.dtype_autocast,
            enabled=self.config.dtype_autocast is not None,
            cache_enabled=False,
        )


@dataclass(slots=True, frozen=True, kw_only=True)
class _PinTrace:
    """Every bounded fixed-HPS node in deterministic visitation order."""

    puzzle_rows: Tensor
    node_index: Tensor
    boards: Tensor
    q_scores: Tensor
    predictions: Tensor


# The fixed node census behind ``source_kind`` 1 and the node corpus: the same tree the
# search engine walks (same :func:`~select_pin_candidates` cell and digit choices), but
# every node within ``budget`` is visited and recorded -- no early acceptance, no
# learned-score pruning. ``node_index`` counts a puzzle's nodes in visitation order
# (attempt-major, breadth-first).
def _exhaustive_pin_trace(
    rollout: Callable[[Tensor, Tensor], LearnedCheckpointRollout],
    *,
    media: Tensor,
    base_logits: Tensor,
    active: Tensor,
    depth: int,
    candidates: int,
    cell_attempts: int,
    budget: int,
    max_rows: int,
    checkpoint_count: int,
) -> _PinTrace:
    """Exhaust deterministic HPS nodes without any acceptance pruning."""
    batch_size, grid_len = media.shape
    device = media.device
    cells_root, digits_root = select_pin_candidates(
        base_logits,
        media,
        n_cells=cell_attempts,
        n_digits=candidates,
    )
    nodes = torch.zeros(batch_size, dtype=torch.int64, device=device)
    puzzle_parts: list[Tensor] = []
    node_parts: list[Tensor] = []
    board_parts: list[Tensor] = []
    q_parts: list[Tensor] = []
    prediction_parts: list[Tensor] = []

    for attempt in range(cell_attempts):
        eligible = active & (nodes + candidates <= budget)
        puzzles = eligible.nonzero(as_tuple=True)[0]
        if not puzzles.shape[0]:
            break
        frontier_puzzle = puzzles.repeat_interleave(candidates)
        boards = media[puzzles].clone().repeat_interleave(candidates, dim=0)
        root_cell = cells_root[puzzles, attempt].repeat_interleave(candidates)
        root_token = digits_root[puzzles, attempt].reshape(-1)
        local_rows = torch.arange(boards.shape[0], device=device)
        boards[local_rows, root_cell] = root_token.to(boards.dtype)

        for level in range(1, depth + 1):
            rollout_parts = [
                rollout(
                    boards[lo : lo + max_rows],
                    frontier_puzzle[lo : lo + max_rows],
                )
                for lo in range(0, boards.shape[0], max_rows)
            ]
            logits = torch.cat([part.logits for part in rollout_parts])
            q_scores = torch.cat([part.q_scores for part in rollout_parts])
            predictions = torch.cat([part.predictions for part in rollout_parts])
            if q_scores.shape != (boards.shape[0], checkpoint_count):
                raise ValueError(
                    "Expected q_scores.shape == (boards.shape[0], checkpoint_count).",
                )

            # Frontier blocks are contiguous and uniformly sized per puzzle
            # (expansion is all-or-nothing per puzzle), so the within-level
            # node offset is a plain remainder.
            rows_per_puzzle = candidates**level
            within_level = torch.arange(boards.shape[0], device=device).remainder(
                rows_per_puzzle,
            )
            puzzle_parts.append(frontier_puzzle.clone())
            node_parts.append(nodes[frontier_puzzle] + within_level)
            board_parts.append(boards.clone())
            q_parts.append(q_scores.float())
            prediction_parts.append(predictions)
            nodes.scatter_add_(0, frontier_puzzle, torch.ones_like(frontier_puzzle))

            if level == depth:
                break
            next_width = candidates ** (level + 1)
            expandable = nodes + next_width <= budget
            keep = expandable[frontier_puzzle].nonzero(as_tuple=True)[0]
            if not keep.shape[0]:
                break
            parent_boards = boards[keep]
            child_cell, child_tokens = select_pin_candidates(
                logits[keep],
                parent_boards,
                n_cells=1,
                n_digits=candidates,
            )
            boards = parent_boards.repeat_interleave(candidates, dim=0)
            frontier_puzzle = frontier_puzzle[keep].repeat_interleave(candidates)
            pinned_cell = child_cell[:, 0].repeat_interleave(candidates)
            pinned_token = child_tokens[:, 0].reshape(-1)
            local_rows = torch.arange(boards.shape[0], device=device)
            boards[local_rows, pinned_cell] = pinned_token.to(boards.dtype)

    if not puzzle_parts:
        return _PinTrace(
            puzzle_rows=torch.empty(0, device=device, dtype=torch.int64),
            node_index=torch.empty(0, device=device, dtype=torch.int64),
            boards=torch.empty((0, grid_len), device=device, dtype=media.dtype),
            q_scores=torch.empty((0, checkpoint_count), device=device),
            predictions=torch.empty(
                (0, checkpoint_count, grid_len),
                device=device,
                dtype=torch.int64,
            ),
        )
    return _PinTrace(
        puzzle_rows=torch.cat(puzzle_parts),
        node_index=torch.cat(node_parts),
        boards=torch.cat(board_parts),
        q_scores=torch.cat(q_parts),
        predictions=torch.cat(prediction_parts),
    )


def _board_rollout(
    rollout: Callable[[Tensor, Tensor, Tensor], LearnedCheckpointRollout],
    boards: Tensor,
    rows: Tensor,
) -> LearnedCheckpointRollout:
    """Bind a from-state rollout to boards-as-feedback (roots and HPS nodes)."""
    return rollout(boards, boards, rows)


def _npz(path: Path) -> NpzFile:
    """Open an npz archive; ``np.load`` itself returns ``Any``."""
    return cast("NpzFile", np.load(path))


# ``np.savez_compressed`` stamps wall-clock zip entry times, so identical arrays produce
# different bytes across runs; a fixed epoch keeps shard bytes (and the manifest's
# sha256 provenance) reproducible. Written via temp file + atomic rename, so a present
# file is complete.
def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write a compressed npz with fixed zip timestamps (byte-deterministic)."""
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, value in arrays.items():
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            with archive.open(info, "w", force_zip64=True) as stream:
                np.save(stream, np.asanyarray(value), allow_pickle=False)
    temporary.replace(path)


def _sha256(path: Path) -> str:
    """Hash one completed immutable shard."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(partial(file.read, 1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Eval engines: HPS full-set eval, agreement-lock committee, verifier sieve.
# ---------------------------------------------------------------------------


class Member(NamedTuple):
    """One committee member: a checkpoint reference under a recorded view.

    Attributes:
      checkpoint: Checkpoint path; a relative logical path is resolved beneath
        the owner's ``base_dir`` at run time. Either checkpoint schema loads
        (module docstring).
      view: The recorded exact symmetry the member evaluates under.
      name: Metrics label; empty falls back to the view name.

    """

    checkpoint: str
    view: View
    name: str = ""


def nine_view_members(checkpoint: str | Path) -> tuple[Member, ...]:
    """Build the frozen nine-view committee over one checkpoint, in tie-break order.

    Args:
      checkpoint: Checkpoint path shared by every pseudo-member.

    Returns:
      members: One :class:`Member` per :data:`NINE_VIEWS` view, frozen order.

    """
    return tuple(Member(str(checkpoint), view) for view in NINE_VIEWS)


def seed_ensemble_members(
    seed_checkpoints: Sequence[str | Path],
) -> tuple[Member, ...]:
    """Build the frozen seed-ensemble committee: plain seeds, then anchor views.

    Members are the seed checkpoints in training order under the canonical
    view (named after their run directory), then the
    :data:`SEED_ENSEMBLE_VIEWS` re-presentations of the FIRST checkpoint
    (the anchor), in the frozen member (tie-break) order.

    Args:
      seed_checkpoints: Checkpoint path per training seed, in order; the
        first is the tie-break anchor.

    Returns:
      members: The plain seed members followed by the anchor's view members.

    """
    if not seed_checkpoints:
        raise ValueError("the seed ensemble needs at least one checkpoint.")
    plain = tuple(
        Member(str(path), NINE_VIEWS[0], name=Path(str(path)).parents[1].name)
        for path in seed_checkpoints
    )
    anchor = str(seed_checkpoints[0])
    return plain + tuple(Member(anchor, view) for view in SEED_ENSEMBLE_VIEWS)


def load_eval_weights(
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> dict[str, Tensor]:
    """Load eval-ready model weights from either checkpoint schema.

    The shared eval-side loader. Accepts BOTH schemas:

    * full trainer checkpoints -- ``state["step"]["model"]`` plus, when the
      run kept one, the flat name-keyed EMA shadow ``state["step"]["ema"]``;
    * HF model-only files -- top-level ``state["model"]``.

    Eval weights are ALWAYS the EMA weights on the recipe (EMA warmup 0):
    when a full checkpoint carries a shadow, the shadow OVERLAYS the live
    parameters -- exactly the trainer's eval-time ``EMA.apply_to`` swap
    (buffers are never shadowed and come from the model state). Model-only
    HF files are exported EMA-applied already, so they load as-is.

    Args:
      path: Checkpoint file in either schema.
      device: ``map_location`` for the loaded tensors.

    Returns:
      weights: Name-keyed state dict ready for ``TRM.load_state_dict``.

    """
    step = _checkpoint_payload(Path(path), device)
    weights = dict(step["model"])
    weights.update(step.get("ema", {}))  # Eval weights are the EMA shadow.
    return weights


class _Weights(TypedDict):
    """One checkpoint's model payload, in either schema."""

    model: dict[str, Tensor]
    ema: NotRequired[dict[str, Tensor]]


def _checkpoint_payload(path: Path, device: torch.device | str) -> _Weights:
    """Load a checkpoint; return its nested full-state or top-level payload."""
    state = from_plain(
        cast(object, torch.load(path, map_location=device, weights_only=True)),
        dict[str, object],
    )
    nested = state.get("step")
    payload = from_plain(nested, dict[str, object]) if nested is not None else state
    weights: _Weights = {"model": from_plain(payload["model"], dict[str, Tensor])}
    if "ema" in payload:
        weights["ema"] = from_plain(payload["ema"], dict[str, Tensor])
    return weights


class HpsEval:
    """Full-set learned-q HPS evaluation of one frozen generator checkpoint.

    The canonical single-model search protocol: a deterministic root rollout
    per puzzle, learned-q acceptance (q >= 7.875 at ACT 24, 28 and 32), and
    the fast pin search ``hps(5, 3, 2, 512)`` on rejected rows -- label-free
    end to end. One member pass of the lock engines under the canonical
    view (the shared :func:`_search_pass` engine), plus the label-joined
    ``solution_visited`` diagnostic. Writes the packed prediction dump (the
    shared packed search-row layout) and a label-joined metrics JSON
    (exact/cell accuracy, acceptance counts, false accepts) under the
    run-dir convention.
    """

    class Config(Fig["HpsEval"]):
        """Checkpoint, search policy, population, and outputs."""

        runtime: SingleProcess.Config = field(
            # Device PINNED: priml defaults to ``None`` (see Trainer.Config).
            default_factory=lambda: SingleProcess.Config(device="cuda"),
        )
        """Process runtime (device + determinism)."""

        experiment_name: str = ""
        """Run identity; keys the run dir ``runs/{name}`` beneath the scratch
        root. Required (stamped by run.py when empty)."""

        doc: str = ""
        """Free-text description retained by the launch harness."""

        base_dir: Path | str | None = None
        """Scratch root; ``None`` resolves beneath ``/opt/scratch`` at
        construction."""

        checkpoint_path: str | Path = "/runs/exp010/checkpoints/step_00019500.pt"
        """The frozen generator checkpoint (either checkpoint schema;
        ``download_checkpoints.py`` places HF files exactly here). A relative
        logical path resolves beneath ``base_dir``."""

        model: TRM.Config = field(default_factory=TRM.Config)
        """Generator architecture (default: the recipe TRM)."""

        dataset: PuzzleDataset.Config = field(default_factory=PuzzleDataset.Config)
        """Puzzle data; the eval population is set from ``evaluation_count``."""

        search: HpsSearch.Config = field(default_factory=HpsSearch.Config)
        """The learned-q HPS policy (defaults are the recipe hps(5, 3, 2,
        512) with q >= 7.875 at ACT (24, 28, 32))."""

        evaluation_count: int = 422_786
        """Ordered puzzles to evaluate (the whole test split by default)."""

        max_eval_seconds: float = 21_600.0
        """Wall-clock cap for the whole pass."""

        emulate_precision_casts: bool = True
        """Set TorchInductor's precision-cast emulation (process-global)
        before model construction -- required for sound compiled bf16 eval
        (the trainer contract)."""

        metrics_path: str | Path = "/runs/{experiment_name}/dumps/hps_eval_metrics.json"
        """Label-joined scalar metrics JSON path resolved beneath ``base_dir``."""

        dump_path: str | Path = "/runs/{experiment_name}/dumps/hps_eval_predictions.npz"
        """Packed per-puzzle search dump (the shared packed search-row layout)."""

        @override
        def finalize(self) -> Self:
            if self.dataset.device is None:
                self.dataset.device = str(self.runtime.device)
            if self.dataset.base_dir is None:
                self.dataset.base_dir = self.base_dir
            return super().finalize()

    def __init__(self, config: Config) -> None:
        if not config.experiment_name:
            raise ValueError(
                "HpsEval.Config.experiment_name is required: it keys the run "
                "directory.",
            )
        if config.evaluation_count < 1:
            raise ValueError("evaluation_count must be positive.")
        self.config = config
        self.runtime = config.runtime.make()
        self.runtime.initialize()
        self.device = self.runtime.device
        # Process-global TorchInductor flag; must be set before the model
        # construction binds torch.compile (the trainer contract).
        inductor_config.emulate_precision_casts = config.emulate_precision_casts
        self._scratch = (
            config.base_dir if config.base_dir is not None else Path("/opt/scratch")
        )

    def run(self, *args: str) -> dict[str, float]:
        """Search the eval prefix; write the dump and metrics artifacts.

        Args:
          *args: Ignored (logged); accepted so launcher passthrough CLI args
            (a launcher calls ``job.run(*unparsed)``) never TypeError.

        Returns:
          metrics: The label-joined scalar metrics also written to
            ``metrics_path`` (``eval/exact_accuracy``, ``eval/false_accepts``,
            node costs, ``eval/total_seconds``).

        """
        if args:
            logger.info("Ignoring launcher passthrough args: %r.", args)
        cfg = self.config
        dataset_cfg = cfg.dataset.copy_tree()
        dataset_cfg.eval_num_instances = cfg.evaluation_count
        model = _eval_model(cfg.model, self._path(cfg.checkpoint_path), self.device)
        started = time.monotonic()
        try:
            rows, media, labels = _search_pass(
                model=model,
                dataset=dataset_cfg.make(),
                view=NINE_VIEWS[0],  # The canonical (identity) view.
                search=cfg.search,
                device=self.device,
                deadline_seconds=cfg.max_eval_seconds,
                label="HpsEval",
                join_solution_visited=True,
            )
        finally:
            del model
            _release()
        _require_population(labels.shape[0], cfg.evaluation_count, label="HpsEval")
        metrics = {
            f"eval/{name}": value
            for name, value in summarize_search(rows, labels).items()
        }
        metrics["eval/total_seconds"] = time.monotonic() - started
        write_member_dump(self._path(cfg.dump_path), rows, media=media, label=labels)
        _write_json(self._path(cfg.metrics_path), metrics)
        return metrics

    def _path(self, template: str | Path) -> Path:
        """Resolve a run-output path beneath the scratch base."""
        return _fill_template(
            template,
            base_dir=self._scratch,
            experiment_name=self.config.experiment_name,
        )


class AgreementLockEval:
    """Lazy sequential member eval under the agreement-lock (V3) rule.

    One-sentence rule: evaluate members in the frozen order; if the first
    two grids agree, lock; on any disagreement, run the full ladder on the
    disagreeing puzzles and take the plain modal (first-listed ties). See
    the module docstring for the soundness argument and output layout.
    """

    class Config(Fig["AgreementLockEval"]):
        """Committee membership, search policy, population, and outputs."""

        runtime: SingleProcess.Config = field(
            # Device PINNED: priml defaults to ``None`` (see Trainer.Config).
            default_factory=lambda: SingleProcess.Config(device="cuda"),
        )
        """Process runtime (device + determinism)."""

        experiment_name: str = ""
        """Run identity; keys the run dir ``runs/{name}`` beneath the scratch
        root. Required (stamped by run.py when empty)."""

        doc: str = ""
        """Free-text description retained by the launch harness."""

        base_dir: Path | str | None = None
        """Scratch root; ``None`` resolves beneath ``/opt/scratch`` at
        construction."""

        members: tuple[Member, ...] = field(
            default_factory=lambda: nine_view_members(
                "/runs/exp010/checkpoints/step_00019500.pt",
            ),
        )
        """Ordered committee members (the order IS the tie-break order).
        The default is the frozen nine-view ladder over the recipe
        generator; :func:`seed_ensemble_members` builds the mixed
        seed-ensemble committee. Member checkpoints are relative logical paths
        resolved beneath ``base_dir``."""

        model: TRM.Config = field(default_factory=TRM.Config)
        """Generator architecture shared by every member checkpoint."""

        dataset: PuzzleDataset.Config = field(default_factory=PuzzleDataset.Config)
        """Puzzle data; the eval subset knobs are managed per member pass."""

        search: HpsSearch.Config = field(default_factory=HpsSearch.Config)
        """The learned-q HPS policy every member runs (defaults are the
        recipe hps(5, 3, 2, 512) with q >= 7.875 at ACT (24, 28, 32))."""

        evaluation_count: int = 422_786
        """Ordered puzzles to evaluate (the whole test split by default)."""

        eval_instance_indices: tuple[int, ...] = ()
        """Explicit ascending global test indices (a targeted union run);
        empty evaluates the ordered ``evaluation_count`` prefix. Survivor
        rounds map their positions through this list."""

        max_member_eval_seconds: float = 21_600.0
        """Per-member evaluation timeout (wall clock). A runaway guard, not
        a target: a full-set member pass (422,786 puzzles, det rollout plus
        searched residuals) needs well over an hour by design -- the
        reference configs pin 21,600 s. Do not lower it for full-set runs."""

        emulate_precision_casts: bool = True
        """Set TorchInductor's precision-cast emulation (process-global)
        before member model construction -- required for sound compiled
        bf16 eval (the trainer contract)."""

        metrics_path: str | Path = (
            "/runs/{experiment_name}/dumps/agreement_lock_metrics.json"
        )
        """Aggregate JSON result path resolved beneath ``base_dir``."""

        dump_path: str | Path = "/runs/{experiment_name}/dumps/agreement_lock_dump.npz"
        """Final grids + lock bookkeeping archive."""

        member_dump_path: str | Path = (
            "/runs/{experiment_name}/dumps/member_{index}_hps_predictions.npz"
        )
        """Per-member packed search dump (``{index}`` = member position)."""

        @override
        def finalize(self) -> Self:
            if self.dataset.device is None:
                self.dataset.device = str(self.runtime.device)
            if self.dataset.base_dir is None:
                self.dataset.base_dir = self.base_dir
            return super().finalize()

    def __init__(self, config: Config) -> None:
        if not config.experiment_name:
            raise ValueError(
                "AgreementLockEval.Config.experiment_name is required: it "
                "keys the run directory.",
            )
        if len(config.members) < 2:
            raise ValueError("the agreement lock needs at least two members.")
        _validate_views(member.view for member in config.members)
        if config.evaluation_count < 1:
            raise ValueError("evaluation_count must be positive.")
        self.config = config
        self.runtime = config.runtime.make()
        self.runtime.initialize()
        self.device = self.runtime.device
        # Process-global TorchInductor flag; must be set before any member
        # model construction binds torch.compile (the trainer contract).
        inductor_config.emulate_precision_casts = config.emulate_precision_casts
        self._scratch = (
            config.base_dir if config.base_dir is not None else Path("/opt/scratch")
        )

    def run(self, *args: str) -> dict[str, float | int | str]:
        """First-pair lock, else full ladder plus modal; write artifacts.

        Args:
          *args: Ignored (logged); accepted so launcher passthrough CLI args
            (a launcher calls ``job.run(*unparsed)``) never TypeError.

        Returns:
          metrics: The scalar metrics also written to ``metrics_path``.

        """
        if args:
            logger.info("Ignoring launcher passthrough args: %r.", args)
        config = self.config
        names = [member.name or member.view.name for member in config.members]
        first, first_seconds = self._member_pass(0, survivors=None)
        _require_population(
            first["label"].shape[0],
            len(config.eval_instance_indices) or config.evaluation_count,
            label="first member pass",
        )
        second, second_seconds = self._member_pass(1, survivors=None)
        if not torch.equal(first["media"], second["media"]) or not torch.equal(
            first["label"],
            second["label"],
        ):
            raise RuntimeError("first-pair member passes are misaligned.")
        labels = first["label"]
        agree = (first["final_prediction"] == second["final_prediction"]).all(dim=-1)
        final = first["final_prediction"].clone()
        survivors = (~agree).nonzero().flatten()
        ladder: list[tuple[dict[str, Tensor], float]] = []
        if survivors.numel():
            collected = [
                first["final_prediction"][survivors],
                second["final_prediction"][survivors],
            ]
            survivor_media = first["media"][survivors]
            for index in range(2, len(names)):
                output, seconds = self._member_pass(index, survivors=survivors)
                if not torch.equal(output["media"], survivor_media):
                    raise RuntimeError("ladder member pass is misaligned.")
                ladder.append((output, seconds))
                collected.append(output["final_prediction"])
            selected, _, _ = modal_grid_predictions(torch.stack(collected))
            final[survivors] = selected
        exact = (final == labels).all(dim=-1)
        n_puzzles = int(labels.shape[0])
        passes = 2 * int(agree.sum()) + len(names) * int(survivors.numel())
        metrics: dict[str, float | int | str] = {
            "eval/exact_accuracy": float(exact.float().mean()),
            "eval/n_correct": int(exact.sum()),
            "eval/n_puzzles": n_puzzles,
            "eval/cell_accuracy": float((final == labels).float().mean()),
            "eval/locked_count": int(agree.sum()),
            "eval/wrong_locks": int((agree & ~exact).sum()),
            "eval/disagreement_count": int(survivors.numel()),
            "eval/mean_passes": passes / n_puzzles if n_puzzles else 0.0,
            "eval/algorithmic_decision_calls": 0,
            "eval/selection": "first_pair_agreement_lock_else_modal",
        }
        outputs = [(first, first_seconds), (second, second_seconds), *ladder]
        for index, (output, seconds) in enumerate(outputs):
            rows = labels if index < 2 else labels[survivors]
            member_exact = (output["final_prediction"] == rows).all(dim=-1)
            metrics[f"eval/member_{index}_name"] = names[index]
            metrics[f"eval/member_{index}_exact_accuracy"] = float(
                member_exact.float().mean(),
            )
            metrics[f"eval/member_{index}_seconds"] = seconds
        metrics["eval/total_seconds"] = math.fsum(seconds for _, seconds in outputs)
        _write_lock_dump(
            self._path(config.dump_path),
            final=final,
            labels=labels,
            locked=agree,
            survivors=survivors,
        )
        _write_json(self._path(config.metrics_path), metrics)
        return metrics

    def _member_pass(
        self,
        index: int,
        *,
        survivors: Tensor | None,
    ) -> tuple[dict[str, Tensor], float]:
        """Evaluate one member, optionally on the disagreement subset only."""
        config = self.config
        member = config.members[index]
        dataset_cfg = config.dataset.copy_tree()
        if survivors is not None:
            dataset_cfg.eval_instance_indices = _survivor_indices(
                config.eval_instance_indices,
                survivors,
            )
            dataset_cfg.eval_num_instances = None
        elif config.eval_instance_indices:
            dataset_cfg.eval_instance_indices = config.eval_instance_indices
            dataset_cfg.eval_num_instances = None
        else:
            dataset_cfg.eval_num_instances = config.evaluation_count
        started = time.monotonic()
        model = _eval_model(
            config.model,
            self._path(member.checkpoint),
            self.device,
        )
        try:
            rows, media, labels = _search_pass(
                model=model,
                dataset=dataset_cfg.make(),
                view=member.view,
                search=config.search,
                device=self.device,
                deadline_seconds=config.max_member_eval_seconds,
                label=f"member {index} ({member.name or member.view.name})",
            )
        finally:
            del model
            _release()
        write_member_dump(
            _prepared(self._path(config.member_dump_path, index=index)),
            rows,
            media=media,
            label=labels,
        )
        grid_len = media.shape[-1]
        outputs = {
            "final_prediction": rows[:, -grid_len:].to(torch.int64),
            "media": media,
            "label": labels,
        }
        return outputs, time.monotonic() - started

    def _path(self, template: str | Path, **extra: object) -> Path:
        """Resolve a run-output path beneath the scratch base."""
        return _fill_template(
            template,
            base_dir=self._scratch,
            experiment_name=self.config.experiment_name,
            **extra,
        )


class SieveEval:
    """Run the verifier-locked progressive sieve over the puzzle population.

    See the module docstring for the round structure (R0 det pass -> view
    ladder on survivors -> escalated tail -> modal over collected) and the
    per-round committee build/teardown contract.
    """

    class Config(Fig["SieveEval"]):
        """Generator, lock committee, round ladder, and outputs."""

        runtime: SingleProcess.Config = field(
            # Device PINNED: priml defaults to ``None`` (see Trainer.Config).
            default_factory=lambda: SingleProcess.Config(device="cuda"),
        )
        """Process runtime (device + determinism)."""

        experiment_name: str = ""
        """Run identity; keys the run dir ``runs/{name}`` beneath the scratch
        root. Required (stamped by run.py when empty)."""

        doc: str = ""
        """Free-text description retained by the launch harness."""

        base_dir: Path | str | None = None
        """Scratch root; ``None`` resolves beneath ``/opt/scratch`` at
        construction."""

        checkpoint_path: str | Path = "/runs/exp010/checkpoints/step_00019500.pt"
        """The frozen generator checkpoint every round evaluates (either
        checkpoint schema; ``download_checkpoints.py`` places HF files
        exactly here). A relative logical path resolves beneath ``base_dir``."""

        verifier_checkpoints: tuple[str | Path, ...] = (
            "/runs/verifier_s0/checkpoints/step_00004000.pt",
            "/runs/verifier_s1/checkpoints/step_00004000.pt",
            "/runs/verifier_s2/checkpoints/step_00004000.pt",
        )
        """The committee member checkpoints, pushed into
        ``acceptor.checkpoint_paths`` in ``finalize()`` when the acceptor is
        a :class:`VerifierAcceptor` config (the source of truth for the
        default committee). Relative logical paths resolve beneath
        ``base_dir``."""

        model: TRM.Config = field(default_factory=TRM.Config)
        """Generator architecture."""

        dataset: PuzzleDataset.Config = field(default_factory=PuzzleDataset.Config)
        """Puzzle data; the eval subset knobs are managed per round."""

        search: HpsSearch.Config = field(default_factory=HpsSearch.Config)
        """The learned-q HPS policy of the view rounds (recipe defaults)."""

        views: tuple[View, ...] = (
            NINE_VIEWS  # house-ignore[globals] -- Frozen evaluation table.
        )
        """The HPS view ladder in round order (canonical first -- the frozen
        nine-view table)."""

        det_halt_threshold: float = 2.0
        """R0 conditional release threshold (middle of the measured accuracy
        plateau)."""

        tail_search: tuple[int, int, int, int] = (7, 4, 3, 8_400)
        """Escalated tail (candidates, depth, attempts, budget) -- the
        proven operating point, not trimmed. The acceptance policy is the
        unchanged ``search`` policy."""

        acceptor: Makeable[CommitteeLock] = field(
            default_factory=VerifierAcceptor.Config,
        )
        """The frozen verifier-committee lock (unanimity above threshold 0).
        Constructed and released around every round's offer."""

        evaluation_count: int = 422_786
        """Ordered puzzles to evaluate (the R0 population)."""

        max_round_eval_seconds: float = 43_200.0
        """Per-round evaluation timeout (runaway guard, not a target).

        Sized ~2x the slowest measured full-set round so a healthy pass
        is never killed one batch short of finishing; only a genuinely
        hung round should trip :class:`EvalTimeLimitError`.
        """

        emulate_precision_casts: bool = True
        """Set TorchInductor's precision-cast emulation (process-global)
        before model construction -- required for sound compiled bf16 eval
        (the trainer contract)."""

        metrics_path: str | Path = "/runs/{experiment_name}/dumps/sieve_metrics.json"
        """Aggregate JSON result path resolved beneath ``base_dir``."""

        dump_path: str | Path = "/runs/{experiment_name}/dumps/sieve_dump.npz"
        """Final grids + per-round lock bookkeeping archive."""

        @override
        def finalize(self) -> Self:
            if self.dataset.device is None:
                self.dataset.device = str(self.runtime.device)
            if self.dataset.base_dir is None:
                self.dataset.base_dir = self.base_dir
            if isinstance(self.acceptor, VerifierAcceptor.Config):
                self.acceptor.checkpoint_paths = self.verifier_checkpoints
                if self.acceptor.device is None:
                    self.acceptor.device = str(self.runtime.device)
                if self.acceptor.base_dir is None:
                    self.acceptor.base_dir = self.base_dir
            return super().finalize()

    def __init__(self, config: Config) -> None:
        if not config.experiment_name:
            raise ValueError(
                "SieveEval.Config.experiment_name is required: it keys the "
                "run directory.",
            )
        _validate_views(config.views)
        if config.evaluation_count < 1:
            raise ValueError("evaluation_count must be positive.")
        self.config = config
        self.runtime = config.runtime.make()
        self.runtime.initialize()
        self.device = self.runtime.device
        # Process-global TorchInductor flag; must be set before any model
        # construction binds torch.compile (the trainer contract).
        inductor_config.emulate_precision_casts = config.emulate_precision_casts
        self._scratch = (
            config.base_dir if config.base_dir is not None else Path("/opt/scratch")
        )

    def run(self, *args: str) -> dict[str, object]:
        """Execute the sieve; write metrics JSON and the npz dump.

        Args:
          *args: Ignored (logged); accepted so launcher passthrough CLI args
            (a launcher calls ``job.run(*unparsed)``) never TypeError.

        Returns:
          metrics: The metrics (incl. the per-round table) also written to
            ``metrics_path``.

        """
        if args:
            logger.info("Ignoring launcher passthrough args: %r.", args)
        cfg = self.config
        n_puzzles = cfg.evaluation_count
        started = time.monotonic()
        grids, media_all, labels_all = self._run_round(0, torch.arange(n_puzzles))
        _require_population(grids.shape[0], n_puzzles, label="sieve round 0")
        final_grids = torch.zeros_like(grids)
        locked = torch.zeros(n_puzzles, dtype=torch.bool)
        collected: list[tuple[Tensor, Tensor]] = []
        round_stats: list[dict[str, float | int | str]] = []
        survivors = torch.arange(n_puzzles)
        media = media_all
        for round_index, round_name in enumerate(self._round_names()):
            if survivors.shape[0] == 0:
                break
            if round_index > 0:
                started = time.monotonic()
                grids, media, _ = self._run_round(round_index, survivors)
                if not torch.equal(media, media_all[survivors]):
                    raise RuntimeError("sieve round media misaligned with round 0.")
            # Decode-clamp every emitted grid to its input givens BEFORE any
            # lock offering or candidate collection. Givens are trusted input
            # conditioning; deep-search and modal emissions can garble them,
            # and that failure mode is a committee blind spot -- verifier
            # training corruptions only touch non-given cells, and measured
            # committee false accepts were given-violating near-misses.
            grids = _clamp_to_givens(grids, media)
            # The committee is made and torn down per round so no resident
            # committee coexists with the next round's compiled generator
            # (the internal run died with a SIGSEGV when it did).
            acceptor = cfg.acceptor.make()
            accepted = self._lock(acceptor, grids, media)
            del acceptor
            _release()
            collected.append((survivors.clone(), grids))
            final_grids[survivors[accepted]] = grids[accepted]
            locked[survivors[accepted]] = True
            on_cuda = self.device.type == "cuda"
            peak_mem_gb = (
                torch.cuda.max_memory_allocated(self.device) / 1e9 if on_cuda else 0.0
            )
            stats: dict[str, float | int | str] = {
                "round": round_name,
                "entering": int(survivors.shape[0]),
                "generated": int(grids.shape[0]),
                "locked": int(accepted.sum()),
                "deferred": int((~accepted).sum()),
                "seconds": time.monotonic() - started,
                "peak_mem_gb": peak_mem_gb,
            }
            round_stats.append(stats)
            logger.info("[sieve] %s", json.dumps(stats))
            if on_cuda:
                torch.cuda.reset_peak_memory_stats(self.device)
            survivors = survivors[~accepted]
        if survivors.shape[0]:
            final_grids[survivors] = _modal_tail(collected, survivors)
        exact = (final_grids == labels_all).all(dim=-1)
        metrics: dict[str, object] = {
            "eval/exact_accuracy": float(exact.float().mean()),
            "eval/n_correct": int(exact.sum()),
            "eval/n_puzzles": n_puzzles,
            "eval/cell_accuracy": float((final_grids == labels_all).float().mean()),
            "eval/locked_total": int(locked.sum()),
            "eval/tail_modal_count": int(survivors.shape[0]),
            "eval/algorithmic_decision_calls": 0,
            "eval/rounds": round_stats,
            "eval/total_seconds": math.fsum(
                float(stat["seconds"]) for stat in round_stats
            ),
        }
        _write_sieve_dump(
            self._path(cfg.dump_path),
            final_grids=final_grids,
            labels=labels_all,
            locked=locked,
            tail=survivors,
        )
        _write_json(self._path(cfg.metrics_path), metrics)
        return metrics

    def _round_names(self) -> list[str]:
        """Return the ordered round names: det, views, escalated tail."""
        return [
            "det_cond_halt",
            *(view.name for view in self.config.views),
            "tail_escalated",
        ]

    def _run_round(
        self,
        round_index: int,
        survivors: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Evaluate one round; returns original-space (grids, media, labels)."""
        cfg = self.config
        if round_index == 0:
            return self._det_round()
        if round_index <= len(cfg.views):
            return self._hps_round(cfg.views[round_index - 1], cfg.search, survivors)
        return self._hps_round(
            View("tail_escalated", IDENTITY_DIGITS, False),
            self._tail_search_config(),
            survivors,
        )

    # Wires to the trainer's conditional-depth eval seam via :func:`~det_pass`; batch
    # pooling is the dataset config's ``eval_batch_size``.
    def _det_round(self) -> tuple[Tensor, Tensor, Tensor]:
        """R0: conditional-halt deterministic pass over the full population."""
        cfg = self.config
        trainer_cfg = Trainer.Config()
        trainer_cfg.experiment_name = f"{cfg.experiment_name}_det_cond_halt"
        trainer_cfg.base_dir = cfg.base_dir
        # Throwaway eval-only trainer: no global reseed (weights are
        # overwritten by the checkpoint) and no per-round run-dir record.
        trainer_cfg.ephemeral = True
        trainer_cfg.runtime = cfg.runtime
        trainer_cfg.model = cfg.model
        trainer_cfg.emulate_precision_casts = cfg.emulate_precision_casts
        # R0 rolls to the same ACT depth and compute dtype as the HPS
        # rounds (one frozen rollout contract across the whole sieve).
        trainer_cfg.max_act_steps = cfg.search.max_act_steps
        trainer_cfg.dtype_autocast = cfg.search.dtype_autocast
        dataset = cfg.dataset.copy_tree()
        dataset.eval_num_instances = cfg.evaluation_count
        dataset.eval_instance_indices = ()
        trainer_cfg.dataset = dataset
        trainer_cfg.checkpointer = None
        trainer_cfg.eval_warmup_batches = 0
        trainer_cfg.max_eval_time = cfg.max_round_eval_seconds
        trainer = trainer_cfg.make()
        try:
            trainer.model.load_state_dict(
                load_eval_weights(
                    self._path(cfg.checkpoint_path),
                    device=trainer.device,
                ),
            )
            result = det_pass(trainer, halt_threshold=cfg.det_halt_threshold)
        finally:
            del trainer
            _release()
        return result.grids, result.media, result.label

    def _hps_round(
        self,
        view: View,
        search: HpsSearch.Config,
        survivors: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """One learned-q HPS round over the surviving puzzles only."""
        cfg = self.config
        dataset_cfg = cfg.dataset.copy_tree()
        # The sieve evaluates the ordered prefix, so survivor positions ARE
        # global test indices; mask-filtering an ``arange`` keeps them ascending.
        dataset_cfg.eval_instance_indices = tuple(
            from_plain(survivors.to(torch.int64).tolist(), list[int]),
        )
        dataset_cfg.eval_num_instances = None
        model = _eval_model(cfg.model, self._path(cfg.checkpoint_path), self.device)
        try:
            rows, media, labels = _search_pass(
                model=model,
                dataset=dataset_cfg.make(),
                view=view,
                search=search,
                device=self.device,
                deadline_seconds=cfg.max_round_eval_seconds,
                label=f"round {view.name}",
            )
        finally:
            del model
            _release()
        grid_len = media.shape[-1]
        return rows[:, -grid_len:].to(torch.int64), media, labels

    def _tail_search_config(self) -> HpsSearch.Config:
        """Build the escalated tail policy: ``tail_search`` over the view policy."""
        cfg = HpsSearch.Config().update(self.config.search)
        candidates, depth, attempts, budget = self.config.tail_search
        cfg.search_candidates = candidates
        cfg.search_depth = depth
        cfg.search_cell_attempts = attempts
        cfg.search_budget = budget
        return cfg

    def _lock(self, acceptor: CommitteeLock, grids: Tensor, media: Tensor) -> Tensor:
        """Offer grids to the committee on the run device; verdicts on CPU."""
        return acceptor(
            grids.to(device=self.device, dtype=torch.long),
            media.to(device=self.device, dtype=torch.long),
        ).cpu()

    def _path(self, template: str | Path, **extra: object) -> Path:
        """Resolve a run-output path beneath the scratch base."""
        return _fill_template(
            template,
            base_dir=self._scratch,
            experiment_name=self.config.experiment_name,
            **extra,
        )


def modal_grid_predictions(predictions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Choose each puzzle's modal complete grid with stable member tie-breaking.

    A member's support on a puzzle is the count of members whose grids are
    pairwise-equal to its own (itself included); the winner is the argmax
    over the member axis, so ties resolve to the FIRST-listed member -- the
    frozen committee orders are the tie-break orders.

    Args:
      predictions: ``[members, puzzles, cells]`` stacked member grids.

    Returns:
      selected: ``[puzzles, cells]`` winning grid per puzzle.
      winning_member: ``[puzzles]`` index of the winning member.
      support: ``[puzzles]`` members agreeing with the winner.

    """
    if predictions.ndim != 3 or predictions.shape[0] < 1:
        raise ValueError("predictions must have shape [members, puzzles, cells].")
    _, puzzles, _ = predictions.shape
    equal = (predictions[:, None] == predictions[None, :]).all(dim=-1)
    support_by_member = equal.sum(dim=1)
    winning_member = support_by_member.argmax(dim=0)
    puzzle_index = torch.arange(puzzles)
    selected = predictions[winning_member, puzzle_index]
    support = support_by_member[winning_member, puzzle_index]
    return selected, winning_member, support


def _validate_views(views: Iterable[View]) -> None:
    """Require every recorded view to be an exact Sudoku symmetry."""
    for view in views:
        if tuple(sorted(view.digit_permutation)) != tuple(range(1, 10)):
            raise ValueError(
                f"view {view.name!r}: digit_permutation must be a permutation "
                f"of 1..9, got {view.digit_permutation}.",
            )
        validate_grid_permutation("row_permutation", view.row_permutation)
        validate_grid_permutation("col_permutation", view.col_permutation)


def _eval_model(
    config: TRM.Config,
    checkpoint_path: Path,
    device: torch.device,
) -> TRM:
    """Build a frozen TRM carrying a checkpoint's eval (EMA) weights."""
    model = config.make()
    model.to(device)
    model.load_state_dict(load_eval_weights(checkpoint_path, device=device))
    model.eval()
    return model


# The search runs entirely in transformed space; root and final grids are inverted back
# before packing, so the returned rows, media, and labels all live in ORIGINAL space
# (labels ride along for post-hoc scoring only -- the search never reads them).
def _search_pass(
    *,
    model: TRM,
    dataset: PuzzleDataset,
    view: View,
    search: HpsSearch.Config,
    device: torch.device,
    deadline_seconds: float,
    label: str,
    join_solution_visited: bool = False,
) -> tuple[Tensor, Tensor, Tensor]:
    """Learned-q HPS over the eval split in ``view`` space, in arrival order."""
    started = time.monotonic()
    rows_parts: list[Tensor] = []
    media_parts: list[Tensor] = []
    label_parts: list[Tensor] = []
    for raw_batch in dataset.eval_dataloader():
        elapsed = time.monotonic() - started
        if elapsed > deadline_seconds:
            raise EvalTimeLimitError(
                f"{label} exceeded its eval budget ({elapsed:.0f}s > "
                f"{deadline_seconds:.0f}s).",
            )
        batch = _to_device(raw_batch, device)
        media = batch["media"]
        valid_count = batch["valid_count"]
        if valid_count == 0:
            continue
        search_batch: dict[str, object] = dict(batch)
        search_batch["media"] = view.apply(media)
        result = run_search(model, search_batch, search)
        # The trace lives in view space, so the labels join there too.
        visited = (
            solution_visited_flags(result, view.apply(batch["label"]))
            if join_solution_visited
            else None
        )
        result = replace(
            result,
            root_predictions=view.invert(result.root_predictions),
            final_predictions=view.invert(result.final_predictions),
        )
        rows_parts.append(
            pack_search_rows(result, solution_visited=visited)[:valid_count].cpu(),
        )
        media_parts.append(media[:valid_count].to(torch.int64).cpu())
        label_parts.append(batch["label"][:valid_count].to(torch.int64).cpu())
    if not rows_parts:
        raise ValueError(f"{label} found no valid eval rows.")
    return torch.cat(rows_parts), torch.cat(media_parts), torch.cat(label_parts)


# Every full-set engine shares this: the dataset silently clips a too-large instance
# cap to the split, so an unchecked pass would publish metrics over fewer puzzles than
# configured.
def _require_population(evaluated: int, expected: int, *, label: str) -> None:
    """Raise unless a pass evaluated exactly the configured population."""
    if evaluated != expected:
        raise RuntimeError(
            f"{label} evaluated {evaluated} puzzles; the configured population "
            f"is {expected} (eval_instance_indices if set, else "
            "evaluation_count).",
        )


def _survivor_indices(
    base_indices: tuple[int, ...],
    survivors: Tensor,
) -> tuple[int, ...]:
    """Global test indices for the disagreement subset."""
    positions = from_plain(survivors.tolist(), list[int])
    if base_indices:
        return tuple(base_indices[position] for position in positions)
    return tuple(positions)


# Blank cells (token 1) and padding (token 0) are untouched; a grid that already
# respects its givens passes through bit-identically.
def _clamp_to_givens(grids: Tensor, media: Tensor) -> Tensor:
    """Copy the given cells of ``media`` (tokens > 1) over ``grids``."""
    return torch.where(media > 1, media, grids)


def _modal_tail(
    collected: list[tuple[Tensor, Tensor]],
    survivors: Tensor,
) -> Tensor:
    """Modal grid over every collected candidate for the tail puzzles."""
    position = {p: i for i, p in enumerate(survivors.tolist())}
    per_round: list[Tensor] = []
    for indices, grids in collected:
        rows = torch.full(
            (survivors.shape[0], grids.shape[1]),
            255,
            dtype=grids.dtype,
        )
        keep = [
            (row, position[p])
            for row, p in enumerate(indices.tolist())
            if p in position
        ]
        for source_row, target_row in keep:
            rows[target_row] = grids[source_row]
        per_round.append(rows)
    stacked = torch.stack(per_round)
    selected, _, _ = modal_grid_predictions(stacked)
    return selected


# A ``Path`` template is a literal override kept verbatim; a ``str`` template has its
# ``{experiment_name}`` (and any ``extra``) fields filled, then the resulting logical
# path is resolved beneath ``base_dir`` (the scratch root).
def _fill_template(
    template: str | Path,
    *,
    base_dir: Path | str | None,
    experiment_name: str,
    **extra: object,
) -> Path:
    """Resolve a run-output path beneath ``base_dir``."""
    if isinstance(template, Path):
        return template
    filled = template.format(experiment_name=experiment_name, **extra)
    return Path(resolve_working_dir(base_dir, filled))


def _prepared(path: Path) -> Path:
    """Create the parent directory of an output path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


# No ``gc.collect()``. One was here, on the theory that a reference cycle might hold
# tensors ``empty_cache`` could then reclaim. Instrumenting the suite measured what it
# actually bought: 77 collections freed 8,887 cycle objects and reclaimed ZERO bytes of
# device memory, because ``memory_allocated()`` was 0 at every one -- the eval path
# builds its models on CPU, so the allocator it was clearing had never allocated.
#
# The cost was not zero. A collection is linear in the live heap, and the heap here is
# every module the tier imported: removing it took this file's slow tier from 21.3s to
# 10.9s on a 128-core Linux host, and the whole file's 25 tests from 22.4s to 10.9s.
#
# Guarding it on ``is_initialized()`` rather than deleting it was measured too, and does
# NOT work: something in the eval path touches the device, so the flag reads True at 71
# of 77 call sites while the allocator still holds nothing. The guard looks right and is
# inert.
#
# ``empty_cache`` alone stays. It is cheap when there is no allocator and correct when a
# caller does run on GPU; only the collection was speculative.
def _release() -> None:
    """Return the CUDA caching allocator's free blocks between passes/rounds."""
    if torch.cuda.is_initialized():
        torch.cuda.empty_cache()


# A crash mid-write must never tear the pipeline's aggregate metrics: the destination
# either keeps its previous content or holds the full new one.
def _write_json(path: Path, metrics: Mapping[str, object]) -> None:
    """Write the aggregate metrics JSON via tmp + atomic rename."""
    destination = _prepared(path)
    temporary = destination.with_name(f"{destination.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    temporary.replace(destination)


def _write_lock_dump(
    path: Path,
    *,
    final: Tensor,
    labels: Tensor,
    locked: Tensor,
    survivors: Tensor,
) -> None:
    """Archive the agreement-lock outcome for offline diagnosis."""
    destination = _prepared(path)
    temporary = destination.with_name(f"{destination.name}.tmp.{os.getpid()}.npz")
    np.savez_compressed(
        temporary,
        final_grids=final.numpy().astype(np.uint8),
        labels=labels.numpy().astype(np.uint8),
        locked=locked.numpy(),
        survivors=survivors.numpy().astype(np.int64),
    )
    temporary.replace(destination)


def _write_sieve_dump(
    path: Path,
    *,
    final_grids: Tensor,
    labels: Tensor,
    locked: Tensor,
    tail: Tensor,
) -> None:
    """Archive the sieve outcome for offline diagnosis."""
    destination = _prepared(path)
    temporary = destination.with_name(f"{destination.name}.tmp.{os.getpid()}.npz")
    np.savez_compressed(
        temporary,
        final_grids=final_grids.numpy().astype(np.uint8),
        labels=labels.numpy().astype(np.uint8),
        locked=locked.numpy(),
        tail=tail.numpy().astype(np.int64),
    )
    temporary.replace(destination)


# ---------------------------------------------------------------------------
# Shared helpers.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# End-to-end reproduction pipelines.
# ---------------------------------------------------------------------------


class Reproduction:
    """One end-to-end from-scratch reproduction pipeline of the 100% result.

    One sequential single-GPU job that TRAINS its models from scratch under
    fresh checkpoint identities, then evaluates the frozen full-set protocol
    against them -- mapping the earliest training step at which the method
    reaches dev-100% and full-set-100%. The three shipped variants
    (experiments.py exp012 / exp013 / exp014) are pure config choices:

    - MULTIVIEW (exp012): one recipe generator trained in checkpoint-cadence
      segments, evaluated under multiple recorded views of each checkpoint;
      a cheap searchless det gate prices every boundary and the nine-view
      agreement-lock dev screen runs only once the gate opens; the first
      perfect dev screen triggers FULL-SET agreement-lock evals at that and
      every later boundary; the final checkpoint is always full-evaluated.
      (``screen="nine_view"``, ``trigger="dev_perfect"``, lock
      ``full_eval``.)
    - COMMITTEE (exp013): stage 0 harvests a candidate corpus from a frozen
      source generator and fits the 3-seed verifier committee from scratch;
      then the generator trains with a committee-locked fast-HPS dev screen
      per boundary; the first perfect AND sound screen (zero committee false
      accepts) triggers full-set sieve evals; final always.
      (``screen="committee"``, ``trigger="dev_perfect"``, sieve
      ``full_eval``.)
    - SEEDS (exp014): several recipe generators (fresh seeds) trained
      sequentially with det-gated single-view learned-q dev screens (curve
      visibility only -- no trigger machinery), then ONE full-set
      agreement-lock committee over the seed checkpoints plus the recorded
      views of the first (the frozen member table).
      (``screen="single_view"``, ``trigger="final_only"``, lock
      ``full_eval``.)

    Training happens in ``eval_every_steps`` segments via a fresh
    :class:`~priml.baselines.sudoku.trainer.Trainer` per segment
    resuming the same checkpoint directory; the segment builder mutates ONLY
    run identity, seed, the segment horizon, the eval cadence, the dedicated
    RNG-stream pins, and checkpoint retention -- the cosine horizon
    (``total_train_steps``) is untouched, so the composed segments replay
    the recipe's single-run schedule. Crash resume is free: a re-run skips
    segments whose horizon checkpoint already exists and resumes a partial
    segment from its latest checkpoint. The resume re-walk re-runs the
    gates, screens, and any already-triggered full evals at pre-crash
    boundaries (screen/eval results are not persisted across runs; gates
    cost seconds, screens minutes, full evals hours), and those stages load
    the OLD boundary checkpoints -- so the segment builder disables the
    recipe's rolling checkpoint retention (``keep_last_n=-1``; one
    checkpoint per boundary, ~0.6 GB each at the recipe size), keeping every
    boundary checkpoint loadable on resume.

    TRACKING is single-outer-run: the runner owns at most ONE W&B run
    (``wandb_project`` -- the trainer convention: set to enable) and
    forwards the stage results through it -- each segment's last EVAL row
    (read back from the segment's ``metrics.json``; the per-step train
    stream is file-local to the segment and is NOT forwarded), screen
    results as ``screen/`` rows, trigger events and full-eval results as
    ``trigger/`` / ``full/`` rows (the final_only committee as
    ``committee/``), and ``summary/`` fields at the end. Inner runs never
    open W&B (the segment builder pins ``wandb_project=None``); the
    aggregate ``metrics.json`` under the runner's own run dir is the
    complete record -- W&B is additive. Per-eval metrics and dumps land
    under ``<scratch>/runs/{experiment_name}/dumps/``.
    """

    class Config(Fig["Reproduction"]):
        """Generator recipe/identities, screens, trigger, and the full eval."""

        runtime: SingleProcess.Config = field(
            # Device PINNED: priml defaults to ``None`` (see Trainer.Config).
            default_factory=lambda: SingleProcess.Config(device="cuda"),
        )
        """The SINGLE process runtime of the whole staged job; ``finalize()``
        pushes it wholesale into every stage template (one process, one
        device)."""

        study_name: str = ""
        """Run-family prefix recorded for provenance; not part of any path."""

        experiment_name: str = ""
        """Run identity; keys the aggregate run dir ``runs/{name}`` beneath
        the scratch root. Required (stamped by run.py when empty)."""

        doc: str = ""
        """Free-text description; forwarded as the W&B run notes."""

        base_dir: Path | str | None = None
        """Scratch root shared by every stage; ``None`` resolves from the
        environment at construction. Pushed into children left unset."""

        generator: Trainer.Config = field(default_factory=Trainer.Config)
        """The FULL generator training recipe segments derive from (the
        recipe seam experiments.py fills). Its ``max_steps`` is the staged
        horizon; ``total_train_steps`` -- the cosine horizon -- is never
        touched by segmentation."""

        generator_names: tuple[str, ...] = ()
        """Fresh checkpoint identities, one per training seed, in order
        (trained sequentially; with several generators the FIRST is the
        final committee's tie-break anchor). Required."""

        generator_seeds: tuple[int, ...] = (44,)
        """Trainer seed per generator; each also seeds its dedicated
        augmentation stream."""

        eval_every_steps: int = 1_000
        """Checkpoint/eval cadence in train steps (the trigger-eval knob)."""

        dev_screen_count: int = 5_000
        """Ordered dev puzzles per boundary screen."""

        screen: Literal["nine_view", "committee", "single_view"] = "nine_view"
        """Per-boundary dev screen kind. ``nine_view``: the agreement-lock
        protocol itself over the dev rows (``full_eval`` must be the lock);
        ``committee``: committee-locked fast HPS -- the sieve's search
        policy and acceptor over the freshly trained verifiers
        (``full_eval`` must be the sieve; entails stage 0); ``single_view``:
        one cheap learned-q pass (curve visibility). The searchless det gate
        below prices the nine_view and single_view screens; committee
        screens are cheap enough to run at every boundary and never gate."""

        screen_gate_det_accuracy: float = 0.98
        """Cheap det-proxy gate for the gated screen kinds: the dev screen
        runs only once the conditional-halt det dev accuracy crosses this
        fraction. Measured internally: below the gate, screens cost 30-90
        min (hundreds of failing roots search at full budget) yet cannot
        read a perfect count -- det below the gate leaves >= 100 dev root
        failures and the measured HPS root-fix rate (~87%) leaves >= 13
        expected misses, so no pre-gate screen can trigger. Above the gate
        screens cost minutes. The final boundary always screens."""

        trigger: Literal["dev_perfect", "final_only"] = "dev_perfect"
        """Full-eval semantics. ``dev_perfect`` (single generator): the
        first perfect dev screen -- perfect AND zero false accepts for
        committee screens -- latches full-set evals at that and every later
        boundary, and the final boundary always full-evaluates.
        ``final_only`` (the seeds variant): no trigger machinery; ONE
        full-set agreement-lock committee over the frozen member table --
        the seed checkpoints in training order plus the recorded views of
        the first (:func:`seed_ensemble_members`) -- after the last
        generator trains."""

        harvest_source_checkpoint: str | Path = ""
        """Frozen generator checkpoint stage 0 rolls out (committee screens
        only, where it is required): train the recipe generator first or
        download the reference checkpoint (the README documents both
        routes); generator transfer is proven, so any recipe-class generator
        works. A relative logical path resolves beneath ``base_dir``."""

        harvest: Harvest.Config = field(default_factory=Harvest.Config)
        """Stage-0 harvest template (committee screens only); the runner
        stamps ``harvest_source_checkpoint`` and SKIPS the stage when the
        corpus manifest already exists (crash resume)."""

        verifier: VerifierFit.Config = field(default_factory=VerifierFit.Config)
        """One committee member's fit recipe (committee screens only); the
        runner re-stamps identity and seed per member (the frozen committee
        trains at seeds 0/1/2), points its data at the fresh harvest, and
        skips members whose final checkpoint already exists (crash
        resume)."""

        verifier_names: tuple[str, ...] = ()
        """Fresh identities for the three verifier committee members, in
        seed order (committee screens only)."""

        full_eval: AgreementLockEval.Config | SieveEval.Config = field(
            default_factory=AgreementLockEval.Config,
        )
        """The full-set protocol template (the Makeable slot the trigger
        fires); the runner stamps run identity, checkpoint paths, the
        population, and output paths per eval. Agreement lock: the member
        VIEWS and tie-break order come from the template -- triggered evals
        re-point every member at the boundary checkpoint (the template's
        member checkpoints are ignored) and the final_only committee
        replaces the member table wholesale; the det pass is structurally
        EXCLUDED (no verifier -- canonical det grids carry a measured
        identical-wrong channel, unsound without one). Sieve: its ``search``
        policy also drives the committee dev screens and its ``acceptor`` is
        copied as the screens' committee lock; the det pass IS included,
        behind the verifier lock. ``evaluation_count`` is the FULL-SET
        population; dev screens use ``dev_screen_count``."""

        wandb_project: str | None = None
        """W&B project for the SINGLE outer tracker (the trainer
        convention: set -> W&B on; with several generators the step axis is
        cumulative across them). Sub-runs never open W&B; the runner
        forwards segment/screen/full-eval metrics through this tracker.
        None keeps the run file-tracked only."""

        metrics_path: str | Path = "/runs/{experiment_name}/metrics.json"
        """Aggregate JSON result path resolved beneath ``base_dir``."""

        @override
        def finalize(self) -> Self:
            self.generator.runtime = self.runtime
            self.full_eval.runtime = self.runtime
            if self.harvest.device is None:
                self.harvest.device = str(self.runtime.device)
            if self.verifier.device is None:
                self.verifier.device = str(self.runtime.device)
            for child in (
                self.generator,
                self.harvest,
                self.verifier,
                self.full_eval,
            ):
                if child.base_dir is None:
                    child.base_dir = self.base_dir
            return super().finalize()

        def fresh_run_dirs(self) -> tuple[Path, ...]:
            """Return persistent child run directories owned by this pipeline.

            Returns:
                run_dirs: Persistent child run directories.

            """
            base = self.base_dir if self.base_dir is not None else Path("/opt/scratch")
            generator_base = (
                self.generator.base_dir if self.generator.base_dir is not None else base
            )
            verifier_base = (
                self.verifier.base_dir if self.verifier.base_dir is not None else base
            )
            directories = [
                resolve_run_dir(generator_base, name) for name in self.generator_names
            ]
            directories.extend(
                resolve_working_dir(
                    verifier_base,
                    str(self.verifier.run_dir).format(experiment_name=name),
                )
                for name in self.verifier_names
            )
            harvest_base = (
                self.harvest.base_dir if self.harvest.base_dir is not None else base
            )
            harvest_out = resolve_working_dir(
                harvest_base,
                str(self.harvest.run_dir).format(
                    experiment_name=self.harvest.experiment_name,
                ),
            )
            directories.append(harvest_out.parent)
            return tuple(directories)

    def __init__(self, config: Config) -> None:
        if not config.experiment_name:
            raise ValueError(
                "Reproduction.Config.experiment_name is required: it keys "
                "the aggregate run directory.",
            )
        if config.eval_every_steps < 1:
            raise ValueError("eval_every_steps must be positive.")
        if not math.isfinite(config.generator.max_steps):
            raise ValueError(
                "the generator recipe needs a finite max_steps horizon (the "
                "staged boundaries derive from it).",
            )
        if not config.generator_names or len(config.generator_names) != len(
            config.generator_seeds,
        ):
            raise ValueError("every generator needs exactly one training seed.")
        if len(set(config.generator_names)) != len(config.generator_names):
            raise ValueError("generator checkpoint identities must be unique.")
        if config.trigger == "dev_perfect" and len(config.generator_names) != 1:
            raise ValueError(
                "the dev_perfect trigger maps ONE generator's frontier; "
                "several generators take trigger='final_only'.",
            )
        if config.trigger == "final_only" and not isinstance(
            config.full_eval,
            AgreementLockEval.Config,
        ):
            raise ValueError(
                "the final_only committee is an agreement lock; full_eval "
                "must be an AgreementLockEval config.",
            )
        if config.screen == "nine_view" and not isinstance(
            config.full_eval,
            AgreementLockEval.Config,
        ):
            raise ValueError(
                "nine_view screens run the agreement-lock protocol; "
                "full_eval must be an AgreementLockEval config.",
            )
        if config.screen == "committee":
            if not str(config.harvest_source_checkpoint):
                raise ValueError(
                    "harvest_source_checkpoint must name a frozen generator "
                    "checkpoint (train the recipe generator first or download "
                    "the reference checkpoint).",
                )
            if len(config.verifier_names) != 3:
                raise ValueError(
                    "the committee is exactly the three verifier seed recipes.",
                )
            if not isinstance(config.full_eval, SieveEval.Config):
                raise ValueError(
                    "committee screens share the sieve's search policy and "
                    "acceptor; full_eval must be a SieveEval config.",
                )
        self.config = config
        self._metrics_path()
        config.fresh_run_dirs()
        self.runtime = config.runtime.make()
        self.runtime.initialize()
        self.device = self.runtime.device
        # Process-global TorchInductor flag; must be set before any screen
        # model construction binds torch.compile (the trainer contract).
        inductor_config.emulate_precision_casts = (
            config.generator.emulate_precision_casts
        )

    def run(self, *args: str) -> None:
        """Run stage 0 if any, train with screens, and run the full evals.

        Args:
          *args: Ignored (logged); accepted so launcher passthrough CLI args
            (a launcher calls ``job.run(*unparsed)``) never TypeError.

        """
        if args:
            logger.info("Ignoring launcher passthrough args: %r.", args)
        cfg = self.config
        outer = _outer_run(cfg.wandb_project, name=cfg.experiment_name, notes=cfg.doc)
        try:
            self._run_staged(outer)
        finally:
            outer.close()

    def _run_staged(self, outer: _OuterRun) -> None:
        """Build the staged body; ``outer`` carries the single W&B lifecycle."""
        cfg = self.config
        stage_seconds = self._load_stage_seconds()
        stages: list[dict[str, float | int | str]] = []
        screens: list[dict[str, object]] = []
        fulls: list[dict[str, object]] = []
        headline: dict[str, object] = {}
        first_dev = 0
        multi = len(cfg.generator_names) > 1
        if cfg.screen == "committee":
            self._run_stage0(outer, stages, stage_seconds)
        total = int(cfg.generator.max_steps)
        boundaries = _eval_boundaries(cfg.eval_every_steps, total)
        generators = zip(cfg.generator_names, cfg.generator_seeds, strict=True)
        for index, (name, seed) in enumerate(generators):
            gate_open = False
            offset = index * total  # Cumulative W&B step axis across seeds.
            screen_prefix = f"screen/{name}/" if multi else "screen/"
            for boundary in boundaries:
                seconds, trained = self._train_segment(name, seed, boundary)
                stage_name = (
                    f"train_{name}_to_{boundary}" if multi else f"train_to_{boundary}"
                )
                seconds = self._record_stage_seconds(
                    stage_name,
                    seconds,
                    completed=trained,
                    stage_seconds=stage_seconds,
                )
                stages.append(
                    {
                        "stage": stage_name,
                        "checkpoint": name,
                        "seconds": seconds,
                    },
                )
                outer.forward_segment(
                    self._segment_metrics_path(name),
                    offset + boundary,
                    prefix=f"{name}/" if multi else "",
                )
                if cfg.screen != "committee" and not gate_open and boundary != total:
                    det_exact, det_of, det_seconds = self._det_gate(name, boundary)
                    gate_open = det_exact >= cfg.screen_gate_det_accuracy * det_of
                    gate_row: dict[str, object] = {
                        "step": boundary,
                        "det_gate_exact": det_exact,
                        "of": det_of,
                        "seconds": det_seconds,
                        "gate": "opened" if gate_open else "closed",
                    }
                    screens.append(
                        {"generator": name, **gate_row} if multi else gate_row,
                    )
                    outer.log(
                        {"det_gate_exact": det_exact, "gate_open": int(gate_open)},
                        offset + boundary,
                        prefix=screen_prefix,
                    )
                    if not gate_open:
                        continue
                exact, of, false_accepts, seconds = self._dev_screen(name, boundary)
                row: dict[str, object] = {"step": boundary, "exact": exact, "of": of}
                if false_accepts is not None:
                    row["false_accepts"] = false_accepts
                row["seconds"] = seconds
                screens.append({"generator": name, **row} if multi else row)
                outer.log(
                    {key: value for key, value in row.items() if key != "step"},
                    offset + boundary,
                    prefix=screen_prefix,
                )
                if cfg.trigger != "dev_perfect":
                    continue
                if (
                    first_dev == 0
                    and exact == cfg.dev_screen_count
                    and (false_accepts is None or false_accepts == 0)
                ):
                    first_dev = boundary
                    outer.log(
                        {"first_dev_perfect_step": first_dev},
                        boundary,
                        prefix="trigger/",
                    )
                if (first_dev and boundary >= first_dev) or boundary == total:
                    headline, seconds = self._full_eval(boundary)
                    fulls.append(self._full_row(boundary, headline, seconds))
                    outer.log(headline, boundary, prefix="full/")
        if cfg.trigger == "dev_perfect":
            first_full = _first_perfect(fulls)
            outer.log(
                {
                    "first_dev_perfect_step": first_dev,
                    "first_fullset_perfect_step": first_full,
                    "final_exact": int(_as_float(fulls[-1]["exact"])) if fulls else 0,
                },
                total,
                prefix="summary/",
            )
            results: dict[str, object] = {
                **headline,
                "eval/stages": stages,
                "eval/dev_screens": screens,
                "eval/first_dev_perfect_step": first_dev,
                "eval/full_evals": fulls,
                "eval/first_fullset_perfect_step": first_full,
                "eval/total_seconds": _total_seconds(stages, screens, fulls),
            }
        else:
            started = time.monotonic()
            headline = self._final_committee_eval(total)
            committee_seconds = time.monotonic() - started
            final_step = len(cfg.generator_names) * total
            outer.log(headline, final_step, prefix="committee/")
            outer.log(
                {"final_exact": int(_as_float(headline["eval/n_correct"]))},
                final_step,
                prefix="summary/",
            )
            results = {
                **headline,
                "eval/stages": stages,
                "eval/dev_screens": screens,
                "eval/committee_seconds": committee_seconds,
                "eval/total_seconds": committee_seconds
                + _total_seconds(stages, screens),
            }
        _write_json(self._metrics_path(), results)

    def _run_stage0(
        self,
        outer: _OuterRun,
        stages: list[dict[str, float | int | str]],
        stage_seconds: dict[str, float],
    ) -> None:
        """Stage 0 (committee screens): harvest, then the three member fits."""
        cfg = self.config
        harvest_dir, seconds = self._run_harvest()
        seconds = self._record_stage_seconds(
            "harvest",
            seconds,
            completed=seconds != 0.0 or "harvest" not in stage_seconds,
            stage_seconds=stage_seconds,
        )
        stages.append(
            {
                "stage": "harvest",
                "checkpoint": str(cfg.harvest_source_checkpoint),
                "seconds": seconds,
            },
        )
        outer.log({"harvest_seconds": seconds}, 0, prefix="train/")
        for index in range(len(cfg.verifier_names)):
            seconds = self._verifier_fit(index, harvest_dir)
            stage_name = f"train_verifier_{index}"
            seconds = self._record_stage_seconds(
                stage_name,
                seconds,
                completed=seconds != 0.0 or stage_name not in stage_seconds,
                stage_seconds=stage_seconds,
            )
            stages.append(
                {
                    "stage": stage_name,
                    "checkpoint": cfg.verifier_names[index],
                    "seconds": seconds,
                },
            )
            outer.log({f"verifier_{index}_seconds": seconds}, 0, prefix="train/")

    def _metrics_path(self) -> Path:
        """Resolve the aggregate metrics path for this pipeline identity."""
        cfg = self.config
        base = cfg.base_dir if cfg.base_dir is not None else Path("/opt/scratch")
        return _fill_template(
            cfg.metrics_path,
            base_dir=base,
            experiment_name=cfg.experiment_name,
        )

    def _progress_path(self) -> Path:
        """Durable completed-stage timing used across crash resumes."""
        metrics_path = self._metrics_path()
        return metrics_path.with_name(f"{metrics_path.stem}.progress.json")

    def _load_stage_seconds(self) -> dict[str, float]:
        """Load completed-stage seconds from progress or legacy final metrics."""
        progress_path = self._progress_path()
        if progress_path.is_file():
            try:
                progress = from_plain(
                    loads(progress_path.read_text()),
                    _ReproductionProgress,
                )
            except ReadError as error:
                raise ValueError(
                    f"invalid reproduction progress: {progress_path}.",
                ) from error
            return progress["stage_seconds"]

        # A legacy final metrics file: best effort, since it was never a resume
        # contract -- any row that is not a timed stage is skipped.
        metrics_path = self._metrics_path()
        if not metrics_path.is_file():
            return {}
        try:
            metrics = from_plain(
                loads(metrics_path.read_text()),
                dict[str, object],
            )
            rows = from_plain(metrics.get("eval/stages"), list[object], default=[])
        except ReadError:
            return {}
        seconds: dict[str, float] = {}
        for row in rows:
            try:
                stage = from_plain(row, _StageRow)
            except ReadError:
                continue
            seconds[stage["stage"]] = stage["seconds"]
        return seconds

    def _record_stage_seconds(
        self,
        stage: str,
        seconds: float,
        *,
        completed: bool,
        stage_seconds: dict[str, float],
    ) -> float:
        """Persist a completed duration or restore it for skipped work."""
        if not completed and stage in stage_seconds:
            return stage_seconds[stage]
        stage_seconds[stage] = seconds
        _write_json(
            self._progress_path(),
            {"schema_version": 1, "stage_seconds": stage_seconds},
        )
        return seconds

    def _run_harvest(self) -> tuple[Path, float]:
        """Run (or reuse) the harvest stage; returns (harvest_dir, seconds)."""
        cfg = self.config
        harvest = cfg.harvest.copy_tree()
        harvest.harvest_source_checkpoint = cfg.harvest_source_checkpoint
        base = (
            harvest.base_dir if harvest.base_dir is not None else Path("/opt/scratch")
        )
        out_dir = resolve_working_dir(
            base,
            str(harvest.run_dir).format(experiment_name=harvest.experiment_name),
        )
        manifest_path = out_dir / "manifest.json"
        if manifest_path.exists():
            # Reuse is keyed on corpus IDENTITY, not mere existence: a corpus
            # rolled out from a different generator must never silently train
            # this pipeline's committee.
            recorded = from_plain(
                loads(manifest_path.read_text()),
                _HarvestSource,
            )["source_checkpoint"]
            source_checkpoint = harvest.harvest_source_checkpoint
            configured = str(
                source_checkpoint
                if isinstance(source_checkpoint, Path)
                else resolve_working_dir(base, source_checkpoint),
            )
            if recorded != configured:
                raise ValueError(
                    f"harvest corpus at {out_dir} was rolled out from "
                    f"{recorded!r}, but this pipeline is configured with "
                    f"{configured!r}; move the stale corpus aside (--fresh) "
                    "or point the harvest at a fresh out_dir.",
                )
            logger.info(
                "harvest manifest already present under %s; reusing the corpus.",
                out_dir,
            )
            return out_dir, 0.0
        started = time.monotonic()
        out_dir = harvest.make().run()
        _release()
        return out_dir, time.monotonic() - started

    def _verifier_fit_config(self, index: int, harvest_dir: Path) -> VerifierFit.Config:
        """One committee member's fit recipe under its fresh identity/seed."""
        fit = self.config.verifier.copy_tree()
        fit.experiment_name = self.config.verifier_names[index]
        fit.seed = index  # The frozen committee trains at seeds 0/1/2.
        fit.dataset.harvest_dir = harvest_dir
        fit.dataset.node_corpus = harvest_dir / "node_corpus.npz"  # Optional.
        return fit

    def _verifier_fit(self, index: int, harvest_dir: Path) -> float:
        """Train one committee member (skipped when already on disk)."""
        if self._verifier_checkpoint_path(index).exists():
            logger.info(
                "verifier %s already trained; skipping.",
                self.config.verifier_names[index],
            )
            return 0.0
        started = time.monotonic()
        self._verifier_fit_config(index, harvest_dir).make().run()
        _release()
        return time.monotonic() - started

    def _verifier_checkpoint_path(self, index: int) -> Path:
        """Resolve final checkpoint path of one committee member."""
        cfg = self.config
        base = (
            cfg.verifier.base_dir
            if cfg.verifier.base_dir is not None
            else Path("/opt/scratch")
        )
        run_dir = resolve_working_dir(
            base,
            str(cfg.verifier.run_dir).format(
                experiment_name=cfg.verifier_names[index],
            ),
        )
        return run_dir / "checkpoints" / f"step_{cfg.verifier.max_steps:08d}.pt"

    # Each is the verifier run-dir logical path (its ``{experiment_name}`` filled) plus
    # the checkpoint suffix; the consuming acceptor resolves it beneath its injected
    # ``base_dir``.
    def _verifier_checkpoints(self) -> tuple[str, ...]:
        """Member checkpoint logical paths for the freshly trained committee."""
        cfg = self.config
        return tuple(
            str(cfg.verifier.run_dir).format(experiment_name=name)
            + f"/checkpoints/step_{cfg.verifier.max_steps:08d}.pt"
            for name in cfg.verifier_names
        )

    def _train_segment(
        self,
        name: str,
        seed: int,
        boundary: int,
    ) -> tuple[float, bool]:
        """Train one generator's segment ending at ``boundary`` (resume-aware)."""
        cfg = self.config
        segment = _generator_segment(
            cfg.generator.copy_tree(),
            study_name=cfg.study_name,
            experiment_name=name,
            num_steps_eval=cfg.eval_every_steps,
            max_steps=boundary,
            seed=seed,
        )
        return _run_training(segment)

    def _segment_metrics_path(self, name: str) -> Path:
        """One generator's own run-dir metrics file (forwarded)."""
        run_dir = resolve_run_dir(self._generator_base(), name)
        return run_dir / "metrics.json"

    def _checkpoint_path(self, name: str, step: int) -> Path:
        """Resolve checkpoint path of one generator at one boundary step."""
        run_dir = resolve_run_dir(self._generator_base(), name)
        return run_dir / "checkpoints" / f"step_{step:08d}.pt"

    def _generator_base(self) -> Path:
        """Return the generator subtree's scratch root (defaults to ``/opt/scratch``)."""
        base = self.config.generator.base_dir
        return Path(base) if base is not None else Path("/opt/scratch")

    def _det_gate(self, name: str, step: int) -> tuple[int, int, float]:
        """Run the cheap det gate at one boundary; (exact, of, seconds)."""
        cfg = self.config
        tag = (
            f"detgate_{name}_{step}"
            if len(cfg.generator_names) > 1
            else f"detgate_{step}"
        )
        return _det_gate_screen(
            recipe=cfg.generator,
            experiment_name=f"{cfg.experiment_name}_{tag}",
            checkpoint_path=self._checkpoint_path(name, step),
            count=cfg.dev_screen_count,
        )

    def _dev_screen(self, name: str, step: int) -> tuple[int, int, int | None, float]:
        """One ``screen``-kind dev screen at one generator boundary."""
        cfg = self.config
        if cfg.screen == "nine_view":
            metrics, seconds = self._lock_eval(
                step,
                count=cfg.dev_screen_count,
                tag="dev9v",
            )
            exact = int(_as_float(metrics["eval/n_correct"]))
            return exact, cfg.dev_screen_count, None, seconds
        if cfg.screen == "committee":
            exact, false_accepts, seconds = self._committee_screen(step)
            return exact, cfg.dev_screen_count, false_accepts, seconds
        exact, of, seconds = self._single_view_screen(name, step)
        return exact, of, None, seconds

    # The committee is the CANDIDATE ACCEPTANCE inside the fast search (``accept_fn`` --
    # the predicate engine with the q-halt@8 early exit); no rule predicate and no
    # labels anywhere on the decision path. Returns (exact, false_accepts, seconds).
    def _committee_screen(self, step: int) -> tuple[int, int, float]:
        """One committee-locked fast-HPS dev screen at one boundary step."""
        cfg = self.config
        assert isinstance(cfg.full_eval, SieveEval.Config)  # __init__ validated.
        started = time.monotonic()
        acceptor = self._screen_acceptor_config().make()
        model = _eval_model(
            cfg.generator.model,
            self._checkpoint_path(cfg.generator_names[0], step),
            self.device,
        )
        dataset = cfg.generator.dataset.copy_tree()
        dataset.eval_num_instances = cfg.dev_screen_count
        dataset.eval_instance_indices = ()
        if dataset.device is None:
            dataset.device = str(cfg.runtime.device)
        try:
            accepted, exact_rows = _screen_pass(
                model=model,
                dataset=dataset.make(),
                search=cfg.full_eval.search,
                device=self.device,
                accept_fn=acceptor,
            )
        finally:
            del model, acceptor
            _release()
        exact = int(exact_rows.sum())
        false_accepts = int((accepted & ~exact_rows).sum())
        return exact, false_accepts, time.monotonic() - started

    def _single_view_screen(self, name: str, step: int) -> tuple[int, int, float]:
        """One cheap single-view learned-q dev screen (curve only)."""
        cfg = self.config
        started = time.monotonic()
        model = _eval_model(
            cfg.generator.model,
            self._checkpoint_path(name, step),
            self.device,
        )
        dataset = cfg.generator.dataset.copy_tree()
        dataset.eval_num_instances = cfg.dev_screen_count
        dataset.eval_instance_indices = ()
        if dataset.device is None:
            dataset.device = str(cfg.runtime.device)
        try:
            _, exact_rows = _screen_pass(
                model=model,
                dataset=dataset.make(),
                search=cfg.full_eval.search,
                device=self.device,
            )
        finally:
            del model
            _release()
        return (
            int(exact_rows.sum()),
            int(exact_rows.shape[0]),
            time.monotonic() - started,
        )

    # A fresh copy of the sieve's committee acceptor re-pointed at the freshly trained
    # verifiers; a non-committee acceptor config passes through untouched (``make()``
    # copies before finalizing).
    def _screen_acceptor_config(self) -> Makeable[CommitteeLock]:
        """Build the screens' lock: the sieve's acceptor over the fresh committee."""
        cfg = self.config
        assert isinstance(cfg.full_eval, SieveEval.Config)  # __init__ validated.
        base = cfg.full_eval.acceptor
        if not isinstance(base, VerifierAcceptor.Config):
            return base
        acceptor = base.copy_tree()
        acceptor.checkpoint_paths = self._verifier_checkpoints()
        if acceptor.device is None:
            acceptor.device = str(cfg.runtime.device)
        if acceptor.base_dir is None:
            acceptor.base_dir = cfg.base_dir
        return acceptor

    def _full_eval(self, step: int) -> tuple[dict[str, object], float]:
        """One triggered full-set eval at one boundary; (metrics, seconds)."""
        if isinstance(self.config.full_eval, SieveEval.Config):
            run = self._sieve_run_config(step)
            started = time.monotonic()
            metrics: dict[str, object] = dict(run.make().run())
            _release()
            return metrics, time.monotonic() - started
        return self._lock_eval(
            step,
            count=int(self.config.full_eval.evaluation_count),
            tag="full9v",
        )

    def _full_row(
        self,
        step: int,
        headline: Mapping[str, object],
        seconds: float,
    ) -> dict[str, object]:
        """One row of the aggregate full-eval table."""
        row: dict[str, object] = {
            "step": step,
            "exact": int(_as_float(headline["eval/n_correct"])),
            "of": int(self.config.full_eval.evaluation_count),
        }
        if isinstance(self.config.full_eval, SieveEval.Config):
            row["rounds"] = headline["eval/rounds"]
        row["seconds"] = seconds
        return row

    def _lock_run_config(
        self,
        step: int,
        *,
        count: int,
        tag: str,
    ) -> AgreementLockEval.Config:
        """One per-eval clone of the lock protocol at one boundary step."""
        cfg = self.config
        assert isinstance(cfg.full_eval, AgreementLockEval.Config)  # Validated.
        run = cfg.full_eval.copy_tree()
        run.experiment_name = f"{cfg.experiment_name}_{tag}_{step}"
        checkpoint = _checkpoint_template(cfg.generator_names[0], step)
        run.members = tuple(
            Member(checkpoint, member.view, member.name) for member in run.members
        )
        run.evaluation_count = count
        dumps = f"/runs/{cfg.experiment_name}/dumps"
        run.metrics_path = f"{dumps}/{tag}_metrics_{step}.json"
        run.dump_path = f"{dumps}/{tag}_dump_{step}.npz"
        run.member_dump_path = (
            f"{dumps}/{tag}_member_{{index}}_hps_predictions_{step}.npz"
        )
        return run

    def _lock_eval(
        self,
        step: int,
        *,
        count: int,
        tag: str,
    ) -> tuple[dict[str, object], float]:
        """Run the agreement-lock ladder at one checkpoint step."""
        run = self._lock_run_config(step, count=count, tag=tag)
        started = time.monotonic()
        metrics: dict[str, object] = dict(run.make().run())
        _release()
        return metrics, time.monotonic() - started

    # ``copy_tree`` (never ``SieveEval.Config().update``): the class-preserving copy
    # lets a sieve SUBCLASS config -- extra fields and overridden round seams -- survive
    # the per-eval rebuild.
    def _sieve_run_config(self, step: int) -> SieveEval.Config:
        """One per-eval clone of the sieve protocol at one boundary step."""
        cfg = self.config
        assert isinstance(cfg.full_eval, SieveEval.Config)  # __init__ validated.
        run = cfg.full_eval.copy_tree()
        run.checkpoint_path = _checkpoint_template(cfg.generator_names[0], step)
        run.verifier_checkpoints = self._verifier_checkpoints()
        # The template is already finalized, so its finalize -- which pushes
        # verifier_checkpoints into the acceptor -- never reruns; without this the
        # sieve locks with the template's committee, not the one just trained.
        run.acceptor = self._screen_acceptor_config()
        run.experiment_name = f"{cfg.experiment_name}_sieve_{step}"
        dumps = f"/runs/{cfg.experiment_name}/dumps"
        run.metrics_path = f"{dumps}/sieve_metrics_{step}.json"
        run.dump_path = f"{dumps}/sieve_dump_{step}.npz"
        return run

    def _committee_run_config(self, step: int) -> AgreementLockEval.Config:
        """Build the final committee: fresh seeds in training order + anchor views."""
        cfg = self.config
        assert isinstance(cfg.full_eval, AgreementLockEval.Config)  # Validated.
        run = cfg.full_eval.copy_tree()
        run.experiment_name = f"{cfg.experiment_name}_committee"
        run.members = seed_ensemble_members(
            tuple(_checkpoint_template(name, step) for name in cfg.generator_names),
        )
        dumps = f"/runs/{cfg.experiment_name}/dumps"
        run.metrics_path = f"{dumps}/committee_metrics.json"
        run.dump_path = f"{dumps}/committee_dump.npz"
        run.member_dump_path = f"{dumps}/member_{{index}}_hps_predictions.npz"
        return run

    def _final_committee_eval(self, step: int) -> dict[str, object]:
        """Run the single full-set committee eval over the final checkpoints."""
        run = self._committee_run_config(step)
        metrics: dict[str, object] = dict(run.make().run())
        _release()
        return metrics


def _generator_segment(
    recipe: Trainer.Config,
    *,
    study_name: str,
    experiment_name: str,
    num_steps_eval: int,
    max_steps: int,
    seed: int,
) -> Trainer.Config:
    """One training segment of ``recipe`` under a fresh identity."""
    cfg = recipe
    cfg.study_name = study_name
    cfg.experiment_name = experiment_name
    cfg.seed = seed
    cfg.num_steps_eval = num_steps_eval
    cfg.max_steps = max_steps
    cfg.halt_exploration_seed = 0  # Explicit pin (the recipe's default).
    cfg.dataset.augment_seed = seed  # Dedicated deterministic augmentation stream.
    cfg.dataset.seed = 0  # Explicit pin (the recipe's train-shuffle stream).
    cfg.wandb_project = None  # No W&B in sub-runs; the outer run owns W&B.
    if cfg.checkpointer is not None:
        # Identity is runner-owned via ``cfg.experiment_name`` above; the
        # trainer's finalize injects the run directory into the checkpointer's
        # ``base_dir``, so the checkpoint dir tracks the fresh identity. The
        # runner copies an ALREADY-finalized ``generator`` seam, so both fields
        # carry a stale run dir resolved under the unstamped identity; reset
        # them to their logical defaults so re-finalization re-injects the
        # freshly stamped run directory.
        cfg.checkpointer.base_dir = None
        default_working_dir = _field_default(cfg.checkpointer, "working_dir")
        assert isinstance(default_working_dir, (str, Path))
        cfg.checkpointer.working_dir = default_working_dir
        _clear_finalized(cfg.checkpointer)
        # The top-level config is finalized too; clear it so ``make()``
        # re-finalizes (and re-injects the checkpointer run dir) instead of
        # skipping finalize on an already-finalized tree.
        _clear_finalized(cfg)
        # The crash-resume re-walk loads OLD boundary checkpoints (gates,
        # screens, full evals); the recipe's rolling retention would prune
        # them past keep_last_n boundaries and wedge every resume, so
        # segments keep every boundary checkpoint.
        cfg.checkpointer.keep_last_n = -1
    return cfg


# ``finalize`` is not idempotent (it resolves logical paths to absolute); a re-finalize
# of an already-finalized subtree is otherwise skipped by the ``_finalized`` guard,
# leaving stale derived values in place.
def _clear_finalized(config: object) -> None:
    """Mark a finalized config pending again so a later finalize re-derives it."""
    object.__setattr__(config, "_finalized", False)


def _field_default(config: DataclassLike, name: str) -> object:
    """Return the declared dataclass default of ``config``'s ``name`` field."""
    for spec in fields(config):
        if spec.name == name:
            return spec.default
    raise AttributeError(f"{type(config).__name__} has no field {name!r}.")


# Crash resume: a completed segment (latest checkpoint at or past the segment horizon)
# is skipped without constructing a trainer; a partial segment resumes from its latest
# checkpoint through the trainer's ordinary resume.
def _run_training(config: Trainer.Config) -> tuple[float, bool]:
    """Run one training segment; skip it when its horizon is already on disk."""
    finalized = config.copy_tree().finalize()
    if finalized.checkpointer is not None:
        steps = finalized.checkpointer.make().available_steps()
        boundary = int(finalized.max_steps)
        if boundary in steps:
            logger.info(
                "segment to step %d already trained (exact boundary checkpoint "
                "present); skipping.",
                boundary,
            )
            return 0.0, False
        later = [step for step in steps if step > boundary]
        if later:
            raise RuntimeError(
                "cannot safely resume segment: exact boundary checkpoint at "
                f"step {boundary} is missing, but later checkpoints exist "
                f"({later}). Refusing to rewind or treat a later state as the "
                "requested boundary.",
            )
    started = time.monotonic()
    trainer = config.make()
    try:
        trainer.run()
    finally:
        del trainer
        _release()
    return time.monotonic() - started, True


# No search anywhere: a throwaway eval-only trainer (the recipe's model / runtime /
# rollout knobs; checkpointing off) loads the boundary checkpoint's eval weights and
# runs the conditional-halt deterministic pass (per-row release at q >= +2, the
# registered bar). Purely a cost gate for the expensive learned-q screens: early-
# training screens explode because every failing root searches at full budget.
def _det_gate_screen(
    *,
    recipe: Trainer.Config,
    experiment_name: str,
    checkpoint_path: Path,
    count: int,
) -> tuple[int, int, float]:
    """Cheap conditional-halt det dev screen; returns (exact, of, seconds)."""
    cfg = Trainer.Config()
    cfg.experiment_name = experiment_name
    cfg.base_dir = recipe.base_dir
    # Throwaway eval-only trainer: no global reseed (weights are overwritten
    # by the checkpoint) and no per-boundary run-dir record.
    cfg.ephemeral = True
    cfg.runtime = recipe.runtime
    cfg.model = recipe.model
    cfg.emulate_precision_casts = recipe.emulate_precision_casts
    cfg.max_act_steps = recipe.max_act_steps
    cfg.dtype_autocast = recipe.dtype_autocast
    dataset = recipe.dataset.copy_tree()
    dataset.eval_num_instances = count
    dataset.eval_instance_indices = ()
    cfg.dataset = dataset
    cfg.checkpointer = None
    cfg.eval_warmup_batches = 0
    started = time.monotonic()
    trainer = cfg.make()
    try:
        trainer.model.load_state_dict(
            load_eval_weights(checkpoint_path, device=trainer.device),
        )
        result = det_pass(trainer)
    finally:
        del trainer
        _release()
    exact = (result.grids == result.label).all(dim=-1)
    return int(exact.sum()), int(exact.shape[0]), time.monotonic() - started


# The learned-q policy runs when ``accept_fn`` is None (the single-view dev screen); a
# grid predicate switches to the committee-locked fast engine. Acceptance is label-free;
# labels join post hoc for the screen metrics only.
def _screen_pass(
    *,
    model: TRM,
    dataset: PuzzleDataset,
    search: HpsSearch.Config,
    device: torch.device,
    accept_fn: GridAcceptor | None = None,
) -> tuple[Tensor, Tensor]:
    """One search pass over the eval split; per-row screen masks."""
    accepted_parts: list[Tensor] = []
    exact_parts: list[Tensor] = []
    for raw_batch in dataset.eval_dataloader():
        batch = _to_device(raw_batch, device)
        valid_count = batch["valid_count"]
        if valid_count == 0:
            continue
        result = run_search(model, batch, search, accept_fn=accept_fn)
        exact = (result.final_predictions == batch["label"]).all(dim=-1)
        accepted_parts.append(result.accepted[:valid_count].cpu())
        exact_parts.append(exact[:valid_count].cpu())
    if not accepted_parts:
        raise ValueError("the dev screen found no valid eval rows.")
    return torch.cat(accepted_parts), torch.cat(exact_parts)


def _to_device(batch: PuzzleBatch, device: torch.device) -> PuzzleBatch:
    """Move a batch's tensors to ``device``."""
    return {
        "media": batch["media"].to(device),
        "label": batch["label"].to(device),
        "valid_count": batch["valid_count"],
        "puzzle_identifiers": batch["puzzle_identifiers"].to(device),
    }


def _checkpoint_template(experiment_name: str, step: int) -> str:
    """Run-dir checkpoint logical path (resolved beneath the consumer's base)."""
    return f"/runs/{experiment_name}/checkpoints/step_{step:08d}.pt"


def _eval_boundaries(every: int, total: int) -> list[int]:
    """Checkpoint boundaries: every ``every`` steps, always ending at total."""
    boundaries = list(range(every, total, every))
    boundaries.append(total)
    return boundaries


class _StageTracker(Protocol):
    """What the outer run needs from the optional W&B tracker."""

    def log_metrics(
        self,
        metrics: Mapping[str, object],
        step: int,
        *,
        prefix: str = "",
    ) -> None:
        """Log scalar metrics at ``step``."""
        ...

    def close(self) -> None:
        """Finish the tracker."""
        ...


def _outer_run(wandb_project: str | None, *, name: str, notes: str) -> _OuterRun:
    """Build the single outer tracker lifecycle (None project = disabled)."""
    if wandb_project is None:
        return _OuterRun(None)
    cfg = WandbTracker.Config()
    cfg.project = wandb_project
    cfg.name = name
    cfg.run_id = hashlib.sha256(f"{wandb_project}\0{name}".encode()).hexdigest()[:16]
    cfg.notes = notes
    return _OuterRun(cfg.make())


class _OuterRun:
    """The single outer tracker lifecycle for one staged job.

    Wraps an already-made tracker (or None = disabled): forwards scalar
    metrics with stage prefixes, clamps the W&B step axis monotonic (nested
    stages may report at earlier steps), and forwards the eval rows finished
    training segments leave in their run-dir ``metrics.json``.
    """

    def __init__(self, tracker: _StageTracker | None) -> None:
        self._tracker = tracker
        self._step = 0

    def log(
        self,
        metrics: Mapping[str, object],
        step: int,
        *,
        prefix: str = "",
    ) -> None:
        """Forward scalar metrics at ``step`` (non-scalars are dropped)."""
        if self._tracker is None:
            return
        self._step = max(self._step, int(step))
        self._tracker.log_metrics(dict(metrics), self._step, prefix=prefix)

    def forward_segment(
        self,
        metrics_path: Path,
        step: int,
        *,
        prefix: str = "",
    ) -> None:
        """Forward a finished segment's last eval row from its metrics file.

        Args:
            metrics_path: Segment metrics JSON path.
            step: W&B step assigned to the forwarded row.
            prefix: Prefix added to forwarded metric names.

        The rows carry their own ``eval/`` key prefixes (the trainer's file
        tracker keeps the last write -- at cadence-length segments the
        boundary row), so the segment trajectory stays step-continuous
        across segments; ``prefix`` disambiguates multi-generator runs.

        """
        if self._tracker is None or not metrics_path.exists():
            return
        rows = from_plain(loads(metrics_path.read_text()), dict[str, object])
        if rows:
            self.log(rows, step, prefix=prefix)

    def close(self) -> None:
        """Finish the tracker (idempotent by tracker contract)."""
        if self._tracker is not None:
            self._tracker.close()


class _ReproductionProgress(TypedDict):
    """The durable completed-stage timing file."""

    schema_version: Literal[1]
    stage_seconds: dict[str, float]


class _StageRow(TypedDict):
    """One timed stage of a legacy final metrics file."""

    stage: str
    seconds: float


class _HarvestSource(TypedDict):
    """The provenance field a reused harvest corpus is keyed on."""

    source_checkpoint: str


def _first_perfect(fulls: Sequence[Mapping[str, object]]) -> int:
    """Return the first full-eval step whose exact count equals its population."""
    for entry in fulls:
        if entry["exact"] == entry["of"]:
            return int(_as_float(entry["step"]))
    return 0


def _total_seconds(*tables: Sequence[Mapping[str, object]]) -> float:
    """Sum the ``seconds`` column across stage/eval tables."""
    return math.fsum(_as_float(row["seconds"]) for table in tables for row in table)


def _as_float(value: object) -> float:
    """Narrow a JSON-ish scalar to float."""
    assert isinstance(value, (int, float, str))
    return float(value)
