"""Hypothesis-pinning search (HPS) with learned-q acceptance.

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
``hps(7, 4, 3, 8400)`` (the sieve's ``tail_search``). Sibling hypotheses
batch together (the fast engine's win) and every forward is capped at
``search_max_rows`` (2,048) rows.

An alternative grid-predicate acceptance path (``accept_fn``) drives the same
tree with the sound label-free validity rule (:func:`accepted_grids`) or a
verifier-committee predicate, with the q-halt@8 early exit pricing false
branches -- the committee dev-screen engine.

Seam contract: :func:`run_search` / :meth:`HpsSearch.run` consume a bare
:class:`~priml.baselines.sudoku.latent_recursion.LatentRecursion`
(eval weights -- the EMA swap is the CALLER's job, e.g. via the trainer's EMA
``apply_to``) and one dataset batch dict (``media`` / ``valid_count`` /
``puzzle_identifiers``; ``label`` is never read).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

import math

from configgle import Fig
from torch import Tensor

import torch

from priml.baselines.sudoku.puzzle_spec import SudokuSpec, sudoku_group_indices
from priml.baselines.sudoku.rollout import (
    family_rollout,
    learned_checkpoint_rollout_rows,
    learned_persistence_scores,
    learned_persistent_rollout_rows,
    segmented_rollout_rows,
)


if TYPE_CHECKING:
    from priml.baselines.sudoku.latent_recursion import LatentRecursion


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


class HpsSearch:
    """The hps(C, K, A, budget) engine with learned-q acceptance.

    ``Config()`` defaults are the recipe policy: ``hps(5, 3, 2, 512)`` with
    q >= 7.875 required at ACT steps 24, 28 AND 32 and full-depth candidate
    rollouts. See the module docstring for the algorithm and the seam
    contract; the sieve's ``tail_search`` is the ``hps(7, 4, 3, 8400)``
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
        model: LatentRecursion,
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
        model: LatentRecursion,
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
        model: LatentRecursion,
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

    def _autocast(self, model: LatentRecursion) -> torch.amp.autocast:
        return torch.amp.autocast(
            device_type=model.device.type,
            dtype=self.config.dtype_autocast,
            enabled=self.config.dtype_autocast is not None,
            cache_enabled=False,
        )


def run_search(
    model: LatentRecursion,
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
        (:func:`~priml.baselines.sudoku.puzzle_spec.sudoku_group_indices`).

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
