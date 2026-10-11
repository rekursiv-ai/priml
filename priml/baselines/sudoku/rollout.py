"""Frozen-model ACT rollouts: the units the search and the harvest replay.

Every rollout starts from a fresh ``init_z`` and threads the decoded grid back
as feedback with the given cells clamped -- the train step's fixed-depth eval
recurrence -- so :func:`family_rollout` on the original boards is
bit-identical to the plain eval. The variants differ in what they keep:
:func:`segmented_rollout_rows` prunes at q@8, and the checkpoint rollouts
retain q and the decoded grid at fixed ACT steps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from torch import Tensor

import torch


if TYPE_CHECKING:
    from priml.baselines.sudoku.latent_recursion import LatentRecursion


@dataclass(slots=True, frozen=True, kw_only=True)
class LearnedCheckpointRollout:
    """Final logits plus learned-q and decoded grids at fixed ACT steps."""

    logits: Tensor
    q_scores: Tensor
    predictions: Tensor


def family_rollout(
    model: LatentRecursion,
    boards: Tensor,
    *,
    step_kwargs: dict[str, Tensor],
    max_steps: int,
) -> tuple[dict[str, Tensor], Tensor]:
    """Full-ACT eval rollout on ``boards`` (the deterministic root unit).

    Fresh ``init_z``, decoded-grid feedback threaded from ``boards`` with
    given cells clamped -- exactly the train step's fixed-depth eval recurrence,
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
    model: LatentRecursion,
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
    model: LatentRecursion,
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
    model: LatentRecursion,
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
