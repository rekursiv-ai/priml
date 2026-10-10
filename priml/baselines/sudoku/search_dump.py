"""Packed search rows: the dump schema of every search pass.

Packed row layout (the dump schema shared with the lock engines, the
harvest, and the repro pipelines; width
``learned_hps_output_width(G) = 7 + 2 * G`` float32 columns)::

    col 0        : score       final learned-q of the emitted grid
    col 1        : accepted    1.0 accepted / 0.0 not
    col 2        : nodes       candidate rollouts spent
    col 3        : depth       pins at acceptance (0 root, -1 unaccepted)
    col 4        : candidates  scored root (1) + nodes
    col 5        : solution_visited   label-joined diagnostic (0 if unjoined)
    col 6        : algorithmic_decision_calls   always 0 (no rule predicate)
    cols 7..7+G  : root grid tokens (deterministic rollout argmax)
    cols 7+G..   : final grid tokens (accepted grid, else the root grid)

:func:`write_member_dump` persists these rows as an npz member dump alongside
``media`` (and optionally ``label``).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import os

from torch import Tensor

import numpy as np
import torch


if TYPE_CHECKING:
    from priml.baselines.sudoku.search import SearchResult


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


def _exact_rows(preds: Tensor, label: Tensor, valid: Tensor) -> Tensor:
    """``[B]`` exact-correctness over label-valid cells (pad rows False)."""
    counts = valid.sum(dim=-1)
    return (((preds == label) & valid).sum(dim=-1) == counts) & (counts > 0)
