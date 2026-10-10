"""Verifier-corpus harvest: on-policy candidate rows from a frozen generator.

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

``out_dir`` and ``harvest_source_checkpoint`` are config fields, and
:meth:`Harvest.run` drives the model directly. Shards carry no per-range
partition column: the verifier trains on GLOBAL group ids below its
``train_group_end`` (700 on the full corpus). One shard = 8 consecutive base
groups (one 16-view eval batch = 8 groups x 2 views).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, cast

import json
import logging
import os
import zipfile

from configgle import Fig
from torch import Tensor
from torch._inductor import config as inductor_config
from tqdm import tqdm
from treekle.codec import from_plain

import numpy as np
import torch

from priml.baselines.sudoku.latent_recursion import LatentRecursion
from priml.baselines.sudoku.puzzle_data import load_puzzle_dataset
from priml.baselines.sudoku.rollout import LearnedCheckpointRollout
from priml.baselines.sudoku.runner import file_sha256, load_eval_weights, open_npz
from priml.baselines.sudoku.search import select_pin_candidates
from priml.paths import resolve_working_dir
from priml.runtime import get_device


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from numpy.typing import NDArray


logger = logging.getLogger(__name__)


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
    model: LatentRecursion,
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
    """The verifier-corpus harvest job (shards + manifest).

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

        model: LatentRecursion.Config = field(default_factory=LatentRecursion.Config)
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
        """Shard/manifest output directory resolved beneath ``base_dir``.
        Existing shards or manifest are never overwritten."""

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
                with open_npz(path) as existing_file:
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
                {"file": path.name, "rows": rows, "sha256": file_sha256(path)},
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
                from_plain(cast("object", kinds.tolist()), list[int]),
                from_plain(cast("object", counts.tolist()), list[int]),
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


# The fixed node census behind ``source_kind`` 1: the same tree the
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
