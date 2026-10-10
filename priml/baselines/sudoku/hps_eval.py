"""The canonical single-checkpoint HPS evaluation and the member pass it shares.

:class:`HpsEval` -- the canonical single-model search protocol over one
frozen checkpoint: a deterministic root rollout per puzzle, learned-q
acceptance, and the fast pin search on rejected rows -- label-free end to
end; labels join post hoc for the metrics (incl. the ``solution_visited``
false-reject diagnostic). :func:`search_pass` is that pass under any recorded
view; the lock engines run one per member or round.
"""

from __future__ import annotations

from dataclasses import field, replace
from pathlib import Path
from typing import Self, override

import logging
import time

from configgle import Fig
from torch import Tensor
from torch._inductor import config as inductor_config

import torch

from priml.baselines.sudoku.latent_recursion import LatentRecursion
from priml.baselines.sudoku.puzzle_data import PuzzleDataset
from priml.baselines.sudoku.runner import (
    eval_model,
    fill_template,
    release,
    require_population,
    to_device,
    write_json,
)
from priml.baselines.sudoku.search import HpsSearch, run_search
from priml.baselines.sudoku.search_dump import (
    pack_search_rows,
    solution_visited_flags,
    summarize_search,
    write_member_dump,
)
from priml.baselines.sudoku.views import NINE_VIEWS, View
from priml.runtime import SingleProcess
from priml.train.train_loop import EvalTimeLimitError


logger = logging.getLogger(__name__)


class HpsEval:
    """Full-set learned-q HPS evaluation of one frozen generator checkpoint.

    The canonical single-model search protocol: a deterministic root rollout
    per puzzle, learned-q acceptance (q >= 7.875 at ACT 24, 28 and 32), and
    the fast pin search ``hps(5, 3, 2, 512)`` on rejected rows -- label-free
    end to end. One member pass of the lock engines under the canonical
    view (the shared :func:`search_pass` engine), plus the label-joined
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
        root. Required (the launcher stamps the factory name when empty)."""

        doc: str = ""
        """Free-text description retained by the launch harness."""

        base_dir: Path | str | None = None
        """Scratch root; ``None`` resolves beneath ``/opt/scratch`` at
        construction."""

        checkpoint_path: str | Path = ""
        """The frozen generator checkpoint, required: a chain rung's final
        checkpoint (either checkpoint schema). A relative logical path
        resolves beneath ``base_dir``."""

        model: LatentRecursion.Config = field(default_factory=LatentRecursion.Config)
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
        if not str(config.checkpoint_path):
            raise ValueError(
                "HpsEval.Config.checkpoint_path is required: the frozen generator "
                "a chain rung trained.",
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
        model = eval_model(cfg.model, self._path(cfg.checkpoint_path), self.device)
        started = time.monotonic()
        try:
            rows, media, labels = search_pass(
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
            release()
        require_population(labels.shape[0], cfg.evaluation_count, label="HpsEval")
        metrics = {
            f"eval/{name}": value
            for name, value in summarize_search(rows, labels).items()
        }
        metrics["eval/total_seconds"] = time.monotonic() - started
        write_member_dump(self._path(cfg.dump_path), rows, media=media, label=labels)
        write_json(self._path(cfg.metrics_path), metrics)
        return metrics

    def _path(self, template: str | Path) -> Path:
        """Resolve a run-output path beneath the scratch base."""
        return fill_template(
            template,
            base_dir=self._scratch,
            experiment_name=self.config.experiment_name,
        )


# The search runs entirely in transformed space; root and final grids are inverted back
# before packing, so the returned rows, media, and labels all live in ORIGINAL space
# (labels ride along for post-hoc scoring only -- the search never reads them).
def search_pass(
    *,
    model: LatentRecursion,
    dataset: PuzzleDataset,
    view: View,
    search: HpsSearch.Config,
    device: torch.device,
    deadline_seconds: float,
    label: str,
    join_solution_visited: bool = False,
) -> tuple[Tensor, Tensor, Tensor]:
    """Learned-q HPS over the eval split in ``view`` space, in arrival order.

    Args:
      model: Frozen generator carrying its eval weights.
      dataset: Eval split; its subset knobs choose the puzzles.
      view: Recorded symmetry the search runs under.
      search: The learned-q HPS policy.
      device: Device every batch moves to.
      deadline_seconds: Wall-clock budget of the whole pass.
      label: Pass name in the timeout and empty-split errors.
      join_solution_visited: Pack the label-joined ``solution_visited`` flag.

    Returns:
      rows: ``[N, 7 + 2G]`` packed search rows of the valid puzzles.
      media: ``[N, G]`` input boards, original space.
      labels: ``[N, G]`` solution tokens, original space.

    Raises:
      EvalTimeLimitError: The pass exceeded ``deadline_seconds``.
      ValueError: The split held no valid rows.

    """
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
        batch = to_device(raw_batch, device)
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
