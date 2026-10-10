"""The agreement-lock committee: lock a grid two symmetric views agree on.

The lock engines evaluate a frozen generator checkpoint (or several) over the
ordered test prefix and decide, label-free, which decoded grids to LOCK as
final answers. Labels join only in the post-hoc metrics.

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
learned-q HPS pass. Members are ``(checkpoint, view)`` pairs; the recipe
committee is one checkpoint under the nine
:data:`~priml.baselines.sudoku.views.NINE_VIEWS` views.

Outputs land under ``<scratch>/runs/{experiment_name}/dumps/``: the metrics
JSON, the lock dump, and one packed member dump per member (the packed
search row layout).
"""

from __future__ import annotations

from dataclasses import field
from pathlib import Path
from typing import NamedTuple, Self, override

import logging
import math
import os
import time

from configgle import Fig
from torch import Tensor
from torch._inductor import config as inductor_config
from treekle.codec import from_plain

import numpy as np
import torch

from priml.baselines.sudoku.hps_eval import search_pass
from priml.baselines.sudoku.latent_recursion import LatentRecursion
from priml.baselines.sudoku.puzzle_data import PuzzleDataset
from priml.baselines.sudoku.runner import (
    eval_model,
    fill_template,
    prepared,
    release,
    require_population,
    write_json,
)
from priml.baselines.sudoku.search import HpsSearch
from priml.baselines.sudoku.search_dump import write_member_dump
from priml.baselines.sudoku.views import NINE_VIEWS, View, validate_views
from priml.runtime import SingleProcess


logger = logging.getLogger(__name__)


class Member(NamedTuple):
    """One committee member: a checkpoint reference under a recorded view.

    Attributes:
      checkpoint: Checkpoint path; a relative logical path is resolved beneath
        the owner's ``base_dir`` at run time. Either checkpoint schema loads
        (:func:`~priml.baselines.sudoku.runner.load_eval_weights`).
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
        root. Required (the launcher stamps the factory name when empty)."""

        doc: str = ""
        """Free-text description retained by the launch harness."""

        base_dir: Path | str | None = None
        """Scratch root; ``None`` resolves beneath ``/opt/scratch`` at
        construction."""

        members: tuple[Member, ...] = field(
            default_factory=lambda: nine_view_members(""),
        )
        """Ordered committee members (the order IS the tie-break order).
        The default is the frozen nine-view ladder with no checkpoint; every
        member needs one before construction. Member checkpoints are relative logical paths resolved
        beneath ``base_dir``."""

        model: LatentRecursion.Config = field(default_factory=LatentRecursion.Config)
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
        if not all(member.checkpoint for member in config.members):
            raise ValueError("every agreement-lock member needs a checkpoint.")
        validate_views(member.view for member in config.members)
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
        require_population(
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
        write_json(self._path(config.metrics_path), metrics)
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
        model = eval_model(
            config.model,
            self._path(member.checkpoint),
            self.device,
        )
        try:
            rows, media, labels = search_pass(
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
            release()
        write_member_dump(
            prepared(self._path(config.member_dump_path, index=index)),
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
        return fill_template(
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


def _survivor_indices(
    base_indices: tuple[int, ...],
    survivors: Tensor,
) -> tuple[int, ...]:
    """Global test indices for the disagreement subset."""
    positions = from_plain(survivors.tolist(), list[int])
    if base_indices:
        return tuple(base_indices[position] for position in positions)
    return tuple(positions)


def _write_lock_dump(
    path: Path,
    *,
    final: Tensor,
    labels: Tensor,
    locked: Tensor,
    survivors: Tensor,
) -> None:
    """Archive the agreement-lock outcome for offline diagnosis."""
    destination = prepared(path)
    temporary = destination.with_name(f"{destination.name}.tmp.{os.getpid()}.npz")
    np.savez_compressed(
        temporary,
        final_grids=final.numpy().astype(np.uint8),
        labels=labels.numpy().astype(np.uint8),
        locked=locked.numpy(),
        survivors=survivors.numpy().astype(np.int64),
    )
    temporary.replace(destination)
