"""The verifier-locked progressive sieve.

:class:`SieveEval`:

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
original run died with a SIGSEGV when it did).

R0 is :func:`det_pass`, which wires to
:meth:`~priml.baselines.sudoku.routed_act_step.RoutedActTrainStep.eval_loss`,
the train step's conditional-depth evaluation (per-row release at q >= +2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Self, override

import json
import logging
import math
import os
import time

from configgle import Fig, Makeable, Makes
from torch import Tensor
from torch._inductor import config as inductor_config
from treekle import from_plain

import numpy as np
import torch

from priml.baselines.sudoku.act import FeedbackCarry
from priml.baselines.sudoku.agreement_lock import modal_grid_predictions
from priml.baselines.sudoku.hps_eval import search_pass
from priml.baselines.sudoku.hypotheses import HypothesisRecursion
from priml.baselines.sudoku.latent_recursion import LatentRecursion
from priml.baselines.sudoku.puzzle_data import PuzzleDataset
from priml.baselines.sudoku.routed_act_step import HaltRelease, RoutedActTrainStep
from priml.baselines.sudoku.runner import (
    eval_model,
    fill_template,
    load_eval_weights,
    prepared,
    release,
    require_population,
    to_device,
    write_json,
)
from priml.baselines.sudoku.search import HpsSearch
from priml.baselines.sudoku.verifier import CommitteeLock, VerifierAcceptor
from priml.baselines.sudoku.verifier_fit import VerifierFit
from priml.baselines.sudoku.views import (
    IDENTITY_DIGITS,
    NINE_VIEWS,
    View,
    validate_views,
)
from priml.paths import resolve_working_dir
from priml.runtime import SingleProcess
from priml.train.parallelism import NoParallel
from priml.train.train_loop import EvalTimeLimitError


logger = logging.getLogger(__name__)


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


def det_pass(step: RoutedActTrainStep, dataset: PuzzleDataset) -> DetPassResult:
    """Conditional-depth deterministic pass over a dataset's eval split.

    The det gate / sieve round-0 pass: every row rolls until its halt logit
    clears the step's ``release`` threshold (+2, the registered release bar)
    or the ACT depth cap. Wires to the train step's own evaluation seam
    (:meth:`RoutedActTrainStep.eval_loss` -- EMA swap, compile guard, and the
    per-row release rollout all live there); this function only collects the
    per-puzzle outputs. Batch pooling (the recorded pass used one pooled
    5,000-row eval batch) is the dataset config's ``eval_batch_size``.

    Args:
      step: Step built with a ``release`` policy, its eval weights loaded.
      dataset: The puzzles to pass over, in eval order.

    Returns:
      result: :class:`DetPassResult` over all valid eval rows, in arrival
        order (CPU tensors; labels are carried for post-hoc scoring only).

    """
    grid_parts: list[Tensor] = []
    q_parts: list[Tensor] = []
    media_parts: list[Tensor] = []
    label_parts: list[Tensor] = []
    for raw_batch in dataset.eval_dataloader():
        batch = step.preprocess_batch(dict(raw_batch))
        media = batch["media"]
        assert isinstance(media, Tensor)
        label = batch["label"]
        assert isinstance(label, Tensor)
        valid_count = batch["valid_count"]
        assert isinstance(valid_count, int)
        if valid_count == 0:
            continue
        packed = step.eval_loss(**batch)["model"]
        q_parts.append(packed[:valid_count, 0].float().cpu())
        grid_parts.append(packed[:valid_count, 1:].to(torch.int64).cpu())
        media_parts.append(media[:valid_count].to(torch.int64).cpu())
        label_parts.append(label[:valid_count].to(torch.int64).cpu())
    if not grid_parts:
        raise ValueError("det_pass found no valid eval rows.")
    return DetPassResult(
        grids=torch.cat(grid_parts),
        q_halt=torch.cat(q_parts),
        media=torch.cat(media_parts),
        label=torch.cat(label_parts),
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
            # Device PINNED: priml's ``None`` falls back to the CPU when no GPU is
            # visible, a silent 100x-slow run.
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
        """The frozen generator checkpoint every round evaluates, required: a
        chain rung's final checkpoint (either checkpoint schema). A relative
        logical path resolves beneath ``base_dir``."""

        verifier_checkpoints: tuple[str | Path, ...] = ()
        """The committee member checkpoints, pushed into
        ``acceptor.checkpoint_paths`` in ``finalize()`` when the acceptor is
        a :class:`VerifierAcceptor` config (the source of truth for the
        default committee). Relative logical paths resolve beneath
        ``base_dir``."""

        model: LatentRecursion.Config = field(default_factory=LatentRecursion.Config)
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
        proven operating point, not trimmed: raising C to 7 (not K) converted
        every residual coverage-exhaustion miss of the ratified default. The
        acceptance policy is the unchanged ``search`` policy."""

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
        (the train step's contract)."""

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
        if not str(config.checkpoint_path):
            raise ValueError(
                "SieveEval.Config.checkpoint_path is required: the frozen "
                "generator a chain rung trained.",
            )
        validate_views(config.views)
        if config.evaluation_count < 1:
            raise ValueError("evaluation_count must be positive.")
        self.config = config
        self.runtime = config.runtime.make()
        self.runtime.initialize()
        self.device = self.runtime.device
        # Process-global TorchInductor flag; must be set before any model
        # construction binds torch.compile (the train step's contract).
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
        require_population(grids.shape[0], n_puzzles, label="sieve round 0")
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
            # (the original run died with a SIGSEGV when it did).
            acceptor = cfg.acceptor.make()
            accepted = self._lock(acceptor, grids, media)
            del acceptor
            release()
            collected.append((survivors.clone(), grids))
            final_grids[survivors[accepted]] = grids[accepted]
            locked[survivors[accepted]] = True
            on_cuda = self.device.type == "cuda"
            peak_mem_gb = (
                torch.cuda.max_memory_allocated(self.device) / 1e9 if on_cuda else 0.0
            )
            stats: dict[str, float | int | str] = {
                "round": round_name,
                "entering": survivors.shape[0],
                "generated": grids.shape[0],
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
            "eval/tail_modal_count": survivors.shape[0],
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
        write_json(self._path(cfg.metrics_path), metrics)
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
            return self._det_round(cfg.model, cfg.checkpoint_path)
        if round_index <= len(cfg.views):
            return self._hps_round(
                cfg.views[round_index - 1],
                cfg.search,
                survivors,
                model=cfg.model,
                checkpoint=cfg.checkpoint_path,
            )
        return self._hps_round(
            View("tail_escalated", IDENTITY_DIGITS, False),
            self._tail_search_config(),
            survivors,
            model=cfg.model,
            checkpoint=cfg.checkpoint_path,
        )

    # Wires to the train step's conditional-depth evaluation via :func:`~det_pass`;
    # batch pooling is the dataset config's ``eval_batch_size``.
    def _det_round(
        self,
        model: LatentRecursion.Config,
        checkpoint: str | Path,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """R0: conditional-halt deterministic pass over the full population."""
        cfg = self.config
        # A throwaway evaluation step, not a loop: no global reseed, since the
        # checkpoint overwrites the weights.
        step = RoutedActTrainStep.Config()
        step.model = model
        step.emulate_precision_casts = cfg.emulate_precision_casts
        # R0 rolls to the same ACT depth and compute dtype as the HPS
        # rounds (one frozen rollout contract across the whole sieve).
        step.pool.max_steps = cfg.search.max_act_steps
        step.dtype_autocast = cfg.search.dtype_autocast
        feedback = step.pool.feedback = FeedbackCarry.Config()
        feedback.givens = (2, cfg.dataset.spec.vocab_size - 1)
        halt_release = step.release = HaltRelease.Config()
        halt_release.threshold = cfg.det_halt_threshold
        placement = step.parallelism = NoParallel.Config()
        placement.device = self.device
        dataset = cfg.dataset.copy_tree()
        dataset.eval_num_instances = cfg.evaluation_count
        dataset.eval_instance_indices = ()
        det_step = step.make()
        try:
            det_step.model.load_state_dict(
                load_eval_weights(self._path(checkpoint), device=det_step.device),
            )
            result = det_pass(det_step, dataset.make())
        finally:
            del det_step
            release()
        return result.grids, result.media, result.label

    def _hps_round(
        self,
        view: View,
        search: HpsSearch.Config,
        survivors: Tensor,
        *,
        model: LatentRecursion.Config,
        checkpoint: str | Path,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """One learned-q HPS round over the surviving puzzles only."""
        cfg = self.config
        generator = eval_model(model, self._path(checkpoint), self.device)
        try:
            rows, media, labels = search_pass(
                model=generator,
                dataset=self._survivor_dataset(survivors).make(),
                view=view,
                search=search,
                device=self.device,
                deadline_seconds=cfg.max_round_eval_seconds,
                label=f"round {view.name}",
            )
        finally:
            del generator
            release()
        grid_len = media.shape[-1]
        return rows[:, -grid_len:].to(torch.int64), media, labels

    def _survivor_dataset(self, survivors: Tensor) -> PuzzleDataset.Config:
        """Restrict the eval split to the surviving puzzles, ascending."""
        dataset = self.config.dataset.copy_tree()
        # The sieve evaluates the ordered prefix, so survivor positions ARE
        # global test indices; mask-filtering an ``arange`` keeps them ascending.
        dataset.eval_instance_indices = tuple(
            from_plain(survivors.to(torch.int64).tolist(), list[int]),
        )
        dataset.eval_num_instances = None
        return dataset

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
        return fill_template(
            template,
            base_dir=self._scratch,
            experiment_name=self.config.experiment_name,
            **extra,
        )


class StaircaseSieveEval(SieveEval):
    """A sieve whose first three rounds read a cheaper, stronger generator.

    The staircase puts a winner-take-all-trained generator
    (:class:`~priml.baselines.sudoku.hypotheses.HypothesisRecursion`, ``hypothesis_model``) in front
    of the :class:`SieveEval` view ladder and REPLACES its deterministic pass:

    - R0 ``hypothesis_det_cond_halt``: the conditional-halt deterministic pass on
      the staircase generator, over the whole population.
    - R1 ``hypothesis_qsel``: on the survivors, roll hypothesis 0 plus
      ``qsel_hypotheses`` fresh latent draws to full depth and emit each
      puzzle's highest-q grid (:func:`qsel_grids`).
    - R2 ``hypothesis_canonical_hps``: the view rounds' learned-q search, identity
      view, on the staircase generator.
    - Then the inherited view rounds and escalated tail on ``model``.

    Every round only changes who generates; the committee lock alone decides
    what is accepted, so soundness is the lock's, unchanged.
    """

    class Config(Makes["StaircaseSieveEval"], SieveEval.Config):
        """The sieve plus the staircase generator and its selection round."""

        hypothesis_model: HypothesisRecursion.Config = field(
            default_factory=HypothesisRecursion.Config,
        )
        """Staircase generator architecture; its hypothesis count must match
        the checkpoint's stored latent rows."""

        hypothesis_checkpoint_path: str | Path = ""
        """The frozen staircase generator checkpoint, required. A relative
        logical path resolves beneath ``base_dir``."""

        qsel_hypotheses: int = 5
        """Fresh latent draws rolled after hypothesis 0 in the selection
        round; the round costs ``1 + qsel_hypotheses`` full rollouts per
        surviving puzzle."""

        qsel_seed: int = 8_902
        """Draw ``i`` of the selection round seeds a dedicated generator at
        ``qsel_seed + i``; the ambient RNG is never read."""

    config: Config  # pyrefly: ignore[bad-override-mutable-attribute] -- Narrowed to this class's own config; nothing assigns the base's.
    """Narrowed from the parent's declaration for the staircase fields."""

    def __init__(self, config: Config) -> None:
        if config.qsel_hypotheses < 1:
            raise ValueError(
                f"qsel_hypotheses must be >= 1, got {config.qsel_hypotheses}.",
            )
        if not str(config.hypothesis_checkpoint_path):
            raise ValueError(
                "StaircaseSieveEval.Config.hypothesis_checkpoint_path is required.",
            )
        super().__init__(config)

    @override
    def _round_names(self) -> list[str]:
        """Return the staircase rounds, then the view rounds and the tail."""
        return [
            "hypothesis_det_cond_halt",
            "hypothesis_qsel",
            "hypothesis_canonical_hps",
            *super()._round_names()[1:],
        ]

    @override
    def _run_round(
        self,
        round_index: int,
        survivors: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Run a staircase round, or the inherited round it displaced."""
        cfg = self.config
        if round_index == 0:
            return self._det_round(cfg.hypothesis_model, cfg.hypothesis_checkpoint_path)
        if round_index == 1:
            return self._qsel_round(survivors)
        if round_index == 2:
            return self._hps_round(
                View("hypothesis_canonical_hps", IDENTITY_DIGITS, False),
                cfg.search,
                survivors,
                model=cfg.hypothesis_model,
                checkpoint=cfg.hypothesis_checkpoint_path,
            )
        # Round 3 is the parent's first view round: its det pass is replaced.
        return super()._run_round(round_index - 2, survivors)

    def _qsel_round(self, survivors: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """R1: highest-q grid over several latent hypotheses, survivors only."""
        cfg = self.config
        model = eval_model(
            cfg.hypothesis_model,
            self._path(cfg.hypothesis_checkpoint_path),
            self.device,
        )
        assert isinstance(model, HypothesisRecursion)
        dtype = cfg.search.dtype_autocast
        started = time.monotonic()
        grid_parts: list[Tensor] = []
        media_parts: list[Tensor] = []
        label_parts: list[Tensor] = []
        try:
            for raw_batch in self._survivor_dataset(survivors).make().eval_dataloader():
                elapsed = time.monotonic() - started
                if elapsed > cfg.max_round_eval_seconds:
                    raise EvalTimeLimitError(
                        f"round hypothesis_qsel exceeded its eval budget ({elapsed:.0f}s "
                        f"> {cfg.max_round_eval_seconds:.0f}s).",
                    )
                batch = to_device(raw_batch, self.device)
                valid_count = batch["valid_count"]
                if valid_count == 0:
                    continue
                with (
                    torch.inference_mode(),
                    torch.amp.autocast(
                        device_type=self.device.type,
                        dtype=dtype,
                        enabled=dtype is not None,
                        cache_enabled=False,
                    ),
                ):
                    grids, _ = qsel_grids(
                        model,
                        batch["media"],
                        puzzle_identifiers=(
                            batch["puzzle_identifiers"]
                            if model.puzzle_emb is not None
                            else None
                        ),
                        hypotheses=cfg.qsel_hypotheses,
                        seed=cfg.qsel_seed,
                        max_steps=cfg.search.max_act_steps,
                    )
                grid_parts.append(grids[:valid_count].cpu())
                media_parts.append(batch["media"][:valid_count].to(torch.int64).cpu())
                label_parts.append(batch["label"][:valid_count].to(torch.int64).cpu())
        finally:
            del model
            release()
        if not grid_parts:
            raise ValueError("round hypothesis_qsel found no valid eval rows.")
        return torch.cat(grid_parts), torch.cat(media_parts), torch.cat(label_parts)


def qsel_grids(
    model: HypothesisRecursion,
    media: Tensor,
    *,
    puzzle_identifiers: Tensor | None,
    hypotheses: int,
    seed: int,
    max_steps: int,
) -> tuple[Tensor, Tensor]:
    """Roll several latent hypotheses per puzzle; keep each puzzle's highest-q grid.

    Hypothesis 0 starts from the deployed fast latent; hypothesis ``i + 1``
    from the first row of :meth:`HypothesisRecursion.fresh_fast_states` seeded
    ``seed + i``. Each rolls the train step's fixed-depth eval recurrence:
    fresh slow latent, decoded-grid feedback with givens clamped. Selection
    is label-free: the final halt logit.

    Args:
      model: The staircase generator, eval weights loaded.
      media: ``[B, G]`` puzzle boards.
      puzzle_identifiers: ``[B]`` ids when the model has a puzzle table.
      hypotheses: Fresh draws after hypothesis 0.
      seed: Base seed of the fresh draws.
      max_steps: ACT depth of every rollout.

    Returns:
      grids: ``[B, G]`` the selected decoded grids.
      q_halt: ``[B]`` float32 final halt logit of each selected grid; ties go
        to the lowest hypothesis.

    """
    rows = media.shape[0]
    _, deployed = model.init_z(rows)
    starts = [deployed]
    starts.extend(
        model.fresh_fast_states(rows, seed=seed + draw)[:, 0].contiguous()
        for draw in range(hypotheses)
    )
    given = (media >= 2) & (media <= 10)
    grids: list[Tensor] = []
    halts: list[Tensor] = []
    for start in starts:
        z_slow, _ = model.init_z(rows)
        z_fast = start
        feedback = media
        out: dict[str, Tensor] = {}
        for _ in range(max_steps):
            out = model.act_step(media, z_slow, z_fast, puzzle_identifiers, feedback)
            z_slow, z_fast = out["z_slow"], out["z_fast"]
            feedback = torch.where(given, media, out["logits"].argmax(dim=-1))
        grids.append(out["logits"].argmax(dim=-1))
        halts.append(out["q_halt"].float())
    stacked_q = torch.stack(halts)
    # ``argmax`` returns the first maximum, so hypothesis 0 wins exact ties.
    selected = stacked_q.argmax(dim=0, keepdim=True)
    stacked = torch.stack(grids)
    picked = stacked.gather(0, selected.unsqueeze(-1).expand(1, -1, stacked.shape[-1]))
    return picked[0], stacked_q.gather(0, selected)[0]


class FittedSieve:
    """Fit any missing committee member, then run the sieve under that committee.

    The member checkpoints are each fit's final checkpoint (the
    :class:`VerifierFit` layout); a member whose checkpoint exists is reused,
    so a rerun after a crash resumes at the first missing member.
    """

    class Config(Fig["FittedSieve"]):
        """The committee members' fits and the sieve that locks with them."""

        experiment_name: str = ""
        """Run identity, stamped onto the sieve. Required."""

        base_dir: Path | str | None = None
        """Scratch root, pushed into every fit and the sieve left unset."""

        members: list[VerifierFit.Config] = field(
            default_factory=list[VerifierFit.Config],
        )
        """One fit per committee member, in committee order."""

        sieve: SieveEval.Config = field(default_factory=SieveEval.Config)
        """The sieve; its ``verifier_checkpoints`` are the members'."""

        @override
        def finalize(self) -> Self:
            for child in (*self.members, self.sieve):
                if child.base_dir is None:
                    child.base_dir = self.base_dir
            for member in self.members:
                if member.device is None:
                    member.device = str(self.sieve.runtime.device)
            self.sieve.experiment_name = self.experiment_name
            self.sieve.verifier_checkpoints = tuple(
                str(member.run_dir).format(experiment_name=member.experiment_name)
                + f"/checkpoints/step_{member.max_steps:08d}.pt"
                for member in self.members
            )
            return super().finalize()

    def __init__(self, config: Config) -> None:
        if not config.experiment_name:
            raise ValueError("FittedSieve.Config.experiment_name is required.")
        if not config.members:
            raise ValueError("the committee needs at least one member fit.")
        self.config = config

    def run(self, *args: str) -> dict[str, object]:
        """Fit every member without a checkpoint, then run the sieve.

        Args:
          *args: Ignored (logged); accepted so launcher passthrough CLI args
            (a launcher calls ``job.run(*unparsed)``) never TypeError.

        Returns:
          metrics: The sieve's metrics.

        """
        if args:
            logger.info("Ignoring launcher passthrough args: %r.", args)
        cfg = self.config
        for member, checkpoint in zip(
            cfg.members,
            cfg.sieve.verifier_checkpoints,
            strict=True,
        ):
            base = member.base_dir if member.base_dir is not None else "/opt/scratch"
            if resolve_working_dir(base, str(checkpoint)).exists():
                logger.info("member %s already fit; reusing.", member.experiment_name)
                continue
            member.make().run()
            release()
        return cfg.sieve.make().run()


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


def _write_sieve_dump(
    path: Path,
    *,
    final_grids: Tensor,
    labels: Tensor,
    locked: Tensor,
    tail: Tensor,
) -> None:
    """Archive the sieve outcome for offline diagnosis."""
    destination = prepared(path)
    temporary = destination.with_name(f"{destination.name}.tmp.{os.getpid()}.npz")
    np.savez_compressed(
        temporary,
        final_grids=final_grids.numpy().astype(np.uint8),
        labels=labels.numpy().astype(np.uint8),
        locked=locked.numpy(),
        tail=tail.numpy().astype(np.int64),
    )
    temporary.replace(destination)
