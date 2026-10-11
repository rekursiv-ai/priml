"""End-to-end reproduction pipelines: train a generator, then lock the full set.

:class:`Reproduction` is the staged from-scratch runner behind the two
end-to-end reproductions (exp008 and exp012).
"""

from __future__ import annotations

from dataclasses import field, fields
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, Self, TypedDict, override

import hashlib
import logging
import math
import time

from configgle import DataclassLike, Fig, Makeable
from torch import Tensor
from torch._inductor import config as inductor_config
from treekle import ReadError, json_loads

import torch

from priml.baselines.sudoku.agreement_lock import AgreementLockEval, Member
from priml.baselines.sudoku.harvest import Harvest
from priml.baselines.sudoku.runner import (
    eval_model,
    fill_template,
    load_eval_weights,
    release,
    to_device,
    write_json,
)
from priml.baselines.sudoku.search import GridAcceptor, HpsSearch, run_search
from priml.baselines.sudoku.sieve import SieveEval, det_pass
from priml.baselines.sudoku.trainer import Trainer, resolve_run_dir
from priml.baselines.sudoku.verifier import CommitteeLock, VerifierAcceptor
from priml.baselines.sudoku.verifier_fit import VerifierFit
from priml.paths import resolve_working_dir
from priml.runtime import SingleProcess
from priml.train.tracker import WandbTracker


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from priml.baselines.sudoku.latent_recursion import LatentRecursion
    from priml.baselines.sudoku.puzzle_data import PuzzleDataset


logger = logging.getLogger(__name__)


class Reproduction:
    """One end-to-end from-scratch reproduction pipeline of the 100% result.

    One sequential single-GPU job that TRAINS its models from scratch under
    fresh checkpoint identities, then evaluates the frozen full-set protocol
    against them -- mapping the earliest training step at which the method
    reaches dev-100% and full-set-100%. The two shipped variants
    (experiments.py exp008 / exp012) are pure config choices:

    - COMMITTEE (exp008): stage 0 harvests a candidate corpus from a frozen
      source generator and fits the 3-seed verifier committee from scratch;
      then the generator trains with a committee-locked fast-HPS dev screen
      per boundary; the first perfect AND sound screen (zero committee false
      accepts) triggers full-set sieve evals; final always.
      (``screen="committee"``, sieve ``full_eval``.)
    - MULTIVIEW (exp012): one recipe generator trained in checkpoint-cadence
      segments, evaluated under multiple recorded views of each checkpoint;
      a cheap searchless det gate prices every boundary and the nine-view
      agreement-lock dev screen runs only once the gate opens; the first
      perfect dev screen triggers FULL-SET agreement-lock evals at that and
      every later boundary; the final checkpoint is always full-evaluated.
      (``screen="nine_view"``, lock ``full_eval``.)

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
    ``trigger/`` / ``full/`` rows, and ``summary/`` fields at the end. Inner
    runs never open W&B (the segment builder pins ``wandb_project=None``);
    the aggregate ``metrics.json`` under the runner's own run dir is the
    complete record -- W&B is additive. Per-eval metrics and dumps land
    under ``<scratch>/runs/{experiment_name}/dumps/``.
    """

    class Config(Fig["Reproduction"]):
        """Generator recipe and identity, screens, and the full eval."""

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
        the scratch root. Required (the launcher stamps the factory name
        when empty)."""

        doc: str = ""
        """Free-text description; forwarded as the W&B run notes."""

        base_dir: Path | str | None = None
        """Scratch root shared by every stage; ``None`` resolves from the
        environment at construction. Pushed into children left unset."""

        generator: Trainer.Config = field(default_factory=Trainer.Config)
        """The FULL generator training recipe segments derive from (the
        recipe seam experiments.py fills). Its ``max_steps`` is the staged
        horizon; ``total_train_steps`` -- the cosine horizon -- is never
        touched by segmentation. Its ``seed`` also seeds the dedicated
        augmentation stream, so it must be set."""

        generator_name: str = ""
        """Fresh checkpoint identity of the generator. Required."""

        eval_every_steps: int = 1_000
        """Checkpoint/eval cadence in train steps (the trigger-eval knob)."""

        dev_screen_count: int = 5_000
        """Ordered dev puzzles per boundary screen."""

        screen: Literal["nine_view", "committee"] = "nine_view"
        """Per-boundary dev screen kind. ``nine_view``: the agreement-lock
        protocol itself over the dev rows (``full_eval`` must be the lock);
        ``committee``: committee-locked fast HPS -- the sieve's search
        policy and acceptor over the freshly trained verifiers
        (``full_eval`` must be the sieve; entails stage 0). The searchless
        det gate below prices the nine_view screens; committee screens are
        cheap enough to run at every boundary and never gate. The first
        perfect dev screen -- perfect AND zero false accepts for committee
        screens -- latches full-set evals at that and every later boundary,
        and the final boundary always full-evaluates."""

        screen_gate_det_accuracy: float = 0.98
        """Cheap det-proxy gate for the nine_view screens: the dev screen
        runs only once the conditional-halt det dev accuracy crosses this
        fraction. Measured on the recorded run: below the gate, screens cost 30-90
        min (hundreds of failing roots search at full budget) yet cannot
        read a perfect count -- det below the gate leaves >= 100 dev root
        failures and the measured HPS root-fix rate (~87%) leaves >= 13
        expected misses, so no pre-gate screen can trigger. Above the gate
        screens cost minutes. The final boundary always screens."""

        harvest_source_checkpoint: str | Path = ""
        """Frozen generator checkpoint stage 0 rolls out (committee screens
        only, where it is required): exp008 reads exp006's final checkpoint,
        so train exp006 first; generator transfer is proven, so any
        recipe-class generator works. A relative logical path resolves
        beneath ``base_dir``."""

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
        member checkpoints are ignored); the det pass is structurally
        EXCLUDED (no verifier -- canonical det grids carry a measured
        identical-wrong channel, unsound without one). Sieve: its ``search``
        policy also drives the committee dev screens and its ``acceptor`` is
        copied as the screens' committee lock; the det pass IS included,
        behind the verifier lock. ``evaluation_count`` is the FULL-SET
        population; dev screens use ``dev_screen_count``."""

        wandb_project: str | None = None
        """W&B project for the SINGLE outer tracker (the trainer
        convention: set -> W&B on). Sub-runs never open W&B; the runner
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
        if not config.generator_name:
            raise ValueError("generator_name must name a fresh checkpoint identity.")
        if config.generator.seed is None:
            raise ValueError(
                "the generator recipe needs a seed: it also seeds the dedicated "
                "augmentation stream that makes resumed segments replay exactly.",
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
                    "checkpoint (train the recipe generator first).",
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
        if cfg.screen == "committee":
            self._run_stage0(outer, stages, stage_seconds)
        total = int(cfg.generator.max_steps)
        gate_open = False
        for boundary in _eval_boundaries(cfg.eval_every_steps, total):
            seconds, trained = self._train_segment(boundary)
            stage_name = f"train_to_{boundary}"
            seconds = self._record_stage_seconds(
                stage_name,
                seconds,
                completed=trained,
                stage_seconds=stage_seconds,
            )
            stages.append(
                {
                    "stage": stage_name,
                    "checkpoint": cfg.generator_name,
                    "seconds": seconds,
                },
            )
            outer.forward_segment(self._segment_metrics_path(), boundary)
            if cfg.screen != "committee" and not gate_open and boundary != total:
                det_exact, det_of, det_seconds = self._det_gate(boundary)
                gate_open = det_exact >= cfg.screen_gate_det_accuracy * det_of
                screens.append(
                    {
                        "step": boundary,
                        "det_gate_exact": det_exact,
                        "of": det_of,
                        "seconds": det_seconds,
                        "gate": "opened" if gate_open else "closed",
                    },
                )
                outer.log(
                    {"det_gate_exact": det_exact, "gate_open": int(gate_open)},
                    boundary,
                    prefix="screen/",
                )
                if not gate_open:
                    continue
            exact, of, false_accepts, seconds = self._dev_screen(boundary)
            row: dict[str, object] = {"step": boundary, "exact": exact, "of": of}
            if false_accepts is not None:
                row["false_accepts"] = false_accepts
            row["seconds"] = seconds
            screens.append(row)
            outer.log(
                {key: value for key, value in row.items() if key != "step"},
                boundary,
                prefix="screen/",
            )
            if (
                first_dev == 0
                and exact == cfg.dev_screen_count
                and (false_accepts or 0) == 0
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
        write_json(self._metrics_path(), results)

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
        return fill_template(
            cfg.metrics_path,
            base_dir=base,
            experiment_name=cfg.experiment_name,
        )

    def _progress_path(self) -> Path:
        """Durable completed-stage timing used across crash resumes."""
        metrics_path = self._metrics_path()
        return metrics_path.with_name(f"{metrics_path.stem}.progress.json")

    def _load_stage_seconds(self) -> dict[str, float]:
        """Load completed-stage seconds recorded by an earlier, crashed run."""
        progress_path = self._progress_path()
        if not progress_path.is_file():
            return {}
        try:
            progress = json_loads(progress_path.read_text(), _ReproductionProgress)
        except ReadError as error:
            raise ValueError(
                f"invalid reproduction progress: {progress_path}.",
            ) from error
        return progress["stage_seconds"]

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
        write_json(
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
            recorded = json_loads(manifest_path.read_text(), _HarvestSource)[
                "source_checkpoint"
            ]
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
                    f"{configured!r}; move the stale corpus aside "
                    "or point the harvest at a fresh out_dir.",
                )
            logger.info(
                "harvest manifest already present under %s; reusing the corpus.",
                out_dir,
            )
            return out_dir, 0.0
        started = time.monotonic()
        out_dir = harvest.make().run()
        release()
        return out_dir, time.monotonic() - started

    def _verifier_fit_config(self, index: int, harvest_dir: Path) -> VerifierFit.Config:
        """One committee member's fit recipe under its fresh identity/seed."""
        fit = self.config.verifier.copy_tree()
        fit.experiment_name = self.config.verifier_names[index]
        fit.seed = index  # The frozen committee trains at seeds 0/1/2.
        fit.dataset.harvest_dir = harvest_dir
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
        release()
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

    def _train_segment(self, boundary: int) -> tuple[float, bool]:
        """Train the generator's segment ending at ``boundary`` (resume-aware)."""
        cfg = self.config
        segment = _generator_segment(
            cfg.generator.copy_tree(),
            study_name=cfg.study_name,
            experiment_name=cfg.generator_name,
            num_steps_eval=cfg.eval_every_steps,
            max_steps=boundary,
        )
        return _run_training(segment)

    def _segment_metrics_path(self) -> Path:
        """Return the generator's own run-dir metrics file (forwarded)."""
        run_dir = resolve_run_dir(self._generator_base(), self.config.generator_name)
        return run_dir / "metrics.json"

    def _checkpoint_path(self, step: int) -> Path:
        """Resolve the generator's checkpoint path at one boundary step."""
        run_dir = resolve_run_dir(self._generator_base(), self.config.generator_name)
        return run_dir / "checkpoints" / f"step_{step:08d}.pt"

    def _generator_base(self) -> Path:
        """Return the generator subtree's scratch root (defaults to ``/opt/scratch``)."""
        base = self.config.generator.base_dir
        return Path(base) if base is not None else Path("/opt/scratch")

    def _det_gate(self, step: int) -> tuple[int, int, float]:
        """Run the cheap det gate at one boundary; (exact, of, seconds)."""
        cfg = self.config
        return _det_gate_screen(
            recipe=cfg.generator,
            experiment_name=f"{cfg.experiment_name}_detgate_{step}",
            checkpoint_path=self._checkpoint_path(step),
            count=cfg.dev_screen_count,
        )

    def _dev_screen(self, step: int) -> tuple[int, int, int | None, float]:
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
        exact, false_accepts, seconds = self._committee_screen(step)
        return exact, cfg.dev_screen_count, false_accepts, seconds

    # The committee is the CANDIDATE ACCEPTANCE inside the fast search (``accept_fn`` --
    # the predicate engine with the q-halt@8 early exit); no rule predicate and no
    # labels anywhere on the decision path. Returns (exact, false_accepts, seconds).
    def _committee_screen(self, step: int) -> tuple[int, int, float]:
        """One committee-locked fast-HPS dev screen at one boundary step."""
        cfg = self.config
        assert isinstance(cfg.full_eval, SieveEval.Config)  # __init__ validated.
        started = time.monotonic()
        acceptor = self._screen_acceptor_config().make()
        model = eval_model(
            cfg.generator.model,
            self._checkpoint_path(step),
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
            release()
        exact = int(exact_rows.sum())
        false_accepts = int((accepted & ~exact_rows).sum())
        return exact, false_accepts, time.monotonic() - started

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
            release()
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
        checkpoint = _checkpoint_template(cfg.generator_name, step)
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
        release()
        return metrics, time.monotonic() - started

    # ``copy_tree`` (never ``SieveEval.Config().update``): the class-preserving copy
    # lets a sieve SUBCLASS config -- extra fields and overridden round seams -- survive
    # the per-eval rebuild.
    def _sieve_run_config(self, step: int) -> SieveEval.Config:
        """One per-eval clone of the sieve protocol at one boundary step."""
        cfg = self.config
        assert isinstance(cfg.full_eval, SieveEval.Config)  # __init__ validated.
        run = cfg.full_eval.copy_tree()
        run.checkpoint_path = _checkpoint_template(cfg.generator_name, step)
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


def _generator_segment(
    recipe: Trainer.Config,
    *,
    study_name: str,
    experiment_name: str,
    num_steps_eval: int,
    max_steps: int,
) -> Trainer.Config:
    """One training segment of ``recipe`` under a fresh identity."""
    cfg = recipe
    cfg.study_name = study_name
    cfg.experiment_name = experiment_name
    cfg.num_steps_eval = num_steps_eval
    cfg.max_steps = max_steps
    cfg.halt_exploration_seed = 0  # Explicit pin (the recipe's default).
    # A dedicated augmentation stream seeded like the trainer: a resumed
    # segment then draws exactly what one uninterrupted run would have.
    cfg.dataset.augment_seed = cfg.seed
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
        release()
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
        release()
    exact = (result.grids == result.label).all(dim=-1)
    return int(exact.sum()), int(exact.shape[0]), time.monotonic() - started


# The learned-q policy runs when ``accept_fn`` is None (the single-view dev screen); a
# grid predicate switches to the committee-locked fast engine. Acceptance is label-free;
# labels join post hoc for the screen metrics only.
def _screen_pass(
    *,
    model: LatentRecursion,
    dataset: PuzzleDataset,
    search: HpsSearch.Config,
    device: torch.device,
    accept_fn: GridAcceptor | None = None,
) -> tuple[Tensor, Tensor]:
    """One search pass over the eval split; per-row screen masks."""
    accepted_parts: list[Tensor] = []
    exact_parts: list[Tensor] = []
    for raw_batch in dataset.eval_dataloader():
        batch = to_device(raw_batch, device)
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

    def forward_segment(self, metrics_path: Path, step: int) -> None:
        """Forward a finished segment's last eval row from its metrics file.

        Args:
            metrics_path: Segment metrics JSON path.
            step: W&B step assigned to the forwarded row.

        The rows carry their own ``eval/`` key prefixes (the trainer's file
        tracker keeps the last write -- at cadence-length segments the
        boundary row), so the segment trajectory stays step-continuous
        across segments.

        """
        if self._tracker is None or not metrics_path.exists():
            return
        rows = json_loads(metrics_path.read_text(), dict[str, object])
        if rows:
            self.log(rows, step)

    def close(self) -> None:
        """Finish the tracker (idempotent by tracker contract)."""
        if self._tracker is not None:
            self._tracker.close()


class _ReproductionProgress(TypedDict):
    """The durable completed-stage timing file."""

    schema_version: Literal[1]
    stage_seconds: dict[str, float]


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
