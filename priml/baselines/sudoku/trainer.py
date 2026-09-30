"""Single-GPU TRM trainer: pool-based ACT training, eval, and run plumbing.

One :class:`Trainer` merges the internal train loop, the TRM ACT train step
(atomic mode), the feedback-recurrence and CSP-auxiliary-loss extensions, and
the conditional-depth eval into a single class, plus the supporting pieces it
needs (optimizers, EMA, checkpointing, trackers, runtime, seeding) so the
module is self-contained.

Each ``train_step()`` call (atomic ACT scheduling):

  1. Slots the incoming batch into halted pool positions (``torch.where``);
     active slots keep their carried ``z_slow``/``z_fast``/feedback state.
  2. Forwards the full pool (one multi-cycle forward), consuming the pool's
     decoded-grid feedback.
  3. Computes the loss, backprops, clips, applies the cosine LR scale, and
     steps the optimizers.
  4. Updates the EMA shadow, the latent carry, and the per-slot halt mask.

Order- and name-sensitive invariants (fidelity contracts -- do not reorder):

* Loss composition: ``lm + q_halt_weight * q_bce`` FIRST, then
  ``+ csp_loss_weight * csp``.
* Step tail: backward -> clip (body params, foreach global norm) -> lr_scale
  -> AdamW.step -> Muon.step -> sparse puzzle-embedding SignSGD step ->
  zero_grad -> EMA update -> carry/halt update (halt-RNG draws).
* Halt-exploration draws per step: ``rand(bs)`` then ``randint(bs)`` from a
  dedicated generator (seed 0). Feedback-scramble draws per step:
  ``rand(bs)`` -> ``rand(bs, 81)`` -> ``randint(2, 11, (bs, 81))`` from a
  second dedicated generator (seed 0).
* Optimizer routing is name-substring based over ``model.named_parameters()``
  order: ``puzzle_emb`` -> SignSGD (sparse rows); names containing ``embed``
  or ``head`` (and every param with ndim < 2) -> AdamW (betas (0.9, 0.95)
  hardcoded); remaining 2D weights -> Muon (``ensemble_dims=0``), 3D ->
  Muon (``ensemble_dims=1``). Parameter names are a contract with trm.py.
* ``torch._inductor.config.emulate_precision_casts`` is set BEFORE model
  construction; when set, eval stays compiled (see
  ``Trainer._eval_compile_disabled``).

Checkpoint schema (``Trainer.state_dict()``, written as
``step_<n:08d>.pt``)::

    {"step":    {"model", "optimizers", "optimizer_puzzle_emb", "ema",
                 "global_step", "halt_rng", "scramble_rng"},
     "dataset": {"train_epochs"},
     "metrics": {},
     "epoch":   int,
     "rng":     {"python", "torch", "cuda"?}}

The ACT pool (carried latents / halt mask / feedback grids) is deliberately
NOT checkpointed: it is in-flight state bound to the specific batches being
processed, and resume continues with the next batch rather than replaying the
interrupted ones.

Output layout: ``<scratch>/runs/{experiment_name}/`` holding
``checkpoints/step_<n:08d>.pt``
and ``metrics.json`` (see :func:`resolve_run_dir`; ``/opt/scratch`` is the
default root).

Downstream seams: :meth:`Trainer.run` trains to the stop condition;
:meth:`Trainer.evaluate` scores the current (possibly resumed) weights over
the eval split and publishes the result; :attr:`Trainer.ema_shadow` exposes
the name-keyed EMA weights (checkpoint-upload contract); ``eval_loss`` /
``preprocess_batch`` are the per-batch eval seams.

``trainer_test.py`` replays frozen trajectories of every recipe bit-for-bit.
"""

from __future__ import annotations

from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    NotRequired,
    Self,
    TypedDict,
    cast,
    override,
)

import difflib
import logging
import math
import re
import time

from configgle import Fig
from torch import Tensor, nn

import torch
import torch._inductor.config

from priml.baselines.sudoku.puzzle_data import PuzzleDataset
from priml.baselines.sudoku.trm import TRM
from priml.math.seed import (
    RngState,
    get_rng_state,
    set_rng_state,
    set_seed_local,
)
from priml.optimizers import lr_scale
from priml.optimizers.muon import Muon
from priml.optimizers.sign_sgd import SignSGD
from priml.runtime import SingleProcess
from priml.train.checkpointer import Checkpointer
from priml.train.ema import EMA
from priml.train.grad_clip import clip_grad_norm_, total_grad_norm
from priml.train.tracker import FileTracker, WandbTracker, scalar_metrics


if TYPE_CHECKING:
    from collections.abc import Generator, Iterable, Iterator, Mapping

    from priml.baselines.sudoku.puzzle_data import PuzzleBatch
    from priml.baselines.sudoku.puzzle_spec import SudokuSpec
    from priml.train.custom_types import TrainStepOutput


__all__ = [
    "Trainer",
    "recipe_checkpointer",
    "recipe_ema",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Seeding and RNG state (subset of the internal seed utilities).
# ---------------------------------------------------------------------------


def recipe_ema() -> EMA.Config:
    """Build the recipe EMA, with every priml default pinned.

    priml defaults to a module shadow, tracks buffers, and skips the warmup
    seed. This recipe needs a name-keyed dict (the checkpoint-upload contract),
    no buffer shadowing (the TRM buffers are derived or sparse-master state),
    and a reference-style seed at the warmup boundary. Each divergence changes
    eval numerics while leaving the state_dict intact, so build every EMA from
    this factory rather than from a bare ``EMA.Config()``.

    Returns:
      config: A recipe-faithful EMA config.

    """
    return EMA.Config(
        shadow_kind="param_dict",
        track_buffers=False,
        warmup_seed=True,
    )


def recipe_checkpointer() -> Checkpointer.Config:
    """Build the recipe checkpointer, with priml's retention defaults pinned.

    priml defaults ``keep_last_n=-1`` and ``keep_every=0`` -- retention OFF, so
    a long run keeps every checkpoint and fills the disk. This recipe keeps a
    rolling window of 8 plus an archival snapshot every 10k steps. Nothing in
    the state_dict reflects the difference, so build every checkpointer from
    this factory rather than a bare ``Checkpointer.Config()``.

    Returns:
      config: A recipe-faithful checkpointer config.

    """
    return Checkpointer.Config(keep_last_n=8, keep_every=10_000)


def sudoku_group_indices(spec: SudokuSpec) -> Tensor:
    """Return the cell indices of every sudoku constraint group.

    Args:
      spec: Dataset-owned grid and constraint-box geometry.

    Returns:
      group_indices: ``[27, 9]`` long tensor by default; rows 0-8 are the 9 grid rows,
        rows 9-17 the 9 columns, rows 18-26 the 9 boxes, each listing the 9
        row-major cell indices (0-80) of that group.

    """
    rows, cols = spec.grid_shape
    box_rows, box_cols = spec.box_shape
    cells = torch.arange(rows * cols).reshape(rows, cols)
    boxes = (
        cells.reshape(rows // box_rows, box_rows, cols // box_cols, box_cols)
        .permute(0, 2, 1, 3)
        .reshape(-1, box_rows * box_cols)
    )
    return torch.cat([cells, cells.T, boxes])


def csp_cardinality_per_sample(
    logits: Tensor,
    labels: Tensor,
    group_indices: Tensor,
    temperature: float = 1.0,
) -> Tensor:
    """Mean squared row/column/box cardinality violation per sample.

    The cardinality loss of arXiv 2307.04895 on the model's per-cell token
    distributions: within every constraint group, each solution symbol's
    probability mass must sum to exactly 1 over the group's 9 cells. The 9
    solution symbols are read from ``labels`` per sample rather than
    hardcoded to tokens 2-10: the on-the-fly augmentation's legacy branch
    permutes token values 1-9 (blank marker included), so which 9 of the 10
    non-pad tokens are digits is sample-dependent. The symbol absent from
    the label (the blank marker) gets target 0. The softmax runs in fp32 so
    bf16-autocast logits do not quantize the small per-group residuals.

    Args:
      logits: ``[B, 81, 11]`` prefix-stripped grid logits.
      labels: ``[B, 81]`` label tokens. Rows holding only ignore labels
        (atomic-mode tail padding) yield an all-zero target and must be
        masked out by the caller.
      group_indices: ``[27, 9]`` cell indices from
        :func:`sudoku_group_indices`, on ``logits``'s device.
      temperature: Softmax temperature for the per-cell distributions (1.0 is
        the published form and the recipe value).

    Returns:
      per_sample: ``[B]`` mean over the 27 groups x 10 non-pad tokens of
        ``(group mass - target) ** 2``.

    References:
      https://arxiv.org/abs/2307.04895
        Yang, Lee & Park. Learning to Solve Constraint Satisfaction Problems
        with Recurrent Transformer. ICLR 2023.

    """
    if logits.shape[-2] != group_indices.shape[-1] ** 2:
        raise ValueError("logits must contain the prefix-stripped grid cells")
    probs = (logits.float() / temperature).softmax(dim=-1)[..., 1:]
    mass = probs[:, group_indices].sum(dim=-2)
    with torch.no_grad():
        present = nn.functional.one_hot(
            labels.long().clamp(min=0),
            num_classes=logits.shape[-1],
        ).sum(dim=-2)
        target = (present > 0).float()[..., 1:].unsqueeze(-2)
    return (mass - target).square().mean(dim=(-2, -1))


# ---------------------------------------------------------------------------
# Run-directory convention.
# ---------------------------------------------------------------------------


def resolve_run_dir(base_dir: str | Path, experiment_name: str) -> Path:
    """Resolve the run directory: ``<base>/runs/{experiment_name}``.

    Args:
      base_dir: Scratch root; ``""`` resolves beneath ``/opt/scratch``.
      experiment_name: Run identity.

    Returns:
      run_dir: The run directory path (not created).

    """
    root = Path(base_dir) if str(base_dir) else Path("/opt/scratch")
    return root / "runs" / experiment_name


class EvalTimeLimitError(RuntimeError):
    """Raised when a single eval pass exceeds ``Trainer.Config.max_eval_time``.

    Surfaced as a hard failure so an over-budget eval fails the experiment
    instead of producing a slow, incomparable score.
    """


# ---------------------------------------------------------------------------
# The trainer.
# ---------------------------------------------------------------------------


class Trainer:
    """Pool-based ACT trainer for the sudoku TRM (single process).

    See the module docstring for the training algorithm, the fidelity
    invariants, and the checkpoint schema. ``Config()`` defaults are the full
    reproduction recipe; the experiment ladder's earlier rungs switch pieces
    off explicitly (``feedback=False``, ``csp_loss_weight=0``, ...).
    """

    class Config(Fig["Trainer"]):
        # -- Run identity and output layout.
        study_name: str = ""
        """Run-family prefix recorded for provenance; not part of any path."""

        experiment_name: str = ""
        """Run identity; keys the run dir ``runs/{name}`` beneath the scratch
        root and the checkpoint dir. Required (stamped by run.py when empty)."""

        doc: str = ""
        """Free-text description (hypothesis/changes/outcome); forwarded to
        the W&B run notes when W&B is enabled."""

        base_dir: Path | str | None = None
        """Scratch root shared by datasets and run outputs. ``None`` resolves
        to the fixed ``/opt/scratch`` root."""

        # -- Parts.
        model: TRM.Config = field(default_factory=TRM.Config)
        """The TRM; defaults are the full recipe architecture."""

        dataset: PuzzleDataset.Config = field(default_factory=PuzzleDataset.Config)
        """Puzzle data; ``dataset.batch_size`` is THE batch size (the pool
        width and the model's puzzle-embedding buffer size follow it)."""

        checkpointer: Checkpointer.Config | None = field(
            default_factory=recipe_checkpointer,
        )
        """Checkpoint engine; None disables checkpointing entirely."""

        runtime: SingleProcess.Config = field(
            # The device is PINNED: priml defaults to "auto", which silently trains on
            # CPU when a GPU is absent. This recipe is CUDA-only; failing loudly
            # beats a 100x-slow run nobody notices.
            default_factory=lambda: SingleProcess.Config(device="cuda"),
        )
        """Process runtime (device + determinism)."""

        wandb_project: str | None = None
        """W&B project name; set to enable the optional W&B tracker on top of
        the always-on ``metrics.json`` file tracker."""

        # -- Stop conditions and cadences.
        seed: int | None = None
        """Global seed for init/train RNG (see :func:`set_seed_local`); None
        draws a fresh seed from OS entropy."""

        max_steps: float = math.inf
        """Optimizer-step stop condition -- the stop for every recipe rung.
        ``run()`` requires max_steps or max_time finite."""

        max_time: float = math.inf
        """Time limit in seconds (clock chosen by ``max_time_kind``); kept for
        segment accounting parity, default off. The reproduction rungs stop
        on ``max_steps`` -- a wall cap would silently truncate the cosine
        schedule on slower GPUs."""

        max_time_kind: Literal["wall", "train"] = "train"
        """Which clock ``max_time`` caps. ``"train"`` (recipe): pure training
        time -- the first step's compile and every mid-loop eval are
        excluded. ``"wall"``: wall-clock since startup."""

        max_eval_time: float = math.inf
        """Wall-clock cap in seconds for a single eval pass; exceeding it
        raises :class:`EvalTimeLimitError`."""

        num_steps_eval: float = 1_000
        """Eval cadence in optimizer steps; ``inf`` disables eval (including
        the final one)."""

        num_steps_log: int = 10
        """Console/tracker logging cadence in optimizer steps."""

        eval_warmup_batches: int = 1
        """Eval batches run once at startup to populate compile caches before
        the training timers start."""

        restore_rng_state: bool = True
        """Restore the checkpointed RNG state on resume. Keep enabled for
        training resumes; eval-only flows on a different device layout may
        disable it."""

        ephemeral: bool = False
        """Throwaway eval-only construction (the det gate / sieve round 0):
        skip the process-global RNG reseed and the run-dir ``config.txt``
        record, so per-boundary constructions neither clobber the ambient
        RNG stream nor litter run directories. The model's random init is
        then unseeded -- the weights MUST be overwritten by a checkpoint
        load before use."""

        # -- Optimizers (AdamW for embeds/heads/1D + Muon for 2D/3D bodies).
        muon_lr: float = 0.02
        """Learning rate for the Muon optimizer (2D/3D weight matrices)."""

        muon_momentum: float = 0.6
        """Nesterov momentum coefficient for Muon."""

        muon_ns_steps: int = 3
        """Number of Newton-Schulz orthogonalization steps in Muon."""

        muon_weight_decay: float | None = None
        """Weight decay for Muon; None defaults to 1e-4 / muon_lr."""

        adamw_lr: float = 1e-4
        """Learning rate for the AdamW optimizer (embeds, heads, 1D)."""

        adamw_weight_decay: float | None = None
        """Weight decay for AdamW; None defaults to 1e-4 / adamw_lr."""

        puzzle_emb_lr: float = 1e-2
        """Learning rate for the SignSGD sparse puzzle-embedding step."""

        puzzle_emb_weight_decay: float = 0.1
        """Weight decay for the sparse puzzle-embedding SignSGD step."""

        total_train_steps: int = 19_500
        """Cosine LR horizon in optimizer steps (the recipe trains exactly to
        it; segmented runs keep it fixed across segments)."""

        warmup_steps: int = 0
        """Linear LR warmup steps from zero to the base LR."""

        lr_min_ratio: float = 0.0
        """Minimum LR as a fraction of base LR at the end of the schedule."""

        grad_clip_max_norm: float | None = 1.0
        """Max global gradient norm over the body params; None disables."""

        grad_clip_foreach: bool | None = True
        """Use fused foreach kernels for grad-norm clipping. Set ``None`` or
        ``False`` on the MPS backend (no foreach kernels there)."""

        log_body_norms: bool = True
        """Emit body gradient and parameter norm diagnostics."""

        norm_log_interval: int = 100
        """Step cadence for computing diagnostic body parameter norms."""

        # -- Loss.
        label_smoothing: float = 0.0
        """Label smoothing epsilon applied to the cross-entropy loss."""

        ignore_label_id: int = -100
        """Label value excluded from the CE loss and the q-halt correctness
        target; atomic-mode tail padding is masked to it."""

        train_q_halt: bool = True
        """Add the q-halt binary cross-entropy loss term (and let q-halt
        drive the halting decision)."""

        q_halt_weight: float = 0.05
        """Weight on the q-halt loss relative to the token CE loss."""

        csp_loss_weight: float = 0.3
        """Weight on the CSP cardinality auxiliary loss; 0 disables the term
        entirely (the pre-CSP ladder rungs)."""

        csp_loss_warmup_steps: int = 0
        """Global steps before the CSP term switches on (0 = always on)."""

        csp_temperature: float = 1.0
        """Softmax temperature for the CSP cardinality term."""

        # -- ACT (atomic scheduling only).
        max_act_steps: int = 32
        """Maximum ACT iterations per sample before forced halt."""

        halt_exploration_prob: float = 0.1
        """Probability of forcing a sample past its q-halt decision."""

        halt_exploration_seed: int = 0
        """Seed for the dedicated ACT halt-exploration RNG stream."""

        min_halt_steps_enabled: bool = True
        """Sample per-slot minimum ACT steps (reference-TRM exploration) so
        q-halt termination is suppressed until the slot has run that many
        steps; False uses the plain random force-continue instead."""

        # -- Feedback recurrence.
        feedback: bool = True
        """Thread the decoded-grid feedback recurrence through training and
        eval. False (the pre-feedback ladder rungs) never feeds the model's
        ``embed_feedback`` table, whose gradient then stays None -- the
        torch-standard skip-None-grad behavior in AdamW/clip/EMA keeps those
        rungs equivalent to a model without the table."""

        feedback_scramble_prob: float = 0.5
        """Per-step probability that a pool slot's stored feedback grid is
        scrambled before the next forward consumes it (corrupted-state
        recovery training). Train-side only; eval always threads the clean
        decoded grid."""

        feedback_scramble_cells: int = 12
        """Scramble intensity: each cell of a selected grid is replaced with
        a uniform random digit token (2-10) with probability ``cells / 81``,
        restricted to non-given positions."""

        feedback_scramble_seed: int = 0
        """Seed for the dedicated feedback-scramble RNG stream."""

        # -- EMA.
        use_ema: bool = True
        """Maintain an EMA shadow of the model parameters; eval always scores
        the EMA weights once past warmup (warmup 0 on the recipe)."""

        ema_decay: float = 0.999
        """EMA decay coefficient applied each optimizer step."""

        ema_warmup_steps: int = 0
        """Steps before EMA tracking begins; live weights are copied into the
        shadow at this step."""

        # -- Precision and compile.
        dtype_autocast: torch.dtype | None = torch.bfloat16
        """Autocast compute dtype for forward passes; None disables autocast.
        Master weights always stay fp32."""

        emulate_precision_casts: bool = True
        """Set TorchInductor's precision-cast emulation (process-global)
        before model construction. Required for sound COMPILED bf16 eval:
        when unset, eval runs eagerly (see ``_eval_compile_disabled``)."""

        # -- Eval rollout.
        eval_act_steps: int = 0
        """Eval ACT depth cap; 0 keeps ``max_act_steps``. Training pool
        dynamics always use ``max_act_steps``."""

        eval_halt_exit: bool = False
        """Release a row from the eval rollout at the first ACT step whose
        q-halt logit clears ``eval_halt_threshold`` (the conditional-depth
        deterministic pass used by the det gate / sieve round 0). False runs
        every row to the depth cap."""

        eval_halt_threshold: float = 0.0
        """Q-halt logit release threshold (the trained halt decision boundary
        is 0.0; the det pass uses +2.0)."""

        eval_min_act_steps: int = 1
        """Earliest ACT step at which a row may exit."""

        @override
        def finalize(self) -> Self:
            # The dataset batch size is THE batch size: the sparse puzzle
            # embedding's local gradient buffer must match it exactly.
            self.dataset.spec.finalize()
            spec = self.dataset.spec
            self.model.puzzle_grid_shape = (math.prod(spec.grid_shape),)
            self.model.vocab_size = spec.vocab_size
            if self.model.pos2d_grid_shape is not None:
                self.model.pos2d_grid_shape = spec.grid_shape
                self.model.pos2d_box_shape = spec.box_shape
            self.model.puzzle_emb_batch_size = self.dataset.batch_size
            # One device for data and compute unless the dataset was pointed
            # elsewhere explicitly.
            if self.dataset.device == "auto":
                self.dataset.device = str(self.runtime.device)
            # Push run context down; explicit child values win. The dataset is
            # a resource child (rooted at the scratch base); the checkpointer is
            # a run-output child (rooted at the run directory).
            if self.dataset.base_dir is None:
                self.dataset.base_dir = self.base_dir
            if self.checkpointer is not None and self.checkpointer.base_dir is None:
                self.checkpointer.base_dir = resolve_run_dir(
                    self.base_dir if self.base_dir is not None else "",
                    self.experiment_name,
                )
            return super().finalize()

    def __init__(self, config: Config) -> None:
        if not config.experiment_name:
            raise ValueError(
                "Trainer.Config.experiment_name is required: it keys the run "
                "and checkpoint directories.",
            )
        if config.eval_act_steps < 0:
            raise ValueError(
                f"eval_act_steps must be >= 0, got {config.eval_act_steps}.",
            )
        if config.eval_min_act_steps < 1:
            raise ValueError(
                f"eval_min_act_steps must be >= 1, got {config.eval_min_act_steps}.",
            )
        rows, cols = config.dataset.spec.grid_shape
        box_rows, box_cols = config.dataset.spec.box_shape
        if config.csp_loss_weight > 0 and (
            rows != cols
            or min(rows, cols, box_rows, box_cols) < 1
            or rows % box_rows
            or cols % box_cols
            or box_rows * box_cols != rows
            or config.model.num_puzzle_grid_tokens != rows * cols
            or config.model.vocab_size != rows + 2
        ):
            raise ValueError(
                "csp_loss_weight > 0 requires sudoku grid, boxes, and vocabulary to agree.",
            )
        self.config = config
        self.runtime = config.runtime.make()
        self.runtime.initialize()
        self.device = self.runtime.device
        self.global_step = 0
        self.local_step = 0
        self.current_epoch = 0

        # Seed BEFORE model construction: the init draws from the global RNG
        # in a pinned order (see model.py), so the seed placement is part of
        # the reproducibility contract. Ephemeral (eval-only) constructions
        # skip the reseed: their weights are always overwritten by a
        # checkpoint load, and reseeding would clobber the ambient stream.
        if not config.ephemeral:
            set_seed_local(config.seed)

        # Dedicated RNG streams (never the ambient global RNG): the training
        # trajectory must not depend on ambient state, so two runs from
        # identical weights stay bit-for-bit identical.
        self._halt_gen = torch.Generator(device=self.device)
        self._halt_gen.manual_seed(config.halt_exploration_seed)
        self._scramble_gen = torch.Generator(device=self.device)
        self._scramble_gen.manual_seed(config.feedback_scramble_seed)

        # Process-global TorchInductor flag; must be set before TRM.__init__
        # binds torch.compile so the first trace captures it.
        torch._inductor.config.emulate_precision_casts = (  # noqa: SLF001 -- Documented inductor knob, no public alias.
            config.emulate_precision_casts
        )
        self.model: TRM = config.model.make()
        if not config.train_q_halt:
            for param in self.model.q_head.parameters():
                param.requires_grad_(False)
        # Master weights stay fp32; autocast handles bf16 compute. The carry
        # pools are fp32 too so backward through autocast does not accumulate
        # bf16 quantization error across ACT steps.
        self.model.to(device=self.device)

        self._ema: EMA | None = None
        if config.use_ema:
            ema_cfg = recipe_ema()
            ema_cfg.decay = config.ema_decay
            ema_cfg.update_after_step = config.ema_warmup_steps
            self._ema = ema_cfg.make()

        self._optimizer_puzzle_emb = self._build_sparse_puzzle_emb_optimizer()
        self._optimizers = self._build_adamw_muon_optimizers()
        for opt in self._optimizers:
            for g in opt.param_groups:
                g["initial_lr"] = g["lr"]
        self._body_params = [
            p
            for n, p in self.model.named_parameters()
            if p.requires_grad and "puzzle_emb" not in n
        ]

        # ACT pool state (atomic mode: one batch per call, halted slots take
        # fresh data, active slots keep their carry).
        bs = self._batch_size = config.dataset.batch_size
        grid_len = config.model.num_puzzle_grid_tokens
        seq_len = config.model.total_seq_len
        hidden = config.model.channels_in
        self._pool_inputs = torch.zeros(
            bs,
            grid_len,
            device=self.device,
            dtype=torch.long,
        )
        self._pool_labels = torch.zeros_like(self._pool_inputs)
        self._pool_z_slow = torch.zeros(
            bs,
            seq_len,
            hidden,
            device=self.device,
            dtype=torch.float32,
        )
        self._pool_z_fast = torch.zeros_like(self._pool_z_slow)
        self._pool_h_step = torch.zeros(bs, device=self.device, dtype=torch.long)
        # All-halted at init so the first call refills every slot.
        self._pool_halted = torch.ones(bs, device=self.device, dtype=torch.bool)
        self._pool_puzzle_ids = torch.zeros(bs, device=self.device, dtype=torch.int32)
        self._pool_feedback = torch.zeros_like(self._pool_inputs)
        self._csp_groups = sudoku_group_indices(config.dataset.spec).to(self.device)

        self.dataset = config.dataset.make()
        self.checkpointer: Checkpointer | None = (
            config.checkpointer.make() if config.checkpointer is not None else None
        )

        self.run_dir = resolve_run_dir(
            config.base_dir if config.base_dir is not None else "",
            config.experiment_name,
        )
        # Priml splits the destination into base_dir + working_dir; the trainer
        # owns an absolute run dir, so base_dir stays None and the absolute path
        # goes in working_dir verbatim.
        metrics_cfg = FileTracker.Config()
        metrics_cfg.working_dir = self.run_dir / "metrics.json"
        self._trackers: list[FileTracker | WandbTracker] = [metrics_cfg.make()]
        if config.wandb_project is not None:
            wandb_cfg = WandbTracker.Config()
            wandb_cfg.project = config.wandb_project
            wandb_cfg.name = config.experiment_name
            wandb_cfg.working_dir = self.run_dir
            self._trackers.append(wandb_cfg.make())
        if config.doc:
            for tracker in self._trackers:
                tracker.log_notes(config.doc)

        self._train_loader: Iterable[PuzzleBatch] | None = None
        self._train_iter: Iterator[PuzzleBatch] | None = None
        self._start_time = time.perf_counter()
        self._train_clock_base = self._start_time

        # Resume + overwrite-guard before the first training step, so a
        # collision is caught at startup, not thousands of steps in.
        resumed = False
        if self.checkpointer is not None:
            resumed = self.checkpointer.load(self, max_steps=config.max_steps)
            if resumed:
                logger.info(
                    "Resumed from checkpoint at step %d under %s.",
                    self.global_step,
                    self.checkpointer.checkpoint_dir,
                )
        if not config.ephemeral:
            self._guard_resume_config(resumed)
        self._warm_eval_compile()
        self._start_time = time.perf_counter()
        self._train_clock_base = self._start_time

    # On fresh start (or when no record exists yet) the finalized config's pprint text
    # is persisted as ``config.txt`` in the run dir, so the file always documents the
    # config the run was launched with. On auto-resume the current text is compared
    # against that record:
    #
    # * identical -- silent; * only stop conditions differ (top-level
    # ``max_steps``/``max_time`` -- horizon extension and the segmented reproduction
    # pipelines move them legitimately) -- an INFO notice; * anything else differs -- a
    # loud warning with the full diff. Never a hard failure: ``--fresh`` remains the
    # escape hatch, and the recorded file is left untouched as the fresh-start evidence.
    def _guard_resume_config(self, resumed: bool) -> None:
        """Record the launch config; warn LOUDLY when a resume changed it."""
        path = self.run_dir / "config.txt"
        text = self.config.pformat(hide_default_values=False)
        if not text.endswith("\n"):
            text += "\n"
        if not resumed or not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
            return
        recorded = path.read_text()
        if recorded == text:
            return
        diff = list(
            difflib.unified_diff(
                recorded.splitlines(),
                text.splitlines(),
                fromfile="config.txt (recorded at fresh start)",
                tofile="current config",
                lineterm="",
                n=0,
            ),
        )
        # Top-level Trainer.Config stop-condition lines in the recorded config
        # pprint. These legitimately change across a resume (horizon extension; the
        # reproduction pipelines' segment re-walk bumps max_steps every segment), so
        # the resume-config guard reports them quietly instead of shouting. Nested
        # fields carry a box-drawing pipe in their indent and never match.
        meaningful = [
            line
            for line in diff
            if line.startswith(("-", "+"))
            and not line.startswith(("---", "+++"))
            and not re.match(r"^\s*max_(steps|time)=", line[1:])
        ]
        if not meaningful:
            logger.info(
                "Resume changed only stop conditions (max_steps/max_time) "
                "relative to the recorded launch config at %s -- expected "
                "for horizon extension and segmented pipelines.",
                path,
            )
            return
        banner = "=" * 72
        logger.warning(
            "\n%s\nCONFIG MISMATCH ON RESUME\n%s\n"
            "This run resumed from a checkpoint, but the current config\n"
            "differs from the launch config recorded at\n"
            "  %s\n"
            "Training onward under a changed config can silently corrupt a\n"
            "resumed run. If the change is intentional, carry on; otherwise\n"
            "restart with --fresh (or move the run dir aside).\n"
            "Diff (recorded launch config -> current):\n%s\n%s",
            banner,
            banner,
            path,
            "\n".join(diff),
            banner,
        )

    # -- Public seams --------------------------------------------------------

    @property
    def ema_shadow(self) -> dict[str, Tensor] | None:
        """Name-keyed EMA shadow params (checkpoint-upload contract), or None.

        Empty until the first ``train_step`` lazily seeds the shadow (a
        resumed shadow is available immediately). None when EMA is disabled.
        """
        if self._ema is None:
            return None
        return self._ema.shadow_params

    def run(self, *args: str) -> None:
        """Train to the stop condition, then final-checkpoint and final-eval.

        Args:
          *args: Ignored (logged); accepted so launcher passthrough CLI args
            (a launcher calls ``job.run(*unparsed)``) never TypeError.

        Raises:
          ValueError: If neither ``max_steps`` nor ``max_time`` is finite
            (the loop would never terminate).

        """
        if args:
            logger.info("Ignoring launcher passthrough args: %r.", args)
        config = self.config
        if config.max_steps == math.inf and config.max_time == math.inf:
            raise ValueError(
                "Trainer has no finite stop condition: set max_steps or "
                "max_time on the config.",
            )
        try:
            trained_any = False
            while self.global_step < config.max_steps and not (
                self._time_limit_reached()
            ):
                batch = self._next_batch()
                # A resumed loop may start exactly on a checkpoint/eval
                # cadence step. Do not re-save or re-score that restored
                # state before this process has advanced training once.
                if self.local_step > 0 and self.checkpointer is not None:
                    self.checkpointer.maybe_save(self, self.global_step)
                if self.local_step > 0:
                    self._maybe_eval()
                self._do_train_step(batch)
                trained_any = True
            elapsed = self._max_time_elapsed()
            if elapsed >= config.max_time:
                logger.warning(
                    "Time limit reached (%.1fs >= %.1fs), stopping early at step %d.",
                    elapsed,
                    config.max_time,
                    self.global_step,
                )
            # Nothing ran (e.g. max_steps already reached on resume): nothing
            # new to persist or measure.
            if not trained_any:
                return
            if self.checkpointer is not None:
                self.checkpointer.save(self, self.global_step)
            # The cadence eval never lands on the final step (the loop exits
            # first), so run one last eval and mark it final: it emits the
            # RESULT line. num_steps_eval=inf disables eval entirely.
            self._maybe_eval(is_final=True)
        finally:
            self._close()

    def evaluate(self) -> dict[str, float]:
        """Run one full eval pass on the current weights and publish it.

        The checkpoint (if any) was already restored at construction by the
        checkpointer's resume policy. Publishes the metrics as the run's
        final eval (RESULT line + ``metrics.json``); trackers stay open so a
        caller may evaluate repeatedly.

        Returns:
          metrics: The scalar eval metrics.

        """
        if self.global_step == 0:
            logger.warning(
                "evaluate() at global_step=0: no checkpoint was resumed; "
                "scoring freshly initialized weights.",
            )
        return self._maybe_eval(is_final=True, force=True) or {}

    def preprocess_batch(self, batch: PuzzleBatch) -> PuzzleBatch:
        """Move every tensor in ``batch`` to the trainer's device.

        Args:
          batch: One dataset batch.

        Returns:
          batch: The same keys, tensors on the trainer's device.

        """
        return cast(
            "PuzzleBatch",
            {
                k: v.to(self.device, non_blocking=True) if isinstance(v, Tensor) else v
                for k, v in batch.items()
            },
        )

    def train_step(self, **batch: object) -> TrainStepOutput:
        """One optimizer step over the ACT pool.

        Args:
          **batch: The dataset batch contract -- ``media`` int32 [B, 81],
            ``label`` int32 [B, 81], ``valid_count`` int,
            ``puzzle_identifiers`` int32 [B].

        Returns:
          out: ``loss`` [1], a tiny ``model`` probe slice, and ``metrics``.

        """
        media = batch["media"]
        assert isinstance(media, Tensor)
        label = batch["label"]
        assert isinstance(label, Tensor)
        raw_count = batch.get("valid_count", media.shape[0])
        assert isinstance(raw_count, int)
        valid_count = raw_count
        pid_raw = batch.get("puzzle_identifiers")
        pid = pid_raw if isinstance(pid_raw, Tensor) else None
        active = self._refill_pool(media, label, valid_count, pid)
        n_active = active.sum()
        fl = self._forward_and_loss(active=active, n_active=n_active)
        train_metrics = fl.train_metrics

        # Step tail -- the fixed order contract (module docstring).
        fl.loss.backward()
        log_norms = self._should_log_norms()
        grad_norm: Tensor | None = None
        if self.config.grad_clip_max_norm is not None:
            grad_norm = clip_grad_norm_(
                self._body_params,
                self.config.grad_clip_max_norm,
                foreach=self.config.grad_clip_foreach,
            )
        elif log_norms:
            grad_norm = total_grad_norm(
                self._body_params,
                foreach=self.config.grad_clip_foreach,
            )
        if self.config.log_body_norms and grad_norm is not None:
            train_metrics["grad_norm"] = grad_norm.detach()
        if log_norms:
            train_metrics["param_norm"] = (
                torch.stack([p.detach().norm(2.0) for p in self._body_params]).norm(2.0)
            ).detach()
        scale = lr_scale(
            self.global_step,
            self.config.total_train_steps,
            self.config.warmup_steps,
            self.config.lr_min_ratio,
        )
        for opt in self._optimizers:
            for g in opt.param_groups:
                g["lr"] = g["initial_lr"] * scale
        train_metrics["lr"] = self._optimizers[0].param_groups[0]["lr"]
        muon_opt = next(
            (opt for opt in self._optimizers if isinstance(opt, Muon)),
            None,
        )
        if muon_opt is not None:
            train_metrics["muon_lr"] = muon_opt.param_groups[0]["lr"]
        for opt in self._optimizers:
            opt.step()
        self._apply_sparse_puzzle_emb_step()
        self.model.zero_grad(set_to_none=True)
        if self._ema is not None:
            self._ema(self.model)
        self._update_carry_and_halt(
            z_slow_out=fl.z_slow_out,
            z_fast_out=fl.z_fast_out,
            q_halt=fl.q_halt,
        )

        # Halted-only train metrics, the reference TRM definition: score only
        # slots that HALT this step. These OVERWRITE the primary metric names
        # (train/exact_accuracy etc. overlay the reference curves); the
        # all-active variants stay queryable under ``active_*``.
        if fl.correctness is not None:
            is_correct, loss_counts, correct = fl.correctness
            with torch.no_grad():
                halted_valid = self._pool_halted & active & (loss_counts > 0)
                n_halted = halted_valid.sum().clamp(min=1)
                train_metrics.update(
                    {
                        "cell_accuracy": torch.where(
                            halted_valid,
                            (
                                is_correct.float()
                                / loss_counts.clamp_min(1).unsqueeze(-1)
                            ).sum(dim=-1),
                            torch.zeros_like(correct),
                        ).sum()
                        / n_halted,
                        "exact_accuracy": (halted_valid & correct.bool()).float().sum()
                        / n_halted,
                        "q_halt_accuracy": (
                            halted_valid & ((fl.q_halt >= 0) == correct.bool())
                        )
                        .float()
                        .sum()
                        / n_halted,
                        "act_steps": torch.where(
                            halted_valid,
                            self._pool_h_step,
                            torch.zeros_like(self._pool_h_step),
                        )
                        .float()
                        .sum()
                        / n_halted,
                        "halted_frac": halted_valid.float().mean(),
                    },
                )

        self.global_step += 1
        self.local_step += 1
        return {
            "loss": fl.loss.detach().unsqueeze(0),
            "model": fl.logits[:1, :1].detach(),
            "metrics": train_metrics,
        }

    def eval_loss(self, **batch: object) -> TrainStepOutput:
        """Full ACT eval on one batch: EMA swap, rollout, metric pack.

        Each ACT step runs the full ``slow_cycles`` forward via ``act_step``
        (the unit the model is trained on), threading the decoded-grid
        feedback when the feedback channel is on. With ``eval_halt_exit``,
        rows are released at the first step whose q-halt logit clears the
        threshold (the conditional-depth deterministic pass); otherwise every
        row runs to the depth cap, which is value-identical to the plain
        fixed-depth eval.

        Args:
          **batch: The dataset batch contract (see :meth:`train_step`).

        Returns:
          out: ``loss`` [1]; ``model`` packed ``[B, 1 + grid_len]`` with
            col 0 the q-halt logit and the rest the predicted grid tokens;
            ``metrics`` scalars.

        """
        config = self.config
        media = batch["media"]
        assert isinstance(media, Tensor)
        label = batch["label"]
        assert isinstance(label, Tensor)
        raw_count = batch.get("valid_count", media.shape[0])
        assert isinstance(raw_count, int)
        valid_count = raw_count
        batch_size = media.shape[0]
        if valid_count == 0:
            grid_len = config.model.num_puzzle_grid_tokens
            return {
                "loss": torch.zeros(1, device=self.device),
                "model": torch.zeros(batch_size, 1 + grid_len, device=self.device),
                "metrics": {
                    "lm_loss": 0.0,
                    "q_halt_loss": 0.0,
                    "cell_accuracy": 0.0,
                    "exact_accuracy": 0.0,
                    "q_halt_accuracy": 0.0,
                    "act_steps": 0.0,
                    "q_continue_loss": 0.0,
                },
            }

        step_kwargs: dict[str, Tensor] = {}
        if self.model.puzzle_emb is not None:
            puzzle_identifiers_value = batch["puzzle_identifiers"]
            assert isinstance(puzzle_identifiers_value, Tensor)
            step_kwargs["puzzle_identifiers"] = puzzle_identifiers_value
        # Apply EMA weights strictly PAST warmup (``>``, not ``>=``): the
        # shadow is seeded on the train step AT the warmup boundary, which
        # runs after the boundary eval -- a ``>=`` gate would score an
        # unseeded shadow there.
        swap: AbstractContextManager[None] = (
            self._ema.apply_to(self.model)
            if self._ema is not None and self.global_step > config.ema_warmup_steps
            else nullcontext()
        )
        self.model.eval()
        with (
            swap,
            self._eval_compile_disabled(),
            torch.inference_mode(),
            self._autocast(),
        ):
            logits, q_halt, act_used = self._eval_rollout(media, step_kwargs)

            logits_v = logits[:valid_count]
            labels_v = label[:valid_count]
            preds = logits.argmax(dim=-1)
            preds_v = preds[:valid_count]
            q_halt_v = q_halt[:valid_count]

            ignore = config.ignore_label_id
            valid_mask = labels_v != ignore
            cell_correct = (preds_v == labels_v) & valid_mask
            loss_counts = valid_mask.sum(dim=-1)
            sample_correct = (cell_correct.sum(dim=-1) == loss_counts) & (
                loss_counts > 0
            )
            active = torch.ones(logits_v.shape[0], dtype=torch.bool, device=self.device)
            lm_loss = self._compute_loss(
                logits=logits_v,
                labels=labels_v,
                active=active,
                n_active=active.sum(),
            )
            q_halt_loss = nn.functional.binary_cross_entropy_with_logits(
                q_halt_v,
                sample_correct.to(q_halt_v.dtype),
            )
            per_sample_cell = cell_correct.float().sum(dim=-1) / loss_counts.clamp(
                min=1,
            )
            cell_acc = per_sample_cell.mean().item()
            cell_weighted_acc = cell_correct.sum().item() / max(
                1,
                int(loss_counts.sum().item()),
            )
            exact_acc = sample_correct.float().mean().item()
            q_halt_acc = ((q_halt_v >= 0) == sample_correct).float().mean().item()
            act_used_v = act_used[:valid_count].float()
            depth_cap = config.eval_act_steps or config.max_act_steps

        # eval/total_loss mirrors train/total_loss: lm + weighted q-halt loss
        # (q_continue is 0 under no-ACT-continue parity).
        total_loss = lm_loss + config.q_halt_weight * q_halt_loss
        model_output = torch.cat(
            [q_halt.reshape(batch_size, 1).float(), preds.float()],
            dim=-1,
        )
        metrics: dict[str, float | Tensor] = {
            "lm_loss": lm_loss.detach(),
            "q_halt_loss": q_halt_loss.detach(),
            "cell_accuracy": cell_acc,
            "cell_weighted_accuracy": cell_weighted_acc,
            "exact_accuracy": exact_acc,
            "q_halt_accuracy": q_halt_acc,
            "act_steps": act_used_v.mean().item(),
            "q_continue_loss": 0.0,
        }
        if config.eval_halt_exit:
            metrics["act_cap_frac"] = (act_used_v >= depth_cap).float().mean().item()
        return {
            "loss": total_loss.detach().reshape(1).float(),
            "model": model_output,
            "metrics": metrics,
        }

    # A host that merely HAS a GPU would otherwise snapshot CUDA generators this run
    # never touched, so a checkpoint minted here and resumed on CPU-only CI carries a
    # ``cuda`` entry whose device count no longer matches and ``set_rng_state`` rejects
    # it.
    def _rng_state(self) -> RngState:
        """Capture RNG state, dropping CUDA entries a CPU run never advanced."""
        state = get_rng_state()
        if self.device.type != "cuda":
            state.pop("cuda", None)
            state.pop("cuda_uuids", None)
        return state

    class StepStateDict(TypedDict):
        """The ``"step"`` payload: model, optimizers, step, RNG streams, EMA."""

        model: dict[str, Tensor]
        optimizers: list[dict[str, Any]]  # pyright: ignore[reportExplicitAny] -- torch's `Optimizer.state_dict()` schema.
        global_step: int
        halt_rng: NotRequired[Tensor]
        scramble_rng: NotRequired[Tensor]
        optimizer_puzzle_emb: NotRequired[dict[str, Any]]  # pyright: ignore[reportExplicitAny] -- torch's `Optimizer.state_dict()` schema.
        ema: NotRequired[dict[str, Tensor]]

    class StateDict(TypedDict):
        """Checkpointed training state (module-docstring schema)."""

        step: Trainer.StepStateDict
        dataset: PuzzleDataset.StateDict
        metrics: dict[str, object]
        epoch: int
        rng: NotRequired[RngState]

    def state_dict(self) -> StateDict:
        """Snapshot the full training state (module-docstring schema).

        Returns:
          state: Nested model, optimizer, dataset, metric, epoch, and RNG state.

        """
        step_state: Trainer.StepStateDict = {
            "model": self.model.state_dict(),
            "optimizers": [opt.state_dict() for opt in self._optimizers],
            "global_step": self.global_step,
            # Both dedicated RNG streams are persisted so a resumed run
            # continues the exact exploration/scramble sequences.
            "halt_rng": self._halt_gen.get_state(),
            "scramble_rng": self._scramble_gen.get_state(),
        }
        if self._optimizer_puzzle_emb is not None:
            step_state["optimizer_puzzle_emb"] = self._optimizer_puzzle_emb.state_dict()
        if self.ema_shadow is not None:
            # Flat name-keyed dict: the cross-tool eval/upload contract
            # (consumers read ``state["step"]["ema"][name]``).
            step_state["ema"] = dict(self.ema_shadow)
        # The ACT pool is deliberately NOT checkpointed (module docstring).
        return {
            "step": step_state,
            "dataset": self.dataset.state_dict(),
            "metrics": {},
            "epoch": self.current_epoch,
            "rng": self._rng_state(),
        }

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        """Restore training state produced by :meth:`state_dict`.

        Args:
          state_dict: Nested training state previously returned by ``state_dict``.

        """
        state = cast(Trainer.StateDict, state_dict)
        step_state = state["step"]
        self.model.load_state_dict(step_state["model"])
        for opt, opt_state in zip(
            self._optimizers,
            step_state["optimizers"],
            strict=True,
        ):
            opt.load_state_dict(opt_state)
        if (
            self._optimizer_puzzle_emb is not None
            and "optimizer_puzzle_emb" in step_state
        ):
            self._optimizer_puzzle_emb.load_state_dict(
                step_state["optimizer_puzzle_emb"],
            )
        self.global_step = step_state["global_step"]
        self.local_step = 0
        # ``Generator.set_state`` requires a CPU ByteTensor; the checkpoint
        # reader maps storages to the compute device, so coerce here.
        if "halt_rng" in step_state:
            self._halt_gen.set_state(step_state["halt_rng"].cpu())
        if "scramble_rng" in step_state:
            self._scramble_gen.set_state(step_state["scramble_rng"].cpu())
        if self._ema is not None and "ema" in step_state:
            self._ema.global_step = self.global_step
            self._ema.load_state_dict(
                {
                    "shadow_params": dict(step_state["ema"]),
                    "global_step": self.global_step,
                },
            )
        self.dataset.load_state_dict(state["dataset"])
        self.current_epoch = state["epoch"]
        if self.config.restore_rng_state and "rng" in state:
            set_rng_state(state["rng"])

    # -- Optimizer construction ------------------------------------------------

    # The name-substring routing is a verbatim port and a parity contract: parameter
    # names pin which optimizer trains each weight.
    def _build_adamw_muon_optimizers(self) -> list[torch.optim.Optimizer]:
        """AdamW for embeds/heads/1D + Muon for 2D/3D matrices."""
        config = self.config
        muon_wd = config.muon_weight_decay
        if muon_wd is None:
            muon_wd = 1e-4 / config.muon_lr
        adamw_wd = config.adamw_weight_decay
        if adamw_wd is None:
            adamw_wd = 1e-4 / config.adamw_lr

        adam_params: list[nn.Parameter] = []
        muon_2d: list[nn.Parameter] = []
        muon_3d: list[nn.Parameter] = []
        for n, p in self.model.named_parameters():
            if not p.requires_grad or "puzzle_emb" in n:
                continue
            if p.ndim >= 2 and "embed" not in n and "head" not in n:
                (muon_3d if p.ndim == 3 else muon_2d).append(p)
            else:
                adam_params.append(p)

        adam_opt = torch.optim.AdamW(
            adam_params,
            lr=config.adamw_lr,
            betas=(0.9, 0.95),  # Hardcoded: a recipe invariant, not a knob.
            weight_decay=adamw_wd,
            fused=False,
        )
        muon_groups: list[dict[str, object]] = []
        if muon_2d:
            muon_groups.append({"params": muon_2d, "ensemble_dims": 0})
        if muon_3d:
            muon_groups.append({"params": muon_3d, "ensemble_dims": 1})
        if not muon_groups:
            return [adam_opt]
        muon_opt = Muon(
            muon_groups,
            lr=config.muon_lr,
            momentum=config.muon_momentum,
            nesterov=True,
            ns_steps=config.muon_ns_steps,
            weight_decay=muon_wd,
        )
        return [adam_opt, muon_opt]

    def _build_sparse_puzzle_emb_optimizer(self) -> SignSGD | None:
        """SignSGD over the sparse puzzle-embedding buffers, or None."""
        # SparsePuzzleEmbedding registers its state as buffers (weights /
        # local_weights / local_ids); SignSGD needs them in the param group so
        # its sparse step can read/write them.
        puzzle_emb = self.model.puzzle_emb
        if puzzle_emb is None:
            return None
        return SignSGD(
            [
                {
                    "params": list(puzzle_emb.buffers()),
                    "lr": self.config.puzzle_emb_lr,
                    "weight_decay": self.config.puzzle_emb_weight_decay,
                },
            ],
        )

    # No-op when the model has no puzzle embedding or no gradient reached its local
    # buffer this step.
    def _apply_sparse_puzzle_emb_step(self) -> None:
        """Apply SignSGD's sparse-row update for the puzzle embedding."""
        if self._optimizer_puzzle_emb is None:
            return
        puzzle_emb = self.model.puzzle_emb
        if puzzle_emb is None or puzzle_emb.local_weights.grad is None:
            return
        self._optimizer_puzzle_emb.step_sparse_embedding(
            puzzle_emb.local_weights.grad.detach(),
            puzzle_emb.local_ids.detach(),
        )
        puzzle_emb.local_weights.grad = None

    # -- Pool refill / forward / halt -------------------------------------------

    # Tail positions past ``valid_count`` get their labels overwritten with the ignore
    # label so they contribute nothing to the loss. Slots that received fresh data
    # restart their feedback grid at the input.
    def _refill_pool(
        self,
        media: Tensor,
        label: Tensor,
        valid_count: int,
        puzzle_ids: Tensor | None,
    ) -> Tensor:
        """Slot the incoming batch into halted pool positions (atomic mode)."""
        bs = self._batch_size
        if media.shape[0] != bs:
            raise ValueError(
                f"atomic ACT scheduling requires incoming batch of size {bs}; "
                f"got {media.shape[0]}.",
            )
        # Snapshot the halted mask BEFORE the base refill mutates pool state
        # (ported ordering contract; the feedback reseed below must key off
        # the same mask the refill used).
        halted = self._pool_halted.clone()
        masked_inputs = media.clone().to(torch.long)
        masked_labels = label.clone().to(torch.long)
        if valid_count < bs:
            masked_labels[valid_count:] = self.config.ignore_label_id
        halted_seq = self._pool_halted.unsqueeze(-1)
        self._pool_inputs = torch.where(halted_seq, masked_inputs, self._pool_inputs)
        self._pool_labels = torch.where(halted_seq, masked_labels, self._pool_labels)
        if puzzle_ids is not None:
            ids = puzzle_ids.to(torch.int32)
            if valid_count < bs:
                ids = ids.clone()
                ids[valid_count:] = 0
            self._pool_puzzle_ids = torch.where(
                self._pool_halted,
                ids,
                self._pool_puzzle_ids,
            )
        z_slow_init, z_fast_init = self.model.init_z(bs)
        halted_z = self._pool_halted.view(-1, 1, 1)
        self._pool_z_slow = torch.where(
            halted_z,
            z_slow_init.to(self._pool_z_slow.dtype),
            self._pool_z_slow,
        )
        self._pool_z_fast = torch.where(
            halted_z,
            z_fast_init.to(self._pool_z_fast.dtype),
            self._pool_z_fast,
        )
        self._pool_h_step = torch.where(
            self._pool_halted,
            torch.zeros_like(self._pool_h_step),
            self._pool_h_step,
        )
        # Feedback restart AFTER the base refill: fresh slots start the
        # recurrence from their (new) input grid.
        self._pool_feedback = torch.where(
            halted.unsqueeze(-1),
            self._pool_inputs,
            self._pool_feedback,
        )
        # Every slot is active in atomic mode -- slots that just received
        # fresh data participate in this step's forward.
        return torch.ones_like(halted)

    # Order: feedback stash -> forward -> lm CE -> ``+ q_halt_weight * q_bce`` -> ``+
    # csp_loss_weight * csp`` -> feedback pool update (argmax + scramble draws). The
    # feedback update runs LAST, matching the internal cooperative-override ordering.
    def _forward_and_loss(self, *, active: Tensor, n_active: Tensor) -> _ForwardLoss:
        """Forward the pool and compose the loss (the fixed-order contract)."""
        config = self.config
        self.model.train()
        if config.feedback:
            self.model.set_feedback(self._pool_feedback)
        forward_kwargs: dict[str, Tensor] = {}
        if self.model.puzzle_emb is not None:
            forward_kwargs["puzzle_identifiers"] = self._pool_puzzle_ids
        with self._autocast():
            out = self.model(
                self._pool_inputs,
                self._pool_z_slow,
                self._pool_z_fast,
                **forward_kwargs,
            )
        logits = out["logits"]
        q_halt = out["q_halt"]
        z_slow_out = out["z_slow"]
        z_fast_out = out["z_fast"]
        assert isinstance(logits, Tensor)
        assert isinstance(q_halt, Tensor)
        assert isinstance(z_slow_out, Tensor)
        assert isinstance(z_fast_out, Tensor)

        lm_loss = self._compute_loss(
            logits=logits,
            labels=self._pool_labels,
            active=active,
            n_active=n_active,
        )
        loss = lm_loss
        train_metrics: dict[str, float | Tensor] = {
            "lm_loss": lm_loss.detach(),
            "q_continue_loss": 0.0,
        }

        # q-halt loss. Ignore-masked seq_is_correct: positions with the
        # ignore label do not count toward correctness (atomic tail padding).
        correctness: tuple[Tensor, Tensor, Tensor] | None = None
        if config.train_q_halt:
            ignore = config.ignore_label_id
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                valid_mask = self._pool_labels != ignore
                is_correct = (preds == self._pool_labels) & valid_mask
                loss_counts = valid_mask.sum(dim=-1)
                correct = (
                    (is_correct.sum(dim=-1) == loss_counts) & (loss_counts > 0)
                ).float()
            q_loss = nn.functional.binary_cross_entropy_with_logits(
                q_halt,
                correct,
                reduction="none",
            )
            q_loss = torch.where(active, q_loss, torch.zeros_like(q_loss))
            q_halt_loss = q_loss.sum() / n_active.clamp(min=1)
            loss = loss + config.q_halt_weight * q_halt_loss
            active_valid = active & (loss_counts > 0)
            n_active_valid = active_valid.sum().clamp(min=1)
            # All-active metrics mix mid-ACT depths; they live under
            # ``active_*`` while the primary names are overwritten with
            # halted-only values in the step tail.
            active_metrics = {
                "cell_accuracy": torch.where(
                    active_valid,
                    (is_correct.float() / loss_counts.clamp_min(1).unsqueeze(-1)).sum(
                        dim=-1,
                    ),
                    torch.zeros_like(correct),
                ).sum()
                / n_active_valid,
                "exact_accuracy": (active_valid & correct.bool()).float().sum()
                / n_active_valid,
                "q_halt_accuracy": (active_valid & ((q_halt >= 0) == correct.bool()))
                .float()
                .sum()
                / n_active_valid,
                "act_steps": torch.where(
                    active_valid,
                    self._pool_h_step + 1,
                    torch.zeros_like(self._pool_h_step),
                )
                .float()
                .sum()
                / n_active_valid,
            }
            train_metrics["q_halt_loss"] = q_halt_loss.detach()
            train_metrics.update(active_metrics)
            train_metrics.update(
                {f"active_{k}": v for k, v in active_metrics.items()},
            )
            correctness = (is_correct, loss_counts, correct)

        # Loss composition order contract: (lm + w_q * q) FIRST, then
        # + w_csp * csp.
        if config.csp_loss_weight > 0:
            if self.global_step < config.csp_loss_warmup_steps:
                train_metrics["csp_loss"] = 0.0
            else:
                csp_loss = self._csp_cardinality_loss(
                    logits,
                    active=active,
                    n_active=n_active,
                )
                loss = loss + config.csp_loss_weight * csp_loss
                train_metrics["csp_loss"] = csp_loss.detach()

        # Advance the feedback recurrence AFTER the loss terms: the pool's
        # next feedback is this forward's decoded grid (givens clamped),
        # optionally scrambled for corrupted-state recovery training.
        if config.feedback:
            with torch.no_grad():
                self._pool_feedback = self._clamp_givens(
                    logits.argmax(dim=-1).detach(),
                    self._pool_inputs,
                )
                if config.feedback_scramble_prob > 0:
                    self._pool_feedback = self._scramble_feedback(
                        self._pool_feedback,
                        self._pool_inputs,
                    )

        return _ForwardLoss(
            loss=loss,
            logits=logits,
            q_halt=q_halt,
            z_slow_out=z_slow_out,
            z_fast_out=z_fast_out,
            correctness=correctness,
            train_metrics=train_metrics,
        )

    def _compute_loss(
        self,
        *,
        logits: Tensor,
        labels: Tensor,
        active: Tensor,
        n_active: Tensor,
    ) -> Tensor:
        """Token CE -> per-sample mean over non-ignore -> mean over active."""
        ignore = self.config.ignore_label_id
        loss_per_token = nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1).long(),
            label_smoothing=self.config.label_smoothing,
            ignore_index=ignore,
            reduction="none",
        ).reshape(logits.shape[0], -1)
        nonpad = (labels != ignore).sum(dim=-1).clamp(min=1)
        loss_per_sample = loss_per_token.sum(dim=-1) / nonpad
        loss_per_sample = torch.where(
            active,
            loss_per_sample,
            torch.zeros_like(loss_per_sample),
        )
        return loss_per_sample.sum() / n_active.clamp(min=1)

    # Only active pool slots contribute; rows whose labels are entirely the ignore label
    # (atomic tail padding) are zeroed too.
    def _csp_cardinality_loss(
        self,
        logits: Tensor,
        *,
        active: Tensor,
        n_active: Tensor,
    ) -> Tensor:
        """Reduce the per-sample cardinality loss exactly like the lm loss."""
        labels = self._pool_labels
        per_sample = csp_cardinality_per_sample(
            logits,
            labels,
            self._csp_groups,
            temperature=self.config.csp_temperature,
        )
        contributes = active & (labels != self.config.ignore_label_id).any(dim=-1)
        per_sample = torch.where(contributes, per_sample, torch.zeros_like(per_sample))
        return per_sample.sum() / n_active.clamp(min=1)

    def _clamp_givens(self, decoded: Tensor, inputs: Tensor) -> Tensor:
        """Return the decoded grid with every given cell forced back to its input digit."""
        given = (inputs >= 2) & (inputs < self.config.dataset.spec.vocab_size)
        return torch.where(given, inputs, decoded)

    # Per-step draw order from the dedicated scramble generator is a contract:
    # ``rand(bs)`` -> ``rand(bs, grid_len)`` -> ``randint(2, vocab_size, (bs, grid_len))``.
    def _scramble_feedback(self, feedback: Tensor, inputs: Tensor) -> Tensor:
        """Corrupt random non-given cells of randomly selected feedback grids."""
        config = self.config
        bs, grid_len = feedback.shape
        slot = (
            torch.rand(bs, device=self.device, generator=self._scramble_gen)
            < config.feedback_scramble_prob
        )
        cell = (
            torch.rand(bs, grid_len, device=self.device, generator=self._scramble_gen)
            < config.feedback_scramble_cells / grid_len
        )
        given = (inputs >= 2) & (inputs < self.config.dataset.spec.vocab_size)
        random_digits = torch.randint(
            2,
            config.dataset.spec.vocab_size,
            (bs, grid_len),
            device=self.device,
            generator=self._scramble_gen,
        )
        return torch.where(
            slot.unsqueeze(-1) & cell & ~given,
            random_digits,
            feedback,
        )

    # With probability ``halt_exploration_prob`` a sample is forced to take a random
    # minimum number of steps in ``[2, max_act]`` before its q-halt head may fire;
    # otherwise the minimum is 1. Draw order from the dedicated halt generator is a
    # contract: ``rand(bs)`` then ``randint(bs)``.
    def _min_halt_steps(self, bs: int, max_act: int) -> Tensor:
        """Per-sample minimum ACT steps before a sample may halt."""
        explore = (
            torch.rand(bs, device=self.device, generator=self._halt_gen)
            < self.config.halt_exploration_prob
        )
        # randint's high is exclusive, so max_act + 1 samples the inclusive
        # range [2, max_act] -- never above the hard cap.
        return torch.where(
            explore,
            torch.randint(
                2,
                max_act + 1,
                (bs,),
                device=self.device,
                generator=self._halt_gen,
            ),
            torch.ones(bs, dtype=torch.long, device=self.device),
        )

    def _force_continue(self, bs: int) -> Tensor:
        """Per-sample mask forcing another ACT step (plain exploration)."""
        return (
            torch.rand(bs, device=self.device, generator=self._halt_gen)
            < self.config.halt_exploration_prob
        )

    def _update_carry_and_halt(
        self,
        *,
        z_slow_out: Tensor,
        z_fast_out: Tensor,
        q_halt: Tensor,
    ) -> None:
        """Persist z_slow/z_fast/h_step and compute the next-step halt mask."""
        self._pool_z_slow.copy_(z_slow_out.to(self._pool_z_slow.dtype))
        self._pool_z_fast.copy_(z_fast_out.to(self._pool_z_fast.dtype))
        self._pool_h_step.add_(1)

        bs = self._batch_size
        max_act = self.config.max_act_steps
        at_max = self._pool_h_step >= max_act
        if self.config.train_q_halt:
            q_positive = q_halt > 0
            if self.config.min_halt_steps_enabled:
                if self.config.halt_exploration_prob > 0:
                    min_halt_steps = self._min_halt_steps(bs, max_act)
                else:
                    min_halt_steps = torch.ones(
                        bs,
                        dtype=torch.long,
                        device=self.device,
                    )
                past_min = self._pool_h_step >= min_halt_steps
                halt = at_max | (q_positive & past_min)
            else:
                halt = at_max | (q_positive & ~self._force_continue(bs))
        else:
            halt = at_max
        # Store the halt mask for the next call's refill.
        self._pool_halted = halt

    # -- Eval internals ----------------------------------------------------------

    def _eval_rollout(
        self,
        media: Tensor,
        step_kwargs: dict[str, Tensor],
    ) -> tuple[Tensor, Tensor, Tensor]:
        """ACT rollout with optional per-row early release."""
        config = self.config
        batch_size = media.shape[0]
        depth_cap = config.eval_act_steps or config.max_act_steps

        rows = torch.arange(batch_size, device=self.device)
        boards = media
        kwargs = step_kwargs
        z_slow, z_fast = self.model.init_z(batch_size)
        feedback = boards if config.feedback else None
        final_logits: Tensor | None = None
        final_q_halt = torch.zeros(batch_size, device=self.device)
        act_used = torch.full(
            (batch_size,),
            depth_cap,
            dtype=torch.int64,
            device=self.device,
        )
        for step in range(1, depth_cap + 1):
            out = self.model.act_step(
                boards,
                z_slow,
                z_fast,
                feedback_ids=feedback,
                **kwargs,
            )
            z_slow = out["z_slow"]
            z_fast = out["z_fast"]
            if feedback is not None:
                feedback = self._clamp_givens(out["logits"].argmax(dim=-1), boards)
            if final_logits is None:
                final_logits = torch.empty(
                    (batch_size, *out["logits"].shape[1:]),
                    dtype=out["logits"].dtype,
                    device=self.device,
                )
            # Every surviving row's CURRENT outputs overwrite its finals, so
            # cap-reaching rows end with step-``depth_cap`` outputs exactly
            # like the fixed-depth eval; released rows stop being overwritten.
            final_logits[rows] = out["logits"]
            final_q_halt[rows] = out["q_halt"].float()
            if not config.eval_halt_exit or step == depth_cap:
                continue
            release = out["q_halt"] >= config.eval_halt_threshold
            if step < config.eval_min_act_steps or not bool(release.any()):
                continue
            act_used[rows[release]] = step
            keep = (~release).nonzero(as_tuple=True)[0]
            if not keep.shape[0]:
                break
            rows = rows[keep]
            boards = boards[keep]
            if feedback is not None:
                feedback = feedback[keep]
            z_slow = z_slow[keep]
            z_fast = z_fast[keep]
            kwargs = {key: value[keep] for key, value in kwargs.items()}
        if final_logits is None:
            raise ValueError("Expected final_logits is not None.")
        return final_logits, final_q_halt, act_used

    # Weighted means of the per-batch step metrics, then the whole-set integer-count
    # exact/cell accuracies overwrite those two keys -- the count ratios are the landed
    # metric definition.
    def _eval_epoch(self) -> dict[str, float]:
        """One pass over the eval split (never augmented)."""
        eval_loader = self.dataset.eval_dataloader()
        total_loss = 0.0
        total_batch_time = 0.0
        total_step_metrics: dict[str, float] = {}
        num_batches = 0
        total_weight = 0
        exact_correct = 0
        n_puzzles = 0
        cell_correct = 0
        cell_total = 0

        eval_start = time.perf_counter()
        batch_start = eval_start
        for raw_batch in eval_loader:
            # Fail fast once the eval pass blows its wall-clock budget rather
            # than producing a slow, incomparable score hours later.
            elapsed_eval = time.perf_counter() - eval_start
            if elapsed_eval > self.config.max_eval_time:
                raise EvalTimeLimitError(
                    f"eval exceeded max_eval_time ({elapsed_eval:.0f}s > "
                    f"{self.config.max_eval_time:.0f}s) after {num_batches} "
                    "batches; reduce eval cost.",
                )
            batch = self.preprocess_batch(raw_batch)
            weight = batch["valid_count"]
            if weight == 0:
                batch_start = time.perf_counter()
                continue
            step_results = self.eval_loss(**batch)
            total_weight += weight
            total_loss += float(step_results["loss"].mean().item()) * weight
            for key, value in step_results.get("metrics", {}).items():
                total_step_metrics[key] = (
                    total_step_metrics.get(key, 0.0)
                    + float(value.mean().item() if isinstance(value, Tensor) else value)
                    * weight
                )
            # Whole-set exact/cell counts from the packed model output (grid
            # preds are the last grid_len columns).
            label = batch["label"]
            grid_len = label.shape[-1]
            preds = step_results["model"][:, -grid_len:].to(torch.int64)[:weight]
            labels = label.to(torch.int64)[:weight]
            valid = labels != self.config.ignore_label_id
            batch_cell_correct = (preds == labels) & valid
            loss_counts = valid.sum(dim=-1)
            seq_correct = (batch_cell_correct.sum(dim=-1) == loss_counts) & (
                loss_counts > 0
            )
            exact_correct += int(seq_correct.sum().item())
            n_puzzles += int((loss_counts > 0).sum().item())
            cell_correct += int(batch_cell_correct.sum().item())
            cell_total += int(loss_counts.sum().item())

            total_batch_time += time.perf_counter() - batch_start
            num_batches += 1
            elapsed_eval = time.perf_counter() - eval_start
            if elapsed_eval > self.config.max_eval_time:
                raise EvalTimeLimitError(
                    f"eval exceeded max_eval_time ({elapsed_eval:.0f}s > "
                    f"{self.config.max_eval_time:.0f}s) after {num_batches} "
                    "batches; reduce eval cost.",
                )
            batch_start = time.perf_counter()

        results: dict[str, float] = {}
        if total_weight > 0:
            results["total_loss"] = total_loss / total_weight
            results["mean_batch_time"] = total_batch_time / num_batches
            for key, value in total_step_metrics.items():
                results[key] = value / total_weight
        results["exact_accuracy"] = exact_correct / max(1, n_puzzles)
        results["cell_accuracy"] = cell_correct / max(1, cell_total)
        return results

    def _maybe_eval(
        self,
        *,
        is_final: bool = False,
        force: bool = False,
    ) -> dict[str, float] | None:
        """Run and publish an eval when due; the final eval emits RESULT."""
        if not force:
            if not math.isfinite(self.config.num_steps_eval):
                return None
            cadence_due = (
                self.global_step != 0
                and self.global_step % self.config.num_steps_eval == 0
            )
            if not is_final and not cadence_due:
                return None
        eval_start = time.perf_counter()
        eval_metrics = self._eval_epoch()
        eval_time = time.perf_counter() - eval_start
        # A cadence eval is not training time: pause the pure-train clock.
        if not is_final:
            self._train_clock_base += eval_time
        scalars = scalar_metrics(eval_metrics)
        payload: dict[str, float] = dict(scalars)
        payload["time"] = eval_time
        for tracker in self._trackers:
            tracker.log_metrics(payload, self.global_step, prefix="eval/")
        if is_final:
            elapsed = time.perf_counter() - self._start_time
            parts = [f"steps={self.global_step}", f"time={elapsed:.1f}s"]
            parts.extend(f"{k}={v:.4f}" for k, v in sorted(scalars.items()))
            logger.info("RESULT: %s", " | ".join(parts))
        else:
            logger.info(
                "Step %d: %s (eval_time=%.3fs)",
                self.global_step,
                scalars,
                eval_time,
            )
        return scalars

    def _warm_eval_compile(self) -> None:
        """Populate eval-only compile caches before timed training/eval."""
        if self.config.eval_warmup_batches <= 0:
            return
        logger.info(
            "Warm eval compile: running %d batch(es).",
            self.config.eval_warmup_batches,
        )
        for batch_index, raw_batch in enumerate(self.dataset.eval_dataloader()):
            if batch_index >= self.config.eval_warmup_batches:
                break
            self.eval_loss(**self.preprocess_batch(raw_batch))

    # torch.compile + bf16 autocast corrupts the repeated-carry eval loop when inductor
    # elides the precision casts autocast inserts; ``emulate_precision_casts=True`` (set
    # before model construction) is the explicit opt-in that keeps eval compiled.
    @contextmanager
    def _eval_compile_disabled(self) -> Generator[None]:
        """Run eval un-compiled unless precision-cast emulation is enabled."""
        if self.config.emulate_precision_casts:
            yield
            return
        old = self.model.config.compile
        self.model.config.compile = False
        try:
            yield
        finally:
            self.model.config.compile = old

    def _autocast(self) -> torch.amp.autocast:
        return torch.amp.autocast(
            device_type=self.device.type,
            dtype=self.config.dtype_autocast,
            enabled=self.config.dtype_autocast is not None,
            cache_enabled=False,
        )

    # -- Loop internals ------------------------------------------------------

    def _next_batch(self) -> PuzzleBatch:
        """Get the next train batch, bumping the epoch on exhaustion."""
        if self._train_loader is None:
            self._train_loader = self.dataset.train_dataloader()
        if self._train_iter is None:
            self._train_iter = iter(self._train_loader)
        for _ in range(2):
            try:
                return self.preprocess_batch(next(self._train_iter))
            except StopIteration:
                self.current_epoch += 1
                self._train_iter = iter(self._train_loader)
        raise RuntimeError("Failed to get next batch after epoch reset")

    def _do_train_step(self, batch: PuzzleBatch) -> None:
        """Execute one training step with timing and cadence logging."""
        step_start = time.perf_counter()
        step_results = self.train_step(**batch)
        step_time = time.perf_counter() - step_start
        if self.local_step == 1:
            # Rebase the pure-train clock AFTER the first step: that step
            # carries the one-time compile and is excluded from "train" time.
            self._train_clock_base = time.perf_counter()
        log_step = (
            self.global_step == 1 or self.global_step % self.config.num_steps_log == 0
        )
        if not log_step:
            return
        loss_value = float(step_results["loss"].mean().item())
        step_metrics = scalar_metrics(step_results.get("metrics", {}))
        elapsed = time.perf_counter() - self._start_time
        extra = " ".join(f"{k}={v:.4f}" for k, v in step_metrics.items())
        line = (
            f"Step {self.global_step}/{self.config.max_steps}: "
            f"loss={loss_value:.4f} step_time={step_time:.3f}s "
            f"elapsed={elapsed:.0f}s"
        )
        logger.info(f"{line} {extra}" if extra else line)
        metrics: dict[str, float] = {
            "total_loss": loss_value,
            "step_time": step_time,
            "time_since_start": elapsed,
            # Pure-train seconds (vs wall ``time_since_start``): the clock a
            # ``max_time_kind="train"`` budget charges against.
            "elapsed": self._train_elapsed(),
            **step_metrics,
        }
        if torch.cuda.is_available():
            metrics["gpu_mem_allocated_gb"] = torch.cuda.max_memory_allocated() / 1e9
            metrics["gpu_mem_reserved_gb"] = torch.cuda.max_memory_reserved() / 1e9
        for tracker in self._trackers:
            tracker.log_metrics(metrics, self.global_step, prefix="train/")

    def _time_limit_reached(self) -> bool:
        """Whether the ``max_time`` cap has elapsed."""
        if self.config.max_time == math.inf:
            return False
        return self._max_time_elapsed() >= self.config.max_time

    def _max_time_elapsed(self) -> float:
        """Seconds charged against ``max_time``, per ``max_time_kind``."""
        if self.config.max_time_kind == "train":
            return self._train_elapsed()
        return time.perf_counter() - self._start_time

    def _train_elapsed(self) -> float:
        """Pure-train seconds: first-step compile and mid-loop evals excluded."""
        return time.perf_counter() - self._train_clock_base

    def _should_log_norms(self) -> bool:
        if not self.config.log_body_norms:
            return False
        step = self.global_step + 1
        return step == 1 or step % self.config.norm_log_interval == 0

    def _close(self) -> None:
        """Release tracker resources at the end of a run."""
        for tracker in self._trackers:
            tracker.close()


# ---------------------------------------------------------------------------
# Private helpers.
# ---------------------------------------------------------------------------


@dataclass(slots=True, kw_only=True)
class _ForwardLoss:
    """Result of the per-step forward + loss, the single train-step seam.

    Attributes:
      loss: Scalar loss to backprop (lm + q-halt + csp terms).
      logits: ``[B, grid_len, V]`` logits for metrics and the probe slice.
      q_halt: ``[B]`` halt logit driving the halting decision.
      z_slow_out: ``[B, S, C]`` updated slow latent for the carry update.
      z_fast_out: ``[B, S, C]`` updated fast latent for the carry update.
      correctness: ``(is_correct, loss_counts, correct)`` when q-halt is
        trained, else None; reused for the halted-only metrics.
      train_metrics: Scalar metrics accumulated this step.

    """

    loss: Tensor
    logits: Tensor
    q_halt: Tensor
    z_slow_out: Tensor
    z_fast_out: Tensor
    correctness: tuple[Tensor, Tensor, Tensor] | None
    train_metrics: dict[str, float | Tensor]
