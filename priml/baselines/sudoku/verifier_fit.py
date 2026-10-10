"""The verifier's dedicated training run and its objective.

:class:`VerifierFit` is the DEDICATED training run: plain AdamW + cosine
annealing, bf16 autocast, gradient clip 1.0, no EMA / Muon / lr_scale.
Deliberately not the generator ``Trainer`` (its schedule cannot express this
recipe; the committee's zero-false-accept property depends on it).

Checkpoint layout (written by :meth:`VerifierFit.run`, consumed by
:class:`~priml.baselines.sudoku.verifier.VerifierAcceptor`):
``{"step": {"model": <state_dict>, "global_step": <int>}}`` at
``<scratch>/runs/<experiment_name>/checkpoints/step_<max_steps:08d>.pt``.
"""

from __future__ import annotations

from dataclasses import field
from pathlib import Path
from typing import TYPE_CHECKING, Self, override

import json
import logging
import math

from configgle import Fig, Makes
from torch import Tensor
from torch.nn import functional
from tqdm import tqdm

import torch

from priml.baselines.sudoku.verifier import SudokuVerifier
from priml.baselines.sudoku.verifier_data import VerifierData
from priml.paths import resolve_working_dir
from priml.runtime import get_device


if TYPE_CHECKING:
    from collections.abc import Mapping


logger = logging.getLogger(__name__)


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


class DenoisingLoss(VerifierLoss):
    """:class:`VerifierLoss` plus the masked-digit cross-entropy, when present.

    Reads a :class:`DenoisingVerifier` training output: the denoising term is
    the mean cross-entropy over masked cells whose candidate digit is correct
    (a wrong digit is not a target). A plain ``[B, 82]`` output scores as
    :class:`VerifierLoss` does.
    """

    class Config(Makes["DenoisingLoss"], VerifierLoss.Config):
        """The verifier loss plus the denoising weight."""

        denoise_weight: float = 1.0
        """Weight of the masked-digit cross-entropy term."""

    config: Config
    """Narrowed from the parent's declaration for the denoising weight."""

    @override
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
          output: ``[B, 82]`` verifier logits, optionally followed by the
            denoising logits and targets.
          label: ``[B]`` grid labels (1 = candidate is the solution).
          cell_label: ``[B, 81]`` per-cell wrongness labels.
          cell_mask: ``[B, 81]`` mask over cells entering the auxiliary term.

        Returns:
          loss: ``[B]`` per-sample loss.

        """
        loss = super().__call__(
            output[:, :82],
            label=label,
            cell_label=cell_label,
            cell_mask=cell_mask,
        )
        if output.shape[-1] == 82:
            return loss
        logits = output[:, 82 : 82 + 81 * 9].reshape(-1, 81, 9).float()
        target = output[:, 82 + 81 * 9 :].long()
        scored = (target >= 0) & (cell_label < 0.5)
        per_cell = functional.cross_entropy(
            logits.reshape(-1, 9),
            target.clamp(min=0).reshape(-1),
            reduction="none",
        ).reshape(-1, 81)
        denoise = (per_cell * scored.float()).sum(-1) / scored.float().sum(-1).clamp(
            min=1,
        )
        return loss + self.config.denoise_weight * denoise


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

        dtype_autocast: torch.dtype | None = torch.bfloat16
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
        if cfg.dtype_autocast is not None:
            # cache_enabled=False preserves exact numerics across forwards.
            with torch.amp.autocast(
                device_type=self.device.type,
                dtype=cfg.dtype_autocast,
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
