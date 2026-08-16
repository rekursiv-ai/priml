"""Example: GAN trainer using TrainStep abstraction.

Demonstrates multi-model pattern with separate optimizers/losses.
Uses train_step() for alternating discriminator/generator training.
"""

from __future__ import annotations

from dataclasses import field
from typing import Any, cast

from configgle import Fig, Makeable, PartialConfig
from torch import Tensor

import torch

from priml.loss.gan import AdversarialLoss
from priml.math.schedules import staircase
from priml.model.special import Identity
from priml.train.custom_types import TrainStepOutput
from priml.train.train_step import TrainStep


class GANTrainStep:
    """GAN trainable model using TrainStep abstraction.

    Demonstrates multi-model training with separate optimizers/losses.
    """

    class Config(Fig["GANTrainStep"]):
        """GAN trainable model configuration."""

        generator: Makeable[TrainStep] = field(
            default_factory=lambda: TrainStep.Config(
                model=Identity.Config(),  # Replace with actual generator
                optimizer=PartialConfig(
                    torch.optim.Adam,
                    lr=2e-4,
                    betas=(0.5, 0.999),
                ),
                # A schedule of PROGRESS needs a horizon: left unset, progress
                # is pinned at zero and the rate below never moves.
                train_budget_steps=400,
                learning_rate_scheduler=PartialConfig(staircase, gamma=0.5),
                loss=AdversarialLoss.Config(),
            ),
        )
        """Generator TrainStep configuration."""
        discriminator: Makeable[TrainStep] = field(
            default_factory=lambda: TrainStep.Config(
                model=Identity.Config(),  # Replace with actual discriminator
                optimizer=PartialConfig(
                    torch.optim.Adam,
                    lr=2e-4,
                    betas=(0.5, 0.999),
                ),
                # Matches the generator's: annealing the two on different
                # horizons is a different recipe, not a different setting.
                train_budget_steps=400,
                learning_rate_scheduler=PartialConfig(staircase, gamma=0.5),
            ),
        )
        """Discriminator TrainStep configuration."""
        n_discriminator_steps: int = 1
        """Number of discriminator steps per generator step."""

    def __init__(self, config: Config) -> None:
        """Initialize GAN trainable model."""
        self.generator = config.generator.make()
        self.discriminator = config.discriminator.make()

        if config.n_discriminator_steps <= 0:
            raise ValueError(
                "n_discriminator_steps must be positive, got "
                f"{config.n_discriminator_steps}",
            )
        self.n_discriminator_steps = config.n_discriminator_steps

        # Track steps independently (source of truth for GANTrainStep)
        self.global_step: int = 0
        self.local_step: int = 0

    def preprocess_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Preprocess batch by delegating to the generator's device transfer.

        Args:
          batch: Raw batch from the dataloader.

        Returns:
          batch: Batch with tensors moved to the generator's device.

        """
        return self.generator.preprocess_batch(batch)

    def call_eval(self, **kwargs: Any) -> Any:
        """Evaluation forward pass (generator only)."""
        return self.generator.call_eval(**kwargs)

    def on_epoch_end(self) -> None:
        """Flush partial accumulation in both sub-steps at the epoch boundary."""
        self.generator.on_epoch_end()
        self.discriminator.on_epoch_end()

    def train_step(self, *, media: Tensor, **batch: Any) -> TrainStepOutput:
        """Single GAN training step with discriminator and generator updates.

        Args:
          media: Real media for discriminator training [B, ...].
          **batch: Additional batch data (e.g., noise, conditional info for generator).

        Returns:
          result: Dict with 'loss' (unreduced, per-sample [B]) and 'model' (generated media).

        """
        real_media = media
        batch_size = real_media.shape[0]

        # Train discriminator for n steps. Accumulate the logging loss as a
        # tensor and sync to host once after the loop (a per-step ``.item()``
        # forces a GPU->CPU sync every iteration).
        d_loss_total: Tensor | None = None
        for _ in range(self.n_discriminator_steps):
            with torch.no_grad():
                fake_media_d = self.generator.model(**batch)

            media = torch.cat([real_media, fake_media_d.detach()], dim=0)
            targets = torch.cat(
                [
                    torch.ones(batch_size, 1, device=media.device),
                    torch.zeros(batch_size, 1, device=media.device),
                ],
                dim=0,
            )

            d_loss_result = self.discriminator.train_step(media=media, label=targets)
            # d_loss_result["loss"] is [2*B] (real + fake); gradients already
            # applied by train_step, this term is logging-only.
            step_loss = d_loss_result["loss"].detach().sum()
            d_loss_total = (
                step_loss if d_loss_total is None else d_loss_total + step_loss
            )
        d_loss_sum = d_loss_total.item() if d_loss_total is not None else 0.0

        # Train generator. The generator backward scores ``D(G(z))`` and so
        # flows THROUGH the discriminator's activations -- which is required --
        # but it must not accumulate gradients ONTO the discriminator's
        # parameters (D's own step already zeroed them; stray grads would
        # pollute the next discriminator update). Freeze D's parameters for the
        # generator phase, then restore each to its PRIOR ``requires_grad`` so a
        # blanket restore does not unfreeze params the user intentionally froze.
        requires_grad_backup = {
            name: param.requires_grad
            for name, param in self.discriminator.model.named_parameters()
        }
        self.discriminator.model.requires_grad_(False)
        try:
            fake_media = self.generator.model(**batch)
            fake_logits = self.discriminator.model(fake_media)
            g_loss_result = self.generator.train_step(
                fake_logits=fake_logits,
                fake_media=fake_media,
                real_media=real_media,
                **batch,
            )
        finally:
            for name, param in self.discriminator.model.named_parameters():
                param.requires_grad_(requires_grad_backup[name])

        # Increment GAN step counters
        self.global_step += 1
        self.local_step += 1

        # Carry the generator's full loss output (preserving auxiliary loss
        # keys for logging), then override the primary loss with the combined
        # GAN loss and the model output with the generated media. The
        # discriminator term is a scalar broadcast for logging only.
        d_loss_avg = d_loss_sum / (self.n_discriminator_steps * batch_size * 2)
        result: dict[str, Any] = dict(g_loss_result)
        result["loss"] = g_loss_result["loss"] + d_loss_avg
        result["model"] = fake_media
        return cast(TrainStepOutput, result)

    def train_loss(self, *, media: Tensor, **batch: Any) -> TrainStepOutput:
        """Compute GAN loss in train mode without backprop."""
        real_media = media
        batch_size = real_media.shape[0]

        # Compute discriminator loss (average over n_discriminator_steps);
        # accumulate as a tensor and sync once after the loop.
        d_loss_total: Tensor | None = None
        for _ in range(self.n_discriminator_steps):
            with torch.no_grad():
                fake_media_d = self.generator.call_eval(**batch)

            media = torch.cat([real_media, fake_media_d.detach()], dim=0)
            targets = torch.cat(
                [
                    torch.ones(batch_size, 1, device=media.device),
                    torch.zeros(batch_size, 1, device=media.device),
                ],
                dim=0,
            )
            d_loss_result = self.discriminator.train_loss(media=media, label=targets)
            step_loss = d_loss_result["loss"].detach().sum()
            d_loss_total = (
                step_loss if d_loss_total is None else d_loss_total + step_loss
            )
        d_loss_sum = d_loss_total.item() if d_loss_total is not None else 0.0

        # Compute generator loss
        fake_media = self.generator.model(**batch)
        fake_logits = self.discriminator.model(fake_media)
        g_loss_result = self.generator.train_loss(
            fake_logits=fake_logits,
            fake_media=fake_media,
            real_media=real_media,
            **batch,
        )

        # Combine losses (consistent with train_step), preserving aux keys.
        d_loss_avg = d_loss_sum / (self.n_discriminator_steps * batch_size * 2)
        result: dict[str, Any] = dict(g_loss_result)
        result["loss"] = g_loss_result["loss"] + d_loss_avg
        result["model"] = fake_media
        return cast(TrainStepOutput, result)

    def eval_loss(self, *, media: Tensor, **batch: Any) -> TrainStepOutput:
        """Compute GAN loss in eval mode (uses EMA models if available)."""
        real_media = media
        batch_size = real_media.shape[0]

        # Compute discriminator loss (average over n_discriminator_steps);
        # accumulate as a tensor and sync once after the loop.
        d_loss_total: Tensor | None = None
        for _ in range(self.n_discriminator_steps):
            fake_media_d = self.generator.call_eval(**batch)

            media = torch.cat([real_media, fake_media_d], dim=0)
            targets = torch.cat(
                [
                    torch.ones(batch_size, 1, device=media.device),
                    torch.zeros(batch_size, 1, device=media.device),
                ],
                dim=0,
            )
            d_loss_result = self.discriminator.eval_loss(media=media, label=targets)
            step_loss = d_loss_result["loss"].detach().sum()
            d_loss_total = (
                step_loss if d_loss_total is None else d_loss_total + step_loss
            )
        d_loss_sum = d_loss_total.item() if d_loss_total is not None else 0.0

        # Compute generator loss
        fake_media = self.generator.call_eval(**batch)
        fake_logits = self.discriminator.call_eval(media=fake_media)
        g_loss_result = self.generator.eval_loss(
            fake_logits=fake_logits,
            fake_media=fake_media,
            real_media=real_media,
            **batch,
        )

        # Combine losses (consistent with train_step), preserving aux keys.
        d_loss_avg = d_loss_sum / (self.n_discriminator_steps * batch_size * 2)
        result: dict[str, Any] = dict(g_loss_result)
        result["loss"] = g_loss_result["loss"] + d_loss_avg
        result["model"] = fake_media
        return cast(TrainStepOutput, result)

    def state_dict(self) -> dict[str, Any]:
        """Get checkpoint state for both generator and discriminator."""
        return {
            "generator": self.generator.state_dict(),
            "discriminator": self.discriminator.state_dict(),
            "global_step": self.global_step,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load checkpoint state for both generator and discriminator."""
        self.generator.load_state_dict(state_dict["generator"])
        self.discriminator.load_state_dict(state_dict["discriminator"])
        self.global_step = state_dict["global_step"]
        self.local_step = 0
