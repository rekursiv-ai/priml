"""GAN trainer composed of two TrainSteps: a generator and a discriminator.

Each sub-step owns its model, optimizer, schedule, and accumulation; this class
owns only the order of their updates. Every forward goes through a sub-step's
own entry point, so autocast, compilation, timers, and device placement apply
to the GAN exactly as they do to a supervised run.
"""

from __future__ import annotations

from dataclasses import field
from functools import partial
from typing import TYPE_CHECKING, TypedDict, cast, override

import math

from configgle import Fig, Makeable, PartialConfig
from torch import Tensor, nn

import torch

from priml.loss.gan import AdversarialLoss
from priml.math.schedules import staircase
from priml.train.custom_types import TrainStepOutput
from priml.train.train_step import TrainStep


if TYPE_CHECKING:
    from collections.abc import Mapping

    from priml.timer import CheckpointableStepTimer
    from priml.train.custom_types import PhaseTimerProtocol


class GANTrainStep:
    """Alternating discriminator and generator updates on one batch.

    Per call: the generator runs ONCE on ``**batch`` (e.g. ``noise``); the
    discriminator then takes ``n_discriminator_steps`` updates on the real
    ``media`` beside that detached fake; finally the generator is scored by the
    frozen, eval-mode discriminator and takes its own update through
    :meth:`TrainStep.train_on_output`.

    The discriminator's loss receives ``label`` shaped like its output (1 for
    real, 0 for fake), so a patch discriminator works unchanged. The
    generator's loss receives ``fake_logits``, ``fake_media`` and
    ``real_media`` beside the batch (see ``loss.gan.AdversarialLoss``).
    """

    class Config(Fig["GANTrainStep"]):
        """GAN configuration: two sub-steps and their update ratio."""

        generator: Makeable[TrainStep] = field(
            default_factory=lambda: TrainStep.Config(
                model=NoiseToMedia.Config(),
                optimizer=PartialConfig(
                    torch.optim.Adam,
                    lr=2e-4,
                    betas=(0.5, 0.999),
                ),
                # A schedule of PROGRESS needs a horizon: left unset, progress
                # is pinned at zero and the rate below never moves.
                train_budget_steps=400,
                lr_schedule=PartialConfig(staircase, gamma=0.5),
                loss=AdversarialLoss.Config(),
            ),
        )
        """Generator sub-step; its model maps the batch (minus ``media``) to media.

        Its ``loss`` receives ``fake_logits``, ``fake_media`` and ``real_media``
        beside the batch."""

        discriminator: Makeable[TrainStep] = field(
            default_factory=lambda: TrainStep.Config(
                model=MediaCritic.Config(),
                optimizer=PartialConfig(
                    torch.optim.Adam,
                    lr=2e-4,
                    betas=(0.5, 0.999),
                ),
                # Matches the generator's: annealing the two on different
                # horizons is a different recipe, not a different setting.
                train_budget_steps=400,
                lr_schedule=PartialConfig(staircase, gamma=0.5),
            ),
        )
        """Discriminator sub-step; its model takes ``media=`` and returns logits.

        Its ``loss`` reads ``label``, shaped like the logits."""

        n_discriminator_steps: int = 1
        """Discriminator updates per generator update."""

    def __init__(self, config: Config) -> None:
        """Build both sub-steps and check they can share one batch and checkpoint.

        Raises:
          ValueError: ``n_discriminator_steps`` is not positive; the sub-steps
            sit on different devices; or the discriminator would be mid-
            accumulation whenever the generator completes an update.

        """
        if config.n_discriminator_steps <= 0:
            raise ValueError(
                "n_discriminator_steps must be positive, got "
                f"{config.n_discriminator_steps}",
            )
        self.generator = config.generator.make()
        self.discriminator = config.discriminator.make()
        self.n_discriminator_steps = config.n_discriminator_steps

        if self.generator.device != self.discriminator.device:
            raise ValueError(
                "generator and discriminator must share one device, got "
                f"{self.generator.device} and {self.discriminator.device}; "
                "the batch is moved once, to that device.",
            )
        # ``global_step`` follows the generator, so the loop saves only when it
        # advances. The discriminator must then be on an update boundary too,
        # or that save raises on its pending accumulation.
        d_micro_per_g_update = (
            self.n_discriminator_steps * self.generator.accumulate_grad_batches
        )
        if d_micro_per_g_update % self.discriminator.accumulate_grad_batches:
            raise ValueError(
                "discriminator.accumulate_grad_batches "
                f"({self.discriminator.accumulate_grad_batches}) must divide "
                "n_discriminator_steps * generator.accumulate_grad_batches "
                f"({d_micro_per_g_update}), so both sub-steps finish an update "
                "together and a checkpoint never lands mid-accumulation.",
            )

    @property
    def global_step(self) -> int:
        """Generator optimizer updates: the GAN's step, as the loop counts it."""
        return self.generator.global_step

    @property
    def timer(self) -> PhaseTimerProtocol | None:
        """The loop's phase timer, shared by both sub-steps."""
        return self.generator.timer

    @timer.setter
    def timer(self, timer: PhaseTimerProtocol | None) -> None:
        self.generator.timer = timer
        self.discriminator.timer = timer

    def bind_epoch_timer(self, timer: CheckpointableStepTimer) -> None:
        """Bind the loader's epoch timer to both sub-steps' schedules."""
        self.generator.bind_epoch_timer(timer)
        self.discriminator.bind_epoch_timer(timer)

    def preprocess_batch(self, batch: dict[str, object]) -> dict[str, object]:
        """Move every tensor in the batch to the device both sub-steps share.

        Args:
          batch: Raw batch from the dataloader.

        Returns:
          batch: Batch with tensors on the sub-steps' device.

        """
        return self.generator.preprocess_batch(batch)

    def call_eval(self, **kwargs: object) -> object:
        """Run the generator's evaluation forward pass.

        Args:
          **kwargs: Preprocessed batch, minus ``media``.

        Returns:
          media: The generator's output.

        """
        return self.generator.call_eval(**kwargs)

    def on_epoch_end(self) -> None:
        """Flush partial accumulation in both sub-steps at the epoch boundary."""
        self.generator.on_epoch_end()
        self.discriminator.on_epoch_end()

    def train_step(self, **preprocessed_batch: object) -> TrainStepOutput:
        """Update the discriminator, then the generator, on one batch.

        Args:
          **preprocessed_batch: Real ``media`` [B, ...] plus the generator's
            inputs (e.g. ``noise``).

        Returns:
          result: The generator's loss output with "model" the generated media
            and "metrics" carrying ``d_loss`` and the discriminator's metrics
            under a ``d_`` prefix.

        """
        media, batch = _split_media(preprocessed_batch)
        fake_media = _tensor(self.generator(**batch), "generator output")
        # One fake serves every discriminator step: the generator has not moved
        # between them, so regenerating would repeat identical work.
        both = _real_and_fake(media, fake_media.detach())

        d_losses: list[Tensor] = []
        d_metrics: dict[str, float | Tensor] = {}
        for _ in range(self.n_discriminator_steps):
            logits = self.discriminator(media=both)
            d_result = self.discriminator.train_on_output(
                logits,
                closure=partial(self._recompute_discriminator_loss, both),
                media=both,
                label=_real_then_fake(_tensor(logits, "discriminator output")),
            )
            d_losses.append(d_result["loss"].detach().mean())
            d_metrics = d_result.get("metrics", {})

        g_result = self.generator.train_on_output(
            fake_media,
            closure=lambda: self._recompute_generator_loss(media, batch),
            **self._generator_loss_inputs(fake_media, media, batch),
        )
        return _with_discriminator(g_result, torch.stack(d_losses).mean(), d_metrics)

    def train_loss(self, **preprocessed_batch: object) -> TrainStepOutput:
        """Compute both losses in train mode, without an update.

        Args:
          **preprocessed_batch: As :meth:`train_step`.

        Returns:
          output: As :meth:`train_step`.

        """
        media, batch = _split_media(preprocessed_batch)
        fake_media = _tensor(self.generator(**batch), "generator output")
        both = _real_and_fake(media, fake_media.detach())
        logits = _tensor(self.discriminator(media=both), "discriminator output")
        d_loss = self.discriminator.loss(
            logits,
            media=both,
            label=_real_then_fake(logits),
        )
        g_result = self.generator.loss(
            fake_media,
            **self._generator_loss_inputs(fake_media, media, batch),
        )
        return _with_discriminator(
            {**g_result, "model": fake_media},
            d_loss["loss"].detach().mean(),
            {},
        )

    def eval_loss(self, **preprocessed_batch: object) -> TrainStepOutput:
        """Compute both losses in eval mode (EMA weights where configured).

        Args:
          **preprocessed_batch: As :meth:`train_step`.

        Returns:
          output: As :meth:`train_step`.

        """
        media, batch = _split_media(preprocessed_batch)
        fake_media = _tensor(self.generator.call_eval(**batch), "generator output")
        both = _real_and_fake(media, fake_media)
        logits = _tensor(
            self.discriminator.call_eval(media=both),
            "discriminator output",
        )
        d_loss = self.discriminator.loss(
            logits,
            media=both,
            label=_real_then_fake(logits),
        )
        fake_logits = self.discriminator.call_eval(media=fake_media)
        g_result = self.generator.loss(
            fake_media,
            fake_logits=fake_logits,
            fake_media=fake_media,
            real_media=media,
            **batch,
        )
        return _with_discriminator(
            {**g_result, "model": fake_media},
            d_loss["loss"].mean(),
            {},
        )

    class StateDict(TypedDict):
        """Both sub-steps' state; the GAN's step is the generator's."""

        generator: TrainStep.StateDict
        discriminator: TrainStep.StateDict

    def state_dict(self) -> StateDict:
        """Get checkpoint state for both sub-steps.

        Returns:
          state: The generator and discriminator states.

        """
        return {
            "generator": self.generator.state_dict(),
            "discriminator": self.discriminator.state_dict(),
        }

    def load_state_dict(
        self,
        state_dict: Mapping[str, object],
        *,
        strict: bool = True,
        load_optimizer: bool = True,
    ) -> None:
        """Load checkpoint state for both sub-steps.

        Args:
          state_dict: State as returned by :meth:`state_dict`.
          strict: Require each model's keys to match exactly.
          load_optimizer: Restore optimizers and EMA; off for finetuning a
            changed architecture.

        """
        state = cast(GANTrainStep.StateDict, state_dict)
        self.generator.load_state_dict(
            state["generator"],
            strict=strict,
            load_optimizer=load_optimizer,
        )
        self.discriminator.load_state_dict(
            state["discriminator"],
            strict=strict,
            load_optimizer=load_optimizer,
        )

    def _generator_loss_inputs(
        self,
        fake_media: Tensor,
        real_media: Tensor,
        batch: Mapping[str, object],
    ) -> dict[str, object]:
        """Score the fake with the frozen discriminator; the generator loss's kwargs."""
        return {
            "fake_logits": self.discriminator.call_frozen(media=fake_media),
            "fake_media": fake_media,
            "real_media": real_media,
            **batch,
        }

    def _recompute_generator_loss(
        self,
        real_media: Tensor,
        batch: Mapping[str, object],
    ) -> Tensor:
        """Re-run the generator objective for a closure-based optimizer."""
        fake_media = _tensor(self.generator(**batch), "generator output")
        inputs = self._generator_loss_inputs(fake_media, real_media, batch)
        return self.generator.loss(fake_media, **inputs)["loss"].sum()

    def _recompute_discriminator_loss(self, both: Tensor) -> Tensor:
        """Re-run the discriminator objective for a closure-based optimizer."""
        logits = _tensor(self.discriminator(media=both), "discriminator output")
        loss = self.discriminator.loss(
            logits,
            media=both,
            label=_real_then_fake(logits),
        )
        return loss["loss"].sum()


class NoiseToMedia(nn.Module):
    """Linear generator from a noise vector of any width to media of a set shape."""

    class Config(Fig["NoiseToMedia"], make_with_kwargs=True):
        media_shape: tuple[int, ...] = (3, 8, 8)
        """Shape of one generated sample, without the batch axis.

        Must match the real ``media`` the dataset yields; the GAN raises on
        the first batch otherwise."""

    def __init__(self, media_shape: tuple[int, ...]) -> None:
        super().__init__()
        self.media_shape = media_shape
        self.proj = nn.LazyLinear(math.prod(media_shape))

    @override
    def forward(self, noise: Tensor) -> Tensor:
        return self.proj(noise.flatten(1)).unflatten(-1, self.media_shape)


class MediaCritic(nn.Module):
    """Linear discriminator: one logit per sample, from media of any shape."""

    class Config(Fig["MediaCritic"]): ...

    def __init__(self, config: Config) -> None:
        del config
        super().__init__()
        self.proj = nn.LazyLinear(1)

    @override
    def forward(self, *, media: Tensor) -> Tensor:
        return self.proj(media.flatten(1))


def _real_and_fake(real: Tensor, fake: Tensor) -> Tensor:
    """Stack real over fake media for one discriminator pass."""
    if real.shape[1:] != fake.shape[1:]:
        raise ValueError(
            f"generator output has sample shape {tuple(fake.shape[1:])} but the "
            f"real media has {tuple(real.shape[1:])}; set the generator model's "
            "output shape (e.g. NoiseToMedia.Config.media_shape) to match the data.",
        )
    return torch.cat([real, fake], dim=0)


def _split_media(
    preprocessed_batch: Mapping[str, object],
) -> tuple[Tensor, dict[str, object]]:
    """Separate the real ``media`` from the generator's inputs."""
    batch = dict(preprocessed_batch)
    return _tensor(batch.pop("media", None), "batch 'media'"), batch


def _tensor(value: object, what: str) -> Tensor:
    """Narrow a sub-model's output to the Tensor the GAN's objective needs."""
    if not isinstance(value, Tensor):
        raise TypeError(
            f"GANTrainStep needs a Tensor {what}, got {type(value).__name__}.",
        )
    return value


def _real_then_fake(logits: Tensor) -> Tensor:
    """Targets shaped like ``logits``: 1 for the leading real half, 0 for the fakes."""
    real, fake = logits.chunk(2, dim=0)
    return torch.cat([torch.ones_like(real), torch.zeros_like(fake)], dim=0)


# ``loss`` stays the generator's alone: the two objectives pull in opposite directions,
# so their sum has no direction a ``best_metric`` could follow.
def _with_discriminator(
    g_result: Mapping[str, object],
    d_loss: Tensor,
    d_metrics: Mapping[str, float | Tensor],
) -> TrainStepOutput:
    """Attach the discriminator's loss and metrics to the generator's output."""
    result = cast(TrainStepOutput, dict(g_result))
    metrics = result.setdefault("metrics", {})
    metrics["d_loss"] = d_loss
    metrics.update({f"d_{key}": value for key, value in d_metrics.items()})
    return result
