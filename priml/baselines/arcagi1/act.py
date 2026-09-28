"""Adaptive computation time for the reference TRM recipes.

Each training call advances every occupied pool slot by one forward, carrying
its latents; a slot leaves when its halt head fires or it reaches the step cap.
Two pieces vary independently, so each is its own slot on the train step:

* the POOL -- how incoming puzzles are seated. :class:`AtomicPool` takes one
  pool-width batch per call and seats it only where a slot just halted.
* the HALTING -- :class:`HaltTraining`: whether the halt head is trained, how
  much its loss weighs, and which exploration keeps it from learning only
  from its own decisions (:class:`SampledMinimum`, :class:`ForcedContinue`).

A pool optionally carries a decoded-grid feedback (:class:`FeedbackCarry`):
each slot's last detached argmax, fed back through the model's
``PredictionFeedback`` channel, optionally corrupted as a repair curriculum.

Draw order from each dedicated generator is a reproducibility contract: the
halt and corruption generators are seeded independently of the global RNG, so
two runs from identical weights stay identical whatever else drew between.
"""

from __future__ import annotations

from dataclasses import field
from typing import TYPE_CHECKING, Protocol, cast, override

import abc
import math

from configgle import Fig, Makeable
from torch import Tensor

import torch

from priml.baselines.sudoku.embedding import PredictionFeedback


if TYPE_CHECKING:
    from priml.baselines.sudoku.model import SudokuNet


class Exploration(Protocol):
    """Decide which slots whose halt head fired may actually halt."""

    def __call__(
        self,
        fired: Tensor,
        *,
        steps: Tensor,
        max_steps: int,
        generator: torch.Generator,
    ) -> Tensor:
        """Apply to the input."""
        ...


class SampledMinimum:
    """Hold a random fraction of slots to a sampled minimum depth.

    With probability ``prob`` a slot draws a minimum in ``[2, max_steps]``
    before its halt head may fire; otherwise the minimum is one step. Draws
    ``rand`` then ``randint``, both for every slot, and nothing at all when
    ``prob`` is 0.
    """

    class Config(Fig["SampledMinimum"]):
        """Exploration rate."""

        prob: float = 0.1
        """Chance a slot is held to a sampled minimum."""

    def __init__(self, config: Config) -> None:
        self.prob = config.prob

    def __call__(
        self,
        fired: Tensor,
        *,
        steps: Tensor,
        max_steps: int,
        generator: torch.Generator,
    ) -> Tensor:
        """Return ``fired`` masked to slots that have run their minimum."""
        n = fired.shape[0]
        device = fired.device
        if self.prob > 0:
            explore = torch.rand(n, device=device, generator=generator) < self.prob
            minimum = torch.where(
                explore,
                # randint's high is exclusive: this samples [2, max_steps].
                torch.randint(
                    2,
                    max_steps + 1,
                    (n,),
                    device=device,
                    generator=generator,
                ),
                torch.ones(n, dtype=torch.long, device=device),
            )
        else:
            minimum = torch.ones(n, dtype=torch.long, device=device)
        return fired & (steps >= minimum)


class ForcedContinue:
    """Override a random fraction of halt decisions with one more step."""

    class Config(Fig["ForcedContinue"]):
        """Exploration rate."""

        prob: float = 0.1
        """Chance a firing slot is kept for another step."""

    def __init__(self, config: Config) -> None:
        self.prob = config.prob

    def __call__(
        self,
        fired: Tensor,
        *,
        steps: Tensor,
        max_steps: int,
        generator: torch.Generator,
    ) -> Tensor:
        """Return ``fired`` with a random ``prob`` fraction cleared."""
        del steps, max_steps
        keep = (
            torch.rand(fired.shape[0], device=fired.device, generator=generator)
            < self.prob
        )
        return fired & ~keep


class HaltTraining:
    """Train the halt head to predict "this grid is already correct"."""

    class Config(Fig["HaltTraining"]):
        """Loss weight, exploration, and the exploration generator's seed."""

        weight: float = 0.05
        """Halt loss weight relative to the token loss."""

        exploration: Makeable[Exploration] = field(
            default_factory=ForcedContinue.Config,
        )
        """Which firing slots may halt."""

        seed: int = 0
        """Seed for the dedicated exploration generator."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.weight = config.weight
        self.exploration: Exploration = config.exploration.make()
        self.generator = torch.Generator()
        self.generator.manual_seed(config.seed)

    def to(self, device: torch.device) -> None:
        """Rebuild the generator on ``device`` from the seed."""
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(self.config.seed)


class FeedbackCarry:
    """Per-slot decoded grid, fed back as the next forward's feedback input.

    Fresh slots restart from their input grid; after every forward the grid
    becomes the detached argmax, each cell replaced by a uniform token in
    ``[corruption_low, corruption_high)`` with probability
    ``corruption_rate``. The objective is unchanged: corruption is a repair
    curriculum, not a target.
    """

    class Config(Fig["FeedbackCarry"]):
        """Corruption rate, its token range, and its generator's seed."""

        corruption_rate: float = 0.0
        """Per-cell replacement probability; 0 feeds back the clean grid."""

        corruption_low: int = 2
        """First replacement token (ARC's first color)."""

        corruption_high: int = 12
        """One past the last replacement token (ARC's vocabulary size)."""

        seed: int = 0
        """Seed for the dedicated corruption generator."""

    def __init__(self, config: Config) -> None:
        rate = config.corruption_rate
        if math.isnan(rate) or rate < 0.0 or rate > 1.0:
            raise ValueError(
                f"corruption_rate must be in [0, 1]; got {config.corruption_rate}.",
            )
        self.config = config
        self.generator = torch.Generator()
        self.generator.manual_seed(config.seed)

    def to(self, device: torch.device) -> None:
        """Rebuild the generator on ``device`` from the seed."""
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(self.config.seed)

    def corrupt(self, grid: Tensor) -> Tensor:
        """Replace random cells with uniform tokens; draws ``rand`` then ``randint``.

        Args:
          grid: Decoded token grid.

        Returns:
          corrupted: ``grid`` with the drawn cells replaced.

        """
        config = self.config
        hit = (
            torch.rand(grid.shape, device=grid.device, generator=self.generator)
            < config.corruption_rate
        )
        tokens = torch.randint(
            config.corruption_low,
            config.corruption_high,
            grid.shape,
            device=grid.device,
            generator=self.generator,
        )
        return torch.where(hit, tokens, grid)


class TrmPool(abc.ABC):
    """Slot state shared by every seating policy; see the module docstring."""

    class Config(Fig["TrmPool"]):
        """Pool geometry and the optional feedback carry."""

        batch_size: int = 192
        """Slots in the pool."""

        max_steps: int = 16
        """Forwards a puzzle may take before it is forced out."""

        feedback: FeedbackCarry.Config | None = None
        """Decoded-grid feedback; requires a ``PredictionFeedback`` channel."""

        grid_len: int = -1
        """Grid tokens per puzzle; inherited from the model."""

        seq_len: int = -1
        """Latent sequence length (prefix + grid); inherited from the model."""

        channels_hidden: int = -1
        """Latent width; inherited from the model."""

        dtype: torch.dtype | None = None
        """Carried-latent dtype; inherited from the model's storage dtype.

        Float32 masters carry float32 latents, so backward through an autocast
        forward does not compound half-precision rounding across steps."""

    def __init__(self, config: Config) -> None:
        if min(config.grid_len, config.seq_len, config.channels_hidden) <= 0:
            raise ValueError(
                "grid_len, seq_len, and channels_hidden must be positive; they are "
                "normally inherited from the model during finalize. Got "
                f"{config.grid_len}, {config.seq_len}, {config.channels_hidden}.",
            )
        self.config = config
        bs = config.batch_size
        self.inputs = torch.zeros(bs, config.grid_len, dtype=torch.long)
        self.labels = torch.zeros_like(self.inputs)
        self.z_slow = torch.zeros(
            bs,
            config.seq_len,
            config.channels_hidden,
            dtype=config.dtype,
        )
        self.z_fast = torch.zeros_like(self.z_slow)
        self.steps = torch.zeros(bs, dtype=torch.long)
        self.active = torch.zeros(bs, dtype=torch.bool)
        # All halted, so the first atomic refill seats every slot.
        self.halted = torch.ones(bs, dtype=torch.bool)
        self.puzzle_ids = torch.zeros(bs, dtype=torch.int32)
        self.carry = None if config.feedback is None else config.feedback.make()
        self.feedback = torch.zeros_like(self.inputs)

    def to(self, device: torch.device) -> None:
        """Move every slot tensor and generator to ``device``.

        Args:
          device: Target device.

        """
        for name in (
            "inputs",
            "labels",
            "z_slow",
            "z_fast",
            "steps",
            "active",
            "halted",
            "puzzle_ids",
            "feedback",
        ):
            setattr(self, name, cast(Tensor, getattr(self, name)).to(device))
        if self.carry is not None:
            self.carry.to(device)

    @abc.abstractmethod
    def refill(
        self,
        net: SudokuNet,
        *,
        media: Tensor,
        labels: Tensor,
        valid_count: int,
        puzzle_ids: Tensor | None,
        ignore_label_id: int,
    ) -> Tensor:
        """Seat incoming puzzles; return the ``[B]`` mask of slots that train.

        Args:
          net: Model whose initial latents a seated slot starts from.
          media: ``[B, grid_len]`` incoming puzzles.
          labels: ``[B, grid_len]`` their solutions.
          valid_count: Real rows; the rest are loader padding.
          puzzle_ids: ``[B]`` task ids, when the batch carries them.
          ignore_label_id: Label the loss skips.

        Returns:
          active: ``[B]`` mask of slots that train this call.

        """

    @abc.abstractmethod
    def update_carry(self, *, z_slow: Tensor, z_fast: Tensor, active: Tensor) -> None:
        """Store this forward's latents and count the step.

        Args:
          z_slow: Updated slow latent.
          z_fast: Updated fast latent.
          active: Slots that trained.

        """

    @abc.abstractmethod
    def release(self, net: SudokuNet, *, halt: Tensor, active: Tensor) -> None:
        """Act on this step's halt decision.

        Args:
          net: Model whose initial latents a refilled slot starts from.
          halt: ``[B]`` slots that stop after this step.
          active: Slots that trained.

        """

    def halted_this_step(self) -> Tensor | None:
        """Slots that halted this step, when metrics score only those."""
        return None

    def set_feedback(self, net: SudokuNet, grid: Tensor | None) -> None:
        """Hand ``grid`` to the model's feedback channel for the next forward."""
        for channel in net.embedding.channels:
            if isinstance(channel, PredictionFeedback):
                channel.set_feedback(grid)

    def decode_feedback(self, logits: Tensor, *, media: Tensor) -> Tensor:
        """Return the grid fed back after a forward: the plain argmax.

        Args:
          logits: ``[B, grid_len, V]`` predictions.
          media: ``[B, grid_len]`` the puzzles those predictions answer.

        Returns:
          grid: ``[B, grid_len]`` token grid for the next forward.

        """
        del media
        return logits.argmax(dim=-1)

    def advance_feedback(
        self,
        carry: FeedbackCarry,
        logits: Tensor,
    ) -> Tensor | float | None:
        """Replace the carried grid with the (corrupted) argmax.

        Args:
          carry: This pool's feedback carry.
          logits: ``[B, grid_len, V]`` this step's predictions.

        Returns:
          changed: Fraction of cells the corruption changed; 0.0 when clean;
            ``None`` from a pool that reports no such metric.

        """
        decoded = self.decode_feedback(logits, media=self.inputs).detach()
        changed: Tensor | float = 0.0
        if carry.config.corruption_rate > 0:
            corrupted = carry.corrupt(decoded)
            changed = (corrupted != decoded).float().mean()
            decoded = corrupted
        self.feedback = decoded
        return changed

    def halt_mask(self, halt: Tensor, *, halting: HaltTraining | None) -> Tensor:
        """Which slots stop after this step: at the cap, or fired and allowed.

        Args:
          halt: ``[B]`` halt logits.
          halting: Halt training; ``None`` halts only at the cap.

        Returns:
          halt: ``[B]`` slots that stop.

        """
        at_cap = self.steps >= self.config.max_steps
        if halting is None:
            return at_cap
        return at_cap | halting.exploration(
            halt > 0,
            steps=self.steps,
            max_steps=self.config.max_steps,
            generator=halting.generator,
        )


class AtomicPool(TrmPool):
    """Seat one pool-width batch per call, only where a slot just halted.

    Every slot trains every call, including ones that were just seated; a
    batch shorter than the pool is padded by the loader and its tail labels
    are masked to the ignore label.
    """

    class Config(TrmPool.Config):
        """Atomic seating."""

    @override
    def refill(
        self,
        net: SudokuNet,
        *,
        media: Tensor,
        labels: Tensor,
        valid_count: int,
        puzzle_ids: Tensor | None,
        ignore_label_id: int,
    ) -> Tensor:
        bs = self.config.batch_size
        if media.shape[0] != bs:
            raise ValueError(
                f"atomic seating requires a batch of {bs}; got {media.shape[0]}.",
            )
        incoming = media.clone().to(torch.long)
        incoming_labels = labels.clone().to(torch.long)
        if valid_count < bs:
            incoming_labels[valid_count:] = ignore_label_id
        halted = self.halted
        seat = halted.unsqueeze(-1)
        self.inputs = torch.where(seat, incoming, self.inputs)
        self.labels = torch.where(seat, incoming_labels, self.labels)
        if puzzle_ids is not None:
            ids = puzzle_ids.to(torch.int32)
            if valid_count < bs:
                ids = ids.clone()
                ids[valid_count:] = 0
            self.puzzle_ids = torch.where(halted, ids, self.puzzle_ids)
        z_slow, z_fast = net.init_latents(bs)
        seat_latent = halted.view(-1, 1, 1)
        self.z_slow = torch.where(
            seat_latent,
            z_slow.to(self.z_slow.dtype),
            self.z_slow,
        )
        self.z_fast = torch.where(
            seat_latent,
            z_fast.to(self.z_fast.dtype),
            self.z_fast,
        )
        self.steps = torch.where(halted, torch.zeros_like(self.steps), self.steps)
        if self.carry is not None:
            self.feedback = torch.where(seat, self.inputs, self.feedback)
        return torch.ones_like(halted)

    @override
    def update_carry(self, *, z_slow: Tensor, z_fast: Tensor, active: Tensor) -> None:
        del active
        self.z_slow.copy_(z_slow.to(self.z_slow.dtype))
        self.z_fast.copy_(z_fast.to(self.z_fast.dtype))
        self.steps.add_(1)

    @override
    def release(self, net: SudokuNet, *, halt: Tensor, active: Tensor) -> None:
        del net, active
        self.halted = halt

    @override
    def halted_this_step(self) -> Tensor | None:
        return self.halted
