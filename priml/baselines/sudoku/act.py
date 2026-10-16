"""Adaptive computation time: a pool of puzzles, each taking one step at a time.

A recurrent solver should spend more steps on a hard puzzle than an easy one.
That is awkward at fixed batch shape -- the batch cannot shrink as puzzles
finish -- so instead the batch IS a pool of slots. Each training call advances
every occupied slot by one forward, carrying its latents; a slot leaves when
its halt head fires or it reaches the step cap, and the next puzzle takes it.

Four pieces vary independently, so each is its own slot on the pool:

* the SEATING -- how incoming puzzles reach a slot. :class:`AtomicPool` takes
  one pool-width batch per call and seats it only where a slot just halted;
  :class:`StreamingPool` queues every incoming puzzle and seats it in the next
  free slot.
* the HALTING -- :class:`HaltTraining`: whether the halt head is trained, how
  much its loss weighs, and which exploration keeps it from learning only from
  its own decisions (:class:`SampledMinimum`, :class:`ForcedContinue`).
* the START -- the latents a newly seated slot begins from: the model's
  learned initial latents (:class:`LearnedStart`) or zeros
  (:class:`ZeroStart`).
* the FEEDBACK -- :class:`FeedbackCarry`: each slot's last detached argmax,
  fed back through the model's feedback channel, with the puzzle's clues
  optionally restored and the grid optionally corrupted as a repair curriculum
  (:class:`CellCorruption`, :class:`SlotScramble`).

The pool never touches the model: a step hands it a :data:`LatentInit` for the
latents a seated slot starts from, and hands the fed-back grid to the model.

Draw order from each dedicated generator is a reproducibility contract: the
halt and corruption generators are seeded independently of the global RNG, so
two runs from identical weights stay identical whatever else drew between.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import field
from typing import Protocol, cast, override

import abc
import math

from configgle import Fig, Makeable, Makes
from torch import Tensor

import torch


type LatentInit = Callable[[int], tuple[Tensor, Tensor]]
"""Return the initial ``(z_slow, z_fast)`` for a number of puzzles."""


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


class Corruption(Protocol):
    """Replace cells of a fed-back grid, drawing from ``generator``."""

    def __call__(
        self,
        grid: Tensor,
        *,
        given: Tensor,
        generator: torch.Generator,
    ) -> Tensor:
        """Apply to the input."""
        ...


class CellCorruption:
    """Replace each cell with a uniform token with probability ``rate``.

    Draws ``rand`` then ``randint``, both over the whole grid. Clue cells are
    not protected.
    """

    class Config(Fig["CellCorruption"]):
        """Rate and replacement token range."""

        rate: float = 0.0
        """Per-cell replacement probability."""

        low: int = 2
        """First replacement token (ARC's first color)."""

        high: int = 12
        """One past the last replacement token (ARC's vocabulary size)."""

    def __init__(self, config: Config) -> None:
        if math.isnan(config.rate) or config.rate < 0.0 or config.rate > 1.0:
            raise ValueError(f"rate must be in [0, 1]; got {config.rate}.")
        self.config = config

    def __call__(
        self,
        grid: Tensor,
        *,
        given: Tensor,
        generator: torch.Generator,
    ) -> Tensor:
        """Return ``grid`` with the drawn cells replaced."""
        del given
        config = self.config
        hit = torch.rand(grid.shape, device=grid.device, generator=generator)
        tokens = torch.randint(
            config.low,
            config.high,
            grid.shape,
            device=grid.device,
            generator=generator,
        )
        return torch.where(hit < config.rate, tokens, grid)


class SlotScramble:
    """Scramble random non-clue cells of a random fraction of slots.

    A slot is selected with probability ``prob``; each of its cells is then
    replaced with probability ``cells / grid_len`` by a uniform token, clue
    cells excepted. Draws ``rand`` over slots, ``rand`` over cells, then
    ``randint`` over cells.
    """

    class Config(Fig["SlotScramble"]):
        """Slot rate, expected cells scrambled, and replacement tokens."""

        prob: float = 0.5
        """Chance a slot's grid is scrambled."""

        cells: int = 12
        """Expected cells replaced in a scrambled grid."""

        low: int = 2
        """First replacement token (the first digit)."""

        high: int = 11
        """One past the last replacement token (the sudoku vocabulary size)."""

    def __init__(self, config: Config) -> None:
        self.config = config

    def __call__(
        self,
        grid: Tensor,
        *,
        given: Tensor,
        generator: torch.Generator,
    ) -> Tensor:
        """Return ``grid`` with the drawn non-clue cells replaced."""
        config = self.config
        rows, grid_len = grid.shape
        slot = torch.rand(rows, device=grid.device, generator=generator) < config.prob
        cell = (
            torch.rand(rows, grid_len, device=grid.device, generator=generator)
            < config.cells / grid_len
        )
        tokens = torch.randint(
            config.low,
            config.high,
            (rows, grid_len),
            device=grid.device,
            generator=generator,
        )
        return torch.where(slot[:, None] & cell & ~given, tokens, grid)


class FeedbackCarry:
    """Per-slot decoded grid, fed back as the next forward's feedback input.

    Fresh slots restart from their input grid; after every forward the grid
    becomes the detached argmax, clue cells optionally restored, then
    optionally corrupted. The objective is unchanged: corruption is a repair
    curriculum, not a target.
    """

    class Config(Fig["FeedbackCarry"]):
        """Clue range, corruption, and the corruption generator's seed."""

        givens: tuple[int, int] | None = None
        """Inclusive token range restored verbatim from the puzzle; ``None``
        feeds back the plain argmax."""

        corruption: Makeable[Corruption] | None = None
        """Repair curriculum applied after decoding; ``None`` feeds back the
        clean grid."""

        seed: int = 0
        """Seed for the dedicated corruption generator."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.corruption: Corruption | None = (
            None if config.corruption is None else config.corruption.make()
        )
        self.generator = torch.Generator()
        self.generator.manual_seed(config.seed)

    def to(self, device: torch.device) -> None:
        """Rebuild the generator on ``device`` from the seed."""
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(self.config.seed)

    def given(self, media: Tensor) -> Tensor:
        """Return the ``[B, grid_len]`` mask of the puzzle's clue cells."""
        if self.config.givens is None:
            return torch.zeros_like(media, dtype=torch.bool)
        low, high = self.config.givens
        return (media >= low) & (media <= high)

    def decode(self, logits: Tensor, *, media: Tensor) -> Tensor:
        """Return the argmax grid with the puzzle's clues restored.

        Args:
          logits: ``[B, grid_len, V]`` predictions.
          media: ``[B, grid_len]`` the puzzles those predictions answer.

        Returns:
          grid: ``[B, grid_len]`` token grid for the next forward.

        """
        predictions = logits.argmax(dim=-1)
        if self.config.givens is None:
            return predictions
        return torch.where(
            self.given(media),
            media.to(predictions.dtype),
            predictions,
        )


class Start(Protocol):
    """Return the latents a seated slot begins from."""

    def __call__(self, init: LatentInit, rows: int) -> tuple[Tensor, Tensor]:
        """Apply to the input."""
        ...


class LearnedStart:
    """Begin from the model's learned initial latents, as evaluation does."""

    class Config(Fig["LearnedStart"]):
        """No parameters."""

    def __init__(self, config: Config) -> None:
        del config

    def __call__(self, init: LatentInit, rows: int) -> tuple[Tensor, Tensor]:
        """Return ``init(rows)``."""
        return init(rows)


class ZeroStart:
    """Begin from zero latents.

    Training then never sees the learned initial latents that evaluation
    starts from; kept so runs trained this way replay exactly.
    """

    class Config(Fig["ZeroStart"]):
        """No parameters."""

    def __init__(self, config: Config) -> None:
        del config

    def __call__(self, init: LatentInit, rows: int) -> tuple[Tensor, Tensor]:
        """Return zeros shaped like ``init(rows)``."""
        z_slow, z_fast = init(rows)
        return torch.zeros_like(z_slow), torch.zeros_like(z_fast)


class Solver(Protocol):
    """The model surface an evaluation rollout drives."""

    def init_latents(self, batch_size: int, /) -> tuple[Tensor, Tensor]:
        """Return the initial ``(z_slow, z_fast)``."""
        ...

    def set_feedback(self, grid: Tensor | None) -> None:
        """Hand ``grid`` to the model's feedback channel for the next forward."""
        ...

    def __call__(
        self,
        tokens: Tensor,
        z_slow: Tensor,
        z_fast: Tensor,
        *,
        collect_intermediates: bool,
        **prefix_kwargs: object,
    ) -> RolloutStep:
        """Run one forward."""
        ...


class RolloutStep(Protocol):
    """What a rollout reads from one forward."""

    @property
    def logits(self) -> Tensor:
        """``[B, grid_len, V]`` predictions."""
        ...

    @property
    def halt(self) -> Tensor:
        """``[B]`` halt logits."""
        ...

    @property
    def z_slow(self) -> Tensor:
        """Carried slow latent."""
        ...

    @property
    def z_fast(self) -> Tensor:
        """Carried fast latent."""
        ...


def rollout(
    model: Solver,
    *,
    media: Tensor,
    max_steps: int,
    carry: FeedbackCarry | None,
    prefix_kwargs: Mapping[str, object] | None = None,
) -> tuple[Tensor, Tensor]:
    """Run a puzzle batch to the step cap, carrying latents and feedback.

    The evaluation counterpart of the training pool: no slots, no halting,
    every row simply runs the full depth the model was trained at.

    Args:
      model: The network to run.
      media: ``[B, grid_len]`` puzzles.
      max_steps: Forwards to run.
      carry: Feedback to thread; ``None`` feeds nothing back.
      prefix_kwargs: Batch fields a prefix module consumes, if any.

    Returns:
      logits: Final ``[B, grid_len, V]`` predictions.
      halt: Final ``[B]`` halt logits.

    Raises:
      ValueError: If ``max_steps`` is not positive.

    """
    if max_steps <= 0:
        raise ValueError(f"max_steps must be positive; got {max_steps}.")
    kwargs = dict(prefix_kwargs or {})
    z_slow, z_fast = model.init_latents(media.shape[0])
    feedback = media if carry is not None else None
    out = _forward(model, media, z_slow, z_fast, feedback=feedback, kwargs=kwargs)
    for _ in range(max_steps - 1):
        if carry is not None:
            feedback = carry.decode(out.logits, media=media)
        out = _forward(
            model,
            media,
            out.z_slow,
            out.z_fast,
            feedback=feedback,
            kwargs=kwargs,
        )
    return out.logits, out.halt


class PoolConfig(Makeable["ActPool"], Protocol):
    """A config that builds a pool and takes its geometry from the model.

    A train step pushes the model's shape down into these fields before
    building, so the pool and the model cannot disagree.
    """

    batch_size: int
    max_steps: int
    halting: HaltTraining.Config | None
    feedback: FeedbackCarry.Config | None
    start: Makeable[Start]
    grid_len: int
    seq_len: int
    channels_hidden: int
    dtype: torch.dtype | None


class ActPool(abc.ABC):
    """Slot state shared by every seating policy; see the module docstring."""

    class Config(Fig["ActPool"]):
        """Pool geometry, halting, and the optional feedback carry."""

        batch_size: int = 192
        """Slots in the pool."""

        max_steps: int = 16
        """Forwards a puzzle may take before it is forced out."""

        halting: HaltTraining.Config | None = field(
            default_factory=HaltTraining.Config,
        )
        """Halt-head training; ``None`` freezes the head and halts at the cap."""

        feedback: FeedbackCarry.Config | None = None
        """Decoded-grid feedback; requires a feedback channel on the model."""

        start: Makeable[Start] = field(default_factory=LearnedStart.Config)
        """Latents a newly seated slot begins from."""

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
        self.feedback = torch.zeros_like(self.inputs)
        self.halting = None if config.halting is None else config.halting.make()
        self.carry = None if config.feedback is None else config.feedback.make()
        self.start: Start = config.start.make()

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
        if self.halting is not None:
            self.halting.to(device)
        if self.carry is not None:
            self.carry.to(device)

    @abc.abstractmethod
    def refill(
        self,
        init: LatentInit,
        *,
        media: Tensor,
        labels: Tensor,
        valid_count: int,
        puzzle_ids: Tensor | None,
        ignore_label_id: int,
    ) -> Tensor:
        """Seat incoming puzzles; return the ``[B]`` mask of slots that train.

        Args:
          init: Initial latents for seated slots.
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
    def release(self, init: LatentInit, *, halt: Tensor, active: Tensor) -> None:
        """Act on this step's halt decision.

        Args:
          init: Initial latents for slots seated from a queue.
          halt: ``[B]`` slots that stop after this step.
          active: Slots that trained.

        """

    def halted_this_step(self) -> Tensor | None:
        """Slots that halted this step, when metrics score only those."""
        return None

    def advance_feedback(self, logits: Tensor) -> Tensor | float:
        """Replace the carried grid with the decoded, possibly corrupted, argmax.

        Args:
          logits: ``[B, grid_len, V]`` this step's predictions.

        Returns:
          changed: Fraction of cells the corruption changed; 0.0 when clean.

        """
        carry = self.carry
        if carry is None:
            raise ValueError("advance_feedback needs a pool with a feedback carry.")
        decoded = carry.decode(logits, media=self.inputs).detach()
        changed: Tensor | float = 0.0
        if carry.corruption is not None:
            corrupted = carry.corruption(
                decoded,
                given=carry.given(self.inputs),
                generator=carry.generator,
            )
            changed = (corrupted != decoded).float().mean()
            decoded = corrupted
        self.feedback = decoded
        return changed

    def halt_mask(self, halt: Tensor) -> Tensor:
        """Which slots stop after this step: at the cap, or fired and allowed.

        Args:
          halt: ``[B]`` halt logits.

        Returns:
          halt: ``[B]`` slots that stop.

        """
        at_cap = self.steps >= self.config.max_steps
        if self.halting is None:
            return at_cap
        return at_cap | self.halting.exploration(
            halt > 0,
            steps=self.steps,
            max_steps=self.config.max_steps,
            generator=self.halting.generator,
        )


class AtomicPool(ActPool):
    """Seat one pool-width batch per call, only where a slot just halted.

    Every slot trains every call, including ones that were just seated; a
    batch shorter than the pool is padded by the loader and its tail labels
    are masked to the ignore label.
    """

    class Config(Makes["AtomicPool"], ActPool.Config):
        """Atomic seating."""

    @override
    def refill(
        self,
        init: LatentInit,
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
        z_slow, z_fast = self.start(init, bs)
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
        self.feedback = torch.where(seat, self.inputs, self.feedback)
        self.active = torch.ones_like(halted)
        return self.active

    @override
    def update_carry(self, *, z_slow: Tensor, z_fast: Tensor, active: Tensor) -> None:
        del active
        self.z_slow.copy_(z_slow.to(self.z_slow.dtype))
        self.z_fast.copy_(z_fast.to(self.z_fast.dtype))
        self.steps.add_(1)

    @override
    def release(self, init: LatentInit, *, halt: Tensor, active: Tensor) -> None:
        del init, active
        self.halted = halt

    @override
    def halted_this_step(self) -> Tensor | None:
        return self.halted


class StreamingPool(ActPool):
    """Queue every incoming puzzle and seat it in the next free slot.

    Only a batch's valid rows are queued; a slot trains only while it holds a
    puzzle, and a halted slot is refilled from the queue at once, so a slot
    may sit empty only while the queue is. Carried task ids and loader padding
    are not supported.
    """

    class Config(Makes["StreamingPool"], ActPool.Config):
        """Streaming seating."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        # Seating keys off ``active``, so nothing has halted until a slot trains.
        self.halted = torch.zeros_like(self.halted)
        self.pending_inputs = torch.zeros(0, config.grid_len, dtype=torch.long)
        self.pending_labels = torch.zeros_like(self.pending_inputs)

    @override
    def to(self, device: torch.device) -> None:
        super().to(device)
        self.pending_inputs = self.pending_inputs.to(device)
        self.pending_labels = self.pending_labels.to(device)

    @override
    def refill(
        self,
        init: LatentInit,
        *,
        media: Tensor,
        labels: Tensor,
        valid_count: int,
        puzzle_ids: Tensor | None,
        ignore_label_id: int,
    ) -> Tensor:
        del ignore_label_id
        if puzzle_ids is not None:
            raise ValueError("streaming seating does not carry task ids.")
        self.pending_inputs = torch.cat(
            [self.pending_inputs, media[:valid_count].to(torch.long)],
        )
        self.pending_labels = torch.cat(
            [self.pending_labels, labels[:valid_count].to(torch.long)],
        )
        self._seat(init)
        return self.active

    @override
    def update_carry(self, *, z_slow: Tensor, z_fast: Tensor, active: Tensor) -> None:
        self.z_slow[active] = z_slow[active].to(self.z_slow.dtype)
        self.z_fast[active] = z_fast[active].to(self.z_fast.dtype)
        self.steps[active] += 1

    @override
    def release(self, init: LatentInit, *, halt: Tensor, active: Tensor) -> None:
        self.halted = active & halt
        self.active[self.halted] = False
        self._seat(init)

    # Seats in slot order, oldest queued puzzle first.
    def _seat(self, init: LatentInit) -> None:
        free = (~self.active).nonzero(as_tuple=True)[0]
        count = min(len(free), len(self.pending_inputs))
        if count == 0:
            return
        slots = free[:count]
        self.inputs[slots] = self.pending_inputs[:count]
        self.labels[slots] = self.pending_labels[:count]
        z_slow, z_fast = self.start(init, count)
        self.z_slow[slots] = z_slow.to(self.z_slow.dtype)
        self.z_fast[slots] = z_fast.to(self.z_fast.dtype)
        self.steps[slots] = 0
        self.active[slots] = True
        self.feedback[slots] = self.inputs[slots]
        self.pending_inputs = self.pending_inputs[count:]
        self.pending_labels = self.pending_labels[count:]


def _forward(
    model: Solver,
    media: Tensor,
    z_slow: Tensor,
    z_fast: Tensor,
    *,
    feedback: Tensor | None,
    kwargs: Mapping[str, object],
) -> RolloutStep:
    """Stash ``feedback`` when there is one, then run one forward."""
    if feedback is not None:
        model.set_feedback(feedback)
    return model(media, z_slow, z_fast, collect_intermediates=False, **kwargs)
