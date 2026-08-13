"""Adaptive computation time: a pool of puzzles, each taking one step at a time.

A recurrent solver should spend more steps on a hard puzzle than an easy one.
That is awkward at fixed batch shape -- the batch cannot shrink as puzzles
finish -- so instead the batch IS a pool of slots. Each training call advances
every occupied slot by one reasoning step, carrying its latent state forward;
when the model's halt head says a slot is done, that slot is released and the
next incoming puzzle takes it. An easy puzzle leaves after a few steps, a hard
one keeps its slot, and the tensor shape never changes.

Everything here is meaningless without a recurrence, which is exactly why it is
a separate injected piece: a plain feedforward experiment's config carries none
of these fields.

Three mechanisms make the scheme work:

* **Halt supervision.** The halt head is trained to predict whether the current
  grid is already correct, so halting is learned rather than a fixed depth.
* **Exploration.** A halt head trained only on its own decisions never sees
  what a deeper rollout would have produced. A fraction of slots are forced to
  keep going regardless, which supplies that counterfactual.
* **Prediction feedback.** The decoded grid can be fed back as input for the
  next step, with the puzzle's given cells clamped back to their true values,
  so the model refines its own answer instead of re-reading a blank grid.

Draw order from the dedicated RNG is a reproducibility contract: the halt
generator is seeded independently of the ambient global RNG, so two runs from
identical weights stay identical regardless of what else drew in between.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self, override

from configgle import Fig
from torch import Tensor, nn

import torch


if TYPE_CHECKING:
    from priml.baselines.sudoku.model import SudokuNet


class ActPool:
    """A fixed set of slots, each holding one puzzle mid-solve.

    See the module docstring for why the pool exists. The pool owns the carried
    latent state, the per-slot step counter, the halt mask, and the fed-back
    grid; the model owns the parameters.
    """

    class Config(Fig["ActPool"]):
        """Pool geometry, halting policy, and feedback."""

        batch_size: int = 384
        """Slots in the pool; also the incoming batch size per step."""

        max_steps: int = 32
        """Reasoning steps a puzzle may take before being forced out."""

        halt_weight: float = 0.05
        """Weight on the halt loss relative to the token loss."""

        halt_exploration_prob: float = 0.1
        """Chance a slot is forced past its halt decision, to supply the
        counterfactual a self-supervised halt head never sees."""

        halt_exploration_seed: int = 0
        """Seed for the dedicated halting RNG.

        Separate from the global stream so the training trajectory does not
        depend on ambient RNG state; two runs from identical weights stay
        bit-for-bit identical."""

        min_steps_sampled: bool = True
        """Sample a per-slot minimum step count rather than forcing a single
        extra step. Spreads exploration over depths instead of always
        producing one-step-deeper rollouts."""

        feedback: bool = True
        """Feed the decoded grid back as the next step's input."""

        given_low: int = 2
        """First token value treated as a puzzle-given clue."""

        given_high: int = 10
        """Last token value treated as a puzzle-given clue.

        Cells holding a value in ``[given_low, given_high]`` are the puzzle's
        clues, restored verbatim in the fed-back grid: the model must not be
        allowed to overwrite what it was told."""

        grid_len: int = -1
        """Grid tokens per puzzle; inherited from the model."""

        seq_len: int = -1
        """Latent sequence length (prefix + grid); inherited from the model."""

        hidden_size: int = -1
        """Latent width; inherited from the model."""

        @override
        def finalize(self) -> Self:
            if self.given_low > self.given_high:
                raise ValueError(
                    f"given_low {self.given_low} exceeds given_high {self.given_high}.",
                )
            return super().finalize()

    def __init__(self, config: Config) -> None:
        if min(config.grid_len, config.seq_len, config.hidden_size) <= 0:
            raise ValueError(
                "grid_len, seq_len, and hidden_size must be positive; they are "
                "normally inherited from the model during finalize. Got "
                f"{config.grid_len}, {config.seq_len}, {config.hidden_size}.",
            )
        self.config = config
        bs = config.batch_size
        self.device = torch.device("cpu")
        self._generator = torch.Generator()
        self._generator.manual_seed(config.halt_exploration_seed)
        self.inputs = torch.zeros(bs, config.grid_len, dtype=torch.long)
        self.labels = torch.zeros_like(self.inputs)
        self.feedback = torch.zeros_like(self.inputs)
        self.z_slow = torch.zeros(bs, config.seq_len, config.hidden_size)
        self.z_fast = torch.zeros_like(self.z_slow)
        self.steps = torch.zeros(bs, dtype=torch.long)
        # Every slot starts halted so the first call fills the whole pool.
        self.halted = torch.ones(bs, dtype=torch.bool)

    def to(self, device: torch.device) -> None:
        """Move pool state to ``device`` and reseed the RNG there."""
        self.device = device
        self.inputs = self.inputs.to(device)
        self.labels = self.labels.to(device)
        self.feedback = self.feedback.to(device)
        self.z_slow = self.z_slow.to(device)
        self.z_fast = self.z_fast.to(device)
        self.steps = self.steps.to(device)
        self.halted = self.halted.to(device)
        self._generator = torch.Generator(device=device)
        self._generator.manual_seed(self.config.halt_exploration_seed)

    def latents(self) -> tuple[Tensor, Tensor]:
        """The carried ``(z_slow, z_fast)`` for this step's forward."""
        return self.z_slow, self.z_fast

    def refill(
        self,
        media: Tensor,
        *,
        labels: Tensor,
        valid_count: int,
        ignore_label_id: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Seat incoming puzzles in halted slots; occupied slots keep solving.

        Args:
          media: ``[B, grid_len]`` incoming puzzles.
          labels: ``[B, grid_len]`` their solutions.
          valid_count: Real rows; the tail is padding whose labels are masked
            out so a short final batch does not train on filler.
          ignore_label_id: Label value the loss skips.

        Returns:
          media: The pool's current puzzles, after seating.
          labels: Their solutions.
          active: ``[B]`` mask of slots participating this step (all of them:
            a slot that just took fresh data still runs this step's forward).

        Raises:
          ValueError: If the incoming batch is not exactly pool-width.

        """
        bs = self.config.batch_size
        if media.shape[0] != bs:
            raise ValueError(
                f"the pool holds {bs} slots; got a batch of {media.shape[0]}.",
            )
        incoming = media.to(torch.long)
        incoming_labels = labels.clone().to(torch.long)
        if valid_count < bs:
            incoming_labels[valid_count:] = ignore_label_id
        halted = self.halted
        seat = halted.unsqueeze(-1)
        self.inputs = torch.where(seat, incoming, self.inputs)
        self.labels = torch.where(seat, incoming_labels, self.labels)
        self.steps = torch.where(halted, torch.zeros_like(self.steps), self.steps)
        # A fresh slot restarts its latent state and its feedback grid; both
        # belong to the puzzle that just left, not the one arriving.
        seat_latent = halted.view(-1, 1, 1)
        self.z_slow = torch.where(
            seat_latent, torch.zeros_like(self.z_slow), self.z_slow
        )
        self.z_fast = torch.where(
            seat_latent, torch.zeros_like(self.z_fast), self.z_fast
        )
        self.feedback = torch.where(seat, self.inputs, self.feedback)
        return self.inputs, self.labels, torch.ones_like(halted)

    def advance(
        self,
        z_slow: Tensor,
        *,
        z_fast: Tensor,
        logits: Tensor,
        halt: Tensor,
        media: Tensor,
    ) -> None:
        """Store this step's state and decide which slots are done.

        Args:
          z_slow: Updated slow latent from the forward.
          z_fast: Updated fast latent.
          logits: ``[B, grid_len, V]`` this step's predictions.
          halt: ``[B]`` halt logits.
          media: The puzzles just solved against, for clamping givens.

        """
        self.z_slow = z_slow.to(self.z_slow.dtype)
        self.z_fast = z_fast.to(self.z_fast.dtype)
        self.steps = self.steps + 1
        if self.config.feedback:
            decoded = logits.argmax(dim=-1).detach()
            self.feedback = self.clamp_givens(decoded, media=media)
        self.halted = self._halt_mask(halt)

    def rollout(
        self,
        model: SudokuNet,
        *,
        media: Tensor,
        prefix_kwargs: dict[str, Any] | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Run one puzzle batch to the step cap, carrying latents throughout.

        The evaluation counterpart of the training pool: no slots, no halting,
        every row simply runs the full depth the model was trained at.

        Args:
          model: The network to run.
          media: ``[B, grid_len]`` puzzles.
          prefix_kwargs: Batch fields a prefix module consumes, if any.

        Returns:
          logits: Final ``[B, grid_len, V]`` predictions.
          halt: Final ``[B]`` halt logits.

        """
        z_slow, z_fast = model.init_latents(media.shape[0])
        feedback = media if self.config.feedback else None
        logits = halt = None
        for _ in range(self.config.max_steps):
            self._set_feedback(model, feedback=feedback)
            out = model(media, z_slow, z_fast, **(prefix_kwargs or {}))
            z_slow, z_fast = out.z_slow, out.z_fast
            logits, halt = out.logits, out.halt
            if feedback is not None:
                feedback = self.clamp_givens(out.logits.argmax(dim=-1), media=media)
        assert logits is not None
        assert halt is not None
        return logits, halt

    def halt_loss(
        self,
        logits: Tensor,
        *,
        labels: Tensor,
        halt: Tensor,
        active: Tensor,
        ignore_label_id: int,
    ) -> tuple[Tensor, dict[str, float | Tensor]]:
        """Train the halt head to predict "this grid is already correct".

        Args:
          logits: ``[B, grid_len, V]`` predictions.
          labels: ``[B, grid_len]`` solutions.
          halt: ``[B]`` halt logits.
          active: ``[B]`` participating slots.
          ignore_label_id: Label value excluded from correctness.

        Returns:
          loss: The weighted halt term.
          metrics: Halt loss and the fraction of slots halting this step.

        """
        with torch.no_grad():
            predictions = logits.argmax(dim=-1)
            counted = labels != ignore_label_id
            correct_cells = (predictions == labels) & counted
            per_row = counted.sum(dim=-1)
            solved = ((correct_cells.sum(dim=-1) == per_row) & (per_row > 0)).to(
                halt.dtype
            )
        per_sample = nn.functional.binary_cross_entropy_with_logits(
            halt,
            solved,
            reduction="none",
        )
        n_active = active.sum().clamp(min=1)
        loss = (
            torch.where(active, per_sample, torch.zeros_like(per_sample)).sum()
            / n_active
        )
        weighted = self.config.halt_weight * loss
        return weighted, {
            "halt_loss": loss.detach(),
            "halted_frac": self.halted.float().mean(),
            "act_steps": self.steps.float().mean(),
        }

    def clamp_givens(self, decoded: Tensor, *, media: Tensor) -> Tensor:
        """Restore the puzzle's clue cells in a decoded grid.

        The model may revise its own guesses freely but must not overwrite what
        the puzzle told it, so clue cells are copied back verbatim.
        """
        given = (media >= self.config.given_low) & (media <= self.config.given_high)
        return torch.where(given, media.to(decoded.dtype), decoded)

    def state_dict(self) -> dict[str, Any]:
        """Snapshot the halting RNG.

        The pool's puzzles and latents are deliberately NOT saved: they are
        in-flight state bound to specific puzzles, and a resumed run continues
        with the next batch rather than replaying interrupted ones. The RNG is
        saved because the exploration sequence must not restart.
        """
        return {"halt_rng": self._generator.get_state()}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore the halting RNG produced by :meth:`state_dict`."""
        if "halt_rng" in state_dict:
            # ``set_state`` wants a CPU byte tensor; a checkpoint read onto the
            # compute device would otherwise be rejected here.
            self._generator.set_state(state_dict["halt_rng"].cpu())

    def _set_feedback(self, model: SudokuNet, *, feedback: Tensor | None) -> None:
        """Hand the fed-back grid to whichever channel consumes it."""
        if feedback is None:
            return
        for channel in model.embedding.channels:
            setter = getattr(channel, "set_feedback", None)
            if setter is not None:
                setter(feedback)

    def _halt_mask(self, halt: Tensor) -> Tensor:
        """Which slots stop after this step.

        A slot halts at the step cap unconditionally, or when the halt head
        fires AND the slot has run its sampled minimum. The draw order --
        ``rand`` then ``randint`` -- is a reproducibility contract.
        """
        config = self.config
        bs = config.batch_size
        at_cap = self.steps >= config.max_steps
        fired = halt > 0
        explore = (
            torch.rand(bs, device=self.device, generator=self._generator)
            < config.halt_exploration_prob
        )
        if config.min_steps_sampled:
            # randint's upper bound is exclusive, so max_steps + 1 samples the
            # inclusive range and never exceeds the cap.
            sampled = torch.randint(
                2,
                config.max_steps + 1,
                (bs,),
                device=self.device,
                generator=self._generator,
            )
            minimum = torch.where(explore, sampled, torch.ones_like(sampled))
            return at_cap | (fired & (self.steps >= minimum))
        return at_cap | (fired & ~explore)
