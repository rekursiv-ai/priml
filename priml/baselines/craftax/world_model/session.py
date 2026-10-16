"""Play sessions: one engine row driven by a person, a policy, or the model.

A session records its full token stream, so a played game saves and reloads
to identical tokens. Every frame slot and decision value carries a ``Mark``:
the model's own samples, archived data, choices forced by a person or a
policy, and play overrides are never confused with each other.

The stream interleaves frames and decisions: frame ``n`` is observed by
decision ``n``, and frame ``n + 1`` is that decision's next frame, or, when
decision ``n`` ends its episode, the new episode's first frame, generated
after a ``start`` position.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import cast

import dataclasses
import enum

from torch import Tensor

import torch

from priml.baselines.craftax.world_model.batch import Segment
from priml.baselines.craftax.world_model.codec import decode
from priml.baselines.craftax.world_model.engine import (
    Control,
    Engine,
    Outcome,
    Policy,
    Prefix,
)


class Mark(enum.IntEnum):
    """Where a token came from."""

    MODEL = 0
    DATA = 1
    FORCED = 2
    OVERRIDE = 3


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Stream:
    """A session's token stream of ``N`` decisions and ``N + 1`` frames.

    Attributes:
      starts_episode: Whether frame 0 follows a ``start`` position.
      cells: Cell values per frame, uint8 ``[N + 1, cells, fields]``.
      aux: Auxiliary values per frame, int16 ``[N + 1, scalars]``.
      frame_marks: ``Mark`` of each frame slot, uint8 ``[N + 1, frame slots]``.
      frame_logp: Model log-probability of each frame slot, 0 unless the model
        generated it, float32 ``[N + 1, frame slots]``.
      action: Executed actions, uint8 ``[N]``.
      reward: Rewards, int16 ``[N]``.
      done: Terminal flags, bool ``[N]``.
      decision_marks: ``Mark`` of each action, reward, and done, uint8 ``[N, 3]``.
      decision_logp: Their model log-probabilities, float32 ``[N, 3]``.

    """

    starts_episode: bool
    cells: Tensor
    aux: Tensor
    frame_marks: Tensor
    frame_logp: Tensor
    action: Tensor
    reward: Tensor
    done: Tensor
    decision_marks: Tensor
    decision_logp: Tensor

    def save(self, path: Path) -> None:
        """Write the stream to ``path`` as a tensor dictionary."""
        tensors = {
            field.name: getattr(self, field.name) for field in dataclasses.fields(self)
        }
        tensors["starts_episode"] = torch.tensor(self.starts_episode)
        torch.save(tensors, path)

    @classmethod
    def load(cls, path: Path) -> "Stream":
        """Read a stream written by ``save``.

        Args:
          path: File that ``save`` wrote.

        Returns:
          stream: The saved stream.

        """
        # torch.load is annotated `-> Any`; weights_only=True admits only tensors
        # and containers, and ``save`` wrote a flat dictionary of tensors.
        tensors = cast(dict[str, Tensor], torch.load(path, weights_only=True))
        return cls(
            starts_episode=bool(tensors["starts_episode"]),
            cells=tensors["cells"],
            aux=tensors["aux"],
            frame_marks=tensors["frame_marks"],
            frame_logp=tensors["frame_logp"],
            action=tensors["action"],
            reward=tensors["reward"],
            done=tensors["done"],
            decision_marks=tensors["decision_marks"],
            decision_logp=tensors["decision_logp"],
        )


class Session:
    """One engine row, played decision by decision, with its recorded stream."""

    def __init__(
        self,
        engine: Engine,
        *,
        row: int,
        stream: Stream | None = None,
    ) -> None:
        """Bind a session to ``row``.

        Args:
          engine: The engine that owns the row.
          row: The engine row this session plays.
          stream: The record so far of a row already in that state, as
            ``branch`` passes; None for a session that ``prefill`` starts.

        """
        self.engine = engine
        self.row = row
        self._starts = True
        self._frames: list[_Frame] = []
        self._decisions: list[_Decision] = []
        if stream is not None:
            self._restore(stream)

    def prefill(self, source: Segment | Stream | None = None) -> None:
        """Start the row over from ``source``.

        Args:
          source: None to generate a new episode from ``start``; an archived
            ``Segment`` whose frames include the one after its last decision
            and that ends no episode; or a saved ``Stream``, whose last episode
            is prefilled.

        """
        if source is None:
            self._starts, self._frames, self._decisions = True, [], []
            self.engine.reset(self._rows_mask())
            begun = self.engine.start(self._control())
            self._record_frame(begun.job.logp[self.row, 2:], mark=Mark.MODEL)
            return
        stream = _data_stream(source) if isinstance(source, Segment) else source
        self._restore(stream)
        ends = stream.done.nonzero()[:, 0]
        first = int(ends[-1]) + 1 if len(ends) else 0
        self.engine.prefill(
            torch.tensor([self.row]),
            Prefix(
                cells=stream.cells[None, first:],
                aux=stream.aux[None, first:],
                actions=stream.action[None, first:],
                starts_episode=stream.starts_episode or first > 0,
            ),
        )

    def act(self, action: int) -> Outcome:
        """Force ``action``; return the reward, done, and the frame now observed."""
        return self._decide(action, mark=Mark.FORCED)

    def autoplay(self, decisions: int, *, policy: Policy | None = None) -> None:
        """Let the model's action head, or a policy, play several decisions.

        Args:
          decisions: Decisions to play.
          policy: Reads the decoded observation; None samples the action head.
            Policy actions are marked ``FORCED``, sampled ones ``MODEL``.

        """
        state = self.engine.state
        for _ in range(decisions):
            if policy is None:
                self._decide(None, mark=Mark.MODEL)
                continue
            observation = decode(state.cells[self.row, None], state.aux[self.row, None])
            self._decide(int(policy(observation)[0]), mark=Mark.FORCED)

    def override(self, slot: int, value: int | Sequence[int]) -> None:
        """Replace one slot of the current frame before the model encodes it.

        Args:
          slot: Frame slot: a cell index, or ``cells + i`` for auxiliary field ``i``.
          value: A cell's field values, or an auxiliary value.

        """
        state = self.engine.state
        cells, aux = state.cells[self.row].clone(), state.aux[self.row].clone()
        board = len(cells)
        if slot < board:
            cells[slot] = torch.as_tensor(value)
        else:
            aux[slot - board] = torch.as_tensor(value)
        self.engine.set_frame(self.row, cells=cells, aux=aux)
        last = self._frames[-1]
        marks, logp = last.marks.clone(), last.logp.clone()
        marks[slot] = Mark.OVERRIDE
        logp[slot] = 0
        self._frames[-1] = _Frame(
            cells=cells.to(torch.uint8).cpu(),
            aux=aux.to(torch.int16).cpu(),
            marks=marks,
            logp=logp,
        )

    def branch(self, row: int) -> "Session":
        """Copy this session's row onto ``row`` to explore from the same point."""
        self.engine.copy_row(self.row, row)
        return Session(self.engine, row=row, stream=self.stream())

    def stream(self) -> Stream:
        """Return the token stream recorded so far.

        Returns:
          stream: Frames and decisions with their marks and log-probabilities.

        """
        frames, decisions = self._frames, self._decisions
        values = _stack([d.values for d in decisions], empty=(0, 3), dtype=torch.long)
        return Stream(
            starts_episode=self._starts,
            cells=torch.stack([f.cells for f in frames]),
            aux=torch.stack([f.aux for f in frames]),
            frame_marks=torch.stack([f.marks for f in frames]),
            frame_logp=torch.stack([f.logp for f in frames]),
            action=values[:, 0].to(torch.uint8),
            reward=values[:, 1].to(torch.int16),
            done=values[:, 2].bool(),
            decision_marks=_stack(
                [d.marks for d in decisions],
                empty=(0, 3),
                dtype=torch.uint8,
            ),
            decision_logp=_stack(
                [d.logp for d in decisions],
                empty=(0, 3),
                dtype=torch.float32,
            ),
        )

    def _decide(self, action: int | None, *, mark: Mark) -> Outcome:
        """Run one decision of this row; record it and the frame it leads to."""
        engine, row = self.engine, self.row
        engine.ensure_room()
        control = self._control()
        if action is not None:
            control.action_forced[row] = True
            control.action[row] = action
        decision = engine.decide(control)
        outcome = engine.outcome(decision.job.tokens[row])
        logp = decision.job.logp[row]
        self._decisions.append(
            _Decision(
                values=torch.stack(
                    [decision.action[row], outcome.reward, outcome.done.long()],
                ).cpu(),
                marks=torch.tensor([mark, Mark.MODEL, Mark.MODEL], dtype=torch.uint8),
                logp=torch.stack([decision.action_logp[row], logp[0], logp[1]]).cpu(),
            ),
        )
        frame_logp = logp[2:]
        if bool(outcome.done):
            frame_logp = engine.start(control).job.logp[row, 2:]
        else:
            engine.skip_start(control)
        self._record_frame(frame_logp, mark=Mark.MODEL)
        state = engine.state
        return Outcome(
            reward=outcome.reward,
            done=outcome.done,
            cells=state.cells[row].clone(),
            aux=state.aux[row].clone(),
        )

    def _record_frame(self, logp: Tensor, *, mark: Mark) -> None:
        """Append the row's current frame with one mark for every slot."""
        state = self.engine.state
        self._frames.append(
            _Frame(
                cells=state.cells[self.row].to(torch.uint8).cpu(),
                aux=state.aux[self.row].to(torch.int16).cpu(),
                marks=torch.full(logp.shape, int(mark), dtype=torch.uint8),
                logp=logp.float().cpu(),
            ),
        )

    def _restore(self, stream: Stream) -> None:
        """Replace the record with ``stream``'s frames and decisions."""
        self._starts = stream.starts_episode
        self._frames = [
            _Frame(cells=cells, aux=aux, marks=marks, logp=logp)
            for cells, aux, marks, logp in zip(
                stream.cells,
                stream.aux,
                stream.frame_marks,
                stream.frame_logp,
                strict=True,
            )
        ]
        values = torch.stack(
            [stream.action.long(), stream.reward.long(), stream.done.long()],
            -1,
        )
        self._decisions = [
            _Decision(values=v, marks=m, logp=p)
            for v, m, p in zip(
                values,
                stream.decision_marks,
                stream.decision_logp,
                strict=True,
            )
        ]

    def _rows_mask(self) -> Tensor:
        """Return a bool ``[B]`` mask of this session's row."""
        mask = torch.zeros(self.engine.rows, dtype=torch.bool)
        mask[self.row] = True
        return mask

    def _control(self) -> Control:
        """Return a free control in which only this row is active."""
        control = self.engine.control()
        control.active.copy_(self._rows_mask())
        return control


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _Frame:
    cells: Tensor
    aux: Tensor
    marks: Tensor
    logp: Tensor


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _Decision:
    values: Tensor
    marks: Tensor
    logp: Tensor


def _data_stream(segment: Segment) -> Stream:
    """Return an archived segment as a stream of ``DATA`` tokens."""
    frames = len(segment.cells)
    slots = segment.cells.shape[1] + segment.aux.shape[1]
    decisions = len(segment.actions)
    return Stream(
        starts_episode=segment.starts_episode,
        cells=segment.cells.to(torch.uint8),
        aux=segment.aux.to(torch.int16),
        frame_marks=torch.full((frames, slots), int(Mark.DATA), dtype=torch.uint8),
        frame_logp=torch.zeros(frames, slots),
        action=segment.actions.to(torch.uint8),
        reward=segment.reward.to(torch.int16),
        done=segment.done.bool(),
        decision_marks=torch.full((decisions, 3), int(Mark.DATA), dtype=torch.uint8),
        decision_logp=torch.zeros(decisions, 3),
    )


def _stack(
    items: list[Tensor],
    *,
    empty: tuple[int, ...],
    dtype: torch.dtype,
) -> Tensor:
    """Stack ``items``, or return an empty tensor when there are none."""
    return torch.stack(items).to(dtype) if items else torch.zeros(empty, dtype=dtype)
