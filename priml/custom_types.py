"""Torch-free ML types shared across priml.

Importable without dragging in torch + jaxtyping + numpy. Tensor type
aliases and ``convert_to_tensor`` live in
:mod:`priml.math.custom_types`; channel Protocols live in
:mod:`priml.model.custom_types`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable


if TYPE_CHECKING:
    from pathlib import Path

    from priml.cost import Cost


__all__ = [
    "CheckpointableProtocol",
    "HasCost",
    "HasNormalizedWorkingDirPattern",
    "JobProtocol",
    "LaunchableExperiment",
    "Matrix",
    "MetricObjective",
    "Vector",
]


@runtime_checkable
class HasCost(Protocol):
    """A config that costs the module it builds.

    Every argument is named: the protocol fixes no keyword, and each
    implementation declares the ones it reads (``seq_len``, ``batch_size``,
    ``dtype``, ...) as required keyword-only parameters, so a caller that
    forgets one fails there rather than costing a guessed batch. The rest of
    the bus passes through ``**kwargs`` so a container can forward it unread.
    """

    def cost(self, **kwargs: object) -> Cost:
        """Cost one token through the module ``self`` builds.

        Args:
          **kwargs: The open bus, named arguments only; each implementation
            declares what it reads.

        Returns:
          cost: Per-token cost.

        """
        ...


@runtime_checkable
class JobProtocol(Protocol):
    """Protocol for a runnable job."""

    def run(self, *args: str) -> None:
        """Run."""
        ...


@runtime_checkable
class LaunchableExperiment(Protocol):
    """A config the launcher can stamp with run identity and a docstring.

    The launcher auto-derives ``study_name`` (run-family prefix from the module
    path) and ``experiment_name`` (the factory function name) when either is left
    empty. It attaches the factory's docstring to ``doc`` when unset. A config
    opts in by declaring these fields; a standalone job lacking them is launched
    untouched.
    """

    study_name: str

    experiment_name: str

    doc: str


@runtime_checkable
class HasNormalizedWorkingDirPattern(Protocol):
    """A path-owning Config that inherits its location from its owner.

    The structural contract behind parent-to-child path propagation: a parent
    fills a child's ``base_dir`` (only when ``None``) with its own resolved
    ``working_dir``, and the child resolves ``working_dir`` beneath it (see
    ``priml.lib.userdirs.resolve_working_dir``). ``base_dir`` is inherited
    plumbing; ``working_dir`` is the Config's own opinionated logical location.
    """

    base_dir: Path | str | None

    working_dir: Path | str


# Type aliases for embedding and score data structures.
Vector = Sequence[float]
"""1D float sequence (embeddings, scores)."""

Matrix = Sequence[Vector]
"""2D float sequence (batch of embeddings/scores)."""


@dataclass(frozen=True, kw_only=True, slots=True)
class MetricObjective:
    """The primary metric a scored run optimizes: its key and direction.

    ``metric_key`` is the full tracker key (e.g. ``"eval/total_loss"``) so a
    consumer never guesses the prefix; ``direction`` says whether lower or higher
    is better. Ranking (leaderboards) and "did it beat the bar" checks need these
    as DATA, not MANIFEST prose.
    """

    metric_key: str
    """Full tracker metric key, e.g. ``"eval/total_loss"`` or ``"eval/roc_auc"``."""

    direction: Literal["minimize", "maximize"]
    """Whether a lower (minimize) or higher (maximize) value is better."""

    def is_better(self, candidate: float, incumbent: float) -> bool:
        """Whether ``candidate`` beats ``incumbent`` under this direction.

        Args:
          candidate: Candidate.
          incumbent: Incumbent.

        Returns:
          result: The bool.

        """
        if self.direction == "minimize":
            return candidate < incumbent
        return candidate > incumbent


@runtime_checkable
class CheckpointableProtocol(Protocol):
    """Protocol for a checkpointable object.

    The checkpointer serializes the payload and never reads it, so the
    protocol is deliberately schema-free: ``Mapping[str, object]`` lets an
    implementation return its nested ``StateDict`` TypedDict (a TypedDict is a
    ``Mapping``, never a ``dict``). An implementation narrows the parameter with
    one ``cast(Cls.StateDict, state_dict)`` at the top of ``load_state_dict``.
    Torch's own ``dict[str, Any]`` payloads are converted at the checkpointer,
    the one place that talks to torch, so no ``Any`` reaches this contract.
    """

    def state_dict(self) -> Mapping[str, object]:
        """Get state for checkpointing.

        Returns:
          state: The state mapping; the schema belongs to the implementation.

        """
        ...

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        """Load state from checkpoint.

        Args:
          state_dict: State mapping as returned by :meth:`state_dict`.

        """
        ...
