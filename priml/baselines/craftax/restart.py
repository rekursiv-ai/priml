"""When finished workers' fresh worlds are generated.

Generating a world is the most expensive thing the environment does, and its
cost is paid per CALL far more than per world: one world takes most of the time
sixty-four do. Two policies trade that cost against the random numbers drawn.

``RestartOnDemand`` generates, on the step a worker finishes, exactly the worlds
that step needs. It is the policy every golden was minted with.

``RestartFromReserve``, the default, generates a whole pool at once and deals
from it until it runs out, so a batch of a thousand workers pays for generation
every twenty or so steps rather than every step. Each world is still an independent draw from
the same generator; only WHEN it is drawn, and so which numbers it consumes,
changes -- which is why the two do not reproduce each other bit for bit.
"""

from __future__ import annotations

from typing import NamedTuple, Protocol, TypedDict

from configgle import Fig


class RestartPlan(NamedTuple):
    """What one step's restart does, decided on the host.

    Attributes:
      generate: Worlds to generate into the front of the pool first; 0 for none.
      offset: Pool index the first finished worker takes.
      modulus: Pool indices wrap at this; later workers share worlds past it.

    """

    generate: int

    offset: int

    modulus: int


class RestartStateDict(TypedDict):
    """A restart policy's checkpoint: the next unused pool world, if any."""

    cursor: int | None


class RestartPolicy(Protocol):
    """Decides, each step, which pool worlds the finished workers take."""

    def plan(self, *, finished: int, pool_size: int) -> RestartPlan:
        """Plan one step's restart for ``finished`` workers."""
        ...

    def state_dict(self) -> RestartStateDict:
        """Return what the policy must remember across a checkpoint."""
        ...

    def load_state_dict(self, state_dict: RestartStateDict) -> None:
        """Restore what :meth:`state_dict` saved."""
        ...


class RestartOnDemand:
    """Generate exactly the worlds each step's finished workers need."""

    class Config(Fig["RestartOnDemand"]):
        """Nothing to configure: the pool size is the environment's."""

    def __init__(self, config: Config) -> None:
        del config

    def plan(self, *, finished: int, pool_size: int) -> RestartPlan:
        """Generate one world per finished worker, up to the pool, and deal them.

        Args:
          finished: Workers whose episode ended this step.
          pool_size: Worlds the pool holds.

        Returns:
          plan: The worlds to generate and how to deal them.

        """
        wanted = min(finished, pool_size)
        return RestartPlan(generate=wanted, offset=0, modulus=max(wanted, 1))

    def state_dict(self) -> RestartStateDict:
        """Return an empty cursor: no world outlives the step it was made for."""
        return {"cursor": None}

    def load_state_dict(self, state_dict: RestartStateDict) -> None:
        """Nothing to restore."""
        del state_dict


class RestartFromReserve:
    """Generate a whole pool at once and deal from it until it runs out."""

    class Config(Fig["RestartFromReserve"]):
        """Nothing to configure: the pool size is the environment's."""

    def __init__(self, config: Config) -> None:
        del config
        self._cursor: int | None = None
        """The next unused pool world; ``None`` before the first fill."""

    def plan(self, *, finished: int, pool_size: int) -> RestartPlan:
        """Deal the next unused worlds, refilling the pool when too few remain.

        Worlds a refill leaves unused are discarded rather than dealt alongside
        the new batch: a step rarely finishes more than a few workers, so the
        waste is a few worlds per pool.

        Args:
          finished: Workers whose episode ended this step.
          pool_size: Worlds the pool holds.

        Returns:
          plan: The worlds to generate, if any, and how to deal them.

        """
        if not finished:
            return RestartPlan(generate=0, offset=0, modulus=1)
        generate = 0
        if self._cursor is None or self._cursor + finished > pool_size:
            generate, self._cursor = pool_size, 0
        offset = self._cursor
        self._cursor = min(offset + finished, pool_size)
        return RestartPlan(generate=generate, offset=offset, modulus=pool_size)

    def state_dict(self) -> RestartStateDict:
        """Return the next unused pool world."""
        return {"cursor": self._cursor}

    def load_state_dict(self, state_dict: RestartStateDict) -> None:
        """Resume dealing where the checkpoint stopped."""
        self._cursor = state_dict["cursor"]
