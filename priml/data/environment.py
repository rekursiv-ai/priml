"""The batched-environment surface consumed by on-policy learners.

An environment differs from a dataset in one way that shapes everything else:
its next observation depends on the action the policy just chose, so the data
cannot be prepared in advance. The protocol here is therefore about stepping,
not iterating, and a rollout is assembled by whoever owns the policy.

Every method is batched over a leading environment axis. A learner that wants
1,024 parallel games gets one call per step, not 1,024, so the batch dimension
is where the parallelism lives -- there is no per-environment Python loop and
no vectorizing transform.

Episodes end at different times across the batch, so the surface auto-resets: a
worker whose episode terminated is returned to a fresh episode on the same step
that reports ``done``. The observation returned alongside a set ``done`` flag is
therefore the RESET observation, and the terminal observation is not visible --
which is what lets a rollout keep a rectangular shape.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable


if TYPE_CHECKING:
    from torch import Tensor


__all__ = [
    "BatchedEnvironmentProtocol",
    "EnvironmentStep",
]


class EnvironmentStep(Protocol):
    """One transition across every parallel worker.

    Attributes:
      observation: Next observations, ``[envs, ...]``. Post-reset for any
        worker whose ``done`` is set.
      reward: Reward earned by the transition, ``[envs]``.
      done: Whether the transition ended an episode, ``[envs]``, boolean.
      info: Per-transition extras keyed by name, each ``[envs, ...]``. Used
        for diagnostics an agent must not train on.

    """

    observation: Tensor
    reward: Tensor
    done: Tensor
    info: dict[str, Tensor]


@runtime_checkable
class BatchedEnvironmentProtocol(Protocol):
    """A batched, auto-resetting environment.

    Attributes:
      num_actions: Size of the discrete action space.
      observation_size: Width of one flattened observation.
      reward_ceiling: Maximum attainable episodic return, used to normalize a
        score into a comparable percentage.

    """

    num_actions: int
    observation_size: int
    reward_ceiling: float

    def reset(self, num_envs: int) -> Tensor:
        """Start fresh episodes in every worker.

        Args:
          num_envs: Number of parallel workers to run from here on.

        Returns:
          observation: Initial observations, ``[num_envs, observation_size]``.

        """
        ...

    def step(self, actions: Tensor) -> EnvironmentStep:
        """Advance every worker by one action.

        Args:
          actions: Integer actions, ``[envs]``, in ``[0, num_actions)``.

        Returns:
          step: The resulting transition across every worker.

        """
        ...

    def state_dict(self) -> dict[str, Any]:
        """Return the simulator state required to resume mid-rollout."""
        ...

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore simulator state produced by :meth:`state_dict`."""
        ...
