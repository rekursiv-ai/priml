"""The score, as PufferLib reports it.

``perf`` is the achievement return as a fraction of the 226 available,
averaged over every episode that finished during evaluation. PufferLib plays
whole rollouts until at least ``num_episodes`` have finished
(``evaluation.Evaluation.play``), then reports the mean of each field of the
environments' episode logs; this module is that reduction.

The mean is PufferLib's arithmetic, not merely its definition: each
environment's fp32 log is added in environment order, skipping environments
that finished no episode, and every field is then divided by the episode count
(``pufferl.cu:922-937, 954-966, 1363-1383``). fp32 addition is not associative,
so a pairwise sum of the same logs lands on different bits.

References:
    https://github.com/PufferAI/PufferLib
        Suarez. PufferLib (MIT license), ``src/pufferl.cu``, pin ``6ffa5b10``.

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, TypedDict, cast

from configgle import Fig

import numpy as np

from priml.baselines.craftax.evaluation import Played
from priml.baselines.craftax.game.state import (
    LOG_DTYPE,
    Achievement,
    env_log,
)


if TYPE_CHECKING:
    from collections.abc import Mapping

    from numpy.typing import NDArray
    from torch import Tensor


LOG_FIELDS: Final = LOG_DTYPE.itemsize // np.dtype(np.float32).itemsize
"""fp32 values in one episode log: PufferLib's ``LOG_NF``."""

FLOOR_NAMES: Final = (
    "floor_0_overworld",
    "floor_1_dungeon",
    "floor_2_gnomish_mines",
    "floor_3_sewers",
    "floor_4_vault",
    "floor_5_troll_mines",
    "floor_6_fire_realm",
    "floor_7_ice_realm",
    "floor_8_graveyard",
)
"""PufferLib's ``puf_log`` names for the floors field, in order."""


def aggregate_logs(logs: NDArray[np.void]) -> NDArray[np.float32]:
    """Return PufferLib's mean of per-environment episode logs, bit for bit.

    Args:
      logs: ``LOG_DTYPE [num_envs]``.

    Returns:
      mean: fp32 ``[LOG_FIELDS]``: every field summed in environment order over
        the environments with ``n != 0``, then divided by the summed ``n``. The
        last entry holds the summed ``n`` itself, as PufferLib reports it, rather
        than ``n / n``. All zeros when no episode finished.

    """
    fields = np.ascontiguousarray(logs).view(np.float32).reshape(len(logs), LOG_FIELDS)
    # NumPy walks this row-major array in memory order, adding a row at a time
    # into one fp32 sum per field: environment order. A copy that makes each
    # field's values contiguous is summed pairwise and rounds differently.
    total = np.sum(fields[np.not_equal(fields[:, -1], 0)], axis=0, dtype=np.float32)
    count = np.float32(total.item(-1))
    if count > 0:
        mean = total / count
        mean[-1] = count
        return mean
    return total


def log_metrics(mean: NDArray[np.float32]) -> dict[str, float]:
    """Name an :func:`aggregate_logs` result with PufferLib's ``puf_log`` keys.

    Args:
      mean: fp32 ``[LOG_FIELDS]`` from :func:`aggregate_logs`.

    Returns:
      metrics: ``perf``, ``achievement_rate``, ``score``, ``episode_return``,
        ``episode_length``, one key per floor, and ``n``.

    """
    record = env_log(mean.view(LOG_DTYPE), 0)
    metrics = {
        "perf": float(record.perf),
        "achievement_rate": float(record.achievement_rate),
        "score": float(record.score),
        "episode_return": float(record.episode_return),
        "episode_length": float(record.episode_length),
    }
    metrics.update(
        {
            name: float(value)
            for name, value in zip(FLOOR_NAMES, record.floors, strict=True)
        },
    )
    metrics["n"] = float(record.n)
    return metrics


def report_metrics(mean: NDArray[np.float32]) -> dict[str, float]:
    """Name an :func:`aggregate_logs` result as the port reports it: without repeats.

    ``score`` and ``episode_return`` both sum the achievement rewards an episode
    unlocked (``game/step.py``), under every rule set: ``perf`` times 226. Only
    ``perf`` is reported. ``floor_9_finish`` is the fraction of completed
    episodes that defeated the Necromancer, as the recipe runs' progression key counts.

    Args:
      mean: fp32 ``[LOG_FIELDS]`` from :func:`aggregate_logs`.

    Returns:
      metrics: :func:`log_metrics` without ``score`` and ``episode_return``,
        plus ``floor_9_finish`` from the averaged boss-defeat achievement.

    """
    metrics = {
        name: value
        for name, value in log_metrics(mean).items()
        if name not in ("score", "episode_return")
    }
    metrics["floor_9_finish"] = float(
        env_log(mean.view(LOG_DTYPE), 0).achievements[Achievement.DEFEAT_NECROMANCER],
    )
    return metrics


class CraftaxScore:
    """PufferLib's score of an evaluation: the mean of its episode logs."""

    class Config(Fig["CraftaxScore"]):
        """Nothing to configure: the evaluation decides what is played."""

    def __init__(self, config: Config) -> None:
        """Prepare an empty score.

        Args:
          config: Empty.

        """
        del config
        self._metrics: dict[str, float] = {}

    def update(self, logits: Tensor | None = None, **batch: object) -> None:
        """Reduce one evaluation's logs to the score.

        Args:
          logits: Unused; present to satisfy the metric interface.
          **batch: Must carry ``played``, what an evaluation played.

        Raises:
          TypeError: The batch carries no played evaluation.

        """
        del logits
        played = batch.get("played")
        if not isinstance(played, Played):
            raise TypeError("CraftaxScore requires a played evaluation batch entry")
        self._metrics = report_metrics(aggregate_logs(played.logs))
        self._metrics["rollouts"] = float(played.rollouts)
        self._metrics["gameplay_seconds"] = played.gameplay_seconds

    def compute(self) -> dict[str, object]:
        """Return the last evaluation's metrics.

        Returns:
          metrics: PufferLib's ``env/*`` means without repeats
            (:func:`report_metrics`; ``perf`` is the headline), the episode
            count ``n``, the rollouts played, and gameplay seconds.

        """
        return dict(self._metrics)

    def reset(self) -> None:
        """Forget the last evaluation."""
        self._metrics = {}

    class StateDict(TypedDict):
        """The last evaluation's metrics."""

        metrics: dict[str, float]

    def state_dict(self) -> StateDict:
        """Return the last evaluation's metrics.

        Returns:
          state: The metrics dict.

        """
        return {"metrics": dict(self._metrics)}

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        """Restore metrics saved by :meth:`state_dict`.

        Args:
          state_dict: State dict.

        """
        state = cast(CraftaxScore.StateDict, state_dict)
        self._metrics = {name: float(value) for name, value in state["metrics"].items()}
