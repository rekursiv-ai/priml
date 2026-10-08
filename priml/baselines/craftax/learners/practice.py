"""Frontier practice: restart some rows, each rollout, in worlds the policy rarely reaches.

A training option of the environment (``CraftaxEnv.Config.practice``). Its
last ``num_donors`` environments play natural episodes, and whenever one first
crosses into a new level of achievement return -- ``level_width`` points per
level -- it saves its whole environment into an archive (``game.archive``
describes the archive and its rules). Once per rollout, between rollouts,
:meth:`FrontierPractice.prepare` sizes the rollout's practice rows and
restores them, rows ``0..selected-1``, from entries of rarely reached levels.
A restored row plays a practice branch until its episode ends: its
transitions train the policy like any others, but its episode's end reaches no
log, and the natural episode it interrupted is never logged either.

The controller aims at ``fraction`` of all transitions spent in branches. A
branch lasts, on average, the branch steps so far over the restores so far
(the horizon before the first); the rollout restores enough rows to cover the
deficit against the target, counting this rollout's transitions, at most
``min(num_envs - num_donors, ceil(num_envs * fraction))``. The horizon is read
from the steps the environments took since the last prepare.

The rollout keeps each entry's carry (``rollout.Rollout``): ``save_slots`` and
``restore_slots`` tell it where the donors saved and which entry each row
starts from, and :attr:`FrontierPractice.carry_slots` sizes its archive.

The archive is numpy, and its rules Numba kernels in ``game/``, because the
env's ``nogil`` buffer loop inserts the donors' saves (``game.archive``'s
docstring); this class allocates it and runs the controller between rollouts.

"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import math

from configgle import Fig

import numpy as np

from priml.baselines.craftax.game.archive import (
    restart_donors_numba,
    restore_rows_numba,
)
from priml.baselines.craftax.game.state import (
    TRAINING_STATS_DTYPE,
    Archive,
    new_states,
    new_stats,
)
from priml.baselines.craftax.lib.arrays import typed


if TYPE_CHECKING:
    from numpy.typing import NDArray
    from torch import Tensor

    import torch

    from priml.baselines.craftax.game.step import Batch
else:
    from wrapt import lazy_import

    torch = lazy_import("torch")  # ~1050 ms; only a state dict touches it.


class FrontierPractice:
    """The practice archive of an environment, and the controller that restores from it.

    Attributes:
      archive: The saved worlds and the donors' records, shared with the
        step's kernels.
      controller: One :attr:`Config.CONTROLLER` record, the controller's counters.
      carry_slots: Entries in the archive: the carries the rollout keeps.

    """

    class Config(Fig["FrontierPractice"]):
        """The archive's shape, the donors, and the share of practice."""

        CONTROLLER: ClassVar[np.dtype[np.void]] = np.dtype(
            [
                ("restores", np.int64),
                ("steps", np.int64),
                ("branch_steps", np.int64),
                ("horizon", np.int64),
                ("selected", np.int64),
                ("levels", np.int64),
            ],
        )
        """The controller's counters: restores so far; the environment steps and
        the branch steps counted at the last prepare; the last rollout's steps
        per row; the rows the last prepare restored, and the sum of their levels."""

        fraction: float = 0.2
        """Share of all transitions to spend in practice branches."""

        num_donors: int = 64
        """Environments, the last ones, whose natural episodes fill the archive;
        they must lie in the last buffer."""

        level_width: float = 8.0
        """Achievement return per level."""

        num_levels: int = 32
        """Return levels (not floors); a return past the last counts as the last."""

        entries_per_level: int = 32
        """Saved worlds each level holds."""

        entries_per_world: int = 4
        """Saved worlds a level keeps from one world."""

        reach_decay: float = 0.999
        """Each donor episode start multiplies every level's reach by this, so
        reach remembers about ``1 / (1 - reach_decay)`` episodes."""

        seed: int = 1_973
        """The archive stream's first ``rand_r`` state."""

    def __init__(
        self,
        config: Config,
        *,
        num_envs: int,
        envs_per_buffer: int,
        save_slots: np.ndarray,
        restore_slots: np.ndarray,
    ) -> None:
        """Allocate an empty archive for ``num_envs`` environments.

        Args:
          config: The archive's shape, the donors and the share.
          num_envs: Environments in total.
          envs_per_buffer: Environments per buffer.
          save_slots: ``int32 [num_envs]``, the env's; the donors write it.
          restore_slots: ``int32 [num_envs]``, the env's; restores write it.

        Raises:
          ValueError: :meth:`check` refuses the config.

        """
        self.check(config, num_envs=num_envs, envs_per_buffer=envs_per_buffer)
        self.fraction = config.fraction
        self.num_envs = num_envs
        self.num_donors = config.num_donors
        self.carry_slots = config.num_levels * config.entries_per_level
        levels, slots, donors = config.num_levels, self.carry_slots, config.num_donors
        # Numpy, not torch: the step's Numba kernels read and write every array.
        self.archive = Archive(
            states=new_states(slots),
            rngs=np.zeros(slots, dtype=np.uint32),
            stats=new_stats(slots, TRAINING_STATS_DTYPE),
            actions=np.zeros(slots, dtype=np.float32),
            rewards=np.zeros(slots, dtype=np.float32),
            worlds=np.zeros(slots, dtype=np.uint64),
            sizes=np.zeros(levels, dtype=np.int64),
            # float64, numpy's default, as the kernels read them.
            reach=np.zeros(levels),
            weights=np.zeros(levels),
            stream=np.array([config.seed], dtype=np.uint32),
            donor_levels=np.zeros(donors, dtype=np.int64),
            donor_worlds=np.zeros(donors, dtype=np.uint64),
            donor_steps=np.zeros(donors, dtype=np.int64),
            counts=np.zeros(num_envs // envs_per_buffer, dtype=np.int64),
            save_slots=save_slots,
            restore_slots=restore_slots,
            first_donor=num_envs - donors,
            envs_per_buffer=envs_per_buffer,
            per_level=config.entries_per_level,
            per_world=config.entries_per_world,
            level_width=np.float32(config.level_width),
            reach_decay=config.reach_decay,
        )
        self.controller = np.zeros((), dtype=config.CONTROLLER)

    @classmethod
    def check(cls, config: Config, *, num_envs: int, envs_per_buffer: int) -> None:
        """Refuse what the constructor would, before anything is built.

        Args:
          config: The archive's shape, the donors and the share.
          num_envs: Environments in total.
          envs_per_buffer: Environments per buffer.

        Raises:
          ValueError: A count is not positive, the width is not a positive
            finite fp32 (the archive divides returns by it), the fraction or
            the decay is outside ``(0, 1]``, the seed is not 32-bit, there are
            no natural rows besides the donors, or the donors do not fit in the
            last buffer: donors in two buffers would save from two threads at
            once, in whatever order the threads ran.

        """
        counts = (
            config.num_donors,
            config.num_levels,
            config.entries_per_level,
            config.entries_per_world,
        )
        width = np.float32(config.level_width)
        if min(counts) <= 0 or not (np.isfinite(width) and width > 0):
            raise ValueError(
                "practice's counts must be positive and level_width a positive "
                f"finite fp32, not {config.level_width}",
            )
        for name, value in (
            ("fraction", config.fraction),
            ("reach_decay", config.reach_decay),
        ):
            if math.isnan(value) or value <= 0 or value > 1:
                raise ValueError(f"{name} must be in (0, 1], not {value}")
        if config.seed < 0 or config.seed >= 2**32:
            raise ValueError(f"seed must be in [0, 2**32), not {config.seed}")
        if config.num_donors >= num_envs:
            raise ValueError("practice needs natural rows besides its donors")
        if config.num_donors > envs_per_buffer:
            raise ValueError(
                f"practice's {config.num_donors} donors must fit in the last "
                f"buffer of {envs_per_buffer} environments",
            )

    def reset(self, batch: Batch) -> None:
        """Start every donor's episode afresh after the environments' reset.

        Args:
          batch: The environments, just reset.

        """
        self.archive.save_slots[:] = -1
        self.archive.restore_slots[:] = -1
        restart_donors_numba(batch, self.archive)

    def prepare(self, batch: Batch) -> None:
        """Size the next rollout's practice rows and restore them; call it between rollouts.

        The last rollout's saves were taken by its carries already, so the save
        slots are cleared.

        Args:
          batch: The environments.

        """
        steps = int(self.archive.counts.sum())
        branch_steps = int(np.sum(typed(batch.stats["branch_steps"], np.int64)))
        controller = self.controller
        stepped = steps - int(controller["steps"])
        if stepped:
            controller["horizon"] = stepped // self.num_envs
        controller["steps"] = steps
        controller["branch_steps"] = branch_steps
        selected = self._selected(steps, branch_steps)
        controller["restores"] += selected
        controller["selected"] = selected
        self.archive.save_slots[:] = -1
        controller["levels"] = restore_rows_numba(batch, self.archive, selected)

    def metrics(self) -> dict[str, float]:
        """Return practice's metrics as the last prepare left them.

        Returns:
          metrics: ``practice/fraction``, branch steps over all environment
            steps until then; ``practice/populated_levels`` and
            ``practice/archive_entries``; ``practice/sample_level_mean``, the
            mean level of that prepare's restores; ``practice/selected``, how
            many rows it restored.

        """
        controller = self.controller
        steps = int(controller["steps"])
        branch_steps = int(controller["branch_steps"])
        sizes = self.archive.sizes
        selected = int(controller["selected"])
        return {
            "practice/fraction": branch_steps / steps if steps else 0.0,
            "practice/populated_levels": float(np.count_nonzero(sizes)),
            "practice/archive_entries": float(sizes.sum()),
            "practice/sample_level_mean": (
                int(controller["levels"]) / selected if selected else 0.0
            ),
            "practice/selected": float(selected),
        }

    def state_dict(self) -> dict[str, Tensor]:
        """Return every array the archive and the controller hold, as bytes.

        The environment's state dict carries these, and its ``load_state_dict``
        copies them back into the live memory.

        Returns:
          state: One entry per array, the live memory, not copies; the save
            and restore slots are the environment's.

        """
        arrays: dict[str, NDArray[np.generic]] = {
            name: value
            for name, value in zip(Archive._fields, self.archive, strict=True)
            if isinstance(value, np.ndarray)
            and name not in {"save_slots", "restore_slots"}
        }
        # A view of the record's own memory: a 0-d array's bytes view needs a shape.
        arrays["controller"] = self.controller.reshape(1)
        return {
            name: torch.from_numpy(array.view(np.uint8))
            for name, array in arrays.items()
        }

    def _selected(self, steps: int, branch_steps: int) -> int:
        """Return how many rows the next rollout restores."""
        horizon = int(self.controller["horizon"])
        if not self.archive.sizes.any() or not horizon:
            return 0
        restores = int(self.controller["restores"])
        duration = branch_steps / restores if restores else horizon
        duration = min(max(duration, 1.0), horizon)
        deficit = (steps + self.num_envs * horizon) * self.fraction - branch_steps
        maximum = min(
            self.num_envs - self.num_donors,
            math.ceil(self.num_envs * self.fraction),
        )
        return min(max(math.ceil(deficit / duration), 0), maximum)
