"""CIFAR-100 batches carrying each image's superclass beside its class.

The vision task's :class:`~priml.baselines.mlsbench.pytorch_vision.data.Cifar100Data`,
held whole in a child slot, its every batch extended with ``label_coarse``: the superclass of
``label`` under the dataset's fixed fine-to-coarse map. The lookup draws
nothing, so shuffles and augmentations are the vision task's, already proven
against the reference loader.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sized
from dataclasses import field
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, Self, override

from configgle import Fig
from torch import Tensor

import torch

from priml.baselines.mlsbench.pytorch_vision.data import Cifar100Data


if TYPE_CHECKING:
    from collections.abc import Mapping

    from priml.timer import CheckpointableStepTimer

COARSE_OF_FINE: Final = (
    *(4, 1, 14, 8, 0, 6, 7, 7, 18, 3),
    *(3, 14, 9, 18, 7, 11, 3, 9, 7, 11),
    *(6, 11, 5, 10, 7, 6, 13, 15, 3, 15),
    *(0, 11, 1, 10, 12, 14, 16, 9, 11, 5),
    *(5, 19, 8, 8, 15, 13, 14, 17, 18, 10),
    *(16, 4, 17, 4, 2, 0, 17, 4, 18, 17),
    *(10, 3, 2, 12, 12, 16, 12, 1, 9, 19),
    *(2, 10, 0, 1, 16, 12, 9, 13, 15, 13),
    *(16, 19, 2, 4, 6, 19, 5, 5, 8, 19),
    *(18, 1, 2, 15, 6, 0, 17, 8, 14, 13),
)
"""The superclass of each of CIFAR-100's 100 classes (Krizhevsky 2009)."""

NUM_COARSE: Final = 20
"""Superclasses of CIFAR-100."""


class MultiTaskCifar100Data:
    """CIFAR-100 yielding ``media``, ``label`` and ``label_coarse`` batches."""

    class Config(Fig["MultiTaskCifar100Data"]):
        """The vision task's data; the superclass map is a fact, not a field."""

        base_dir: Path | str | None = None
        """Resource root supplied during parent finalization, pushed to ``data``."""

        # The loop pushes base_dir only into a config with working_dir too
        # (HasNormalizedWorkingDirPattern); without it ``data`` stays unresolved.
        working_dir: Path | str = "/datasets"
        """Unread; ``data`` resolves its own path beneath ``base_dir``."""

        data: Cifar100Data.Config = field(default_factory=Cifar100Data.Config)
        """Where the prepared tensors live and how batches are made."""

        @override
        def finalize(self) -> Self:
            if self.data.base_dir is None:
                self.data.base_dir = self.base_dir
            return super().finalize()

    def __init__(self, config: Config) -> None:
        self.data = config.data.make()
        self.timer_epoch: CheckpointableStepTimer = self.data.timer_epoch
        """The vision loader's pass timer; it alone sees a pass end."""

    def train_dataloader(self) -> WithCoarse:
        """Return the re-iterable training loader, each batch with its superclasses.

        Returns:
          loader: Each iteration is one shuffled, augmented pass.

        """
        return WithCoarse(self.data.train_dataloader())

    def eval_dataloader(self) -> WithCoarse:
        """Return the re-iterable test loader, each batch with its superclasses.

        Returns:
          loader: Each iteration is one sequential pass over the test split.

        """
        return WithCoarse(self.data.eval_dataloader())

    def state_dict(self) -> Cifar100Data.StateDict:
        """Return the vision loader's state; the superclass lookup holds none."""
        return self.data.state_dict()

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        """Restore the vision loader's state.

        Args:
          state_dict: Checkpoint from :meth:`state_dict`.

        """
        self.data.load_state_dict(state_dict)


class SizedBatches(Sized, Iterable[dict[str, Tensor]], Protocol):
    """A re-iterable loader of batches with a length."""


class WithCoarse:
    """Adds ``label_coarse`` to every batch of a re-iterable loader."""

    def __init__(self, loader: SizedBatches) -> None:
        self.loader = loader
        self.coarse_of = torch.tensor(COARSE_OF_FINE)

    def __iter__(self) -> Iterator[dict[str, Tensor]]:
        for batch in self.loader:
            yield batch | {"label_coarse": self.coarse_of[batch["label"]]}

    def __len__(self) -> int:
        return len(self.loader)
