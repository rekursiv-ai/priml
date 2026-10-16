"""A training loader the loop can iterate once per pass over the data.

``TrainLoop`` builds the training loader once and calls ``iter`` on it again at
every epoch boundary. A generator returned from ``train_dataloader`` is its
own iterator, so the second ``iter`` hands back the exhausted first pass.
:class:`Passes` wraps the function that draws one pass, so each ``iter``
starts a fresh one.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass


@dataclass(frozen=True, slots=True, kw_only=True)
class Passes[T]:
    """Re-iterable batches; every ``iter`` calls ``draw`` for a new pass.

    Attributes:
      draw: Returns one pass of batches. Called once per ``iter``, so a pass
        may reshuffle or reseed.

    """

    draw: Callable[[], Iterator[T]]

    def __iter__(self) -> Iterator[T]:
        """Start a new pass."""
        return self.draw()
