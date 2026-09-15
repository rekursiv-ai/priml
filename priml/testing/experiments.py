"""Types shared by the baselines' experiment-chain tests."""

from __future__ import annotations

from typing import Protocol


class ExperimentFactory[T_co](Protocol):
    """An ``expNNN`` factory, named so a parametrized test can report which failed."""

    __name__: str

    def __call__(self) -> T_co:
        """Build the experiment's config tree."""
        ...
