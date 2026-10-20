"""Subject-disjoint validation policies: grouped k-fold or one fixed cohort."""

from __future__ import annotations

from dataclasses import field
from typing import TYPE_CHECKING, cast

from configgle import Fig
from treekle import from_plain

from priml.data.folds import grouped_folds


if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    import numpy as np

    from priml.baselines.motor_imagery.data import Trial
else:
    from wrapt import lazy_import

    np = lazy_import("numpy")


class GroupedFolds:
    """Balance whole people across folds exactly as the public notebook does."""

    class Config(Fig["GroupedFolds"]):
        count: int = 2
        """GroupKFold fold count."""

    def __init__(self, config: Config) -> None:
        self.count = config.count

    def __call__(self, trials: Sequence[Trial]) -> Iterator[list[str]]:
        """Yield each fold's validation people in bin order.

        Args:
          trials: Training cues; every window counts toward its person's size.

        Yields:
          people: Sorted validation people of one fold.

        """
        groups = np.asarray(
            [trial.subject for trial in trials for _ in range(len(trial.windows))],
            dtype=np.str_,
        )
        for _, validation in grouped_folds(groups, count=self.count):
            yield from_plain(
                cast(object, np.unique(groups[validation]).tolist()),
                list[str],
            )


class FixedCohort:
    """Hold out one fixed set of people."""

    class Config(Fig["FixedCohort"]):
        subjects: list[str] = field(default_factory=list[str])
        """People excluded completely from training."""

    def __init__(self, config: Config) -> None:
        self.subjects = config.subjects

    def __call__(self, trials: Sequence[Trial]) -> Iterator[list[str]]:
        """Yield the one fold, after checking it leaves a training cohort.

        Args:
          trials: Training cues whose people must contain the cohort.

        Yields:
          people: The configured validation people.

        Raises:
          ValueError: The cohort is empty, has unknown people, or is everyone.

        """
        people = set(self.subjects)
        available = {trial.subject for trial in trials}
        if not people or people - available or people == available:
            raise ValueError("Fixed validation people must leave a training cohort.")
        yield list(self.subjects)
