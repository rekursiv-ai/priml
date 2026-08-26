"""Transactional evaluation helpers shared by Craftax trainers."""

from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING

import copy


if TYPE_CHECKING:
    from torch import nn


@contextmanager
def evaluation_mode(model: nn.Module) -> Generator[None]:
    """Temporarily evaluate a model, restoring its incoming mode."""
    training = model.training
    model.eval()
    try:
        yield
    finally:
        model.train(training)


@contextmanager
def evaluation_transaction[StateT](
    *,
    model: nn.Module,
    save: Callable[[], StateT],
    restore: Callable[[StateT], None],
) -> Generator[None]:
    """Run evaluation against a snapshot and restore all training state."""
    state = copy.deepcopy(save())
    with evaluation_mode(model):
        try:
            yield
        finally:
            restore(state)
