"""Generated MLS-Bench experiments, one module per task.

Every (task, baseline, seed) is a function ``expNNN()`` in
``priml.baselines.mlsbench.tasks.<task_module>``, where ``<task_module>`` is the
task name with ``-`` replaced by ``_``. The modules are written by
``scripts/generate_experiments.py`` from ``registry.json``; numbering is local
to a task. This package stays light on purpose: it imports no task module, so
launching ``priml.baselines.mlsbench.tasks.llm_dllm_demask_strategy.exp002``
with ``uv --quiet run --frozen python -m priml`` loads one task.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import importlib
import pkgutil


if TYPE_CHECKING:
    from collections.abc import Callable

    from priml.baselines.mlsbench.experiments import MlsbenchEval


def module_of(task: str) -> str:
    """Return the module name of MLS-Bench task ``task`` (``-`` -> ``_``)."""
    return task.replace("-", "_")


def task_modules() -> tuple[str, ...]:
    """Return every generated task module name, sorted, without importing any."""
    return tuple(sorted(m.name for m in pkgutil.iter_modules(__path__)))


def resolve(name: str) -> Callable[[], MlsbenchEval]:
    """Import one task module and return its experiment function.

    Args:
      name: ``<task_module>.<expNNN>``, e.g. ``"llm_dllm_demask_strategy.exp002"``.

    Returns:
      factory: The experiment function.

    Raises:
      KeyError: ``name`` is not ``<task_module>.<function>`` of a known task.

    """
    mod, _, fn = name.partition(".")
    if mod not in task_modules() or not fn.startswith("exp"):
        raise KeyError(name)
    module = importlib.import_module(f"{__name__}.{mod}")
    try:
        return cast("Callable[[], MlsbenchEval]", getattr(module, fn))
    except AttributeError:
        raise KeyError(name) from None
