"""Example Usage

```python
# foo.py
import argparse

from configgle import Fig, Makeable

class SomeJob:

    class Config(Fig["SomeJob"]):
        a: int = 123

    def __init__(self, config: Config):
        self.a = config.a

    def run(self, *args: str) -> None:
        parser = argparse.ArgumentParser(description="A specific job.")
        parser.add_argument('--verbose', type=bool, default=False)
        args = parser.parse_args(args)
        print(f"{args=} {self.a=}")

def experiment() -> Makeable[SomeJob]:
    return SomeJob.Config()
```

Then to execute (single process). ``--verbose`` is unknown to the launcher,
so it passes through to ``SomeJob.run``'s own argument parser:

```sh
uv --quiet run --frozen python -m priml.launch foo.experiment --verbose=True
```

Override config fields with ``--override PATH=VALUE`` (cast to the field's
declared type; repeatable; nested paths allowed):

```sh
uv --quiet run --frozen python -m priml.launch foo.experiment --override a=456
```

Multi-GPU runs under ``torch.distributed.run`` (the launcher reads the world
size it sets). Prefer ``python -m torch.distributed.run`` over the ``torchrun``
console script: the module form runs the ``torch`` from the resolved uv
environment, whereas the ``torchrun`` shim is a separate entry point that may
resolve to a different (e.g. global) install. To resume, point at the same
``name`` and enable ``resume``:

```sh
uv --quiet run --frozen python -m torch.distributed.run \
  --standalone --nproc_per_node=8 -m priml.launch \
  your_project.experiments.my_experiment \
  --override name=my_run \
  --override resume=True \
  --override resume_step=-1 \
  --override eval_only=False
```

"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from typing import get_type_hints

import argparse
import dataclasses
import importlib
import inspect
import json
import logging
import platform
import signal
import socket
import time

from configgle import Makeable
from torch.distributed.elastic.multiprocessing.errors import (  # ty: ignore[unresolved-import] -- real torch submodule; ty cannot resolve it, basedpyright (authoritative) does
    record,
)

import torch

from priml.lib.custom_json import decode
from priml.lib.custom_types import JobProtocol, LaunchableExperiment
from priml.logger import setup_logging


@record
def main() -> None:
    t0 = time.perf_counter()
    parser = argparse.ArgumentParser(
        description="A generic launcher.",
        add_help=False,
    )
    _ = parser.add_argument(
        "config",
        type=str,
        help="Path to a callable which returns a Makeable[JobProtocol].",
    )
    _ = parser.add_argument(
        "--loglevel",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)",
    )
    _ = parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="PATH=VALUE",
        help="Override a (possibly nested) config field, e.g. --override "
        "step.lr=3e-4. Repeatable. The value is cast to the field's type.",
    )
    args, unparsed = parser.parse_known_args()

    # Setup logging with specified level
    setup_logging(level=args.loglevel)

    _log_hardware()
    config_str = args.config
    module_name, function_name = config_str.rsplit(".", 1)

    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise ImportError(f"Cannot import module '{module_name}': {e}") from e

    try:
        function = getattr(module, function_name)
    except AttributeError as e:
        raise AttributeError(
            f"Module '{module_name}' has no attribute '{function_name}'",
        ) from e

    if not callable(function):
        raise TypeError(f"'{config_str}' ({function}) is not callable.")

    config = function()
    if not isinstance(config, Makeable):
        raise TypeError(
            f"'{config_str}' returned {type(config).__name__}, not a Makeable.",
        )

    # Apply --override before the name auto-derive so an explicit
    # ``--override experiment_name=...`` wins (and derived fields key off it at
    # finalize).
    apply_overrides(config, args.override)

    docstring = inspect.getdoc(function) or ""
    _log_docstring(config_str, docstring)
    # A LaunchableExperiment gets its run identity + docstring stamped here when
    # unset; any other config (a standalone job) is launched untouched.
    if isinstance(config, LaunchableExperiment):
        _stamp_run_identity(
            config, module_name=module_name, function_name=function_name
        )
        if docstring and not config.doc:
            config.doc = docstring
    _log_config(config_str, config)

    job = config.make()
    if not isinstance(job, JobProtocol):
        raise TypeError(
            f"'{config_str}' returned {type(job).__name__}, not a JobProtocol.",
        )

    with _graceful_sigterm():
        job.run(*unparsed)
    logging.getLogger(__name__).info(
        "Total program time: %.1fs", time.perf_counter() - t0
    )


def apply_overrides(config: Makeable[object], overrides: list[str]) -> None:
    """Apply ``PATH=VALUE`` config overrides in place.

    Each override names a (possibly nested) field by a dotted path and a
    raw value. Traversal validates every hop against the node's declared
    fields (never an arbitrary attribute or method); the leaf is set to the
    raw value coerced to its declared type. ``VALUE`` is first parsed as a
    JSON literal (so ``true`` / ``3e-4`` / ``[1, 2]`` decode naturally),
    falling back to the bare string, then cast against the field annotation.

    Args:
      config: The configgle config (a dataclass) to mutate.
      overrides: ``PATH=VALUE`` strings, e.g. ``["step.lr=3e-4", "name=run"]``.

    Raises:
      ValueError: An override lacks ``=``, traverses a non-config node, names
        a field absent from the node's declared fields, or carries a value
        that cannot coerce to the field's declared type.

    """
    for override in overrides:
        if "=" not in override:
            raise ValueError(
                f"Malformed override `{override}`; expected PATH=VALUE.",
            )
        path, _, raw = override.partition("=")
        keys = path.split(".")
        if not path or "" in keys:
            raise ValueError(
                f"Malformed override `{override}`; PATH must be a dotted "
                "field path with no empty segments (e.g. `step.lr`).",
            )
        node = config
        for key in keys[:-1]:
            _, node = _override_field(node, key, path)
        leaf = keys[-1]
        annotation, _ = _override_field(node, leaf, path)
        try:
            value = decode(annotation, _parse_override_value(raw))
        except (TypeError, ValueError) as e:
            # ``decode`` raises ``TypeError`` when ``raw`` cannot coerce to the
            # field type -- e.g. a scalar for a nested-config field
            # (``child=5`` instead of ``child.lr=5``). Re-raise as a
            # path-naming ``ValueError`` so the caller sees one error contract.
            raise ValueError(
                f"Override path `{path}` cannot set `{raw}` "
                f"on a field of type {annotation}: {e}",
            ) from e
        # We own ``config`` and mutate it in place, writing the field directly on
        # the real (per-instance, ``default_factory``-built, never shared) node.
        # ``object.__setattr__`` is the primitive configgle itself uses to write
        # frozen Figs (see ``fig.py`` ``finalize``): it writes through
        # ``frozen=True`` while still honoring ``slots`` (an unknown leaf raises
        # ``AttributeError``). ``_override_field`` already rejected unknown
        # fields, so only declared, type-checked fields reach here.
        object.__setattr__(node, leaf, value)


def _override_field(node: object, key: str, path: str) -> tuple[object, object]:
    """Return ``(annotation, value)`` of ``node``'s declared field ``key``.

    Validates ``key`` against ``node``'s resolved type hints -- a declared
    field, never an arbitrary attribute or method -- so each hop of an
    override path is checked uniformly. The annotation drives leaf casting;
    the value continues the traversal.

    Raises:
      ValueError: ``node`` is not a dataclass, or has no declared field ``key``.

    """
    if not dataclasses.is_dataclass(node):
        raise ValueError(
            f"Override path `{path}` traverses non-config "
            f"{type(node).__name__}, which has no field `{key}`.",
        )
    hints = get_type_hints(type(node))
    if key not in hints:
        raise ValueError(
            f"Override path `{path}` has no field `{key}` on {type(node).__name__}.",
        )
    return hints[key], getattr(node, key)


def _parse_override_value(raw: str) -> object:
    """Parse a raw override value as a JSON literal, else the bare string."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _log_config(config_str: str, config: Makeable[object]) -> None:
    """Pretty-print the full resolved experiment config to the log.

    Logs every field (defaults included) of the launched config so a run's
    exact setup is recoverable from its log alone. ``pformat`` is a configgle
    ``Fig`` method; a non-Fig ``Makeable`` falls back to ``repr``.
    """
    pformat = getattr(config, "pformat", None)
    if callable(pformat):
        body = pformat(finalize=True, hide_default_values=False)
    else:
        body = repr(config)
    logging.getLogger(__name__).info("experiment config %s:\n%s", config_str, body)


def _stamp_run_identity(
    config: LaunchableExperiment,
    *,
    module_name: str,
    function_name: str,
) -> None:
    """Fill a launchable experiment's run-identity fields when left unset.

    ``experiment_name`` defaults to the factory function name; ``study_name`` to
    the run-family prefix derived from the module path (see
    :func:`_derive_study_name`). An explicit value on either field wins -- this
    only fills a blank, so the experiment stays the source of truth. The
    config's own ``finalize`` then keys its paths off these fields.
    """
    if not config.experiment_name:
        config.experiment_name = function_name
    if not config.study_name:
        config.study_name = _derive_study_name(module_name)


def _derive_study_name(module_name: str) -> str:
    """Derive the ``study_name`` prefix from an ``experimental`` module path.

    ``<pkg>.experimental.<a>.<b>.experiments`` -> ``"<a>/<b>/"``. A path without
    the ``experimental``/``experiments`` markers yields ``""`` (no prefix).
    """
    parts = module_name.split(".")
    try:
        exp_idx = parts.index("experimental")
        between = parts[exp_idx + 1 : parts.index("experiments")]
    except ValueError:
        return ""
    study = "/".join(between)
    return f"{study}/" if study else ""


def _log_docstring(config_str: str, docstring: str) -> None:
    """Print the experiment function's docstring so the run log says WHAT it is.

    The docstring is the experiment's contract -- hypothesis, changes, and (once
    finished) outcome. Printing it at launch makes a job log self-describing: a
    reader sees the intent next to the metrics, not just an opaque config.
    """
    if not docstring:
        logging.getLogger(__name__).info(
            "experiment %s: no docstring (add hypothesis/changes/outcome)",
            config_str,
        )
        return
    logging.getLogger(__name__).info(
        "experiment %s docstring:\n%s", config_str, docstring
    )


@contextmanager
def _graceful_sigterm() -> Generator[None]:
    """Turn SIGTERM into KeyboardInterrupt so the job's cleanup can run.

    A scheduler stops a job with SIGTERM, whose default action is an immediate
    process exit -- skipping the job's ``try/finally`` cleanup. That cleanup is
    what flushes buffered tracker state (e.g. ``WandbTracker.close`` ->
    ``run.finish``, which transmits history not yet sent under the W&B flush
    interval), so a hard SIGTERM silently drops the run's last metrics. Raising
    ``KeyboardInterrupt`` from the handler instead unwinds through the job's
    ``finally``, flushing before exit. Only rank-aware code below the job runs;
    the handler itself is reentrant-safe (it just raises). Restores the prior
    handler on exit so nested/repeat launches are unaffected.
    """

    def _raise_on_sigterm(signum: int, _frame: object) -> None:
        del _frame
        logging.getLogger(__name__).info(
            "received signal %d; raising to run cleanup before exit", signum
        )
        raise KeyboardInterrupt

    previous = signal.signal(signal.SIGTERM, _raise_on_sigterm)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def _log_hardware() -> None:
    """Log hardware banner: hostname, GPU/device, OS."""
    if torch.cuda.is_available():
        dev = torch.cuda.get_device_name()
    elif torch.backends.mps.is_available():
        dev = "mps"
    else:
        dev = "cpu"
    logging.getLogger(__name__).info(
        "hw: %s | %s | %s",
        socket.gethostname(),
        dev,
        platform.platform(),
    )


if __name__ == "__main__":
    main()
