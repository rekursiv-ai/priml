"""Utilities for torch.compile diagnostics and lazy-compilation decorators.

Module-level ``@torch.compile`` and ``@torch.compiler.assume_constant_result``
load ~400 torch modules (dynamo, inductor, functorch, functorch's symbolic
shapes, ...) just to construct the lazy trampoline -- about 1s each. ``lazy_compile``
and ``lazy_assume_constant_result`` defer that construction to first call, so
modules that decorate but never run in a given process pay no import-time cost.
First call is slower by the deferred amount; subsequent calls are identical to
the non-lazy version.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Final, cast, overload

import functools
import logging
import traceback

import torch


logger = logging.getLogger(__name__)

_MAX_TRACES_PER_KEY: Final = 32
"""Stacks kept per key: enough to diagnose a guard failure, bounded for a run."""

# Process-global by necessity: ``trace_compile`` runs inside compiled code via
# ``assume_constant_result``, which can thread no state through its arguments.
_compile_traces = dict[str, list[str]]()
_compile_counts = dict[str, int]()


@overload
def lazy_torch_compile[**P, R](fn: Callable[P, R], /) -> Callable[P, R]: ...


@overload
def lazy_torch_compile[**P, R](
    **compile_kwargs: object,
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


def lazy_torch_compile(
    *compile_args: object,
    **compile_kwargs: object,
) -> Callable[..., object]:
    """Lazy ``@torch.compile`` -- defers dynamo/inductor imports to first call.

    Mirrors ``torch.compile``'s dual calling convention: use bare
    (``@lazy_torch_compile``) or parameterized
    (``@lazy_torch_compile(fullgraph=True)``). Parameterized arguments
    (``fullgraph``, ``dynamic``, ``mode``, ``backend``, ``options``, etc.)
    are forwarded verbatim on first invocation; see ``help(torch.compile)``
    for the full reference.

    Module-level ``@torch.compile`` forces ~400 torch internals
    (dynamo, inductor, functorch, ...) to load at import time just
    to construct the trampoline. This wrapper defers that to first
    call, so processes that import but never invoke pay zero cost.

    Args:
      *compile_args: The function to decorate when applied bare; any other
        positional raises ``TypeError``.
      **compile_kwargs: Keyword arguments forwarded to torch.compile().

    Returns:
      result: The lazily-compiled function when applied bare, otherwise a
        decorator awaiting the function to compile.

    """
    # Bare ``@lazy_torch_compile``: the lone positional is the decorated
    # function, not a ``torch.compile`` argument -- decorate it directly.
    if compile_args:
        if len(compile_args) == 1 and callable(compile_args[0]) and not compile_kwargs:
            return _make_lazy_compiled(compile_args[0])
        raise TypeError("lazy_torch_compile accepts only keyword compile arguments.")

    def decorator(fn: Callable[..., object]) -> Callable[..., object]:
        return _make_lazy_compiled(fn, **compile_kwargs)

    return decorator


def lazy_assume_constant_result[**P, R](fn: Callable[P, R]) -> Callable[P, R]:
    """Lazy ``@torch.compiler.assume_constant_result`` -- defers dynamo imports to first call.

    Same semantics as ``torch.compiler.assume_constant_result``; see
    its docs for what the wrapping buys you inside a compiled region.
    This version pays the dynamo import cost on first call rather
    than at module load.

    Args:
      fn: The function whose result dynamo may treat as constant.

    Returns:
      wrapper: Calls ``fn`` through ``assume_constant_result``, wrapping it on
        first use.

    """
    wrapped: Callable[P, R] | None = None

    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        nonlocal wrapped
        target = wrapped
        if target is None:
            target = torch.compiler.assume_constant_result(fn)
            wrapped = target
        return target(*args, **kwargs)

    return wrapper


@lazy_assume_constant_result
def trace_compile(
    key: str,
    *,
    max_compiles: int = -1,
    always_log: bool = False,
) -> int:
    """Track and optionally limit recompilations. Safe to call from compiled code.

    Call this inside a compiled function to record each (re)compilation.
    When ``max_compiles`` is exceeded, raises ``RuntimeError`` with the
    collected stack traces (the most recent ``_MAX_TRACES_PER_KEY``) so you
    can diagnose guard failures.

    Enable verbose torch recompilation logging with::

        TORCH_LOGS="recompiles_verbose" python script.py

    Args:
        key: Identifier for this compilation site.
        max_compiles: Raise after this many compiles (-1 = unlimited).
        always_log: Log the stack trace at WARNING on every compile.

    Returns:
        count: Number of compiles seen so far for this key.

    """
    trace = "".join(traceback.format_stack()[:-1])
    traces = _compile_traces.setdefault(key, [])
    traces.append(trace)
    del traces[:-_MAX_TRACES_PER_KEY]
    count = _compile_counts[key] = _compile_counts.get(key, 0) + 1
    if always_log:
        logger.warning("Compile %s of %s:\n%s", count, key, trace)
    if max_compiles > -1 and count > max_compiles:
        traces_str = "" if always_log else ("\n" + "\n--------\n".join(traces))
        raise RuntimeError(f"Too many compiles ({count}) for {key}.{traces_str}")
    return count


def _make_lazy_compiled[**P, R](
    fn: Callable[P, R],
    **compile_kwargs: object,
) -> Callable[P, R]:
    """Wrap ``fn`` so ``torch.compile`` runs on first call, not at decoration."""
    compiled: Callable[P, R] | None = None

    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        nonlocal compiled
        target = compiled
        if target is None:
            # ``torch.compile`` is overloaded on a keyword-only signature; the
            # forwarded splat is opaque to it, so the decorator it returns is
            # rebuilt as the ``Callable[P, R]`` it is documented to be.
            compile_fn = cast(
                Callable[..., Callable[[Callable[P, R]], Callable[P, R]]],
                torch.compile,
            )
            target = compile_fn(**compile_kwargs)(fn)
            compiled = target
        return target(*args, **kwargs)

    return wrapper
