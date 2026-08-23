"""Utilities for torch.compile diagnostics and lazy-compilation decorators.

Module-level ``@torch.compile`` and ``@torch.compiler.assume_constant_result``
load ~400 torch modules (dynamo, inductor, functorch, functorch's symbolic
shapes, ...) just to construct the lazy trampoline — about 1s each. ``lazy_compile``
and ``lazy_assume_constant_result`` defer that construction to first call, so
modules that decorate but never run in a given process pay no import-time cost.
First call is slower by the deferred amount; subsequent calls are identical to
the non-lazy version.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast, overload

import functools
import traceback

import torch


_compile_traces = dict[str, list[str]]()


@overload
def lazy_torch_compile[**P, R](fn: Callable[P, R], /) -> Callable[P, R]: ...


@overload
def lazy_torch_compile[**P, R](
    **compile_kwargs: Any,
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


def lazy_torch_compile(*compile_args: Any, **compile_kwargs: Any) -> Callable[..., Any]:
    """Lazy ``@torch.compile`` — defers dynamo/inductor imports to first call.

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

    Returns:
      result: The lazily-compiled function when applied bare, otherwise a
        decorator awaiting the function to compile.

    """
    # Bare ``@lazy_torch_compile``: the lone positional is the decorated
    # function, not a ``torch.compile`` argument -- decorate it directly.
    if len(compile_args) == 1 and not compile_kwargs and callable(compile_args[0]):
        return _make_lazy_compiled(compile_args[0])

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        return _make_lazy_compiled(fn, *compile_args, **compile_kwargs)

    return decorator


def _make_lazy_compiled[**P, R](
    fn: Callable[P, R], *compile_args: Any, **compile_kwargs: Any
) -> Callable[P, R]:
    """Wrap ``fn`` so ``torch.compile`` runs on first call, not at decoration."""
    compiled: Callable[P, R] | None = None

    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        nonlocal compiled
        target = compiled
        if target is None:
            target = cast(
                Callable[P, R], torch.compile(*compile_args, **compile_kwargs)(fn)
            )
            compiled = target
        return target(*args, **kwargs)

    return wrapper


def lazy_assume_constant_result(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Lazy ``@torch.compiler.assume_constant_result`` — defers dynamo imports to first call.

    Same semantics as ``torch.compiler.assume_constant_result``; see
    its docs for what the wrapping buys you inside a compiled region.
    This version pays the dynamo import cost on first call rather
    than at module load.
    """
    wrapped: Callable[..., Any] | None = None

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
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
    always_print: bool = False,
) -> int:
    """Track and optionally limit recompilations. Safe to call from compiled code.

    Call this inside a compiled function to record each (re)compilation.
    When ``max_compiles`` is exceeded, raises ``RuntimeError`` with all
    collected stack traces so you can diagnose guard failures.

    Enable verbose torch recompilation logging with::

        TORCH_LOGS="recompiles_verbose" python script.py

    Args:
        key: Identifier for this compilation site.
        max_compiles: Raise after this many compiles (-1 = unlimited).
        always_print: Print the stack trace on every compile.

    Returns:
        Number of compiles seen so far for this key.

    """
    trace = "".join(traceback.format_stack()[:-1])
    traces = _compile_traces.setdefault(key, [])
    traces.append(trace)
    if always_print:
        print(trace)  # noqa: T201
    if max_compiles > -1 and len(traces) > max_compiles:
        traces_str = "" if always_print else ("\n" + "\n--------\n".join(traces))
        raise RuntimeError(f"Too many compiles ({len(traces)}) for {key}.{traces_str}")
    return len(traces)
