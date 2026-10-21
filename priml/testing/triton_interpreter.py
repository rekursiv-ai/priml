"""Run Triton kernels on the CPU through Triton's interpreter, then undo it.

Triton's interpreter rewrites ``triton.language`` in place. A device-function
call patches the language modules and never restores them, and interpreting a
library function executes interpreter code in that library's globals. Any of
it left behind makes every later GPU compile in the process fail, so
:func:`triton_interpreter` snapshots every module it or the interpreter can set
attributes on, its own runtime hooks included, and restores them on exit.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from functools import partial
from typing import TYPE_CHECKING, Protocol, cast

import importlib
import inspect
import os
import pkgutil
import sys

from triton.runtime import interpreter, jit

import triton


if TYPE_CHECKING:
    from collections.abc import Callable, Generator
    from types import ModuleType
    from typing_extensions import TypeIs

    from numpy.typing import NDArray

    import numpy as np


@contextmanager
def triton_interpreter(*caches: _Cache) -> Generator[None]:
    """Interpret every Triton launch inside the block on the CPU.

    A kernel jitted outside the block compiles and one jitted inside interprets,
    so each cache of jitted kernels is cleared on entry and on exit. Every
    ``triton.language`` submodule is imported first, so the snapshot taken before
    any patch covers each module the interpreter can touch; whatever step fails,
    the steps before it are undone.

    Args:
      caches: The ``functools.cache`` factories that jit the kernels under test.

    Yields:
      none: Control, with ``TRITON_INTERPRET`` set and the interpreter patched.

    """
    with ExitStack() as undo:
        for cache in caches:
            cache.cache_clear()
            undo.callback(cache.cache_clear)
        for module in pkgutil.walk_packages(
            triton.language.__path__,
            "triton.language.",
        ):
            importlib.import_module(module.name)
        for owner in _patched_owners():
            undo.callback(_restore, owner, dict(cast("dict[str, object]", vars(owner))))
        undo.enter_context(_interpret_flag())
        jit._normalize_ty = partial(  # noqa: SLF001 -- Triton has no public hook for it.
            _normalize_bare_constexpr,
            normalize=jit._normalize_ty,  # noqa: SLF001 -- Likewise.
        )
        interpreter._patch_lang_tensor = partial(  # noqa: SLF001 -- Likewise.
            _patch_scalar_index,
            patch=interpreter._patch_lang_tensor,  # noqa: SLF001 -- Likewise.
        )
        _jit_library()
        yield


def is_jit(value: object) -> TypeIs[jit.JITFunction[..., object]]:
    """Whether ``value`` is a jitted Triton function."""
    return isinstance(value, jit.JITFunction)


# The interpreter matches a constexpr only by its bare name; postponed annotations
# spell it ``language.constexpr``.
def _normalize_bare_constexpr(
    ty: object,
    *,
    normalize: Callable[[object], str],
) -> str:
    return normalize(
        ty.strip("'").removeprefix("language.") if isinstance(ty, str) else ty,
    )


# The interpreter holds a scalar as a one-element array, which numpy 2.5 no longer
# converts with ``int``; a runtime loop bound needs it.
def _patch_scalar_index(
    tensor: type,
    scope: _PatchScope,
    *,
    patch: Callable[[type, _PatchScope], None],
) -> None:
    patch(tensor, scope)
    scope.set_attr(tensor, "__index__", _scalar_index)


def _scalar_index(tensor: _InterpretedTensor) -> int:
    return int(tensor.handle.data.item())  # pyrefly: ignore[unnecessary-type-conversion] -- The stub's type hides the runtime value (numpy/torch scalar or non-str key); the conversion is load-bearing.


@contextmanager
def _interpret_flag() -> Generator[None]:
    """Set ``TRITON_INTERPRET`` for the block and put the caller's value back."""
    interpret = os.environ.get("TRITON_INTERPRET")
    os.environ["TRITON_INTERPRET"] = "1"
    try:
        yield
    finally:
        if interpret is None:
            del os.environ["TRITON_INTERPRET"]
        else:
            os.environ["TRITON_INTERPRET"] = interpret


# Triton jitted its library functions at import, compiled-mode, which the interpreter
# refuses to call.
def _jit_library() -> None:
    """Rebind each jitted library function as an interpreted one."""
    for module in (triton.language, triton.language.standard):
        for name, value in inspect.getmembers(module, is_jit):
            setattr(module, name, interpreter.InterpretedFunction(value.fn))


class _Cache(Protocol):
    def cache_clear(self) -> None: ...


class _PatchScope(Protocol):
    def set_attr(self, obj: object, name: str, value: object) -> None: ...


class _InterpretedHandle(Protocol):
    data: NDArray[np.integer]


class _InterpretedTensor(Protocol):
    handle: _InterpretedHandle


def _patched_owners() -> list[object]:
    """Every module and class the interpreter or this context sets attributes on."""
    hooks: list[object] = [jit, interpreter]
    modules: list[ModuleType] = [
        module
        for name, module in list(sys.modules.items())
        if name == "triton.language" or name.startswith("triton.language.")
    ]
    classes = {
        id(value): value
        for module in modules
        for value in cast("dict[str, object]", vars(module)).values()
        if isinstance(value, type) and value.__module__.startswith("triton.language")
    }
    return [*hooks, *modules, *classes.values()]


def _restore(owner: object, attributes: dict[str, object]) -> None:
    """Put back each attribute ``owner`` held and drop each one it gained."""
    current = cast("dict[str, object]", vars(owner))
    # A submodule imported inside the block binds to its package and stays in
    # ``sys.modules``; dropping the binding breaks every later attribute import.
    gained = [
        name
        for name, value in current.items()
        if name not in attributes and not _is_submodule(value, of=owner)
    ]
    for name in gained:
        delattr(owner, name)
    for name, value in attributes.items():
        if current.get(name) is not value:
            setattr(owner, name, value)


def _is_submodule(value: object, *, of: object) -> bool:
    """Whether ``value`` is a module inside the package ``of``."""
    return (
        inspect.ismodule(value)
        and inspect.ismodule(of)
        and value.__name__ == f"{of.__name__}.{value.__name__.rpartition('.')[2]}"
    )
