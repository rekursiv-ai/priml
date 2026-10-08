"""Test support for the game: the compiled code of a kernel, for the tests that read it.

Some of ``jit``'s rules cannot be checked by running the game -- no fused
multiply-add in the machine code, no float64 in a float32 kernel -- so their
tests read what Numba compiled, and an intrinsic's tests read the IR its
codegen emits. Nothing at runtime inspects a kernel, so this lives beside the
tests rather than in ``jit``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, Literal, TypedDict, cast

import re

from llvmlite import ir
from numba import njit


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from numba.core.dispatcher import Dispatcher


FMA_MNEMONIC: Final = re.compile(
    r"\bv?fn?m(add|sub)(132|213|231)?[sp][sd]\b|\bfn?m(add|sub|la|ls)\b",
)
"""x86 (``vfmadd231ss`` ...) and arm64 (``fmadd``, ``fmla`` ...) fused multiply-adds."""


def kernel_llvm(dispatcher: Dispatcher[Callable[..., object]]) -> str:
    """Return the LLVM IR of every compiled kernel body, without Python wrappers.

    Numba emits the kernel, a ``cpython`` wrapper that unboxes arguments and
    a ``cfunc`` wrapper. The wrappers handle Python floats as ``double``, so a
    test for float64 leaking into a kernel must look only at the kernel.

    Args:
      dispatcher: A ``jit`` function that has been called at least once.

    Returns:
      ir: The ``define`` blocks of the kernels, concatenated.

    """
    return _kernel_llvm(_inspectable(dispatcher).inspect_llvm().values())


def kernel_assembly(dispatcher: Dispatcher[Callable[..., object]]) -> str:
    """Return the machine assembly of every compiled overload of a ``jit`` function."""
    return "\n".join(_inspectable(dispatcher).inspect_asm().values())


def ir_builder(*arguments: ir.Type) -> tuple[ir.IRBuilder, tuple[ir.Argument, ...]]:
    """Return a builder at the entry of a fresh ``void`` function, and its arguments.

    An intrinsic's codegen runs while a kernel compiles, which a warm cache
    skips, so its test calls the codegen on this builder instead.

    Args:
      *arguments: The function's argument types.

    Returns:
      builder: Positioned in the function's only block.
      args: The function's arguments, one per type.

    """
    function = ir.Function(
        ir.Module(),
        ir.FunctionType(ir.VoidType(), arguments),
        name="kernel",
    )
    return ir.IRBuilder(function.append_basic_block()), function.args


def kernel_inspection(
    dispatcher: Dispatcher[Callable[..., object]],
) -> tuple[str, str]:
    """Return assembly and filtered LLVM from one fresh twin of every overload.

    Args:
      dispatcher: A ``jit`` function that has been called at least once.

    Returns:
      assembly: The machine assembly of every compiled overload.
      llvm: The kernel-body LLVM of every compiled overload, without wrappers.

    """
    twin = _inspectable(dispatcher)
    return "\n".join(twin.inspect_asm().values()), _kernel_llvm(
        twin.inspect_llvm().values(),
    )


def _kernel_llvm(modules: Iterable[str]) -> str:
    """Join each LLVM kernel body after dropping the Python-call wrappers."""
    bodies: list[str] = []
    for module in modules:
        for block in re.finditer(r"^define .*?^}", module, re.DOTALL | re.MULTILINE):
            header = block.group(0).partition("\n")[0]
            if "cpython" in header or "cfunc" in header:
                continue
            bodies.append(block.group(0))
    return "\n".join(bodies)


class _Options(TypedDict, total=False):
    """The ``targetoptions`` a ``jit`` kernel records, as ``njit`` takes them back."""

    nogil: bool
    parallel: bool
    fastmath: bool
    error_model: Literal["python", "numpy"]
    boundscheck: bool


# Numba returns an empty, invalid result from ``inspect_llvm`` and ``inspect_asm`` on a
# kernel it loaded from the on-disk cache, so a test that inspected the cached kernel
# would pass on a warm cache having checked nothing. The twin is compiled here and now
# with the same options.
def _inspectable(
    dispatcher: Dispatcher[Callable[..., object]],
) -> Dispatcher[Callable[..., object]]:
    """Compile an uncached twin of ``dispatcher`` for its recorded signatures."""
    # `njit` sets nopython itself, and warns when it is passed.
    options = dict(dispatcher.targetoptions)
    del options["nopython"]
    twin = njit(**cast("_Options", options))(dispatcher.py_func)
    if not dispatcher.signatures:
        raise ValueError("call the kernel before inspecting it")
    for signature in dispatcher.signatures:
        twin.compile(signature)
    return twin
