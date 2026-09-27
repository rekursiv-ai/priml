"""Build Triton kernels in modules that import Triton lazily.

A module whose kernels only run on CUDA imports ``triton`` through
``wrapt.lazy_import``, so it still imports on a host without the Triton wheel
(macOS has none) and pays nothing until a kernel is first built. Triton
compiles a kernel from its SOURCE and resolves ``language.*`` and
``libdevice.*`` through the function's globals, where it would find those lazy
proxies rather than the modules -- and hashing the kernel for its cache walks
those globals, which cannot copy a proxy. :func:`jit_kernel` rebinds the two
names to the real modules first, so a kernel body reads exactly as it would
with eager imports.

:func:`require_power_of_two` checks a launch size when a config sets it,
before the first compile.
"""

from __future__ import annotations

from importlib import import_module
from types import FunctionType
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Callable

    import triton
else:
    from wrapt import lazy_import

    triton = lazy_import("triton")


def jit_kernel(
    function: Callable[..., object],
    **helpers: triton.JITFunction[..., object],
) -> triton.JITFunction[..., object]:
    """Return ``triton.jit`` of ``function`` with the Triton modules bound.

    Args:
      function: A kernel body written against ``language`` and ``libdevice``.
      **helpers: Jitted device functions the body calls, by the name it uses;
        Triton resolves a callee through the caller's globals, so a helper
        jitted elsewhere must be bound here under that name.

    Returns:
      kernel: The jitted kernel, launched as ``kernel[grid](...)``, or a
        device function another kernel binds as a helper.

    """
    assert isinstance(function, FunctionType)
    bound = FunctionType(
        function.__code__,
        function.__globals__
        | {
            "language": import_module("triton.language"),
            "libdevice": import_module("triton.language.extra.cuda.libdevice"),
        }
        | helpers,
        function.__name__,
        function.__defaults__,
    )
    bound.__annotations__ = function.__annotations__
    return triton.jit(bound)


def require_power_of_two(**sizes: int) -> None:
    """Refuse a launch size Triton cannot tile: each must be a power of two.

    ``language.arange`` extents and warp counts are powers of two, so a
    config that sets another value fails here, at construction, rather than
    inside the first compile.

    Args:
      **sizes: Each size, by the config field that set it.

    Raises:
      ValueError: A size is not a positive power of two; the message names it.

    """
    for name, size in sizes.items():
        if size <= 0 or size & (size - 1):
            raise ValueError(f"{name} must be a positive power of two, not {size}")
