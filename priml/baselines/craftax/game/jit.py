"""The one Numba decorator every game kernel uses, C's clamps, and builtins Numba lacks.

The game is numpy compiled by Numba rather than torch; the package docstring
gives the measured reasons.

The options are pinned because each default breaks bit parity with the C env:

- ``error_model="numpy"``: the ``"python"`` default raises on a float divide by
  zero, where C produces ``inf`` or ``NaN``.
- ``fastmath=False``: fast math permits FMA contraction and reassociation,
  either of which changes float32 bits. The C env's host code has no FMA.
- ``boundscheck=False``: the port never reads out of bounds (every C alias is
  an explicit branch), so the check would only cost time. Tests may re-enable
  it through ``NUMBA_BOUNDSCHECK=1``.
- ``nogil=True``: the env steps on Python threads, so a kernel must release
  the GIL.
- Caching: compiled kernels persist under ``NUMBA_CACHE_DIR`` (else beside the
  source, else in Numba's user-wide cache, as Numba's own chain), stamped with
  :func:`source_stamp` -- a hash of every ``game/`` source and of the kernel's
  own file -- rather than, as Numba's own stamp is, the kernel's file alone. A
  kernel's machine code embeds its callees, so under Numba's stamp a caller
  cached before a callee's file changed ran the old callee (measured); now an
  edit to ``game/`` or to the kernel's file recompiles it on first use. So a
  kernel outside ``game/`` calls only kernels of ``game/`` and of its own file
  (``jit_test.py`` checks it): a callee in another file is outside its stamp,
  and an edit there would leave it stale.
  The stamp leaves the rest of the kernel's package out on purpose: with it,
  an edit to any ``.py`` beside ``env.py`` (``experiments.py`` included)
  recompiled the env's kernels, 34-55 s on the Xeon. The stamp also
  names the platform's libm (:func:`platform_key`): the step embeds light
  levels computed through it when it compiled (``rules.daylight_table``), so a
  cache shared across libcs would replay another libm's bits.

Kernels also keep a few spellings that ``jit_test.py`` enforces: float32
constants everywhere; ``np.cos``, ``np.sin``, ``np.sqrt``, ``np.power`` and
``np.fmod`` on float32 scalars (typed below as :data:`cosf`, :data:`sinf`,
:data:`sqrtf`, :data:`powf` and :data:`fmodf`), which Numba lowers to the libm
calls of those names that the C env makes, where ``x ** 3`` would become
multiplications and ``x ** 0.5`` a different rounding; and explicit uint32
masking.

C's ``/`` and ``%`` on ``int`` are Python's ``//`` and ``%`` here. Python
floors toward negative infinity and C truncates toward zero, which differ when
the operands' signs differ; the game divides only indices, sizes and
``rand_r`` draws, which are never negative, so the two agree.

The clamps mirror ``clampi`` and ``clampf`` in ``craftax.h``, including their
order of comparison: a value below ``low`` returns ``low`` before ``high`` is
consulted, which matters when a caller passes ``low > high``. They are kernels
rather than ``min(max(value, low), high)`` because Numba's ``min`` and ``max``
refuse the ``IntEnum`` members (``MobType``, ``ProjectileType``) that several
callers pass.

``popcount`` and ``trailing_zeros`` are ``__builtin_popcountll`` and
``__builtin_ctzll``: LLVM's ``ctpop`` and ``cttz`` intrinsics, one instruction
each on the Xeon, which Numba exposes no spelling for. ``prefetch`` is
``__builtin_prefetch``, LLVM's ``prefetch``: a hint, with no effect on any value.

References:
    https://github.com/PufferAI/PufferLib
        Suarez. PufferLib (MIT license), ``ocean/craftax/craftax.h``, pin
        ``6ffa5b10``.

"""

from __future__ import annotations

from functools import cache, partial
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, cast, override, runtime_checkable

import hashlib
import platform
import sys

from llvmlite import ir
from numba import njit
from numba.core.caching import (
    CompileResultCacheImpl,
    FunctionCache,
    InTreeCacheLocator,
    UserProvidedCacheLocator,
    UserWideCacheLocator,
)
from numba.extending import intrinsic

import numba.core.types as nbtypes
import numpy as np


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from llvmlite.ir import IRBuilder, Value
    from numba.core.base import BaseContext
    from numba.core.dispatcher import Dispatcher
    from numba.core.typing.templates import Signature


_CWD: Final = Path(__file__).resolve().parent
"""``game/``: the package whose sources stamp every kernel's cache."""


def jit[F: Callable[..., object]](fn: F) -> Dispatcher[F]:
    """Compile a game kernel with the pinned options; the only decorator ``game/`` may use.

    Args:
      fn: The kernel's Python source.

    Returns:
      kernel: The compiled dispatcher.

    """
    return _cached(
        njit(
            nogil=True,
            cache=False,
            error_model="numpy",
            fastmath=False,
            boundscheck=False,
        )(fn),
    )


def jit_parallel[F: Callable[..., object]](fn: F) -> Dispatcher[F]:
    """Compile with ``prange`` support, for whole batches off the step path.

    The world pool build and the parity suite's random-policy statistics use
    it, and nothing on the step path.

    A parallel region needs Numba's threading layer, and concurrent parallel
    regions from several Python threads need the ``omp`` or ``tbb`` layer, so
    the step path stays on plain ``nogil`` calls from the buffer threads.

    Args:
      fn: The kernel's Python source.

    Returns:
      kernel: The compiled dispatcher.

    """
    return _cached(
        njit(
            parallel=True,
            nogil=True,
            cache=False,
            error_model="numpy",
            fastmath=False,
            boundscheck=False,
        )(fn),
    )


def package_digest(directory: Path) -> str:
    """Return a hash of every non-test ``*.py`` in ``directory``, names and bytes.

    Args:
      directory: The package, ``game/`` for its kernels' cache stamp.

    Returns:
      digest: 16 hex digits.

    """
    digest = hashlib.sha1(usedforsecurity=False)
    for path in sorted(directory.glob("*.py")):
        if not path.name.endswith("_test.py"):
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def source_stamp(source: Path) -> str:
    """Return the cache stamp of the kernels ``source`` defines.

    The digest of ``game/``; the hash of ``source`` itself, which the digest
    leaves out when it is a test or lies outside ``game/``; then the
    :func:`platform_key` whose libm the kernels' constants were computed
    through.

    Args:
      source: The file that defines the kernels.

    Returns:
      stamp: The digest, the hash and the platform key, joined by ``-``.

    """
    own = hashlib.sha1(source.read_bytes(), usedforsecurity=False)
    return "-".join((package_digest(_CWD), own.hexdigest()[:16], platform_key()))


@cache
def platform_key() -> str:
    """Return the key a libm-dependent golden is minted under.

    ``cosf``, ``sinf``, ``powf`` and ``fmodf`` are the platform libm's, so a
    golden of their outputs holds only where that libm runs. PufferLib's
    reference container image is ``linux-x86_64-glibc2.35``; ``glibc2.39`` is
    Ubuntu 24.04's. A test whose golden carries another key skips and
    names the mint command. Taken once per process: ``libc_ver`` scans the
    interpreter binary, and every kernel file's cache stamp needs the key.

    Returns:
      key: ``<os>-<machine>-<libc><version>``.

    """
    libc, version = platform.libc_ver()
    return f"{sys.platform}-{platform.machine()}-{libc or 'libsystem'}{version}"


def _cached[F: Callable[..., object]](dispatcher: Dispatcher[F]) -> Dispatcher[F]:
    """Give ``dispatcher`` the on-disk cache stamped with :func:`source_stamp`."""
    # Numba's ``enable_caching`` builds its own ``FunctionCache``; the dispatcher
    # has no other door to a cache class, so the stamped one is set directly.
    assert isinstance(dispatcher, _Caching)
    dispatcher._cache = _PackageCache(dispatcher.py_func)  # noqa: SLF001 -- Numba offers no public hook for a dispatcher's cache class.
    return dispatcher


@cache
def _stamp(source: Path) -> str:
    """Return :func:`source_stamp`, taken once per file and process."""
    return source_stamp(source)


@runtime_checkable
class _Caching(Protocol):
    """A dispatcher's cache slot, which ``Dispatcher.enable_caching`` fills."""

    _cache: FunctionCache


class _UserProvidedPackageLocator(UserProvidedCacheLocator):
    """Under ``NUMBA_CACHE_DIR``, stamped with :func:`source_stamp`."""

    @override
    def get_source_stamp(self) -> str:
        """Return :func:`source_stamp` of the kernel's file."""
        return _stamp(Path(self._py_file))


class _InTreePackageLocator(InTreeCacheLocator):
    """Beside the source in ``__pycache__``, stamped with :func:`source_stamp`."""

    @override
    def get_source_stamp(self) -> str:
        """Return :func:`source_stamp` of the kernel's file."""
        return _stamp(Path(self._py_file))


class _UserWidePackageLocator(UserWideCacheLocator):
    """In Numba's user-wide cache directory, stamped with :func:`source_stamp`."""

    @override
    def get_source_stamp(self) -> str:
        """Return :func:`source_stamp` of the kernel's file."""
        return _stamp(Path(self._py_file))


class _PackageCacheImpl(CompileResultCacheImpl):
    """Numba's compile-result cache, finding its files through the stamped locators.

    Numba's own chain for a kernel in a source file, in its order; without the
    user-wide fallback, a read-only tree with ``NUMBA_CACHE_DIR`` unset could not
    import a kernel module at all.
    """

    _locator_classes = (
        _UserProvidedPackageLocator,
        _InTreePackageLocator,
        _UserWidePackageLocator,
    )


class _PackageCache(FunctionCache):
    """Numba's function cache over :class:`_PackageCacheImpl`."""

    _impl_class = _PackageCacheImpl


# The clamps compile through ``jit`` as the module imports, so they follow
# everything ``jit`` reaches then: moved above the cache classes, the import
# raises a NameError.
@jit
def clampi_numba(value: int, low: int, high: int) -> int:
    """Return ``value`` clamped to ``[low, high]``, as ``clampi`` in ``craftax.h``."""
    if value < low:
        return low
    if value > high:
        return high
    return value


@jit
def clampf_numba(value: np.float32, low: np.float32, high: np.float32) -> np.float32:
    """Return ``value`` clamped to ``[low, high]``, as ``clampf`` in ``craftax.h``."""
    if value < low:
        return low
    if value > high:
        return high
    return value


unliteral = cast("Callable[[int], int]", np.int64)
"""Hand a module-level int to a kernel as a plain int64: ``np.int64``, typed as ``int``.

Numba types a module-level int as a literal of its value, and a kernel called
with a literal argument compiles a copy of itself for that value, which changes
what LLVM inlines around the call: the five ``_write_mob_obs_numba`` calls of the
observation stayed out of line (measured). So a constant that a kernel binds to
a local or passes to another kernel goes through this. ``int()`` gives an int64
too, but a local bound through it still changed world generation's machine code
(measured). Callable from kernels, where it costs nothing.
"""

cosf = cast("Callable[[np.float32], np.float32]", np.cos)
sinf = cast("Callable[[np.float32], np.float32]", np.sin)
sqrtf = cast("Callable[[np.float32], np.float32]", np.sqrt)
powf = cast("Callable[[np.float32, np.float32], np.float32]", np.power)
fmodf = cast("Callable[[np.float32, np.float32], np.float32]", np.fmod)
"""The ufuncs kernels apply to float32 scalars, typed as the C calls they lower to.

Each is the numpy ufunc itself, so a kernel compiles exactly as through
``np.cos``; numpy's stubs type a ufunc's scalar result as ``Any``.
"""


def _popcount_signature(
    typingctx: object,
    value: nbtypes.Type,
) -> tuple[Signature, Callable[..., Value]]:
    """Type :data:`popcount` as ``int64(uint64)``, lowered to LLVM's ``ctpop``."""
    del typingctx, value
    return nbtypes.int64(nbtypes.uint64), partial(
        _call_bit_intrinsic,
        "llvm.ctpop.i64",
        (),
    )


def _trailing_zeros_signature(
    typingctx: object,
    value: nbtypes.Type,
) -> tuple[Signature, Callable[..., Value]]:
    """Type :data:`trailing_zeros` as ``int64(uint64)``, lowered to LLVM's ``cttz``."""
    del typingctx, value
    return nbtypes.int64(nbtypes.uint64), partial(
        _call_bit_intrinsic,
        "llvm.cttz.i64",
        (ir.Constant(ir.IntType(1), 0),),
    )


popcount = cast("Callable[[np.uint64], int]", intrinsic(_popcount_signature))
"""Count the set bits of a ``uint64``, as ``__builtin_popcountll``, as an ``int64``.

Signed, like the C builtin's ``int``: Numba compares and unifies a ``uint64``
with an ``int64`` in float64. Callable from kernels only.
"""

trailing_zeros = cast(
    "Callable[[np.uint64], int]",
    intrinsic(_trailing_zeros_signature),
)
"""Count a ``uint64``'s zero bits below its lowest set bit, as ``__builtin_ctzll``.

Signed as :data:`popcount` is. Unlike the C builtin, a zero input is defined
and returns 64. Callable from kernels only.
"""


def _call_bit_intrinsic(  # noqa: PLR0917 -- Numba calls a codegen with these four positionally, after the two bound by partial.
    name: str,
    flags: Sequence[ir.Constant],
    context: BaseContext,
    builder: IRBuilder,
    signature: Signature,
    args: Sequence[Value],
) -> Value:
    """Emit a call of the ``i64 -> i64`` LLVM intrinsic ``name`` on ``args[0]``."""
    del context, signature
    word = ir.IntType(64)
    function = builder.module.declare_intrinsic(
        name,
        fnty=ir.FunctionType(word, [word, *(flag.type for flag in flags)]),
    )
    return builder.call(function, [args[0], *flags])


def _prefetch_signature(
    typingctx: object,
    record: nbtypes.Type,
    offset: nbtypes.Type,
) -> tuple[Signature, Callable[..., Value]] | None:
    """Type :data:`prefetch` as ``void(record, intp)``; any other first argument fails."""
    del typingctx, offset
    if not isinstance(record, nbtypes.Record):
        return None
    return nbtypes.void(record, nbtypes.intp), _emit_prefetch


prefetch = cast("Callable[[object, int], None]", intrinsic(_prefetch_signature))
"""Ask the cache for the line holding byte ``offset`` of a record, as ``__builtin_prefetch``.

A read, kept in every level (``__builtin_prefetch(p, 0, 3)``). It changes no
value, and an offset past the record faults nothing. Callable from kernels only.
"""


def _emit_prefetch(
    context: BaseContext,
    builder: IRBuilder,
    signature: Signature,
    args: Sequence[Value],
) -> Value:
    """Emit LLVM's ``prefetch`` of the record's data pointer plus the byte offset."""
    del signature
    byte = ir.IntType(8).as_pointer()
    word = ir.IntType(32)
    hint = builder.module.declare_intrinsic(
        "llvm.prefetch.p0",
        fnty=ir.FunctionType(ir.VoidType(), [byte, word, word, word]),
    )
    address = builder.gep(builder.bitcast(args[0], byte), [args[1]])
    builder.call(hint, [address, *(ir.Constant(word, flag) for flag in (0, 3, 1))])
    return context.get_dummy_value()
