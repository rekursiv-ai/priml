"""Test support for the game: kernels run as Python, small worlds, and compiled code.

A unit test runs the kernels it reaches as their Python source
(:func:`eager_kernels`) on worlds built by hand (:func:`small_world`): a
compile costs seconds, the Python milliseconds on a world this small.

Some of ``jit``'s rules cannot be checked by running the game -- no fused
multiply-add in the machine code, no float64 in a float32 kernel -- so their
tests read what Numba compiled, and an intrinsic's tests read the IR its
codegen emits. Nothing at runtime inspects a kernel, so this lives beside the
tests rather than in ``jit``.
"""

from __future__ import annotations

from functools import cache, wraps
from typing import (
    TYPE_CHECKING,
    Final,
    Literal,
    Protocol,
    SupportsFloat,
    TypedDict,
    cast,
    override,
    runtime_checkable,
)

import ctypes
import re
import sys

from llvmlite import ir
from numba import njit
from numba.core.dispatcher import Dispatcher

import numpy as np

from priml.baselines.craftax.game import rules
from priml.baselines.craftax.game.rng import rand_r_numba
from priml.baselines.craftax.game.state import (
    BOSS_SPAWN_TURNS,
    MAP_SIZE,
    MAX_RANGED_MOBS,
    NUM_POTIONS,
    Action,
    BlockType,
    new_states,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from numpy.typing import NDArray

    import pytest

    from priml.baselines.craftax.game.state import (
        Array1,
        Array2,
        EnvState,
        Records,
    )


FMA_MNEMONIC: Final = re.compile(
    r"\bv?fn?m(add|sub)(132|213|231)?[sp][sd]\b|\bfn?m(add|sub|la|ls)\b",
)
"""x86 (``vfmadd231ss`` ...) and arm64 (``fmadd``, ``fmla`` ...) fused multiply-adds."""


def eager_kernels(
    monkeypatch: pytest.MonkeyPatch,
    **stand_ins: Callable[..., object],
) -> None:
    """Run every kernel of the port as its Python source until ``monkeypatch`` undoes it.

    Each ``jit`` dispatcher bound in a loaded module of the port becomes its
    Python function, which views each plain array it is handed (and each in
    a tuple, a ``Batch``'s) as a record array, so a record's fields read as
    attributes, as a kernel reads them, and its writes land in the caller's
    memory. Kernels call one another through those module globals, so a
    whole step runs as Python, as ``NUMBA_DISABLE_JIT=1`` would; that
    variable is read once per process, and the bit-for-bit goldens need the
    machine code. Each intrinsic, callable from kernels only, becomes a
    stand-in under the name modules bind it to: the game's here, and
    ``stand_ins``, which may also replace a kernel. ``cosf``, ``sinf`` and
    ``powf`` become the platform libm's float32 calls, which the compiled
    kernels make and numpy's own ufuncs may round apart from.

    A test's own ``monkeypatch.setattr`` of a kernel is undone first, then
    these, so a kernel a test replaced comes back compiled; a context manager
    restoring these before a ``monkeypatch`` set up ahead of it undid the
    test's would leave a test's Python kernel behind for the next test.

    Args:
      monkeypatch: The test's, which records every replacement.
      **stand_ins: Python stand-ins for the intrinsics outside ``game/``, or
        for kernels, by the name modules bind them to.

    """
    named: dict[str, Callable[..., object]] = {
        "popcount": _popcount,
        "trailing_zeros": _trailing_zeros,
        "prefetch": _prefetch,
        "daylight_table": _daylight_table,
        "_training": _identity,
        **stand_ins,
    }
    libm = _Float32Libm(_libm())
    replacements: dict[int, Callable[..., object]] = {
        id(np.cos): libm.cosf,
        id(np.sin): libm.sinf,
        id(np.power): libm.powf,
    }
    port = __name__.removesuffix(".game.testing")
    for name, module in tuple(sys.modules.items()):
        if name != port and not name.startswith(f"{port}."):
            continue
        for attribute, value in tuple(cast("dict[str, object]", vars(module)).items()):
            # ``type``, not ``isinstance``, which would import a module a
            # ``lazy_import`` proxy stands for: triton, say, absent on a Mac.
            if attribute in named:
                replacement = named[attribute]
            elif issubclass(type(value), Dispatcher):
                kernel = cast("Dispatcher[Callable[..., object]]", value)
                replacement = replacements.setdefault(
                    id(value),
                    _on_records(kernel.py_func),
                )
            elif id(value) in replacements:
                replacement = replacements[id(value)]
            else:
                continue
            monkeypatch.setattr(module, attribute, replacement)


def small_world(world: np.void, marker: int) -> None:
    """Build a world a step can play into a zeroed record, as world generation would.

    The surface is sand, lit everywhere, and the other floors are empty: sand
    is neither land, grave nor water, so nothing spawns and the spawn bitsets
    stay zero, as the map says. The player stands at the centre facing a
    tree, so a first ``DO`` earns an achievement, with world generation's
    meters, attributes and creature slots, and an unshuffled potion table.
    The marker, a stone on the top row at column ``marker``, tells the
    worlds of a pool apart.

    Args:
      world: A zeroed ``STATE_DTYPE`` record, filled in place.
      marker: Which world; below ``MAP_SIZE``.

    """
    centre = MAP_SIZE // 2
    world["map"][0] = BlockType.SAND
    world["map"][0, 0, marker] = BlockType.STONE
    world["map"][0, centre - 1, centre] = BlockType.TREE
    world["light_map"][0] = 255
    for creatures in ("melee_mobs", "passive_mobs", "mob_projectiles"):
        world[creatures]["health"] = np.float32(1.0)
    world["player_projectiles"]["health"] = np.float32(1.0)
    world["ranged_mobs"]["health"][:, :MAX_RANGED_MOBS] = np.float32(1.0)
    world["mob_projectile_dirs"] = 1
    world["player_projectile_directions"] = 1
    world["potion_mapping"] = np.arange(NUM_POTIONS)
    world["monsters_killed"][0] = 10
    world["player_position"] = centre
    world["player_direction"] = Action.UP
    world["player_health"] = 9.0
    for meter in ("player_food", "player_drink", "player_energy", "player_mana"):
        world[meter] = 9
    for attribute in ("player_dexterity", "player_strength", "player_intelligence"):
        world[attribute] = 1
    world["boss_timestep_to_spawn_this_round"] = BOSS_SPAWN_TURNS


def small_worlds(count: int) -> NDArray[np.void]:
    """Return ``count`` :func:`small_world` worlds, world ``k`` marked ``k``."""
    worlds = new_states(count)
    for marker in range(count):
        small_world(cast("np.void", worlds[marker]), marker)
    return worlds


def build_small_pool(
    pool: Records[EnvState],
    pool_bytes: Array2[int],
    smooth_configs: object,
    dungeon_configs: object,
    first_seed: int,
) -> None:
    """Stand in for ``build_pool_numba``: world ``k`` is :func:`small_world` marked by its seed.

    Args:
      pool: ``STATE_DTYPE [num_worlds]``, overwritten.
      pool_bytes: ``pool`` as bytes, zeroed first as the pool builder does.
      smooth_configs: Unread: no floor is generated.
      dungeon_configs: Unread.
      first_seed: The seed of world 0; world ``k`` is marked
        ``(first_seed + k) % MAP_SIZE``.

    """
    del smooth_configs, dungeon_configs
    for k in range(pool.shape[0]):
        pool_bytes[k, :] = 0
        small_world(cast("np.void", pool[k]), (first_seed + k) % MAP_SIZE)


def generate_small_world(
    state: EnvState,
    rng: Array1[np.uint32],
    smooth_configs: object,
    dungeon_configs: object,
) -> None:
    """Stand in for ``generate_world_numba``: one draw, which marks a :func:`small_world`.

    Args:
      state: A zeroed world record, filled in place.
      rng: The world's stream, advanced by one draw.
      smooth_configs: Unread: no floor is generated.
      dungeon_configs: Unread.

    """
    del smooth_configs, dungeon_configs
    small_world(cast("np.void", state), int(rand_r_numba(rng) % MAP_SIZE))


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


@cache
def _libm() -> _Libm:
    """Return the platform libm's ``cosf``, ``sinf`` and ``powf``, declared once per process."""
    # The process's own symbols: the libm the interpreter and the kernels link.
    library = ctypes.CDLL(None)
    for function in (library.cosf, library.sinf):
        function.restype = ctypes.c_float
        function.argtypes = [ctypes.c_float]
    library.powf.restype = ctypes.c_float
    library.powf.argtypes = [ctypes.c_float, ctypes.c_float]
    return cast("_Libm", library)


class _Libm(Protocol):
    """The libm calls the compiled kernels make, as :func:`_libm` declares them."""

    def cosf(self, x: SupportsFloat, /) -> float: ...
    def sinf(self, x: SupportsFloat, /) -> float: ...
    def powf(self, x: SupportsFloat, y: SupportsFloat, /) -> float: ...


class _Float32Libm:
    """:class:`_Libm`'s calls returning ``np.float32``, as a kernel's ``cosf`` does."""

    def __init__(self, libm: _Libm) -> None:
        self._libm = libm

    def cosf(self, x: np.float32) -> np.float32:
        """Return ``cosf(x)``."""
        return np.float32(self._libm.cosf(x))

    def sinf(self, x: np.float32) -> np.float32:
        """Return ``sinf(x)``."""
        return np.float32(self._libm.sinf(x))

    def powf(self, x: np.float32, y: np.float32) -> np.float32:
        """Return ``powf(x, y)``."""
        return np.float32(self._libm.powf(x, y))


def _popcount(value: np.uint64) -> int:
    """Stand in for ``jit.popcount``."""
    return int(value).bit_count()


def _trailing_zeros(value: np.uint64) -> int:
    """Stand in for ``jit.trailing_zeros``: 64 for zero, as the intrinsic returns."""
    word = int(value)
    return (word & -word).bit_length() - 1 if word else 64


def _prefetch(record: object, offset: int) -> None:
    """Stand in for ``jit.prefetch``, a cache hint that changes no value."""
    del record, offset


def _daylight_table() -> _DaylightTable:
    """Stand in for ``rules.daylight_table``."""
    return _DaylightTable()


class _DaylightTable:
    """``rules.daylight_table`` as the step reads it: its length, and a light per timestep."""

    shape: Final = (rules.DAYLIGHT_TIMESTEPS,)

    def __getitem__(self, timestep: int) -> np.float32:
        """Return the light at ``timestep``, from ``daylight_numba`` as it runs now."""
        return rules.daylight_numba(timestep)


def _identity[T](value: T) -> T:
    """Stand in for ``step._training``, the identity on a stats record."""
    return value


def _on_records(function: Callable[..., object]) -> Callable[..., object]:
    """Return ``function`` called with its plain arrays viewed as record arrays."""

    @wraps(function)
    def call(*args: object) -> object:
        return function(*map(_as_records, args))

    return call


def _as_records(value: object) -> object:
    """View a plain array, or each one a tuple holds, as a record array; wrap a record."""
    if type(value) is np.ndarray:
        return value.view(np.recarray)
    if type(value) is np.record:
        return _Fields(value)
    if not isinstance(value, tuple):
        return value
    items = cast("tuple[object, ...]", value)
    if not any(type(item) is np.ndarray for item in items):
        return items
    records = map(_as_records, items)
    return items._make(records) if isinstance(items, _NamedTuple) else tuple(records)


class _Fields:
    """A record whose array and record fields, views that never move, are read once.

    numpy builds a new view on every read of a record's field, which made a
    step as Python spend half its time there: a world's grids are read per
    tile. A scalar field is read and written through the record each time.
    """

    __slots__ = ("_record", "_views")

    def __init__(self, record: np.record) -> None:
        self._record = record
        self._views: dict[str, object] = {}

    def __getattr__(self, name: str) -> object:
        """Return the field ``name``: a cached view, or a scalar read now."""
        if name in self._views:
            return self._views[name]
        value = cast("object", getattr(self._record, name))
        if type(value) is np.record:
            value = _Fields(value)
        if isinstance(value, np.ndarray | _Fields):
            self._views[name] = value
        return value

    @override
    def __setattr__(self, name: str, value: object) -> None:
        """Write the field ``name`` in the record's memory; set the proxy's own slots."""
        if name in _Fields.__slots__:
            object.__setattr__(self, name, value)
        else:
            setattr(self._record, name, value)


@runtime_checkable
class _NamedTuple(Protocol):
    """A ``NamedTuple``'s constructor from its fields, which a ``Batch`` has."""

    def _make(self, fields: Iterable[object], /) -> object: ...


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
