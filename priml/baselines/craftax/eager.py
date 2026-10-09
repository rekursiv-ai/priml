"""Test support: the game's kernels run as plain Python, on tiny hand-built worlds.

A test that calls a compiled kernel pays for Numba twice. The first dispatch
in a process imports Numba's lowering registries and scipy.linalg: 0.3 s on
the x86 host, 0.7 s on the Mac, whether the kernel then compiles or loads from
the cache. A kernel the cache lacks then compiles, 45-68 s for the step. A
kernel's Python source costs neither, and on a few decisions of a hand-built
world it runs in milliseconds. So a unit test runs its kernels inside
:func:`eager`. While the block runs, every ``jit`` kernel of every loaded
module of this package, test modules included, is its ``py_func``.

Five things stand in for what Python cannot run as Numba does:

- Numba's ``error_model="numpy"`` and its wrapping integer arithmetic are
  numpy's with every floating-point error ignored, so a float divide by zero
  gives ``inf`` and a uint64 product wraps, as the compiled kernel does,
  where ``filterwarnings = error`` would raise. A thread's outermost call
  sets numpy's error state, and the kernels it calls run inside it: the
  state is the thread's own, and a rollout steps its buffers on threads of
  its own.
- A kernel reads a record's fields as attributes (``states[0].map``), which
  a numpy record allows only through a ``recarray`` view, and at 1 us a read:
  it builds the field's view anew each time, and a step reads hundreds of
  fields, a map scan ``state.map`` per tile. So an eager kernel is handed
  each structured array argument, and each one a NamedTuple argument holds,
  as a :class:`_Records`, and each record argument as a :class:`_Fields`,
  whose array fields, once read, are plain attributes and whose scalar
  fields are properties over one-element views, 0.1 us a read. A structured
  array keeps its :class:`_Records` for the block, so every kernel handed
  the State reads the views the first one made.
- Each intrinsic, which Python cannot call, has a Python stand-in:
  ``popcount``, ``trailing_zeros``, ``prefetch`` (nothing), the daylight
  table (each light computed on demand by ``daylight_numba``'s source, the
  table's own recipe), and ``step._training`` (the record itself).
- The two FNV-1a kernels hash with Python ints, remembered by content: the
  same integers, 4x faster than numpy's uint64 scalars over a State's 80,248
  bytes, and a State replay hashes again costs nothing more.
- ``jit.cosf``, ``sinf`` and ``powf``, numpy's ufuncs under the names the
  kernels bind, are the platform libm's float32 calls, which the compiled
  kernels make and numpy's own may round apart from: the daylight curve and
  world generation's noise come out as compiled.

So eager numbers are the compiled ones. With ``world``, world generation is
that function instead of the nine-floor recipe, which in Python is slow:
:func:`tiny_world` is such a function.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from functools import cache, lru_cache, partial
from typing import TYPE_CHECKING, Final, Protocol, SupportsFloat, SupportsIndex, cast
from unittest import mock

import ctypes
import operator
import sys
import threading

from numba.core.dispatcher import Dispatcher

import numpy as np

from priml.baselines.craftax.game import jit, rules, step
from priml.baselines.craftax.game.rules import GRAVE_BLOCKS, LAND_BLOCKS
from priml.baselines.craftax.game.state import (
    BOSS_SPAWN_TURNS,
    MAP_SIZE,
    NUM_LEVELS,
    NUM_POTIONS,
    Action,
    BlockType,
    ItemType,
    env_state,
    env_stats,
    new_stats,
)
from priml.baselines.craftax.game.step import Rules
from priml.baselines.craftax.game.world_gen import generate_world_numba
from priml.baselines.craftax.lib.arrays import ints, typed
from priml.baselines.craftax.world_model import replay
from priml.baselines.craftax.world_model.archive import Receipt, Record
from priml.baselines.craftax.world_model.capture import step as capture_step


if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Sequence

    from numpy.typing import NDArray

    import torch

    from priml.baselines.craftax.game.state import Array1, EnvState
else:
    from wrapt import lazy_import

    torch = lazy_import("torch")


_PACKAGE: Final = __name__.rpartition(".")[0]
"""``priml.baselines.craftax``: the modules whose kernels go eager."""

_SEEN: Final[dict[int, _Records]] = {}
"""Each structured array handed to a kernel in the block, by ``id``, as its view.

The view holds the array, so no other array takes its ``id`` before the block
ends and the table empties.
"""

LADDER_DOWN: Final = (MAP_SIZE // 2, MAP_SIZE // 2 + 2)
"""Where :func:`tiny_world` puts every floor's down ladder: two tiles right of the start."""

LADDER_UP: Final = (MAP_SIZE // 2, MAP_SIZE // 2 - 2)
"""Where :func:`tiny_world` puts every floor's up ladder: two tiles left of the start."""


_DAYLIGHT: Final = rules.daylight_numba.py_func
"""The light at a timestep, by the recipe the daylight table is computed with."""


@contextmanager
def eager(
    *,
    world: Callable[[EnvState, Array1[np.uint32]], None] | None = None,
) -> Generator[None]:
    """Run every kernel of this package as Python while the block runs.

    Args:
      world: Fills a zeroed State from its world stream, in place of world
        generation; None generates worlds by the recipe, in Python.

    Yields:
      None: The block runs eagerly; on exit every kernel is compiled again.

    """
    stand_ins = _stand_ins()
    if world is not None:
        stand_ins[id(generate_world_numba)] = _generator(world)
    with ExitStack() as stack:
        for name, module in list(sys.modules.items()):
            if name == __name__ or not name.startswith(_PACKAGE):
                continue
            members = cast("dict[str, object]", vars(module))
            for attribute, value in list(members.items()):
                stand_in = stand_ins.get(id(value))
                # ``isinstance`` would import the module a ``lazy_import`` proxy
                # stands for: an optional dependency this host may lack.
                if stand_in is None and issubclass(type(value), Dispatcher):
                    kernel = cast("Dispatcher[Callable[..., object]]", value)
                    stand_in = _recarrays(kernel.py_func)
                if stand_in is not None:
                    stack.enter_context(mock.patch.object(module, attribute, stand_in))
        stack.callback(_SEEN.clear)
        yield


def tiny_world(state: EnvState, rng: Array1[np.uint32], *, timestep: int = 0) -> None:
    """Fill a zeroed State with a world of grass, set up as the reset sets it up.

    The surface and the floor below it are grass, lit throughout, and the
    deeper floors are left zero (``INVALID``), so the State is mostly zero bytes, which the hash
    stand-in skips. Every floor has its down ladder at :data:`LADDER_DOWN` and
    its up ladder at :data:`LADDER_UP`, and the surface has a tree just above
    the start.
    The player, the creature slots and the potions are as
    ``world_gen._init_mobs_and_player_numba`` leaves them, without its draws:
    the potions keep their order. The spawn bitsets hold the map's land, so
    creatures spawn by the game's own rules. The stream, the world seed at a
    reset, puts a stone at the surface's ``(0, seed % 48)``, so worlds of
    different seeds hash apart.

    Args:
      state: A zeroed world record, filled in place.
      rng: The world's stream; not advanced.
      timestep: The clock, so an episode can start this close to the timeout.

    """
    blocks = typed(np.asarray(state.map), np.uint8)
    blocks[:2] = BlockType.GRASS
    typed(np.asarray(state.light_map), np.uint8)[:2] = 255
    blocks[0, MAP_SIZE // 2 - 1, MAP_SIZE // 2] = BlockType.TREE
    blocks[0, 0, int(rng[0]) % MAP_SIZE] = BlockType.STONE
    items = typed(np.asarray(state.item_map), np.uint8)
    items[(slice(None), *LADDER_DOWN)] = ItemType.LADDER_DOWN
    items[(slice(None), *LADDER_UP)] = ItemType.LADDER_UP
    typed(np.asarray(state.down_ladders), np.int32)[...] = LADDER_DOWN
    typed(np.asarray(state.up_ladders), np.int32)[...] = LADDER_UP
    for level in range(NUM_LEVELS):
        for mobs in (
            state.melee_mobs[level],
            state.passive_mobs[level],
            state.ranged_mobs[level],
            state.mob_projectiles[level],
            state.player_projectiles[level],
        ):
            mobs.health[:] = np.float32(1.0)
    typed(np.asarray(state.mob_projectile_dirs), np.int32)[...] = 1
    typed(np.asarray(state.player_projectile_directions), np.int32)[...] = 1
    typed(np.asarray(state.potion_mapping), np.int32)[...] = np.arange(NUM_POTIONS)
    typed(np.asarray(state.player_position), np.int32)[...] = MAP_SIZE // 2
    state.monsters_killed[0] = 10
    state.player_level = 0
    state.player_direction = Action.UP
    state.player_health = np.float32(9.0)
    state.player_food = state.player_drink = state.player_energy = 9
    state.player_mana = 9
    state.player_dexterity = state.player_strength = state.player_intelligence = 1
    state.boss_timestep_to_spawn_this_round = BOSS_SPAWN_TURNS
    state.timestep = timestep
    state.light_level = _DAYLIGHT(0)
    columns = np.uint64(1) << np.arange(MAP_SIZE, dtype=np.uint64)
    for bitset, table in (
        (state.spawn_land, LAND_BLOCKS),
        (state.spawn_grave, GRAVE_BLOCKS),
        (state.spawn_water, np.arange(len(LAND_BLOCKS)) == BlockType.WATER),
    ):
        typed(np.asarray(bitset), np.uint64)[...] = np.bitwise_or.reduce(
            np.where(table[blocks] != 0, columns, np.uint64(0)),
            axis=-1,
        )


def scripted(
    actions: Sequence[int],
    *,
    world_seed: int,
    sampling_seed: int = 0,
    truncated: bool = False,
) -> Record:
    """Play ``actions`` from the reset of ``world_seed``; return their record.

    Inside :func:`eager`, a record of a few chosen decisions of a tiny world.
    The hashes are the ones capture takes: before every 256th decision, and
    after the last.

    Args:
      actions: Each decision's action.
      world_seed: The world.
      sampling_seed: What the receipt names as the sampler's seed.
      truncated: Whether the record says its last decision is not terminal.

    Returns:
      record: As capture records it, for arm 0 of the training split.

    """
    states, rng = replay.reset_world(world_seed)
    stats = new_stats(1)
    hashes: list[int] = []
    for t, action in enumerate(actions):
        if t % replay.HASH_STRIDE == 0:
            hashes.append(int(replay.fnv1a_numba(states.view(np.uint8))))
        step.play_numba(env_state(states, 0), rng, env_stats(stats, 0), action, Rules())
    hashes.append(int(replay.fnv1a_numba(states.view(np.uint8))))
    return Record(
        receipt=Receipt(
            world_seed=world_seed,
            sampling_seed=sampling_seed,
            initial_state_hash=hashes[0],
            arm=0,
            split=0,
        ),
        actions=torch.tensor(actions, dtype=torch.uint8),
        hashes=torch.from_numpy(np.array(hashes, np.uint64).view(np.int64)),
        truncated=truncated,
    )


class _Daylight:
    """The daylight table as a kernel reads it, each entry computed when read."""

    shape: Final = (rules.DAYLIGHT_TIMESTEPS,)

    def __getitem__(self, timestep: int) -> np.float32:
        """Return the light at ``timestep``."""
        return _DAYLIGHT(int(timestep))


def _stand_ins() -> dict[int, Callable[..., object]]:
    """Return each intrinsic's and fast kernel's Python stand-in, by its object's id."""
    table = _Daylight()
    libm = _Float32Libm(_libm())
    return {
        id(np.cos): libm.cosf,
        id(np.sin): libm.sinf,
        id(np.power): libm.powf,
        id(jit.popcount): _popcount,
        id(jit.trailing_zeros): _trailing_zeros,
        id(jit.prefetch): _prefetch,
        id(rules.daylight_table): lambda: table,
        id(step._training): _record,  # noqa: SLF001 -- The intrinsic the step calls; its stand-in is the identity it compiles to.
        id(replay.fnv1a_numba): _fnv1a,
        id(capture_step.fnv1a_numba): _fnv1a,
    }


@cache
def _libm() -> _Libm:
    """Return the platform libm's ``cosf``, ``sinf`` and ``powf``, declared once."""
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
    """Count the set bits, as ``jit.popcount``."""
    return int(value).bit_count()


def _trailing_zeros(value: np.uint64) -> int:
    """Count the zeros below the lowest set bit, 64 for zero, as ``jit.trailing_zeros``."""
    word = int(value)
    return (word & -word).bit_length() - 1 if word else 64


def _prefetch(record: object, offset: int) -> None:
    """Do nothing, as ``jit.prefetch`` changes no value."""
    del record, offset


def _record[T](stats: T) -> T:
    """Return the record itself, as ``step._training`` compiles to."""
    return stats


def _fnv1a(data: NDArray[np.uint8]) -> np.uint64:
    """Return ``replay.fnv1a_numba``'s hash of ``data``, in Python ints."""
    return np.uint64(_fnv1a_bytes(data.tobytes()))


# A replay hashes one State several times over: the reset world before decision 0
# twice and the last State once more, then each later replay of the record again.
# A zero byte only multiplies by the prime, so a run of ``k`` of them multiplies by its
# ``k``-th power at once: a tiny world's State is mostly zeros.
@lru_cache(maxsize=64)
def _fnv1a_bytes(data: bytes) -> int:
    """Return the FNV-1a hash of ``data``, remembered for the same bytes."""
    prime, mask = 1_099_511_628_211, 0xFFFF_FFFF_FFFF_FFFF
    value, after = 1_469_598_103_934_665_603, 0
    for at in ints(np.flatnonzero(np.frombuffer(data, np.uint8))):
        value = (value * pow(prime, at - after, mask + 1) ^ data[at]) * prime & mask
        after = at + 1
    return value * pow(prime, len(data) - after, mask + 1) & mask


def _generator(
    world: Callable[[EnvState, Array1[np.uint32]], None],
) -> Callable[..., None]:
    """Return ``world`` called as world generation is, its level configs ignored."""

    def generate(state: EnvState, rng: Array1[np.uint32], *configs: object) -> None:
        del configs
        world(state, rng)

    return generate


class _Ignoring(threading.local):
    """Whether this thread runs inside an eager kernel, numpy's errors ignored."""

    inside = False


_IGNORING: Final = _Ignoring()
"""Each thread's own: whether its numpy errors are already ignored."""


def _recarrays(kernel: Callable[..., object]) -> Callable[..., object]:
    """Return ``kernel`` taking ``recarray`` views of its structured array arguments."""

    def call(*arguments: object) -> object:
        viewed = [a if type(a) in _PASSED else _viewed(a) for a in arguments]
        if _IGNORING.inside:
            return kernel(*viewed)
        # Entering numpy's error state takes as long as a short kernel: a step
        # calls thousands of them.
        _IGNORING.inside = True
        try:
            with np.errstate(all="ignore"):
                return kernel(*viewed)
        finally:
            _IGNORING.inside = False

    return call


class _Fields:
    """Record ``index`` of a structured array, its fields read and written as attributes.

    A scalar field is a property of the record's dtype (:func:`_fields_type`);
    any other field, once read, is a plain attribute holding its view: an
    array field's array, a record field's :class:`_Fields`, a field of records'
    :class:`_Records`.
    """

    def __init__(self, array: np.recarray, index: int) -> None:
        self._array = array
        self._index = index
        self._scalars: dict[str, np.ndarray] = {}

    def __getattr__(self, name: str) -> object:
        """Read the non-scalar field ``name``, and keep its view as an attribute."""
        kind = self._array.dtype[name]
        if kind.shape:
            whole = cast("np.ndarray", self._array[name][self._index])
            value: object = (
                whole if kind.base.names is None else _Records(whole.view(np.recarray))
            )
        else:
            one = self._array[name][self._index : self._index + 1]
            value = _fields_type(kind)(one.view(np.recarray), 0)
        object.__setattr__(self, name, value)
        return value

    def read_field(self, name: str) -> object:
        """Return scalar field ``name``'s value."""
        return cast("object", self.scalar(name)[0])

    def write_field(self, value: object, name: str) -> None:
        """Write scalar field ``name``."""
        self.scalar(name)[0] = value

    def scalar(self, name: str) -> np.ndarray:
        """Return scalar field ``name`` as a one-element view, made at its first use."""
        if name not in self._scalars:
            self._scalars[name] = self._array[name][self._index : self._index + 1]
        return self._scalars[name]

    def record(self) -> np.record:
        """Return the record as numpy reads it."""
        return cast("np.record", self._array[self._index])


@cache
def _fields_type(dtype: np.dtype[np.void]) -> type[_Fields]:
    """Return the :class:`_Fields` of ``dtype``: a property per scalar field."""
    fields = type(
        "_Fields",
        (_Fields,),
        {
            name: property(
                partial(_Fields.read_field, name=name),
                partial(_Fields.write_field, name=name),
            )
            for name in dtype.names or ()
            if not dtype[name].shape and dtype[name].names is None
        },
    )
    _PASSED.add(fields)
    return fields


def _fields_of(record: np.record) -> _Fields:
    """Return a record of a one-dimensional structured array as :class:`_Fields`."""
    base = cast("object", record.base)
    if not isinstance(base, np.recarray) or base.ndim != 1:
        raise TypeError("Expected a record of a one-dimensional recarray.")
    # The record's address: its ``__array_interface__`` builds the dtype's whole
    # description first, 4x slower.
    at = np.frombuffer(memoryview(record), np.uint8).ctypes.data
    index = (at - base.ctypes.data) // base.strides[0]
    return _fields_type(base.dtype)(base, index)


def _records_of(array: np.ndarray) -> _Records:
    """Return a structured array's :class:`_Records`, the one made first in the block."""
    records = _SEEN.get(id(array))
    if records is None:
        records = _SEEN[id(array)] = _Records(array.view(np.recarray))
    return records


class _Records:
    """A structured array whose records read as :class:`_Fields`, each made once.

    A kernel reads ``states[0]`` and passes it on to the kernels it calls, so
    one :class:`_Fields` per record keeps the views it has read for all of
    them. Everything else is the array's.
    """

    def __init__(self, array: np.recarray) -> None:
        self.array = array
        self.records: dict[int, _Fields] = {}

    def __getitem__(self, index: SupportsIndex | slice) -> object:
        """Return record ``index`` as its :class:`_Fields`; a slice as the array's."""
        if isinstance(index, slice):
            return cast("object", self.array[index])
        key = operator.index(index)
        if key not in self.records:
            self.records[key] = _fields_type(self.array.dtype)(self.array, key)
        return self.records[key]

    def __setitem__(self, index: int, value: object) -> None:
        """Write record ``index``, from a record or its :class:`_Fields`."""
        self.array[index] = value.record() if isinstance(value, _Fields) else value

    def __getattr__(self, name: str) -> object:
        """Return the array's attribute ``name``."""
        return cast("object", getattr(self.array, name))

    def __len__(self) -> int:
        return len(self.array)


_PASSED: Final[set[type]] = {_Records}
"""Argument types a kernel takes as they are, passed on at a glance.

The record views, and each type :func:`_viewed` has met that never holds a
record: most arguments, numbers and enums.
"""


def _viewed(value: object) -> object:
    """Return a structured array as :class:`_Records`, a record as :class:`_Fields`."""
    kind = type(value)
    if kind is np.record:
        return _fields_of(cast("np.record", value))
    if kind is np.ndarray or kind is np.recarray:
        array = cast("np.ndarray", value)
        if array.dtype.names is None:
            return value
        return _records_of(array)
    if not issubclass(kind, tuple):
        _PASSED.add(kind)
        return value
    members = cast("tuple[object, ...]", value)
    fields = [_viewed(member) for member in members]
    if all(a is b for a, b in zip(fields, members, strict=True)):
        return members
    return cast("Callable[..., object]", kind)(*fields)
