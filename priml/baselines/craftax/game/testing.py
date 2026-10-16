"""Test support for the game: small worlds, and the code a kernel compiles to.

A unit test runs the kernels it reaches as their Python source (``eager`` of
``craftax.eager``) on worlds built by hand (:func:`small_world`): a compile
costs seconds, the Python milliseconds on a world this small.

Some of ``jit``'s rules cannot be checked by running the game -- no fused
multiply-add in the machine code, no float64 in a float32 kernel -- so their
tests read what Numba compiled, and an intrinsic's tests read the IR its
codegen emits. Nothing at runtime inspects a kernel, so this lives beside the
tests rather than in ``jit``.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Final,
    Literal,
    TypedDict,
    cast,
)

import re

from llvmlite import ir
from numba import njit

import numpy as np

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

    from numba.core.dispatcher import Dispatcher
    from numpy.typing import NDArray

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
