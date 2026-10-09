"""Tests of the small worlds, and that the code a kernel compiles to can be inspected."""

from __future__ import annotations

from typing import TYPE_CHECKING

from llvmlite import ir

import numpy as np
import pytest

from priml.baselines.craftax.game import testing
from priml.baselines.craftax.game.jit import (
    jit,
    jit_parallel,
)
from priml.baselines.craftax.game.rules import LAND_BLOCKS
from priml.baselines.craftax.game.state import (
    MAP_SIZE,
    STATE_DTYPE,
    Action,
    BlockType,
)
from priml.baselines.craftax.game.testing import (
    _inspectable,
    _kernel_llvm,
    ir_builder,
    kernel_llvm,
    small_worlds,
)
from priml.baselines.craftax.lib.arrays import typed


if TYPE_CHECKING:
    from collections.abc import Callable

    from numba.core.dispatcher import Dispatcher


def test_small_worlds_are_playable_quiet_and_told_apart_by_their_marker() -> None:
    worlds = small_worlds(3)
    assert worlds.dtype == STATE_DTYPE
    centre = MAP_SIZE // 2
    maps = typed(worlds["map"], np.uint8)
    assert [maps.item(k, 0, 0, k) for k in range(3)] == [BlockType.STONE] * 3
    assert np.equal(maps[:, 0, centre, centre], BlockType.SAND).all()
    assert not LAND_BLOCKS[BlockType.SAND]
    assert not typed(worlds["spawn_land"], np.uint64).any()
    assert np.equal(typed(worlds["light_map"], np.uint8)[:, 0], 255).all()
    assert np.equal(typed(worlds["player_position"], np.int32), centre).all()
    assert np.equal(typed(worlds["player_direction"], np.int32), Action.UP).all()
    assert np.equal(typed(worlds["player_health"], np.float32), 9.0).all()
    ranged = typed(worlds["ranged_mobs"]["health"], np.float32)
    assert np.array_equal(ranged[:, 0], np.tile([1.0, 1.0, 0.0], (3, 1)))
    potions = typed(worlds["potion_mapping"], np.int32)
    assert np.array_equal(potions, np.tile(np.arange(6), (3, 1)))
    assert len({worlds[k : k + 1].tobytes() for k in range(3)}) == 3


@jit
def _identity(x: int) -> int:
    return x


@jit
def _mad(a: np.float32, b: np.float32, c: np.float32) -> np.float32:
    return np.float32(a * b + c)


@jit_parallel
def _plus_one(x: np.ndarray) -> np.ndarray:
    return x + np.float32(1.0)


@jit
def _never_called(x: int) -> int:
    return x


@pytest.mark.compute_large_fixture
def test_an_inspected_twin_compiles_with_its_kernels_options() -> None:
    # A twin built from a restated option list compiled a jit_parallel kernel
    # serially, so an inspection of it checked code that never runs.
    _identity(1)
    _plus_one(np.zeros(4, dtype=np.float32))
    for kernel in (_identity, _plus_one):
        twin = _inspectable(kernel)
        assert kernel.targetoptions.items() <= twin.targetoptions.items(), kernel


@pytest.mark.compute_large_fixture
def test_kernel_llvm_drops_the_python_wrappers_and_survives_the_cache() -> None:
    _mad(np.float32(1.5), np.float32(2.5), np.float32(0.5))
    body = kernel_llvm(_mad)
    assert "define" in body
    assert "cpython" not in body
    # The twin, not the possibly cached dispatcher: Numba returns nothing from
    # inspect_llvm on cached code, which is why kernel_llvm recompiles.
    assert "cpython" in "\n".join(_inspectable(_mad).inspect_llvm().values())


def test_kernel_llvm_judges_each_block_by_its_define_line_alone() -> None:
    # A kernel may call a helper whose name says cpython; only a wrapper's own
    # define line does. Every block after a dropped wrapper is still read.
    wrapper = (
        'define i8* @"cpython.__main__.kern"(i8* %py) {\nentry:\n  ret i8* null\n}'
    )
    kernel = (
        'define float @"_ZN8__main__4kern"(float %a) {\n'
        "entry:\n"
        "  call void @cpython_helper()\n"
        "  ret float %a\n"
        "}"
    )
    cfunc = 'define float @"cfunc._ZN8__main__4kern"(float %a) {\n  ret float %a\n}'
    later = 'define i32 @"_ZN8__main__5later"(i32 %b) {\n  ret i32 %b\n}'
    modules = [f"{wrapper}\n\n{kernel}\n", f"{cfunc}\n{later}\n"]
    assert _kernel_llvm(modules) == f"{kernel}\n{later}"


def test_kernel_inspection_reuses_one_twin_for_every_signature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @jit
    def identity(x: np.ndarray) -> np.ndarray:
        return x

    values = np.arange(4, dtype=np.float32)
    identity(values)
    identity(values[::2])
    twins: list[Dispatcher[Callable[..., object]]] = []
    inspectable = _inspectable

    def record_twin(
        dispatcher: Dispatcher[Callable[..., object]],
    ) -> Dispatcher[Callable[..., object]]:
        twin = inspectable(dispatcher)
        twins.append(twin)
        return twin

    monkeypatch.setattr(testing, "_inspectable", record_twin)
    assembly, llvm = testing.kernel_inspection(identity)

    assert len(twins) == 1
    twin = twins[0]
    assert twin.signatures == identity.signatures
    assert identity.targetoptions.items() <= twin.targetoptions.items()
    assembly_by_signature = twin.inspect_asm()
    llvm_by_signature = twin.inspect_llvm()
    assert len(assembly_by_signature) == len(twin.signatures)
    assert len(llvm_by_signature) == len(twin.signatures)
    assert all(assembly_by_signature.values())
    assert all(llvm_by_signature.values())
    assert assembly == "\n".join(assembly_by_signature.values())
    assert llvm.count("define ") >= len(twin.signatures)
    assert "cpython" not in llvm
    assert "cfunc" not in llvm


def test_inspecting_an_uncalled_kernel_fails_loudly() -> None:
    with pytest.raises(ValueError, match="call the kernel"):
        kernel_llvm(_never_called)


def test_an_ir_builder_writes_into_a_fresh_function_of_its_arguments() -> None:
    word, flag = ir.IntType(64), ir.IntType(1)
    builder, (first, second) = ir_builder(word, flag)
    assert (first.type, second.type) == (word, flag)
    builder.ret_void()
    assert 'define void @"kernel"(i64 %".1", i1 %".2")' in str(builder.module)


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
