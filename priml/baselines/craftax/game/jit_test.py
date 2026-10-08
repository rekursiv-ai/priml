"""Tests that the Numba rules the port depends on hold, and C's clamps.

Three rules cannot be checked by running the game: no fused multiply-add in
the generated code, no float64 leaking into a float32 kernel, and no power
operator in ``game/`` (``x ** 3`` multiplies where C calls ``powf``). Each has
a test here so a violation fails before it changes bits.
"""

from __future__ import annotations

from ctypes.util import find_library
from importlib import util
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, cast

import ast
import ctypes
import math
import platform
import re

from llvmlite import ir
from numba.core import config
from numba.core.registry import cpu_target
from numba.np.numpy_support import from_dtype

import numba.core.types as nbtypes
import numpy as np
import pytest

from priml.baselines.craftax.game.jit import (
    _Caching,
    _emit_prefetch,
    _popcount_signature,
    _prefetch_signature,
    _trailing_zeros_signature,
    clampf_numba,
    clampi_numba,
    jit,
    jit_parallel,
    package_digest,
    platform_key,
    source_stamp,
    unliteral,
)
from priml.baselines.craftax.game.mobs import spawn_mobs_numba
from priml.baselines.craftax.game.state import NOISE_PI2
from priml.baselines.craftax.game.step import step_range_numba
from priml.baselines.craftax.game.testing import (
    FMA_MNEMONIC,
    ir_builder,
    kernel_assembly,
    kernel_llvm,
)


if TYPE_CHECKING:
    from collections.abc import Callable

    from numba.core.typing.templates import Signature

    from priml.baselines.craftax.game.state import Array1

_CWD: Final = Path(__file__).resolve().parent
_SLOTS: Final = 3

TRIG_ANCHORS: Final = {
    1: (0x3CC9D9B5, 0x3F7FEC1B, 0x3CC9D47A),
    64: (0x3FC9D9B5, 0xBBC9DA0A, 0x3F7FFEC2),
    128: (0x4049D9B5, 0xBF7FFB07, 0xBC49D90F),
    191: (0x4096996E, 0xBBC9D793, 0xBF7FFEC2),
    255: (0x40C90FDB, 0x3F800000, 0x343BBD2E),
}
"""Angle, ``cosf`` and ``sinf`` of 256 angles on ``[0, 2 pi]``, as fp32 bits, by index.

Read from glibc 2.39's libm and macOS's, which agree here; they round ``sinf``
apart at index 8.
"""


class _Twice(Protocol):
    """The module a cache-stamp test writes and loads from a path."""

    twice: Callable[[int], int]


class _AddOne(Protocol):
    """The module the read-only-tree test writes and loads from a path."""

    add_one: Callable[[int], int]


@jit
def _identity(x: int) -> int:
    return x


@jit_parallel
def _parallel_identity(x: int) -> int:
    return x


@jit
def _mad(a: np.float32, b: np.float32, c: np.float32) -> np.float32:
    return np.float32(a * b + c)


@jit
def _dot(x: Array1[np.float32], y: Array1[np.float32]) -> np.float32:
    total = np.float32(0.0)
    for i in range(x.shape[0]):
        total += x[i] * y[i]
    return np.float32(total)


@jit
def _cos_loop(x: Array1[np.float32], out: Array1[np.float32]) -> None:
    for i in range(x.shape[0]):
        out[i] = math.cos(x[i])


@jit
def _sin_loop(x: Array1[np.float32], out: Array1[np.float32]) -> None:
    for i in range(x.shape[0]):
        out[i] = math.sin(x[i])


@pytest.mark.compute_large_fixture
def test_the_decorator_pins_the_options_parity_depends_on() -> None:
    _identity(1)
    options = _identity.targetoptions
    assert options["nogil"] is True
    assert options["fastmath"] is False
    assert options["error_model"] == "numpy"
    assert options["boundscheck"] is False
    assert options.get("parallel", False) is False


def test_the_parallel_decorator_adds_prange_to_the_same_options() -> None:
    options = _parallel_identity.targetoptions
    assert options["parallel"] is True
    assert options["nogil"] is True
    assert options["fastmath"] is False
    assert options["error_model"] == "numpy"
    assert options["boundscheck"] is False


def test_game_kernels_use_no_power_operator() -> None:
    offenders = [
        f"{path.name}:{node.lineno}"
        for path in sorted(_CWD.glob("*.py"))
        # A file without the two characters holds no power node, so only those with
        # them are parsed and walked.
        if not path.name.endswith("_test.py")
        and "**" in path.read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, (ast.BinOp, ast.AugAssign)) and isinstance(node.op, ast.Pow)
    ]
    assert not offenders, offenders


@pytest.mark.compute_large_fixture
def test_float32_kernels_emit_no_fused_multiply_add() -> None:
    _mad(np.float32(1.5), np.float32(2.5), np.float32(0.5))
    _dot(np.ones(8, np.float32), np.ones(8, np.float32))
    for kernel in (_mad, _dot):
        assert not FMA_MNEMONIC.search(kernel_assembly(kernel)), kernel


@pytest.mark.compute_large_fixture
def test_float32_kernels_never_widen_to_float64() -> None:
    _mad(np.float32(1.5), np.float32(2.5), np.float32(0.5))
    _dot(np.ones(8, np.float32), np.ones(8, np.float32))
    for kernel in (_mad, _dot):
        body = kernel_llvm(kernel)
        assert "define" in body
        assert "double" not in body, kernel
        assert "fpext" not in body, kernel


def test_the_package_digest_changes_with_any_non_test_source(tmp_path: Path) -> None:
    for name in ("mobs.py", "step.py", "mobs_test.py"):
        (tmp_path / name).write_text(f"# {name}\n")
    before = package_digest(tmp_path)
    (tmp_path / "mobs_test.py").write_text("# edited\n")
    assert package_digest(tmp_path) == before
    (tmp_path / "mobs.py").write_text("# edited\n")
    assert package_digest(tmp_path) != before


def test_the_source_stamp_covers_game_the_kernels_file_and_the_libm(
    tmp_path: Path,
) -> None:
    for name in ("kernels.py", "experiments.py", "case_test.py"):
        (tmp_path / name).write_text(f"# {name}\n")
    kernels = tmp_path / "kernels.py"
    in_test = tmp_path / "case_test.py"
    before = source_stamp(kernels), source_stamp(in_test)
    assert before[0].startswith(f"{package_digest(_CWD)}-")
    assert before[0].endswith(f"-{platform_key()}")
    # A file beside the kernels', which none of them calls, recompiles nothing.
    (tmp_path / "experiments.py").write_text("# edited\n")
    assert (source_stamp(kernels), source_stamp(in_test)) == before
    kernels.write_text("# edited\n")
    assert source_stamp(kernels) != before[0]
    in_test.write_text("# edited\n")
    assert source_stamp(in_test) != before[1]


def test_a_kernel_outside_game_calls_no_kernel_of_another_file_outside_game() -> None:
    """Its stamp covers ``game/`` and its own file, so such a callee could go stale."""
    modules = _kernel_modules_outside_game()
    kernels = {
        module: {kernel.name for kernel in _kernels(tree)}
        for module, tree in modules.items()
    }
    offenders: list[str] = []
    for module, tree in modules.items():
        imported = _imported_names(tree)
        for kernel in _kernels(tree):
            offenders += [
                f"{module}.{kernel.name} calls {imported[node.func.id]}.{node.func.id}"
                for node in ast.walk(kernel)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in kernels.get(imported.get(node.func.id, ""), set())
            ]
    assert offenders == []


def test_every_kernel_is_cached_under_its_source_stamp(tmp_path: Path) -> None:
    # A caller's machine code embeds its callees, so a caller's cache must go
    # stale when any source it may call does, not only its own file.
    module = tmp_path / "twice.py"
    module.write_text(
        "from priml.baselines.craftax.game.jit import jit\n\n\n"
        "@jit\ndef twice(x: int) -> int:\n    return 2 * x\n",
    )
    spec = util.spec_from_file_location("twice", module)
    assert spec is not None
    assert spec.loader is not None
    loaded = util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    twice = cast("_Twice", loaded).twice
    kernels: list[tuple[object, Path]] = [(_identity, _CWD / "jit_test.py")]
    kernels += [(_parallel_identity, _CWD / "jit_test.py")]
    kernels += [(step_range_numba, _CWD / "step.py")]
    kernels += [(spawn_mobs_numba, _CWD / "mobs.py"), (twice, module)]
    for kernel, source in kernels:
        assert isinstance(kernel, _Caching)
        stamp = kernel._cache._impl.locator.get_source_stamp()
        assert stamp == source_stamp(source), source


def test_a_kernel_in_a_read_only_tree_caches_in_the_user_wide_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Without NUMBA_CACHE_DIR and with an unwritable __pycache__, Numba's own
    # chain falls back to its user-wide cache; importing the module raised here.
    monkeypatch.setattr(config, "CACHE_DIR", "")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    tree = tmp_path / "tree"
    tree.mkdir()
    module = tree / "read_only.py"
    module.write_text(
        "from priml.baselines.craftax.game.jit import jit\n\n\n"
        "@jit\ndef add_one(x: int) -> int:\n    return x + 1\n",
    )
    tree.chmod(0o555)
    try:
        spec = util.spec_from_file_location("read_only", module)
        assert spec is not None
        assert spec.loader is not None
        loaded = util.module_from_spec(spec)
        spec.loader.exec_module(loaded)
        add_one = cast("_AddOne", loaded).add_one
        assert add_one(1) == 2
        assert isinstance(add_one, _Caching)
        locator = add_one._cache._impl.locator
        assert Path(locator.get_cache_path()).is_relative_to(tmp_path)
        assert locator.get_source_stamp() == source_stamp(module)
    finally:
        tree.chmod(0o755)


def test_platform_key_names_the_libc() -> None:
    key = platform_key()
    assert key.startswith(f"{platform.system().lower()}-{platform.machine()}-")
    libc, version = platform.libc_ver()
    if libc:
        assert key.endswith(f"{libc}{version}")


def test_stamping_kernel_files_reads_the_libc_version_at_most_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # platform.libc_ver() scans the interpreter binary (30 ms on the Mac); once
    # per kernel file it cost importing game.step 385 ms.
    calls: list[tuple[object, ...]] = []
    libc_ver = platform.libc_ver

    def counted(*args: object) -> tuple[str, str]:
        calls.append(args)
        return libc_ver()

    monkeypatch.setattr(platform, "libc_ver", counted)
    for name in ("kernels.py", "more_kernels.py"):
        (tmp_path / name).write_text(f"# {name}\n")
        source_stamp(tmp_path / name)
    assert len(calls) <= 1


@pytest.mark.compute_large_fixture
def test_a_vectorizable_cos_loop_matches_the_platform_libm() -> None:
    # On the Xeon this fails when SVML is enabled (NUMBA_DISABLE_INTEL_SVML
    # unset): LLVM vectorizes the loop through SVML's cos, whose bits differ
    # from glibc's cosf, which the C calls.
    libm = ctypes.CDLL(find_library("m"))
    libm.cosf.restype = ctypes.c_float
    libm.cosf.argtypes = [ctypes.c_float]
    x = np.linspace(0.0, 6.2831854, 1 << 16, dtype=np.float32)
    out = np.empty_like(x)
    _cos_loop(x, out)
    expected = np.fromiter((libm.cosf(float(v)) for v in x), np.float32, len(x))
    assert np.array_equal(out.view(np.uint32), expected.view(np.uint32))


@pytest.mark.compute_large_fixture
def test_cos_and_sin_match_the_platform_libm() -> None:
    """Every angle against this libm, live; the anchors against glibc's and macOS's."""
    x = np.linspace(0.0, float(NOISE_PI2), 256, dtype=np.float32)
    libm = ctypes.CDLL(find_library("m"))
    libm.cosf.restype = ctypes.c_float
    libm.cosf.argtypes = [ctypes.c_float]
    libm.sinf.restype = ctypes.c_float
    libm.sinf.argtypes = [ctypes.c_float]
    cos_expected = np.fromiter(
        (libm.cosf(x.item(i)) for i in range(len(x))),
        np.float32,
        len(x),
    )
    sin_expected = np.fromiter(
        (libm.sinf(x.item(i)) for i in range(len(x))),
        np.float32,
        len(x),
    )
    cos_actual = np.empty_like(x)
    sin_actual = np.empty_like(x)
    _cos_loop(x, cos_actual)
    _sin_loop(x, sin_actual)
    assert np.array_equal(cos_actual.view(np.uint32), cos_expected.view(np.uint32))
    assert np.array_equal(sin_actual.view(np.uint32), sin_expected.view(np.uint32))
    columns = [values.view(np.uint32) for values in (x, cos_actual, sin_actual)]
    assert {
        index: tuple(column.item(index) for column in columns) for index in TRIG_ANCHORS
    } == TRIG_ANCHORS


@pytest.mark.compute_large_fixture
def test_clamps_check_low_before_high_like_c() -> None:
    assert clampi_numba(5, 0, 3) == 3
    assert clampi_numba(-1, 0, 3) == 0
    assert clampi_numba(2, 0, 3) == 2
    # With low > high, C returns low for a value below low, else high.
    assert clampi_numba(1, 4, 2) == 4
    assert clampi_numba(9, 4, 2) == 2
    assert clampf_numba(
        np.float32(1.5),
        np.float32(0.0),
        np.float32(1.0),
    ) == np.float32(1.0)
    assert clampf_numba(
        np.float32(-0.5),
        np.float32(0.0),
        np.float32(1.0),
    ) == np.float32(0.0)
    assert clampf_numba(
        np.float32(0.25),
        np.float32(0.0),
        np.float32(1.0),
    ) == np.float32(
        0.25,
    )


@pytest.mark.compute_large_fixture
def test_clampf_keeps_float32() -> None:
    result = clampf_numba(np.float32(0.1), np.float32(0.0), np.float32(1.0))
    assert isinstance(result, float)
    assert result == float(np.float32(0.1))
    assert "double" not in kernel_llvm(clampf_numba)


@jit
def _successor(value: int) -> int:
    return value + 1


@jit
def _bare_successor() -> int:
    return _successor(_SLOTS)


@jit
def _unliteral_successor() -> int:
    return _successor(unliteral(_SLOTS))


@pytest.mark.compute_large_fixture
def test_unliteral_hands_a_module_constant_on_as_a_plain_int64() -> None:
    # The bare call is the control: Numba compiles the callee for the literal.
    assert _bare_successor() == _unliteral_successor() == 4
    kernel_llvm(_bare_successor)
    kernel_llvm(_unliteral_successor)
    assert {str(signature[0]) for signature in _successor.signatures} == {
        "Literal[int](3)",
        "int64",
    }


@pytest.mark.parametrize(
    ("typing", "emitted"),
    [
        (_popcount_signature, 'call i64 @"llvm.ctpop.i64"(i64 %".1")'),
        (_trailing_zeros_signature, 'call i64 @"llvm.cttz.i64"(i64 %".1", i1 0)'),
    ],
    ids=["popcount", "trailing_zeros"],
)
def test_each_bit_count_is_one_llvm_intrinsic_on_a_word(
    typing: Callable[[object, nbtypes.Type], tuple[Signature, Callable[..., ir.Value]]],
    emitted: str,
) -> None:
    signature, emit = typing(None, nbtypes.uint64)
    assert signature == nbtypes.int64(nbtypes.uint64)
    builder, args = ir_builder(ir.IntType(64))
    emit(cpu_target.target_context, builder, signature, args)
    assert emitted in str(builder.module)


def test_a_prefetch_reads_a_records_line_and_types_no_other_argument() -> None:
    record = from_dtype(np.dtype([("x", np.int64)]))
    signature = _prefetch_signature(None, record, nbtypes.intp)
    assert signature == (nbtypes.void(record, nbtypes.intp), _emit_prefetch)
    assert _prefetch_signature(None, nbtypes.int64, nbtypes.intp) is None
    builder, args = ir_builder(ir.IntType(64).as_pointer(), ir.IntType(64))
    _emit_prefetch(cpu_target.target_context, builder, nbtypes.void(), args)
    # A read (0), kept in every level (3), of data (1): ``__builtin_prefetch(p, 0, 3)``.
    prefetched = (
        r'call void @"llvm\.prefetch\.p0"\(i8\* %"\.\d+", i32 0, i32 3, i32 1\)'
    )
    assert re.search(prefetched, str(builder.module))


def _kernel_modules_outside_game() -> dict[str, ast.Module]:
    """Parse every module of the port outside ``game/`` that defines a kernel."""
    root = _CWD.parent
    modules: dict[str, ast.Module] = {}
    for path in sorted(root.rglob("*.py")):
        parts = path.relative_to(root).with_suffix("").parts
        if parts[0] == "game":
            continue
        text = path.read_text(encoding="utf-8")
        if "@jit" in text:
            modules[".".join(("priml.baselines.craftax", *parts))] = ast.parse(
                text,
            )
    return modules


def _kernels(tree: ast.Module) -> list[ast.FunctionDef]:
    """Return a module's top-level functions decorated ``@jit`` or ``@jit_parallel``."""
    return [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and any(
            isinstance(decorator, ast.Name) and decorator.id in {"jit", "jit_parallel"}
            for decorator in node.decorator_list
        )
    ]


def _imported_names(tree: ast.Module) -> dict[str, str]:
    """Map each name a module imports through ``from ... import`` to its module."""
    return {
        alias.asname or alias.name: node.module
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module is not None
        for alias in node.names
    }


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
