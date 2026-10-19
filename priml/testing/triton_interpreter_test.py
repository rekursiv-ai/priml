"""Tests for running Triton kernels through Triton's CPU interpreter."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import functools
import importlib
import os
import sys

from triton import language

import pytest
import torch
import triton

from priml.testing.triton_interpreter import (
    _is_submodule,
    _normalize_bare_constexpr,
    triton_interpreter,
)


if TYPE_CHECKING:
    from types import ModuleType


def test_a_kernel_runs_on_the_cpu_and_leaves_triton_as_it_was() -> None:
    before = {name: dict(vars(module)) for name, module in _language_modules()}
    tensor_before = dict(vars(language.tensor))
    values = torch.arange(6, dtype=torch.float32)
    total = torch.zeros(1)
    with triton_interpreter():
        assert os.environ["TRITON_INTERPRET"] == "1"
        triton.jit(_sum_kernel)[(1,)](values, total, 6, block=8)
    assert total.tolist() == [15.0]
    assert "TRITON_INTERPRET" not in os.environ
    after = {name: dict(vars(module)) for name, module in _language_modules()}
    assert _changed(before, after) == []
    assert (
        _changed({"tensor": tensor_before}, {"tensor": dict(vars(language.tensor))})
        == []
    )


def test_a_submodule_imported_inside_stays_bound_to_its_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extra = sys.modules["triton.language.extra"]
    monkeypatch.delattr(extra, "libdevice", raising=False)
    monkeypatch.delitem(sys.modules, "triton.language.extra.libdevice", raising=False)
    with triton_interpreter():
        importlib.import_module("triton.language.extra.libdevice")
    bound = cast("dict[str, object]", vars(extra))
    assert bound["libdevice"] is sys.modules["triton.language.extra.libdevice"]


def test_the_caller_interpret_setting_is_restored_on_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRITON_INTERPRET", "0")
    with triton_interpreter():
        assert os.environ["TRITON_INTERPRET"] == "1"
    assert os.environ["TRITON_INTERPRET"] == "0"


def test_a_module_bound_inside_that_is_no_submodule_is_dropped() -> None:
    standard = sys.modules["triton.language.standard"]
    with triton_interpreter():
        vars(standard)["leaked"] = os
    assert "leaked" not in vars(standard)


def test_the_named_caches_are_cleared_on_entry_and_on_exit() -> None:
    calls: list[int] = []

    @functools.cache
    def kernel() -> int:
        calls.append(len(calls))
        return len(calls)

    kernel()
    with triton_interpreter(kernel):
        kernel()
    kernel()
    assert calls == [0, 1, 2]


def test_an_interpreted_scalar_bounds_a_runtime_loop() -> None:
    values = torch.arange(6, dtype=torch.float32)
    out = torch.zeros(1)
    with triton_interpreter():
        triton.jit(_loop_kernel)[(1,)](values, out, torch.tensor([3]))
    assert out.tolist() == [3.0]


@pytest.mark.parametrize(
    ("annotation", "expected"),
    [
        ("'language.constexpr'", "constexpr"),
        ("language.constexpr", "constexpr"),
        ("'tensor'", "tensor"),
        ("'XX'", "XX"),
        (5, 5),
    ],
)
def test_a_postponed_constexpr_annotation_reads_as_bare(
    annotation: object,
    expected: object,
) -> None:
    assert _normalize_bare_constexpr(annotation, normalize=repr) == repr(expected)


@pytest.fixture(autouse=True)
def _no_interpret_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TRITON_INTERPRET", raising=False)


def _sum_kernel(
    values_ptr: language.tensor,
    total_ptr: language.tensor,
    count: int,
    block: language.constexpr,
) -> None:
    idx = language.arange(0, block)
    values = language.load(values_ptr + idx, mask=idx < count, other=0.0)
    language.store(total_ptr, language.sum(values, axis=0))


def _loop_kernel(
    values_ptr: language.tensor,
    out_ptr: language.tensor,
    stop_ptr: language.tensor,
) -> None:
    acc = language.load(values_ptr) * 0.0
    for index in range(language.load(stop_ptr)):
        acc += language.load(values_ptr + index)
    language.store(out_ptr, acc)


def _language_modules() -> list[tuple[str, ModuleType]]:
    return [
        (name, module)
        for name, module in sorted(sys.modules.items())
        if name.startswith("triton.language")
    ]


def _changed(
    before: dict[str, dict[str, object]],
    after: dict[str, dict[str, object]],
) -> list[str]:
    # A submodule imported on first use stays bound, as an import outside would.
    return [
        f"{name}.{key}"
        for name, old in before.items()
        for key in sorted(old.keys() | after[name].keys())
        if key != "__annotations_cache__"
        and old.get(key) is not after[name].get(key)
        and not (
            key not in old and _is_submodule(after[name][key], of=sys.modules[name])
        )
    ]


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
