"""Tests for priml.compile lazy compilation decorators."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from unittest.mock import patch

from priml.compile import (
    lazy_assume_constant_result,
    lazy_torch_compile,
)


def _identity(fn: Callable[..., Any]) -> Callable[..., Any]:
    return fn


def _identity_compile(*_args: Any, **_kwargs: Any) -> Callable[..., Any]:
    """Stand-in for torch.compile: returns a no-op decorator."""
    return _identity


class TestLazyTorchCompile:
    def test_bare_decorator(self) -> None:
        # Bare @lazy_torch_compile must decorate the function, not bind it
        # as a torch.compile argument (issue CORE-006).
        with patch("priml.compile.torch.compile", _identity_compile):

            @lazy_torch_compile
            def f(x: int) -> int:
                return x + 1

            assert f(1) == 2

    def test_parameterized_decorator(self) -> None:
        with patch("priml.compile.torch.compile", _identity_compile):

            @lazy_torch_compile(fullgraph=True)
            def f(x: int) -> int:
                return x * 2

            assert f(3) == 6

    def test_compile_deferred_to_first_call(self) -> None:
        calls: list[int] = []

        def _tracking_compile(*_args: Any, **_kwargs: Any) -> Callable[..., Any]:
            calls.append(1)
            return _identity

        with patch("priml.compile.torch.compile", _tracking_compile):

            @lazy_torch_compile
            def f(x: int) -> int:
                return x

            assert calls == []  # not compiled at decoration time
            assert f(5) == 5
            assert len(calls) == 1
            f(6)
            assert len(calls) == 1  # compiled exactly once


class TestLazyAssumeConstantResult:
    def test_defers_and_calls(self) -> None:
        with patch(
            "priml.compile.torch.compiler.assume_constant_result",
            _identity,
        ):

            @lazy_assume_constant_result
            def f(x: int) -> int:
                return x + 10

            assert f(1) == 11


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
