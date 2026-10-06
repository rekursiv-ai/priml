"""Tests for priml.compile lazy compilation decorators."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import logging

import pytest
import torch

from priml.compile import (
    _MAX_TRACES_PER_KEY,
    lazy_assume_constant_result,
    lazy_torch_compile,
    trace_compile,
)


if TYPE_CHECKING:
    from collections.abc import Callable


def _identity(fn: Callable[..., object]) -> Callable[..., object]:
    return fn


def _identity_compile(*_args: object, **_kwargs: object) -> Callable[..., object]:
    """Stand-in for torch.compile: returns a no-op decorator."""
    return _identity


class TestLazyTorchCompile:
    @pytest.mark.parametrize("compile_args", [(1,), (1, 2)])
    def test_invalid_positional_arguments_are_rejected(
        self,
        compile_args: tuple[object, ...],
    ) -> None:
        with pytest.raises(TypeError) as raised:
            lazy_torch_compile(*compile_args)
        assert str(raised.value) == (
            "lazy_torch_compile accepts only keyword compile arguments."
        )

    def test_bare_decorator(self) -> None:
        # Bare @lazy_torch_compile must decorate the function, not bind it
        # as a torch.compile argument (issue CORE-006).
        with patch("priml.compile.torch.compile", _identity_compile):

            @lazy_torch_compile
            def f(x: int) -> int:
                return x + 1

            assert f(1) == 2

    def test_empty_parameterized_decorator(self) -> None:
        with patch("priml.compile.torch.compile", _identity_compile):

            @lazy_torch_compile()
            def f(x: int) -> int:
                return x - 1

            assert f(3) == 2

    def test_parameterized_decorator(self) -> None:
        calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

        def _tracking_compile(
            *args: object,
            **kwargs: object,
        ) -> Callable[..., object]:
            calls.append((args, kwargs))
            return _identity

        with patch("priml.compile.torch.compile", _tracking_compile):

            @lazy_torch_compile(fullgraph=True)
            def f(x: int) -> int:
                return x * 2

            assert f(3) == 6

        assert calls == [((), {"fullgraph": True})]

    def test_compile_deferred_to_first_call(self) -> None:
        calls: list[int] = []

        def _tracking_compile(
            *_args: object,
            **_kwargs: object,
        ) -> Callable[..., object]:
            calls.append(1)
            return _identity

        with patch("priml.compile.torch.compile", _tracking_compile):

            @lazy_torch_compile
            def f(x: int) -> int:
                return x

            assert calls == []  # Not compiled at decoration time.
            assert f(5) == 5
            assert len(calls) == 1
            f(6)
            assert len(calls) == 1  # Compiled exactly once.


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


@pytest.fixture
def isolated_traces(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[str]]:
    """Give each test its own recompile ledger and keep dynamo unimported."""
    traces: dict[str, list[str]] = {}
    monkeypatch.setattr("priml.compile._compile_traces", traces)
    monkeypatch.setattr("priml.compile._compile_counts", {})
    monkeypatch.setattr(torch.compiler, "assume_constant_result", _identity)
    return traces


def test_trace_compile_counts_each_call_per_key(
    isolated_traces: dict[str, list[str]],
) -> None:
    assert trace_compile("site_a") == 1
    assert trace_compile("site_a") == 2
    assert trace_compile("site_b") == 1
    assert set(isolated_traces) == {"site_a", "site_b"}
    # The recorded stack ends at the caller; trace_compile's own frame is dropped.
    assert "format_stack" not in isolated_traces["site_a"][0]
    assert "test_trace_compile_counts_each_call_per_key" in isolated_traces["site_a"][0]


def test_trace_compile_raises_past_max_compiles_with_the_stacks(
    isolated_traces: dict[str, list[str]],
) -> None:
    assert trace_compile("site", max_compiles=1) == 1
    with pytest.raises(
        RuntimeError,
        match=r"Too many compiles \(2\) for site\.",
    ) as raised:
        trace_compile("site", max_compiles=1)
    assert "--------" in str(raised.value)
    assert len(isolated_traces["site"]) == 2


def test_trace_compile_always_log_logs_the_stack_and_omits_it_from_the_error(
    isolated_traces: dict[str, list[str]],
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger="priml.compile")
    trace_compile("site", max_compiles=1, always_log=True)
    with pytest.raises(RuntimeError) as raised:
        trace_compile("site", max_compiles=1, always_log=True)
    assert "--------" not in str(raised.value)
    assert capsys.readouterr().out == ""
    assert caplog.text.count("test_trace_compile_always_log") == 2
    assert len(isolated_traces["site"]) == 2


def test_trace_compile_keeps_a_bounded_history_per_key(
    isolated_traces: dict[str, list[str]],
) -> None:
    counts = [trace_compile("site") for _ in range(_MAX_TRACES_PER_KEY + 3)]
    assert counts[-1] == _MAX_TRACES_PER_KEY + 3
    assert len(isolated_traces["site"]) == _MAX_TRACES_PER_KEY


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
