"""Tests for the per-pass training loader."""

from __future__ import annotations

from typing import TYPE_CHECKING

from priml.data.passes import Passes


if TYPE_CHECKING:
    from collections.abc import Iterator


def _counting(calls: list[int]) -> Iterator[int]:
    calls.append(len(calls))
    yield from range(3)


def test_every_iteration_draws_a_fresh_pass() -> None:
    calls: list[int] = []
    loader = Passes(draw=lambda: _counting(calls))
    assert list(loader) == list(loader) == [0, 1, 2]
    assert calls == [0, 1]


def test_a_pass_is_not_its_own_loader() -> None:
    loader = Passes(draw=lambda: iter(range(2)))
    assert iter(loader) is not loader


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
