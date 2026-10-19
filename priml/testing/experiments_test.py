"""Tests for the experiment-factory protocol."""

from __future__ import annotations

from typing import get_type_hints

from priml.testing import experiments


def exp000() -> int:
    return 0


def _named(factory: experiments.ExperimentFactory[int]) -> tuple[str, int]:
    return factory.__name__, factory()


def test_a_factory_function_satisfies_the_protocol() -> None:
    assert _named(exp000) == ("exp000", 0)


def test_the_factory_protocol_names_its_factory() -> None:
    assert get_type_hints(experiments.ExperimentFactory)["__name__"] is str


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
