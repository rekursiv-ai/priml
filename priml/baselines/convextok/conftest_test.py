"""Tests for the fixture that runs ConvexTok's kernels as Python."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest

from priml.baselines.convextok import conftest
from priml.baselines.convextok.presolver import core
from priml.baselines.convextok.presolver.numba_api import Dispatcher


if TYPE_CHECKING:
    from collections.abc import Mapping


# ``compiled`` keeps this test's own autouse ``kernels`` from swapping them first.
@pytest.mark.parametrize("kernels", ["compiled"], indirect=True)
@pytest.mark.parametrize(
    ("param", "python"),
    [("interpreted", True), ("compiled", False)],
)
def test_kernels_are_python_unless_a_test_asks_for_the_machine_code(
    param: str,
    python: bool,
) -> None:
    compiled = {
        name: value
        for name, value in cast("Mapping[str, object]", vars(core)).items()
        if isinstance(value, Dispatcher)
    }
    assert compiled
    request = SimpleNamespace(param=param)
    with pytest.MonkeyPatch.context() as patch:
        conftest.kernels._get_wrapped_function()(request, patch)
        assert all(
            (vars(core)[name] is kernel.py_func) is python
            for name, kernel in compiled.items()
        )
        assert (vars(core)["_new"] is conftest._attribute_state) is python


def test_the_state_stand_in_is_a_fresh_attribute_bag() -> None:
    first = conftest._attribute_state(object())
    first.x = 1
    assert vars(conftest._attribute_state(object())) == {}


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
