"""Tests for priml's package fixtures that no test reaches by requesting them."""

from __future__ import annotations

from typing import TYPE_CHECKING, Self, cast

import pytest

from priml import conftest
from priml.distributed import testing


if TYPE_CHECKING:
    from collections.abc import Generator
    from types import TracebackType

    from priml.distributed.testing import WarmPoolGetter


_LOG: list[str] = []
"""What each ``_Pool`` did, in order; cleared by the test."""


class _Pool:
    """A ``WorkerPool`` stand-in that records being entered and exited."""

    class Config:
        def __init__(self, *, mesh_dims: dict[str, int]) -> None:
            self.mesh_dims = mesh_dims

        def make(self) -> _Pool:
            return _Pool(self.mesh_dims)

    def __init__(self, mesh_dims: dict[str, int]) -> None:
        self.mesh_dims = mesh_dims

    def __enter__(self) -> Self:
        _LOG.append(f"enter {self.mesh_dims}")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        del exc_type, exc_val, exc_tb
        _LOG.append(f"exit {self.mesh_dims}")


def test_warm_pools_reuses_one_pool_per_mesh_and_closes_each_at_the_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(testing, "WorkerPool", _Pool)
    _LOG.clear()
    fixture = cast(
        "Generator[WarmPoolGetter]",
        conftest.warm_pools._get_wrapped_function()(),
    )

    get_pool = next(fixture)
    first = get_pool({"dp": 2})
    assert get_pool({"dp": 2}) is first
    other = get_pool({"dp": 1, "tp": 2})
    assert other is not first
    assert _LOG == ["enter {'dp': 2}", "enter {'dp': 1, 'tp': 2}"]

    with pytest.raises(StopIteration):
        next(fixture)
    assert _LOG[2:] == ["exit {'dp': 1, 'tp': 2}", "exit {'dp': 2}"]


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
