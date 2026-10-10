"""Tests for the hooks that compile the kernels before a session's first test."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, cast

import pytest

from priml.baselines.craftax import conftest


_CWD: Final = Path(__file__).resolve().parent

_INTEGRATION: Final = _CWD / "world_model" / "codec_test.py"
"""A file of this package whose test the integration tier selects."""

_UNIT: Final = _CWD / "env_test.py"
"""A file of this package whose tests run in the unit and slow tiers."""

_ELSEWHERE: Final = _CWD.parents[2] / "testing" / "golden_test.py"
"""A test file outside this package."""


@dataclass(frozen=True, slots=True, kw_only=True)
class _Item:
    """What the hooks read of a collected test: its file and its marks."""

    path: Path
    marks: tuple[str, ...] = ()

    def get_closest_marker(self, name: str) -> pytest.Mark | None:
        """Return the mark ``name`` if the test carries it."""
        if name not in self.marks:
            return None
        return cast("pytest.MarkDecorator", getattr(pytest.mark, name)).mark


@dataclass(frozen=True, slots=True, kw_only=True)
class _PluginManager:
    """Whether xdist's controller plugin is registered."""

    controller: bool = False

    def has_plugin(self, name: str) -> bool:
        """Return whether plugin ``name`` is registered."""
        return self.controller and name == "dsession"


@dataclass(frozen=True, slots=True, kw_only=True)
class _Config:
    """The stash the hooks keep their state in, and the plugin manager."""

    stash: pytest.Stash = field(default_factory=pytest.Stash)
    pluginmanager: _PluginManager = field(default_factory=_PluginManager)


@dataclass(frozen=True, slots=True, kw_only=True)
class _Session:
    """A session as the hooks see it."""

    config: _Config = field(default_factory=_Config)
    items: tuple[_Item, ...] = ()


@pytest.fixture
def compiled(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    """Record each compile the hooks start, in place of the child process."""
    calls: list[object] = []
    monkeypatch.setattr(conftest, "_compile", calls.append)
    monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)
    return calls


@pytest.mark.parametrize(
    "items",
    [
        pytest.param((), id="no-test"),
        pytest.param(
            (_Item(path=_INTEGRATION, marks=("integration",)),) * 2,
            id="integration-tests",
        ),
        pytest.param((_Item(path=_UNIT, marks=("skip",)),), id="skipped-test"),
        pytest.param(
            (
                _Item(path=_INTEGRATION, marks=("integration",)),
                _Item(path=_ELSEWHERE),
            ),
            id="integration-test-beside-another-packages",
        ),
    ],
)
def test_a_session_whose_tests_call_no_kernel_compiles_nothing(
    compiled: list[object],
    items: tuple[_Item, ...],
) -> None:
    conftest.pytest_collection_finish(_hooked(_Session(items=items)))
    assert compiled == []


@pytest.mark.parametrize(
    "items",
    [
        pytest.param((_Item(path=_UNIT),), id="unit-test"),
        pytest.param((_Item(path=_UNIT, marks=("slow",)),), id="slow-test"),
        pytest.param(
            (
                _Item(path=_INTEGRATION, marks=("integration",)),
                _Item(path=_UNIT, marks=("skip",)),
                _Item(path=_UNIT),
            ),
            id="one-test-among-integration-and-skipped",
        ),
    ],
)
def test_a_session_with_a_test_that_may_call_a_kernel_compiles_once(
    compiled: list[object],
    items: tuple[_Item, ...],
) -> None:
    session = _Session(items=items)
    conftest.pytest_collection_finish(_hooked(session))
    assert compiled == [session.config]


def test_a_serial_run_that_loaded_the_file_at_startup_compiles_once_the_tests_are_known(
    compiled: list[object],
) -> None:
    integration = _Session(items=(_Item(path=_INTEGRATION, marks=("integration",)),))
    unit = _Session(items=(_Item(path=_UNIT),))
    for session in (integration, unit):
        conftest.pytest_sessionstart(_hooked(session))
        conftest.pytest_collection_finish(_hooked(session))
    assert compiled == [unit.config]


def test_an_xdist_controller_compiles_at_startup_for_its_workers(
    compiled: list[object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _Session(config=_Config(pluginmanager=_PluginManager(controller=True)))
    conftest.pytest_sessionstart(_hooked(controller))
    assert compiled == [controller.config]
    monkeypatch.setenv("PYTEST_XDIST_WORKER", "gw0")
    worker = _Session(items=(_Item(path=_UNIT),))
    conftest.pytest_sessionstart(_hooked(worker))
    conftest.pytest_collection_finish(_hooked(worker))
    assert compiled == [controller.config]


def _hooked(session: _Session) -> pytest.Session:
    """Hand ``session`` to a hook, which reads only what it carries."""
    return cast("pytest.Session", cast("object", session))


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
