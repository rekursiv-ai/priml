"""Tests for the golden regeneration flags."""

from __future__ import annotations

from typing import TYPE_CHECKING

from priml.testing import regenerate


if TYPE_CHECKING:
    import pytest


def test_flags_default_off_and_follow_the_command_line(
    pytestconfig: pytest.Config,
) -> None:
    assert regenerate.golden() is bool(pytestconfig.getoption("--regenerate-golden"))
    assert regenerate.b4b() is bool(pytestconfig.getoption("--regenerate-b4b"))


def test_overriding_one_flag_leaves_the_other(monkeypatch: pytest.MonkeyPatch) -> None:
    regenerate.override(monkeypatch, golden=False, b4b=False)
    regenerate.override(monkeypatch, golden=True)
    assert regenerate.golden()
    assert not regenerate.b4b()
    regenerate.override(monkeypatch, b4b=True)
    assert regenerate.golden()
    assert regenerate.b4b()


def test_suppressed_turns_both_flags_off_then_restores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    regenerate.override(monkeypatch, golden=True, b4b=True)
    with regenerate.suppressed():
        assert not regenerate.golden()
        assert not regenerate.b4b()
    assert regenerate.golden()
    assert regenerate.b4b()


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
