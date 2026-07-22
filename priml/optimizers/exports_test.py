"""Tests for optimizer package exports (LOSSOPT-010)."""

from __future__ import annotations

from priml import optimizers
from priml.optimizers import Muon, Newton


def test_muon_and_newton_are_exported() -> None:
    """Muon and Newton must be importable from the package root."""
    assert Muon.__name__ == "Muon"
    assert Newton.__name__ == "Newton"
    assert "Muon" in optimizers.__all__
    assert "Newton" in optimizers.__all__


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
