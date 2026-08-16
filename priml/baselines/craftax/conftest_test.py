"""The parity-skip guard must track what the parity tests actually import.

``HAS_CRAFTAX`` decides whether every parity test in this directory runs or
skips. Probing the top-level ``craftax`` package answers a weaker question than
the tests ask: a half-removed install leaves the package directory in place
while its submodules are gone, so the guard says "present", the tests run, and
28 of them fail at ``ModuleNotFoundError`` instead of skipping. That is exactly
what a ``uv sync`` without the optional group leaves behind.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import importlib.util

import pytest

from priml.baselines.craftax.conftest import (
    _REFERENCE_PROBES,
    HAS_CRAFTAX,
    _reference_is_installed,
    requires_craftax,
)


if TYPE_CHECKING:
    from importlib.machinery import ModuleSpec


def test_guard_probes_every_reference_root_the_tests_import() -> None:
    """Both reference roots are probed, since the tests reach into both.

    ``constants``/``envs``/``world_gen`` live under ``craftax.craftax``, while
    ``world_gen_test`` reads ``craftax.craftax_classic``. A guard covering one
    root leaves the other free to be missing.
    """
    roots = {name.split(".")[1] for name in _REFERENCE_PROBES}

    assert roots == {"craftax", "craftax_classic"}, (
        f"the guard probes {sorted(roots)}; the parity tests import both "
        f"`craftax.craftax` and `craftax.craftax_classic`."
    )


def test_every_probe_is_a_submodule_not_the_bare_package() -> None:
    """A bare ``craftax`` probe passes on a half-removed install."""
    shallow = [name for name in _REFERENCE_PROBES if name.count(".") < 2]

    assert not shallow, (
        f"{shallow} probe the package root, which exists even when every "
        f"submodule has been removed. Probe a module the tests import."
    )


@pytest.mark.usefixtures("absent_craftax")
def test_guard_survives_a_wholly_absent_reference() -> None:
    """A missing parent raises from ``find_spec``; the guard must not.

    ``importlib.util.find_spec("craftax.craftax.constants")`` returns ``None``
    for a missing leaf but RAISES ``ModuleNotFoundError`` when an intermediate
    parent is absent -- the wholly-uninstalled case this guard exists for.
    """
    assert _reference_is_installed() is False


def test_guard_agrees_with_a_real_import() -> None:
    """Whatever the guard claims, the import it gates must match.

    ``find_spec`` RAISES rather than returning ``None`` when an intermediate
    parent is missing, which is the ordinary state of an export env that never
    installs the optional group -- so probing it here needs the same guard the
    module under test uses.
    """
    for name in _REFERENCE_PROBES:
        try:
            importable = importlib.util.find_spec(name) is not None
        except ModuleNotFoundError:
            importable = False
        assert importable == HAS_CRAFTAX, (
            f"guard says installed={HAS_CRAFTAX} but {name} importable={importable}"
        )


def test_the_marker_is_wired_to_the_guard() -> None:
    """The exported marker must be the skipif the guard computes."""
    assert requires_craftax.name == "skipif"
    assert requires_craftax.args == (not HAS_CRAFTAX,)


@pytest.fixture(name="absent_craftax")
def fixture_absent_craftax(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every ``craftax`` probe raise, as a full uninstall does."""
    real = importlib.util.find_spec

    def raising(name: str, package: str | None = None) -> ModuleSpec | None:
        if name.startswith("craftax"):
            raise ModuleNotFoundError(
                f"No module named {name.split('.', maxsplit=1)[0]!r}"
            )
        return real(name, package)

    monkeypatch.setattr(importlib.util, "find_spec", raising)


if __name__ == "__main__":
    from priml.lib.testing import test_main

    test_main(__file__)
