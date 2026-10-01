"""Run ConvexTok's Numba kernels as Python in its tests, unless a test asks otherwise.

A JIT compile costs about 45 s per process on x86 with a cold cache, while the
same kernels run as Python take milliseconds on the test fixtures, so every test
here sees the Python by default. A test that needs the machine code -- that every
kernel compiles, or how Numba specializes one -- asks for it with
``@pytest.mark.parametrize("kernels", ["compiled"], indirect=True)``.
"""

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest

from priml.baselines.convextok.presolver import core
from priml.baselines.convextok.presolver.numba_api import Dispatcher


if TYPE_CHECKING:
    from collections.abc import Mapping


@pytest.fixture(autouse=True)
def kernels(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace each kernel in ``core`` with its Python source, unless ``compiled``.

    Kernels call one another through ``core``'s globals, so replacing every
    dispatcher there runs the whole presolve as Python, as ``NUMBA_DISABLE_JIT=1``
    would; that variable is process-wide, and other suites need the JIT. Numba's
    ``structref.new`` exists only compiled, so the state becomes an attribute bag.
    """
    if getattr(request, "param", "interpreted") == "compiled":
        return
    for name, value in cast("Mapping[str, object]", vars(core)).items():
        if isinstance(value, Dispatcher):
            monkeypatch.setattr(core, name, value.py_func)
    monkeypatch.setattr(core, "_new", _attribute_state)


def _attribute_state(struct_type: object) -> SimpleNamespace:
    """Stand in for ``structref.new``: a fresh attribute bag for the kernels to fill."""
    del struct_type
    return SimpleNamespace()
