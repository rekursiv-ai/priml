"""Every MLS-Bench docstring cross-reference names a module the referring module binds.

The ports import their shared modules under qualified names wherever two
tasks' modules share a leaf (``reference as vision_reference``), so a
reference written against an older binding resolves to nothing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import re


_THIS: Final = Path(__file__).resolve()
_CWD: Final = _THIS.parent

_ROOT: Final = _CWD.parents[1]

_REFERENCE: Final = re.compile(r":(?:func|data|class|meth|attr):`~?([A-Za-z_]\w*)\.\w")

_IMPORTED: Final = re.compile(
    r"^\s*(?:from \S+ )?import (\([^)]*\)|[^\n]*)",
    re.MULTILINE,
)

_EVERYWHERE: Final = frozenset({"priml", "torch", "np", "nn"})
"""Packages a reference may name by their full path or ecosystem alias."""


def dangling(source: str) -> list[str]:
    """Return each cross-reference whose first name ``source`` never binds.

    A name is bound by an import (``import x``, ``as x``, or ``x,`` in a
    parenthesized import), a definition, or an assignment or parameter.
    Parsing every module would cost a quarter of a second; text suffices.
    """
    imported = {
        name.split(" as ")[-1].strip()
        for match in _IMPORTED.finditer(source)
        for name in match.group(1).replace("(", "").replace(")", "").split(",")
    }
    heads = [match.group(1) for match in _REFERENCE.finditer(source)]
    return [
        head
        for head in heads
        if head not in _EVERYWHERE | imported
        and not re.search(
            rf"(?:\bdef|\bclass) {head}\b|^\s*{head}\b\s*[:=]",
            source,
            re.MULTILINE,
        )
    ]


def test_a_reference_to_an_unbound_module_is_found() -> None:
    source = (
        '"""See :func:`port.x`, :data:`~reference.SEED`, :func:`run.x`,\n'
        ':func:`upstream.y` and :data:`lone.z`."""\n'
        "from x import reference\n"
        "from y import (\n"
        "    upstream,\n"
        ")\n"
        "def run() -> None: ...\n"
        'text = "lone"\n'
    )
    assert dangling(source) == ["port", "lone"]


def test_every_mlsbench_docstring_reference_resolves() -> None:
    # This file's own cases name unbound modules on purpose.
    found = {
        str(path.relative_to(_ROOT)): heads
        for path in sorted(_ROOT.rglob("*.py"))
        if path != _THIS and (heads := dangling(path.read_text()))
    }
    assert found == {}


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
