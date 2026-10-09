"""The port stands alone: it imports no other baseline and leaves no dead name.

Its parity checks live outside priml, beside their fixtures, and import the
port, never the reverse. So the port imports nothing but priml, configgle and
the shared modules priml ships, and no other priml baseline; and a public name
that only those checks read would be dead code here, where no test notices it,
so every public top-level name must be read inside the package -- by its own
module, another module or a test.

The checks are textual, as a grep is: a name counts as read wherever it appears
as a word in code, with comments and string literals removed, so two unrelated
names spelled alike hide each other. Parsing instead would be exact, but it
took 110-140 ms per test on an M-series Mac, over the unit tier's 100 ms.
"""

from __future__ import annotations

from collections import Counter
from functools import cache
from pathlib import Path
from typing import Final

import re


_CWD: Final = Path(__file__).resolve().parent

_PACKAGE: Final = "priml.baselines.craftax"

# Unrolled loops, each a run of one character class to its closing quote: the lazy
# ``[\s\S]*?`` and the per-character alternation they replace match the same.
_STRING: Final = (
    r'''(?:"""[^"]*(?:"(?!"")[^"]*)*"""|'''
    r"""'''[^']*(?:'(?!'')[^']*)*'''|"""
    r""""[^"\\\n]*(?:\\.[^"\\\n]*)*"|"""
    r"""'[^'\\\n]*(?:\\.[^'\\\n]*)*')"""
)

# Every match starts with ``#``, a quote or a prefix letter; the lookahead says so, and
# re then skips to those characters: 40 ms over the package rather than 100 on a Mac.
_PROSE: Final = re.compile(
    rf"(?=[#'\"rRbBuUfF])"
    rf"(?:(?P<fstring>(?<!\w)[rR]?[fF][rR]?{_STRING})|#[^\n]*|(?<!\w)[rRbBuU]*{_STRING})",
)
"""Comments and string literals, docstrings included; an f-string is its own group."""

_DEFINITION: Final = re.compile(
    r"^(?:(?:async\s+)?def|class)\s+(\w+)|^(\w+)\s*(?::[^\n=]*)?=(?!=)",
    re.MULTILINE,
)
"""A top-level ``def``, ``class`` or assignment: one that starts its line."""

_IMPORT: Final = re.compile(
    r"^\s*(?:from\s+([\w.]+)\s+import|import\s+([\w.]+))",
    re.MULTILINE,
)


def test_the_port_imports_only_what_priml_ships() -> None:
    """The port imports priml, configgle and priml's shared modules, no other baseline."""
    offenders = [
        f"{path.relative_to(_CWD)}: {module}"
        for path, code in _sources().items()
        for module in _imports(code)
        if (
            module.startswith("loop.")
            and module.split(".")[1] not in {"priml", "configgle", "lib"}
        )
        or (
            module.startswith("priml.baselines")
            and not f"{module}.".startswith(f"{_PACKAGE}.")
        )
    ]

    assert offenders == []


def test_every_public_name_is_read_inside_the_package() -> None:
    """A name only the parity checks read belongs with them; one nobody reads is dead."""
    sources = _sources()
    # A definition is one occurrence: a name read nowhere else occurs once in all.
    occurrences = Counter(_words(" ".join(sources.values())))
    unread = [
        f"{path.relative_to(_CWD)}: {name}"
        for path, code in sources.items()
        if not path.name.endswith("_test.py")
        for name in _public_names(code)
        # Pytest calls a conftest's hooks by name.
        if not (path.name == "conftest.py" and name.startswith("pytest_"))
        and occurrences[name] < 2
    ]

    assert unread == []


# Both tests read them; whichever runs first blanks them.
@cache
def _sources() -> dict[Path, str]:
    """Return every source of the package, as :func:`_code`."""
    return {
        path: _code(path.read_text(encoding="utf-8"))
        for path in sorted(_CWD.rglob("*.py"))
    }


def _code(source: str) -> str:
    """Blank ``source``'s comments and strings; an f-string stays, for its fields."""
    return _PROSE.sub(lambda match: match.group("fstring") or " ", source)


# Every character ``\w`` does not match becomes a space, which ``split`` drops: the
# runs ``re.findall(r"\w+", code)`` returns, in a fifth of its time. Blanked code is
# ASCII as a rule, so its table is ASCII's; other code's lists the characters it holds.
def _words(code: str) -> list[str]:
    """Return the runs of word characters in ``code``, in order."""
    chars = map(chr, range(128)) if code.isascii() else set(code)
    spaces = {ord(char): " " for char in chars if not (char.isalnum() or char == "_")}
    return code.translate(spaces).split()


def _imports(code: str) -> list[str]:
    """Return the module each import statement names."""
    return [match[1] or match[2] or "" for match in _IMPORT.finditer(code)]


def _public_names(code: str) -> list[str]:
    """Return the public names a module defines at top level."""
    return [
        name
        for match in _DEFINITION.finditer(code)
        for name in (match.group(1) or match.group(2),)
        if not name.startswith("_")
    ]


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
