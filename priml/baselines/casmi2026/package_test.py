"""The package layout forms an acyclic dependency graph between its modules."""

from __future__ import annotations

from pathlib import Path
from typing import Final, cast

import re


_CWD: Final = Path(__file__).resolve().parent
_PREFIX: Final = "priml.baselines.casmi2026"
_IMPORT: Final = re.compile(
    rf"^[ \t]*(?:from|import)[ \t]+{re.escape(_PREFIX)}\.([A-Za-z0-9_.]+)",
    re.MULTILINE,
)


def _unit(parts: list[str]) -> str:
    return parts[0]


def _graph() -> dict[str, set[str]]:
    graph: dict[str, set[str]] = {}
    for path in _CWD.rglob("*.py"):
        parts = list(path.relative_to(_CWD).with_suffix("").parts)
        if path.name.endswith("_test.py") or parts[0] in {"testing", "conftest"}:
            continue
        unit = _unit(parts)
        for target in cast("list[str]", _IMPORT.findall(path.read_text())):
            used = _unit(target.split("."))
            if used != unit:
                graph.setdefault(unit, set()).add(used)
    return graph


def _cycle(graph: dict[str, set[str]]) -> list[str]:
    done: set[str] = set()
    for unit in sorted(graph):
        if found := _visit(graph, unit, trail=[], done=done):
            return found
    return []


def _visit(
    graph: dict[str, set[str]],
    unit: str,
    *,
    trail: list[str],
    done: set[str],
) -> list[str]:
    if unit in trail:
        return [*trail[trail.index(unit) :], unit]
    if unit in done:
        return []
    for used in sorted(graph.get(unit, ())):
        if found := _visit(graph, used, trail=[*trail, unit], done=done):
            return found
    done.add(unit)
    return []


def test_subpackages_never_import_each_other_in_a_cycle() -> None:
    assert _cycle(_graph()) == []


def test_no_package_module_imports_the_offline_scripts() -> None:
    users = sorted(unit for unit, used in _graph().items() if "scripts" in used)
    assert users == []


def test_cycle_search_names_the_loop_it_finds() -> None:
    graph: dict[str, set[str]] = {"a": {"b"}, "b": {"c"}, "c": {"a", "d"}, "d": set()}
    assert _cycle(graph) == ["a", "b", "c", "a"]
    assert _cycle({"a": {"b"}, "b": set()}) == []


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
