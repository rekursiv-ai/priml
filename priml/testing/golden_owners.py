"""Check that every priml text golden has exactly the test that owns it.

A golden is a ``testdata/<name>.txt`` beside the ``*_test.py`` that reads or
asserts it. Two failures are possible: a test claims a golden that is not
there, or a golden lies in ``testdata/`` that no test claims. Both are local to
one directory -- a test claims goldens only in its own ``testdata/`` -- so the
package's test checks it directory by directory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import ast
import re


__all__ = ["golden_directories", "ownership"]


def golden_directories(package: Path) -> list[Path]:
    """Return every directory of ``package`` holding tests or goldens.

    Args:
      package: A package root.

    Returns:
      directories: Each directory with a ``*_test.py`` or a ``testdata/``, sorted.

    """
    return sorted(
        {test.parent for test in package.rglob("*_test.py")}
        | {data.parent for data in package.rglob("testdata")},
    )


def ownership(directory: Path) -> tuple[set[Path], set[Path]]:
    """Return one directory's claimed-but-missing and unclaimed goldens.

    Args:
      directory: A directory holding ``*_test.py`` files and their
        ``testdata/``.

    Returns:
      missing: pprint goldens a test claims that do not exist.
      unclaimed: Text goldens in ``testdata/`` no test claims.

    """
    pprint_claims: set[Path] = set()
    other_text_claims: set[Path] = set()
    testdata = directory / "testdata"
    for test_file in sorted(directory.glob("*_test.py")):
        source = test_file.read_text(
            encoding="utf-8",
        )  # pragma: no mutate -- this host's locale is UTF-8 too.
        if (
            re.search(r"_golden\(|\.txt|read_sites\(|assert_sites_reached\(", source)
            is None
        ):
            continue
        scan = _scan_tree(ast.parse(source, filename=str(test_file)))
        for function, call in scan.golden_calls:
            names = _literal_name_values(function, call)
            if not names:
                raise ValueError(
                    f"{test_file}:{call.lineno}: name= must resolve to literals",
                )
            if any(keyword.arg == "host" for keyword in call.keywords):
                other_text_claims.update(
                    path for name in names for path in testdata.glob(f"{name}_*.txt")
                )
                continue
            claims = (
                pprint_claims
                if _call_name(call) == "assert_pprint_golden"
                else other_text_claims
            )
            claims.update(testdata / f"{name}.txt" for name in names)
        other_text_claims.update(_used_text_paths(scan, test_file))
    text_goldens = set(testdata.glob("*.txt"))
    return (
        pprint_claims - text_goldens,
        text_goldens - pprint_claims - other_text_claims,
    )


def _call_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _strings_in(expression: ast.expr) -> set[str]:
    return {
        node.value
        for node in ast.walk(expression)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }


def _parameter_names(expression: ast.expr) -> list[str]:
    if isinstance(expression, ast.Constant) and isinstance(expression.value, str):
        return [name.strip() for name in expression.value.split(",")]
    if isinstance(expression, (ast.List, ast.Tuple)):
        return [
            item.value
            for item in expression.elts
            if isinstance(item, ast.Constant) and isinstance(item.value, str)
        ]
    return []


def _parametrized_values(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    parameter: str,
) -> set[str]:
    values: set[str] = set()
    for decorator in function.decorator_list:
        if (
            not isinstance(decorator, ast.Call)
            or _call_name(decorator) != "parametrize"
        ):
            continue
        if len(decorator.args) < 2:
            continue
        names = _parameter_names(decorator.args[0])
        if parameter not in names:
            continue
        index = names.index(parameter)
        rows = decorator.args[1]
        if not isinstance(rows, (ast.List, ast.Tuple)):
            continue
        for row in rows.elts:
            value = row
            if len(names) > 1 and isinstance(row, (ast.List, ast.Tuple)):
                value = row.elts[index]
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                values.add(value.value)
    return values


def _literal_name_values(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    call: ast.Call,
) -> set[str]:
    keyword = next((item for item in call.keywords if item.arg == "name"), None)
    if keyword is None:
        return set()
    return _expression_values(function, keyword.value)


def _expression_values(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    expression: ast.expr,
) -> set[str]:
    if isinstance(expression, ast.Constant) and isinstance(expression.value, str):
        return {expression.value}
    if isinstance(expression, ast.JoinedStr):
        # Every literal part joined with every value of each plain ``{name}``.
        joined: set[str] = {""}
        for part in expression.values:
            if isinstance(part, ast.FormattedValue):
                plain = part.conversion == -1 and part.format_spec is None
                options = (
                    _expression_values(function, part.value) if plain else set[str]()
                )
            else:
                options = _strings_in(part)
            joined = {prefix + option for prefix in joined for option in options}
        return joined
    if not isinstance(expression, ast.Name):
        return set()
    values = _parametrized_values(function, expression.id)
    for node in function.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(
            isinstance(target, ast.Name) and target.id == expression.id
            for target in targets
        ):
            continue
        if node.value is not None:
            values.update(_strings_in(node.value))
    return values


def _text_path_name(expression: ast.expr) -> str | None:
    if (
        isinstance(expression, ast.BinOp)
        and isinstance(expression.op, ast.Div)
        and isinstance(expression.right, ast.Constant)
        and isinstance(expression.right.value, str)
        and expression.right.value.endswith(".txt")
    ):
        return Path(expression.right.value).name
    return None


# The helper's own tests pass a ``tmp_path`` ``test_file``; their goldens never exist in
# the tree, so they claim nothing.
def _owns_beside_test_file(call: ast.Call) -> bool:
    """Return whether `call` addresses a golden beside the module's own file."""
    keyword = next((item for item in call.keywords if item.arg == "test_file"), None)
    if keyword is None:
        return True
    return any(
        isinstance(node, ast.Name) and node.id == "__file__"
        for node in ast.walk(keyword.value)
    )


@dataclass(slots=True, kw_only=True)
class _TreeScan:
    """Everything one test module contributes to golden ownership, from one walk."""

    golden_calls: list[tuple[ast.FunctionDef | ast.AsyncFunctionDef, ast.Call]]
    text_variables: dict[str, str]
    text_calls: list[ast.Call]
    reads_sites: bool = False


def _scan_tree(tree: ast.Module) -> _TreeScan:
    """Collect every ownership-relevant node from one walk of the module."""
    scan = _TreeScan(golden_calls=[], text_variables={}, text_calls=[])
    functions: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
    golden_calls: list[ast.Call] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            call_name = _call_name(node)
            # The last three are ``baselines/craftax/testing.py``'s text goldens.
            if call_name in {
                "assert_pprint_golden",
                "assert_text_golden",
                "assert_golden",
                "read_golden",
                "require_golden",
            } and _owns_beside_test_file(node):
                golden_calls.append(node)
            elif call_name in {"read_text", "write_text"} and isinstance(
                node.func,
                ast.Attribute,
            ):
                scan.text_calls.append(node)
            elif call_name in {"read_sites", "assert_sites_reached"}:
                # ``mlsbench.dl.vision``'s sites file, read from a testdata directory.
                scan.reads_sites = True
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            _scan_assignment(node, scan=scan)
    for call in golden_calls:
        function = _enclosing_function(functions, call)
        if function is not None:
            scan.golden_calls.append((function, call))
    return scan


def _enclosing_function(
    functions: list[ast.FunctionDef | ast.AsyncFunctionDef],
    call: ast.Call,
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    """Return the innermost function whose line span contains `call`, if any."""
    enclosing = [
        function
        for function in functions
        if function.end_lineno is not None
        and function.lineno <= call.lineno <= function.end_lineno
    ]
    if not enclosing:
        return None
    return max(enclosing, key=_function_lineno)


def _function_lineno(function: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    return function.lineno


def _scan_assignment(node: ast.Assign | ast.AnnAssign, *, scan: _TreeScan) -> None:
    if node.value is None:
        return
    name = _text_path_name(node.value)
    if name is None:
        return
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    scan.text_variables.update(
        (target.id, name) for target in targets if isinstance(target, ast.Name)
    )


def _used_text_paths(scan: _TreeScan, test_file: Path) -> set[Path]:
    claims: set[Path] = set()
    for call in scan.text_calls:
        assert isinstance(call.func, ast.Attribute)
        receiver = call.func.value
        if isinstance(receiver, ast.Name):
            name = scan.text_variables.get(receiver.id)
        else:
            name = _text_path_name(receiver)
        if name is not None:
            claims.add(test_file.parent / "testdata" / name)
    if scan.reads_sites:
        claims.add(test_file.parent / "testdata" / "reference_sites.txt")
    return claims
