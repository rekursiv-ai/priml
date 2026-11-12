"""Tests for the golden-ownership gate."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from priml.testing import golden_owners
from priml.testing.golden_owners import main, ownership


if TYPE_CHECKING:
    from pathlib import Path


def _model(tmp_path: Path, source: str, goldens: tuple[str, ...] = ()) -> Path:
    """Write one test directory: ``example_test.py`` and its ``testdata/``."""
    model = tmp_path / "model"
    (model / "testdata").mkdir(parents=True)
    (model / "example_test.py").write_text(source, encoding="utf-8")
    for name in goldens:
        (model / "testdata" / name).write_text("golden\n", encoding="utf-8")
    return model


def test_async_pprint_test_claims_its_golden(tmp_path: Path) -> None:
    model = _model(
        tmp_path,
        "async def test_case():\n"
        "    assert_pprint_golden(test_file=__file__, name='async_case', config=None)\n",
        ("async_case.txt",),
    )

    assert ownership(model) == (set(), set())


def test_a_pprint_claim_without_its_golden_is_missing(tmp_path: Path) -> None:
    model = _model(
        tmp_path,
        "def test_case():\n"
        "    assert_pprint_golden(test_file=__file__, name='gone', config=None)\n",
    )

    assert ownership(model) == ({model / "testdata" / "gone.txt"}, set())


def test_golden_call_on_a_tmp_test_file_claims_nothing(tmp_path: Path) -> None:
    model = _model(
        tmp_path,
        "def test_case(tmp_path):\n"
        "    assert_pprint_golden(test_file=str(tmp_path), name='x', config=None)\n",
    )

    assert ownership(model) == (set(), set())


@pytest.mark.parametrize(
    "source",
    [
        (
            '@pytest.mark.parametrize("case", ["a", "b"])\n'
            "def test_case(case):\n"
            '    assert_text_golden(test_file=__file__, name=f"env_{case}", text="")\n'
        ),
        (
            '@pytest.mark.parametrize(("case", "n"), [("a", 1), ("b", 2), (x, 3)])\n'
            "def test_case(case, n):\n"
            '    assert_golden(name=f"env_{case}", value=n)\n'
        ),
        (
            '@pytest.mark.parametrize("case,n", (("a", 1), ("b", 2)))\n'
            "def test_case(case, n):\n"
            '    read_golden(name=f"env_{case}")\n'
        ),
        (
            "def test_case(flag, holder):\n"
            "    setup()\n"
            '    other = "env_c"\n'
            '    holder.attr = "env_c"\n'
            "    name: str\n"
            '    name: str = "env_a" if flag else "env_b"\n'
            "    require_golden(name=name)\n"
        ),
        # Only the parametrize naming ``case``, with rows it can read, counts.
        (
            "@pytest.mark.cli_git\n"
            '@pytest.mark.skipif("case", ["c"])\n'
            '@pytest.mark.parametrize("case")\n'
            '@pytest.mark.parametrize("n", [1, 2])\n'
            '@pytest.mark.parametrize("case", CASES)\n'
            '@pytest.mark.parametrize(["case", "n", OTHER], [("a", 1), ("b", 2), ["x", 3]])\n'
            '@pytest.mark.parametrize("case", [("c", "c")])\n'
            '@pytest.mark.parametrize(NAMES, [("c", 1)])\n'
            "def test_case(case, n):\n"
            "    factory()()\n"
            '    assert_golden(name=f"env_{case}")\n'
        ),
    ],
    ids=["one-name", "tuple-rows", "comma-names", "assigned", "decorators"],
)
def test_a_text_golden_claims_every_literal_its_name_can_take(
    tmp_path: Path,
    source: str,
) -> None:
    model = _model(tmp_path, source, ("env_a.txt", "env_b.txt", "env_c.txt"))

    assert ownership(model) == (set(), {model / "testdata" / "env_c.txt"})


def test_host_keyed_golden_claims_each_host_of_its_parametrized_names(
    tmp_path: Path,
) -> None:
    """A later golden call in the same file still claims its own golden."""
    model = _model(
        tmp_path,
        '@pytest.mark.parametrize("case", ["a", "b"])\n'
        "def test_case(case):\n"
        '    read_golden(test_file=__file__, name=f"env_{case}", host=key())\n'
        '    read_golden(name="after")\n',
        (
            *(f"env_{case}_{host}.txt" for case in ("a", "b") for host in "xy"),
            "env_c_x.txt",
            "after.txt",
        ),
    )

    assert ownership(model) == (set(), {model / "testdata" / "env_c_x.txt"})


def test_only_a_pprint_claim_can_be_missing(tmp_path: Path) -> None:
    """Text goldens are written by their tests, so an absent one is no failure."""
    model = _model(
        tmp_path,
        "def test_case():\n    assert_text_golden(name='later', text='')\n",
    )

    assert ownership(model) == (set(), set())


def test_the_innermost_function_resolves_a_name(tmp_path: Path) -> None:
    """A one-line function encloses its call; a module-level call claims nothing."""
    model = _model(
        tmp_path,
        "def test_outer():\n"
        '    name = "env_a"\n'
        "    def inner():\n"
        '        name = "env_b"\n'
        "        read_golden(name=name)\n"
        'def test_one_line(): read_golden(name="env_c")\n'
        'read_golden(name="module")\n',
        ("env_a.txt", "env_b.txt", "env_c.txt", "module.txt"),
    )

    assert ownership(model) == (
        set(),
        {model / "testdata" / "env_a.txt", model / "testdata" / "module.txt"},
    )


@pytest.mark.parametrize(
    "name",
    ['f"env_{case!r}"', 'f"env_{case:>3}"', "pick()"],
    ids=["conversion", "format-spec", "call"],
)
def test_a_name_that_resolves_to_no_literal_is_an_error(
    tmp_path: Path,
    name: str,
) -> None:
    model = _model(
        tmp_path,
        '@pytest.mark.parametrize("case", ["a"])\n'
        f"def test_case(case):\n    assert_pprint_golden(name={name}, config=None)\n",
    )

    with pytest.raises(ValueError, match=r"example_test\.py:3: name= must resolve"):
        ownership(model)


def test_a_golden_call_with_no_name_is_an_error(tmp_path: Path) -> None:
    model = _model(tmp_path, "def test_case():\n    assert_pprint_golden(config=1)\n")

    with pytest.raises(ValueError, match=r"example_test\.py:2: name= must resolve"):
        ownership(model)


def test_an_unparsable_test_file_names_itself(tmp_path: Path) -> None:
    model = _model(tmp_path, "def test_case(:\n    read_golden(name='x')\n")

    with pytest.raises(SyntaxError) as raised:
        ownership(model)

    assert raised.value.filename == str(model / "example_test.py")


def test_every_test_file_of_the_directory_claims(tmp_path: Path) -> None:
    """A file with no golden call does not stop the scan of the next one."""
    model = _model(tmp_path, "", ("b.txt",))
    (model / "z_test.py").write_text(
        "def test_b():\n    read_golden(name='b')\n",
        encoding="utf-8",
    )

    assert ownership(model) == (set(), set())


@pytest.mark.parametrize("call", ["read_sites(DATA)", "assert_sites_reached(DATA)"])
def test_a_sites_read_claims_the_sites_file_beside_the_test(
    tmp_path: Path,
    call: str,
) -> None:
    model = _model(
        tmp_path,
        f"def test_case():\n    {call}\n",
        ("reference_sites.txt",),
    )

    assert ownership(model) == (set(), set())


def test_a_read_text_golden_is_claimed_through_its_variable(tmp_path: Path) -> None:
    model = _model(
        tmp_path,
        "from pathlib import Path\n"
        'DATA: Path = Path(__file__).parent / "testdata" / "a.txt"\n'
        'def test_case():\n    DATA.read_text()\n    (DIR / "b.txt").write_text("")\n',
        ("a.txt", "b.txt"),
    )

    assert ownership(model) == (set(), set())


def test_unrelated_text_literal_does_not_claim_a_golden(tmp_path: Path) -> None:
    model = _model(
        tmp_path,
        "from pathlib import Path\n"
        'UNRELATED = Path(__file__).parent / "testdata" / "orphan.txt"\n'
        "def test_case():\n    UNRELATED.exists()\n",
        ("orphan.txt",),
    )

    assert ownership(model) == (set(), {model / "testdata" / "orphan.txt"})


def test_a_test_file_with_no_golden_call_is_not_parsed(tmp_path: Path) -> None:
    model = _model(tmp_path, "def test_case(:\n", ("orphan.txt",))

    assert ownership(model) == (set(), {model / "testdata" / "orphan.txt"})


def test_main_checks_the_directory_of_each_changed_test_or_golden(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An unrelated file selects nothing; a clean directory reports nothing."""
    model = _model(
        tmp_path,
        "def test_case():\n    assert_pprint_golden(name='gone', config=None)\n",
        ("orphan.txt",),
    )
    clean = tmp_path / "clean"
    clean.mkdir()
    (clean / "clean_test.py").write_text("", encoding="utf-8")

    assert main([str(tmp_path / "notes.md"), str(clean / "clean_test.py")]) == 0
    for changed in (model / "example_test.py", model / "testdata" / "orphan.txt"):
        assert main([str(changed)]) == 1
        assert capsys.readouterr().err.splitlines() == [
            f"{model / 'testdata' / 'gone.txt'}: claimed, but no such golden",
            f"{model / 'testdata' / 'orphan.txt'}: no test claims this golden",
        ]


@pytest.mark.parametrize(
    ("docstring", "after_usage"),
    [
        (golden_owners.__doc__, "Check that every priml text golden"),
        (None, "positional arguments:"),
    ],
    ids=["docstring", "optimized away"],
)
def test_help_describes_the_gate_from_its_docstring(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    docstring: str | None,
    after_usage: str,
) -> None:
    """``python -OO`` strips the docstring; the help then has no description."""
    monkeypatch.setattr(golden_owners, "__doc__", docstring)

    with pytest.raises(SystemExit):
        main(["--help"])

    out = capsys.readouterr().out
    assert out.split("\n\n")[1].startswith(after_usage)
    assert ("golden_owners priml/a/b_test.py" in out) == bool(docstring)
    assert "Changed test files or goldens." in out


def test_a_change_to_the_rule_checks_its_whole_package(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Directories holding only tests, or only goldens, are both rechecked."""
    rule = tmp_path / "testing" / golden_owners.__file__.rsplit("/", 1)[1]
    rule.parent.mkdir()
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "a_test.py").write_text(
        "def test_a():\n    assert_pprint_golden(name='gone', config=None)\n",
        encoding="utf-8",
    )
    (tmp_path / "b" / "testdata").mkdir(parents=True)
    (tmp_path / "b" / "testdata" / "orphan.txt").write_text("", encoding="utf-8")

    assert main([str(rule)]) == 1
    assert capsys.readouterr().err.splitlines() == [
        f"{tmp_path / 'a' / 'testdata' / 'gone.txt'}: claimed, but no such golden",
        f"{tmp_path / 'b' / 'testdata' / 'orphan.txt'}: no test claims this golden",
    ]


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
