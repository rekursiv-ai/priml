"""Tests for cutting the openai/math library into lemma tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import json
import tomllib

from priml.baselines.openaimath.scripts import lemma_tasks
from priml.baselines.openaimath.scripts.lemma_tasks import (
    Challenge,
    Lemma,
    blank_comments_and_strings,
    is_test_family,
    lemmas_in,
    split_modules,
    usable_module,
)


if TYPE_CHECKING:
    from pathlib import Path


MODULE = """\
import Mathlib.Data.Nat.Basic

/-! Header comment with a fake theorem:
theorem not_real : True := trivial -/

namespace OAI.Demo

def double (n : ℕ) : ℕ := 2 * n

/-- Doubling is additive. -/
@[simp]
theorem double_add (m n : ℕ) : double (m + n) = double m + double n := by
  unfold double
  ring

private theorem hidden : double 0 = 0 := rfl

section Inner
variable (k : ℕ)

lemma double_pos (h : 0 < k := by decide) : 0 < double k := by
  unfold double; omega

theorem _root_.rootDouble : double 1 = 2 := rfl

theorem byCases : ∀ n : ℕ, double n = n + n
  | 0 => rfl
  | n + 1 => by unfold double; ring

end Inner

theorem double_le.{u} (n : ℕ) : n ≤ double n := by
  unfold double
  omega

/-- A docstring for the next declaration, which is not a proof. -/
end OAI.Demo
"""


def test_blank_comments_and_strings_keeps_offsets() -> None:
    """Comments and strings vanish, code and newlines stay where they were."""
    text = 'a /- x /- nested -/ y -/ b -- tail\n"s := t" c'
    blanked = blank_comments_and_strings(text)
    assert len(blanked) == len(text)
    assert blanked.split() == ["a", "b", "c"]
    assert blanked.index("\n") == text.index("\n")


def test_lemmas_in_names_every_public_theorem() -> None:
    """Namespaces qualify names; private, matched, and commented theorems are skipped."""
    names = [lemma.theorem for lemma in lemmas_in("OAI.Demo", MODULE)]
    assert names == [
        "OAI.Demo.double_add",
        "OAI.Demo.double_pos",
        "rootDouble",
        "OAI.Demo.double_le",
    ]


def test_lemmas_in_cuts_the_challenge_at_the_statement() -> None:
    """The challenge is the module verbatim up to the statement, then sorry and scope ends."""
    lemma = lemmas_in("OAI.Demo", MODULE)[1]
    prefix = MODULE[: MODULE.index("lemma double_pos")]
    assert lemma.challenge == (
        prefix
        + "lemma double_pos (h : 0 < k := by decide) : 0 < double k := by\n"
        + "  sorry\n"
        + "end Inner\nend Demo\nend OAI\n"
    )


def test_lemmas_in_keeps_attributes_and_docstrings_before_the_statement() -> None:
    """The lemma's own docstring and attribute stay attached to it."""
    challenge = lemmas_in("OAI.Demo", MODULE)[0].challenge
    assert challenge.endswith(
        "/-- Doubling is additive. -/\n@[simp]\n"
        "theorem double_add (m n : ℕ) : double (m + n) = double m + double n"
        " := by\n  sorry\nend Demo\nend OAI\n",
    )


def test_lemmas_in_reference_stops_at_the_proof() -> None:
    """The reference keeps the proof and drops the next declaration's docstring."""
    lemma = lemmas_in("OAI.Demo", MODULE)[-1]
    assert lemma.reference.endswith(
        "theorem double_le.{u} (n : ℕ) : n ≤ double n := by\n"
        "  unfold double\n  omega\nend Demo\nend OAI\n",
    )
    assert lemma.challenge.startswith(
        lemma.reference[: MODULE.index("theorem double_le")],
    )


def test_lemmas_in_refuses_unbalanced_scopes() -> None:
    """A stray end means the file is not read the way Lean reads it."""
    assert lemmas_in("OAI.Bad", "end Foo\ntheorem t : True := trivial\n") == []


def test_lemma_names_are_stable_and_shell_safe() -> None:
    """Task names come from the module and theorem only."""
    lemma = Lemma(module="OAI.A", theorem="OAI.A.t", challenge="x", reference="y")
    other = Lemma(module="OAI.A", theorem="OAI.A.t", challenge="z", reference="w")
    assert lemma.name == other.name
    assert lemma.name.startswith("lemma-")
    assert lemma.challenge_name.isalnum()


def test_is_test_family_holds_out_a_fifth_by_hash() -> None:
    """The split rule is a pure function of the family's name."""
    assert [is_test_family(str(n)) for n in range(1, 7)] == [
        False,
        False,
        False,
        True,
        True,
        True,
    ]
    held_out = sum(is_test_family(str(n)) for n in range(1, 1_001))
    assert 150 < held_out < 250


def test_split_modules_keeps_shared_and_unclaimed_modules_out_of_training() -> None:
    """Only modules that training challenges alone import are trained on."""
    graph = {
        "OAI.TrainRoot": ("OAI.TrainOnly", "OAI.Shared", "Mathlib.Data"),
        "OAI.TestRoot": ("OAI.Shared", "OAI.TestOnly"),
        "OAI.LostRoot": ("OAI.Lost",),
        "OAI.TrainOnly": (),
        "OAI.Shared": (),
        "OAI.TestOnly": (),
        "OAI.Lost": (),
        "OAI.Orphan": (),
    }
    split = split_modules(
        graph,
        [
            Challenge("Train", "1", "OAI.TrainRoot"),
            Challenge("Test", "4", "OAI.TestRoot"),
            Challenge("Unpublished", None, "OAI.LostRoot"),
        ],
    )
    assert split.train_families == ("1",)
    assert split.test_families == ("4",)
    assert split.train_modules == {"OAI.TrainRoot", "OAI.TrainOnly"}
    assert split.test_modules == {"OAI.TestRoot", "OAI.Shared", "OAI.TestOnly"}


def test_usable_module_refuses_what_the_image_or_grader_cannot_take() -> None:
    """Only plain, small, Mathlib-only modules with no escape hatches are cut."""
    ok = "import Mathlib\ntheorem t : True := trivial\n"
    assert usable_module(ok, ["Mathlib"], plain=True)
    assert not usable_module(ok, ["OAI.Other"], plain=True)
    assert not usable_module(ok, ["PrimeNumberTheoremAnd"], plain=True)
    assert not usable_module(ok, ["Mathlib"], plain=False)
    assert not usable_module(ok.replace("trivial", "sorry"), ["Mathlib"], plain=True)
    assert not usable_module(
        ok + "theorem u : True := by native_decide\n",
        ["Mathlib"],
        plain=True,
    )
    big = ok + "-- " + "x" * lemma_tasks.MAX_MODULE_BYTES
    assert not usable_module(big, ["Mathlib"], plain=True)


def test_header_imports_reads_only_the_header() -> None:
    """Imports stop at the first command; module-system files are not plain."""
    imports, plain = lemma_tasks.header_imports(
        "/- c -/\nimport Mathlib.A\n-- x\nimport Lean\n\ndef f := 1\nimport Late\n",
    )
    assert (imports, plain) == (["Mathlib.A", "Lean"], True)
    assert not lemma_tasks.header_imports("module\nimport Mathlib\n")[1]


def _template(root: Path) -> Path:
    """Write a published task's environment and grader."""
    task = root / "published"
    (task / "environment").mkdir(parents=True)
    (task / "environment" / "Dockerfile").write_text("FROM ubuntu\n", encoding="utf-8")
    (task / "tests").mkdir()
    (task / "tests" / "grade.py").write_text("print('grade')\n", encoding="utf-8")
    (task / "tests" / "test.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    return task


def test_write_task_uses_the_published_grader_and_image(tmp_path: Path) -> None:
    """A lemma task is a dataset task: same image, same grader, its own challenge."""
    lemma = lemmas_in("OAI.Demo", MODULE)[0]
    template = _template(tmp_path)
    lemma_tasks.write_task(
        lemma,
        tmp_path / "train",
        template=template,
        split="train",
        source_commit="c" * 40,
    )
    task = tmp_path / "train" / lemma.name
    assert (task / "environment" / "Dockerfile").read_bytes() == b"FROM ubuntu\n"
    assert (task / "tests" / "grade.py").read_text(
        encoding="utf-8",
    ) == "print('grade')\n"
    assert (task / "tests" / "Challenge.lean").read_text(
        encoding="utf-8",
    ) == lemma.challenge
    assert (task / "solution" / "Submission.lean").read_text(
        encoding="utf-8",
    ) == lemma.reference
    assert (task / "solution" / "solve.sh").stat().st_mode & 0o111
    config = cast(
        dict[str, object],
        json.loads((task / "tests" / "task.json").read_text(encoding="utf-8")),
    )
    assert config == {
        "challenge": lemma.challenge_name,
        "comparator": {
            "theorem_names": ["OAI.Demo.double_add"],
            "definition_names": [],
            "permitted_axioms": ["propext", "Classical.choice", "Quot.sound"],
        },
        "grade_timeout_sec": lemma_tasks.GRADE_TIMEOUT_SEC,
    }
    with (task / "task.toml").open("rb") as stream:
        toml = tomllib.load(stream)
    assert toml["task"]["name"] == f"openai-math/{lemma.name}"
    assert toml["metadata"]["split"] == "train"
    assert toml["verifier"]["environment_mode"] == "separate"
    assert toml["agent"]["network_mode"] == "no-network"
    assert lemma.challenge.rstrip() in (task / "instruction.md").read_text(
        encoding="utf-8",
    )


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
