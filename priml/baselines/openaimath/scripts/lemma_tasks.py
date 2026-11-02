"""Cut openai/math's Lean library into lemma tasks, split by family.

A whole openai/math challenge is a research result whose reference proof spans
67 library modules at the median; a 4B policy should prove almost none of them,
so training on them alone would give a reward that is zero nearly everywhere. The library those proofs are built from
is also a corpus of ordinary lemmas, each with a statement, the file context
before it, and a proof the kernel accepted. This module turns those lemmas into
tasks in exactly the dataset's own format, so Harbor, the shared Daytona
snapshot, and the dataset's own grader (``tests/grade.py``, Comparator) run them
unchanged.

The split is by family, because the release shares one Lean library across all
challenges: a lemma from a module that some held-out challenge's reference proof
imports would teach the policy part of that proof. So:

- A family is held out when its salted hash lands in the bottom fifth
  (:func:`is_test_family`). The rule needs no list, so it cannot drift.
- Test-side modules are every OAI module a held-out challenge's reference proof
  imports, directly or not. Training lemmas come only from modules a training
  challenge imports and no held-out or unpublished challenge does. Modules no
  challenge imports are left out too: nothing says which family they serve.
- Held-out lemmas (from test-side modules) are a progress metric only; the
  protected score is the official grader on the held-out families' own
  challenges.

A lemma task's challenge is its module, verbatim, up to the lemma, with the
lemma's proof replaced by ``sorry``. Earlier declarations keep their proofs and
the lemma's own proof is the reference solution. Only modules that import
nothing beyond Mathlib and Lean's own libraries are used, because the image has
no OAI library installed, and only modules small enough that the whole prefix
fits the episode's prompt.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import hashlib
import json
import re
import shutil
import tomllib


if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence


SPLIT_SALT: Final = "openaimath-family-split"
"""Salt of the family hash; changing it re-draws the whole split."""

TEST_FAMILY_CUTOFF: Final = 52
"""A family is held out when its digest's first byte is below this: 52/256, about 20%."""

MAX_MODULE_BYTES: Final = 24_000
"""Largest module a lemma is cut from.

The instruction embeds the prefix, and the episode's prompt budget is 16,384
tokens; Lean source runs about three bytes a token, so a 24 KB prefix leaves
room for the rest of the instruction and the protocol.
"""

LEAN_ROOTS: Final = frozenset(
    {"Aesop", "Batteries", "Init", "Lean", "Mathlib", "Plausible", "Qq", "Std"},
)
"""Import roots the image provides: Mathlib, its dependencies, and Lean itself."""

PERMITTED_AXIOMS: Final = ("propext", "Classical.choice", "Quot.sound")
"""The axioms the dataset's own challenges permit."""

GRADE_TIMEOUT_SEC: Final = 1_800
"""Seconds the grader gets for one lemma.

Building a challenge statement takes about 5 s at the median in the dataset's
own measurements, and these modules are at most 24 KB, against a 3-hour floor
sized for whole reference proofs.
"""

AGENT_TIMEOUT_SEC: Final = 7_200
"""Seconds an episode may take; its token budget, not this, normally ends it."""

OFFICIAL_TEST_TASKS_NAME: Final = "official_test_tasks.txt"
"""The held-out families' published tasks, one per line: the protected score's tasks."""

SORRY_PROOF: Final = " := by\n  sorry\n"
"""What replaces the target lemma's proof in its challenge."""

_REJECTED_SOURCE: Final = re.compile(
    r"\b(?:sorry|admit|native_decide)\b|^\s*axiom\s|@\[\s*(?:implemented_by|extern)\b|\bunsafe\b",
    re.MULTILINE,
)
"""Source the grader would refuse, or that could break the statement check."""

_IMPORT: Final = re.compile(
    r"^\s*(?:public\s+|private\s+)?(?:meta\s+)?import\s+(?:all\s+)?([A-Za-z_«]\S*)",
)
_HEADER_SKIP: Final = re.compile(r"^\s*(?:--|/-|$)")
_MODULE_SYSTEM: Final = re.compile(r"^\s*(?:module|prelude)\b")
_TARGET: Final = re.compile(
    r"(?:@\[[^\]\n]*\]\s*)?"
    r"(?P<mods>(?:(?:private|protected|noncomputable|nonrec)\s+)*)"
    r"(?:theorem|lemma)\s+(?P<name>(?:[^\s(\[{:⦃.]|\.(?!\{))+(?:\.\{[^}\n]*\})?)",
)
_NAMESPACE: Final = re.compile(r"namespace\s+(\S+)")
_SECTION: Final = re.compile(r"(?:noncomputable\s+)?section\b[ \t]*(\S*)")
_END: Final = re.compile(r"end\b[ \t]*(\S*)")
_IDENTIFIER: Final = re.compile(r"[^\W\d][\w'!?₀-₉]*(?:\.[^\W\d][\w'!?₀-₉]*)*")
_CONTINUATION: Final = ("|", ")", "]", "}", "⟩", ",")
_CONTINUATION_WORDS: Final = frozenset(
    {"where", "deriving", "termination_by", "decreasing_by"},
)
_OPEN: Final = "([{⟨⦃"
_CLOSE: Final = ")]}⟩⦄"


def is_test_family(family: str) -> bool:
    """Return whether a family is held out from training."""
    digest = hashlib.sha256(f"{SPLIT_SALT}:{family}".encode()).digest()
    return digest[0] < TEST_FAMILY_CUTOFF


@dataclass(frozen=True, slots=True)
class Challenge:
    """One official challenge: its family and the module its reference proof roots at."""

    name: str
    family: str | None
    """The family, or ``None`` for a challenge the dataset did not publish."""

    solution_module: str


@dataclass(frozen=True, slots=True)
class Lemma:
    """One lemma task, before it is written out."""

    module: str
    theorem: str
    """The lemma's fully qualified name, as Comparator checks it."""

    challenge: str
    """The module up to the lemma, with the lemma's proof replaced by ``sorry``."""

    reference: str
    """The same prefix with the lemma's own proof: the reference solution."""

    @property
    def digest(self) -> str:
        """Return a short stable id for the lemma."""
        key = f"{self.module}:{self.theorem}".encode()
        return hashlib.sha256(key).hexdigest()[:12]

    @property
    def name(self) -> str:
        """Return the task's directory name, which is also its Harbor task name."""
        return f"lemma-{self.digest}"

    @property
    def challenge_name(self) -> str:
        """Return the challenge's name: letters and digits only, safe in a shell."""
        return f"Lemma{self.digest}"


@dataclass(frozen=True, slots=True)
class Split:
    """Which modules each side may use, and why."""

    test_families: tuple[str, ...]
    train_families: tuple[str, ...]
    test_modules: frozenset[str]
    """Every OAI module a held-out challenge's reference proof imports."""

    train_modules: frozenset[str]
    """Modules only training challenges import."""


def header_imports(text: str) -> tuple[list[str], bool]:
    """Read a module's imports, and whether it is plain Lean.

    Lean requires imports before any other command, so reading stops at the
    first line that is not an import, a comment, or blank. A module-system file
    (``module`` or ``prelude`` header) is not plain: its declarations need
    ``public`` to be visible to another module, which a cut prefix cannot keep.

    Returns:
      imports: The imported module names.
      plain: Whether the file uses neither header.

    """
    imports: list[str] = []
    in_comment = False
    for line in text.splitlines():
        if in_comment:
            in_comment = "-/" not in line
            continue
        if line.lstrip().startswith("/-") and "-/" not in line:
            in_comment = True
            continue
        if _MODULE_SYSTEM.match(line):
            return imports, False
        match = _IMPORT.match(line)
        if match:
            imports.append(match.group(1).replace("«", "").replace("»", ""))
        elif not _HEADER_SKIP.match(line):
            break
    return imports, True


def import_graph(lean_dir: Path) -> dict[str, tuple[str, ...]]:
    """Return every OAI module's imports, keyed by module name."""
    graph: dict[str, tuple[str, ...]] = {}
    for path in sorted((lean_dir / "OAI").rglob("*.lean")):
        name = ".".join(path.relative_to(lean_dir).with_suffix("").parts)
        imports, _ = header_imports(path.read_text(encoding="utf-8", errors="replace"))
        graph[name] = tuple(imports)
    return graph


def closure(graph: Mapping[str, Sequence[str]], root: str) -> set[str]:
    """Return every OAI module reachable from ``root`` through imports."""
    seen: set[str] = set()
    stack = [root]
    while stack:
        module = stack.pop()
        if module in seen or module not in graph:
            continue
        seen.add(module)
        stack.extend(name for name in graph[module] if name.startswith("OAI"))
    return seen


def split_modules(
    graph: Mapping[str, Sequence[str]],
    challenges: Iterable[Challenge],
) -> Split:
    """Assign modules to the training and held-out sides.

    Args:
      graph: Every OAI module's imports.
      challenges: Every challenge in the release, published or not.

    Returns:
      split: The families on each side and the modules each may draw from.

    """
    test_families: set[str] = set()
    train_families: set[str] = set()
    test_modules: set[str] = set()
    train_modules: set[str] = set()
    unassigned: set[str] = set()
    for challenge in challenges:
        modules = closure(graph, challenge.solution_module)
        if challenge.family is None:
            unassigned |= modules
        elif is_test_family(challenge.family):
            test_families.add(challenge.family)
            test_modules |= modules
        else:
            train_families.add(challenge.family)
            train_modules |= modules
    return Split(
        test_families=tuple(sorted(test_families)),
        train_families=tuple(sorted(train_families)),
        test_modules=frozenset(test_modules),
        train_modules=frozenset(train_modules - test_modules - unassigned),
    )


def blank_comments_and_strings(text: str) -> str:
    """Return ``text`` with comments and string literals replaced by spaces.

    Lengths and newlines are kept, so an offset found in the result is the same
    offset in the source. Block comments nest, as Lean's do.
    """
    out = list(text)
    size = len(text)
    index = 0
    while index < size:
        if text.startswith("/-", index):
            depth, end = 1, index + 2
            while end < size and depth:
                if text.startswith("/-", end):
                    depth, end = depth + 1, end + 2
                elif text.startswith("-/", end):
                    depth, end = depth - 1, end + 2
                else:
                    end += 1
        elif text.startswith("--", index):
            end = text.find("\n", index)
            end = size if end < 0 else end
        elif text[index] == '"':
            end = index + 1
            while end < size and text[end] != '"':
                end += 2 if text[end] == "\\" else 1
            end = min(end + 1, size)
        else:
            index += 1
            continue
        for position in range(index, end):
            if out[position] != "\n":
                out[position] = " "
        index = end
    return "".join(out)


def command_starts(code: str) -> list[int]:
    """Return the offsets of top-level commands in comment-blanked source.

    A command starts on a line that begins in column 0, unless that line
    continues the previous command (a match arm, a closing bracket, ``where``).
    """
    starts: list[int] = []
    offset = 0
    for line in code.splitlines(keepends=True):
        stripped = line.rstrip("\n")
        if stripped and not stripped[0].isspace():
            word = stripped.split(maxsplit=1)[0]
            if (
                not stripped.startswith(_CONTINUATION)
                and word not in _CONTINUATION_WORDS
            ):
                starts.append(offset)
        offset += len(line)
    return starts


def _statement_end(code: str, begin: int, end: int) -> int | None:
    """Return the offset of the ``:=`` that ends a lemma's statement.

    That is the first ``:=`` outside every bracket. A lemma proved by pattern
    matching (``=>`` or ``where`` first) has none, and neither does a malformed
    one; both return ``None`` and are skipped.
    """
    depth = 0
    for index in range(begin, end):
        char = code[index]
        if char in _OPEN:
            depth += 1
        elif char in _CLOSE:
            depth -= 1
            if depth < 0:
                return None
        elif depth == 0:
            if code.startswith(":=", index):
                return index
            if code.startswith("=>", index):
                return None
            if code.startswith("where", index) and not code[index - 1].isalnum():
                return None
    return None


def _close_scopes(scopes: Sequence[tuple[str, str]]) -> str:
    """Return ``end`` commands that close every open scope, innermost first."""
    return "".join(f"\nend {name}".rstrip() for _, name in reversed(scopes))


def lemmas_in(module: str, text: str) -> list[Lemma]:
    """Return one lemma task per public top-level theorem of a module.

    Args:
      module: The module's name.
      text: The module's source.

    Returns:
      lemmas: The module's lemmas in source order; empty when the scopes do not
        balance, which means the file is not read the way Lean reads it.

    """
    code = blank_comments_and_strings(text)
    starts = command_starts(code)
    scopes: list[tuple[str, str]] = []
    lemmas: list[Lemma] = []
    for position, start in enumerate(starts):
        end = starts[position + 1] if position + 1 < len(starts) else len(code)
        newline = code.find("\n", start, end)
        line = code[start : newline if newline >= 0 else end]
        if match := _NAMESPACE.match(line):
            scopes.extend(("namespace", part) for part in match.group(1).split("."))
            continue
        if match := _SECTION.match(line):
            scopes.append(("section", match.group(1)))
            continue
        if line.split(maxsplit=1)[0] == "mutual":
            scopes.append(("mutual", ""))
            continue
        if match := _END.match(line):
            if not scopes:
                return []
            name = match.group(1)
            count = len(name.split(".")) if name and scopes[-1][0] == "namespace" else 1
            if count > len(scopes):
                return []
            del scopes[-count:]
            continue
        lemma = _lemma_at(module, text, code, start, end, line, scopes)
        if lemma is not None:
            lemmas.append(lemma)
    return lemmas


def _lemma_at(  # noqa: PLR0917 -- One command's context, passed whole.
    module: str,
    text: str,
    code: str,
    start: int,
    end: int,
    line: str,
    scopes: Sequence[tuple[str, str]],
) -> Lemma | None:
    """Return the lemma task a command defines, or ``None`` when it is not one."""
    match = _TARGET.match(line)
    if match is None or "private" in match.group("mods").split():
        return None
    if any(kind == "mutual" for kind, _ in scopes):
        return None
    name = re.sub(r"\.\{[^}]*\}$", "", match.group("name"))
    if name.startswith("_root_."):
        name = name.removeprefix("_root_.")
        namespace = cast(list[str], [])
    else:
        namespace = [part for kind, part in scopes if kind == "namespace"]
    if not _IDENTIFIER.fullmatch(name):
        return None
    statement_end = _statement_end(code, start + match.end(), end)
    if statement_end is None:
        return None
    closing = _close_scopes(scopes).lstrip("\n") + ("\n" if scopes else "")
    # Comments after the proof belong to the next declaration (a docstring
    # before ``end`` does not parse), so the reference stops at the proof's
    # last code character.
    proof_end = len(code[:end].rstrip())
    return Lemma(
        module=module,
        theorem=".".join([*namespace, name]),
        challenge=text[:statement_end].rstrip() + SORRY_PROOF + closing,
        reference=text[:proof_end] + "\n" + closing,
    )


def usable_module(text: str, imports: Sequence[str], *, plain: bool) -> bool:
    """Return whether lemmas may be cut from a module.

    It must import only what the image provides, fit the prompt, be plain Lean,
    and contain nothing the grader would refuse in a prefix it must accept.
    """
    return (
        plain
        and all(name.split(".")[0] in LEAN_ROOTS for name in imports)
        and len(text.encode()) <= MAX_MODULE_BYTES
        and _REJECTED_SOURCE.search(text) is None
    )


def collect_lemmas(lean_dir: Path, modules: Iterable[str]) -> list[Lemma]:
    """Return every lemma task the given modules yield, ordered by task name."""
    lemmas: dict[str, Lemma] = {}
    for module in sorted(modules):
        path = (lean_dir / Path(*module.split("."))).with_suffix(".lean")
        text = path.read_text(encoding="utf-8")
        imports, plain = header_imports(text)
        if not usable_module(text, imports, plain=plain):
            continue
        for lemma in lemmas_in(module, text):
            lemmas.setdefault(lemma.name, lemma)
    return [lemmas[name] for name in sorted(lemmas)]


def official_challenges(tasks_dir: Path, lean_dir: Path) -> list[Challenge]:
    """Return every challenge of the release, with the family of each published one.

    The published tasks' ``task.toml`` files name each one's family and
    reference module. A challenge the dataset left out has no family there,
    but its reference proof still claims modules, so it is returned with
    ``family=None`` and its modules stay out of training.
    """
    published: dict[str, Challenge] = {}
    for path in sorted(tasks_dir.glob("*/task.toml")):
        with path.open("rb") as stream:
            metadata = cast("dict[str, object]", tomllib.load(stream)["metadata"])
        name = str(metadata["challenge"])
        published[name] = Challenge(
            name=name,
            family=str(metadata["family"]),
            solution_module=str(metadata["reference_module"]),
        )
    challenges = dict(published)
    for path in sorted((lean_dir / "ComparatorChallenges").glob("*.json")):
        if path.stem in challenges:
            continue
        config = cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8")))
        challenges[path.stem] = Challenge(
            name=path.stem,
            family=None,
            solution_module=str(config["solution_module"]),
        )
    return [challenges[name] for name in sorted(challenges)]


def instruction(lemma: Lemma) -> str:
    """Return a lemma task's instruction, in the dataset's own wording where it applies."""
    return f"""# openai/math lemma `{lemma.theorem}`

Prove the lemma `{lemma.theorem}` from module `{lemma.module}` of OpenAI's
[openai/math](https://github.com/openai/math) Lean library in Lean 4, with a proof the Lean
kernel accepts.

`/workspace/Submission.lean` already holds the file below: the module up to this lemma, with the
lemma's proof replaced by `sorry`. The declarations before it keep their proofs, and you may use
them.

```lean
{lemma.challenge.rstrip()}
```

## What to submit

Replace the `sorry` in `/workspace/Submission.lean` with a proof. The file is graded on three things:

- **Same statements.** `{lemma.theorem}` must keep exactly the statement shown above: same name,
  namespaces, binders and types. Every declaration before it must stay exactly as written.
- **Standard axioms only.** Proofs may use only `propext`, `Quot.sound` and `Classical.choice`.
  `sorry`, `admit`, new `axiom`s and `native_decide` (it introduces an axiom of its own) are rejected.
- **Kernel-checked.** The proofs are re-checked by the Lean kernel, not just the elaborator.

Helper lemmas may go just before `{lemma.theorem}`, or in a separate `Submission/*.lean` module
(module names `Submission.Foo`) imported from `Submission.lean`. Do not add instances or notation:
they can change how the statement elaborates, and then it no longer matches.

## Environment

- Lean `v4.34.1` and Mathlib at commit `d13f23b` are installed and prebuilt; `/workspace` is a
  Lake project.
- The sandbox has 4 CPUs and 8 GB of memory. Check your work with
  `cd /workspace && lake build Submission`. Add `#print axioms <name>` to see which axioms a proof
  uses.
- There is no internet access. OpenAI's own proofs are not installed.

## Grading

When you finish, `Submission.lean` and `Submission/**.lean` are copied to a fresh machine and
checked with [Comparator](https://github.com/leanprover/comparator), the Lean FRO's proof checker.
The reward is 1 if Comparator accepts the proof and 0 otherwise.
"""


def task_toml(lemma: Lemma, *, split: str, source_commit: str) -> str:
    """Return a lemma task's ``task.toml``, matching the dataset's except where noted."""

    def quote(value: str) -> str:
        return json.dumps(value, ensure_ascii=False)

    return f"""schema_version = "1.4"

[task]
name = {quote(f"openai-math/{lemma.name}")}
version = "1.0.0"
description = {quote(f"Prove the lemma {lemma.theorem} from openai/math module {lemma.module} in Lean 4 with Mathlib; graded by Comparator in a separate sandbox.")}
keywords = ["lean4", "mathlib", "theorem-proving", "openai-math", "lemma"]

[metadata]
source = "https://github.com/openai/math"
source_commit = {quote(source_commit)}
source_license = "Apache-2.0"
source_notice = {quote(f"tests/Challenge.lean is lean/{lemma.module.replace('.', '/')}.lean from openai/math, (c) OpenAI, cut before the lemma's proof")}
kind = "lemma"
split = {quote(split)}
challenge = {quote(lemma.challenge_name)}
module = {quote(lemma.module)}
theorems = [{quote(lemma.theorem)}]

[agent]
timeout_sec = {AGENT_TIMEOUT_SEC}
network_mode = "no-network"

[environment]
build_timeout_sec = 5400
cpus = 4
memory_mb = 8192
storage_mb = 10240
workdir = "/workspace"
# Lean sees all host CPUs rather than the sandbox's 4; this bounds `lake build` to 3 parallel jobs.
env = {{ LEAN_NUM_THREADS = "3" }}

[verifier]
timeout_sec = {GRADE_TIMEOUT_SEC + 900}
environment_mode = "separate"
network_mode = "no-network"

[[verifier.collect]]
command = "python3 /opt/openai-math/pack_submission.py"
timeout_sec = 300
"""


SOLVE_SH: Final = """#!/usr/bin/env bash
# Oracle: submit openai/math's own proof of this lemma, the module as written up to its end.
set -euo pipefail
cp /solution/Submission.lean /workspace/Submission.lean
echo "placed the reference proof"
"""


def write_task(
    lemma: Lemma,
    directory: Path,
    *,
    template: Path,
    split: str,
    source_commit: str,
) -> None:
    """Write one lemma task beside the dataset's own, with its grader and image.

    ``environment/`` and the grader are copied byte for byte from a published
    task (``template``): the shared Daytona snapshot is named after the image
    recipe, and the grader is the one the dataset validated.

    Args:
      lemma: The lemma to write.
      directory: Where the split's tasks go.
      template: A published task directory.
      split: ``"train"`` or ``"test"``.
      source_commit: The openai/math commit the module was read at.

    """
    task = directory / lemma.name
    if task.exists():
        shutil.rmtree(task)
    shutil.copytree(template / "environment", task / "environment")
    (task / "tests").mkdir()
    (task / "solution").mkdir()
    for name in ("grade.py", "test.sh"):
        shutil.copy2(template / "tests" / name, task / "tests" / name)
    _write(task / "instruction.md", instruction(lemma))
    _write(
        task / "task.toml",
        task_toml(lemma, split=split, source_commit=source_commit),
    )
    _write(task / "tests" / "Challenge.lean", lemma.challenge)
    _write(
        task / "tests" / "task.json",
        json.dumps(
            {
                "challenge": lemma.challenge_name,
                "comparator": {
                    "theorem_names": [lemma.theorem],
                    "definition_names": [],
                    "permitted_axioms": list(PERMITTED_AXIOMS),
                },
                "grade_timeout_sec": GRADE_TIMEOUT_SEC,
            },
            indent=2,
        ),
    )
    _write(task / "solution" / "Submission.lean", lemma.reference)
    _write(task / "solution" / "solve.sh", SOLVE_SH)
    (task / "solution" / "solve.sh").chmod(0o755)


def _write(path: Path, text: str) -> None:
    """Write UTF-8 text with Unix newlines, whatever the host."""
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def generate(
    lean_dir: Path,
    tasks_dir: Path,
    out_dir: Path,
    *,
    source_commit: str,
) -> dict[str, object]:
    """Write the training and held-out lemma tasks and the split record.

    Args:
      lean_dir: The openai/math checkout's ``lean/`` directory.
      tasks_dir: The dataset's published ``tasks/`` directory.
      out_dir: Where ``train/``, ``test/``, and ``split.json`` go.
      source_commit: The commit ``lean_dir`` is checked out at.

    Returns:
      record: What ``split.json`` holds.

    Raises:
      FileNotFoundError: There is no published task to copy the grader from.

    """
    templates = sorted(path.parent for path in tasks_dir.glob("*/task.toml"))
    if not templates:
        raise FileNotFoundError(f"No published tasks under {tasks_dir}.")
    template = templates[0]
    challenges = official_challenges(tasks_dir, lean_dir)
    split = split_modules(import_graph(lean_dir), challenges)
    sides = {
        "train": collect_lemmas(lean_dir, split.train_modules),
        "test": collect_lemmas(lean_dir, split.test_modules),
    }
    for side, lemmas in sides.items():
        directory = out_dir / side
        directory.mkdir(parents=True, exist_ok=True)
        for lemma in lemmas:
            write_task(
                lemma,
                directory,
                template=template,
                split=side,
                source_commit=source_commit,
            )
    task_of = {
        _challenge_of(path): path.parent.name
        for path in sorted(tasks_dir.glob("*/task.toml"))
    }
    official = sorted(
        task_of[challenge.name]
        for challenge in challenges
        if challenge.family is not None and is_test_family(challenge.family)
    )
    record: dict[str, object] = {
        "schema": 1,
        "salt": SPLIT_SALT,
        "test_family_cutoff": TEST_FAMILY_CUTOFF,
        "source_commit": source_commit,
        "template_task": template.name,
        "test_families": list(split.test_families),
        "train_families": list(split.train_families),
        "official_test_tasks": official,
        "test_modules": len(split.test_modules),
        "train_modules": len(split.train_modules),
        "train_lemmas": len(sides["train"]),
        "test_lemmas": len(sides["test"]),
    }
    _write(
        out_dir / OFFICIAL_TEST_TASKS_NAME,
        "".join(f"{name}\n" for name in official),
    )
    # Publish the split only after the entire task set and its list are written.
    # An interrupted generation is retried rather than mistaken for complete.
    split_path = out_dir / "split.json"
    pending = out_dir / "split.json.pending"
    _write(pending, json.dumps(record, indent=2, sort_keys=True) + "\n")
    pending.replace(split_path)
    return record


def _challenge_of(task_toml_path: Path) -> str:
    """Return the challenge a published task's ``task.toml`` names."""
    with task_toml_path.open("rb") as stream:
        metadata = cast("dict[str, object]", tomllib.load(stream)["metadata"])
    return str(metadata["challenge"])
