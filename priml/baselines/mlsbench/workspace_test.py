"""Tests for workspace materialization and the diff guard."""

# ruff: noqa: TC003, PT018
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from priml.baselines.mlsbench.task import TaskSpec, load_task
from priml.baselines.mlsbench.testing import RUN_PY, build_task
from priml.baselines.mlsbench.workspace import (
    DiffGuardViolation,
    EditError,
    check_diff,
    manifest,
    materialize,
    workspace_changes,
)
from priml.baselines.mlsbench.workspace_harbor import MissingBundleError


@pytest.fixture
def task(tmp_path: Path) -> TaskSpec:
    return load_task("toy-task", root=build_task(tmp_path / "up"))


def _trees(tmp_path: Path, edited_text: str | None = RUN_PY) -> tuple[Path, Path]:
    a, b = tmp_path / "a", tmp_path / "b"
    for d in (a, b):
        d.mkdir(parents=True, exist_ok=True)
    (a / "run.py").write_text(RUN_PY)
    (b / "run.py").unlink(missing_ok=True)
    if edited_text is not None:
        (b / "run.py").write_text(edited_text)
    return a, b


def _replace(n: int, new: str) -> str:
    lines = RUN_PY.splitlines(keepends=True)
    lines[n - 1 : n] = [new]
    return "".join(lines)


def test_untouched_passes(task: TaskSpec, tmp_path: Path):
    assert check_diff(task, *_trees(tmp_path)) == ()


@pytest.mark.parametrize("n", [5, 7, 10, 15, 16])
def test_edit_inside_range_passes(task: TaskSpec, tmp_path: Path, n: int):
    assert check_diff(task, *_trees(tmp_path, _replace(n, "X\nY\n"))) == ()


@pytest.mark.parametrize("n", [1, 4, 11, 14, 17, 20])
def test_edit_outside_range_fails(task: TaskSpec, tmp_path: Path, n: int):
    (v,) = check_diff(task, *_trees(tmp_path, _replace(n, "X\n")))
    assert (v.path, v.kind, v.line) == ("run.py", "outside_range", n)


def test_hunk_spanning_range_and_gap_fails(task: TaskSpec, tmp_path: Path):
    lines = RUN_PY.splitlines(keepends=True)
    lines[9:12] = ["X\n"]  # replaces lines 10-12, crossing the range end
    (v,) = check_diff(task, *_trees(tmp_path, "".join(lines)))
    assert v.kind == "outside_range"


def test_growth_does_not_shift_later_ranges(task: TaskSpec, tmp_path: Path):
    lines = RUN_PY.splitlines(keepends=True)
    lines[5:6] = ["a\n"] * 10  # grow the first range by 9 lines
    assert check_diff(task, *_trees(tmp_path, "".join(lines))) == ()
    # line 11 (pristine) sits between the ranges; editing it after growth is still illegal
    lines[5 + 10 + 4] = "tampered\n"
    (v,) = check_diff(task, *_trees(tmp_path, "".join(lines)))
    assert v.kind == "outside_range" and v.line == 11
    # ... and pristine line 15 is still legal, even though it is now line 24 of the file
    lines = RUN_PY.splitlines(keepends=True)
    lines[5:6] = ["a\n"] * 10
    lines[14 + 9] = "ok\n"
    assert check_diff(task, *_trees(tmp_path, "".join(lines))) == ()


def test_deletion_inside_range_passes(task: TaskSpec, tmp_path: Path):
    lines = RUN_PY.splitlines(keepends=True)
    del lines[5:10]
    assert check_diff(task, *_trees(tmp_path, "".join(lines))) == ()


def test_insertion_at_boundaries(task: TaskSpec, tmp_path: Path):
    lines = RUN_PY.splitlines(keepends=True)
    lines.insert(4, "before\n")  # directly before line 5
    assert check_diff(task, *_trees(tmp_path, "".join(lines))) == ()
    lines = RUN_PY.splitlines(keepends=True)
    lines.insert(2, "gap\n")
    (v,) = check_diff(task, *_trees(tmp_path, "".join(lines)))
    assert v.kind == "outside_range"


def test_added_deleted_unlisted(task: TaskSpec, tmp_path: Path):
    a, b = _trees(tmp_path)
    (b / "new.py").write_text("x")
    assert [(v.path, v.kind) for v in check_diff(task, a, b)] == [
        ("new.py", "file_added"),
    ]
    (b / "new.py").unlink()
    (b / "run.py").unlink()
    assert [(v.path, v.kind) for v in check_diff(task, a, b)] == [
        ("run.py", "file_deleted"),
    ]
    a2, b2 = _trees(tmp_path / "u")  # unlisted file present in both, changed
    (a2 / "other.txt").write_text("1")
    (b2 / "other.txt").write_text("2")
    assert [(v.path, v.kind) for v in check_diff(task, a2, b2)] == [
        ("other.txt", "unlisted_file_changed"),
    ]


def test_symlink_tamper(task: TaskSpec, tmp_path: Path):
    a, b = _trees(tmp_path)
    (b / "run.py").unlink()
    (b / "run.py").symlink_to("/etc/hosts")
    (v,) = check_diff(task, a, b)
    assert v.kind == "unlisted_file_changed"


def test_binary_and_crlf_change_detected(task: TaskSpec, tmp_path: Path):
    (v,) = check_diff(task, *_trees(tmp_path, RUN_PY.replace("line1\n", "line1\r\n")))
    assert (v.kind, v.line) == ("outside_range", 1)


def test_materialize_pristine(task: TaskSpec, tmp_path: Path):
    ws = materialize(task, tmp_path / "run", edit=None)
    assert ws.edit is None and ws.root == tmp_path / "run/workspace"
    assert (ws.root / "run.py").read_text() == RUN_PY
    assert not (ws.root / "run.py").is_symlink()
    assert set(manifest(ws.root)) == {"run.py"}


def test_materialize_baseline_edit(task: TaskSpec, tmp_path: Path):
    ws = materialize(task, tmp_path / "run", edit=task.baseline("base1").edit)
    assert ws.edit == "base1"
    assert "LINE6\nextra\n" in (ws.root / "run.py").read_text()
    assert (task.root / "run.py").read_text() == RUN_PY  # upstream untouched
    ws2 = materialize(task, tmp_path / "run2", edit=task.baseline("base1").edit)
    assert ws2.digest == ws.digest
    assert ws.digest != materialize(task, tmp_path / "run3", edit=None).digest


def test_materialize_agent_edit_and_violation(task: TaskSpec, tmp_path: Path):
    bad = tmp_path / "bad.edit.py"
    bad.write_text(
        "import sys, pathlib\np = pathlib.Path(sys.argv[1]) / 'run.py'\n"
        "p.write_text(p.read_text().replace('line1\\n', 'HACK\\n'))\n",
    )
    with pytest.raises(DiffGuardViolation) as ei:
        materialize(task, tmp_path / "run", edit=bad)
    assert ei.value.violations[0].kind == "outside_range"
    good = tmp_path / "good.edit.py"
    good.write_text((task.baseline("base1").edit or tmp_path).read_text())
    assert materialize(task, tmp_path / "run2", edit=good).edit == "agent"


def test_edit_crash_and_timeout(task: TaskSpec, tmp_path: Path):
    crash = tmp_path / "crash.edit.py"
    crash.write_text("raise SystemExit('boom')\n")
    with pytest.raises(EditError, match="boom"):
        materialize(task, tmp_path / "r1", edit=crash)
    slow = tmp_path / "slow.edit.py"
    slow.write_text("import time; time.sleep(30)\n")
    with pytest.raises(EditError, match="timed out"):
        materialize(task, tmp_path / "r2", edit=slow, edit_timeout=0.01)


def test_edit_env_is_sanitised(
    task: TaskSpec,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("MLSBENCH_ROOT", "/secret")
    probe = tmp_path / "probe.edit.py"
    probe.write_text(
        "import os, sys, pathlib\n"
        "assert 'MLSBENCH_ROOT' not in os.environ, 'leak'\n"
        "assert os.environ['PATH'] == '/usr/bin:/bin'\n",
    )
    materialize(task, tmp_path / "run", edit=probe)


def test_edit_adding_file_is_violation(task: TaskSpec, tmp_path: Path):
    add = tmp_path / "add.edit.py"
    add.write_text(
        "import sys, pathlib\n(pathlib.Path(sys.argv[1]) / 'evil.py').write_text('x')\n",
    )
    with pytest.raises(DiffGuardViolation) as ei:
        materialize(task, tmp_path / "run", edit=add)
    assert ei.value.violations[0].kind == "file_added"


def test_double_materialize_refused(task: TaskSpec, tmp_path: Path):
    materialize(task, tmp_path / "run", edit=None)
    with pytest.raises(FileExistsError):
        materialize(task, tmp_path / "run", edit=None)


def test_workspace_changes_flags_modified_deleted_and_symlinked_files(
    task: TaskSpec,
    tmp_path: Path,
):
    ws = materialize(task, tmp_path / "run", edit=None)
    assert workspace_changes(ws) == ()
    (ws.root / "out").mkdir()
    (ws.root / "out" / "result.txt").write_text("outputs are not changes\n")
    assert workspace_changes(ws) == ()
    (ws.root / "run.py").write_text(RUN_PY + "x\n")
    assert workspace_changes(ws) == ("modified:run.py",)
    (ws.root / "run.py").unlink()
    assert workspace_changes(ws) == ("deleted:run.py",)
    (ws.root / "elsewhere").write_text(RUN_PY)
    (ws.root / "run.py").symlink_to(ws.root / "elsewhere")
    assert workspace_changes(ws) == ("modified:run.py",)


def test_workspace_changes_falls_back_to_the_digest_without_a_snapshot(
    task: TaskSpec,
    tmp_path: Path,
):
    ws = replace(materialize(task, tmp_path / "run", edit=None), files=())
    assert workspace_changes(ws) == ()
    (ws.root / "run.py").write_text("changed\n")
    assert workspace_changes(ws) == ("digest:<tree>",)


def test_real_clone_task_without_a_bundle_raises_a_typed_error(tmp_path: Path):
    root = build_task(tmp_path / "up")
    task = load_task("toy-task", root=root)
    (root / "tasks/toy-task/run.py").unlink()  # real tasks do not ship editable files
    # a clone with harbor/ but no bundle for this task
    (root / "harbor/tasks-docker").mkdir(parents=True)
    with pytest.raises(
        MissingBundleError,
        match=r"no Harbor bundle at .*mls-bench__toy-task",
    ):
        materialize(task, tmp_path / "r1", edit=None)
    # a bundle exists but the caller did not pass it: the message names how to fix it
    bundle = root / "harbor/tasks-docker/mls-bench__toy-task"
    (bundle / "tests/meta").mkdir(parents=True)
    with pytest.raises(
        MissingBundleError,
        match=r"call with bundle=.*mls-bench__toy-task",
    ):
        materialize(task, tmp_path / "r2", edit=None)
    # still a FileNotFoundError for callers that caught the old error
    with pytest.raises(FileNotFoundError):
        materialize(task, tmp_path / "r3", edit=None)


def test_synthetic_task_missing_file_keeps_the_plain_error(tmp_path: Path):
    root = build_task(tmp_path / "up")
    task = load_task("toy-task", root=root)
    (root / "tasks/toy-task/run.py").unlink()
    with pytest.raises(FileNotFoundError) as ei:
        materialize(task, tmp_path / "r", edit=None)
    assert not isinstance(ei.value, MissingBundleError)


@pytest.mark.parametrize(
    ("ranges", "n", "legal"),
    [
        (((-1, -1),), 1, True),  # whole file
        (((-1, -1),), 20, True),
        (((15, -1),), 20, True),  # to EOF
        (((15, -1),), 14, False),
    ],
)
def test_whole_file_and_to_eof_ranges(
    task: TaskSpec,
    tmp_path: Path,
    ranges: tuple[tuple[int, int], ...],
    n: int,
    legal: bool,
):
    (ef,) = task.files
    task = replace(task, files=(replace(ef, ranges=ranges),))
    found = check_diff(task, *_trees(tmp_path, _replace(n, "X\n")))
    assert (found == ()) is legal
