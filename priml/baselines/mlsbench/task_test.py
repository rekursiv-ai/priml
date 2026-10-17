"""Tests for strict TaskSpec parsing (synthetic fixtures; the real clone is opt-in below)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import json
import math
import os

import pytest

from priml.baselines.mlsbench.task import (
    MissingFileError,
    TaskSpecError,
    UnknownFieldError,
    load_all,
    load_task,
    parse_time,
    task_names,
)
from priml.baselines.mlsbench.testing import (
    build_task,
    default_config,
    leaderboard_csv,
)


if TYPE_CHECKING:
    from priml.baselines.mlsbench.task import TaskSpec


def test_round_trip(tmp_path: Path):
    t = load_task("toy-task", root=build_task(tmp_path))
    assert (t.name, t.family, t.package, t.seeds) == (
        "toy-task",
        "toy",
        "toy-pkg",
        (42,),
    )
    assert t.seeds_declared
    assert t.files[0].path == "run.py"
    assert t.files[0].ranges == ((5, 10), (15, 16))
    assert t.files[0].read_ranges == ((-1, -1),)
    assert [s.label for s in t.settings] == ["a-x", "b"]
    a, b = t.setting("a-x"), t.setting("b")
    assert (a.time_limit, a.compute, a.group, a.hidden) == (60.0, 0.5, 1, False)
    assert (b.time_limit, b.compute, b.group, b.hidden) == (30.0, 1.0, 2, True)
    assert a.compute_for("H200") == 0.5
    assert b.compute_for("H100") == 1.0
    assert b.compute_for("H200") == 2.0
    assert b.h200 is not None
    assert b.h200.env == {"BATCH_SIZE": "64"}
    assert t.visible_settings() == (a,)
    base = t.baseline("base1")
    assert base.edit == t.root / "edits/base1.edit.py"
    assert (base.cmd, dict(base.env), base.labels) == (None, {}, None)
    assert t.row("base1").values == {"accuracy_a-x": 0.5, "loss_b": 1.25}
    assert t.score_spec is None
    assert t.parser.name == "parser.py"
    assert t.packages == ("toy-pkg",)


def test_absent_seeds_default_to_42(tmp_path: Path):
    cfg = default_config()
    del cfg["seeds"]
    t = load_task("toy-task", root=build_task(tmp_path, config=cfg))
    assert t.seeds == (42,)
    assert not t.seeds_declared


def test_optional_task_flags(tmp_path: Path):
    cfg = default_config()
    cfg.update(
        rigorous_codebase=True,
        use_cuda=True,
        requires_network=True,
        ephemeral_inputs=True,
        guard_exclude=["x/y"],
        hidden_metrics=["piqa_*"],
        leaderboard={"base1": {"accuracy_a-x": 0.5}},
    )
    t = load_task("toy-task", root=build_task(tmp_path, config=cfg))
    assert (
        t.rigorous_codebase,
        t.use_cuda,
        t.requires_network,
        t.ephemeral_inputs,
    ) == (
        True,
        True,
        True,
        True,
    )
    assert (t.guard_exclude, t.hidden_metrics) == (("x/y",), ("piqa_*",))
    assert t.inline_leaderboard == {"base1": {"accuracy_a-x": 0.5}}


def test_baselines_without_an_edit(tmp_path: Path):
    cfg = default_config()
    cfg["baselines"] = {
        "shipped": {},
        "viacmd": {"cmd": "scripts/a.sh", "env": {"K": "v"}, "labels": ["b"]},
    }
    lb = leaderboard_csv(["a"], [("shipped", 42, [1.0]), ("viacmd", 42, [2.0])])
    t = load_task("toy-task", root=build_task(tmp_path, config=cfg, leaderboard=lb))
    assert t.baseline("shipped").edit is None
    v = t.baseline("viacmd")
    assert (v.cmd, dict(v.env), v.labels, v.edit) == (
        "scripts/a.sh",
        {"K": "v"},
        ("b",),
        None,
    )


def test_whole_file_and_to_eof_ranges(tmp_path: Path):
    cfg = default_config()
    cfg["files"] = [
        {
            "filename": "run.py",
            "read": [{"start": 1, "end": 4}],
            "edit": [{"start": -1, "end": -1}],
        },
        {
            "filename": "b.py",
            "read": [{"start": -1, "end": -1}],
            "edit": [{"start": 3, "end": -1}],
        },
        {"filename": "c.py", "read": [{"start": -1, "end": -1}]},
    ]
    t = load_task("toy-task", root=build_task(tmp_path, config=cfg))
    assert [f.ranges for f in t.files] == [((-1, -1),), ((3, -1),), ()]


def test_spread_comes_from_the_seed_rows(tmp_path: Path):
    lb = leaderboard_csv(
        ["accuracy_a-x"],
        [("base1", 1, [0.4]), ("base1", 2, [0.6]), ("base1", "mean", [0.5])],
    )
    t = load_task("toy-task", root=build_task(tmp_path, leaderboard=lb))
    assert t.row("base1").values == {"accuracy_a-x": 0.5}
    assert t.row("base1").spread["accuracy_a-x"] == pytest.approx(0.1414213562)


def test_columns_need_not_end_in_a_setting_label(tmp_path: Path):
    # Upstream parsers name their own columns; some do not end in the label.
    lb = leaderboard_csv(["pass_rate_deepseek"], [("base1", 42, [0.3])])
    t = load_task("toy-task", root=build_task(tmp_path, leaderboard=lb))
    assert t.row("base1").values == {"pass_rate_deepseek": 0.3}


type Mutate = Callable[[dict[str, object]], object]


def _node(c: object, path: tuple[str | int, ...]) -> object:
    for k in path:
        c = (
            cast("dict[str, object]", c)[k]
            if isinstance(k, str)
            else cast("list[object]", c)[k]
        )
    return c


def _d(c: object, *path: str | int) -> dict[str, object]:
    return cast("dict[str, object]", _node(c, path))


def _l(c: object, *path: str | int) -> list[object]:
    return cast("list[object]", _node(c, path))


def _bad(tmp_path: Path, mutate: Mutate) -> None:
    cfg = default_config()
    mutate(cfg)
    load_task("toy-task", root=build_task(tmp_path, config=cfg))


_UNKNOWN_FIELDS: Final[list[Mutate]] = [
    lambda c: c.update(extra=1),
    lambda c: c.update(name="toy-task"),  # Not a real key.
    lambda c: _d(c, "files", 0).update(oops=1),
    lambda c: _d(c, "files", 0, "edit", 0).update(oops=1),
    lambda c: _d(c, "test_cmds", 0).update(oops=1),
    lambda c: _d(c, "test_cmds", 1, "h200").update(oops=1),
    lambda c: _d(c, "baselines", "base1").update(oops=1),
]


@pytest.mark.parametrize("mutate", _UNKNOWN_FIELDS)
def test_unknown_field_anywhere(
    tmp_path: Path,
    mutate: Mutate,
):
    with pytest.raises(UnknownFieldError, match="unknown field"):
        _bad(tmp_path, mutate)


_VIOLATIONS: Final[list[tuple[Mutate, str]]] = [
    (lambda c: _d(c, "test_cmds", 1).update(label="a-x"), "duplicate setting label"),
    (lambda c: _l(c, "test_cmds").clear(), "empty"),
    (lambda c: _d(c, "baselines").clear(), "empty"),
    (
        lambda c: _l(c, "files").append(dict(_d(c, "files", 0))),
        "duplicate",
    ),
    (
        lambda c: _d(c, "files", 0).update(edit=[{"start": 10, "end": 5}]),
        "start <= end",
    ),
    (
        lambda c: _d(c, "files", 0).update(
            edit=[{"start": 1, "end": 5}, {"start": 5, "end": 8}],
        ),
        "overlapping",
    ),
    (
        lambda c: _d(c, "files", 0).update(
            edit=[{"start": 1, "end": -1}, {"start": 5, "end": 8}],
        ),
        "overlapping",
    ),
    (
        lambda c: _d(c, "files", 0).update(
            edit=[{"start": -1, "end": -1}, {"start": 5, "end": 8}],
        ),
        "stand alone",
    ),
    (lambda c: _d(c, "files", 0).update(filename="../x"), "relative"),
    (lambda c: _d(c, "files", 0).update(filename=""), "empty string"),
    (lambda c: _d(c, "test_cmds", 0).update(compute=float("inf")), "not finite"),
    (lambda c: _d(c, "files", 0).update(read=[{"start": 0, "end": 3}]), "start <= end"),
    (lambda c: _d(c, "test_cmds", 0).update(time="1:00"), "H:MM:SS"),
    (lambda c: _d(c, "test_cmds", 0).update(time="0:00:00"), "> 0"),
    (lambda c: _d(c, "test_cmds", 0).update(time=60), "expected"),
    (lambda c: _d(c, "test_cmds", 0).update(compute=-1), "compute"),
    (lambda c: _d(c, "test_cmds", 0).update(hidden="yes"), "expected"),
    (lambda c: _d(c, "test_cmds", 0).update(mem=0), "mem"),
    (lambda c: _d(c, "test_cmds", 1, "h200").update(compute=0), "h200.compute"),
    (lambda c: _d(c, "test_cmds", 0).pop("group"), "missing"),
    (lambda c: _d(c, "test_cmds", 0).pop("package"), "missing"),
    (lambda c: c.pop("files"), "missing"),
    (lambda c: c.update(seeds=[True]), "expected"),
    (lambda c: c.update(seeds=[]), "seeds"),
    (lambda c: c.update(seeds=[1, 1]), "seeds"),
    (lambda c: c.update(use_cuda="yes"), "expected"),
    (lambda c: _d(c, "baselines").update(bad={"labels": ["nope"]}), "not settings"),
    (lambda c: c.update(leaderboard={"ghost": {"m": 1}}), "no baseline"),
]


@pytest.mark.parametrize(("mutate", "msg"), _VIOLATIONS)
def test_schema_violations(
    tmp_path: Path,
    mutate: Mutate,
    msg: str,
):
    with pytest.raises(TaskSpecError, match=msg):
        _bad(tmp_path, mutate)


@pytest.mark.parametrize(
    ("text", "seconds"),
    [
        ("0:00:01", 1.0),
        ("4:00:00", 14400.0),
        ("00:59:00", 3540.0),
        ("17:00:00", 61200.0),
    ],
)
def test_parse_time(text: str, seconds: float):
    assert parse_time(text) == seconds


def test_missing_script_edit_cmd_and_mid_edit(tmp_path: Path):
    root = build_task(tmp_path)
    (root / "tasks/toy-task/scripts/a.sh").unlink()
    with pytest.raises(MissingFileError, match="script"):
        load_task("toy-task", root=root)
    (root / "tasks/toy-task/scripts/a.sh").write_text("")
    (root / "tasks/toy-task/edits/base1.edit.py").unlink()
    with pytest.raises(MissingFileError, match="edit file"):
        load_task("toy-task", root=root)
    (root / "tasks/toy-task/edits/base1.edit.py").write_text("")
    cfg = cast(
        "dict[str, object]",
        json.loads((root / "tasks/toy-task/config.json").read_text()),
    )
    cfg["mid_edit"] = "edits/mid_edit.py"
    (root / "tasks/toy-task/config.json").write_text(json.dumps(cfg))
    with pytest.raises(MissingFileError, match="mid_edit"):
        load_task("toy-task", root=root)
    del cfg["mid_edit"]
    _d(cfg, "baselines")["base1"] = {"cmd": "scripts/none.sh"}
    (root / "tasks/toy-task/config.json").write_text(json.dumps(cfg))
    with pytest.raises(MissingFileError, match="cmd script"):
        load_task("toy-task", root=root)


def test_editable_files_need_not_exist_in_the_task_dir(tmp_path: Path):
    # They live in the package codebase (the image), not in tasks/<name>/.
    root = build_task(tmp_path)
    (root / "tasks/toy-task/run.py").unlink()
    assert load_task("toy-task", root=root).files[0].path == "run.py"


@pytest.mark.parametrize(
    ("lb", "msg"),
    [
        ("name,acc\nbase1,1\n", "header"),
        (leaderboard_csv(["a"], [("other", 42, [1.0])]), "no leaderboard row"),
        (leaderboard_csv(["a", "a"], [("base1", 42, [1.0, 1.0])]), "duplicate columns"),
    ],
)
def test_leaderboard_problems(tmp_path: Path, lb: str, msg: str):
    with pytest.raises(TaskSpecError, match=msg):
        load_task("toy-task", root=build_task(tmp_path, leaderboard=lb))


def test_blank_leaderboard_lines_and_comments_are_skipped(tmp_path: Path):
    cfg = default_config()
    _d(cfg, "files", 0).update(_comment="why")
    _d(cfg, "files", 0, "edit", 0).update(_comment="why")
    lb = leaderboard_csv(["a"], [("base1", 42, [1.0])]) + "\n"
    task = load_task("toy-task", root=build_task(tmp_path, config=cfg, leaderboard=lb))
    assert task.leaderboard[0].values == {"a": 1.0}


def test_missing_leaderboard_and_parser(tmp_path: Path):
    root = build_task(tmp_path)
    (root / "tasks/toy-task/parser.py").unlink()
    with pytest.raises(MissingFileError, match=r"parser\.py"):
        load_task("toy-task", root=root)
    (root / "tasks/toy-task/parser.py").write_text("")
    (root / "tasks/toy-task/leaderboard.csv").unlink()
    with pytest.raises(MissingFileError, match=r"leaderboard\.csv"):
        load_task("toy-task", root=root)


def test_a_non_object_config_is_refused(tmp_path: Path):
    root = build_task(tmp_path)
    (root / "tasks/toy-task/config.json").write_text("[]")
    with pytest.raises(TaskSpecError, match="expected object"):
        load_task("toy-task", root=root)


def test_bad_json_and_missing_task(tmp_path: Path):
    root = build_task(tmp_path)
    (root / "tasks/toy-task/config.json").write_text("{nope")
    with pytest.raises(TaskSpecError, match="invalid json"):
        load_task("toy-task", root=root)
    with pytest.raises(MissingFileError):
        load_task("absent", root=root)


def test_unknown_lookup(tmp_path: Path):
    t = load_task("toy-task", root=build_task(tmp_path))
    for f in (t.setting, t.baseline, t.row):
        with pytest.raises(TaskSpecError):
            f("nope")


def test_task_names_and_load_all(tmp_path: Path):
    root = build_task(tmp_path)
    assert task_names(root) == ("toy-task",)
    assert [t.name for t in load_all(root)] == ["toy-task"]
    with pytest.raises(MissingFileError):
        task_names(tmp_path / "empty")


def test_config_is_json(tmp_path: Path):
    root = build_task(tmp_path)
    assert json.loads((root / "tasks/toy-task/config.json").read_text())["seeds"] == [
        42,
    ]


# -- opt-in: the real pinned upstream clone ---------------------------------

UPSTREAM_COMMIT: Final = "80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c"
EXPECTED: Final = {
    "tasks": 140,
    "families": 26,
    "settings": 424,
    "baselines": 480,
    "hidden": 135,
}

_upstream_root = os.environ.get("MLSBENCH_ROOT", "")
upstream = pytest.mark.skipif(
    not (_upstream_root and Path(_upstream_root, "tasks").is_dir()),
    reason="set MLSBENCH_ROOT to a read-only MLS-Bench clone",
)


@pytest.fixture(scope="module")
def real() -> tuple[TaskSpec, ...]:
    return load_all(Path(_upstream_root))


@pytest.mark.integration
@upstream
def test_every_real_task_parses_strictly_with_the_plans_counts(
    real: tuple[TaskSpec, ...],
):
    assert len(real) == EXPECTED["tasks"]
    assert len({t.family for t in real}) == EXPECTED["families"]
    assert sum(len(t.settings) for t in real) == EXPECTED["settings"]
    assert sum(len(t.baselines) for t in real) == EXPECTED["baselines"]
    assert sum(s.hidden for t in real for s in t.settings) == EXPECTED["hidden"]
    assert all(t.name == t.root.name for t in real)


@pytest.mark.integration
@upstream
def test_real_seed_census(real: tuple[TaskSpec, ...]):
    multi = [t for t in real if len(t.seeds) > 1]
    assert len(multi) == 13  # "127 tasks use one seed; 13 use three or five"
    assert sum(1 for t in real if not t.seeds_declared) == 27


@pytest.mark.integration
@upstream
def test_real_leaderboards_cover_every_baseline_and_are_finite(
    real: tuple[TaskSpec, ...],
):
    for t in real:
        for b in t.baselines:
            row = t.row(b.name)
            assert row.values, (t.name, b.name)
            assert all(math.isfinite(v) for v in row.values.values())
        for row in t.leaderboard:
            if len(row.seeds) > 1:
                assert row.spread, (t.name, row.name)


@pytest.mark.integration
@upstream
def test_real_packages_agree_with_harbors(real: tuple[TaskSpec, ...]):
    root = Path(_upstream_root)
    for t in real:
        meta = root / "harbor/tasks-docker" / f"mls-bench__{t.name}" / "tests/meta"
        if (meta / "package").is_file():
            assert (meta / "package").read_text().strip() == t.package, t.name


@pytest.mark.integration
@upstream
def test_real_dllm_task_matches_the_briefing(real: tuple[TaskSpec, ...]):
    t = next(x for x in real if x.name == "llm-dllm-demask-strategy")
    assert [(s.label, s.time_limit, s.hidden) for s in t.settings] == [
        ("llada-math", 14400.0, False),
        ("llada-humaneval", 6300.0, False),
        ("dream-text", 6300.0, True),
    ]
    assert [b.name for b in t.baselines] == [
        "topk_margin",
        "confidence_greedy",
        "klass",
    ]
    assert t.baseline("klass").edit == t.root / "edits/klass_kl.edit.py"
    assert [(f.path, f.ranges) for f in t.files] == [
        ("LLaDA/custom_demask_eval.py", ((59, 151),)),
    ]
    assert t.seeds == (42,)
    assert t.package == "LLaDA"
    assert t.row("topk_margin").values["accuracy_llada-math"] == 0.322


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
