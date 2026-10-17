"""Small synthetic MLS-Bench-shaped tasks, for tests and ``exp_smoke``.

Copies nothing from upstream; the layout is the one documented in
:mod:`priml.baselines.mlsbench.task`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import json


if TYPE_CHECKING:
    from pathlib import Path


RUN_PY: Final = "".join(f"line{i}\n" for i in range(1, 21))  # 20 lines.


def default_config() -> dict[str, object]:
    """Return a config in the schema upstream really uses (see :mod:`...task`).

    Returns:
      config: A two-setting, one-baseline ``config.json`` as plain data.

    """
    return {
        "allow_create": False,
        "seeds": [42],
        "files": [
            {
                "filename": "run.py",
                "read": [{"start": -1, "end": -1}],
                "edit": [{"start": 5, "end": 10}, {"start": 15, "end": 16}],
            },
        ],
        "test_cmds": [
            {
                "cmd": "scripts/a.sh",
                "label": "a-x",
                "group": 1,
                "compute": 0.5,
                "time": "0:01:00",
                "package": "toy-pkg",
            },
            {
                "cmd": "scripts/b.sh",
                "label": "b",
                "group": 2,
                "compute": 1.0,
                "time": "0:00:30",
                "package": "toy-pkg",
                "hidden": True,
                "h200": {"compute": 2.0, "env": {"BATCH_SIZE": "64"}},
            },
        ],
        "baselines": {"base1": {"edit_ops": "edits/base1.edit.py"}},
    }


LB_HEADER: Final = "timestamp,model,is_final,seed,"


def leaderboard_csv(
    columns: list[str],
    rows: list[tuple[str, str | int, list[float | None]]],
    *,
    final: bool = True,
) -> str:
    """Render upstream's long ``leaderboard.csv``: ``(baseline, seed|"mean", cells)`` rows."""
    out = LB_HEADER + ",".join(columns) + "\n"
    for name, seed, cells in rows:
        vals = ",".join("" if c is None else repr(c) for c in cells)
        out += f"{'2026-01-01T00:00:00+00:00'},baseline:{name},{'true' if final else 'false'},{seed},{vals}\n"
    return out


EDIT_OK: Final = """import sys, pathlib
p = pathlib.Path(sys.argv[1]) / "run.py"
p.write_text(p.read_text().replace("line6\\n", "LINE6\\nextra\\n"))
"""


KV_PARSER: Final = """class _Result:
    def __init__(self, feedback, metrics):
        self.feedback = feedback
        self.metrics = metrics


class Parser:
    def parse(self, cmd_label, raw_output):
        out = {}
        for line in raw_output.splitlines():
            if "=" in line:
                k, _, v = line.partition("=")
                out[k.strip() + "_" + cmd_label] = float(v)
        return _Result(raw_output, out)
"""
"""Parser for logs of ``key=value`` lines, standing in for upstream's ``parser.py``.

Same contract as upstream: ``Parser().parse(cmd_label, raw_output)`` returns an
object with ``metrics`` already keyed ``<metric>_<label>``.
"""

SYNTH_TASK: Final = "synth-three-setting"
"""Name of the neutral fixture task; deliberately absent from ``registry.json``."""
SYNTH_LABELS: Final = ("set-a", "set-b", "set-c")
SYNTH_BASELINES: Final = ("base_a", "base_b", "base_c")


def synth_config() -> dict[str, object]:
    """Return a config with three settings and three baselines (synthetic).

    Returns:
      config: A three-setting, three-baseline ``config.json`` as plain data.

    """
    cmds = [
        {
            "cmd": f"scripts/{lab}.sh",
            "label": lab,
            "group": 1,
            "compute": 1.0,
            "time": "0:01:40",
            "package": "synth-pkg",
        }
        for lab in SYNTH_LABELS
    ]
    return {
        "allow_create": False,
        "seeds": [42],
        "files": [
            {
                "filename": "run.py",
                "read": [{"start": -1, "end": -1}],
                "edit": [{"start": 5, "end": 10}],
            },
        ],
        "test_cmds": cmds,
        "baselines": {n: {"edit_ops": f"edits/{n}.edit.py"} for n in SYNTH_BASELINES},
    }


def build_synth_task(root: Path) -> Path:
    """Build a synthetic neutral three-setting task under ``root``.

    Args:
      root: Directory to hold the synthetic clone.

    Returns:
      root: ``root``, to pass as ``load_task(root=...)``.

    """
    cfg = synth_config()
    cols = ["accuracy_set-a", "avg_steps_set-a", "pass_at_1_set-b"]
    lb = leaderboard_csv(cols, [(n, 42, [0.322, 256.0, 0.39]) for n in SYNTH_BASELINES])
    build_task(
        root,
        config=cfg,
        leaderboard=lb,
        parser_source=KV_PARSER,
        dirname=SYNTH_TASK,
    )
    d = root / "tasks" / SYNTH_TASK
    for n in SYNTH_BASELINES:
        (d / "edits" / f"{n}.edit.py").write_text(EDIT_OK)
    for lab in SYNTH_LABELS:
        (d / "scripts" / f"{lab}.sh").write_text("echo x\n")
    return root


def build_task(
    root: Path,
    *,
    config: dict[str, object] | None = None,
    leaderboard: str | None = None,
    edit_source: str = EDIT_OK,
    parser_source: str = "def parse(log_text, setting):\n    return {}\n",
    dirname: str = "toy-task",
) -> Path:
    """Write ``<root>/tasks/toy-task``; returns ``root`` (pass as ``load_task(root=...)``).

    Args:
      root: Directory to hold the synthetic clone.
      config: ``config.json`` contents; ``None`` is :func:`default_config`.
      leaderboard: ``leaderboard.csv`` text; ``None`` is one ``base1`` row.
      edit_source: Source of ``edits/base1.edit.py``.
      parser_source: Source of ``parser.py``.
      dirname: Task directory name.

    Returns:
      root: ``root``, to pass as ``load_task(root=...)``.

    """
    d = root / "tasks" / dirname
    (d / "scripts").mkdir(parents=True)
    (d / "edits").mkdir()
    (d / "run.py").write_text(RUN_PY)
    (d / "scripts" / "a.sh").write_text("echo a\n")
    (d / "scripts" / "b.sh").write_text("echo b\n")
    (d / "edits" / "base1.edit.py").write_text(edit_source)
    (d / "parser.py").write_text(parser_source)
    if leaderboard is None:
        leaderboard = leaderboard_csv(
            ["accuracy_a-x", "loss_b"],
            [("base1", 42, [0.5, 1.25])],
        )
    (d / "leaderboard.csv").write_text(leaderboard)
    (d / "config.json").write_text(
        json.dumps(config if config is not None else default_config()),
    )
    return root
