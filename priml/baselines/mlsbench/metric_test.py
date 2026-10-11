"""Tests for the suite metric: a real TrainLoop pass with a fake executor."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import json
import os
import shutil
import sys

import pytest
import torch

from priml.baselines.mlsbench import leaderboard
from priml.baselines.mlsbench.executor import FakeExecutor, Outcome, RunRequest
from priml.baselines.mlsbench.experiments import (
    SMOKE_LOGS,
    MlsbenchEval,
    candidate_experiment,
    exp_smoke,
    mlsbench_experiment,
)
from priml.baselines.mlsbench.leaderboard import compare, read_metrics_file
from priml.baselines.mlsbench.metric import (
    MlsbenchSuite,
    ParseError,
    data_mounts,
    elapsed_columns,
    expand_env,
    expected_columns,
    inputgen_dir,
    package_env,
    run_parser,
    time_budget,
)
from priml.baselines.mlsbench.task import load_task
from priml.baselines.mlsbench.testing import (
    SYNTH_TASK,
    build_synth_task,
    build_task,
    default_config,
    leaderboard_csv,
)
from priml.baselines.mlsbench.workspace_harbor import EvalScriptsError
from priml.baselines.mlsbench.workspace_harbor_test import make_bundle
from priml.train.results import read_metrics


def _read_json(path: Path) -> dict[str, object]:
    return cast("dict[str, object]", json.loads(path.read_text()))


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return build_synth_task(tmp_path / "up")


def _cfg(
    tmp_path: Path,
    root: Path,
    fake: FakeExecutor.Config,
    baseline: str = "base_a",
) -> MlsbenchEval:
    cfg = mlsbench_experiment(SYNTH_TASK, baseline, executor=fake)
    cfg.base_dir = tmp_path / "out"
    cfg.dataset.mlsbench_root = str(root)
    suite = cfg.metrics_eval[""]
    assert isinstance(suite, MlsbenchSuite.Config)
    suite.mlsbench_root = str(root)
    return cfg


def _run(cfg: MlsbenchEval, tmp_path: Path) -> tuple[dict[str, float], MlsbenchSuite]:
    loop = cfg.make()
    loop.run()
    path = tmp_path / "out/runs/mlsbench" / cfg.experiment_name / "metrics.json"
    suite = loop.metrics_eval[""]
    assert isinstance(suite, MlsbenchSuite)
    return dict(read_metrics(path)), suite


def _fake(**kw: object) -> FakeExecutor.Config:
    fake = FakeExecutor.Config()
    fake.outputs = dict(SMOKE_LOGS)
    for k, v in kw.items():
        setattr(fake, k, v)
    return fake


def test_end_to_end_metrics_use_leaderboard_column_names(
    tmp_path: Path,
    root: Path,
) -> None:
    metrics, suite = _run(_cfg(tmp_path, root, _fake()), tmp_path)
    assert metrics["eval/accuracy_set-a"] == 0.322
    assert metrics["eval/avg_steps_set-a"] == 256.0
    assert metrics["eval/pass_at_1_set-b"] == 0.3902
    assert metrics["eval/gen_ppl_set-c"] == 237.0499
    assert [r.label for r in suite.runs] == [
        "set-a",
        "set-b",
        "set-c",
    ]
    assert all(r.error is None for r in suite.runs)


def test_settings_json_records_every_setting(tmp_path: Path, root: Path) -> None:
    cfg = _cfg(tmp_path, root, _fake())
    _run(cfg, tmp_path)
    rec = (
        tmp_path / "out/runs/mlsbench" / cfg.experiment_name / "runs/settings.json"
    ).read_text()
    assert "set-a" in rec
    assert '"outcome": "ok"' in rec


def test_run_reaches_a_leaderboard_verdict(tmp_path: Path, root: Path) -> None:
    cfg = _cfg(tmp_path, root, _fake())
    _run(cfg, tmp_path)
    run_dir = tmp_path / "out/runs/mlsbench" / cfg.experiment_name
    spec = load_task(SYNTH_TASK, root=root)
    report = compare(spec, "base_a", [read_metrics_file(run_dir)])
    by = {c.column: c for c in report.comparisons}
    assert by["accuracy_set-a"].verdict == "ok"
    assert by["avg_steps_set-a"].verdict == "ok"
    # 0.3902 vs 0.39 published, within the 2-point accuracy tolerance
    assert by["pass_at_1_set-b"].verdict == "ok"
    # Measured columns with no published cell (set-c) are shown as unverified
    # but are not the row's to judge: the published cells all reproduce.
    assert by["gen_ppl_set-c"].verdict == "unverified"
    assert report.verdict == "ok"


def test_a_wrong_number_diverges(tmp_path: Path, root: Path) -> None:
    fake = _fake()
    fake.outputs = {**SMOKE_LOGS, "set-a": "accuracy=0.5\navg_steps=256.0\n"}
    cfg = _cfg(tmp_path, root, fake)
    _run(cfg, tmp_path)
    run_dir = tmp_path / "out/runs/mlsbench" / cfg.experiment_name
    report = compare(
        load_task(SYNTH_TASK, root=root),
        "base_a",
        [read_metrics_file(run_dir)],
    )
    assert report.verdict == "diverged"


@pytest.mark.parametrize(
    ("outcome", "error"),
    [(Outcome.TIMEOUT, "timeout"), (Outcome.CRASHED, "crashed:1")],
)
def test_timeout_and_crash_leave_the_key_absent(
    tmp_path: Path,
    root: Path,
    outcome: Outcome,
    error: str,
) -> None:
    metrics, suite = _run(
        _cfg(tmp_path, root, _fake(outcomes={"set-a": outcome})),
        tmp_path,
    )
    assert "eval/accuracy_set-a" not in metrics
    assert "eval/avg_steps_set-a" not in metrics
    assert "eval/pass_at_1_set-b" in metrics
    assert suite.runs[0].error == error


def test_missing_metric_is_recorded_and_absent(tmp_path: Path, root: Path) -> None:
    fake = _fake()
    fake.outputs = {**SMOKE_LOGS, "set-a": "avg_steps=256.0\n"}
    metrics, suite = _run(_cfg(tmp_path, root, fake), tmp_path)
    assert "eval/accuracy_set-a" not in metrics
    assert metrics["eval/avg_steps_set-a"] == 256.0
    assert suite.runs[0].error == "missing_metric:accuracy_set-a"


@pytest.mark.parametrize("bad", ["accuracy=nan\n", "accuracy=inf\n", "junk\nx=y\n"])
def test_parser_failure_is_recorded_and_absent(
    tmp_path: Path,
    root: Path,
    bad: str,
) -> None:
    fake = _fake()
    fake.outputs = {**SMOKE_LOGS, "set-a": bad}
    metrics, suite = _run(_cfg(tmp_path, root, fake), tmp_path)
    assert not any(k.endswith("_set-a") for k in metrics)
    err = suite.runs[0].error
    assert err is not None
    assert err.startswith(("parse:", "missing_metric:"))
    if bad != "junk\nx=y\n":  # a non-finite cell is missing, not a parse failure
        assert err == "missing_metric:accuracy_set-a"


def test_a_non_finite_metric_keeps_the_settings_other_metrics(
    tmp_path: Path,
    root: Path,
) -> None:
    fake = _fake()
    fake.outputs = {**SMOKE_LOGS, "set-a": "accuracy=nan\navg_steps=256.0\n"}
    metrics, suite = _run(_cfg(tmp_path, root, fake), tmp_path)
    assert "eval/accuracy_set-a" not in metrics
    assert metrics["eval/avg_steps_set-a"] == 256.0
    assert suite.runs[0].error == "missing_metric:accuracy_set-a"


def test_a_raising_parser_is_a_parse_error(tmp_path: Path) -> None:
    src = (
        "class Parser:\n"
        "    def parse(self, cmd_label, raw_output):\n"
        "        raise RuntimeError('boom')\n"
    )
    task = load_task("toy-task", root=build_task(tmp_path / "up", parser_source=src))
    with pytest.raises(ParseError, match="boom"):
        run_parser(task, "", "a-x")


def test_a_parser_without_a_parser_class_is_rejected(tmp_path: Path) -> None:
    r = build_task(tmp_path / "up", parser_source="x = 1\n")
    with pytest.raises(ParseError, match="no class Parser"):
        run_parser(load_task("toy-task", root=r), "", "a-x")


def test_a_parser_without_parse_is_rejected(tmp_path: Path) -> None:
    r = build_task(tmp_path / "up", parser_source="class Parser:\n    pass\n")
    with pytest.raises(ParseError, match="no parse"):
        run_parser(load_task("toy-task", root=r), "", "a-x")


def test_a_result_without_metrics_is_rejected(tmp_path: Path) -> None:
    src = (
        "class Parser:\n    def parse(self, cmd_label, raw_output):\n        return 3\n"
    )
    r = build_task(tmp_path / "up", parser_source=src)
    with pytest.raises(ParseError, match="no metrics mapping"):
        run_parser(load_task("toy-task", root=r), "", "a-x")


def test_the_parser_gets_label_then_log_and_its_keys_pass_through(
    tmp_path: Path,
) -> None:
    src = (
        "class Parser:\n"
        "    def parse(self, cmd_label, raw_output):\n"
        "        class R:\n"
        "            metrics = {'acc_' + cmd_label: float(raw_output)}\n"
        "        return R()\n"
    )
    task = load_task("toy-task", root=build_task(tmp_path / "up", parser_source=src))
    assert run_parser(task, "0.25", "a-x") == {"acc_a-x": 0.25}


def test_a_non_finite_value_drops_that_column_only(tmp_path: Path) -> None:
    src = (
        "class Parser:\n"
        "    def parse(self, cmd_label, raw_output):\n"
        "        class R:\n"
        "            metrics = {'acc_' + cmd_label: 0.5, 'loss_' + cmd_label: float('nan'),\n"
        "                       'ppl_' + cmd_label: float('inf')}\n"
        "        return R()\n"
    )
    task = load_task("toy-task", root=build_task(tmp_path / "up", parser_source=src))
    assert run_parser(task, "", "a-x") == {"acc_a-x": 0.5}


def test_the_parser_may_import_from_the_clone_src_dir(tmp_path: Path) -> None:
    src = (
        "from toy_upstream_helper import VALUE\n"
        "class Parser:\n"
        "    def parse(self, cmd_label, raw_output):\n"
        "        class R:\n"
        "            metrics = {'v_' + cmd_label: VALUE}\n"
        "        return R()\n"
    )
    root = build_task(tmp_path / "up", parser_source=src)
    (root / "src").mkdir()
    (root / "src" / "toy_upstream_helper.py").write_text("VALUE = 7.0\n")
    before = list(sys.path)
    assert run_parser(load_task("toy-task", root=root), "", "b") == {"v_b": 7.0}
    assert sys.path == before  # the clone's src is not left on the path


def test_the_parser_may_import_host_modules_from_the_bundle_meta(
    tmp_path: Path,
) -> None:
    src = (
        "import dgp\n"
        "class Parser:\n"
        "    def parse(self, cmd_label, raw_output):\n"
        "        class R:\n"
        "            metrics = {'r2_' + cmd_label: dgp.truth(cmd_label)}\n"
        "        return R()\n"
    )
    root = build_task(tmp_path / "up", parser_source=src)
    task = load_task("toy-task", root=root)
    with pytest.raises(ParseError, match="No module named 'dgp'"):
        run_parser(task, "", "b")
    meta = tmp_path / "bundle" / "tests" / "meta"
    meta.mkdir(parents=True)
    (meta / "dgp.py").write_text("def truth(label):\n    return 0.5\n")
    before = list(sys.path)
    assert run_parser(task, "", "b", meta=meta) == {"r2_b": 0.5}
    assert sys.path == before
    assert "dgp" not in sys.modules  # another task's dgp must not be reused


def test_the_parser_may_import_inside_parse(tmp_path: Path) -> None:
    # optimization-variance-reduction imports dgp lazily, from within parse()
    src = (
        "class Parser:\n"
        "    def parse(self, cmd_label, raw_output):\n"
        "        import dgp\n"
        "        import toy_lazy_helper\n"
        "        class R:\n"
        "            metrics = {'r2_' + cmd_label: dgp.truth(cmd_label),\n"
        "                       'v_' + cmd_label: toy_lazy_helper.VALUE}\n"
        "        return R()\n"
    )
    root = build_task(tmp_path / "up", parser_source=src)
    (root / "src").mkdir()
    (root / "src" / "toy_lazy_helper.py").write_text("VALUE = 7.0\n")
    meta = tmp_path / "bundle" / "tests" / "meta"
    meta.mkdir(parents=True)
    (meta / "dgp.py").write_text("def truth(label):\n    return 0.5\n")
    before = list(sys.path)
    task = load_task("toy-task", root=root)
    assert run_parser(task, "", "b", meta=meta) == {"r2_b": 0.5, "v_b": 7.0}
    assert sys.path == before
    assert "dgp" not in sys.modules


def test_elapsed_columns_are_the_runners_not_the_parsers(tmp_path: Path) -> None:
    lb = leaderboard_csv(
        ["accuracy_a-x", "elapsed_a-x", "loss_b"],
        [("base1", 42, [0.5, 12.0, 1.25])],
    )
    task = load_task("toy-task", root=build_task(tmp_path / "up", leaderboard=lb))
    assert expected_columns(task, "a-x") == ("accuracy_a-x",)
    assert elapsed_columns(task, "a-x") == ("elapsed_a-x",)
    assert elapsed_columns(task, "b") == ()


def test_a_baseline_cmd_replaces_the_script_and_the_package_follows_the_setting(
    tmp_path: Path,
) -> None:
    cfg = default_config()
    cfg["test_cmds"][1]["package"] = "other-pkg"
    cfg["baselines"] = {"viacmd": {"cmd": "scripts/alt.sh", "labels": ["b"]}}
    root = build_task(
        tmp_path / "up",
        config=cfg,
        leaderboard=leaderboard_csv(["accuracy_a-x"], [("viacmd", 42, [0.5])]),
    )
    (root / "tasks/toy-task/scripts/alt.sh").write_text("echo alt\n")
    fake = FakeExecutor.Config()
    fake.outputs = {"a-x": "accuracy=0.5\n", "b": "loss=1.0\n"}
    suite = MlsbenchSuite.Config(
        task="toy-task",
        baseline="viacmd",
        mlsbench_root=str(root),
        working_dir=tmp_path / "w",
        executor=fake,
    ).make()
    for label in ("a-x", "b"):
        suite.update(torch.empty(0), setting=label)
    assert isinstance(suite.executor, FakeExecutor)
    seen = {
        c.setting.label: (c.setting.script, c.task.package)
        for c in suite.executor.calls
    }
    assert seen == {
        "a-x": ("scripts/a.sh", "toy-pkg"),
        "b": ("scripts/alt.sh", "other-pkg"),
    }
    for label, script, cmd in (
        ("a-x", "scripts/a.sh", None),
        ("b", "scripts/alt.sh", "scripts/alt.sh"),
    ):
        rec = _read_json(tmp_path / f"w/logs/{label}.request.json")
        assert (rec["script"], rec["cmd"], rec["baseline"]) == (script, cmd, "viacmd")
        assert (rec["SEED"], rec["ENV"], rec["OUTPUT_DIR"]) == ("42", label, "/output")
        assert str(rec["output_dir_host"]).endswith(f"logs/{label}.out")


def test_the_parser_runs_once_per_setting(tmp_path: Path, root: Path) -> None:
    fake = _fake()
    cfg = _cfg(tmp_path, root, fake).copy_tree().finalize()
    suite_cfg = cfg.metrics_eval[""]
    assert isinstance(suite_cfg, MlsbenchSuite.Config)
    suite = suite_cfg.make()
    for _ in range(2):
        suite.update(torch.empty(0), setting="set-a")
    assert len(suite.runs) == 1
    assert isinstance(suite.executor, FakeExecutor)
    assert len(suite.executor.calls) == 1
    assert suite.executor.calls[0].seed == 42


def test_baseline_edit_is_applied_to_the_workspace(tmp_path: Path, root: Path) -> None:
    cfg = _cfg(tmp_path, root, _fake())
    _run(cfg, tmp_path)
    ws = tmp_path / "out/runs/mlsbench" / cfg.experiment_name / "runs/workspace/run.py"
    assert "LINE6" in ws.read_text()


def test_pristine_run_applies_no_edit(tmp_path: Path, root: Path) -> None:
    cfg = _cfg(tmp_path, root, _fake())
    suite = cfg.metrics_eval[""]
    assert isinstance(suite, MlsbenchSuite.Config)
    suite.baseline = ""
    _run(cfg, tmp_path)
    ws = tmp_path / "out/runs/mlsbench" / cfg.experiment_name / "runs/workspace/run.py"
    assert "LINE6" not in ws.read_text()


def test_a_candidate_edit_runs_and_is_judged_like_a_row(
    tmp_path: Path,
    root: Path,
    capfd: pytest.CaptureFixture[str],
) -> None:
    edits = tmp_path / "out" / "edits"
    edits.mkdir(parents=True)
    task = load_task(SYNTH_TASK, root=root)
    edit = task.baseline("base_a").edit
    assert edit is not None
    _ = shutil.copy(edit, edits / "my-idea.edit.py")
    cfg = candidate_experiment(
        SYNTH_TASK,
        "my-idea",
        "/edits/my-idea.edit.py",
        executor=_fake(),
    )
    cfg.base_dir = tmp_path / "out"
    cfg.dataset.mlsbench_root = str(root)
    suite = cfg.metrics_eval[""]
    assert isinstance(suite, MlsbenchSuite.Config)
    suite.mlsbench_root = str(root)
    assert cfg.experiment_name == f"{SYNTH_TASK}/my-idea/seed42"
    metrics, _ = _run(cfg, tmp_path)
    assert metrics["eval/accuracy_set-a"] == 0.322
    ws = tmp_path / "out/runs/mlsbench" / cfg.experiment_name / "runs/workspace/run.py"
    assert "LINE6" in ws.read_text()
    argv = ["--task", SYNTH_TASK, "--row", "base_a", "--candidate", "my-idea"]
    code = leaderboard.main(
        [*argv, "--runs", str(tmp_path / "out"), "--root", str(root)],
    )
    assert "0.3220" in capfd.readouterr().out
    assert code == 0


def test_config_validation() -> None:
    with pytest.raises(ValueError, match="task is required"):
        MlsbenchSuite.Config().finalize()
    with pytest.raises(ValueError, match="mutually exclusive"):
        MlsbenchSuite.Config(task="t", baseline="b", edit_path="e.py").finalize()


def test_state_dict_round_trips(tmp_path: Path, root: Path) -> None:
    _, suite = _run(_cfg(tmp_path, root, _fake()), tmp_path)
    state = suite.state_dict()
    other = MlsbenchSuite(MlsbenchSuite.Config(task="t"))
    other.load_state_dict(state)
    assert other.runs == suite.runs


def test_expected_columns_prefer_the_longest_label(tmp_path: Path) -> None:
    task = load_task("toy-task", root=build_task(tmp_path / "up"))
    assert expected_columns(task, "a-x") == ("accuracy_a-x",)
    assert expected_columns(task, "b") == ("loss_b",)


def test_time_budget_sums_limits_with_slack(tmp_path: Path) -> None:
    task = load_task("toy-task", root=build_task(tmp_path / "up"))
    assert time_budget(task) == pytest.approx(90 * 1.1)
    assert time_budget(task, ("b",)) == pytest.approx(30 * 1.1)


def test_smoke_experiment_runs_end_to_end(tmp_path: Path) -> None:
    from priml.baselines.mlsbench.scripts import prepare_data as pd  # noqa: PLC0415

    # The documented flow: ``prepare_data --smoke`` then ``python -m priml``.
    assert pd.main(["--smoke", str(tmp_path / "out/mlsbench_smoke")]) == 0
    cfg = exp_smoke()
    cfg.base_dir = tmp_path / "out"
    cfg.make().run()
    metrics = read_metrics(tmp_path / "out/runs/mlsbench/exp_smoke/metrics.json")
    assert metrics["eval/accuracy_set-a"] == 0.322


# -- opt-in: the real upstream parser --------------------------------------
# Upstream ships no sample logs, so the log below is built from the parser's
# own documented shape (TRAIN_METRICS progress lines, one TEST_METRICS line).

_CLONE = os.environ.get("MLSBENCH_ROOT", "")


@pytest.mark.integration
@pytest.mark.skipif(not _CLONE, reason="set MLSBENCH_ROOT to a read-only clone")
def test_a_real_upstream_parser_runs_through_run_parser() -> None:
    task = load_task("llm-dllm-demask-strategy", root=Path(_CLONE))
    log = (
        "noise\nTRAIN_METRICS step=1 total=2\n"
        "TEST_METRICS accuracy=0.322 avg_steps=256.0 n_samples=500\n"
    )
    assert run_parser(task, log, "llada-math") == {
        "accuracy_llada-math": 0.322,
        "avg_steps_llada-math": 256.0,
    }
    assert run_parser(task, "crash\n", "llada-math") == {}
    columns = {c for r in task.leaderboard for c in r.values}
    assert set(expected_columns(task, "llada-math")) <= columns
    for setting in task.settings:
        assert expected_columns(task, setting.label), setting.label


def _bundled(
    tmp_path: Path,
    root: Path,
    fake: FakeExecutor.Config,
) -> tuple[MlsbenchEval, Path]:
    """Build a run config with a synthetic Harbor bundle mirroring the task's scripts."""
    bundle = make_bundle(tmp_path / "b", declared="run.py", ranges=[[5, 10]])
    ref = bundle / "tests/eval/scripts"
    for p in (root / "tasks" / SYNTH_TASK / "scripts").glob("*.sh"):
        (ref / p.name).write_bytes(p.read_bytes())
    (ref / "a.sh").unlink()
    cfg = _cfg(tmp_path, root, fake)
    suite = cfg.metrics_eval[""]
    assert isinstance(suite, MlsbenchSuite.Config)
    suite.bundle = str(bundle)
    return cfg, ref


def test_an_edited_eval_script_is_refused_before_any_setting_runs(
    tmp_path: Path,
    root: Path,
) -> None:
    fake = _fake()
    cfg, ref = _bundled(tmp_path, root, fake)
    (ref / "set-a.sh").write_text("echo tampered\n")
    loop = cfg.make()
    with pytest.raises(EvalScriptsError, match=r"eval_script_modified set-a.sh"):
        loop.run()
    suite = loop.metrics_eval[""]
    assert isinstance(suite, MlsbenchSuite)
    assert suite.runs == []
    assert not (
        tmp_path / "out/runs/mlsbench" / cfg.experiment_name / "runs/workspace"
    ).exists()


def test_a_swapped_eval_script_is_refused_between_settings(
    tmp_path: Path,
    root: Path,
) -> None:
    cfg, _ = _bundled(tmp_path, root, _fake())
    suite = cfg.make().metrics_eval[""]
    assert isinstance(suite, MlsbenchSuite)
    suite.update(torch.zeros(1), setting="set-a")
    (root / "tasks" / SYNTH_TASK / "scripts" / "set-c.sh").write_text("echo swap\n")
    with pytest.raises(EvalScriptsError, match=r"set-c.sh"):
        suite.update(torch.zeros(1), setting="set-c")
    assert [r.label for r in suite.runs] == ["set-a"]


def test_an_intact_bundle_run_logs_adapted_scripts_and_passes(
    tmp_path: Path,
    root: Path,
) -> None:
    cfg, _ = _bundled(tmp_path, root, _fake())
    metrics, _ = _run(cfg, tmp_path)
    assert metrics["eval/accuracy_set-a"] == 0.322
    ws_json = next((tmp_path / "out").rglob("workspace.json")).read_text()
    assert '"harbor_adapted_scripts": []' in ws_json


def test_a_workspace_changed_between_settings_fails_that_setting_only(
    tmp_path: Path,
    root: Path,
) -> None:
    fake = _fake()
    cfg = _cfg(tmp_path, root, fake)
    suite = cfg.make().metrics_eval[""]
    assert isinstance(suite, MlsbenchSuite)
    suite.update(torch.zeros(1), setting="set-a")
    ws = next((tmp_path / "out").rglob("workspace/run.py"))
    # outputs a setting adds are fine; rewriting the edit surface is not
    (ws.parent / "outputs.txt").write_text("fine\n")
    suite.update(torch.zeros(1), setting="set-b")
    ws.write_text(ws.read_text() + "tampered\n")
    suite.update(torch.zeros(1), setting="set-c")
    by = {r.label: r for r in suite.runs}
    assert by["set-a"].error is None
    assert by["set-b"].error is None
    bad = by["set-c"]
    assert bad.outcome is Outcome.CRASHED
    assert bad.error == "workspace_changed:modified:run.py"
    assert bad.metrics == {}
    assert [r.setting.label for r in fake_calls(suite)] == [
        "set-a",
        "set-b",
    ]
    out = suite.compute()
    assert out["workspace_changed_set-c"] == 1.0
    assert "workspace_changed_set-a" not in out
    assert "accuracy_set-a" in out
    assert '"workspace_changed:modified:run.py"' in (
        next((tmp_path / "out").rglob("settings.json")).read_text()
    )


def fake_calls(suite: MlsbenchSuite) -> list[RunRequest]:
    ex = suite.executor
    assert isinstance(ex, FakeExecutor)
    return ex.calls


def test_the_workspace_change_marker_reaches_the_metrics_file(
    tmp_path: Path,
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg(tmp_path, root, _fake())
    loop = cfg.make()
    suite = loop.metrics_eval[""]
    assert isinstance(suite, MlsbenchSuite)
    orig = suite.update

    def tamper(logits: torch.Tensor, **batch: object) -> None:
        orig(logits, **batch)
        if batch["setting"] == "set-a":
            ws = next((tmp_path / "out").rglob("workspace/run.py"))
            ws.write_text("changed\n")

    monkeypatch.setattr(suite, "update", tamper)
    loop.run()
    path = tmp_path / "out/runs/mlsbench" / cfg.experiment_name / "metrics.json"
    got = dict(read_metrics(path))
    assert got["eval/accuracy_set-a"] == 0.322
    assert got["eval/workspace_changed_set-b"] == 1.0
    assert got["eval/workspace_changed_set-c"] == 1.0
    assert "eval/pass_at_1_set-b" not in got


# -- real-run wiring: package env, data mounts, held-out inputs, seeds ---------


def test_expand_env_mirrors_harbor_templates() -> None:
    base = {"A": "1"}
    assert expand_env("/w:${PYTHONPATH}", base) == "/w:"  # unset -> empty
    assert expand_env("$A-${A}-${A:-9}-${B:-9}", base) == "1-1-1-"  # default ignored
    assert expand_env("plain", base) == "plain"


def _bundle_with_envs(tmp_path: Path, envs: dict[str, dict[str, str]]) -> Path:
    bundle = make_bundle(tmp_path / "b", declared="run.py", ranges=[[5, 10]])
    (bundle / "tests/meta/package_envs.json").write_text(json.dumps(envs))
    return bundle


def test_package_env_follows_the_setting_package_then_the_task_default(
    tmp_path: Path,
    root: Path,
) -> None:
    task = load_task(SYNTH_TASK, root=root)  # default package: synth-pkg
    bundle = _bundle_with_envs(
        tmp_path,
        {
            "synth-pkg": {
                "HF_HOME": "/opt/hf",
                "PYTHONPATH": "/workspace/SynthPkg:${PYTHONPATH}",
                "HOME": "/root/${X}",
            },
            "other": {"HF_HOME": "/other"},
        },
    )
    got = dict(package_env(task, "synth-pkg", bundle))
    assert got["HF_HOME"] == "/opt/hf"
    assert got["PYTHONPATH"] == "/workspace/SynthPkg:"  # Harbor unsets the base one
    assert got["HOME"] == "/root/${X}"  # HOME is taken literally
    assert got["DATA_ROOT"] == "/data"
    assert dict(package_env(task, "other", bundle))["HF_HOME"] == "/other"
    # a package with no block falls back to the task's default package's
    assert dict(package_env(task, "unlisted", bundle))["HF_HOME"] == "/opt/hf"


def test_a_baseline_env_overlays_and_expands_against_the_package_env(
    tmp_path: Path,
    root: Path,
) -> None:
    task = load_task(SYNTH_TASK, root=root)
    bundle = _bundle_with_envs(tmp_path, {"synth-pkg": {"BASE": "/b"}})
    got = dict(
        package_env(
            task,
            "synth-pkg",
            bundle,
            overlay={"X": "${BASE}/x", "BASE": "/c"},
        ),
    )
    assert got["X"] == "/b/x"
    assert got["BASE"] == "/c"


def test_package_env_falls_back_to_the_native_pkg_config_without_a_bundle(
    root: Path,
) -> None:
    cfg = root / "vendor/pkg_configs/Synth-pkg"
    cfg.mkdir(parents=True)
    (cfg / "config.json").write_text(json.dumps({"env": {"HF_HUB_OFFLINE": "1"}}))
    task = load_task(SYNTH_TASK, root=root)
    assert dict(package_env(task, "synth-pkg", None))["HF_HUB_OFFLINE"] == "1"
    assert dict(package_env(task, "no-such", None)) == {"DATA_ROOT": "/data"}


def _suite(
    tmp_path: Path,
    root: Path,
    *,
    bundle: Path | None = None,
    seed: int = 42,
    baseline: str = "base_a",
    data_root: Path | str = "",
    staged_data: Path | str = "",
) -> MlsbenchSuite:
    return MlsbenchSuite.Config(
        task=SYNTH_TASK,
        baseline=baseline,
        seed=seed,
        mlsbench_root=str(root),
        data_root=data_root,
        staged_data=staged_data,
        bundle="" if bundle is None else str(bundle),
        working_dir=tmp_path / f"w{seed}",
        executor=_fake(),
    ).make()


def _bundle_for(tmp_path: Path, root: Path) -> Path:
    bundle = make_bundle(tmp_path / "b", declared="run.py", ranges=[[5, 10]])
    ref = bundle / "tests/eval/scripts"
    for p in (root / "tasks" / SYNTH_TASK / "scripts").glob("*.sh"):
        (ref / p.name).write_bytes(p.read_bytes())
    (ref / "a.sh").unlink()
    return bundle


def test_every_request_carries_the_package_env(tmp_path: Path, root: Path) -> None:
    bundle = _bundle_for(tmp_path, root)
    (bundle / "tests/meta/package_envs.json").write_text(
        json.dumps({"synth-pkg": {"HF_HOME": "/opt/hf", "P": "/w:${P}"}}),
    )
    suite = _suite(tmp_path, root, bundle=bundle)
    for lab in ("set-a", "set-c"):
        suite.update(torch.empty(0), setting=lab)
    for call in fake_calls(suite):
        env = dict(call.env)
        assert env["HF_HOME"] == "/opt/hf"
        assert env["P"] == "/w:"
    record = _read_json(tmp_path / "w42/logs/set-a.request.json")
    assert cast("dict[str, str]", record["env"])["HF_HOME"] == "/opt/hf"
    assert record["seed"] == 42


def test_data_root_and_staged_data_become_request_binds(
    tmp_path: Path,
    root: Path,
) -> None:
    from priml.baselines.mlsbench.scripts import prepare_data as pd  # noqa: PLC0415

    dr = tmp_path / "datasets"
    (dr / "weights").mkdir(parents=True)
    cfg = root / "vendor/pkg_configs/synth-pkg"
    cfg.mkdir(parents=True)
    (cfg / "config.json").write_text(
        json.dumps(
            {
                "data_deps": [
                    {
                        "host_path": "{data_root}/weights",
                        "container_path": "/data/w",
                    },
                    {"host_path": "{data_root}/absent", "container_path": "/data/x"},
                ],
            },
        ),
    )
    (root / "tasks" / SYNTH_TASK / "data").mkdir()
    (root / "tasks" / SYNTH_TASK / "data" / "d.txt").write_text("d")
    staged = tmp_path / "staged"
    pd.stage_task(root, SYNTH_TASK, staged)
    suite = _suite(tmp_path, root, data_root=dr, staged_data=staged)
    suite.update(torch.empty(0), setting="set-a")
    (call,) = fake_calls(suite)
    assert f"{dr}/weights:/data/w" in call.binds
    assert not any("absent" in b for b in call.binds)  # never bind a missing dir
    assert any(b.endswith(":/workspace/_task/data:ro") for b in call.binds)
    rec = _read_json(tmp_path / "w42/logs/set-a.request.json")
    assert rec["skipped_binds"] == [f"{dr}/absent:/data/x"]


def test_a_tampered_staged_copy_is_a_typed_missing_input(
    tmp_path: Path,
    root: Path,
) -> None:
    from priml.baselines.mlsbench.scripts import prepare_data as pd  # noqa: PLC0415

    (root / "tasks" / SYNTH_TASK / "data").mkdir()
    (root / "tasks" / SYNTH_TASK / "data" / "d.txt").write_text("d")
    staged = tmp_path / "staged"
    pd.stage_task(root, SYNTH_TASK, staged)
    (staged / SYNTH_TASK / "data" / "d.txt").write_text("evil")
    suite = _suite(tmp_path, root, staged_data=staged)
    suite.update(torch.empty(0), setting="set-a")
    assert fake_calls(suite) == []  # the script never ran
    (run,) = suite.runs
    assert run.outcome is Outcome.MISSING_INPUT
    assert run.error is not None
    assert run.error.startswith("missing_input:staged data:")
    assert suite.compute() == {}  # absent, never NaN


def test_data_mounts_without_configuration_are_empty(root: Path) -> None:
    task = load_task(SYNTH_TASK, root=root)
    assert data_mounts(task, "synth-pkg") == data_mounts(
        task,
        "synth-pkg",
        data_root="",
    )
    assert data_mounts(task, "synth-pkg").binds == ()


def test_ephemeral_inputs_without_a_generator_are_a_typed_missing_input(
    tmp_path: Path,
) -> None:
    cfg = default_config()
    cfg["ephemeral_inputs"] = True
    root = build_task(tmp_path / "up", config=cfg)
    suite = MlsbenchSuite.Config(
        task="toy-task",
        mlsbench_root=str(root),
        working_dir=tmp_path / "w",
        executor=FakeExecutor.Config(),
    ).make()
    suite.update(torch.empty(0), setting="a-x")
    assert fake_calls(suite) == []
    (run,) = suite.runs
    assert run.outcome is Outcome.MISSING_INPUT
    assert run.error is not None
    assert "ephemeral_inputs" in run.error
    assert "no Harbor bundle" in run.error


def test_a_bundle_generator_is_handed_to_the_executor(
    tmp_path: Path,
    root: Path,
) -> None:
    bundle = _bundle_for(tmp_path, root)
    assert inputgen_dir(bundle) is None
    gen = bundle / "tests/eval/_inputgen"
    gen.mkdir(parents=True)
    (gen / "apply.py").write_text("# stand-in\n")
    assert inputgen_dir(bundle) == gen
    suite = _suite(tmp_path, root, bundle=bundle)
    suite.update(torch.empty(0), setting="set-a")
    (call,) = fake_calls(suite)
    assert call.inputgen == gen
    # a bundle without one hands over nothing
    (gen / "apply.py").unlink()
    other = _suite(tmp_path / "o", root, bundle=bundle)
    other.update(torch.empty(0), setting="set-a")
    assert fake_calls(other)[0].inputgen is None


def test_a_missing_input_from_the_executor_is_recorded_typed(
    tmp_path: Path,
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite = _suite(tmp_path, root)
    ex = suite.executor
    assert isinstance(ex, FakeExecutor)

    def run(request: RunRequest) -> object:
        from priml.baselines.mlsbench.executor import RunResult  # noqa: PLC0415

        return RunResult(
            Outcome.MISSING_INPUT,
            None,
            request.log_path,
            0.1,
            "inputgen exited 3",
        )

    monkeypatch.setattr(ex, "run", run)
    suite.update(torch.empty(0), setting="set-a")
    assert suite.runs[0].error == "missing_input:inputgen exited 3"


def _seeded_root(tmp_path: Path) -> Path:
    cfg = default_config()
    cfg["seeds"] = [42, 123]
    return build_task(tmp_path / "up", config=cfg)


def test_each_seed_is_its_own_env_and_run_dir(tmp_path: Path) -> None:
    root = _seeded_root(tmp_path)
    seen: dict[int, tuple[str, str]] = {}
    for seed in (42, 123):
        cfg = mlsbench_experiment("toy-task", "base1", seed=seed, executor=_fake())
        cfg.base_dir = tmp_path / "out"
        cfg.dataset.mlsbench_root = str(root)
        suite_cfg = cfg.metrics_eval[""]
        assert isinstance(suite_cfg, MlsbenchSuite.Config)
        suite_cfg.mlsbench_root = str(root)
        suite_cfg.executor = FakeExecutor.Config()
        cfg = cfg.copy_tree().finalize()
        suite_cfg = cfg.metrics_eval[""]
        assert isinstance(suite_cfg, MlsbenchSuite.Config)
        suite = suite_cfg.make()
        suite.update(torch.empty(0), setting="a-x")
        (call,) = fake_calls(suite)
        assert call.seed == seed
        seen[seed] = (cfg.experiment_name, str(suite_cfg.working_dir))
        ws = _read_json(Path(suite_cfg.working_dir) / "workspace.json")
        assert ws["seed"] == seed
    assert seen[42][0].endswith("seed42")
    assert seen[123][0].endswith("seed123")
    assert seen[42][1] != seen[123][1]
    assert "seed123" in seen[123][1]


def test_a_seed_the_task_does_not_publish_is_refused(tmp_path: Path) -> None:
    root = build_task(tmp_path / "up")  # seeds [42]
    suite = MlsbenchSuite.Config(
        task="toy-task",
        seed=7,
        mlsbench_root=str(root),
        working_dir=tmp_path / "w",
        executor=FakeExecutor.Config(),
    ).make()
    with pytest.raises(ValueError, match=r"seed 7 is not one of toy-task's seeds"):
        suite.update(torch.empty(0), setting="a-x")
