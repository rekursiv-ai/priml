"""Tests for a setting's environment, data mounts, columns and upstream parser."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import json
import os
import sys

import pytest

from priml.baselines.mlsbench.harbor_test import make_bundle
from priml.baselines.mlsbench.setting import (
    Mounts,
    ParseError,
    data_mounts,
    elapsed_columns,
    expand_env,
    expected_columns,
    package_env,
    run_parser,
)
from priml.baselines.mlsbench.task import load_task
from priml.baselines.mlsbench.testing import (
    SYNTH_TASK,
    build_synth_task,
    build_task,
    leaderboard_csv,
)


_CLONE: Final = os.environ.get("MLSBENCH_ROOT", "")


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return build_synth_task(tmp_path / "up")


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


def test_a_non_numeric_metric_is_a_parse_error(tmp_path: Path) -> None:
    src = (
        "class Parser:\n"
        "    def parse(self, cmd_label, raw_output):\n"
        "        class R:\n"
        "            metrics = {'acc_' + cmd_label: 'high'}\n"
        "        return R()\n"
    )
    task = load_task("toy-task", root=build_task(tmp_path / "up", parser_source=src))
    with pytest.raises(ParseError, match="non-numeric 'high'"):
        run_parser(task, "", "a-x")


def test_the_bundle_parser_runs_in_place_of_the_tasks(tmp_path: Path) -> None:
    task = load_task("toy-task", root=build_task(tmp_path / "up"))
    meta = tmp_path / "bundle" / "tests" / "meta"
    meta.mkdir(parents=True)
    (meta / "parser.py").write_text(
        "class Parser:\n"
        "    def parse(self, cmd_label, raw_output):\n"
        "        class R:\n"
        "            metrics = {'from_bundle_' + cmd_label: 1.0}\n"
        "        return R()\n",
    )
    assert run_parser(task, "", "b", meta=meta) == {"from_bundle_b": 1.0}


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
    assert sys.path == before  # The clone's src is not left on the path.


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
    assert "dgp" not in sys.modules  # Another task's dgp must not be reused.


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


def test_expected_columns_prefer_the_longest_label(tmp_path: Path) -> None:
    task = load_task("toy-task", root=build_task(tmp_path / "up"))
    assert expected_columns(task, "a-x") == ("accuracy_a-x",)
    assert expected_columns(task, "b") == ("loss_b",)


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


def test_expand_env_mirrors_harbor_templates() -> None:
    base = {"A": "1"}
    assert expand_env("/w:${PYTHONPATH}", base) == "/w:"  # Unset -> empty.
    assert expand_env("$A-${A}-${A:-9}-${B:-9}", base) == "1-1-1-"  # Default ignored.
    assert expand_env("plain", base) == "plain"


def _bundle_with_envs(tmp_path: Path, envs: dict[str, dict[str, str]]) -> Path:
    bundle = make_bundle(tmp_path / "b", declared="run.py", ranges=[[5, 10]])
    (bundle / "tests/meta/package_envs.json").write_text(json.dumps(envs))
    return bundle


def test_package_env_follows_the_setting_package_then_the_task_default(
    tmp_path: Path,
    root: Path,
) -> None:
    task = load_task(SYNTH_TASK, root=root)  # Default package: synth-pkg.
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
    assert got["PYTHONPATH"] == "/workspace/SynthPkg:"  # Harbor unsets the base one.
    assert got["HOME"] == "/root/${X}"  # HOME is taken literally.
    assert got["DATA_ROOT"] == "/data"
    assert dict(package_env(task, "other", bundle))["HF_HOME"] == "/other"
    # A package with no block falls back to the task's default package's.
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
    (cfg / "config.json").write_text(json.dumps({"env": {"HF_HUB_OFFLINE": 1}}))
    task = load_task(SYNTH_TASK, root=root)
    assert dict(package_env(task, "synth-pkg", None))["HF_HUB_OFFLINE"] == "1"
    assert dict(package_env(task, "no-such", None)) == {"DATA_ROOT": "/data"}


def test_data_mounts_expand_templates_and_dedupe(tmp_path: Path, root: Path) -> None:
    dr = tmp_path / "data"
    for sub in ("a", "b"):
        (dr / sub).mkdir(parents=True)
    _pkg(
        root,
        "Synth-Pkg",
        {
            "data_bind": ["{data_root}/a:/data/a", "{data_root}/gone:/data/gone"],
            "data_deps": [
                {"host_path": "{data_root}/a", "container_path": "/data/a"},
                {"host_path": "{data_root}/b", "container_path": "/data/b"},
                {"host_path": "{project_root}/vendor", "container_path": "/v"},
                {"name": "no-paths"},
                "not-a-dict",
            ],
        },
    )
    got = data_mounts(load_task(SYNTH_TASK, root=root), "synth-pkg", data_root=dr)
    assert got.binds == (f"{dr}/a:/data/a", f"{dr}/b:/data/b", f"{root}/vendor:/v")
    assert got.skipped == (f"{dr}/gone:/data/gone",)


def test_a_scalar_data_bind_is_accepted(tmp_path: Path, root: Path) -> None:
    dr = tmp_path / "data"
    dr.mkdir()
    _pkg(root, "synth-pkg", {"data_bind": "{data_root}:/data"})
    task = load_task(SYNTH_TASK, root=root)
    assert data_mounts(task, "synth-pkg", data_root=dr).binds == (f"{dr}:/data",)
    assert data_mounts(task, "unknown", data_root=dr) == Mounts()


def _pkg(root: Path, name: str, cfg: dict[str, object]) -> None:
    d = root / "vendor" / "pkg_configs" / name
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps(cfg))


def test_data_mounts_without_configuration_are_empty(root: Path) -> None:
    task = load_task(SYNTH_TASK, root=root)
    assert data_mounts(task, "synth-pkg") == data_mounts(
        task,
        "synth-pkg",
        data_root="",
    )
    assert data_mounts(task, "synth-pkg").binds == ()


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
