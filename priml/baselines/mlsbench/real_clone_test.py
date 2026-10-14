"""Opt-in sweep against a real upstream clone (``MLSBENCH_ROOT``), no docker.

Every task and baseline is materialized from the Harbor bundle's pristine tree,
the baseline edit is applied and the diff guard runs. A handful of tasks also go
through the whole metric path with a canned-log executor and the real parser.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import importlib
import json
import os
import shutil
import tempfile

import pytest

from priml.baselines.mlsbench.executor import FakeExecutor
from priml.baselines.mlsbench.experiments import mlsbench_experiment
from priml.baselines.mlsbench.metric import MlsbenchSuite, expected_columns
from priml.baselines.mlsbench.scripts.generate_experiments import NATIVE
from priml.baselines.mlsbench.task import TaskSpec, load_all
from priml.baselines.mlsbench.workspace import materialize
from priml.baselines.mlsbench.workspace_harbor import (
    ADAPTED_EVAL_SCRIPTS,
    NORMALISED_ONLY,
    EvalScriptsError,
    MissingBundleError,
    allow_internet,
    bundle_dir,
    check_eval_scripts,
    eval_scripts_intact,
    load_bundle,
    normalise_harbor_script,
)
from priml.train.results import read_metrics


WARNED_BASELINES = frozenset(
    {
        "cv-diffusion-cfg/cfg",
        "cv-diffusion-cfg/cfgpp",
        "cv-diffusion-cfg/zeroinit",
        "cv-diffusion-efficiency/ddim",
        "cv-diffusion-efficiency/dpm3m_sde",
        "cv-diffusion-efficiency/dpm2s",
        "ml-continual-regularization/ewc",
        "mlsys-sparse-attention-inference/dense",
        "mlsys-sparse-attention-inference/streaming_llm",
        "mlsys-sparse-attention-inference/bigbird",
        "mlsys-sparse-attention-inference/block_topk",
    },
)
"""The 11 baselines whose upstream data trips a guard/manifest finding (non-strict)."""

pytestmark = [pytest.mark.integration, pytest.mark.timeout(1200)]

_ROOT = os.environ.get("MLSBENCH_ROOT", "")
needs_clone = pytest.mark.skipif(not _ROOT, reason="set MLSBENCH_ROOT to a clone")

FULL_PATH_TASKS = (
    "llm-dllm-demask-strategy",
    "ml-calibration",
    "cv-diffusion-cfg",
    "ml-continual-regularization",
    "optimization-bilevel",
)


def _root() -> Path:
    return Path(_ROOT)


# Baselines with no edit script: nothing to apply, so the sweep only proves the
# pristine tree materializes and passes the guard for them. 22 run a script of
# their own (``cmd``) instead of an edit; 6 (``default`` on the robo tasks) are the
# unmodified starting code and have neither.
NOOP_BASELINES = (
    ("robo-diffusion-guidance", "default"),
    ("robo-diffusion-policy", "default"),
    ("robo-diffusion-sampling-method", "default"),
    ("robomimic-bc-loss", "default"),
    ("robomimic-iql-vf", "default"),
    ("robomimic-obs-encoder", "default"),
)
N_BASELINES, N_EDIT, N_CMD_ONLY = 480, 452, 22


@needs_clone
def test_every_task_and_baseline_materializes_edits_and_passes_the_guard() -> None:
    tasks = load_all(_root())
    assert len(tasks) == 140
    swept: list[tuple[str, str]] = []
    with_edit = cmd_only = 0
    noop: list[tuple[str, str]] = []
    warned: list[str] = []
    with tempfile.TemporaryDirectory() as td:
        for t in tasks:
            bundle = bundle_dir(t.name, root=_root())
            for bl in t.baselines:
                n = len(swept)
                swept.append((t.name, bl.name))
                # Published baselines run non-strict; warnings are the reported set.
                ws = materialize(
                    t,
                    Path(td) / f"w{n}",
                    edit=bl.edit,
                    bundle=bundle,
                    strict=False,
                )
                warned.extend(f"{t.name}/{bl.name}" for _ in ws.warnings)
                if bl.edit is not None:
                    with_edit += 1
                elif bl.cmd is not None:
                    cmd_only += 1
                else:
                    noop.append((t.name, bl.name))
            # An agent edit (no edit at all) must be clean and strict.
            materialize(t, Path(td) / f"p{len(swept)}", edit=None, bundle=bundle)
    # The sweep covers exactly the registry's 480 baselines, none skipped.
    registry = cast(
        "dict[str, dict[str, list[object]]]",
        json.loads((Path(__file__).parent / "registry.json").read_text()),
    )
    assert len(swept) == len(set(swept)) == N_BASELINES
    assert {(t, str(b)) for t, v in registry.items() for b in v["baselines"]} == {
        (t.name, b.name) for t in tasks if t.name not in NATIVE for b in t.baselines
    }
    # ... and the ones where no edit is applied are an explicit, justified list.
    assert (with_edit, cmd_only, tuple(noop)) == (N_EDIT, N_CMD_ONLY, NOOP_BASELINES)
    assert with_edit + cmd_only + len(noop) == N_BASELINES
    # Baselines upstream itself does not hold to its guard: keep the list explicit.
    got = sorted(set(warned))
    want = sorted(WARNED_BASELINES)
    assert got == want, (
        f"warned baselines changed; unexpected: {sorted(set(got) - set(want))}, "
        f"no longer warning: {sorted(set(want) - set(got))}"
    )


@needs_clone
def test_harbor_adapted_eval_scripts_are_the_pinned_set() -> None:
    """Exactly 32 tasks differ from tasks/*/scripts on some file; 29 on ``*.sh``.

    12 differ only by the documented Harbor additions (``cd /workspace[/pkg]``,
    the ``_inputgen/apply.py`` call, blank lines); 20 carry substantive patches.
    """
    diff: dict[str, tuple[str, ...]] = {}
    sh_only: set[str] = set()
    normalised: set[str] = set()
    for t in load_all(_root()):
        hb = load_bundle(bundle_dir(t.name, root=_root()))
        found = eval_scripts_intact(hb, t.root / "scripts")
        if not found:
            continue
        diff[t.name] = tuple(sorted(v.path for v in found))
        if any(v.path.endswith(".sh") for v in found):
            sh_only.add(t.name)
        ref = hb.root / "tests" / "eval" / "scripts"
        if all(
            v.kind == "eval_script_modified"
            and normalise_harbor_script((ref / v.path).read_text())
            == normalise_harbor_script((t.root / "scripts" / v.path).read_text())
            for v in found
        ):
            normalised.add(t.name)
    assert diff == ADAPTED_EVAL_SCRIPTS
    assert len(diff) == 32
    assert len(sh_only) == 29
    assert sorted(set(diff) - sh_only) == [
        "quant-concept-drift",
        "quant-graph-stock",
        "quant-stock-prediction",
    ]
    assert normalised == NORMALISED_ONLY
    assert len(normalised) == 12


@needs_clone
def test_check_eval_scripts_accepts_every_pinned_task_and_refuses_a_swap(
    tmp_path: Path,
) -> None:
    for t in load_all(_root()):
        hb = load_bundle(bundle_dir(t.name, root=_root()))
        got = check_eval_scripts(t.name, hb, t.root / "scripts")
        assert got == ADAPTED_EVAL_SCRIPTS.get(t.name, ())
    # a copy of one script tree with a swapped script is refused
    t = next(t for t in load_all(_root()) if t.name == "llm-dllm-demask-strategy")
    copy = tmp_path / "scripts"
    shutil.copytree(t.root / "scripts", copy)
    victim = next(copy.glob("*.sh"))
    victim.write_text(victim.read_text() + "\necho tampered\n")
    hb = load_bundle(bundle_dir(t.name, root=_root()))
    with pytest.raises(EvalScriptsError, match=victim.name):
        check_eval_scripts(t.name, hb, copy)


@needs_clone
def test_materialize_without_a_bundle_names_the_missing_bundle(tmp_path: Path) -> None:
    t = next(t for t in load_all(_root()) if t.name == "llm-dllm-demask-strategy")
    with pytest.raises(MissingBundleError, match="mls-bench__llm-dllm-demask-strategy"):
        materialize(t, tmp_path / "r", edit=None)


@needs_clone
def test_network_policy_is_read_from_every_bundle() -> None:
    seen = {
        t.name: allow_internet(bundle_dir(t.name, root=_root()))
        for t in load_all(_root())
    }
    assert all(v in {True, False, None} for v in seen.values())


def _canned(task: TaskSpec) -> dict[str, str]:
    out: dict[str, str] = {}
    for s in task.settings:
        cols = expected_columns(task, s.label)
        if task.name == "cv-diffusion-cfg":  # this parser reads its own line format
            out[s.label] = f"GENERATION_METRICS model={s.label.split('-')[0]} fid=0.5\n"
            continue
        pairs = " ".join(f"{c.removesuffix('_' + s.label)}=0.5" for c in cols)
        out[s.label] = f"TRAIN_METRICS: step=1 loss=1.0\nTEST_METRICS: {pairs}\n"
    return out


@needs_clone
@pytest.mark.parametrize("name", FULL_PATH_TASKS)
def test_full_path_with_fake_executor_and_real_parser(
    name: str,
    tmp_path: Path,
) -> None:
    task = next(t for t in load_all(_root()) if t.name == name)
    fake = FakeExecutor.Config()
    fake.outputs = _canned(task)
    cfg = mlsbench_experiment(name, task.baselines[0].name, executor=fake)
    cfg.base_dir = tmp_path / "out"
    cfg.dataset.mlsbench_root = _ROOT
    suite = cfg.metrics_eval[""]
    assert isinstance(suite, MlsbenchSuite.Config)
    suite.mlsbench_root = _ROOT
    loop = cfg.make()
    loop.run()
    path = tmp_path / "out/runs/mlsbench" / cfg.experiment_name / "metrics.json"
    got = dict(read_metrics(path))
    for s in task.settings:
        for col in expected_columns(task, s.label):
            assert f"eval/{col}" in got, (name, col)
    ws = next(tmp_path.rglob("workspace.json"))
    assert '"bundle": "' in ws.read_text()


# Experiments whose baseline applies no edit script (nothing to materialize but
# the pristine tree): the 22 ``cmd``-only baselines and the 6 ``default`` ones,
# each once per seed of its task. Every other experiment applies an edit.
N_EXPERIMENTS = 563


@needs_clone
def test_every_generated_experiment_materializes_with_its_edit_and_guards() -> None:
    from priml.baselines.mlsbench import tasks as task_pkg  # noqa: PLC0415

    tasks = {t.name: t for t in load_all(_root()) if t.name not in NATIVE}
    seen: set[tuple[str, str, int]] = set()
    no_edit: list[tuple[str, str, int]] = []
    with tempfile.TemporaryDirectory() as td:
        for i, m in enumerate(task_pkg.task_modules()):
            mod = importlib.import_module(f"{task_pkg.__name__}.{m}")
            index = cast("dict[str, tuple[str, str, int]]", mod.INDEX)
            for fn, ident in index.items():
                cfg = task_pkg.resolve(f"{m}.{fn}")()
                suite = cfg.metrics_eval[""]
                assert isinstance(suite, MlsbenchSuite.Config)
                task_name, baseline, seed = ident
                assert (suite.task, suite.baseline, suite.seed) == ident
                task = tasks[task_name]
                assert seed in task.seeds
                bl = next(b for b in task.baselines if b.name == baseline)
                seen.add(ident)
                if bl.edit is None:
                    no_edit.append(ident)
                # The same materialization the suite performs for this run.
                run = Path(td) / f"{i}-{len(seen)}"
                materialize(
                    task,
                    run,
                    edit=bl.edit,
                    bundle=bundle_dir(task_name, root=_root()),
                    strict=False,
                )
    assert len(seen) == N_EXPERIMENTS
    assert seen == {
        (t.name, b.name, s)
        for t in tasks.values()
        for b in t.baselines
        for s in t.seeds
    }
    assert len(no_edit) == sum(
        len(tasks[t].seeds)
        for t, _ in [
            (t.name, b.name)
            for t in tasks.values()
            for b in t.baselines
            if b.edit is None
        ]
    )
    assert {(t, b) for t, b, _ in no_edit} >= set(NOOP_BASELINES)
