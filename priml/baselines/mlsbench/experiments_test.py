"""Tests for the harness experiment configs and their pprint goldens."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Final, cast

import importlib
import json
import os

import pytest

from priml.baselines.mlsbench import tasks
from priml.baselines.mlsbench.data import SettingBatches
from priml.baselines.mlsbench.executor import FakeExecutor
from priml.baselines.mlsbench.experiments import (
    REGISTRY,
    REGISTRY_PATH,
    MlsbenchEval,
    TaskEntry,
    candidate_experiment,
    exp_smoke,
    mlsbench_experiment,
    registry,
    task_entry,
)
from priml.baselines.mlsbench.metric import MlsbenchSuite
from priml.baselines.mlsbench.scripts.generate_experiments import (
    registry_from_clone,
    write_registry,
)
from priml.baselines.mlsbench.testing import (
    SYNTH_BASELINES,
    SYNTH_TASK,
    build_synth_task,
)
from priml.baselines.mlsbench.train_step import NullStep
from priml.runtime import SingleProcess
from priml.testing.experiments import ExperimentFactory  # noqa: TC001
from priml.testing.golden import assert_pprint_golden
from priml.train.tracker import FileTracker
from priml.train.train_loop import TrainLoop


ALL: Final[list[ExperimentFactory[MlsbenchEval]]] = [exp_smoke]
FIRST_TASK: Final = next(iter(REGISTRY))
FIRST_BASELINES: Final = REGISTRY[FIRST_TASK]


def _suite(cfg: MlsbenchEval) -> MlsbenchSuite.Config:
    suite = cfg.metrics_eval[""]
    assert isinstance(suite, MlsbenchSuite.Config)
    return suite


def test_slots_carry_their_concrete_types() -> None:
    cfg = MlsbenchEval()
    assert isinstance(cfg.step, NullStep.Config)
    assert isinstance(cfg.dataset, SettingBatches.Config)
    assert MlsbenchEval.parent_class is TrainLoop


@pytest.mark.parametrize("factory", ALL, ids=[f.__name__ for f in ALL])
def test_every_experiment_finalizes(factory: ExperimentFactory[MlsbenchEval]) -> None:
    assert factory().copy_tree().finalize() is not None


@pytest.mark.parametrize("factory", ALL, ids=[f.__name__ for f in ALL])
def test_nothing_trains_and_numbers_leave_through_a_file(
    factory: ExperimentFactory[MlsbenchEval],
) -> None:
    cfg = factory()
    assert cfg.eval_only is True
    assert cfg.checkpointer is None
    assert list(cfg.metrics_eval) == [""]
    assert isinstance(cfg.tracker, FileTracker.Config)
    runtime = cfg.runtime
    assert isinstance(runtime, SingleProcess.Config)
    assert runtime.deterministic is False
    assert cfg.study_name == "mlsbench"


@pytest.mark.parametrize("factory", ALL, ids=[f.__name__ for f in ALL])
def test_experiment_matches_its_golden_config(
    factory: ExperimentFactory[MlsbenchEval],
) -> None:
    assert_pprint_golden(test_file=__file__, name=factory.__name__, config=factory())


def test_smoke_uses_the_fake_executor() -> None:
    cfg = exp_smoke()
    assert isinstance(_suite(cfg).executor, FakeExecutor.Config)
    assert cfg.experiment_name == "exp_smoke"


def test_the_executor_is_injected() -> None:
    fake = FakeExecutor.Config()
    assert _suite(mlsbench_experiment("t", "b", executor=fake)).executor is fake


def test_factory_rejects_an_unknown_baseline_of_a_registered_task() -> None:
    with pytest.raises(ValueError, match="unknown baseline"):
        mlsbench_experiment(FIRST_TASK, "nope")


def test_other_tasks_go_through_the_factory() -> None:
    cfg = mlsbench_experiment("some-task", "some-base", seed=7)
    assert cfg.experiment_name == "some-task/some-base/seed7"
    assert _suite(cfg).seed == 7


def test_max_eval_time_is_a_pure_function_of_the_registry_entry(
    tmp_path: Path,
) -> None:
    entry = cast("TaskEntry", task_entry(FIRST_TASK))
    expected = mlsbench_experiment(FIRST_TASK, FIRST_BASELINES[0]).max_eval_time
    assert expected == entry.max_eval_time
    assert expected == pytest.approx(sum(entry.settings.values()) * 1.1, abs=1e-3)
    # a clone, even a different one, does not change it
    cfg = mlsbench_experiment(FIRST_TASK, FIRST_BASELINES[0])
    cfg.mlsbench_root = build_synth_task(tmp_path / "up")
    assert cfg.copy_tree().finalize().max_eval_time == expected


def test_a_task_outside_the_registry_has_no_budget() -> None:
    assert mlsbench_experiment("some-task", "b").max_eval_time == float("inf")


def test_the_committed_registry_has_the_plans_shape() -> None:
    reg = registry()
    assert len(reg) == 140
    assert sum(len(b) for b in reg.values()) == 480
    assert sum(len(cast("TaskEntry", task_entry(t)).settings) for t in reg) == 424
    assert reg == REGISTRY


def test_registry_round_trips_through_a_clone(tmp_path: Path) -> None:
    root = build_synth_task(tmp_path / "up")
    got = registry_from_clone(root)
    assert set(got) == {SYNTH_TASK}
    assert got[SYNTH_TASK].baselines == SYNTH_BASELINES
    out = tmp_path / "registry.json"
    assert write_registry(root, out) == 1
    assert json.loads(out.read_text())[SYNTH_TASK]["baselines"] == list(SYNTH_BASELINES)


def _generated() -> list[tuple[str, str, tuple[str, str, int]]]:
    out: list[tuple[str, str, tuple[str, str, int]]] = []
    for m in tasks.task_modules():
        mod = importlib.import_module(f"{tasks.__name__}.{m}")
        index = cast("dict[str, tuple[str, str, int]]", mod.INDEX)
        out.extend((m, fn, ident) for fn, ident in index.items())
    return out


GENERATED: Final = _generated()
"""(task module, function name, (task, baseline, seed)) for every generated experiment."""

GOLDEN_DIR: Final = Path(__file__).with_name("testdata") / "tasks"


def _raw_registry() -> dict[str, dict[str, list[object]]]:
    return cast(
        "dict[str, dict[str, list[object]]]",
        json.loads(REGISTRY_PATH.read_text()),
    )


def test_experiment_count_is_baselines_times_seeds_of_the_registry() -> None:
    raw = _raw_registry()
    total = sum(len(d["baselines"]) * len(d.get("seeds") or [42]) for d in raw.values())
    assert total == len(GENERATED) == 566
    assert sum(len(d["baselines"]) for d in raw.values()) == 480


def test_generated_names_are_unique() -> None:
    assert len({f"{f}.{n}" for f, n, _ in GENERATED}) == len(GENERATED)
    assert len({i for _, _, i in GENERATED}) == len(GENERATED)
    assert len({f"{t}/{b}/seed{s}" for _, _, (t, b, s) in GENERATED}) == len(GENERATED)


@pytest.mark.parametrize(
    ("mod", "fn", "ident"),
    GENERATED,
    ids=[f"{f}.{n}" for f, n, _ in GENERATED],
)
def test_generated_experiment_round_trips_and_matches_its_golden(
    mod: str,
    fn: str,
    ident: tuple[str, str, int],
) -> None:
    task, baseline, seed = ident
    cfg = tasks.resolve(f"{mod}.{fn}")()
    suite = _suite(cfg)
    assert (suite.task, suite.baseline, suite.seed) == ident
    assert cfg.dataset.task == task
    assert cfg.experiment_name == f"{task}/{baseline}/seed{seed}"
    assert cfg.max_eval_time == cast("TaskEntry", task_entry(task)).max_eval_time
    assert cfg.copy_tree().finalize() is not None
    assert_pprint_golden(test_file=__file__, name=f"tasks/{mod}/{fn}", config=cfg)


def test_every_experiment_has_a_golden_and_no_golden_is_orphaned() -> None:
    want = {f"{f}/{n}.txt" for f, n, _ in GENERATED}
    have = {str(p.relative_to(GOLDEN_DIR)) for p in GOLDEN_DIR.rglob("*.txt")}
    assert have == want
    assert not list(GOLDEN_DIR.parent.glob("task_*.txt")), "retired per-task goldens"


def test_goldens_are_host_independent() -> None:
    bad = ("/Users/", "/home/", "/private/", "/tmp/", "/var/folders")  # noqa: S108
    for p in GOLDEN_DIR.rglob("*.txt"):
        text = p.read_text()
        assert not any(b in text for b in bad), p
        # every resource path is the logical default beneath the default base_dir
        assert "mlsbench_root=PosixPath('/opt/scratch/mlsbench" in text, p


def test_other_baselines_of_a_task_differ_only_in_name() -> None:
    for task, names in REGISTRY.items():
        base = _flatten(mlsbench_experiment(task, names[0]))
        for other in names[1:]:
            got = _flatten(mlsbench_experiment(task, other))
            moved = {k for k in base.keys() | got.keys() if base.get(k) != got.get(k)}
            assert moved == {"experiment_name", "metrics_eval..baseline"}, (task, other)


def test_the_factory_accepts_every_registered_pair() -> None:
    for task, names in REGISTRY.items():
        for n in names:
            assert mlsbench_experiment(task, n).experiment_name == f"{task}/{n}/seed42"


_CLONE: Final = os.environ.get("MLSBENCH_ROOT", "")


@pytest.mark.integration
@pytest.mark.skipif(not _CLONE, reason="set MLSBENCH_ROOT to a read-only clone")
def test_the_committed_registry_matches_the_clone() -> None:
    assert registry_from_clone(Path(_CLONE)) == {
        t: cast("TaskEntry", task_entry(t)) for t in REGISTRY
    }


@pytest.mark.integration
@pytest.mark.skipif(not _CLONE, reason="set MLSBENCH_ROOT to a read-only clone")
def test_generated_experiments_are_the_clones_baselines_times_seeds() -> None:
    from priml.baselines.mlsbench.task import load_all  # noqa: PLC0415

    tasks = load_all(Path(_CLONE))
    want = {(t.name, b.name, s) for t in tasks for b in t.baselines for s in t.seeds}
    assert sum(len(t.baselines) for t in tasks) == 480
    assert len(want) == len(GENERATED) == 566
    assert {i for _, _, i in GENERATED} == want


def test_resource_paths_follow_base_dir(tmp_path: Path) -> None:
    cfg = mlsbench_experiment(FIRST_TASK, FIRST_BASELINES[0])
    cfg.base_dir = tmp_path
    cfg = cfg.finalize()
    suite = _suite(cfg)
    assert suite.working_dir == (
        tmp_path / f"runs/mlsbench/{FIRST_TASK}/{FIRST_BASELINES[0]}/seed42/runs"
    )
    # Shared resources resolve beneath the resource root, not the run directory.
    assert cfg.dataset.mlsbench_root == tmp_path / "mlsbench"
    assert suite.mlsbench_root == tmp_path / "mlsbench"
    assert suite.data_root == tmp_path / "datasets/mlsbench"
    assert suite.staged_data == tmp_path / "datasets/mlsbench_staged"


def test_a_candidate_is_its_own_row_with_a_logical_edit(tmp_path: Path) -> None:
    cfg = candidate_experiment(FIRST_TASK, "my-idea", "/edits/my-idea.edit.py")
    assert cfg.experiment_name == f"{FIRST_TASK}/my-idea/seed42"
    entry = task_entry(FIRST_TASK)
    assert entry is not None
    assert cfg.max_eval_time == entry.max_eval_time
    cfg.base_dir = tmp_path
    suite = _suite(cfg.copy_tree().finalize())
    assert suite.baseline == ""
    assert suite.edit_path == tmp_path / "edits/my-idea.edit.py"
    with pytest.raises(ValueError, match="must be new"):
        _ = candidate_experiment(FIRST_TASK, FIRST_BASELINES[0], "/e.py")
    with pytest.raises(ValueError, match="must be new"):
        _ = candidate_experiment(FIRST_TASK, "", "/e.py")
    with pytest.raises(ValueError, match="edit_path is required"):
        _ = candidate_experiment(FIRST_TASK, "my-idea", "")
    assert cfg.dataset.settings == ()
    screen = candidate_experiment(
        FIRST_TASK,
        "my-idea-screen",
        "/e.py",
        settings=("x",),
    )
    assert screen.dataset.settings == ("x",)


def test_resource_paths_are_one_field_each(tmp_path: Path) -> None:
    cfg = exp_smoke()
    cfg.base_dir = tmp_path
    cfg.mlsbench_root = "/elsewhere"
    cfg = cfg.copy_tree().finalize()
    assert cfg.dataset.mlsbench_root == tmp_path / "elsewhere"
    assert _suite(cfg).mlsbench_root == tmp_path / "elsewhere"


def test_smoke_reads_the_clone_prepare_data_builds() -> None:
    cfg = exp_smoke().copy_tree().finalize()
    assert cfg.dataset.mlsbench_root == Path("/opt/scratch/mlsbench_smoke")


def _flatten(config: object, prefix: str = "") -> dict[str, object]:
    assert is_dataclass(config)
    flat: dict[str, object] = {}
    for spec in fields(config):
        value = cast("object", getattr(config, spec.name))
        dotted = f"{prefix}{spec.name}"
        if is_dataclass(value):
            flat[dotted] = type(value)
            flat.update(_flatten(value, prefix=f"{dotted}."))
        elif isinstance(value, dict):
            for k, item in cast("dict[str, object]", value).items():
                sub = f"{dotted}.{k}"
                if is_dataclass(item):
                    flat[sub] = type(item)
                    flat.update(_flatten(item, prefix=f"{sub}."))
                else:
                    flat[sub] = item
        else:
            flat[dotted] = value
    return flat
