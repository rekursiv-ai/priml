"""Tests for the generator and the modules it wrote from ``registry.json``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, cast

import importlib
import sys

import pytest

from priml.baselines.mlsbench import tasks
from priml.baselines.mlsbench.experiments import MlsbenchEval, TaskEntry, snapshot
from priml.baselines.mlsbench.scripts import generate_experiments as gen


if TYPE_CHECKING:
    from collections.abc import Callable


N_EXPERIMENTS: Final = 563
"""Sum over tasks of baselines x seeds, pinned to upstream commit 80cf5c5c."""


def _all() -> list[tuple[str, str, tuple[str, str, int]]]:
    """Return (task module, function name, identity) for every generated function."""
    out: list[tuple[str, str, tuple[str, str, int]]] = []
    for m in tasks.task_modules():
        mod = importlib.import_module(f"{tasks.__name__}.{m}")
        index = cast("dict[str, tuple[str, str, int]]", mod.INDEX)
        out.extend((m, fn, ident) for fn, ident in index.items())
    return out


ALL: Final = _all()

REGISTRY: Final = snapshot()


def _entry(task: str) -> TaskEntry:
    return REGISTRY[task]


def test_generated_files_are_up_to_date() -> None:
    assert gen.stale(gen.generate()) == []


def test_generated_files_match_the_generator_byte_for_byte() -> None:
    for path, text in gen.generate().items():
        assert path.read_text() == text, path.name


def test_there_is_one_module_per_registry_task() -> None:
    assert set(tasks.task_modules()) == {gen.module_of(t) for t in REGISTRY}
    assert len(tasks.task_modules()) == len(REGISTRY) == 139
    assert {p.stem for p in gen.TASKS_DIR.glob("*.py")} - {"__init__"} == set(
        tasks.task_modules(),
    )


def test_each_module_holds_exactly_its_own_task() -> None:
    for m in tasks.task_modules():
        assert {t for f, _, (t, _, _) in ALL if f == m} == {
            next(t for t in REGISTRY if gen.module_of(t) == m),
        }


def test_the_total_is_baselines_times_seeds() -> None:
    assert (
        sum(len(_entry(t).baselines) * len(_entry(t).seeds) for t in REGISTRY)
        == N_EXPERIMENTS
    )
    assert len(ALL) == N_EXPERIMENTS
    assert len({ident for _, _, ident in ALL}) == N_EXPERIMENTS
    assert len({f"{mod}.{fn}" for mod, fn, _ in ALL}) == N_EXPERIMENTS


def test_every_registry_triple_has_exactly_one_function() -> None:
    want = {
        (t, b, s)
        for t in REGISTRY
        for b in _entry(t).baselines
        for s in _entry(t).seeds
    }
    assert {ident for _, _, ident in ALL} == want


def test_names_are_sequential_within_a_task() -> None:
    for m in tasks.task_modules():
        names = [fn for f, fn, _ in ALL if f == m]
        assert names == [f"exp{i:03d}" for i in range(len(names))]


def test_numbering_does_not_depend_on_other_tasks() -> None:
    full = gen.plan(REGISTRY)
    for task, entry in REGISTRY.items():
        m = gen.module_of(task)
        assert gen.plan({task: entry}) == {m: full[m]}


def test_functions_are_exactly_the_index() -> None:
    for m in tasks.task_modules():
        mod = importlib.import_module(f"{tasks.__name__}.{m}")
        funcs = {
            n
            for n, v in cast("dict[str, object]", vars(mod)).items()
            if n.startswith("exp") and callable(v)
        }
        assert funcs == set(cast("dict[str, object]", mod.INDEX))


def test_multi_seed_and_hidden_registry_facts() -> None:
    seeds = {t: _entry(t).seeds for t in REGISTRY}
    assert sum(len(s) > 1 for s in seeds.values()) == 13
    assert seeds["optimization-nas"] == (0, 1, 2, 3, 4)
    assert sum(s == (42,) for s in seeds.values()) == 126
    assert sum(len(_entry(t).hidden) for t in REGISTRY) == 134
    for t in REGISTRY:
        assert set(_entry(t).hidden) <= set(_entry(t).settings), t


def test_resolve_rejects_unknown_names() -> None:
    mod = ALL[0][0]
    for bad in ("nope.exp000", mod, f"{mod}.exp999", f"{mod}.", f"{mod}.INDEX"):
        with pytest.raises(KeyError):
            _ = tasks.resolve(bad)


def test_the_package_import_is_lazy() -> None:
    mods = [m for m in sys.modules if m.startswith(f"{tasks.__name__}.")]
    saved = {m: sys.modules.pop(m) for m in mods}
    try:
        _ = importlib.reload(tasks)
        _ = tasks.task_modules()
        assert not [m for m in sys.modules if m.startswith(f"{tasks.__name__}.")]
    finally:
        sys.modules.update(saved)


def test_list_prints_dotted_paths(capsys: pytest.CaptureFixture[str]) -> None:
    mod, fn, (task, baseline, seed) = ALL[0]
    want = [
        f"{tasks.__name__}.{f}.{n}  {t}  {bl}  {sd}"
        for f, n, (t, bl, sd) in ALL
        if t == task
    ]
    assert gen.main(["list", "--task", task]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines == want
    assert lines[0] == f"{tasks.__name__}.{mod}.{fn}  {task}  {baseline}  {seed}"
    assert len(gen.list_rows()) == N_EXPERIMENTS


def test_module_of_maps_dashes_and_rejects_non_identifiers() -> None:
    assert gen.module_of("llm-dllm-demask-strategy") == "llm_dllm_demask_strategy"
    assert tasks.module_of("llm-dllm-demask-strategy") == "llm_dllm_demask_strategy"
    for bad in ("1-task", "Upper-case", "has.dot", "class"):
        with pytest.raises(ValueError, match="not a module name"):
            _ = gen.module_of(bad)


def test_plan_rejects_two_tasks_with_one_module_name() -> None:
    with pytest.raises(ValueError, match="share"):
        _ = gen.plan(
            {
                "a-b": TaskEntry(baselines=("x",), settings={}),
                "a_b": TaskEntry(baselines=("y",), settings={}),
            },
        )


def test_plan_sorts_by_task_baseline_order_then_seed() -> None:
    reg = {
        "x-b": TaskEntry(baselines=("z", "a"), settings={}, seeds=(2, 1)),
        "x-a": TaskEntry(baselines=("m",), settings={}),
    }
    assert gen.plan(reg) == {
        "x_a": [("x-a", "m", 42)],
        "x_b": [
            ("x-b", "z", 1),
            ("x-b", "z", 2),
            ("x-b", "a", 1),
            ("x-b", "a", 2),
        ],
    }


def test_callable_type_of_resolve() -> None:
    fn: Callable[[], MlsbenchEval] = tasks.resolve(f"{ALL[0][0]}.{ALL[0][1]}")
    assert fn().experiment_name.endswith("/seed42")


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
