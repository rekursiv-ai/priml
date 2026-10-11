"""Tests for :mod:`priml.baselines.mlsbench.leaderboard` (synthetic fixtures only)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, override

import json
import sys
import types

import pytest

from priml.baselines.mlsbench import leaderboard as lb
from priml.baselines.mlsbench.leaderboard import (
    MetricKind,
    PublishedRow,
    TaskOverrides,
    Verdict,
    classify,
    compare,
    format_report,
    headline,
    read_leaderboard_csv,
    read_metrics_file,
)


if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class _Task:
    name: str
    rows: Mapping[str, PublishedRow]

    def row(self, name: str) -> PublishedRow:
        return self.rows[name]


def _task(values: dict[str, float], spread: dict[str, float] | None = None) -> _Task:
    return _Task("t", {"r": PublishedRow("r", values, spread or {})})


def _one(
    task: _Task,
    runs: list[dict[str, float]],
    overrides: TaskOverrides = TaskOverrides(),  # noqa: B008
) -> lb.Comparison:
    (c,) = compare(task, "r", runs, overrides).comparisons
    return c


def test_classify() -> None:
    assert classify("avg_steps_x") is MetricKind.EXACT
    assert classify("steps_x") is MetricKind.EXACT
    assert classify("num_tokens_x") is MetricKind.EXACT
    assert classify("accuracy_x") is MetricKind.ACCURACY
    assert classify("pass_at_1_x") is MetricKind.ACCURACY
    assert classify("gen_ppl_x") is MetricKind.CONTINUOUS
    assert classify("weird") is MetricKind.CONTINUOUS


def test_accuracy_boundary() -> None:
    t = _task({"accuracy_s": 0.322})
    assert _one(t, [{"accuracy_s": 0.342}]).verdict is Verdict.OK
    assert _one(t, [{"accuracy_s": 0.302}]).verdict is Verdict.OK
    c = _one(t, [{"accuracy_s": 0.3421}])
    assert c.verdict is Verdict.DIVERGED
    assert c.delta == pytest.approx(0.0201)


def test_accuracy_scale_override() -> None:
    t = _task({"accuracy_s": 32.2})
    o = TaskOverrides(scale={"accuracy_s": 100.0})
    assert _one(t, [{"accuracy_s": 34.2}], overrides=o).verdict is Verdict.OK
    assert _one(t, [{"accuracy_s": 34.3}], overrides=o).verdict is Verdict.DIVERGED


def test_exact_off_by_one_diverges() -> None:
    t = _task({"avg_steps_s": 256.0})
    assert _one(t, [{"avg_steps_s": 256.0}]).verdict is Verdict.OK
    c = _one(t, [{"avg_steps_s": 255.0}])
    assert c.verdict is Verdict.DIVERGED
    assert c.atol == 0.0


def test_continuous_needs_measured_noise() -> None:
    t = _task({"gen_ppl_s": 100.0})
    c = _one(t, [{"gen_ppl_s": 100.0}])
    assert c.verdict is Verdict.UNVERIFIED
    assert "noise not measured" in c.reason
    o = TaskOverrides(continuous_atol={"gen_ppl_s": 5.0})
    assert _one(t, [{"gen_ppl_s": 105.0}], overrides=o).verdict is Verdict.OK
    assert _one(t, [{"gen_ppl_s": 105.5}], overrides=o).verdict is Verdict.DIVERGED


def test_kind_override() -> None:
    t = _task({"gen_ppl_s": 10.0})
    o = TaskOverrides(kinds={"gen_ppl_s": MetricKind.EXACT})
    assert _one(t, [{"gen_ppl_s": 10.5}], overrides=o).verdict is Verdict.DIVERGED


def test_missing_values_unverified() -> None:
    rep = compare(
        _task({"accuracy_s": 0.3, "accuracy_u": 0.4}),
        "r",
        [{"accuracy_s": 0.3, "extra_s": 1.0}],
    )
    by = {c.column: c for c in rep.comparisons}
    assert by["accuracy_s"].verdict is Verdict.OK
    assert by["accuracy_u"].verdict is Verdict.UNVERIFIED
    assert by["accuracy_u"].reason == "not measured"
    assert by["extra_s"].verdict is Verdict.UNVERIFIED
    assert by["extra_s"].reason == "no published value"
    assert rep.verdict is Verdict.UNVERIFIED
    assert compare(_task({"accuracy_s": 0.3}), "r", []).verdict is Verdict.UNVERIFIED


def test_unpublished_columns_neither_fail_nor_hold_back_the_row() -> None:
    # llm-pretrain parsers also emit hidden-split metrics the leaderboard omits
    rep = compare(
        _task({"accuracy_s": 0.3}),
        "r",
        [{"accuracy_s": 0.3, "piqa_s": 0.7, "workspace_changed_s": 1.0}],
    )
    by = {c.column: c for c in rep.comparisons}
    assert by["piqa_s"].verdict is Verdict.UNVERIFIED
    assert [c.column for c in rep.comparisons] == [
        "accuracy_s",
        "piqa_s",
        "workspace_changed_s",
    ]
    assert rep.verdict is Verdict.OK


def test_multi_seed() -> None:
    t = _task({"accuracy_s": 0.30}, {"accuracy_s": 0.03})
    # single run with published spread: 0.03 is wider than the 0.02 default
    assert _one(t, [{"accuracy_s": 0.33}]).verdict is Verdict.OK
    c = _one(t, [{"accuracy_s": 0.34}])
    assert c.verdict is Verdict.DIVERGED
    assert c.kind is MetricKind.MULTI_SEED
    # several runs: mean, and sample std widens the tolerance
    runs = [{"accuracy_s": 0.20}, {"accuracy_s": 0.40}, {"accuracy_s": 0.36}]
    c = _one(_task({"accuracy_s": 0.30}), runs)
    assert c.measured == pytest.approx(0.32)
    assert c.verdict is Verdict.OK  # std ~0.107
    runs = [{"accuracy_s": 0.50}, {"accuracy_s": 0.50}]
    assert _one(_task({"accuracy_s": 0.30}), runs).verdict is Verdict.DIVERGED


def test_multi_seed_exact_stays_exact() -> None:
    t = _task({"avg_steps_s": 256.0}, {"avg_steps_s": 10.0})
    c = _one(t, [{"avg_steps_s": 250.0}, {"avg_steps_s": 250.0}])
    assert c.kind is MetricKind.EXACT
    assert c.verdict is Verdict.DIVERGED


def test_headline_excludes_diverged() -> None:
    ok = compare(_task({"accuracy_s": 0.3}), "r", [{"accuracy_s": 0.3}])
    bad = compare(_task({"accuracy_s": 0.3}), "r", [{"accuracy_s": 0.9}])
    un = compare(_task({"accuracy_s": 0.3}), "r", [])
    assert (ok.verdict, bad.verdict, un.verdict) == (
        Verdict.OK,
        Verdict.DIVERGED,
        Verdict.UNVERIFIED,
    )
    assert headline([ok, bad, un, ok]) == (2, 1, 1)
    assert "exceeds" in bad.comparisons[0].reason


def test_format_report() -> None:
    rep = compare(_task({"accuracy_s": 0.3}), "r", [{"accuracy_s": 0.9}])
    text = format_report([rep])
    assert "[diverged] t / r" in text
    assert "accuracy_s" in text
    assert "headline: ok=0 diverged=1 unverified=0" in text


_H = "timestamp,model,is_final,seed,"
_T = "2026-01-01T00:00:00+00:00"


def _row(name: str, seed: str | int, cells: str, final: str = "true") -> str:
    return f"{_T},baseline:{name},{final},{seed},{cells}\n"


def test_read_csv_single_seed_and_blank_cells(tmp_path: Path) -> None:
    p = tmp_path / "leaderboard.csv"
    p.write_text(
        _H
        + "accuracy_a,avg_steps_a\n"
        + _row("x", 42, "0.5,10")
        + _row("y", 42, ",12"),
    )
    x, y = read_leaderboard_csv(p)
    assert x.values == {"accuracy_a": 0.5, "avg_steps_a": 10.0}
    assert x.spread == {}
    assert x.seeds == {42: {"accuracy_a": 0.5, "avg_steps_a": 10.0}}
    assert y.values == {"avg_steps_a": 12.0}


def test_read_csv_multi_seed_uses_the_mean_row_and_the_seed_std(tmp_path: Path) -> None:
    p = tmp_path / "leaderboard.csv"
    p.write_text(
        _H
        + "score\n"
        + _row("x", 1, "0.2")
        + _row("x", 2, "0.4")
        + _row("x", 3, "0.6")
        + _row("x", "mean", "0.4001")
        + _row("z", 1, "1.0")
        + _row("z", 2, "3.0"),
    )
    x, z = read_leaderboard_csv(p)
    assert x.values == {"score": 0.4001}  # the published mean row wins
    assert x.spread["score"] == pytest.approx(0.2)
    assert z.values == {"score": 2.0}  # no mean row: mean of the seeds
    assert set(z.seeds) == {1, 2}


def test_read_csv_flags_unfinished_baselines(tmp_path: Path) -> None:
    p = tmp_path / "leaderboard.csv"
    p.write_text(_H + "a\n" + _row("x", 42, "1", final="false"))
    assert read_leaderboard_csv(p)[0].is_final is False


@pytest.mark.parametrize(
    "text",
    [
        "name,a\nx,1\n",  # the old wide format
        _H + "a\n" + _row("x", 42, "abc"),
        _H + "a\n" + _row("x", 42, "1") + _row("x", 42, "2"),
        _H + "a\n" + _row("x", 42, "1,2"),
        _H + "a\n" + _row("x", 42, "nan"),
        _H + "a\n" + _row("x", "avg", "1"),
        _H + "a\n" + f"{_T},x,true,42,1\n",
        _H + "a\n" + _row("x", 42, "1", final="maybe"),
    ],
)
def test_read_csv_rejects(tmp_path: Path, text: str) -> None:
    p = tmp_path / "l.csv"
    p.write_text(text)
    with pytest.raises(ValueError):  # noqa: PT011
        read_leaderboard_csv(p)


def test_read_metrics_file(tmp_path: Path) -> None:
    (tmp_path / "metrics.json").write_text(
        json.dumps({"eval/accuracy_a": 0.5, "train/loss": 1.0}),
    )
    assert read_metrics_file(tmp_path) == {"accuracy_a": 0.5}


def test_main(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    @dataclass(frozen=True)
    class _Spec(_Task):
        baselines: tuple[_B, ...] = ()

    spec = _Spec("t", {"r": PublishedRow("r", {"accuracy_s": 0.3}, {})}, (_B("r"),))
    fake = types.ModuleType("priml.baselines.mlsbench.task")

    def _load_task(name: str, root: Path | None = None) -> object:
        del name, root
        return spec

    vars(fake)["load_task"] = _load_task
    monkeypatch.setitem(sys.modules, "priml.baselines.mlsbench.task", fake)
    runs = tmp_path / "runs"
    (runs / "r" / "seed42").mkdir(parents=True)
    (runs / "r" / "seed42" / "metrics.json").write_text(
        json.dumps({"eval/accuracy_s": 0.3}),
    )
    assert lb.main(["--task", "t", "--runs", str(runs)]) == 0
    (runs / "r" / "seed42" / "metrics.json").write_text(
        json.dumps({"eval/accuracy_s": 0.9}),
    )
    assert lb.main(["--task", "t", "--runs", str(runs), "--row", "r"]) == 1
    assert "diverged" in capsys.readouterr().out
    # a base_dir is accepted as well as the runs directory itself
    nested = tmp_path / "base" / "runs" / "mlsbench" / "t" / "r" / "seed42"
    nested.mkdir(parents=True)
    (nested / "metrics.json").write_text(json.dumps({"eval/accuracy_s": 0.3}))
    assert lb.main(["--task", "t", "--runs", str(tmp_path / "base"), "--row", "r"]) == 0
    # a candidate's runs are judged against the published row
    cand = tmp_path / "base" / "runs" / "mlsbench" / "t" / "my-idea" / "seed42"
    cand.mkdir(parents=True)
    (cand / "metrics.json").write_text(json.dumps({"eval/accuracy_s": 0.9}))
    _ = capsys.readouterr()
    argv = ["--task", "t", "--runs", str(tmp_path / "base"), "--row", "r"]
    assert lb.main([*argv, "--candidate", "my-idea"]) == 1
    out = capsys.readouterr().out
    assert "0.9000" in out
    assert "0.3000" in out
    assert lb.main([*argv, "--candidate", "nobody"]) == 0  # nothing measured
    with pytest.raises(SystemExit):
        _ = lb.main(["--task", "t", "--runs", str(runs), "--candidate", "my-idea"])
    with pytest.raises(SystemExit):
        _ = lb.main(["--all", "--runs", str(runs), "--candidate", "my-idea"])


# ---- whole-benchmark aggregation (synthetic runs) ----


@dataclass(frozen=True)
class _B:
    name: str


@dataclass(frozen=True)
class _Spec:
    name: str
    rows: Mapping[str, PublishedRow]
    seeds: tuple[int, ...] = (42,)

    @property
    def baselines(self) -> tuple[_B, ...]:
        return tuple(_B(n) for n in self.rows)

    def row(self, name: str) -> PublishedRow:
        return self.rows[name]


def _write_run(root: Path, task: str, row: str, seed: int, **m: float) -> None:
    d = root / task / row / f"seed{seed}"
    d.mkdir(parents=True)
    (d / "metrics.json").write_text(json.dumps({f"eval/{k}": v for k, v in m.items()}))


def _specs() -> list[_Spec]:
    return [
        _Spec(
            "a-one",
            {
                "good": PublishedRow("good", {"accuracy_s": 0.5}, {}),
                "bad": PublishedRow("bad", {"accuracy_s": 0.5}, {}),
                "idle": PublishedRow("idle", {"accuracy_s": 0.5}, {}),
            },
        ),
        _Spec(
            "b-multi",
            {
                "full": PublishedRow("full", {"score_s": 1.0}, {"score_s": 0.3}),
                "part": PublishedRow("part", {"score_s": 1.0}, {"score_s": 0.3}),
                "far": PublishedRow("far", {"score_s": 1.0}, {"score_s": 0.3}),
            },
            seeds=(42, 123, 456),
        ),
        _Spec("c-idle", {"x": PublishedRow("x", {"accuracy_s": 0.5}, {})}),
    ]


def test_aggregate_counts_per_task_and_overall(tmp_path: Path) -> None:
    _write_run(tmp_path, "a-one", "good", 42, accuracy_s=0.51)
    _write_run(tmp_path, "a-one", "bad", 42, accuracy_s=0.9)
    for seed, v in ((42, 0.8), (123, 1.1), (456, 1.2)):
        _write_run(tmp_path, "b-multi", "full", seed, score_s=v)
    _write_run(tmp_path, "b-multi", "part", 42, score_s=1.0)
    for seed in (42, 123, 456):
        _write_run(tmp_path, "b-multi", "far", seed, score_s=5.0)
    one, multi, idle = lb.aggregate(_specs(), tmp_path)
    assert one.counts == (1, 1, 1)  # good ok, bad diverged, idle never run
    assert [r.verdict for r in one.reports] == [
        Verdict.OK,
        Verdict.DIVERGED,
        Verdict.UNVERIFIED,
    ]
    assert one.reports[2].seed_count == 0
    assert multi.counts == (1, 1, 1)
    full, part, far = multi.reports
    # mean 1.0333 vs published 1.0: inside the published spread 0.3
    assert (full.verdict, full.seed_count) == (Verdict.OK, 3)
    assert full.comparisons[0].kind is MetricKind.MULTI_SEED
    # a single seed of a three-seed task never reaches ok
    assert part.verdict is Verdict.UNVERIFIED
    assert multi.missing_seeds == {"part": (123, 456)}
    assert far.verdict is Verdict.DIVERGED
    assert idle.counts == (0, 0, 1)
    assert lb.overall([one, multi, idle]) == (2, 2, 3)


def test_a_partial_multi_seed_row_that_diverges_stays_diverged(tmp_path: Path) -> None:
    _write_run(tmp_path, "b-multi", "part", 42, score_s=9.0)
    (_, multi, _) = lb.aggregate(_specs(), tmp_path)
    assert multi.reports[1].verdict is Verdict.DIVERGED


def test_a_row_without_a_published_cell_source_is_unverified(tmp_path: Path) -> None:
    spec = _Spec("t-x", {})
    spec_bad = _Spec("t-x", {"r": PublishedRow("r", {}, {})})

    class _NoRow(_Spec):
        @property
        @override
        def baselines(self) -> tuple[_B, ...]:
            return (_B("ghost"),)

    _write_run(tmp_path, "t-x", "ghost", 42, accuracy_s=0.5)
    (s,) = lb.aggregate([_NoRow("t-x", {})], tmp_path)
    assert s.reports[0].verdict is Verdict.UNVERIFIED
    assert s.reports[0].seed_count == 1
    assert lb.aggregate([spec], tmp_path)[0].reports == ()
    assert lb.aggregate([spec_bad], tmp_path)[0].reports[0].seed_count == 0


def test_discover_and_seed_layouts(tmp_path: Path) -> None:
    _write_run(tmp_path, "a-one", "good", 7, accuracy_s=0.5)
    _write_run(tmp_path, "z-two", "r", 1, accuracy_s=0.5)
    (tmp_path / "a-one" / "good" / "seedX").mkdir()
    assert lb.discover_tasks(tmp_path) == ["a-one", "z-two"]
    assert list(lb.find_seed_runs(tmp_path, "a-one", "good")) == [7]
    assert lb.find_seed_runs(tmp_path, "a-one", "nope") == {}


def test_format_summary(tmp_path: Path) -> None:
    _write_run(tmp_path, "b-multi", "part", 42, score_s=1.0)
    text = lb.format_summary(lb.aggregate(_specs(), tmp_path), verbose=True)
    lines = text.splitlines()
    assert lines[0].split() == ["task", "ok", "diverged", "unverified"]
    assert lines[-1] == "overall: tasks=3 rows=7 ok=0 diverged=0 unverified=7"
    assert "missing seeds [123, 456]" in text


def test_main_aggregates_a_base_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from priml.baselines.mlsbench import task as task_mod  # noqa: PLC0415

    specs = {s.name: s for s in _specs()}

    def _load_task(name: str, root: Path) -> object:
        del root
        return specs[name]

    def _task_names(root: Path) -> tuple[str, ...]:
        del root
        return tuple(specs)

    monkeypatch.setattr(task_mod, "load_task", _load_task)
    monkeypatch.setattr(task_mod, "task_names", _task_names)
    runs = tmp_path / "runs" / "mlsbench"
    _write_run(runs, "a-one", "good", 42, accuracy_s=0.5)
    assert lb.main(["--all", "--runs", str(tmp_path), "--root", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "a-one" in out
    assert "b-multi" not in out
    assert "overall: tasks=1 rows=3 ok=1 diverged=0 unverified=2" in out
    assert lb.main(["--runs", str(tmp_path), "--include-missing"]) == 0
    assert (
        "overall: tasks=3 rows=7 ok=1 diverged=0 unverified=6"
        in capsys.readouterr().out
    )
    _write_run(runs, "a-one", "bad", 42, accuracy_s=0.99)
    assert lb.main(["--all", "--runs", str(runs)]) == 1
    with pytest.raises(SystemExit):
        _ = lb.main(["--all", "--task", "a-one", "--runs", str(runs)])


def test_a_partial_run_is_judged_against_its_own_published_seeds() -> None:
    # published mean 0.33 with spread 0.02, but seed 42 alone scored 0.99
    row = PublishedRow(
        "r",
        {"score_s": 0.33},
        {"score_s": 0.02},
        {42: {"score_s": 0.99}, 123: {"score_s": 0.0}, 456: {"score_s": 0.0}},
    )
    task = _Task("t", {"r": row})
    run = [{"score_s": 0.99}]
    assert lb.compare(task, "r", run).verdict is Verdict.DIVERGED
    got = lb.compare(task, "r", run, seeds=[42])
    assert got.verdict is Verdict.OK
    assert got.comparisons[0].published == pytest.approx(0.99)
    # an unpublished seed falls back to the all-seed mean
    assert lb.compare(task, "r", run, seeds=[7]).verdict is Verdict.DIVERGED
