"""Check budgets, mechanism holdouts, and causal scoring in the release suite."""

from __future__ import annotations

from dataclasses import field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Self, override

import json
import sys

from configgle import Fig, LateBound, Makes, Makes2
from torch import Tensor
from treekle import ReadError, from_plain, json_loads, to_plain

import pytest
import torch

from priml import launch
from priml.baselines.oscillator.benchmark import BenchmarkRun
from priml.baselines.oscillator.data import ForecastData, InitialStates, Oscillator
from priml.baselines.oscillator.discovery import SparseSimulator
from priml.baselines.oscillator.experiments import exp040
from priml.baselines.oscillator.forces import CorrectedOscillator, WeightedForce
from priml.baselines.oscillator.mechanisms import SineForce, SystemFamily
from priml.baselines.oscillator.pretrained import PretrainedForecast
from priml.baselines.oscillator.pretrained_test import _PointForecast
from priml.baselines.oscillator.suite import (
    BenchmarkRecipe,
    BenchmarkSuite,
    BudgetStudy,
    ReferenceBudget,
    suite_digest,
)
from priml.baselines.oscillator.train_step import DLinearForecast, ForecastGeometry


if TYPE_CHECKING:
    from priml.baselines.oscillator.data import Windows
    from priml.baselines.oscillator.train_step import FitReport


def test_release_holdouts_and_planned_counts_are_explicit() -> None:
    cfg = exp040()
    suite = cfg.make()
    assert {family.name for family in cfg.development}.isdisjoint(
        family.name for family in cfg.audit
    )
    assert len(list(suite.tasks("audit"))) == 16
    assert len(list(suite.plan("audit"))) == 544
    assert [budget.train + budget.validation for budget in cfg.budgets] == [96, 384]
    for recipe in cfg.recipes:
        assert recipe.study.seed is None
        if recipe.access == "zero-shot":
            assert recipe.seeds == [2021]


def test_suite_geometry_reaches_methods_before_their_first_finalization() -> None:
    cfg = exp040()
    cfg.data.seq_len = 8
    cfg.data.pred_len = 3
    cfg.audit = cfg.audit[:1]
    cfg.audit[0].instances = 1
    cfg.budgets = cfg.budgets[:1]
    for recipe in cfg.recipes:
        recipe.seeds = recipe.seeds[:1]
    suite = cfg.make()
    for _, _, _, planned in suite.plan("audit"):
        study = planned.finalized()
        if isinstance(study.method, ForecastGeometry):
            assert (study.method.seq_len, study.method.pred_len) == (8, 3)
        assert (study.data.seq_len, study.data.pred_len) == (8, 3)


def test_audit_force_configs_do_not_alias_development() -> None:
    cfg = exp040()
    drag = cfg.development[2].reference.corrections[0]
    before = drag.copy_tree()
    held_out = cfg.audit[0].reference.corrections[0]
    held_out.coefficient += 1
    assert drag == before
    assert held_out.term is not drag.term


def test_study_resolves_data_geometry_before_finalizing_the_method() -> None:
    cfg = BudgetStudy.Config()
    cfg.data = _ParentBoundData.Config()
    cfg.data.seq_len = 6
    cfg.data.pred_len = 3
    cfg.scoring_trajectories = 4
    cfg.seed = 7
    method = cfg.method = DLinearForecast.Config()
    method.updates = 2
    study = cfg.make()
    assert isinstance(study.config.method, DLinearForecast.Config)
    assert study.config.data.pred_len == study.config.method.pred_len == 4
    assert study.run()["timesteps_per_query"] == 10
    assert cfg.data.pred_len == 3


@pytest.mark.parametrize("finalized", [False, True])
def test_template_owners_run_their_own_finalizers(finalized: bool) -> None:
    cfg = _DevelopmentSuite.Config()
    cfg.update(tiny_suite())
    family = _CubicFamily.Config()
    family.coefficient_scale = (1.0, 1.0)
    if finalized:
        cfg = cfg.finalized()
        family = family.finalized()
    suite = cfg.make()
    assert suite.config.phase == "development"
    sampled = family.make()
    assert sampled.draw(0).cubic == 0.4
    assert sampled.bound_root is sampled
    planned = next(suite.plan("audit"))[3]
    assert planned.make().config.data.seq_len == cfg.data.seq_len


@pytest.mark.parametrize("diagnostic", ["oracle", "refined"])
def test_diagnostics_reject_nonfinite_scores_before_aggregating(
    diagnostic: str,
) -> None:
    cfg = tiny_suite()
    if diagnostic == "refined":
        reference = _NonfiniteRefinement.Config()
        reference.cubic = 0.4
        cfg.audit[0].reference = reference
    suite = cfg.make()
    system_id, data = next(suite.tasks("audit"))
    if diagnostic == "oracle":
        data.train.amplitude = (1e20, 2e20)
    with pytest.raises(ValueError, match="finite"):
        suite.diagnose(system_id, data, budget="small")


def test_enclosing_finalization_preserves_recipe_and_task_boundaries() -> None:
    cfg = _SuiteOwner.Config()
    cfg.suite.update(tiny_suite())
    recipe = _DerivedRecipe.Config()
    recipe.update(cfg.suite.recipes[0])
    study = _BoundStudy.Config()
    study.update(recipe.study)
    recipe.study = study
    cfg.suite.recipes[0] = recipe
    owner = cfg.make()
    resolved = owner.suite.config.recipes[0]
    assert resolved.seeds == [11, 12, 17]
    assert isinstance(resolved.study, _BoundStudy.Config)
    assert resolved.study.bindings == []
    planned = next(owner.suite.plan("audit"))[3].finalized()
    assert isinstance(planned, _BoundStudy.Config)
    assert planned.bindings == [(6, 4, 11)]
    assert recipe.seeds == [11, 12]
    assert study.bindings == []


def test_native_launcher_rejects_flags_before_writing_or_fitting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_output(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("A native flag reached output creation.")

    def keep_logging(level: str) -> None:
        del level

    monkeypatch.setattr(launch, "setup_logging", keep_logging)
    monkeypatch.setattr(Path, "mkdir", reject_output)
    monkeypatch.setattr(
        sys,
        "argv",
        ["priml", "priml.baselines.oscillator.experiments.exp040", "--help"],
    )
    with pytest.raises(TypeError, match="Path"):
        launch.main()


def test_data_budgets_are_nested_and_every_method_receives_the_same_task() -> None:
    cfg = tiny_suite()
    before = cfg.copy_tree()
    state = torch.random.get_rng_state()
    plan = list(cfg.make().plan("audit"))
    first = plan[0][3]
    second = next(
        study
        for _, system, _, study in plan
        if system == plan[0][1] and study.data.train.count == 8
    )
    small = first.data.make().windows(
        first.data.train,
        simulator=first.data.reference.make(),
    )
    large = second.data.make().windows(
        second.data.train,
        simulator=second.data.reference.make(),
    )
    assert torch.equal(small.media, large.media[: len(small.media)])
    assert torch.equal(small.label, large.label[: len(small.label)])
    torch.testing.assert_close(
        first.make().score_scale(small),
        second.make().score_scale(large),
    )
    same_task = [
        study.data
        for _, system, _, study in plan
        if system == plan[0][1] and study.data.train.count == 4
    ]
    assert all(data == same_task[0] for data in same_task)
    assert cfg == before
    assert torch.equal(torch.random.get_rng_state(), state)


def test_extra_reference_data_cannot_change_the_scoring_denominator(
    tmp_path: Path,
) -> None:
    suite = tiny_suite().make()
    records, diagnostics = suite.run(tmp_path / "audit", phase="audit")
    assert all(record["status"] == "complete" for record in records)
    assert len(diagnostics) == 6
    assert all(row["oracle_scaled_mse"] == 0 for row in diagnostics)
    small = records[0]
    large = next(
        row
        for row in records
        if row["system_id"] == small["system_id"]
        and row["budget"] == "large"
        and row["method"] == small["method"]
        and row["model_seed"] == small["model_seed"]
    )
    assert small["result"] is not None
    assert large["result"] is not None
    assert small["result"]["scores"] == large["result"]["scores"]
    assert small["result"]["reference_budget"] == 9
    assert large["result"]["reference_budget"] == 18
    with pytest.raises(FileExistsError):
        suite.run(tmp_path / "audit", phase="audit")


@pytest.mark.parametrize("phase", ["audit", "development"])
@pytest.mark.parametrize("failed", [False, True])
def test_native_suite_run_uses_config_defaults_and_writes_every_attempt(
    tmp_path: Path,
    phase: Literal["development", "audit"],
    failed: bool,
) -> None:
    cfg = tiny_suite()
    assert cfg.phase == "audit"
    cfg.phase = phase
    cfg.working_dir = tmp_path / "native"
    cfg.audit = cfg.audit[:1]
    cfg.audit[0].instances = 1
    cfg.budgets = cfg.budgets[:1]
    cfg.recipes[0].seeds = [11]
    if failed:
        cfg.recipes[0].study.method = TypeErrorForecast.Config()
    before = cfg.copy_tree()
    if failed:
        with pytest.raises(RuntimeError, match="failed"):
            cfg.make().run()
    else:
        cfg.make().run()
    report = json_loads(
        (cfg.working_dir / "results.json").read_text(),
        dict[str, object],
    )
    assert report["version"] == cfg.version
    assert report["phase"] == phase
    runs = from_plain(report["runs"], list[BenchmarkRun])
    assert [run["status"] for run in runs] == [
        "failed" if failed else "complete",
        "complete",
    ]
    for run in runs:
        assert (
            json_loads(
                (cfg.working_dir / f"{run['run_id']}.json").read_text(),
                BenchmarkRun,
            )
            == run
        )
    diagnostics = from_plain(report["diagnostics"], list[dict[str, object]])
    assert len(diagnostics) == 1
    assert diagnostics[0]["oracle_scaled_mse"] == 0
    summary = from_plain(report["summary"], list[dict[str, object]])
    assert len(summary) == 4
    for row in summary:
        expected_failures = int(failed and row["method"] == "dlinear")
        assert row["failed_runs"] == expected_failures
        assert (row["scaled_mse"] is None) == bool(expected_failures)
    assert (cfg.working_dir / "protocol.txt").read_text() == cfg.pformat(
        hide_default_values=False,
        mask_memory_addresses=True,
    ) + "\n"
    assert cfg == before
    with pytest.raises(FileExistsError):
        cfg.make().run()


def test_nonfinite_model_runs_are_retained_as_failures(tmp_path: Path) -> None:
    cfg = tiny_suite()
    cfg.audit = cfg.audit[:1]
    cfg.audit[0].instances = 1
    cfg.budgets = cfg.budgets[:1]
    cfg.recipes = cfg.recipes[:1]
    cfg.recipes[0].seeds = [11]
    pipeline = _PointForecast.Config()
    pipeline.nonfinite = True
    method = PretrainedForecast.Config()
    method.pipeline = pipeline
    cfg.recipes[0].study.method = method
    records, _ = cfg.make().run(tmp_path / "failed", phase="audit")
    assert len(records) == 1
    assert records[0]["status"] == "failed"
    assert records[0]["result"] is None
    assert records[0]["error"] is not None


def test_adapter_type_errors_are_recorded_and_later_recipes_continue(
    tmp_path: Path,
) -> None:
    cfg = tiny_suite()
    cfg.audit = cfg.audit[:1]
    cfg.audit[0].instances = 1
    cfg.budgets = cfg.budgets[:1]
    cfg.recipes[0].seeds = [11]
    cfg.recipes[0].study.method = TypeErrorForecast.Config()
    records, _ = cfg.make().run(tmp_path / "attempts", phase="audit")
    assert [record["status"] for record in records] == ["failed", "complete"]
    assert records[0]["error"] == "TypeError: Upstream forecast shape changed."
    assert records[0]["result"] is None
    saved = json_loads(
        (tmp_path / "attempts" / f"{records[0]['run_id']}.json").read_text(),
        dict[str, object],
    )
    assert saved["error"] == records[0]["error"]


def test_process_interrupts_leave_the_current_attempt_unrecorded(
    tmp_path: Path,
) -> None:
    cfg = tiny_suite()
    cfg.recipes[0].study.method = InterruptedForecast.Config()
    directory = tmp_path / "interrupted"
    with pytest.raises(KeyboardInterrupt):
        cfg.make().run(directory, phase="audit")
    assert not list(directory.glob("*--*.json"))


def test_diagnostic_failures_abort_before_attempting_methods(tmp_path: Path) -> None:
    cfg = tiny_suite()
    cfg.data.train = RepeatedInitialStates.Config()
    directory = tmp_path / "bad_task"
    with pytest.raises(ValueError, match="initial state"):
        cfg.make().run(directory, phase="audit")
    assert not list(directory.glob("*--*.json"))


@pytest.mark.parametrize("method", ["pretrained", "sparse"])
def test_resume_retains_attempts_and_rejects_changed_provenance(
    tmp_path: Path,
    method: str,
) -> None:
    cfg = tiny_suite()
    cfg.audit = cfg.audit[:1]
    cfg.audit[0].instances = 1
    if method == "pretrained":
        pretrained = PretrainedForecast.Config()
        pretrained.pipeline = _PointForecast.Config()
        cfg.recipes[0].study.method = pretrained
    else:
        sparse = SparseSimulator.Config()
        sparse.max_terms = 0
        cfg.recipes[0].study.method = sparse
    suite = cfg.make()
    directory = tmp_path / "run"
    records, _ = suite.run(directory, phase="audit", provenance={"source": "first"})
    retained = directory / f"{records[0]['run_id']}.json"
    original = retained.read_bytes()
    missing = directory / f"{records[-1]['run_id']}.json"
    missing.unlink()
    resumed, _ = suite.run(
        directory,
        phase="audit",
        resume=True,
        provenance={"source": "first"},
    )
    assert len(resumed) == len(records)
    assert resumed[0] == records[0]
    assert retained.read_bytes() == original
    assert missing.is_file()
    assert suite.summarize(resumed, phase="audit")
    with pytest.raises(ValueError, match="resume"):
        suite.run(
            directory,
            phase="audit",
            resume=True,
            provenance={"source": "changed"},
        )
    cfg.budgets[0].train += 1
    with pytest.raises(ValueError, match="resume"):
        cfg.make().run(
            directory,
            phase="audit",
            resume=True,
            provenance={"source": "first"},
        )
    (directory / "foreign--run.json").write_text(json.dumps(to_plain(records[0])))
    with pytest.raises(ValueError, match="unplanned"):
        suite.run(directory, phase="audit", resume=True, provenance={"source": "first"})


@pytest.mark.parametrize("bad", ["missing_count", "wrong_count", "run_id", "score"])
def test_resume_validates_core_fields_without_discarding_fit_extensions(
    tmp_path: Path,
    bad: str,
) -> None:
    cfg = tiny_suite()
    cfg.audit = cfg.audit[:1]
    cfg.audit[0].instances = 1
    cfg.budgets = cfg.budgets[:1]
    cfg.recipes = cfg.recipes[:1]
    cfg.recipes[0].seeds = [11]
    suite = cfg.make()
    directory = tmp_path / "resume"
    records, _ = suite.run(directory, phase="audit")
    path = directory / f"{records[0]['run_id']}.json"
    raw = json_loads(path.read_text(), dict[str, object])
    result = from_plain(raw["result"], dict[str, object])
    if bad == "missing_count":
        result.pop("reference_budget")
    elif bad == "wrong_count":
        result["reference_budget"] = "wrong"
    elif bad == "run_id":
        raw["run_id"] = 42
    else:
        result["scores"] = {"familiar": {"method": {"mse": "wrong"}}}
    raw["result"] = result
    path.write_text(json.dumps(to_plain(raw)))
    with pytest.raises(ReadError):
        suite.run(directory, phase="audit", resume=True)


def test_direct_resume_rejects_an_ambient_default_dtype_change(tmp_path: Path) -> None:
    suite = tiny_suite().make()
    directory = tmp_path / "dtype"
    suite.run(directory, phase="audit")
    original = torch.get_default_dtype()
    manifest = json_loads(
        (directory / "run-manifest.json").read_text(),
        dict[str, object],
    )
    assert manifest["torch_default_dtype"] == str(original)
    try:
        torch.set_default_dtype(
            torch.float64 if original == torch.float32 else torch.float32,
        )
        with pytest.raises(ValueError, match="environment"):
            suite.run(directory, phase="audit", resume=True)
    finally:
        torch.set_default_dtype(original)


@pytest.mark.parametrize(
    "counts",
    [
        [(4, 5), (8, 4)],
        [(8, 4), (4, 5)],
    ],
)
def test_crossing_reference_budgets_are_rejected(counts: list[tuple[int, int]]) -> None:
    cfg = tiny_suite()
    for budget, (train, validation) in zip(cfg.budgets, counts, strict=True):
        budget.train, budget.validation = train, validation
    with pytest.raises(ValueError, match="nested"):
        cfg.make()


@pytest.mark.parametrize(
    "counts",
    [[(8, 10), (4, 5)], [(4, 5), (4, 10)], [(4, 5), (4, 5)]],
)
def test_nested_reference_budgets_allow_reordering_and_ties(
    counts: list[tuple[int, int]],
) -> None:
    cfg = tiny_suite()
    for budget, (train, validation) in zip(cfg.budgets, counts, strict=True):
        budget.train, budget.validation = train, validation
    assert cfg.make().config.budgets == cfg.budgets


def test_task_templates_finalize_once_after_every_task_input_is_assigned() -> None:
    cfg = tiny_suite()
    data = _BoundData.Config()
    data.update(cfg.data)
    cfg.data = data
    study = _BoundStudy.Config()
    study.update(cfg.recipes[0].study)
    cfg.recipes[0].study = study
    reference = _BoundReference.Config()
    reference.cubic = 0.4
    cfg.audit[0].reference = reference
    suite = cfg.make()
    stored = suite.config.recipes[0].study
    assert isinstance(stored, _BoundStudy.Config)
    assert stored.bindings == []
    _, _, _, planned = next(suite.plan("audit"))
    finalized = planned.finalized()
    assert isinstance(finalized, _BoundStudy.Config)
    assert finalized.bindings == [(6, 4, 11)]
    assert isinstance(finalized.data, _BoundData.Config)
    assert finalized.data.binding == (4, finalized.data.train.seed)
    assert finalized.data.finalizations == 1
    task_id, task = next(suite.tasks("audit"))
    assert isinstance(task.reference, _BoundReference.Config)
    drawn = task.reference.finalized()
    assert drawn.bound_cubic == drawn.cubic
    assert drawn.finalizations == 1
    assert suite.diagnose(task_id, task, budget="small")["oracle_scaled_mse"] == 0
    assert stored.bindings == []
    assert reference.bound_cubic == -1


@pytest.mark.parametrize("pred_len", [3, 4])
def test_diagnostics_use_parent_finalized_dynamics_splits_and_geometry(
    pred_len: int,
) -> None:
    cfg = tiny_suite()
    data = _ParentBoundData.Config()
    data.seq_len = 6
    data.pred_len = pred_len
    data.cheap.damping = 2.0
    before = data.copy_tree()
    diagnostics = cfg.make().diagnose("parent-bound", data, budget="small")
    assert diagnostics["oracle_scaled_mse"] == 0
    assert 0 < diagnostics["finer_grid_scaled_mse"] < cfg.integration_error_limit
    assert diagnostics["unique_initial_states"] == 4 + 5 + 6 + 7
    assert data == before


@pytest.mark.parametrize("template", ["study", "data", "family"])
def test_finalized_template_inputs_are_rejected(template: str) -> None:
    cfg = tiny_suite()
    if template == "study":
        cfg.recipes[0].study.finalize()
    elif template == "data":
        cfg.data.finalize()
    else:
        cfg.audit[0].reference.finalize()
    with pytest.raises(ValueError, match="unfinalized"):
        cfg.make()


@pytest.mark.parametrize(
    "bad",
    [
        "name",
        "duplicate_recipe",
        "baseline",
        "budget",
        "holdout_name",
        "holdout_law",
        "system_seed",
        "seed_range",
        "grid",
        "tolerance",
    ],
)
def test_invalid_suite_protocols_fail_before_running(bad: str) -> None:
    cfg = tiny_suite()
    if bad == "name":
        cfg.audit[0].name = "../bad"
    elif bad == "duplicate_recipe":
        cfg.recipes[1].name = cfg.recipes[0].name
    elif bad == "baseline":
        cfg.baseline = "missing"
    elif bad == "budget":
        cfg.budgets[1].name = cfg.budgets[0].name
    elif bad == "holdout_name":
        cfg.audit[0].name = cfg.development[0].name
    elif bad == "holdout_law":
        cfg.audit[0].reference = cfg.development[0].reference.copy_tree()
    elif bad == "system_seed":
        cfg.audit[0].seed = cfg.development[0].seed
    elif bad == "seed_range":
        cfg.audit[0].seed = 2**32
    elif bad == "grid":
        cfg.integration_refinement = 1
    else:
        cfg.integration_error_limit = float("inf")
    with pytest.raises(
        ValueError,
        match=r"labels|Recipes|Budget|families|prototype|seed|Integration",
    ):
        cfg.make()


def test_invalid_budget_recipe_and_scoring_prefix_are_rejected() -> None:
    for cfg in (
        ReferenceBudget.Config(train=0),
        BenchmarkRecipe.Config(seeds=[]),
        BenchmarkRecipe.Config(seeds=[1, 1]),
        BenchmarkRecipe.Config(seeds=[-1]),
    ):
        with pytest.raises(ValueError, match=r"budget|recipe|Recipe|Model"):
            cfg.make()
    cfg = BudgetStudy.Config()
    cfg.scoring_trajectories = 0
    with pytest.raises(ValueError, match="prefix"):
        cfg.make()


def test_protocol_digest_covers_recipes_budgets_and_holdout_equations() -> None:
    cfg = tiny_suite()
    first = suite_digest(cfg)
    cfg.audit[0].reference.cubic *= 2
    assert suite_digest(cfg) != first
    cfg = tiny_suite()
    cfg.budgets[0].train += 1
    assert suite_digest(cfg) != first
    cfg = tiny_suite()
    cfg.recipes[0].seeds.append(13)
    assert suite_digest(cfg) != first


def test_unknown_phase_is_rejected_before_drawing_tasks() -> None:
    with pytest.raises(ValueError, match="phase"):
        list(tiny_suite().make().tasks("unknown"))


@pytest.mark.parametrize("bad", ["duplicate_states", "continuation", "integration"])
def test_diagnostics_reject_invalid_task_generation(bad: str) -> None:
    cfg = tiny_suite()
    if bad == "duplicate_states":
        cfg.data.train = RepeatedInitialStates.Config()
    elif bad == "continuation":
        cfg.audit[0].reference = WrongContinuation.Config()
    else:
        cfg.integration_error_limit = 1e-20
    suite = cfg.make()
    system_id, data = next(suite.tasks("audit"))
    message = {
        "duplicate_states": "initial state",
        "continuation": "continue",
        "integration": "discrepancy",
    }[bad]
    with pytest.raises(ValueError, match=message):
        suite.diagnose(system_id, data, budget="small")


def tiny_suite() -> BenchmarkSuite.Config:
    cfg = BenchmarkSuite.Config()
    cfg.data.seq_len = 6
    cfg.data.pred_len = 3
    cfg.data.synthetic.count = 11
    cfg.data.test.count = 7
    cfg.data.shifted.count = 9
    cfg.budgets = [
        ReferenceBudget.Config(train=4, validation=5),
        ReferenceBudget.Config(name="large", train=8, validation=10),
    ]
    first = BenchmarkRecipe.Config()
    first.study.method = RepeatForecast.Config()
    first.seeds = [11, 12]
    second = BenchmarkRecipe.Config()
    second.name = "zero"
    second.study.method = RepeatForecast.Config(factor=0.0)
    second.seeds = [11]
    cfg.recipes = [first, second]
    cfg.development = [SystemFamily.Config(seed=10, instances=1)]
    family = SystemFamily.Config()
    family.name = "cubic"
    family.reference.cubic = 0.4
    family.seed = 100
    family.instances = 2
    other = SystemFamily.Config()
    other.name = "sine"
    other.seed = 200
    other.instances = 1
    other.reference.corrections = [
        WeightedForce.Config(term=SineForce.Config(), coefficient=0.8),
    ]
    cfg.audit = [family, other]
    cfg.summary.resamples = 20
    return cfg


class RepeatForecast:
    class Config(Fig["RepeatForecast"]):
        factor: float = 1.0
        """Scale the repeated last observation."""

    def __init__(self, config: Config) -> None:
        self.factor = config.factor
        self.horizon = 0

    def fit(
        self,
        train: Windows,
        *,
        validation: Windows,
        synthetic: Windows,
        simulator: Oscillator,
    ) -> FitReport:
        del validation, synthetic
        assert simulator.config.cubic == 0
        self.horizon = train.label.shape[1]
        return {"updates": 0, "best_update": 0, "pretrain_updates": 0}

    def predict(self, media: Tensor) -> Tensor:
        return self.factor * media[:, -1:].expand(
            len(media),
            self.horizon,
            media.shape[-1],
        )


class TypeErrorForecast(RepeatForecast):
    class Config(Makes["TypeErrorForecast"], RepeatForecast.Config):
        """Fail at the adapter boundary after fitting."""

    @override
    def predict(self, media: Tensor) -> Tensor:
        del media
        raise TypeError("Upstream forecast shape changed.")


class RepeatedInitialStates(InitialStates):
    class Config(Makes["RepeatedInitialStates"], InitialStates.Config):
        """Give multiple trajectories the same initial state."""

    @override
    def sample(self) -> Tensor:
        return torch.zeros(self.config.count, 2)


class WrongContinuation(CorrectedOscillator):
    class Config(Makes2["WrongContinuation"], CorrectedOscillator.Config):
        """Produce history continuations inconsistent with complete rollouts."""

    @override
    def forecast_history(
        self,
        media: Tensor,
        *,
        steps: int,
        substeps: int | None = None,
    ) -> Tensor:
        return -super().forecast_history(media, steps=steps, substeps=substeps)


class InterruptedForecast(RepeatForecast):
    class Config(Makes["InterruptedForecast"], RepeatForecast.Config):
        """Interrupt a method attempt without turning it into a failed score."""

    @override
    def predict(self, media: Tensor) -> Tensor:
        del media
        raise KeyboardInterrupt


class _BoundStudy(BudgetStudy):
    class Config(Makes2["_BoundStudy"], BudgetStudy.Config):
        bindings: list[tuple[int, int, int | None]] = field(default_factory=list)
        """Inputs observed by each finalization of this study copy."""

        @override
        def finalize(self) -> Self:
            self.bindings.append((self.data.seq_len, self.data.train.count, self.seed))
            return super().finalize()


class _BoundData(ForecastData):
    class Config(Makes["_BoundData"], ForecastData.Config):
        binding: tuple[int, int] = (-1, -1)
        """Training count and stream seed at finalization."""

        finalizations: int = 0
        """Number of finalizations of this data copy."""

        @override
        def finalize(self) -> Self:
            self.binding = self.train.count, self.train.seed
            self.finalizations += 1
            return super().finalize()


class _BoundReference(CorrectedOscillator):
    class Config(Makes2["_BoundReference"], CorrectedOscillator.Config):
        bound_cubic: float = -1
        """Cubic coefficient observed during finalization."""

        bound_substeps: int = -1
        """Integration refinement observed during finalization."""

        finalizations: int = 0
        """Number of finalizations of this equation copy."""

        @override
        def finalize(self) -> Self:
            self.bound_cubic = self.cubic
            self.bound_substeps = self.substeps
            self.finalizations += 1
            return super().finalize()

    def __init__(self, config: Config) -> None:
        if (config.bound_cubic, config.bound_substeps) != (
            config.cubic,
            config.substeps,
        ):
            raise ValueError("Equation inputs changed after finalization.")
        super().__init__(config)


class _ParentBoundData(ForecastData):
    class Config(Makes["_ParentBoundData"], ForecastData.Config):
        """Derive equation, geometry, and streams in the parent finalizer."""

        @override
        def finalize(self) -> Self:
            self.reference.damping = self.cheap.damping
            self.pred_len = 4
            for offset, stream in enumerate(
                (self.train, self.validation, self.test, self.shifted),
                start=1,
            ):
                stream.seed = 100 + offset
                stream.count = 3 + offset
            return super().finalize()


class _DevelopmentSuite(BenchmarkSuite):
    class Config(Makes["_DevelopmentSuite"], BenchmarkSuite.Config):
        @override
        def finalize(self) -> Self:
            self.phase = "development"
            return super().finalize()


class _CubicFamily(SystemFamily, LateBound):
    class Config(Makes["_CubicFamily"], SystemFamily.Config):
        @override
        def finalize(self) -> Self:
            self.reference.cubic = 0.4
            return super().finalize()

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.bound_root: object | None = None

    @override
    def bind(self, root: object) -> None:
        self.bound_root = root


class _DerivedRecipe(BenchmarkRecipe):
    class Config(Makes["_DerivedRecipe"], BenchmarkRecipe.Config):
        @override
        def finalize(self) -> Self:
            self.seeds.append(17)
            return super().finalize()


class _SuiteOwner:
    class Config(Fig["_SuiteOwner"]):
        suite: BenchmarkSuite.Config = field(default_factory=BenchmarkSuite.Config)
        """Template owner finalized by an enclosing config."""

    def __init__(self, config: Config) -> None:
        self.suite = config.suite.make()


class _NonfiniteRefinement(CorrectedOscillator):
    class Config(Makes2["_NonfiniteRefinement"], CorrectedOscillator.Config):
        """Expose a nonfinite refined score after a valid ordinary rollout."""

    @override
    def forecast_history(
        self,
        media: Tensor,
        *,
        steps: int,
        substeps: int | None = None,
    ) -> Tensor:
        values = super().forecast_history(media, steps=steps, substeps=substeps)
        return values if substeps is None else torch.full_like(values, float("nan"))


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
