"""Check budgets, mechanism holdouts, and causal scoring in the release suite."""

from __future__ import annotations

from typing import TYPE_CHECKING

import json

from configgle import Fig
from torch import Tensor

import pytest
import torch

from priml.baselines.oscillator.experiments import exp040
from priml.baselines.oscillator.forces import WeightedForce
from priml.baselines.oscillator.mechanisms import SineForce, SystemFamily
from priml.baselines.oscillator.suite import (
    BenchmarkRecipe,
    BenchmarkSuite,
    BudgetStudy,
    ReferenceBudget,
    suite_digest,
)


if TYPE_CHECKING:
    from pathlib import Path

    from priml.baselines.oscillator.data import Oscillator, Windows
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


def test_nonfinite_model_runs_are_retained_as_failures(tmp_path: Path) -> None:
    cfg = tiny_suite()
    cfg.audit = cfg.audit[:1]
    cfg.audit[0].instances = 1
    cfg.budgets = cfg.budgets[:1]
    cfg.recipes = cfg.recipes[:1]
    cfg.recipes[0].seeds = [11]
    cfg.recipes[0].study.method = RepeatForecast.Config(factor=float("nan"))
    records, _ = cfg.make().run(tmp_path / "failed", phase="audit")
    assert len(records) == 1
    assert records[0]["status"] == "failed"
    assert records[0]["result"] is None
    assert records[0]["error"] is not None


def test_resume_retains_attempts_and_rejects_changed_provenance(tmp_path: Path) -> None:
    cfg = tiny_suite()
    cfg.audit = cfg.audit[:1]
    cfg.audit[0].instances = 1
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
    (directory / "foreign--run.json").write_text(json.dumps(records[0]))
    with pytest.raises(ValueError, match="unplanned"):
        suite.run(directory, phase="audit", resume=True, provenance={"source": "first"})


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
