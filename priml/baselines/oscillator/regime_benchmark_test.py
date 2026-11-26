"""Check declared attempts, paired scoring, and retained audit evidence."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict, cast
from unittest import mock

import json
import math

from configgle import Makeable, PartialConfig
from treekle import json_loads

import pytest
import torch

from priml.baselines.oscillator.adaptation import (
    LastObservation,
    ObservedNeuralForecast,
)
from priml.baselines.oscillator.experiments import (
    exp041,
    exp042,
    exp043,
    exp044,
    exp045,
    exp046,
    exp047,
    exp048,
)
from priml.baselines.oscillator.regime_benchmark import (
    EpisodeScores,
    ObservedRecipe,
    RegimeBenchmark,
    RegimeCampaign,
    RegimeFamily,
    RegimeRun,
)
from priml.baselines.oscillator.regimes import (
    DynamicsDomain,
    EvaluatorEpisodes,
    Observation,
    ObservedChange,
    ObservedWindows,
    PositionObservation,
    RegimeData,
    RegimeDynamics,
)
from priml.testing.golden import assert_pprint_golden


if TYPE_CHECKING:
    from pathlib import Path


class _Evidence(TypedDict):
    runs: list[RegimeRun]
    summary: list[dict[str, object]]
    comparisons: list[dict[str, object]]


def _small_study() -> RegimeBenchmark.Config:
    cfg = RegimeBenchmark.Config()
    cfg.data.seq_len, cfg.data.pred_len = 11, 5
    cfg.train_episodes, cfg.validation_episodes, cfg.scoring_episodes = 7, 3, 7
    cfg.development[0].episodes = 3
    cfg.resamples = 11
    cfg.recipes[0].method = LastObservation.Config()
    cfg.recipes[0].seeds = [13, 17]
    return cfg


def _records(cfg: RegimeBenchmark.Config) -> list[RegimeRun]:
    return [
        RegimeRun(
            run_id=f"{r.name}--{seed}",
            method=r.name,
            seed=seed,
            status="complete",
            error=None,
            fit={},
            elapsed_seconds=0.1,
            families={
                f.name: EpisodeScores(
                    scaled_mse=[float(i + 1) for i in range(f.episodes)],
                    horizon_scaled_mse=[2.0] * cfg.data.pred_len,
                )
                for f in cfg.development
            },
            diagnostics={f.name: 0.0 for f in cfg.development},
        )
        for r in cfg.recipes
        for seed in r.seeds
    ]


def test_config_goldens_pin_defaults_and_hidden_velocity_propagation() -> None:
    assert_pprint_golden(
        test_file=__file__,
        name="regime_defaults",
        config=RegimeBenchmark.Config(),
    )
    assert_pprint_golden(
        test_file=__file__,
        name="regime_position_change",
        config=exp044(),
    )


def test_factories_isolate_observation_schedule_phase_and_data_budget() -> None:
    base, position, change, combined = exp041(), exp042(), exp043(), exp044()
    assert isinstance(position.data, RegimeData.Config)
    assert isinstance(change.data, RegimeData.Config)
    assert isinstance(position.data.observation, PositionObservation.Config)
    assert isinstance(change.data.schedule, ObservedChange.Config)
    for declaration, parent, attribute in (
        (position, base, "observation"),
        (change, base, "schedule"),
        (combined, change, "observation"),
    ):
        child = declaration.copy_tree()
        setattr(child.data, attribute, getattr(parent.data, attribute))
        child.experiment_name, child.working_dir = (
            parent.experiment_name,
            parent.working_dir,
        )
        assert child == parent
    development, audit, large = exp045(), exp046(), exp047()
    assert [s.experiment_name for s in development.studies] == [
        f"exp{i:03d}" for i in range(41, 45)
    ]
    audit.working_dir = development.working_dir
    for before, after in zip(development.studies, audit.studies, strict=True):
        assert after.phase == "audit"
        after.phase = before.phase
    assert audit == development
    audit = exp046()
    large.working_dir = audit.working_dir
    for small, big in zip(audit.studies, large.studies, strict=True):
        assert big.train_episodes == small.train_episodes * 4
        assert big.validation_episodes == small.validation_episodes * 4
        assert big.scoring_episodes == small.scoring_episodes
        big.train_episodes, big.validation_episodes = (
            small.train_episodes,
            small.validation_episodes,
        )
    assert large == audit
    longer = exp048()
    longer.working_dir = audit.working_dir
    for short, long in zip(audit.studies, longer.studies, strict=True):
        for a, b in zip(short.recipes, long.recipes, strict=True):
            if isinstance(a.method, ObservedNeuralForecast.Config):
                assert isinstance(b.method, ObservedNeuralForecast.Config)
                assert (
                    b.method.step.train_budget_steps
                    == 4 * a.method.step.train_budget_steps
                )
                b.method.step.train_budget_steps = a.method.step.train_budget_steps
    assert longer == audit


def test_run_preserves_rng_and_threads_and_resume_keeps_evidence(
    tmp_path: Path,
) -> None:
    cfg = _small_study()
    state = torch.get_rng_state().clone()
    threads = torch.get_num_threads()
    study = cfg.make()
    first = study.run(tmp_path / "run")
    assert torch.equal(state, torch.get_rng_state())
    assert torch.get_num_threads() == threads
    with mock.patch.object(LastObservation, "fit", side_effect=AssertionError("refit")):
        second = study.run(tmp_path / "run", resume=True)
    assert first == second
    records = cast(
        "_Evidence",
        json.loads((tmp_path / "run" / "results.json").read_text()),
    )["runs"]
    assert len(records) == 2
    assert all(r["diagnostics"] == {"in_domain": 0.0} for r in records)
    assert (tmp_path / "run" / "protocol.txt").is_file()
    with pytest.raises(FileExistsError):
        study.run(tmp_path / "run")


def test_changed_manifest_and_unplanned_attempts_cannot_resume(tmp_path: Path) -> None:
    cfg = _small_study()
    cfg.make().run(tmp_path / "run")
    changed = cfg.copy_tree()
    changed.train_seed += 1
    with pytest.raises(ValueError, match="Cannot resume changed"):
        changed.make().run(tmp_path / "run", resume=True)
    manifest_path = tmp_path / "run" / "manifest.json"
    original = manifest_path.read_text()
    manifest = cast("dict[str, object]", json.loads(original))
    manifest["python"] = "other"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="environment"):
        cfg.make().run(tmp_path / "run", resume=True)
    manifest_path.write_text(original)
    (tmp_path / "run" / "unknown--13.json").write_text("{}")
    with pytest.raises(ValueError, match="unplanned"):
        cfg.make().run(tmp_path / "run", resume=True)


def test_wrong_retained_identity_cannot_resume(tmp_path: Path) -> None:
    cfg = _small_study()
    cfg.make().run(tmp_path / "run")
    path = tmp_path / "run" / "dlinear--13.json"
    record = json_loads(path.read_text(), RegimeRun)
    record["run_id"] = "other--13"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="identity"):
        cfg.make().run(tmp_path / "run", resume=True)


def test_failed_attempt_is_retained_and_never_gets_a_finite_score(
    tmp_path: Path,
) -> None:
    cfg = _small_study()
    cfg.comparisons = [("dlinear", "control")]
    cfg.recipes.append(
        ObservedRecipe.Config(
            name="control",
            method=LastObservation.Config(),
            seeds=[19],
        ),
    )
    threads = torch.get_num_threads()
    with (
        mock.patch.object(
            LastObservation,
            "fit",
            side_effect=RuntimeError("fit failed"),
        ),
        pytest.raises(RuntimeError, match="attempts are retained"),
    ):
        cfg.make().run(tmp_path / "run")
    report = cast(
        "_Evidence",
        json.loads((tmp_path / "run" / "results.json").read_text()),
    )
    assert len(report["runs"]) == 3
    assert all(r["error"] == "RuntimeError: fit failed" for r in report["runs"])
    assert all(r["scaled_mse"] is None for r in report["summary"])
    assert report["comparisons"][0]["status"] == "failed"
    assert torch.get_num_threads() == threads


@pytest.mark.parametrize("invalid", ["shape", "nan", "overflow"])
def test_invalid_predictions_invalidate_the_attempt(
    tmp_path: Path,
    invalid: str,
) -> None:
    cfg = _small_study()
    output = torch.zeros(3, 5, 2, dtype=torch.float64)
    if invalid == "shape":
        output = output[:, :2]
    else:
        output.fill_(1e300 if invalid == "overflow" else math.nan)
    with (
        mock.patch.object(LastObservation, "predict", return_value=output),
        pytest.raises(RuntimeError, match="failed"),
    ):
        cfg.make().run(tmp_path / "run")
    record = json_loads((tmp_path / "run" / "dlinear--13.json").read_text(), RegimeRun)
    assert record["error"] is not None
    assert (
        "nonfinite scored errors"
        if invalid == "overflow"
        else "invalid observed forecast"
    ) in record["error"]


def test_nonfinite_fit_diagnostics_invalidate_the_attempt(tmp_path: Path) -> None:
    cfg = _small_study()
    with (
        mock.patch.object(LastObservation, "fit", return_value={"loss": math.nan}),
        pytest.raises(RuntimeError, match="failed"),
    ):
        cfg.make().run(tmp_path / "run")
    record = json_loads((tmp_path / "run" / "dlinear--13.json").read_text(), RegimeRun)
    assert record["error"] == "ValueError: Fit diagnostics must be finite."


def test_position_scores_do_not_reward_an_extra_velocity_channel() -> None:
    cfg = _small_study()
    study = cfg.make()
    data = cfg.data.make()
    windows = data.sample(count=3, seed=73).windows
    method = LastObservation.Config().make()
    method.fit(windows, validation=windows, dynamics=data.dynamics)
    first, _ = study.evaluate(method, data=data, scale=torch.tensor(2.0))
    original = method.predict

    def corrupt_velocity(media: torch.Tensor) -> torch.Tensor:
        output = original(media)
        output[..., 1] = 1e10
        return output

    with mock.patch.object(method, "predict", side_effect=corrupt_velocity):
        second, _ = study.evaluate(method, data=data, scale=torch.tensor(2.0))
    assert first == second
    scores = first["in_domain"]
    assert sum(scores["scaled_mse"]) / 3 == pytest.approx(
        sum(scores["horizon_scaled_mse"]) / 5,
    )


def test_reference_continuation_must_match_targets() -> None:
    cfg = _small_study()
    data = cfg.data.make()
    windows = data.sample(count=3, seed=79).windows
    method = LastObservation.Config().make()
    method.fit(windows, validation=windows, dynamics=data.dynamics)
    with (
        mock.patch.object(data.dynamics, "forecast", return_value=torch.zeros(3, 5, 2)),
        pytest.raises(ValueError, match="Reference continuation"),
    ):
        cfg.make().evaluate(method, data=data, scale=torch.tensor(1.0))


def test_audit_uses_reserved_families_and_constructs_them_after_fit(
    tmp_path: Path,
) -> None:
    cfg = _small_study()
    cfg.phase = "audit"
    cfg.audit = [RegimeFamily.Config(name="fresh", seed=83, episodes=3)]
    cfg.recipes[0].seeds = [13]
    order: list[int | str] = []
    sample, fit = RegimeData.sample, LastObservation.fit

    def record_sample(
        self: RegimeData,
        *,
        count: int,
        seed: int,
        domain: DynamicsDomain.Config | None = None,
    ) -> EvaluatorEpisodes:
        order.append(seed)
        return sample(self, count=count, seed=seed, domain=domain)

    def record_fit(
        self: LastObservation,
        train: ObservedWindows,
        *,
        validation: ObservedWindows,
        dynamics: RegimeDynamics,
    ) -> dict[str, int | float]:
        order.append("fit")
        return fit(self, train, validation=validation, dynamics=dynamics)

    with (
        mock.patch.object(RegimeData, "sample", record_sample),
        mock.patch.object(LastObservation, "fit", record_fit),
    ):
        report = cfg.make().run(tmp_path / "run")
    assert order == [cfg.train_seed, cfg.validation_seed, "fit", 83]
    assert report["phase"] == "audit"


def test_seed_average_and_paired_bootstrap_keep_episode_unit() -> None:
    cfg = _small_study()
    cfg.recipes.append(
        ObservedRecipe.Config(
            name="control",
            method=LastObservation.Config(),
            seeds=[19],
        ),
    )
    cfg.comparisons = [("dlinear", "control")]
    records = _records(cfg)
    records[0]["families"]["in_domain"]["scaled_mse"] = [0.0, 1.0, 2.0]
    records[1]["families"]["in_domain"]["scaled_mse"] = [2.0, 3.0, 4.0]
    records[2]["families"]["in_domain"]["scaled_mse"] = [1.5, 2.5, 3.5]
    study = cfg.make()
    summary = study.summarize(records)
    assert summary[0]["scaled_mse"] == 2.0
    assert summary[0]["independent_episodes"] == 3
    assert summary[0]["model_seeds"] == 2
    comparison = study.compare(records)[0]
    assert comparison["scaled_mse_difference"] == -0.5
    assert comparison["interval"] == [-0.5, -0.5]


@pytest.mark.parametrize(
    "tamper",
    [
        "missing",
        "duplicate",
        "identity",
        "status",
        "elapsed",
        "families",
        "horizon",
        "episodes",
        "negative",
        "nonfinite",
        "error",
    ],
)
def test_summary_rejects_incomplete_or_tampered_evidence(tamper: str) -> None:
    cfg = _small_study()
    records = _records(cfg)
    if tamper == "missing":
        records.pop()
    elif tamper == "duplicate":
        records.append(records[0])
    elif tamper == "identity":
        records[0]["seed"] = 19
    elif tamper == "status":
        records[0]["status"] = "unknown"
    elif tamper == "elapsed":
        records[0]["elapsed_seconds"] = math.inf
    elif tamper == "families":
        records[0]["families"]["extra"] = records[0]["families"]["in_domain"]
    elif tamper == "horizon":
        records[0]["families"]["in_domain"]["horizon_scaled_mse"] = [1.0]
    elif tamper == "episodes":
        records[0]["families"]["in_domain"]["scaled_mse"] = [1.0]
    elif tamper == "negative":
        records[0]["families"]["in_domain"]["scaled_mse"][0] = -1.0
    elif tamper == "nonfinite":
        records[0]["families"]["in_domain"]["scaled_mse"][0] = math.nan
    else:
        records[0]["error"] = "failed"
    with pytest.raises(ValueError, match=r"attempt|Retained|retained|complete"):
        cfg.make().summarize(records)


def test_campaign_runs_and_resumes_each_condition(tmp_path: Path) -> None:
    cfg = RegimeCampaign.Config(studies=[_small_study()])
    first = cfg.make().run(tmp_path / "campaign")
    assert cfg.make().run(tmp_path / "campaign", resume=True) == first
    with pytest.raises(ValueError, match="distinct"):
        RegimeCampaign.Config().make()
    cfg.studies *= 2
    with pytest.raises(ValueError, match="distinct"):
        cfg.make()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("train_episodes", 0),
        ("resamples", 0),
        ("scoring_episodes", 8),
        ("phase", "other"),
        ("phase", "audit"),
        ("validation_seed", 1101),
    ],
)
def test_invalid_budgets_phases_and_streams_fail(field: str, value: object) -> None:
    cfg = _small_study()
    setattr(cfg, field, value)
    with pytest.raises(ValueError, match=r"budgets|phase|audit|streams"):
        cfg.make()


def test_invalid_recipe_family_and_comparison_declarations_fail() -> None:
    cfg = _small_study()
    cfg.recipes *= 2
    with pytest.raises(ValueError, match="distinct"):
        cfg.make()
    cfg = _small_study()
    cfg.comparisons = [("missing", "dlinear")]
    with pytest.raises(ValueError, match="comparisons"):
        cfg.make()
    with pytest.raises(ValueError, match="Recipes"):
        ObservedRecipe.Config(seeds=[13, 13]).make()
    with pytest.raises(ValueError, match="families"):
        RegimeFamily.Config(episodes=0).make()
    cfg = _small_study()
    assert isinstance(cfg.data, RegimeData.Config)
    cfg.data.observation = cast("Makeable[Observation]", PartialConfig(torch.clone))
    with pytest.raises(TypeError, match="declare its channels"):
        cfg.make()


def test_neural_initialization_and_evaluation_leave_global_rng_alone(
    tmp_path: Path,
) -> None:
    cfg = _small_study()
    method = ObservedNeuralForecast.Config()
    method.step.train_budget_steps = 2
    method.batch_size = 4
    cfg.recipes[0].method = method
    before = torch.get_rng_state().clone()
    cfg.make().run(tmp_path / "run")
    assert torch.equal(before, torch.get_rng_state())


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
