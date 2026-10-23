"""Tests for the openai/math experiment configurations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import pytest
import torch

from priml.baselines.openaimath import experiments
from priml.baselines.openaimath.experiments import OpenMathTrainLoop, exp000, exp_smoke
from priml.baselines.openaimath.live import LiveOpenMathRolloutData
from priml.baselines.tmax.rollouts import read_rollout_records
from priml.baselines.tmax.train_step import _row_metrics
from priml.math.seed import set_seed_local
from priml.runtime import MultiProcess, SingleProcess
from priml.testing.golden import assert_pprint_golden


if TYPE_CHECKING:
    from priml.testing.experiments import ExperimentFactory
    from priml.train.custom_types import TrainStepProtocol


@dataclass
class _PadOnlyStep:
    """A learner that exposes only its resolved pad id."""

    pad_token_id: int


FIXTURE = Path(__file__).resolve().parent / "testdata" / "smoke_rollout.jsonl"

ALL_EXPERIMENTS: Final[list[ExperimentFactory[OpenMathTrainLoop]]] = [
    exp000,
    exp_smoke,
]


_FIXTURE_ROOT = str(Path(__file__).resolve().parent / "testdata")


def _normalize_smoke_golden(rendered: str) -> str:
    """Keep the smoke config golden independent of the checkout path."""
    return rendered.replace(_FIXTURE_ROOT, "<fixture>")


@pytest.mark.parametrize(
    "factory",
    ALL_EXPERIMENTS,
    ids=[f.__name__ for f in ALL_EXPERIMENTS],
)
def test_every_experiment_finalizes(
    factory: ExperimentFactory[OpenMathTrainLoop],
) -> None:
    """Every experiment config copies and finalizes."""
    assert factory().copy_tree().finalize() is not None


@pytest.mark.parametrize(
    "factory",
    ALL_EXPERIMENTS,
    ids=[f.__name__ for f in ALL_EXPERIMENTS],
)
def test_experiment_name_matches_the_factory(
    factory: ExperimentFactory[OpenMathTrainLoop],
) -> None:
    """The configured experiment name matches the factory name."""
    assert factory().experiment_name == factory.__name__


@pytest.mark.parametrize(
    "factory",
    ALL_EXPERIMENTS,
    ids=[f.__name__ for f in ALL_EXPERIMENTS],
)
def test_construction_reads_no_files(
    factory: ExperimentFactory[OpenMathTrainLoop],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Configs build without artifacts or a GPU."""

    def boom(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("experiment construction must not load tensors")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(torch, "load", boom)
    _ = factory().copy_tree().finalize()


def test_smoke_is_marked_as_carrying_no_result() -> None:
    """The smoke docstring marks it as not a result."""
    assert "Not a result" in (exp_smoke.__doc__ or "")


def test_module_docstring_lists_every_experiment() -> None:
    """The module docstring lists every experiment."""
    documented = experiments.__doc__ or ""
    for factory in ALL_EXPERIMENTS:
        assert factory.__name__ in documented


@pytest.mark.compute_large_fixture
def test_exp000_matches_its_golden_config() -> None:
    """Pin the complete finalized exp000 config.

    Rendering the full 4B model config takes about two seconds, so this runs in
    the slow tier, as nanochat's does.
    """
    assert_pprint_golden(test_file=__file__, name="exp000", config=exp000())


def test_exp_smoke_matches_its_golden_config() -> None:
    """Pin the complete finalized exp_smoke config."""
    assert_pprint_golden(
        test_file=__file__,
        name="exp_smoke",
        config=exp_smoke(),
        normalize=_normalize_smoke_golden,
    )


def test_exp000_trains_from_live_openai_math_rollouts() -> None:
    """The TMax 4B recipe's settings survive where the substrate allows."""
    config = exp000()
    assert config.study_name == "openaimath"
    assert config.experiment_name == "exp000"
    # Unseeded, as a real run is; the task order's drawn seed is checkpointed.
    assert config.seed is None
    assert config.step.model_path == "/models/Qwen3.5-4B"
    assert config.step.pad_token_id == 248_044
    assert config.step.train_budget_steps == 500
    assert config.max_epochs == 1
    assert isinstance(config.dataset, LiveOpenMathRolloutData.Config)
    assert config.dataset.runner.tasks_dir == Path(
        "/datasets/openaimath/openai-math/lemmas/train",
    )
    assert config.dataset.runner.filter_zero_std_samples is True
    assert config.dataset.runner.active_sampling is True
    assert config.dataset.runner.max_draws_per_update == 64
    assert config.dataset.runner.num_prompts == 8
    assert config.dataset.runner.samples_per_prompt == 32
    assert config.dataset.runner.records_per_update == 256
    assert isinstance(config.runtime, MultiProcess.Config)
    assert config.num_steps_eval == 0


def test_exp000_finalize_shares_settings_with_the_runner(tmp_path: Path) -> None:
    """One configuration drives both the learner and the rollout runner."""
    config = exp000()
    config.base_dir = tmp_path
    config = config.finalize()
    dataset = config.dataset
    assert isinstance(dataset, LiveOpenMathRolloutData.Config)
    runner = dataset.runner
    assert runner.num_updates == config.step.train_budget_steps
    assert runner.seed == config.seed
    assert runner.temperature == config.step.temperature
    assert runner.checkpoint == tmp_path / "models/Qwen3.5-4B"
    assert runner.output_dir == dataset.working_dir
    assert dataset.records_per_update == runner.records_per_update
    assert dataset.num_samples_per_prompt == runner.samples_per_prompt
    assert dataset.export_dir == tmp_path / "runs/openaimath/exp000/export"


def test_exp000_finalize_keeps_an_explicit_runner_setting(tmp_path: Path) -> None:
    """The loop fills a runner setting only where the experiment left it unset."""
    config = exp000()
    config.base_dir = tmp_path
    assert isinstance(config.dataset, LiveOpenMathRolloutData.Config)
    config.dataset.runner.seed = 7
    config.dataset.runner.checkpoint = "/models/other"
    config = config.finalize()
    assert isinstance(config.dataset, LiveOpenMathRolloutData.Config)
    assert config.dataset.runner.seed == 7
    assert config.dataset.runner.checkpoint == tmp_path / "models/other"


def _stage_exp000_task(base_dir: Path) -> None:
    """Stage one task where exp000's runner looks for the dataset."""
    task = base_dir / "datasets/openaimath/openai-math/lemmas/train/lemma-alpha"
    task.mkdir(parents=True)
    (task / "instruction.md").write_text("prove alpha\n", encoding="utf-8")


def test_exp000_live_data_builds_its_runner(tmp_path: Path) -> None:
    """The finalized recipe leaves no per-run runner setting unset."""
    _stage_exp000_task(tmp_path)
    config = exp000()
    config.base_dir = tmp_path
    data = config.finalize().dataset.make()
    assert isinstance(data, LiveOpenMathRolloutData)


def test_exp000_live_data_refuses_a_pad_token_mismatch(tmp_path: Path) -> None:
    """Packing and scoring must remove the same token."""
    _stage_exp000_task(tmp_path)
    config = exp000()
    config.base_dir = tmp_path
    config.dataset.pad_token_id = 1
    data = config.finalize().dataset.make()
    with pytest.raises(ValueError, match="pad_token_id"):
        data.bind_step(cast("TrainStepProtocol", _PadOnlyStep(pad_token_id=248_044)))


def test_exp_smoke_runs_one_cpu_update_from_the_fixture() -> None:
    """The smoke run is small, offline, and needs no checkpoint."""
    config = exp_smoke()
    assert config.experiment_name == "exp_smoke"
    assert isinstance(config.runtime, SingleProcess.Config)
    assert config.step.model_path is None
    assert config.step.train_budget_steps == 1
    assert config.max_steps == 1
    assert config.checkpointer is None
    assert config.step.export_dir is None
    assert config.dataset.records_per_update == 2
    assert config.dataset.num_samples_per_prompt == 2
    assert config.dataset.mask_tool_use is False
    assert config.dataset.working_dir == FIXTURE.name
    assert not isinstance(config.dataset, LiveOpenMathRolloutData.Config)


def test_exp_smoke_fixture_is_one_prompt_group_with_both_outcomes() -> None:
    """The fixture holds one accepted and one rejected episode."""
    records = read_rollout_records(FIXTURE)
    assert len(records) == 2
    assert {record.step for record in records} == {0}
    assert {record.prompt_idx for record in records} == {0}
    assert sorted(record.reward for record in records) == [0.0, 1.0]
    for record in records:
        assert record.finish_reason == "stop"
        assert record.tool_mask is not None
        assert len(record.logprobs) == len(record.response_tokens)


@pytest.mark.compute_training
def test_exp_smoke_update_moves_weights() -> None:
    """The substrate's fixture trains a real DPPO update end to end.

    The other smoke tests read the fixture's shape. This one runs the loop's own
    train step over it, so the transcript-to-record conversion, the packing, the
    scoring, and the objective are checked together rather than in pieces: the
    ratio starts at 1 because the fixture's logprobs are the smoke model's own,
    the two rewards differ so the centered advantage is nonzero, and the weights
    actually move. That is what makes the fixture evidence that the substrate
    trains, rather than evidence that a file parses.
    """
    config = exp_smoke()
    set_seed_local(config.seed)
    step = config.step.make()
    data = config.dataset.make()
    batch = next(iter(data.train_dataloader()))
    preprocessed = step.preprocess_batch(batch)

    output = step.train_loss(**preprocessed)
    metrics = _row_metrics(output)
    # Two episodes of four response tokens each.
    assert float(metrics["response_tokens"]) == 8.0
    assert float(metrics["ratio_mean"]) == pytest.approx(1.0, abs=1e-5)
    assert float(metrics["divergence_mean"]) == 0.0
    output["loss"].backward()
    magnitude = sum(
        float(parameter.grad.abs().sum())
        for parameter in step.model.parameters()
        if parameter.grad is not None
    )
    # A collapsed ratio produces a much smaller gradient.
    assert magnitude > 1.0

    # Recreate the model that scored the fixture, then step it once.
    set_seed_local(config.seed)
    update = config.step.make()
    before = next(update.model.parameters()).detach().clone()
    result = update.train_step(**update.preprocess_batch(batch))
    assert update.global_step == 1
    assert torch.isfinite(result["loss"])
    assert not torch.equal(before, next(update.model.parameters()).detach())


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
