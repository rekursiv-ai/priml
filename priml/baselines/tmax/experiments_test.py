"""Tests for the TMax experiment configs."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import math

from configgle.testing import assert_pprint_golden

import pytest
import torch

from priml.baselines.tmax import experiments
from priml.baselines.tmax.experiments import TMaxTrainLoop, exp000, exp_smoke
from priml.baselines.tmax.live import LiveTMaxRolloutData
from priml.baselines.tmax.train_step import _row_metrics
from priml.math.seed import set_seed_local
from priml.runtime import MultiProcess, SingleProcess
from priml.train.activation import SelectiveActivationCheckpointing
from priml.train.checkpointer import Checkpointer
from priml.train.parallelism import FullySharded, NoParallel


if TYPE_CHECKING:
    from priml.testing.experiments import ExperimentFactory


ALL_EXPERIMENTS: Final[list[ExperimentFactory[TMaxTrainLoop]]] = [
    exp000,
    exp_smoke,
]


@pytest.mark.parametrize(
    "factory",
    ALL_EXPERIMENTS,
    ids=[f.__name__ for f in ALL_EXPERIMENTS],
)
def test_every_experiment_finalizes(
    factory: ExperimentFactory[TMaxTrainLoop],
) -> None:
    """Every experiment config copies and finalizes."""
    assert factory().copy_tree().finalize() is not None


@pytest.mark.parametrize(
    "factory",
    ALL_EXPERIMENTS,
    ids=[f.__name__ for f in ALL_EXPERIMENTS],
)
def test_experiment_name_matches_the_factory(
    factory: ExperimentFactory[TMaxTrainLoop],
) -> None:
    """The configured experiment name matches the factory name."""
    assert factory().experiment_name == factory.__name__


@pytest.mark.parametrize(
    "factory",
    ALL_EXPERIMENTS,
    ids=[f.__name__ for f in ALL_EXPERIMENTS],
)
def test_construction_reads_no_files(
    factory: ExperimentFactory[TMaxTrainLoop],
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


def test_exp000_matches_its_golden_config() -> None:
    """Pin the complete finalized exp000 config."""
    assert_pprint_golden(test_file=__file__, name="exp000", config=exp000())


def test_exp000_names_the_artifact_backed_cluster_recipe() -> None:
    """Check the published cluster recipe."""
    config = exp000().finalize()
    assert config.study_name == "tmax"
    assert config.experiment_name == "exp000"
    assert config.step.model_path == Path("/opt/scratch/models/Qwen3.5-4B")
    assert config.step.model.channels_in == 2_560
    assert config.step.model.channels_out == 248_320
    assert isinstance(config.step.model.block, list)
    assert len(config.step.model.block) == 32
    assert config.step.pad_token_id == 248_044
    assert isinstance(config.step.parallelism, FullySharded.Config)
    assert config.step.parallelism.mp_param_dtype == torch.bfloat16
    assert isinstance(config.runtime, MultiProcess.Config)
    assert config.runtime.process_group_timeout_sec == 72_000
    assert isinstance(config.dataset, LiveTMaxRolloutData.Config)
    assert config.dataset.working_dir == Path(
        "/opt/scratch/runs/tmax/exp000/rollouts",
    )
    assert config.dataset.runner.output_dir == config.dataset.working_dir
    assert config.step.export_dir == Path("/opt/scratch/runs/tmax/exp000/export")
    assert config.dataset.records_per_update == 256
    assert config.dataset.num_samples_per_prompt == 32
    assert config.dataset.mask_tool_use is True
    # Active sampling replaces groups removed by the reward-spread filter.
    assert config.dataset.filter_zero_std_samples is True
    assert config.dataset.active_sampling is True
    assert config.step.divergence_type == "tv"
    assert config.step.divergence_threshold == 0.1
    assert config.max_steps == math.inf
    assert config.max_epochs == 1
    assert config.step.train_budget_steps == 500
    assert config.dataset.runner.num_updates == 500
    assert config.step.gradient_clip_norm == 1.0
    # The cluster recipe limits memory with recomputation and chunked scoring.
    assert config.step.head_chunk_size == 2_048
    assert config.step.fp32_head is True
    assert isinstance(
        config.step.activation_memoization,
        SelectiveActivationCheckpointing.Config,
    )
    # Match the released seed.
    assert config.seed == 42
    assert config.checkpointer is not None
    checkpointer = cast(Checkpointer.Config, config.checkpointer)
    assert checkpointer.save_every == 10
    assert checkpointer.keep_last_n == 3
    assert checkpointer.keep_every == 50


def test_exp000_moves_every_path_with_base_dir(tmp_path: Path) -> None:
    """base_dir moves every model, rollout, export, and asset path."""
    config = exp000()
    config.base_dir = tmp_path
    config = config.finalize()

    run_dir = tmp_path / "runs/tmax/exp000"
    assert isinstance(config.dataset, LiveTMaxRolloutData.Config)
    assert config.working_dir == run_dir
    assert config.step.model_path == tmp_path / "models/Qwen3.5-4B"
    assert config.step.export_dir == run_dir / "export"
    assert config.dataset.working_dir == run_dir / "rollouts"
    assert config.dataset.runner.output_dir == run_dir / "rollouts"
    assert config.dataset.runner.dataset_path == (
        tmp_path / "datasets/tmax/tmax-15k-open-instruct/data/"
        "train-00000-of-00001.parquet"
    )
    assert config.dataset.runner.upstream_root == (
        tmp_path / "datasets/tmax/upstream/tmax"
    )
    assert config.dataset.runner.task_data_dir == (
        tmp_path / "datasets/tmax/tmax-15k-open-instruct/task-data.tar.gz.extracted"
    )


def test_live_data_cannot_escape_the_run_directory(tmp_path: Path) -> None:
    """Live rollouts stay inside the run directory."""
    config = exp000()
    assert isinstance(config.dataset, LiveTMaxRolloutData.Config)
    config.dataset.base_dir = tmp_path / "other"
    with pytest.raises(ValueError, match="must inherit"):
        config.finalize()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("active_sampling", False),
        ("filter_zero_std_samples", False),
    ],
)
def test_live_data_filter_settings_must_match_the_runner(
    field: str,
    value: bool,
) -> None:
    """The learner and rollout runner use the same filters."""
    config = exp000()
    assert isinstance(config.dataset, LiveTMaxRolloutData.Config)
    setattr(config.dataset, field, value)
    with pytest.raises(ValueError, match=field):
        config.finalize()


def test_smoke_is_a_real_learner_over_the_published_fixture() -> None:
    """The smoke recipe is offline and CPU-friendly."""
    config = exp_smoke().finalize()
    assert config.experiment_name == "exp_smoke"
    assert config.step.model_path is None
    assert isinstance(config.runtime, SingleProcess.Config)
    assert isinstance(config.step.parallelism, NoParallel.Config)
    assert Path(config.dataset.working_dir).name == "smoke_rollout.jsonl"
    # Two rewards are needed for a nonzero centered advantage.
    assert config.dataset.records_per_update == 2
    assert config.dataset.num_samples_per_prompt == 2
    assert config.dataset.mask_tool_use is False
    assert config.max_steps == 1
    assert config.step.train_budget_steps == 1
    assert config.checkpointer is None
    assert config.step.export_dir is None
    # The smoke recipe omits the cluster memory settings.
    assert config.step.head_chunk_size is None
    assert config.step.fp32_head is False
    assert not isinstance(
        config.step.activation_memoization,
        SelectiveActivationCheckpointing.Config,
    )
    # The fixture logprobs were computed with this seeded model.
    assert config.seed == 0


@pytest.mark.compute_training
def test_smoke_update_moves_weights() -> None:
    """The smoke update has ratio 1, a gradient, and a weight change."""
    config = exp_smoke()
    set_seed_local(config.seed)
    step = config.step.make()
    data = config.dataset.make()
    batch = next(iter(data.train_dataloader()))
    pre = step.preprocess_batch(batch)

    gradient = step.train_loss(**pre)
    metrics = _row_metrics(gradient)
    assert float(metrics["response_tokens"]) == 8.0
    assert float(metrics["ratio_mean"]) == pytest.approx(1.0, abs=1e-5)
    assert float(metrics["divergence_mean"]) == 0.0
    gradient["loss"].backward()
    magnitude = sum(
        float(param.grad.abs().sum())
        for param in step.model.parameters()
        if param.grad is not None
    )
    # A collapsed ratio produces a much smaller gradient.
    assert magnitude > 1.0

    # Recreate the model used to score the fixture.
    set_seed_local(config.seed)
    update = config.step.make()
    before = next(update.model.parameters()).detach().clone()
    result = update.train_step(**update.preprocess_batch(batch))
    assert update.global_step == 1
    assert torch.isfinite(result["loss"])
    assert not torch.equal(before, next(update.model.parameters()).detach())
