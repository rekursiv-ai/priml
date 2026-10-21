"""Check the native study: family training, fresh inference, records and publication."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final, NotRequired, TypedDict, cast

import argparse
import gc
import json
import os
import platform
import sys

from configgle import Fig, PartialConfig
from treekle import json_loads, to_plain

import numpy as np
import pytest
import torch

from priml.baselines.wellbore_geology import pipeline
from priml.baselines.wellbore_geology.data import (
    FoldObservations,
    PredictionRows,
    load_split,
)
from priml.baselines.wellbore_geology.inference import (
    FamilyArtifacts,
    PredictionTable,
    SingleFamilyPrediction,
    read_table,
    write_table,
)
from priml.baselines.wellbore_geology.pipeline import (
    FamilyPrediction,
    Flags,
    FreshInference,
    NativeFamilyRunner,
    WellboreStudy,
    _add_prediction_arguments,
    accumulate_prediction,
    family_prediction,
    fold_identity,
    fold_settings,
    predict_family,
    release_accelerator,
    train_family,
)
from priml.baselines.wellbore_geology.recipes import (
    Family,
    family_0729_v3,
    family_0803_v2,
)
from priml.baselines.wellbore_geology.testing import (
    tiny_family,
    write_study_inputs,
    write_training_wells,
)
from priml.baselines.wellbore_geology.train_loop import WellboreFoldLoop
from priml.optimizers.lr_schedule import RecursiveCosine
from priml.runtime import SingleProcess, runtime_initialized
from priml.train.budget import WallBudget
from priml.train.epoch_loop import EarlyStopping, first_periodic_and_last


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


def test_flattened_fold_settings_retain_all_native_numerics() -> None:
    family = family_0729_v3().finalized()
    actual = fold_settings(family, index=11, training=(), validation=()).finalized()
    assert actual.seed == 18
    assert actual.max_epochs == 200
    assert isinstance(actual.early_stopping, EarlyStopping.Config)
    assert (actual.early_stopping.min_epochs, actual.early_stopping.patience) == (
        150,
        50,
    )
    assert actual.step.accumulate_grad_batches == 1
    assert actual.step.gradient_clip_norm == 10
    assert actual.step.dtype_autocast is torch.bfloat16
    assert actual.eval_epochs == PartialConfig(first_periodic_and_last, every=5)
    assert actual.dataset.loader.seed == 11_008
    assert actual.dataset.loader.dataset.streams.seed == 18
    assert isinstance(actual.runtime, SingleProcess.Config)
    assert actual.runtime.deterministic is False
    assert actual.runtime.float32_matmul_precision == "high"
    assert isinstance(family.training.runtime, SingleProcess.Config)
    family.training.runtime.deterministic = True
    family.training.runtime.float32_matmul_precision = "medium"
    changed = fold_settings(family, index=11, training=(), validation=())
    assert isinstance(changed.runtime, SingleProcess.Config)
    assert changed.runtime.deterministic is True
    assert changed.runtime.float32_matmul_precision == "medium"


def test_complete_repeat_major_checkpoint_identities() -> None:
    family = family_0729_v3()
    assert [fold_identity(family, index=index) for index in range(15)] == [
        f"repeat{repeat}_fold{fold}" for repeat in range(3) for fold in range(5)
    ]
    family.cv.repeats = 1
    assert fold_identity(family, index=2) == "fold2"


def test_empty_validation_partition_fails_before_model_construction(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "data"
    write_study_inputs(dataset, rows=13)
    family = tiny_family(particle=False)
    family.dataset_directory = dataset
    raw = load_split(dataset / "train")
    with pytest.raises(ValueError, match="nonempty validation"):
        train_family(
            family,
            raw={"well0": raw["well0"]},
            output_directory=tmp_path / "unused",
        )


def test_fold_cleanup_preserves_cpu_and_flushes_the_declared_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    def collect() -> int:
        events.append("collect")
        return 0

    def synchronize(device: str) -> None:
        events.append(f"synchronize:{device}")

    def empty_cache() -> None:
        events.append("empty")

    monkeypatch.setattr(gc, "collect", collect)
    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    monkeypatch.setattr(torch.cuda, "empty_cache", empty_cache)
    release_accelerator("cpu")
    assert events == []
    release_accelerator("cuda:3")
    assert events == ["collect", "synchronize:cuda:3", "empty"]


def test_loader_construction_preserves_global_torch_draws_and_initial_model(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "dataset"
    write_study_inputs(directory)
    names = tuple(load_split(directory / "train"))
    family = tiny_family(particle=False).finalized()
    config = family.training.dataset.loader
    config.seed = 719
    config.dataset.streams.seed = 7
    observations = FoldObservations.Config()
    observations.directory = directory / "train"
    observations.geography = family.geography.copy_tree()
    observations.well_ids = observations.support_well_ids = names
    config.dataset.observations = observations
    torch.manual_seed(719)
    before = torch.get_rng_state().clone()
    loader = config.make().loader
    assert torch.equal(torch.get_rng_state(), before)
    after_data = family.training.step.model.make()
    after_data_rng = torch.get_rng_state().clone()
    torch.manual_seed(719)
    before_data = family.training.step.model.make()
    second = config.make().loader
    assert torch.equal(torch.get_rng_state(), after_data_rng)
    assert len(loader) == len(second)
    assert after_data.state_dict().keys() == before_data.state_dict().keys()
    for name, value in after_data.state_dict().items():
        assert torch.equal(value, before_data.state_dict()[name]), name


def test_finalized_family_resolves_a_fresh_complete_fold_horizon() -> None:

    family = family_0729_v3()
    family.training.max_epochs = 3
    family.training.step.accumulate_grad_batches = 2
    family.training.dataset.loader.batch_size = 2
    family.training.dataset.loader.drop_last = False
    family = family.finalized()
    names = ("well0", "well1", "well2", "well3", "well4")
    config = fold_settings(family, index=11, training=names, validation=())
    schedule = config.step.lr_schedule
    assert isinstance(schedule, RecursiveCosine.Config)
    assert schedule.T_max == 6
    original = family.training.dataset.loader.dataset.observations
    assert isinstance(original, FoldObservations.Config)
    assert original.well_ids == ()


@pytest.mark.parametrize("startup_seconds", [42.0, 271.0])
def test_cold_loop_startup_consumes_outer_allowance_and_expiration_closes_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    startup_seconds: float,
) -> None:

    elapsed = 10.0

    def clock() -> float:
        return elapsed

    initialize = WellboreFoldLoop.__init__
    train = WellboreFoldLoop.train
    caps: list[float] = []

    def construct(loop: WellboreFoldLoop, config: WellboreFoldLoop.Config) -> None:
        nonlocal elapsed
        initialize(loop, config)
        elapsed += startup_seconds

    def run(loop: WellboreFoldLoop) -> None:
        caps.append(loop.max_time)
        train(loop)

    monkeypatch.setattr(WellboreFoldLoop, "__init__", construct)
    monkeypatch.setattr(WellboreFoldLoop, "train", run)
    dataset = tmp_path / "data"
    write_study_inputs(dataset)
    output = tmp_path / "trained"
    output.mkdir()
    family = tiny_family(particle=False)
    family.first_n_folds = 1
    family.dataset_directory = dataset
    family.pf_directory = tmp_path / "pf"
    config = WallBudget.Config(seconds=300, reserve_evaluation_sec=30)
    config.clock = clock
    budget = config.make()
    if startup_seconds >= 270:
        with pytest.raises(TimeoutError, match="No training time remains"):
            train_family(
                family,
                raw=load_split(dataset / "train"),
                output_directory=output,
                budget=budget,
            )
        assert caps == []
    else:
        train_family(
            family,
            raw=load_split(dataset / "train"),
            output_directory=output,
            budget=budget,
        )
        assert caps == [228.0]
    assert not runtime_initialized()


def test_training_requires_the_family_dataset_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="dataset directory"):
        train_family(
            tiny_family(particle=False),
            raw={},
            output_directory=tmp_path / "unused",
        )


def test_training_requires_a_runtime_device(tmp_path: Path) -> None:
    dataset = tmp_path / "data"
    write_study_inputs(dataset)
    family = tiny_family(particle=False)
    family.dataset_directory = dataset
    assert isinstance(family.training.runtime, SingleProcess.Config)
    family.training.runtime.device = None
    with pytest.raises(ValueError, match="runtime device"):
        train_family(
            family,
            raw=load_split(dataset / "train"),
            output_directory=tmp_path / "unused",
        )


def test_particle_family_requires_a_cache_directory_or_a_prepared_cache(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "data"
    write_study_inputs(dataset)
    family = tiny_family(particle=True)
    family.dataset_directory = dataset
    raw = load_split(dataset / "train")
    with pytest.raises(ValueError, match="cache directory"):
        train_family(family, raw=raw, output_directory=tmp_path / "unused")
    family.pf_directory = tmp_path / "pf"
    with pytest.raises(FileNotFoundError, match="not prepared"):
        train_family(family, raw=raw, output_directory=tmp_path / "unused")


def test_budget_expiring_before_any_update_is_a_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    def skip(loop: WellboreFoldLoop) -> None:
        del loop

    monkeypatch.setattr(WellboreFoldLoop, "train", skip)
    dataset = tmp_path / "data"
    write_study_inputs(dataset)
    output = tmp_path / "trained"
    output.mkdir()
    family = tiny_family(particle=False)
    family.dataset_directory = dataset
    budget = WallBudget.Config(seconds=300, reserve_evaluation_sec=30).make()
    with pytest.raises(TimeoutError, match="before an optimizer update"):
        train_family(
            family,
            raw=load_split(dataset / "train"),
            output_directory=output,
            budget=budget,
        )
    assert not runtime_initialized()


def test_training_without_a_held_out_loader_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = fold_settings

    def unvalidated(
        family: Family.Config,
        *,
        index: int,
        training: Sequence[str],
        validation: Sequence[str],
    ) -> WellboreFoldLoop.Config:
        config = real(family, index=index, training=training, validation=validation)
        config.dataset.validation = None
        config.early_stopping = None
        return config

    monkeypatch.setattr(pipeline, "fold_settings", unvalidated)
    dataset = tmp_path / "data"
    write_study_inputs(dataset)
    output = tmp_path / "trained"
    output.mkdir()
    family = tiny_family(particle=False)
    family.first_n_folds = 1
    family.dataset_directory = dataset
    with pytest.raises(ValueError, match="validation is not None"):
        train_family(family, raw=load_split(dataset / "train"), output_directory=output)


def test_fifteen_model_bin_mean_divides_each_model_in_insertion_order() -> None:
    pattern = (np.arange(12) % 7).reshape(2, 6).astype(np.float32)
    total = None
    for index in range(15):
        prediction = pattern / np.float32(11 + index) - np.float32(0.2) * (index % 3)
        total = accumulate_prediction(total, prediction, count=15)
    # Dividing once at the end, or in reverse order, rounds these bins differently.
    mean = [-0.20000002, -0.14086735, -0.08173467, -0.022602012, 0.03653065, 0.0956633]
    np.testing.assert_array_equal(
        total,
        np.asarray([mean, [0.15479596, *mean[:5]]], dtype=np.float32),
        strict=True,
    )


@pytest.mark.parametrize("count", [0, -1])
def test_model_bin_reduction_requires_a_positive_complete_count(count: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        accumulate_prediction(None, np.zeros((2, 6), dtype=np.float32), count=count)


def test_single_model_mean_owns_its_complete_running_bins() -> None:
    prediction = np.asarray(
        [[1.25, -3.5, 7.125], [-2.5, 4.75, -8.625]],
        dtype=np.float32,
    )
    expected = prediction.copy()
    total = accumulate_prediction(None, prediction, count=1)
    assert total is not prediction
    np.testing.assert_array_equal(total, expected, strict=True)
    np.testing.assert_array_equal(prediction, expected, strict=True)


def test_inference_uses_native_config_all_checkpoint_paths_and_fresh_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []

    def run(command: list[str], *, cwd: Path, check: bool) -> None:
        assert cwd == tmp_path
        assert check is True
        commands.append(command)

    monkeypatch.setattr(
        "priml.baselines.wellbore_geology.pipeline.subprocess.run",
        run,
    )
    cfg = FreshInference.Config()
    cfg.working_directory = tmp_path
    path = cfg.make().predict(
        configuration=tmp_path / "family.json",
        directory=tmp_path / "trained",
    )
    assert path == tmp_path / "trained/inference/submission_details.pqt"
    assert commands == [
        [
            sys.executable,
            "-m",
            "priml",
            "priml.baselines.wellbore_geology.pipeline.family_prediction",
            "--configuration",
            str(tmp_path / "family.json"),
            "--directory",
            str(tmp_path / "trained"),
            "--output",
            str(tmp_path / "trained/inference"),
        ],
    ]


def test_inference_rejects_empty_checkpoints_before_creating_outputs(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "trained"
    directory.mkdir()
    (directory / "checkpoints.json").write_text("[]")
    output = tmp_path / "inference"
    family = tiny_family(particle=False)
    family.dataset_directory = tmp_path / "data"
    with pytest.raises(ValueError, match="every selected checkpoint"):
        predict_family(family, directory=directory, output=output)
    assert not output.exists()


def test_inference_main_deserializes_the_complete_native_family(
    tmp_path: Path,
) -> None:
    config = tmp_path / "family.json"
    family = tiny_family(particle=False)
    family.dataset_directory = tmp_path / "data"
    config.write_text(json.dumps(to_plain(family)))
    directory = tmp_path / "trained"
    directory.mkdir()
    (directory / "checkpoints.json").write_text("[]")
    command: list[str] = []
    for name, path in (
        ("configuration", config),
        ("directory", directory),
        ("output", tmp_path / "inference"),
    ):
        command.extend((f"--{name}", str(path)))
    with pytest.raises(ValueError, match="every selected checkpoint"):
        FamilyPrediction.Config().make().run(*command)


def test_fresh_main_deserializes_and_forwards_every_declared_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    family_config = tiny_family(particle=True)
    configuration = tmp_path / "family.json"
    configuration.write_text(json.dumps(to_plain(family_config)))
    calls: list[tuple[Family.Config, dict[str, object]]] = []

    def record(family: Family.Config, **paths: object) -> Path:
        calls.append((family, paths))
        return tmp_path / "prediction.pqt"

    paths = {name: tmp_path / name for name in ("directory", "output")}
    command = [
        argument
        for name, path in {"configuration": configuration, **paths}.items()
        for argument in (f"--{name}", str(path))
    ]
    monkeypatch.setattr(
        "priml.baselines.wellbore_geology.pipeline.predict_family",
        record,
    )
    FamilyPrediction.Config().make().run(*command)
    assert len(calls) == 1
    actual, forwarded = calls[0]
    assert to_plain(actual) == to_plain(family_config), actual.udiff(
        family_config,
        finalize=False,
        color=False,
    )
    assert forwarded == {"directory": paths["directory"], "output": paths["output"]}


def test_fresh_cli_requires_all_input_and_output_paths(tmp_path: Path) -> None:
    parser = argparse.ArgumentParser()
    _add_prediction_arguments(parser)
    with pytest.raises(SystemExit):
        parser.parse_args([])
    command = [
        argument
        for name in ("configuration", "directory", "output")
        for argument in (f"--{name}", str(tmp_path / name))
    ]
    flags = cast("Flags", parser.parse_args(command))
    assert flags.configuration == tmp_path / "configuration"
    assert flags.directory == tmp_path / "directory"
    assert flags.output == tmp_path / "output"


def test_fresh_entry_point_reports_missing_paths(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit, match=r"^2$"):
        family_prediction().make().run()
    error = capsys.readouterr().err
    for name in ("configuration", "directory", "output"):
        assert f"--{name}" in error


def test_inference_requires_the_family_dataset_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="dataset directory and device"):
        predict_family(
            tiny_family(particle=False),
            directory=tmp_path / "trained",
            output=tmp_path / "inference",
        )


def test_particle_inference_requires_a_cache_directory(tmp_path: Path) -> None:
    directory = tmp_path / "trained"
    directory.mkdir()
    (directory / "checkpoints.json").write_text(json.dumps(["fold0.pt"]))
    (directory / "fold0.pt").write_bytes(b"")
    write_study_inputs(tmp_path / "data")
    particle = tiny_family(particle=True)
    particle.dataset_directory = tmp_path / "data"
    with pytest.raises(ValueError, match="cache directory"):
        predict_family(particle, directory=directory, output=tmp_path / "inference")


def test_particle_inference_requires_a_prepared_test_cache(tmp_path: Path) -> None:
    directory = tmp_path / "trained"
    directory.mkdir()
    (directory / "checkpoints.json").write_text(json.dumps(["fold0.pt"]))
    (directory / "fold0.pt").write_bytes(b"")
    write_study_inputs(tmp_path / "data")
    particle = tiny_family(particle=True)
    particle.dataset_directory = tmp_path / "data"
    particle.pf_directory = tmp_path / "pf"
    with pytest.raises(FileNotFoundError, match="not prepared"):
        predict_family(particle, directory=directory, output=tmp_path / "inference")


_CWD: Final = Path(__file__).resolve().parent


class Receipt(TypedDict):
    """Shared durable receipt with caller-owned wellbore metadata."""

    status: str
    budget_seconds: float
    elapsed_seconds: float
    metadata: dict[str, object]
    error: NotRequired[str]


class RecordingFamilyRunner:
    """Provide complete native artifact files at the study's external family seam."""

    class Config(Fig["RecordingFamilyRunner"]):
        fail: bool = False
        """Raise the injected external family failure before publication."""

    def __init__(self, config: Config) -> None:
        self.config = config

    def run(
        self,
        family: Family.Config,
        *,
        output_directory: Path,
        budget: WallBudget | None,
    ) -> FamilyArtifacts:
        """Publish complete native probe rows for the configured study caller."""
        runtime = family.training.runtime
        assert isinstance(runtime, SingleProcess.Config)
        assert runtime.device == "cpu"
        if self.config.fail:
            raise ValueError("native family failed")
        output_directory.mkdir()
        (output_directory / "budget.txt").write_text(
            "none" if budget is None else str(budget.seconds),
        )
        rows = PredictionRows(
            well_ids=("well0", "well1", "well2", "well3"),
            submit_indices=np.asarray([4, 4, 4, 4], dtype=np.int64),
            tvt=np.asarray([0.25, 0.5, 0.75, 1], dtype=np.float32),
            z=np.zeros(4, dtype=np.float32),
            target=np.zeros(4, dtype=np.float32),
        )
        oof, test = output_directory / "oof.parquet", output_directory / "test.parquet"
        write_table(oof, PredictionTable(rows=rows))
        write_table(test, PredictionTable(rows=rows))
        return FamilyArtifacts(oof=oof, test=test)


class InProcessInference:
    """Exercise complete native reconstruction while isolating process startup."""

    class Config(Fig["InProcessInference"]):
        pass

    def __init__(self, config: Config) -> None:
        del config

    def predict(
        self,
        *,
        configuration: Path,
        directory: Path,
    ) -> Path:
        """Run every real inference stage from its serialized native config."""
        family = Family.Config.deserialize(json_loads(configuration.read_text()))
        return predict_family(
            family,
            directory=directory,
            output=directory / "inference",
        )


def test_declared_output_templates_resolve_without_touching_the_filesystem(
    tmp_path: Path,
) -> None:
    config = WellboreStudy.Config(base_dir=tmp_path, experiment_name="exp005")
    resolved = config.finalized()
    assert resolved.working_dir == tmp_path / "runs/wellbore_geology/exp005"
    assert resolved.pf_directory == tmp_path / "caches/wellbore_geology/exp005-pf"
    assert not (tmp_path / "runs").exists()
    assert resolved.copy_tree().finalized() == resolved
    config.working_dir = tmp_path / "explicit-run"
    config.pf_directory = tmp_path / "explicit-cache"
    explicit = config.finalized()
    assert explicit.working_dir == config.working_dir
    assert explicit.pf_directory == config.pf_directory


def test_real_native_family_trains_all_folds_and_reconstructs_every_test_row(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "data"
    write_study_inputs(dataset)
    family = tiny_family(particle=True)
    family.first_n_folds = 2
    family.prepare_particles = True
    family.dataset_directory = dataset
    family.pf_directory = tmp_path / "pf"
    config = NativeFamilyRunner.Config()
    config.inference = InProcessInference.Config()
    artifacts = config.make().run(
        family,
        output_directory=tmp_path / "trained",
        budget=WallBudget.Config(seconds=300, reserve_evaluation_sec=30).make(),
    )
    assert artifacts.oof.name == "oof_df.pqt"
    assert artifacts.test.name == "submission_details.pqt"
    assert json_loads(
        (tmp_path / "trained/checkpoints.json").read_text(),
        list[str],
    ) == ["fold0.pt", "fold1.pt"]
    summaries = json_loads(
        (tmp_path / "trained/folds.json").read_text(),
        list[dict[str, object]],
    )
    assert [record["fold"] for record in summaries] == ["fold0", "fold1"]
    assert [record["best_epoch"] for record in summaries] == [1, 1]
    for path, target_present in ((artifacts.oof, True), (artifacts.test, False)):
        table = read_table(path)
        assert len(table.rows.tvt) == 4 * 29
        assert set(table.rows.well_ids) == {"well0", "well1", "well2", "well3"}
        assert table.rows.tvt.dtype == np.float32
        assert (table.rows.target is not None) is target_present
        assert np.count_nonzero(~np.isfinite(table.rows.tvt)) == 0
        assert "pred_lookup_GR" in table.diagnostics
        assert "geo_prior_TVT" in table.diagnostics
        assert "PF_TVT" in table.diagnostics
        assert "geo_nbr_distance_q10" in table.diagnostics
    assert len(list((tmp_path / "pf/train").glob("*.npz"))) == 4
    assert len(list((tmp_path / "pf/test").glob("*.npz"))) == 4


def _recorded_study(
    tmp_path: Path,
    *,
    fail: bool = False,
    budget: bool = True,
) -> WellboreStudy.Config:
    cfg = WellboreStudy.Config()
    cfg.device = "cpu"
    cfg.dataset_directory = tmp_path / "data"
    cfg.working_dir = tmp_path / "run"
    cfg.pf_directory = tmp_path / "pf"
    cfg.families = [tiny_family(particle=True)]
    if budget:
        cfg.budget = WallBudget.Config(seconds=300, reserve_evaluation_sec=30)
    cfg.runner = RecordingFamilyRunner.Config(fail=fail)
    cfg.results.prediction = SingleFamilyPrediction.Config()
    write_training_wells(cfg.dataset_directory / "train")
    write_training_wells(cfg.dataset_directory / "test")
    (cfg.dataset_directory / "sample_submission.csv").write_text(
        "id,tvt\nwell3_4,0\nwell0_4,0\nwell2_4,0\nwell1_4,0\n",
    )
    return cfg


@pytest.mark.parametrize("fail", [False, True])
def test_study_records_measured_publication_and_external_family_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fail: bool,
) -> None:
    _stub_config_text(monkeypatch)
    cfg = _recorded_study(tmp_path, fail=fail)
    run = tmp_path / "run"
    if fail:
        with pytest.raises(ValueError, match="native family failed"):
            cfg.make().run()
    else:
        cfg.make().run()
    record = json_loads(
        (run / "run.json").read_text(),
        Receipt,
    )
    assert record["status"] == ("failed" if fail else "complete")
    metadata = record["metadata"]
    assert metadata["train_wells"] == metadata["test_wells"] == 4
    assert metadata["metric_direction"] == "minimize"
    if fail:
        assert "error" in record
        assert record["error"] == "ValueError: native family failed"
    else:
        assert metadata["metrics"] == {
            "rmse": ((0.25**2 + 0.5**2 + 0.75**2 + 1) / 4) ** 0.5,
        }
        assert (run / "submission.csv").exists()
    with pytest.raises(FileExistsError):
        cfg.make().run()


def test_missing_inputs_fail_before_output_creation(tmp_path: Path) -> None:
    cfg = WellboreStudy.Config()
    cfg.families = [tiny_family(particle=True)]
    cfg.dataset_directory = tmp_path / "missing"
    cfg.working_dir = tmp_path / "run"
    with pytest.raises(FileNotFoundError, match="No paired well CSVs"):
        cfg.make().run()
    assert not cfg.working_dir.exists()


def test_study_requires_a_complete_family_before_creating_outputs(
    tmp_path: Path,
) -> None:
    config = WellboreStudy.Config()
    config.working_dir = tmp_path / "unused"
    with pytest.raises(
        ValueError,
        match=r"^A wellbore study requires at least one complete family\.$",
    ):
        config.make().run()
    assert not config.working_dir.exists()


def test_pprint_needs_neither_datasets_nor_pretrained_weights(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = WellboreStudy.Config()
    cfg.families = [family_0803_v2()]
    cfg.dataset_directory = tmp_path / "missing"
    cfg.working_dir = tmp_path / "run"
    monkeypatch.setattr(WellboreStudy.Config, "pprint", _stub_pprint)
    cfg.make().run("--pprint")
    assert capsys.readouterr().out == "stub config\n"
    assert not cfg.working_dir.exists()


def _no_flush(descriptor: int) -> None:
    del descriptor


def _stub_pformat(config: WellboreStudy.Config, **kwargs: object) -> str:
    del config, kwargs
    return "stub config\n"


def _stub_pprint(config: WellboreStudy.Config, **kwargs: object) -> None:
    del config, kwargs
    sys.stdout.write("stub config\n")


def _stub_config_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the two config renderers and disk flushes of a study run."""
    monkeypatch.setattr(os, "fsync", _no_flush)
    monkeypatch.setattr(WellboreStudy.Config, "pformat", _stub_pformat)
    monkeypatch.setattr(WellboreStudy.Config, "pprint", _stub_pprint)


def _record_hidden_defaults(
    monkeypatch: pytest.MonkeyPatch,
    method: str,
) -> list[object]:
    """Record the exact ``hide_default_values`` argument each call receives."""
    original = cast("Callable[..., object]", getattr(WellboreStudy.Config, method))
    seen: list[object] = []

    def wrapper(
        config: WellboreStudy.Config,
        *args: object,
        **kwargs: object,
    ) -> object:
        seen.append(kwargs.get("hide_default_values", "unset"))
        return original(config, *args, **kwargs)

    monkeypatch.setattr(WellboreStudy.Config, method, wrapper)
    return seen


@pytest.mark.parametrize("budget", [True, False])
def test_receipt_names_the_study_scope_and_forwards_the_outer_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    budget: bool,
) -> None:
    cfg = _recorded_study(tmp_path, budget=budget)
    seen: list[object] = []

    def pformat(config: WellboreStudy.Config, **kwargs: object) -> str:
        del config
        seen.append(kwargs.get("hide_default_values", "unset"))
        return "stub config\n"

    monkeypatch.setattr(os, "fsync", _no_flush)
    monkeypatch.setattr(WellboreStudy.Config, "pformat", pformat)
    cfg.make().run()
    assert seen == [False]
    run = tmp_path / "run"
    metadata = json_loads((run / "run.json").read_text(), Receipt)["metadata"]
    assert metadata["schema_version"] == 1
    assert metadata["study"] == "wellbore_geology"
    assert metadata["experiment"] == "exp000"
    assert metadata["reference_revision"] == cfg.reference_revision
    assert metadata["competition"] == "rogii-wellbore-geology-prediction"
    assert metadata["metric"] == "rmse"
    assert metadata["device"] == "cpu"
    assert metadata["host"] == platform.node()
    assert metadata["outer_budget_sec"] == (300 if budget else None)
    timing_scope = metadata["timing_scope"]
    assert isinstance(timing_scope, str)
    assert timing_scope.startswith("Study.run including cold preprocessing")
    assert timing_scope.endswith("excludes downloads and launcher imports")
    assert (run / "0803_V2/budget.txt").read_text() == ("300" if budget else "none")
    assert (run / "config.txt").read_text() == "stub config\n"


def test_run_creates_missing_parents_and_protects_its_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_config_text(monkeypatch)
    cfg = _recorded_study(tmp_path)
    cfg.working_dir = tmp_path / "deep/new/run"
    cfg.make().run()
    assert (tmp_path / "deep/new/run/run.json").exists()
    inside = _recorded_study(tmp_path / "other")
    assert isinstance(inside.dataset_directory, Path)
    inside.working_dir = inside.dataset_directory
    with pytest.raises(ValueError, match="data"):
        inside.make().run()
    cache = _recorded_study(tmp_path / "third")
    cache.working_dir = cache.pf_directory
    with pytest.raises(ValueError, match="pf"):
        cache.make().run()


@pytest.mark.parametrize("prepares", [True, False])
def test_budgeted_cold_run_rejects_only_a_used_cache_it_will_prepare(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepares: bool,
) -> None:
    _stub_config_text(monkeypatch)
    cfg = _recorded_study(tmp_path)
    cfg.families[0].prepare_particles = prepares
    used = tmp_path / "pf/0803_V2"
    used.mkdir(parents=True)
    (used / "old.npz").write_bytes(b"")
    if prepares:
        with pytest.raises(FileExistsError, match="empty PF cache"):
            cfg.make().run()
    else:
        cfg.make().run()


def test_unbudgeted_run_never_checks_for_a_cold_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_config_text(monkeypatch)
    cfg = _recorded_study(tmp_path, budget=False)
    cfg.families[0].prepare_particles = True
    used = tmp_path / "pf/0803_V2"
    used.mkdir(parents=True)
    (used / "old.npz").write_bytes(b"")
    cfg.make().run()


def test_run_help_describes_the_study_and_pprint_shows_every_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = _recorded_study(tmp_path)
    with pytest.raises(SystemExit):
        cfg.make().run("--help")
    assert "Train the complete native study" in capsys.readouterr().out
    seen = _record_hidden_defaults(monkeypatch, method="pprint")
    cfg.make().run("--pprint")
    assert seen == [False]
    assert "prepare_particles=False" in capsys.readouterr().out


def test_native_runner_requires_the_family_dataset_directory(tmp_path: Path) -> None:
    runner = NativeFamilyRunner.Config().make()
    with pytest.raises(ValueError, match="dataset directory"):
        runner.run(
            tiny_family(particle=False),
            output_directory=tmp_path / "unused",
            budget=None,
        )
    assert not (tmp_path / "unused").exists()


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
