"""The benchmark config is printable without competition data or weights."""

from __future__ import annotations

from functools import partialmethod
from pathlib import Path
from typing import TYPE_CHECKING, NotRequired, TypedDict, override

import csv
import math

from configgle import Makes
from treekle import json_loads

import numpy as np
import pytest
import torch

from priml.baselines.name_that_llm.benchmark import Benchmark
from priml.baselines.name_that_llm.encoder import FrozenQwen
from priml.baselines.name_that_llm.model import (
    CenteredHeads,
    QwenHeads,
    Split,
    SvmBranch,
)
from priml.baselines.name_that_llm.testing import (
    miniature_split,
    recipe,
)
from priml.launch import _stamp_run_identity
from priml.train.budget import WallBudget


if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray


@pytest.fixture
def hub_extra() -> None:
    # A run records the installed ``transformers`` version, priml's ``hub`` extra.
    pytest.importorskip("transformers")


@pytest.fixture
def wiring_config(tmp_path: Path) -> Benchmark.Config:
    with (tmp_path / "train.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "prompt_id", "text", "model"])
        writer.writerows(
            [
                [
                    f"tr-{group}-{label}",
                    f"p-{group}",
                    f"Topic {group} Style {label}",
                    label,
                ]
                for group in range(6)
                for label in ["a", "b", "c"]
            ],
        )
    (tmp_path / "test.csv").write_text("id,text\nte-1,Topic Test\nte-2,Another Topic\n")
    config = Benchmark.Config()
    config.dataset.directory = tmp_path
    config.dataset.expected_rows = 18
    config.dataset.expected_prompts = 6
    config.dataset.labels = ["a", "b", "c"]
    config.encoder = _FixtureEncoder.Config()
    config.base_dir = None
    config.working_dir = tmp_path / "run"
    return config


def test_benchmark_exposes_complete_public_recipe_without_io() -> None:
    config = recipe()
    assert config.outer_folds == 5
    assert list(config.branches) == ["svm", "plain", "centred"]
    assert config.encoder.layers == [4, 6, 8, 22]
    assert config.blend.weights == {"svm": 0.4, "plain": 0.3, "centred": 0.3}
    assert isinstance(config.budget, WallBudget.Config)
    assert math.isinf(config.budget.seconds)


def test_output_identity_resolves_without_io_and_preserves_literal_paths(
    tmp_path: Path,
) -> None:
    config = Benchmark.Config()
    config.base_dir = tmp_path
    config.study_name = "study"
    config.experiment_name = "exp007"
    resolved = config.copy_tree().finalize()
    assert resolved.working_dir == tmp_path / "artifacts/study/exp007"
    assert resolved.dataset.directory == tmp_path / "datasets/name-that-llm/raw"
    assert (
        resolved.encoder.directory
        == tmp_path / "datasets/name-that-llm/qwen-3-8b-base-v1"
    )
    assert config.working_dir == "/artifacts/{study_name}/{experiment_name}"
    config.base_dir = None
    config.working_dir = tmp_path / "literal-{experiment_name}"
    assert config.copy_tree().finalize().working_dir == config.working_dir


def test_unset_identity_uses_the_launched_factory_and_retains_overrides(
    tmp_path: Path,
) -> None:
    config = Benchmark.Config(base_dir=tmp_path)
    assert config.experiment_name == ""
    _stamp_run_identity(
        config,
        module_name="priml.baselines.name_that_llm.experiments",
        function_name="exp017",
    )
    resolved = config.copy_tree().finalize()
    assert isinstance(resolved.working_dir, Path)
    assert resolved.working_dir == tmp_path / "artifacts/name-that-llm/exp017"
    assert resolved.budget.receipt_path == resolved.working_dir / "result.json"
    assert config.working_dir == "/artifacts/{study_name}/{experiment_name}"
    config.experiment_name = "explicit"
    _stamp_run_identity(
        config,
        module_name="priml.baselines.name_that_llm.experiments",
        function_name="exp018",
    )
    assert (
        config.copy_tree().finalize().working_dir
        == tmp_path / "artifacts/name-that-llm/explicit"
    )


@pytest.mark.usefixtures("hub_extra")
def test_wiring_records_score_inputs_timing_and_submission(
    wiring_config: Benchmark.Config,
) -> None:
    _WiringBenchmark(wiring_config.finalized()).run()
    result = json_loads(
        (Path(wiring_config.working_dir) / "result.json").read_text(),
        target=_Result,
    )
    assert result["status"] == "complete"
    metadata = result["metadata"]
    assert metadata["balanced_accuracy"] == 1 / 3
    assert metadata["validation_rows"] == 18
    assert metadata["validation_prompts"] == 6
    assert metadata["submission_rows"] == 2
    assert (Path(wiring_config.working_dir) / "submission.parquet").is_file()
    assert (Path(wiring_config.working_dir) / "validation.npz").is_file()
    assert "accelerator" in metadata
    assert "input_sha256" in metadata
    assert "elapsed_seconds" in result
    assert result["budget_seconds"] == math.inf


def test_all_branches_reach_the_public_source_blend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, list[NDArray[np.float64]]] = {}

    for name, branch in {
        "svm": SvmBranch,
        "plain": QwenHeads,
        "centred": CenteredHeads,
    }.items():
        monkeypatch.setattr(
            branch,
            name="predict",
            value=partialmethod(
                _observe_call,
                method=branch.predict,
                name=name,
                observed=observed,
            ),
        )
    config = recipe()
    scores = config.make()._predict(miniature_split())
    assert {name: len(values) for name, values in observed.items()} == {
        "svm": 1,
        "plain": 2,
        "centred": 1,
    }
    assert np.array_equal(
        scores,
        a2=config.blend.make()(
            {
                "svm": observed["svm"][0],
                "plain": observed["plain"][0],
                "centred": observed["centred"][0],
            },
        ),
    )


@pytest.mark.usefixtures("hub_extra")
def test_failed_run_preserves_durable_error(wiring_config: Benchmark.Config) -> None:
    wiring_config.dataset.directory = Path(wiring_config.dataset.directory) / "missing"
    with pytest.raises(FileNotFoundError):
        _WiringBenchmark(wiring_config.finalized()).run()
    result = json_loads(
        (Path(wiring_config.working_dir) / "result.json").read_text(),
        target=dict[str, object],
    )
    assert result["status"] == "failed"
    assert "error" in result
    assert "elapsed_seconds" in result


@pytest.mark.usefixtures("hub_extra")
def test_over_budget_candidate_cannot_receive_a_score(
    wiring_config: Benchmark.Config,
) -> None:
    wiring_config.budget.seconds = 1e-15
    with pytest.raises(TimeoutError, match="exceeded"):
        _WiringBenchmark(wiring_config.finalized()).run()
    result = json_loads(
        (Path(wiring_config.working_dir) / "result.json").read_text(),
        target=_Result,
    )
    assert result["status"] == "failed"
    assert "balanced_accuracy" not in result["metadata"]


@pytest.mark.parametrize("seconds", argvalues=[0.0, -1.0, math.nan])
def test_invalid_budget_is_rejected_before_io(
    wiring_config: Benchmark.Config,
    seconds: float,
) -> None:
    wiring_config.budget.seconds = seconds
    with pytest.raises(ValueError, match="positive"):
        _WiringBenchmark(wiring_config.finalized()).run()
    assert not Path(wiring_config.working_dir).exists()


@pytest.mark.usefixtures("hub_extra")
def test_result_publication_counts_toward_budget(
    wiring_config: Benchmark.Config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _PublicationClock()
    original_replace = Path.replace

    def delayed_replace(path: Path, target: str | Path) -> Path:
        replaced = original_replace(path, target=target)
        clock.elapsed += 2 * (Path(target).name == "result.json")
        return replaced

    monkeypatch.setattr(Path, name="replace", value=delayed_replace)
    wiring_config.budget.clock = clock
    wiring_config.budget.seconds = 1.0
    with pytest.raises(TimeoutError, match="exceeded"):
        _WiringBenchmark(wiring_config.finalized()).run()
    result = json_loads(
        (Path(wiring_config.working_dir) / "result.json").read_text(),
        target=dict[str, object],
    )
    assert result["status"] == "failed"
    assert result["elapsed_seconds"] == 2
    assert "TimeoutError" in str(result["error"])


@pytest.mark.usefixtures("hub_extra")
def test_configured_budget_checks_extraction_before_any_fold_score(
    wiring_config: Benchmark.Config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, list[NDArray[np.float64]]] = {}
    monkeypatch.setattr(
        _WiringBenchmark,
        name="_predict",
        value=partialmethod(
            _observe_call,
            method=_WiringBenchmark._predict,
            name="prediction",
            observed=observed,
        ),
    )
    wiring_config.budget = _CheckpointBudget.Config()
    with pytest.raises(TimeoutError, match="Configured extraction checkpoint"):
        _WiringBenchmark(wiring_config.finalized()).run()
    result = json_loads(
        (Path(wiring_config.working_dir) / "result.json").read_text(),
        _Result,
    )
    metadata = result["metadata"]
    assert result["status"] == "failed"
    assert "accelerator" in metadata
    assert "input_sha256" in metadata
    assert "balanced_accuracy" not in metadata
    assert not (Path(wiring_config.working_dir) / "validation.npz").exists()
    assert observed == {}


@pytest.mark.usefixtures("hub_extra")
def test_explicit_receipt_path_is_retained_without_mutating_config(
    wiring_config: Benchmark.Config,
    tmp_path: Path,
) -> None:
    wiring_config.budget.receipt_path = tmp_path / "custom.json"
    _WiringBenchmark(wiring_config.finalized()).run()
    result = json_loads((tmp_path / "custom.json").read_text(), dict[str, object])
    assert result["status"] == "complete"
    assert not (Path(wiring_config.working_dir) / "result.json").exists()
    assert wiring_config.budget.receipt_path == tmp_path / "custom.json"


def test_existing_output_directory_is_rejected(wiring_config: Benchmark.Config) -> None:
    directory = Path(wiring_config.working_dir)
    directory.mkdir()
    (directory / "result.json").write_text("previous result\n")
    with pytest.raises(FileExistsError):
        _WiringBenchmark(wiring_config.finalized()).run()
    assert (directory / "result.json").read_text() == "previous result\n"


@pytest.mark.usefixtures("hub_extra")
def test_extraction_exception_identity_reaches_the_caller(
    wiring_config: Benchmark.Config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = RuntimeError("Fixture extraction failed.")

    def failed_encode(
        encoder: _FixtureEncoder,
        texts: list[str],
    ) -> NDArray[np.float32]:
        del encoder, texts
        raise error

    monkeypatch.setattr(_FixtureEncoder, "encode", failed_encode)
    with pytest.raises(RuntimeError, match="Fixture extraction failed") as caught:
        _WiringBenchmark(wiring_config.finalized()).run()
    assert caught.value is error
    result = json_loads(
        (Path(wiring_config.working_dir) / "result.json").read_text(),
        _Result,
    )
    assert result["status"] == "failed"
    assert result.get("error") == f"RuntimeError: {error}"


@pytest.mark.usefixtures("hub_extra")
def test_cuda_encoder_without_an_accelerator_fails_the_run(
    wiring_config: Benchmark.Config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    wiring_config.encoder.device = "cuda"
    with pytest.raises(RuntimeError, match="CUDA accelerator"):
        _WiringBenchmark(wiring_config.finalized()).run()
    result = json_loads(
        (Path(wiring_config.working_dir) / "result.json").read_text(),
        target=dict[str, object],
    )
    assert result["status"] == "failed"


def test_unexpected_flags_are_rejected_before_io() -> None:
    with pytest.raises(ValueError, match="configuration fields only"):
        Benchmark.Config().make().run("--untracked-hyperparameter")


def _observe_call(
    *args: object,
    method: Callable[..., NDArray[np.float64]],
    name: str,
    observed: dict[str, list[NDArray[np.float64]]],
    **kwargs: object,
) -> NDArray[np.float64]:
    result = method(*args, **kwargs)
    observed.setdefault(name, []).append(result)
    return result


class _Result(TypedDict):
    """Read the shared envelope and nested literal config tags in one pass."""

    status: str
    budget_seconds: float
    elapsed_seconds: float
    metadata: dict[str, object]
    error: NotRequired[str]


class _PublicationClock:
    def __init__(self) -> None:
        self.elapsed = 0.0

    def __call__(self) -> float:
        return self.elapsed


class _CheckpointBudget(WallBudget):
    """Verify configured cooperative checks reach the shared runtime."""

    class Config(Makes["_CheckpointBudget"], WallBudget.Config):
        """Inject a failing checkpoint without replacing receipt publication."""

    @override
    def check(self) -> None:
        raise TimeoutError("Configured extraction checkpoint.")


class _FixtureEncoder(FrozenQwen):
    """Fixture-only adapter; real encoder numerics are tested in encoder_test."""

    class Config(Makes["_FixtureEncoder"], FrozenQwen.Config):
        """Inject the wiring fixture through the existing encoder slot."""

    @override
    def encode(self, texts: list[str]) -> NDArray[np.float32]:
        return np.sin(np.arange(len(texts) * 4 * 5, dtype=np.float32)).reshape(
            len(texts),
            4,
            5,
        )


class _WiringBenchmark(Benchmark):
    """Test result persistence independently of the separately checked solvers."""

    @override
    def _predict(self, split: Split) -> NDArray[np.float64]:
        output = np.zeros((len(split.predict_texts), 3))
        output[:, 0] = 1
        return output


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
