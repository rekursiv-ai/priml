"""Complete-stage orchestration and real Priml epoch/RNG integration."""

from __future__ import annotations

from collections import Counter
from dataclasses import fields
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast, override

import copy
import functools
import signal

from configgle import Fig, InlineConfig, Makes, Makes2
from torch import Tensor
from treekle import from_plain, json_loads

import numpy as np
import pytest
import torch

from priml.baselines.motor_imagery import campaign
from priml.baselines.motor_imagery.campaign import (
    MotorImageryCampaign,
    _build_cpu,
    _diagnostic_draws,
    predict,
)
from priml.baselines.motor_imagery.data import Preprocessing, Trial, WindowData
from priml.baselines.motor_imagery.experiments import exp000, exp_smoke
from priml.baselines.motor_imagery.folds import FixedCohort, GroupedFolds
from priml.baselines.motor_imagery.metric import EEGAccuracy
from priml.baselines.motor_imagery.model import (
    AttentionEEGNet,
    MultiScaleEEGNet,
    ShallowConvNet,
)
from priml.baselines.motor_imagery.testing import (
    compare_record,
    model_record,
    optimizer_record,
    random_record,
    source_step,
)
from priml.baselines.motor_imagery.train_loop import MotorImageryLoop
from priml.baselines.motor_imagery.train_step import MotorImageryStep
from priml.math.seed import get_rng_state
from priml.model.conv import Conv2d
from priml.model.pool import AvgPool2d
from priml.optimizers.lr_schedule import RecursiveCosine
from priml.runtime import SingleProcess, runtime_initialized
from priml.testing.bfb import host_agnostic_numerics, portable_half_precision
from priml.testing.golden import (
    NativeDraws,
    PinnedRandom,
    read_tensors,
)
from priml.train.ema import EMA


if TYPE_CHECKING:
    from collections.abc import Sequence

    from priml.math.seed import RngState


_CWD: Final = Path(__file__).resolve().parent


def test_fold_configuration_declares_inputs_without_live_trials() -> None:
    assert "trials" not in {entry.name for entry in fields(WindowData.Config)}


def test_training_before_the_next_ensemble_constructor_changes_initial_state() -> None:
    cfg = exp_smoke().finalized()
    # MultiScaleEEGNet.forward consumes one Conv2d input plane.
    media = torch.arange(4 * 8 * 100).reshape(4, 1, 8, 100).float() / 100
    initial = torch.Generator(device="cpu").manual_seed(42).get_state()
    captures: list[list[dict[str, Tensor]]] = []
    states: list[list[Tensor]] = []
    with torch.random.fork_rng(devices=[]):
        for interleave in (False, True):
            torch.set_rng_state(initial)
            models: list[torch.nn.Module] = []
            records: list[dict[str, Tensor]] = []
            before: list[Tensor] = []
            with (
                host_agnostic_numerics(),
                torch.no_grad(),
            ):
                for config in cfg.ensemble:
                    before.append(torch.get_rng_state())
                    model = _build_cpu(config)
                    models.append(model)
                    records.append(model_record(model, prefix="initial"))
                    if interleave:
                        model(media)
                if not interleave:
                    for model in models:
                        model(media)
            captures.append(records)
            states.append(before)
    compare_record(captures[0][0], captures[1][0])
    assert torch.equal(states[0][0], states[1][0])
    assert not torch.equal(states[0][1], states[1][1])
    name = next(
        name
        for name, value in captures[0][1].items()
        if not torch.equal(value, captures[1][1][name])
    )
    expected = captures[0][1][name].flatten()
    actual = captures[1][1][name].flatten()
    assert name == "initial/conv_t.weight"
    assert int((expected != actual).sum()) > 0
    assert tuple(captures[0][1][name].shape) == (2, 1, 1, 25)
    assert int((expected != actual).sum()) == 50
    assert float(expected[0]).hex() == "-0x1.a8082a0000000p-4"
    assert float(actual[0]).hex() == "-0x1.ff1ece0000000p-4"


class SyntheticInput(Preprocessing):
    """Supply external raw-ingestion results without replacing model numerics."""

    class Config(Makes["SyntheticInput"], Preprocessing.Config):
        training_trial_count: int = 4
        """Number of generated training trials."""

        windows_per_trial: int = 4
        """Number of windows in each generated training trial."""

        testing_subject: str = "S008"
        """Subject identifier assigned to generated test trials."""

        testing_trial_count: int = 1
        """Number of generated masked test trials."""

        testing_windows_per_trial: int = 2
        """Number of windows in each generated masked test trial."""

        failure: BaseException | None = None
        """External ingestion failure injected by lifecycle regressions."""

        alarm: bool = False
        """Deliver the operating system deadline signal during ingestion."""

    def __init__(self, config: Config) -> None:
        self.inputs = config

    @override
    def read(self, paths: Sequence[Path], *, labelled: bool) -> list[Trial]:
        del paths
        if self.inputs.failure is not None:
            raise self.inputs.failure
        if self.inputs.alarm:
            assert 0 < signal.getitimer(signal.ITIMER_REAL)[0] <= 1
            signal.raise_signal(signal.SIGALRM)
        if labelled:
            windows = torch.zeros(
                self.inputs.windows_per_trial,
                len(self.inputs.channels),
                self.inputs.window_size,
                dtype=torch.float32,
                device="cpu",
            )
            return [
                Trial(
                    subject=f"S{index + 1:03}",
                    trial_id=index,
                    label=index % 2,
                    windows=windows,
                )
                for index in range(self.inputs.training_trial_count)
            ]
        return [
            Trial(
                subject=self.inputs.testing_subject,
                trial_id=index,
                label=-1,
                windows=torch.zeros(
                    self.inputs.testing_windows_per_trial,
                    len(self.inputs.channels),
                    self.inputs.window_size,
                    dtype=torch.float32,
                    device="cpu",
                ),
            )
            for index in range(self.inputs.testing_trial_count)
        ]


class CountingLoop(MotorImageryLoop):
    """Exercise fit orchestration independently of the numerical replay tests."""

    class Config(Makes2["CountingLoop"], MotorImageryLoop.Config):
        """Retain the actual Priml tree while replacing only fit execution."""

        fail_fit: bool = False
        """Inject a fit failure after preprocessing and runtime acquisition."""

        reseed: bool = False
        """Emulate Priml's local reseeding without hardware initialization."""

        @property
        @override
        def parent_class(self) -> type[CountingLoop]:
            return CountingLoop

        @override
        def make(self) -> CountingLoop:
            """Construct the test fit using the normal finalized-copy guard."""
            return CountingLoop(self.finalized())

    def __init__(
        self,
        config: MotorImageryLoop.Config,
        *,
        step: MotorImageryStep | None = None,
        dataset: WindowData | None = None,
    ) -> None:
        self.inputs = config
        self.step = config.step.make() if step is None else step
        self.dataset = config.dataset.make() if dataset is None else dataset
        assert isinstance(config, CountingLoop.Config)
        if config.reseed:
            torch.manual_seed(1_381_806_158)
        metric = EEGAccuracy.Config().make()
        if len(self.dataset.eval_dataset):
            _, labels, identifiers = self.dataset.eval_dataset.tensors
            metric.best_probabilities = labels.float()
            metric.best_labels = labels.clone()
            metric.best_trials = identifiers.clone()
        metric.best_correct = len(metric.best_labels)
        self.metrics_eval = {"": metric}

    @override
    def train(self) -> None:
        """Validate that each fit receives its finite source epoch budget."""
        assert self.inputs.max_epochs == self.inputs.step.train_budget_epochs
        assert isinstance(self.inputs, CountingLoop.Config)
        if self.inputs.fail_fit:
            raise ValueError("fit failure")


def synthetic_campaign(tmp_path: Path) -> MotorImageryCampaign.Config:
    """Inject raw ingestion and fit execution into a real campaign config."""
    cfg = exp_smoke()
    validation, ensemble = CountingLoop.Config(), CountingLoop.Config()
    validation.update(cfg.validation)
    ensemble.update(cfg.ensemble_loop)
    cfg.validation, cfg.ensemble_loop = validation, ensemble
    inputs = validation.dataset.preprocessing = SyntheticInput.Config()
    inputs.channels = ("Fz", "C3", "Cz")
    inputs.window_size = 100
    validation.dataset.base_dir = None
    validation.dataset.working_dir = tmp_path / "raw"
    cfg.base_dir = None
    cfg.working_dir = tmp_path / "result"
    for name in ("Training set", "Cross participant test set"):
        (tmp_path / "raw" / name).mkdir(parents=True)
    for loop in (validation, ensemble):
        loop.base_dir = None
        loop.working_dir = tmp_path / "fits"
        loop.runtime = SingleProcess.Config(device="cpu")
    cfg.expected_submission_rows = 1
    return cfg


def test_synthetic_input_reconstructs_the_original_trial_fixtures() -> None:
    config = SyntheticInput.Config()
    config.channels = ("Fz", "C3", "Cz")
    config.window_size = 100
    processor = config.make()
    rng_before = get_rng_state()
    first_training = processor.read([], labelled=True)
    first_testing = processor.read([], labelled=False)
    second_training = processor.read([], labelled=True)
    second_testing = processor.read([], labelled=False)
    rng_after = get_rng_state()

    expected_training_windows = torch.zeros(
        (4, 3, 100),
        dtype=torch.float32,
        device="cpu",
    )
    expected_testing_windows = torch.zeros(
        (2, 3, 100),
        dtype=torch.float32,
        device="cpu",
    )
    expected_training = [
        Trial(
            subject=f"S{index + 1:03}",
            trial_id=index,
            label=index % 2,
            windows=expected_training_windows,
        )
        for index in range(4)
    ]
    expected_testing = [
        Trial(
            subject="S008",
            trial_id=0,
            label=-1,
            windows=expected_testing_windows,
        ),
    ]
    for actual, expected in (
        (first_training, expected_training),
        (first_testing, expected_testing),
        (second_training, expected_training),
        (second_testing, expected_testing),
    ):
        assert [trial.subject for trial in actual] == [
            trial.subject for trial in expected
        ]
        assert [trial.trial_id for trial in actual] == [
            trial.trial_id for trial in expected
        ]
        assert [trial.label for trial in actual] == [trial.label for trial in expected]
        for trial, original in zip(actual, expected, strict=True):
            assert trial.windows.shape == original.windows.shape
            assert trial.windows.dtype == original.windows.dtype
            assert trial.windows.device == original.windows.device
            assert torch.equal(trial.windows, original.windows)
    assert first_training is not second_training
    assert first_testing is not second_testing
    assert all(
        first is not second
        for first, second in zip(first_training, second_training, strict=True)
    )
    assert first_testing[0] is not second_testing[0]
    assert first_training[0].windows is first_training[1].windows
    assert first_training[0].windows.data_ptr() != second_training[0].windows.data_ptr()
    assert first_testing[0].windows.data_ptr() != second_testing[0].windows.data_ptr()
    assert rng_before["python"] == rng_after["python"]
    assert "numpy" in rng_before
    assert "numpy" in rng_after
    assert rng_before["numpy"] == rng_after["numpy"]
    assert torch.equal(rng_before["torch"], rng_after["torch"])
    if "cuda" in rng_before:
        assert "cuda" in rng_after
        before_cuda = rng_before["cuda"]
        after_cuda = rng_after["cuda"]
        assert all(
            torch.equal(before, after)
            for before, after in zip(before_cuda, after_cuda, strict=True)
        )


def test_campaign_reports_all_folds_members_and_official_artifacts(
    tmp_path: Path,
) -> None:
    cfg = synthetic_campaign(tmp_path)
    assert isinstance(cfg.folds, GroupedFolds.Config)
    assert isinstance(cfg.validation.dataset.preprocessing, SyntheticInput.Config)
    cfg.validation.dataset.preprocessing.windows_per_trial = 2
    cfg.make().run()
    output = Path(cfg.working_dir)
    result = json_loads((output / "result.json").read_text(), dict[str, object])
    metadata = from_plain(result["metadata"], dict[str, object])
    assert result["status"] == "complete"
    assert metadata["eval/accuracy"] == 1.0
    assert metadata["validation_trials"] == 4
    assert metadata["reference_notebook_version"] == 353_208_444
    assert "torch" in from_plain(metadata["dependencies"], dict[str, object])
    assert {path.name for path in output.glob("fold-*.pt")} == {
        f"fold-{index}.pt" for index in range(cfg.folds.count)
    }
    assert {path.name for path in output.glob("ensemble-*.pt")} == {
        f"ensemble-{index}.pt" for index in range(len(cfg.ensemble))
    }
    assert (output / "config.txt").is_file()
    assert (output / "submission.csv").read_text().splitlines()[0] == "ID,TARGET"
    # Four equal people: the first and third fill fold 1, the second and fourth fold 0.
    for fold, people in enumerate(({1, 3}, {0, 2})):
        state = cast(
            "dict[str, Tensor]",
            torch.load(output / f"fold-{fold}.pt", weights_only=True),
        )
        best_trials = from_plain(state["best_trials"].tolist(), list[int])
        assert Counter(best_trials) == Counter(dict.fromkeys(people, 2))


def test_campaign_rejects_a_fold_without_a_positive_accuracy_epoch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = synthetic_campaign(tmp_path)

    def no_selected_epoch(loop: CountingLoop) -> None:
        loop.metrics_eval[""] = EEGAccuracy.Config().make()

    monkeypatch.setattr(CountingLoop, name="train", value=no_selected_epoch)
    with pytest.raises(ValueError, match="No validation epoch exceeded zero"):
        cfg.make().run()
    output = Path(cfg.working_dir)
    result = json_loads((output / "result.json").read_text(), dict[str, object])
    assert result["status"] == "failed"
    assert (
        result["error"]
        == "ValueError: No validation epoch exceeded zero window accuracy."
    )
    assert not list(output.glob("ensemble-*.pt"))
    assert not (output / "submission.csv").exists()


class TiedEpochLoop(MotorImageryLoop):
    """Select one deterministic epoch at the fit boundary."""

    class Config(Makes2["TiedEpochLoop"], MotorImageryLoop.Config):
        """Retain the normal native dataset and step configuration."""

        @property
        @override
        def parent_class(self) -> type[TiedEpochLoop]:
            return TiedEpochLoop

        @override
        def make(self) -> TiedEpochLoop:
            """Build the deterministic epoch fixture from its resolved tree."""
            return TiedEpochLoop(self.finalized())

    def __init__(
        self,
        config: Config,
        *,
        step: MotorImageryStep | None = None,
        dataset: WindowData | None = None,
    ) -> None:
        del step
        data = config.dataset.make() if dataset is None else dataset
        metric = EEGAccuracy.Config().make()
        _, labels, identifiers = cast(
            tuple[Tensor, Tensor, Tensor],
            data.eval_dataset.tensors,
        )
        for identifier in dict.fromkeys(
            from_plain(cast(object, identifiers.tolist()), list[int]),
        ):
            selected = identifiers == identifier
            metric.update(
                torch.zeros(int(selected.sum().item()), 2),
                label=labels[selected],
                trial=identifiers[selected],
            )
        metric.compute()
        self.metrics_eval = {"": metric}

    @override
    def train(self) -> None:
        """Use the selected epoch without replacing reporting arithmetic."""


def test_validation_reports_the_selected_argmax_window_score(tmp_path: Path) -> None:
    cfg = synthetic_campaign(tmp_path)
    loop = TiedEpochLoop.Config()
    loop.update(cfg.validation, skip_missing=True)
    cfg.validation = loop
    cohort = cfg.folds = FixedCohort.Config()
    cohort.subjects = ["S001", "S002", "S003"]
    assert isinstance(cfg.validation.dataset.preprocessing, SyntheticInput.Config)
    result = cfg.make()._validate(
        cfg.validation.dataset.preprocessing.make().read([], labelled=True),
        output=tmp_path,
    )
    assert result["source_window_accuracy"] == 2 / 3
    assert result["trial_accuracy"] == 1 / 3


class CopyPreparation:
    """Return independently owned model storage through the native transform slot."""

    class Config(Fig["CopyPreparation"]):
        """Construct the copy-returning preparation."""

    def __init__(self, config: Config) -> None:
        del config

    def __call__(self, model: torch.nn.Module) -> torch.nn.Module:
        """Return the replacement module that must own fitting and inference."""
        return copy.deepcopy(model)


class ObservedCampaign(MotorImageryCampaign):
    """Observe real full-cohort fitting while isolating subject validation."""

    class Config(Makes["ObservedCampaign"], MotorImageryCampaign.Config):
        """Use the actual configured ensemble and training loops."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.offered: list[torch.nn.Module] = []
        self.fitted: list[MotorImageryStep] = []
        self.inferred: list[list[Tensor]] = []

    @override
    def _validate(self, trials: list[Trial], *, output: Path) -> dict[str, float]:
        del trials, output
        return {"trial_accuracy": 0.0}

    @override
    def _make_loop(
        self,
        model: torch.nn.Module,
        *,
        config: MotorImageryLoop.Config,
        trials: list[Trial],
        validation_subjects: list[str],
    ) -> MotorImageryLoop:
        loop = super()._make_loop(
            model,
            config=config,
            trials=trials,
            validation_subjects=validation_subjects,
        )
        assert isinstance(loop.step, MotorImageryStep)
        self.offered.append(model)
        self.fitted.append(loop.step)
        predictions: list[Tensor] = []
        self.inferred.append(predictions)
        loop.step.model.register_forward_hook(
            functools.partial(_record_inference, predictions),
        )
        return loop


def _record_inference(
    predictions: list[Tensor],
    model: torch.nn.Module,
    args: tuple[object, ...],
    output: object,
) -> None:
    del args
    assert isinstance(output, Tensor)
    if not model.training:
        predictions.append(output.detach().clone())


@pytest.mark.parametrize("policy", ["copy", "ema_module", "ema_params", "autocast"])
def test_campaign_exports_and_predicts_the_complete_fitted_inference_state(
    tmp_path: Path,
    policy: str,
) -> None:
    cfg = ObservedCampaign.Config()
    cfg.update(synthetic_campaign(tmp_path))
    assert isinstance(cfg.validation.dataset.preprocessing, SyntheticInput.Config)
    cfg.validation.dataset.preprocessing.training_trial_count = 2
    cfg.validation.dataset.preprocessing.windows_per_trial = 5
    cfg.validation.dataset.preprocessing.testing_windows_per_trial = 5
    cfg.validation.dataset.preprocessing.window_size = 32
    cfg.diagnostic_batch_size = 5
    shallow = next(
        model for model in cfg.ensemble if isinstance(model, ShallowConvNet.Config)
    )
    assert isinstance(shallow.conv_t, Conv2d.Config)
    assert isinstance(shallow.pool, AvgPool2d.Config)
    shallow.conv_t.kernel_size = (1, 3)
    shallow.pool.kernel_size = (1, 16)
    validation, ensemble = MotorImageryLoop.Config(), MotorImageryLoop.Config()
    validation.update(cfg.validation, skip_missing=True)
    ensemble.update(cfg.ensemble_loop, skip_missing=True)
    cfg.validation, cfg.ensemble_loop = validation, ensemble
    for loop in (validation, ensemble):
        loop.max_epochs = 1
        loop.step.train_budget_epochs = loop.max_epochs
        assert isinstance(loop.step.lr_schedule, RecursiveCosine.Config)
        loop.step.lr_schedule.T_max = loop.max_epochs
        loop.dataset.batch_size = 5
        loop.early_train_log_steps = 0
        if policy == "copy":
            loop.step.model_quantization = CopyPreparation.Config()
        elif policy == "autocast":
            loop.step.dtype_autocast = torch.bfloat16
        else:
            average = loop.step.ema = EMA.Config()
            average.decay = 0.5
            if policy == "ema_params":
                average.shadow_kind = "param_dict"
    runner = cfg.make()
    finalized_shallow = next(
        model
        for model in runner.config.ensemble
        if isinstance(model, ShallowConvNet.Config)
    )
    assert runner.config.validation.dataset.preprocessing.window_size == 32
    assert isinstance(finalized_shallow.conv_t, Conv2d.Config)
    assert finalized_shallow.conv_t.kernel_size == (1, 3)
    assert isinstance(finalized_shallow.pool, AvgPool2d.Config)
    assert finalized_shallow.pool.kernel_size == (1, 16)
    assert finalized_shallow.pool.stride == (1, 15)
    with portable_half_precision():
        runner.run()
    assert len(runner.fitted) == len(cfg.ensemble) == 3
    if policy == "copy":
        for offered, fitted in zip(runner.offered, runner.fitted, strict=True):
            assert fitted.model is not offered
            assert all(
                prepared.data_ptr() != source.data_ptr()
                for prepared, source in zip(
                    fitted.model.parameters(),
                    offered.parameters(),
                    strict=True,
                )
            )
    assert isinstance(cfg.validation.dataset.preprocessing, SyntheticInput.Config)
    testing = cfg.validation.dataset.preprocessing.make().read([], labelled=False)
    media = testing[0].windows[:, None].float()
    for index, (step, observed) in enumerate(
        zip(runner.fitted, runner.inferred, strict=True),
    ):
        if policy in ("ema_module", "ema_params"):
            assert isinstance(step.ema, EMA)
            assert step.ema.global_step == step.global_step == 2
            assert step.ema.local_step == 2
        state = from_plain(
            cast(
                object,
                torch.load(
                    Path(cfg.working_dir) / f"ensemble-{index}.pt",
                    weights_only=True,
                ),
            ),
            dict[str, Tensor],
        )
        with step.ema.apply_to(step.model):
            expected = step.model.state_dict()
            assert state.keys() == expected.keys()
            assert all(
                torch.equal(state[name], tensor) for name, tensor in expected.items()
            )
        assert len(observed) == 1
        prediction = observed[0].clone()
        # run() predicted with oneDNN off; its float32 convolutions round differently.
        with portable_half_precision():
            expected_prediction = step.call_eval(media)
        assert isinstance(expected_prediction, Tensor)
        assert torch.equal(prediction, expected_prediction)


@pytest.mark.parametrize("default", ["cpu", "meta"])
@pytest.mark.parametrize(
    "model_type",
    [MultiScaleEEGNet, ShallowConvNet, AttentionEEGNet],
)
def test_cpu_construction_preserves_complete_model_and_random_state(
    default: str,
    model_type: type[MultiScaleEEGNet | ShallowConvNet | AttentionEEGNet],
) -> None:
    config = model_type.Config(8, 2)
    config.channels_in = 3
    config.num_samples = 100
    config.channels_hidden = 2
    if isinstance(config, (MultiScaleEEGNet.Config, AttentionEEGNet.Config)):
        config.channels_projection = 2
    with (
        torch.device("cpu"),
        host_agnostic_numerics(),
        PinnedRandom() as pins,
    ):
        expected = model_record(config.make(), prefix="model")
        expected.update(random_record(pins))
        expected["rng"] = torch.get_rng_state()
    with (
        host_agnostic_numerics(),
        PinnedRandom() as pins,
    ):
        with torch.device(default):
            model = _build_cpu(config)
        actual = model_record(model, prefix="model")
        actual.update(random_record(pins))
        actual["rng"] = torch.get_rng_state()
    assert all(parameter.device.type == "cpu" for parameter in model.parameters())
    compare_record(expected, actual)


@pytest.mark.parametrize(
    "model_type",
    [MultiScaleEEGNet, ShallowConvNet, AttentionEEGNet],
)
def test_discarded_diagnostics_preserve_every_tensor_and_draw_without_graphs(
    model_type: type[MultiScaleEEGNet | ShallowConvNet | AttentionEEGNet],
) -> None:
    config = model_type.Config(8, 2)
    config.channels_in = 3
    config.num_samples = 100
    config.channels_hidden = 2
    if isinstance(config, (MultiScaleEEGNet.Config, AttentionEEGNet.Config)):
        config.channels_projection = 2
    before = torch.get_rng_state()
    with torch.enable_grad(), NativeDraws() as source:
        model = config.make()
        # The original Conv2d requires a singleton input plane.
        media = torch.randn(4, 1, 3, 100)
        expected_output = model(media)
    assert expected_output.requires_grad
    expected = model_record(model, prefix="model")
    expected["output"] = expected_output.detach().clone()
    expected.update(source.outputs)
    expected_states = [*source.draws, ("final", get_rng_state())]
    modules: list[torch.nn.Module] = []
    outputs: list[Tensor] = []

    def remember(
        model: torch.nn.Module,
        args: tuple[object, ...],
        output: object,
    ) -> None:
        del model, args
        assert isinstance(output, Tensor)
        outputs.append(output)

    def construct() -> torch.nn.Module:
        modules.append(model := config.make())
        model.register_forward_hook(remember)
        return model

    torch.set_rng_state(before)
    with torch.enable_grad(), NativeDraws() as native:
        _diagnostic_draws(
            [InlineConfig(construct)],
            batch_size=4,
            channels=3,
            num_samples=100,
        )
    actual = model_record(modules[0], prefix="model")
    actual["output"] = outputs[0].detach().clone()
    actual.update(native.outputs)
    actual_states = [*native.draws, ("final", get_rng_state())]
    for index, ((site, want), (observed, got)) in enumerate(
        zip(expected_states, actual_states, strict=True),
    ):
        assert site == observed
        assert want.keys() == got.keys()
        for key in ("python", "numpy", "cuda_uuids"):
            assert want.get(key) == got.get(key)
        expected[f"rng/{index}/cpu"] = want["torch"]
        actual[f"rng/{index}/cpu"] = got["torch"]
        for device, (left, right) in enumerate(
            zip(want.get("cuda", []), got.get("cuda", []), strict=True),
        ):
            expected[f"rng/{index}/cuda/{device}"] = left
            actual[f"rng/{index}/cuda/{device}"] = right
    compare_record(expected, actual)
    assert not outputs[0].requires_grad
    assert all(parameter.requires_grad for parameter in modules[0].parameters())


@pytest.mark.parametrize("default", ["cpu", "meta"])
def test_diagnostic_inputs_share_the_models_cpu_device(default: str) -> None:
    cfg = MultiScaleEEGNet.Config(3, 2)
    cfg.num_samples = 100
    cfg.channels_hidden = cfg.channels_projection = 2
    with torch.device(default):
        _diagnostic_draws([cfg], batch_size=4, channels=3, num_samples=100)


@pytest.mark.parametrize("failure", [ValueError("read failure"), KeyboardInterrupt()])
def test_campaign_persists_catchable_ingestion_failure_and_releases_runtime(
    tmp_path: Path,
    failure: BaseException,
) -> None:
    cfg = synthetic_campaign(tmp_path)
    assert isinstance(cfg.validation.dataset.preprocessing, SyntheticInput.Config)
    cfg.validation.dataset.preprocessing.failure = failure
    with pytest.raises(type(failure)):
        cfg.make().run()
    result = json_loads(
        (Path(cfg.working_dir) / "result.json").read_text(),
        dict[str, object],
    )
    assert result["status"] == "failed"
    assert result["error"] == f"{type(failure).__name__}: {failure}"
    metadata = from_plain(result["metadata"], dict[str, object])
    assert metadata["experiment_name"] == "exp_smoke"
    assert from_plain(result["elapsed_seconds"], float) > 0
    assert not runtime_initialized()


@pytest.mark.parametrize("fail_fit", [False, True])
def test_campaign_preserves_caller_owned_runtime_and_settings(
    tmp_path: Path,
    fail_fit: bool,
) -> None:
    cfg = synthetic_campaign(tmp_path)
    cfg.folds = FixedCohort.Config()
    assert isinstance(cfg.folds, FixedCohort.Config)
    cfg.folds.subjects = ["S001"]
    assert isinstance(cfg.validation, CountingLoop.Config)
    cfg.validation.fail_fit = fail_fit
    outer_config = cfg.validation.runtime = SingleProcess.Config(device="cpu")
    outer_config.float32_matmul_precision = "medium"
    cfg.ensemble_loop.runtime = outer_config.copy_tree()
    outer = outer_config.make()
    outer.initialize()
    try:
        if fail_fit:
            with pytest.raises(ValueError, match="fit failure"):
                cfg.make().run()
        else:
            cfg.make().run()
        assert runtime_initialized()
        assert torch.get_float32_matmul_precision() == "medium"
        result = json_loads(
            (Path(cfg.working_dir) / "result.json").read_text(),
            dict[str, object],
        )
        assert result["status"] == ("failed" if fail_fit else "complete")
    finally:
        outer.destroy()


def test_finite_outer_budget_interrupts_preprocessing_and_writes_failure(
    tmp_path: Path,
) -> None:
    cfg = synthetic_campaign(tmp_path)
    cfg.budget.seconds = 1
    assert isinstance(cfg.validation.dataset.preprocessing, SyntheticInput.Config)
    cfg.validation.dataset.preprocessing.alarm = True
    handler = signal.getsignal(signal.SIGALRM)
    timer = signal.getitimer(signal.ITIMER_REAL)
    with pytest.raises(TimeoutError, match="Outer wall budget exceeded"):
        cfg.make().run()
    result = json_loads(
        (Path(cfg.working_dir) / "result.json").read_text(),
        dict[str, object],
    )
    assert result["status"] == "failed"
    assert result["error"] == "TimeoutError: Outer wall budget exceeded."
    assert signal.getsignal(signal.SIGALRM) == handler
    assert signal.getitimer(signal.ITIMER_REAL)[0] <= timer[0]
    assert signal.getitimer(signal.ITIMER_REAL)[1] == timer[1]
    assert not runtime_initialized()


@pytest.mark.parametrize("budget", [0.0, float("nan")])
def test_campaign_rejects_invalid_budget(budget: float) -> None:
    cfg = exp000()
    cfg.budget.seconds = budget
    with pytest.raises(ValueError, match="Wall budget seconds"):
        cfg.make().run()


def test_result_publication_cannot_escape_the_campaign_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = synthetic_campaign(tmp_path)
    cfg.folds = FixedCohort.Config()
    assert isinstance(cfg.folds, FixedCohort.Config)
    cfg.folds.subjects = ["S001"]
    cfg.budget.seconds = 0.5
    clock = _ReceiptClock()
    cfg.budget.clock = clock.perf_counter
    original_replace = Path.replace

    def delayed_replace(path: Path, target: Path | str) -> Path:
        if Path(target).name == "result.json" and clock.elapsed == 0:
            clock.elapsed = 0.6
        return original_replace(path, target)

    monkeypatch.setattr(Path, name="replace", value=delayed_replace)
    with pytest.raises(TimeoutError, match="Outer wall budget exceeded"):
        cfg.make().run()
    result = json_loads(
        (Path(cfg.working_dir) / "result.json").read_text(),
        target=dict[str, object],
    )
    assert result["status"] == "failed"
    assert result["error"] == "TimeoutError: Outer wall budget exceeded."
    assert result["elapsed_seconds"] == 0.6


def test_campaign_requires_both_loops_to_share_one_runtime() -> None:
    cfg = exp000()
    cfg.validation.runtime = SingleProcess.Config(device="cpu")
    with pytest.raises(ValueError, match="share a runtime"):
        cfg.make()


def test_campaign_rejects_test_people_who_are_training_people(tmp_path: Path) -> None:
    cfg = synthetic_campaign(tmp_path)
    assert isinstance(cfg.validation.dataset.preprocessing, SyntheticInput.Config)
    cfg.validation.dataset.preprocessing.testing_subject = "S001"
    with pytest.raises(ValueError, match="overlap"):
        cfg.make().run()


def test_campaign_rejects_missing_data_and_passthrough_arguments(
    tmp_path: Path,
) -> None:
    cfg = exp000()
    cfg.validation.base_dir = None
    cfg.validation.dataset.working_dir = tmp_path / "missing"
    campaign = cfg.make()
    with pytest.raises(FileNotFoundError, match="Raw EEG competition data missing"):
        campaign.run()
    with pytest.raises(ValueError, match="Unexpected campaign arguments"):
        campaign.run("--unknown")


def test_actual_priml_loop_matches_the_step_golden(
    tmp_path: Path,
) -> None:
    cfg = exp_smoke()
    # The full-cohort loop never evaluates, so its draws match the train/ golden's.
    cfg.validation.runtime = SingleProcess.Config(device="cpu")
    cfg.ensemble_loop.runtime = SingleProcess.Config(device="cpu")
    cfg.ensemble_loop.base_dir = None
    cfg.ensemble_loop.working_dir = tmp_path
    cfg.ensemble_loop.early_train_log_steps = 0
    cfg.ensemble_loop.dataset.batch_size = 4
    model_cfg = MultiScaleEEGNet.Config(8, 2)
    model_cfg.channels_in = 3
    model_cfg.num_samples = 100
    model_cfg.channels_hidden = model_cfg.channels_projection = 2
    with (
        host_agnostic_numerics(),
        PinnedRandom() as pins,
    ):
        model = model_cfg.make()
        # Every network's Conv2d requires a singleton input plane; labels are binary.
        media = torch.arange(4 * 3 * 100).reshape(4, 3, 100).float() / 100
        trials = [
            Trial(
                subject="S002",
                trial_id=index,
                label=index % 2,
                windows=media[index : index + 1],
            )
            for index in range(4)
        ]
        runner = cfg.make()
        loop = runner._make_loop(
            model,
            config=runner.config.ensemble_loop,
            trials=trials,
            validation_subjects=[],
        )
        loop.train()
        actual = model_record(model, prefix="final")
        assert isinstance(loop.step, MotorImageryStep)
        actual.update(
            optimizer_record(cast(torch.optim.Optimizer, loop.step.optimizer)),
        )
        actual.update(random_record(pins))
    record = read_tensors(_CWD / "testdata" / "multiscaleeegnet.pt")
    expected = {
        key.removeprefix("train/"): value
        for key, value in record.items()
        if key.startswith(("train/final/", "train/optimizer/", "train/random/"))
    }
    compare_record(expected, actual)
    assert loop.step.global_step == 3
    assert loop.dataset.timer_epoch.global_count == 3


def test_loop_construction_restores_the_pending_first_use_cuda_seed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = synthetic_campaign(tmp_path)
    assert isinstance(cfg.validation, CountingLoop.Config)
    cfg.validation.reseed = True
    cuda_seeds: list[int] = []
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", cuda_seeds.append)
    model = torch.nn.Linear(3, 2)
    torch.manual_seed(42)
    before = torch.get_rng_state()
    runner = cfg.make()
    loop = runner._make_loop(
        model,
        config=runner.config.validation,
        trials=[],
        validation_subjects=[],
    )
    assert torch.equal(torch.get_rng_state(), before)
    assert cuda_seeds[-1] == 42
    assert isinstance(loop, CountingLoop)
    assert loop.inputs.checkpointer is None
    assert loop.inputs.tracker is None
    assert set(loop.inputs.metrics_eval) == {""}
    assert isinstance(loop.inputs.metrics_eval[""], EEGAccuracy.Config)


def test_loop_restores_initialized_cuda_state_without_reseeding_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = synthetic_campaign(tmp_path)
    state = get_rng_state()
    state["cuda"] = [torch.arange(4, dtype=torch.uint8)]
    restored: list[RngState] = []
    cuda_seeds: list[int] = []
    monkeypatch.setattr(campaign, name="get_rng_state", value=lambda: state)
    monkeypatch.setattr(campaign, name="set_rng_state", value=restored.append)
    monkeypatch.setattr(torch.cuda, name="manual_seed_all", value=cuda_seeds.append)
    runner = cfg.make()
    runner._make_loop(
        torch.nn.Linear(3, 2),
        config=runner.config.validation,
        trials=[],
        validation_subjects=[],
    )
    assert restored == [state]
    assert cuda_seeds == []


class LogitModel(torch.nn.Module):
    """Pass supplied logits through a parameterized Priml model."""

    class Config(Fig["LogitModel"]):
        """Construct the deterministic loss-reduction fixture."""

    def __init__(self, config: Config) -> None:
        super().__init__()
        del config
        self.bias = torch.nn.Parameter(torch.zeros(2))

    @override
    def forward(self, media: Tensor) -> Tensor:
        return media.reshape(len(media), self.bias.numel(), -1).mean(-1) + self.bias


def test_priml_eval_weights_a_short_final_batch_by_actual_window_count(
    tmp_path: Path,
) -> None:
    cfg = exp_smoke()
    cfg.validation.base_dir = None
    cfg.validation.working_dir = tmp_path
    cfg.validation.runtime = SingleProcess.Config(device="cpu")
    cfg.ensemble_loop.runtime = SingleProcess.Config(device="cpu")
    cfg.validation.dataset.batch_size = 4
    logits = torch.tensor(
        [
            [2.0, 0.0],
            [1.0, 0.0],
            [0.0, 2.0],
            [0.0, 3.0],
            [0.0, -1.0],
            [1.0, -2.0],
            [-1.0, 1.0],
        ],
    )
    windows = logits[:, :, None].expand(-1, -1, 5)
    trials = [
        Trial(subject="S001", trial_id=0, label=0, windows=torch.zeros(4, 2, 5)),
        Trial(subject="S002", trial_id=1, label=0, windows=windows),
    ]
    runner = cfg.make()
    loop = runner._make_loop(
        LogitModel.Config().make(),
        config=runner.config.validation,
        trials=trials,
        validation_subjects=["S002"],
    )
    try:
        first = loop.step.eval_loss(media=windows[:4], label=torch.zeros(4).long())
        last = loop.step.eval_loss(media=windows[4:], label=torch.zeros(3).long())
        first_prediction, last_prediction = first["model"], last["model"]
        assert torch.equal(first_prediction, logits[:4])
        assert torch.equal(last_prediction, logits[4:])
        expected = (
            first["loss"].mean().item() * 4 + last["loss"].mean().item() * 3
        ) / len(windows)
        assert loop.eval()["total_loss"] == expected
        assert (
            expected != (first["loss"].mean().item() + last["loss"].mean().item()) / 2
        )
    finally:
        loop.close()


def test_fixed_validation_people_are_the_only_fold_and_remain_disjoint(
    tmp_path: Path,
) -> None:
    cfg = synthetic_campaign(tmp_path)
    cohort = cfg.folds = FixedCohort.Config()
    cohort.subjects = ["S002", "S004"]
    cfg.make().run()
    state = cast(
        "dict[str, Tensor]",
        torch.load(
            Path(cfg.working_dir) / "fold-0.pt",
            weights_only=True,
        ),
    )
    assert set(state["best_trials"].tolist()) == {1, 3}
    assert {path.name for path in Path(cfg.working_dir).glob("fold-*.pt")} == {
        "fold-0.pt",
    }


def test_ensemble_probability_matches_every_golden_model_and_window() -> None:
    steps: list[MotorImageryStep] = []
    probabilities: list[Tensor] = []
    expected_logits: dict[str, Tensor] = {}
    for model_type in (MultiScaleEEGNet, ShallowConvNet, AttentionEEGNet):
        record = read_tensors(_CWD / "testdata" / f"{model_type.__name__.lower()}.pt")
        cfg = model_type.Config(8, 2)
        cfg.channels_in = 3
        cfg.num_samples = 100
        cfg.channels_hidden = 2
        if isinstance(cfg, (MultiScaleEEGNet.Config, AttentionEEGNet.Config)):
            cfg.channels_projection = 2
        model = cfg.make()
        model.load_state_dict(
            {
                key.removeprefix("train/final/"): value
                for key, value in record.items()
                if key.startswith("train/final/")
            },
        )
        steps.append(
            MotorImageryStep(source_step(cfg, epochs=1).finalized(), model=model),
        )
        probabilities.append(record["train/probabilities"])
        expected_logits[model_type.__name__] = record["train/eval_logits"]
    media = torch.arange(4 * 3 * 100).reshape(4, 3, 100).float() / 100
    trial = Trial(subject="S008", trial_id=0, label=-1, windows=media)
    with host_agnostic_numerics():
        outputs = {
            name: cast(Tensor, step.call_eval(media[:, None]))
            for name, step in zip(expected_logits, steps, strict=True)
        }
        compare_record(expected_logits, outputs)
        compare_record(
            dict(zip(expected_logits, probabilities, strict=True)),
            {
                name: output.to(dtype=torch.float32).softmax(dim=1)[:, 1]
                for name, output in outputs.items()
            },
        )
        actual = predict(steps, trials=[trial], device=torch.device("cpu"))
    assert actual == [
        float(np.mean([probability.numpy() for probability in probabilities])),
    ]


@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.bfloat16, torch.float32, torch.float64],
)
@pytest.mark.parametrize(
    "model_name",
    ["multiscaleeegnet", "shallowconvnet", "attentioneegnet"],
)
def test_float32_softmax_cast_retains_native_complete_logit_probabilities(
    dtype: torch.dtype,
    model_name: str,
) -> None:
    record = read_tensors(_CWD / "testdata" / f"{model_name}.pt")
    with torch.inference_mode():
        logits = record["train/eval_logits"].to(dtype=dtype)
        expected = logits.softmax(dim=1, dtype=torch.float32)
        actual = logits.to(dtype=torch.float32).softmax(dim=1)
    compare_record({"probabilities": expected}, {"probabilities": actual})


class _ReceiptClock:
    elapsed: float = 0.0

    def perf_counter(self) -> float:
        return self.elapsed

    def time(self) -> float:
        return 0.0


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
