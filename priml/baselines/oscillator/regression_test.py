"""Pin oscillator initialization and short training runs."""

from collections.abc import Callable
from pathlib import Path
from typing import Final

from configgle import PartialConfig
from torch import Tensor

import pytest
import torch

from priml.baselines.etth1.testing import update_record
from priml.baselines.oscillator.benchmark import ForecastStudy
from priml.baselines.oscillator.experiments import exp_smoke
from priml.baselines.oscillator.nonlinear import HistoryNetwork, NeuralForecast
from priml.baselines.oscillator.train_step import DLinearForecast
from priml.model.init import unit_fan_in_uniform
from priml.model.swiglu import SwiGLU
from priml.testing.bfb import host_agnostic_numerics
from priml.testing.golden import (
    assert_tensor_golden,
    expect_golden_mismatch,
    rng_fingerprint,
)
from priml.train.custom_types import TrainStepOutput


_CWD: Final = Path(__file__).resolve().parent
MAX_GOLDEN_BYTES: Final = 32 * 1024
GOLDEN_SEED: Final = 2021


class _RecordedUpdates:
    """Wrap a train step, keeping each update's output."""

    def __init__(self, train_step: Callable[..., TrainStepOutput]) -> None:
        self.train_step = train_step
        self.outputs: list[TrainStepOutput] = []

    def __call__(self, **batch: object) -> TrainStepOutput:
        output = self.train_step(**batch)
        self.outputs.append(output)
        return output


def test_model_matches_golden() -> None:
    path = _CWD / "testdata" / "dlinear_model.pt"
    assert_tensor_golden(path, _model_record())
    assert path.stat().st_size <= MAX_GOLDEN_BYTES


def test_model_golden_detects_one_changed_bit() -> None:
    actual = _model_record()
    # Change one float32 ULP before comparing the record.
    actual["output"] = actual["output"].contiguous()
    actual["output"].view(torch.int32).flatten()[0] += 1
    with expect_golden_mismatch("output"):
        assert_tensor_golden(_CWD / "testdata" / "dlinear_model.pt", actual)


@pytest.mark.parametrize("pretrain_updates", [0, 1])
def test_fit_matches_golden(
    pretrain_updates: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = exp_smoke()
    assert isinstance(cfg.method, DLinearForecast.Config)
    cfg.method.pretrain_updates = pretrain_updates
    path = _CWD / "testdata" / f"dlinear_fit{pretrain_updates}.pt"
    assert_tensor_golden(path, _fit_record(cfg, monkeypatch=monkeypatch))
    assert path.stat().st_size <= MAX_GOLDEN_BYTES


def test_network_model_matches_golden() -> None:
    path = _CWD / "testdata" / "network_model.pt"
    assert_tensor_golden(path, _network_model_record())
    assert path.stat().st_size <= MAX_GOLDEN_BYTES


def test_network_fit_matches_golden(monkeypatch: pytest.MonkeyPatch) -> None:
    path = _CWD / "testdata" / "network_fit.pt"
    assert_tensor_golden(path, _fit_record(_network_config(), monkeypatch=monkeypatch))
    assert path.stat().st_size <= MAX_GOLDEN_BYTES


def test_fit_golden_detects_changed_recipe(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = exp_smoke()
    assert isinstance(cfg.method, DLinearForecast.Config)
    adam = cfg.method.step.optimizer
    assert isinstance(adam, PartialConfig)
    adam.lr = 2e-3
    actual = _fit_record(cfg, monkeypatch=monkeypatch)
    with expect_golden_mismatch("final/parameter/"):
        assert_tensor_golden(_CWD / "testdata" / "dlinear_fit0.pt", actual)


def _model_record() -> dict[str, Tensor]:
    with torch.random.fork_rng(devices=[]), host_agnostic_numerics():
        cfg = exp_smoke().finalize()
        torch.manual_seed(GOLDEN_SEED)
        assert isinstance(cfg.method, DLinearForecast.Config)
        method = cfg.method.make()
        data = cfg.data.make()
        train = data.windows(cfg.data.train, simulator=data.reference)
        with torch.no_grad():
            output = method.step.net(train.media)
        return {
            **{
                f"initial/{key}": value.detach().clone()
                for key, value in method.step.model.state_dict().items()
            },
            "output": output.detach().clone().float(),
            "rng": rng_fingerprint(),
        }


def _network_model_record() -> dict[str, Tensor]:
    with torch.random.fork_rng(devices=[]), host_agnostic_numerics():
        cfg = _network_config().finalize()
        torch.manual_seed(GOLDEN_SEED)
        assert isinstance(cfg.method, NeuralForecast.Config)
        method = cfg.method.make()
        data = cfg.data.make()
        train = data.windows(cfg.data.train, simulator=data.reference)
        assert isinstance(method.step.model, HistoryNetwork)
        with torch.no_grad():
            output = method.step.model(train.media)
        return {
            **{
                f"initial/{key}": value.detach().clone()
                for key, value in method.step.model.state_dict().items()
            },
            "output": output.detach().clone().float(),
            "rng": rng_fingerprint(),
        }


def _network_config() -> ForecastStudy.Config:
    cfg = exp_smoke()
    method = NeuralForecast.Config()
    method.updates, method.batch_size, method.validation_every = 2, 4, 1
    assert isinstance(method.step.model.block, SwiGLU.Config)
    method.step.model.block.channels_hidden = 6
    # The default zero output projection would make the pinned forward all zeros.
    method.step.model.block.init_weight_out = unit_fan_in_uniform
    cfg.method = method
    return cfg


def _fit_record(
    cfg: ForecastStudy.Config,
    *,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Tensor]:
    with torch.random.fork_rng(devices=[]), host_agnostic_numerics():
        cfg = cfg.finalize()
        torch.manual_seed(GOLDEN_SEED)
        assert isinstance(cfg.method, DLinearForecast.Config | NeuralForecast.Config)
        method = cfg.method.make()
        data = cfg.data.make()
        record = {
            f"initial/{key}": value.detach().clone()
            for key, value in method.step.model.state_dict().items()
        }
        recorded = _RecordedUpdates(method.step.train_step)
        monkeypatch.setattr(method.step, "train_step", recorded)
        report = method.fit(
            data.windows(cfg.data.train, simulator=data.reference),
            validation=data.windows(cfg.data.validation, simulator=data.reference),
            synthetic=data.windows(cfg.data.synthetic, simulator=data.cheap),
            simulator=data.cheap,
        )
        last_output = recorded.outputs[-1]
        final = update_record(
            method.step.model,
            optimizer=method.step.optimizer,
            output=last_output["model"],
            loss=last_output["loss"].mean(),
        )
        record.update(
            {
                f"final/{key}": value
                for key, value in final.items()
                if key == "rng" or key.startswith(("parameter/", "optimizer/"))
            },
        )
        assert method.mean is not None
        assert method.scale is not None
        # Both trained forecasters keep singleton axes for channel scaling.
        record.update(mean=method.mean.clone(), scale=method.scale.clone())
        # The network's SimpleLoss keeps per-cell losses; one mean per update.
        record["losses"] = torch.stack(
            [output["loss"].mean() for output in recorded.outputs],
        )
        record.update(
            {f"report/{key}": torch.tensor(value) for key, value in report.items()},
        )
        return {
            key: value.float() if value.is_floating_point() else value
            for key, value in record.items()
        }


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
