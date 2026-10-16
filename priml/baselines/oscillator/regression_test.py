"""Pin oscillator initialization and short training runs."""

from pathlib import Path
from typing import TYPE_CHECKING

from configgle import PartialConfig
from torch import Tensor

import pytest
import torch

from priml.baselines.etth1.testing import update_record
from priml.baselines.oscillator.benchmark import ForecastStudy
from priml.baselines.oscillator.experiments import exp_smoke
from priml.baselines.oscillator.train_step import DLinearForecast
from priml.testing.bfb import host_agnostic_numerics
from priml.testing.golden import (
    assert_tensor_golden,
    expect_golden_mismatch,
    rng_fingerprint,
)
from priml.train.custom_types import TrainStepOutput


if TYPE_CHECKING:
    from collections.abc import Callable


def test_model_matches_golden() -> None:
    actual = _model_record()
    path = Path(__file__).parent / "testdata" / "dlinear_model.pt"
    assert_tensor_golden(path, actual)
    assert path.stat().st_size <= 32768


def test_model_golden_detects_one_changed_bit() -> None:
    actual = _model_record()
    # Change one float32 ULP before comparing the record.
    actual["output"] = actual["output"].contiguous()
    actual["output"].view(torch.int32).flatten()[0] += 1
    with expect_golden_mismatch("output"):
        assert_tensor_golden(
            Path(__file__).parent / "testdata" / "dlinear_model.pt",
            actual,
        )


@pytest.mark.compute_training
@pytest.mark.parametrize("pretrain_updates", [0, 1])
def test_fit_matches_golden(
    pretrain_updates: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = exp_smoke()
    assert isinstance(cfg.method, DLinearForecast.Config)
    cfg.method.pretrain_updates = pretrain_updates
    actual = _fit_record(cfg, monkeypatch=monkeypatch)
    path = Path(__file__).parent / "testdata" / f"dlinear_fit{pretrain_updates}.pt"
    assert_tensor_golden(path, actual)
    assert path.stat().st_size <= 32768


@pytest.mark.compute_training
def test_fit_golden_detects_changed_recipe(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = exp_smoke()
    assert isinstance(cfg.method, DLinearForecast.Config)
    cfg.method.step.optimizer = PartialConfig(torch.optim.Adam, lr=2e-3)
    actual = _fit_record(cfg, monkeypatch=monkeypatch)
    with expect_golden_mismatch("final/parameter/"):
        assert_tensor_golden(
            Path(__file__).parent / "testdata" / "dlinear_fit0.pt",
            actual,
        )


def _model_record() -> dict[str, Tensor]:
    with torch.random.fork_rng(devices=[]), host_agnostic_numerics():
        cfg = exp_smoke().finalize()
        torch.manual_seed(cfg.seed)
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


def _fit_record(
    cfg: ForecastStudy.Config,
    *,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Tensor]:
    with torch.random.fork_rng(devices=[]), host_agnostic_numerics():
        cfg = cfg.finalize()
        torch.manual_seed(cfg.seed)
        assert isinstance(cfg.method, DLinearForecast.Config)
        method = cfg.method.make()
        data = cfg.data.make()
        record = {
            f"initial/{key}": value.detach().clone()
            for key, value in method.step.model.state_dict().items()
        }
        losses: list[Tensor] = []
        outputs: list[TrainStepOutput] = []
        train_step: Callable[..., TrainStepOutput] = method.step.train_step

        def record_update(**batch: object) -> TrainStepOutput:
            output = train_step(**batch)
            outputs.append(output)
            losses.append(output["loss"].clone())
            return output

        monkeypatch.setattr(method.step, "train_step", record_update)
        report = method.fit(
            data.windows(cfg.data.train, simulator=data.reference),
            validation=data.windows(cfg.data.validation, simulator=data.reference),
            synthetic=data.windows(cfg.data.synthetic, simulator=data.cheap),
            simulator=data.cheap,
        )
        last_output = outputs[-1]
        final = update_record(
            method.step.model,
            optimizer=method.step.optimizer,
            output=last_output["model"],
            loss=last_output["loss"],
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
        # DLinearForecast.fit keeps singleton axes for channel scaling.
        record.update(mean=method.mean.clone(), scale=method.scale.clone())
        record["losses"] = torch.cat(losses)
        record.update(
            {f"report/{key}": torch.tensor(value) for key, value in report.items()},
        )
        return {
            key: value.float() if value.is_floating_point() else value
            for key, value in record.items()
        }
