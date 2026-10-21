"""Check offline weight verification and zero-shot input isolation."""

from __future__ import annotations

from pathlib import Path
from pkgutil import resolve_name

import hashlib
import json

from configgle import Fig, InlineConfig
from torch import Tensor
from treekle import json_loads, to_plain

import pytest
import torch

from priml.baselines.oscillator.data import Oscillator, Windows
from priml.baselines.oscillator.pretrained import (
    AssetManifest,
    BoltPipeline,
    Chronos2Pipeline,
    ModelFiles,
    PretrainedForecast,
)


@pytest.mark.parametrize("bad", ["identity", "revision", "files", "digest"])
def test_staged_model_files_are_pinned_and_checked(tmp_path: Path, bad: str) -> None:
    cfg = _assets(tmp_path)
    assert cfg.make().directory == tmp_path
    path = tmp_path / "source.json"
    record = json_loads(path.read_text(), AssetManifest)
    if bad == "identity":
        record["model_id"] = "other/model"
    elif bad == "revision":
        record["revision"] = "0" * 40
    elif bad == "files":
        record["files"]["../outside"] = "bad"
    else:
        (tmp_path / "model.safetensors").write_text("changed")
    path.write_text(json.dumps(to_plain(record)))
    with pytest.raises(ValueError, match=r"identity|manifest|digest"):
        cfg.make()


def test_lazy_pipeline_factory_resolves_instead_of_returning_a_partial() -> None:
    for cfg in (BoltPipeline.Config(), Chronos2Pipeline.Config()):
        assert isinstance(cfg.factory, InlineConfig)
    lazy = InlineConfig(resolve_name, f"{__name__}:_Factory")
    assert lazy.make() is _Factory


def test_pretrained_assets_default_to_the_shared_download_cache() -> None:
    assert ModelFiles.Config().directory == Path(
        "/opt/scratch/caches/chronos/chronos-bolt-tiny",
    )
    assert Chronos2Pipeline.Config().files.directory == Path(
        "/opt/scratch/caches/chronos/chronos-2",
    )


@pytest.mark.parametrize("multivariate", [False, True])
def test_pipelines_keep_channel_order_and_never_enable_cross_trajectory_learning(
    tmp_path: Path,
    multivariate: bool,
) -> None:
    assets = _assets(tmp_path)
    if multivariate:
        cfg = Chronos2Pipeline.Config()
        cfg.factory = _Factory.Config(multivariate=True)
    else:
        cfg = BoltPipeline.Config()
        cfg.factory = _Factory.Config()
    cfg.files = assets
    cfg.batch_size = 4
    pipeline = cfg.make()
    media = torch.arange(3 * 7 * 2).reshape(3, 7, 2).double()
    result = pipeline(media, horizon=5)
    torch.testing.assert_close(result, media[:, -1:].expand(3, 5, 2))
    assert result.dtype == media.dtype
    assert isinstance(pipeline.pipeline, _Factory)
    assert pipeline.pipeline.calls == (1 if multivariate else 2)
    if multivariate:
        assert pipeline.pipeline.cross_learning is False
    cfg.batch_size = 0
    with pytest.raises(ValueError, match="batch"):
        cfg.make()


def test_upstream_shape_contracts_are_enforced(tmp_path: Path) -> None:
    cfg = BoltPipeline.Config()
    cfg.files = _assets(tmp_path)
    cfg.factory = _Factory.Config(multivariate=True)
    with pytest.raises(TypeError, match="tensor"):
        cfg.make()(torch.zeros(3, 7, 2), horizon=5)
    other = Chronos2Pipeline.Config()
    other.files = cfg.files
    other.factory = _Factory.Config()
    with pytest.raises(TypeError, match="trajectory"):
        other.make()(torch.zeros(3, 7, 2), horizon=5)


def test_zero_shot_fit_ignores_target_values_and_rejects_bad_forecasts() -> None:
    cfg = PretrainedForecast.Config()
    cfg.pipeline = _PointForecast.Config()
    method = cfg.make()
    media = torch.arange(3 * 7 * 2).reshape(3, 7, 2).float()
    with pytest.raises(RuntimeError, match="geometry"):
        method.predict(media)
    train = Windows(media=media, label=torch.full((3, 5, 2), float("nan")))
    report = method.fit(
        train,
        validation=train,
        synthetic=train,
        simulator=Oscillator.Config().make(),
    )
    assert report["reference_training_used"] == report["reference_validation_used"] == 0
    assert report["updates"] == 0
    torch.testing.assert_close(method.predict(media), media[:, -1:].expand(3, 5, 2))
    cfg.pipeline = _PointForecast.Config(nonfinite=True)
    bad = cfg.make()
    bad.fit(
        train,
        validation=train,
        synthetic=train,
        simulator=Oscillator.Config().make(),
    )
    assert torch.isnan(bad.predict(media)).all()
    cfg.pipeline = _PointForecast.Config(wrong_shape=True)
    bad = cfg.make()
    bad.fit(
        train,
        validation=train,
        synthetic=train,
        simulator=Oscillator.Config().make(),
    )
    with pytest.raises(ValueError, match="geometry"):
        bad.predict(media)


def _assets(path: Path) -> ModelFiles.Config:
    cfg = ModelFiles.Config()
    cfg.directory = path
    files = {"config.json": "{}", "model.safetensors": "test weights"}
    for name, content in files.items():
        (path / name).write_text(content)
    record = {
        "model_id": cfg.model_id,
        "revision": cfg.revision,
        "files": {
            name: hashlib.sha256(content.encode()).hexdigest()
            for name, content in files.items()
        },
    }
    (path / "source.json").write_text(json.dumps(to_plain(record)))
    return cfg


class _Factory:
    class Config(Fig["_Factory"]):
        multivariate: bool = False
        """Return per-trajectory groups instead of a batched point tensor."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.calls = 0
        self.cross_learning: object = None

    def from_pretrained(self, path: str, **kwargs: object) -> _Factory:
        assert Path(path).is_dir()
        assert kwargs["local_files_only"] is True
        assert kwargs["device_map"] == "cpu"
        assert kwargs["dtype"] == torch.float32
        return self

    def predict_quantiles(
        self,
        inputs: Tensor,
        *,
        prediction_length: int,
        quantile_levels: list[float],
        **kwargs: object,
    ) -> tuple[Tensor | list[Tensor], Tensor | list[Tensor]]:
        assert quantile_levels == [0.5]
        self.calls += 1
        self.cross_learning = kwargs.get("cross_learning")
        point = inputs[..., -1:].expand(*inputs.shape[:-1], prediction_length)
        if self.config.multivariate:
            return list(point), list(point)
        return point, point


class _PointForecast:
    class Config(Fig["_PointForecast"]):
        nonfinite: bool = False
        """Return nonfinite point forecasts to exercise evaluator failure handling."""

        wrong_shape: bool = False
        """Return the observed geometry instead of the requested forecast geometry."""

    def __init__(self, config: Config) -> None:
        self.config = config

    def __call__(self, media: Tensor, *, horizon: int) -> Tensor:
        if self.config.wrong_shape:
            return media
        result = media[:, -1:].expand(len(media), horizon, 2)
        return result * float("nan") if self.config.nonfinite else result


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
