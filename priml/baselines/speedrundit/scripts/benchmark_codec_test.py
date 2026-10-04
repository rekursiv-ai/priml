"""Tests for the codec benchmark's metrics and candidate table."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import math

import pytest
import torch

from priml.baselines.speedrundit.corpus import CorpusMismatchError
from priml.baselines.speedrundit.latent_codec import FloatCodec, ScalarTableCodec
from priml.baselines.speedrundit.scripts import benchmark_codec, prepare_data
from priml.baselines.speedrundit.scripts.prepare_data_test import _imagenet, _source
from priml.model.vision_ae.latent_norm import ScaleLatents


if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from torch import Tensor


def test_identical_latents_score_zero_error() -> None:
    latents = _latents(4, seed=0)
    metrics = benchmark_codec.latent_metrics(
        latents,
        decoded=latents.clone(),
        normalizer=ScaleLatents.Config().make(),
    )
    assert metrics["nmse"] == 0.0
    assert metrics["snr_db"] == math.inf
    assert metrics["max_abs_error"] == 0.0


def test_error_is_measured_after_normalization() -> None:
    """A normalizer that doubles latents doubles the error it reports."""
    latents = _latents(4, seed=0)
    noisy = latents + 0.01
    plain = benchmark_codec.latent_metrics(
        latents,
        decoded=noisy,
        normalizer=ScaleLatents.Config().make(),
    )
    doubled = benchmark_codec.latent_metrics(
        latents,
        decoded=noisy,
        normalizer=ScaleLatents.Config(scale=2.0).make(),
    )
    assert doubled["max_abs_error"] == pytest.approx(2 * plain["max_abs_error"])
    assert doubled["nmse"] == pytest.approx(plain["nmse"])


def test_psnr_of_a_uniform_offset() -> None:
    reference = torch.zeros(1, 3, 4, 4)
    assert benchmark_codec.psnr(reference, other=reference + 0.1) == pytest.approx(20.0)
    assert benchmark_codec.psnr(reference, other=reference) == math.inf


def test_evaluate_fits_on_one_sample_and_scores_on_the_other() -> None:
    report = benchmark_codec.evaluate(
        {
            "float16": FloatCodec.Config(dtype=torch.float16),
            "lloyd": ScalarTableCodec.Config(),
        },
        fit_sample=_latents(32, seed=0),
        eval_sample=_latents(8, seed=1),
        normalizer=ScaleLatents.Config().make(),
    )
    assert report["float16"]["bits_per_scalar"] == 16
    assert report["lloyd"]["bits_per_scalar"] == 8
    assert "index_entropy_bits" in report["lloyd"]
    assert report["float16"]["snr_db"] > report["lloyd"]["snr_db"]


def test_evaluate_scores_images_when_given_a_decoder() -> None:
    report = benchmark_codec.evaluate(
        {"float32": FloatCodec.Config()},
        fit_sample=_latents(8, seed=0),
        eval_sample=_latents(4, seed=1),
        normalizer=ScaleLatents.Config().make(),
        decode=lambda latents: latents.sigmoid(),
    )
    assert report["float32"]["image_psnr_db"] == math.inf


def test_fit_stability_reports_each_size() -> None:
    curve = benchmark_codec.fit_stability(
        _latents(64, seed=0),
        eval_sample=_latents(8, seed=1),
        normalizer=ScaleLatents.Config().make(),
        sizes=[16, 64],
    )
    assert set(curve) == {16, 64}
    assert all(value > 0 for value in curve.values())


def test_unequal_fit_and_eval_requests_get_the_requested_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _source(tmp_path)
    monkeypatch.setattr(benchmark_codec, "dataset_config", partial(_config, config))
    monkeypatch.setattr(benchmark_codec, "candidates", dict)
    report = benchmark_codec.run(
        "test",
        imagenet=_imagenet(tmp_path, count=8),
        num_fit_images=6,
        num_eval_images=2,
        device="cpu",
        batch_size=2,
        decode_images=False,
    )
    assert (report["fit_images"], report["eval_images"]) == (6, 2)


def _config(config: object, experiment: str) -> object:
    del experiment
    return config


def test_the_fit_and_eval_split_ignores_source_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Source order is grouped by class, so evaluation must not be its tail."""
    config = _source(tmp_path)
    encoded: list[list[str]] = []
    monkeypatch.setattr(benchmark_codec, "dataset_config", partial(_config, config))
    monkeypatch.setattr(benchmark_codec, "candidates", dict)
    monkeypatch.setattr(
        benchmark_codec,
        "_encode_all",
        partial(_record_stems, encoded=encoded),
    )
    _ = benchmark_codec.run(
        "test",
        imagenet=_imagenet(tmp_path, count=8),
        num_fit_images=4,
        num_eval_images=4,
        device="cpu",
        batch_size=2,
        decode_images=False,
    )
    assert sorted(encoded[1]) != [f"{index:08d}" for index in range(4, 8)]


def _record_stems(
    autoencoder: object,
    listed: Sequence[prepare_data.Record],
    *,
    encoded: list[list[str]],
    latent_shape: tuple[int, int, int],
    **_: object,
) -> Tensor:
    del autoencoder
    encoded.append([record.stem for record in listed])
    return torch.zeros(len(listed), *latent_shape)


@pytest.mark.parametrize(("fit", "evaluate"), [(0, 1), (1, -1), (2, 1)])
def test_benchmark_needs_positive_counts_the_source_can_supply(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fit: int,
    evaluate: int,
) -> None:
    monkeypatch.setattr(
        benchmark_codec,
        "dataset_config",
        partial(_config, _source(tmp_path)),
    )
    with pytest.raises(ValueError, match=r"positive|too small"):
        benchmark_codec.run(
            "test",
            imagenet=_imagenet(tmp_path, count=2),
            num_fit_images=fit,
            num_eval_images=evaluate,
            device="cpu",
            batch_size=2,
            decode_images=False,
        )


def test_nan_reconstructions_score_nan_rather_than_perfect() -> None:
    latents = _latents(2, seed=0)
    broken = torch.full_like(latents, float("nan"))
    metrics = benchmark_codec.latent_metrics(
        latents,
        decoded=broken,
        normalizer=ScaleLatents.Config().make(),
    )
    assert math.isnan(metrics["snr_db"])
    assert math.isnan(metrics["channel_snr_db_min"])
    assert math.isnan(
        benchmark_codec.psnr(torch.zeros(1), other=torch.tensor([math.nan])),
    )


def test_benchmark_refuses_crops_from_another_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _imagenet(tmp_path / "first", count=2)
    second = _imagenet(tmp_path / "second", count=3)
    config = _source(tmp_path)
    prepare_data.prepare(config, imagenet=first, device="cpu")
    monkeypatch.setattr(benchmark_codec, "dataset_config", partial(_config, config))
    monkeypatch.setattr(benchmark_codec, "candidates", dict)
    with pytest.raises(CorpusMismatchError, match="image source"):
        benchmark_codec.run(
            "test",
            imagenet=second,
            num_fit_images=1,
            num_eval_images=1,
            device="cpu",
            batch_size=2,
            decode_images=False,
        )


def test_preparation_refuses_crops_from_another_benchmarks_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _imagenet(tmp_path / "first", count=2)
    second = _imagenet(tmp_path / "second", count=3)
    config = _source(tmp_path)
    monkeypatch.setattr(benchmark_codec, "dataset_config", partial(_config, config))
    monkeypatch.setattr(benchmark_codec, "candidates", dict)
    benchmark_codec.run(
        "test",
        imagenet=first,
        num_fit_images=1,
        num_eval_images=1,
        device="cpu",
        batch_size=2,
        decode_images=False,
    )
    with pytest.raises(CorpusMismatchError, match="image source"):
        prepare_data.prepare(config, imagenet=second, device="cpu")


def test_output_inside_an_input_is_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _imagenet(tmp_path, count=2)
    monkeypatch.setattr(
        benchmark_codec,
        "dataset_config",
        partial(_config, _source(tmp_path)),
    )
    target = raw / "train" / "report.json"
    monkeypatch.setattr(
        "sys.argv",
        ["benchmark_codec", "--source", str(raw), "--output", str(target)],
    )
    with pytest.raises(SystemExit):
        _ = benchmark_codec.main()
    assert not target.exists()


def test_every_candidate_builds() -> None:
    names = benchmark_codec.candidates()
    assert {"float32", "bfloat16", "float16", "uint8_lloyd_max"} <= names.keys()
    for config in names.values():
        _ = config.make()


def _latents(images: int, *, seed: int) -> Tensor:
    generator = torch.Generator().manual_seed(seed)
    scale = torch.tensor([1.0, 30.0]).view(1, 2, 1, 1)
    return torch.randn(images, 2, 4, 4, generator=generator) * scale


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
