"""Tests for the codec benchmark's metrics and candidate table."""

from __future__ import annotations

from typing import TYPE_CHECKING

import math

import pytest
import torch

from priml.baselines.speedrundit.latent_codec import FloatCodec, ScalarTableCodec
from priml.baselines.speedrundit.scripts import benchmark_codec
from priml.model.vision_ae.latent_norm import ScaleLatents


if TYPE_CHECKING:
    from torch import Tensor


def _latents(images: int, *, seed: int) -> Tensor:
    generator = torch.Generator().manual_seed(seed)
    scale = torch.tensor([1.0, 30.0]).view(1, 2, 1, 1)
    return torch.randn(images, 2, 4, 4, generator=generator) * scale


def test_identical_latents_score_zero_error() -> None:
    latents = _latents(4, seed=0)
    metrics = benchmark_codec.latent_metrics(
        latents,
        latents.clone(),
        ScaleLatents.Config().make(),
    )
    assert metrics["nmse"] == 0.0
    assert metrics["snr_db"] == math.inf
    assert metrics["max_abs_error"] == 0.0


def test_error_is_measured_after_normalization() -> None:
    """A normalizer that doubles latents doubles the error it reports."""
    latents = _latents(4, seed=0)
    noisy = latents + 0.01
    plain = benchmark_codec.latent_metrics(latents, noisy, ScaleLatents.Config().make())
    doubled = benchmark_codec.latent_metrics(
        latents,
        noisy,
        ScaleLatents.Config(scale=2.0).make(),
    )
    assert doubled["max_abs_error"] == pytest.approx(2 * plain["max_abs_error"])
    assert doubled["nmse"] == pytest.approx(plain["nmse"])


def test_psnr_of_a_uniform_offset() -> None:
    reference = torch.zeros(1, 3, 4, 4)
    assert benchmark_codec.psnr(reference, reference + 0.1) == pytest.approx(20.0)
    assert benchmark_codec.psnr(reference, reference) == math.inf


def test_evaluate_fits_on_one_sample_and_scores_on_the_other() -> None:
    report = benchmark_codec.evaluate(
        {
            "float16": FloatCodec.Config(dtype=torch.float16),
            "lloyd": ScalarTableCodec.Config(),
        },
        _latents(32, seed=0),
        _latents(8, seed=1),
        ScaleLatents.Config().make(),
    )
    assert report["float16"]["bits_per_scalar"] == 16
    assert report["lloyd"]["bits_per_scalar"] == 8
    assert "index_entropy_bits" in report["lloyd"]
    assert report["float16"]["snr_db"] > report["lloyd"]["snr_db"]


def test_evaluate_scores_images_when_given_a_decoder() -> None:
    report = benchmark_codec.evaluate(
        {"float32": FloatCodec.Config()},
        _latents(8, seed=0),
        _latents(4, seed=1),
        ScaleLatents.Config().make(),
        decode=lambda latents: latents.sigmoid(),
    )
    assert report["float32"]["image_psnr_db"] == math.inf


def test_fit_stability_reports_each_size() -> None:
    curve = benchmark_codec.fit_stability(
        _latents(64, seed=0),
        _latents(8, seed=1),
        ScaleLatents.Config().make(),
        [16, 64],
    )
    assert set(curve) == {16, 64}
    assert all(value > 0 for value in curve.values())


def test_every_candidate_builds() -> None:
    names = benchmark_codec.candidates()
    assert {"float32", "bfloat16", "float16", "uint8_lloyd_max"} <= names.keys()
    for config in names.values():
        _ = config.make()


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
