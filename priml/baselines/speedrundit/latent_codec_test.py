"""Tests for the latent storage codecs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import Tensor

import pytest
import torch

from priml.baselines.speedrundit.latent_codec import (
    NUM_LEVELS,
    FittedCodec,
    FloatCodec,
    GaussianFit,
    LatentCodec,
    LinearFit,
    LloydMaxFit,
    QuantileFit,
    ScalarTableCodec,
    ScaleGroups,
    SharedTable,
    TableFit,
    bits_per_scalar,
    entropy_bits,
)
from priml.math.scalar_quantization import midpoints


if TYPE_CHECKING:
    from configgle import Makeable


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
@pytest.mark.parametrize(
    "fit",
    [LloydMaxFit.Config, GaussianFit.Config, QuantileFit.Config, LinearFit.Config],
)
@pytest.mark.parametrize("groups", [SharedTable.Config, ScaleGroups.Config])
def test_fitted_codecs_preserve_cuda_device(
    fit: type[
        LloydMaxFit.Config | GaussianFit.Config | QuantileFit.Config | LinearFit.Config
    ],
    groups: type[SharedTable.Config | ScaleGroups.Config],
) -> None:
    sample = _sample(images=4).cuda()
    codec = _fitted(ScalarTableCodec.Config(fit=fit(), groups=groups()), sample=sample)
    stored = codec.encode(sample)
    assert stored.device == sample.device
    assert codec.decode(stored).device == sample.device
    assert codec.saturated(sample).device == sample.device


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_cpu_codec_table_can_code_cuda_inputs() -> None:
    codec = _fitted(ScalarTableCodec.Config(), sample=_sample(images=4))
    sample = _sample(images=4).cuda()
    assert codec.decode(codec.encode(sample)).device == sample.device
    assert codec.saturated(sample).device == sample.device


def test_float32_codec_is_the_identity() -> None:
    codec = FloatCodec.Config().make()
    latent = _sample()
    assert torch.equal(codec.decode(codec.encode(latent)), latent)
    assert codec.stored_dtype == torch.float32


def test_float16_codec_rounds_to_half_and_widens_back() -> None:
    codec = FloatCodec.Config(dtype=torch.float16).make()
    latent = _sample()
    stored = codec.encode(latent)
    assert stored.dtype == torch.float16
    assert torch.equal(codec.decode(stored), latent.half().float())


def test_float_encoding_compiles_without_reading_a_python_scalar() -> None:
    codec = FloatCodec.Config(dtype=torch.float16).make()
    latent = _sample(images=1)
    compiled = torch.compile(codec.encode, backend="eager", fullgraph=True)
    assert torch.equal(compiled(latent), codec.encode(latent))


def test_float_codec_preserves_the_floating_overflow_contract() -> None:
    codec = FloatCodec.Config(dtype=torch.float16).make()
    assert codec.encode(torch.tensor([1e6])).isinf().all()


@pytest.mark.parametrize("dtype", [torch.uint8, torch.float8_e4m3fn])
def test_float_codec_needs_a_dtype_the_corpus_can_store(dtype: torch.dtype) -> None:
    with pytest.raises(ValueError, match="FloatCodec stores"):
        _ = FloatCodec.Config(dtype=dtype).make()


def test_a_float_codec_has_no_table_to_fit() -> None:
    assert not isinstance(FloatCodec.Config().make(), FittedCodec)


def test_table_codec_stores_one_byte_per_scalar() -> None:
    sample = _sample()
    codec = _fitted(ScalarTableCodec.Config(), sample=sample)
    stored = codec.encode(sample)
    assert stored.dtype == torch.uint8
    assert stored.shape == sample.shape
    assert bits_per_scalar(codec) == 8
    assert bits_per_scalar(FloatCodec.Config(dtype=torch.float16).make()) == 16


def test_levels_themselves_decode_exactly() -> None:
    codec = _fitted(ScalarTableCodec.Config(), sample=_sample())
    levels = codec.table()["levels"]
    latent = levels.T.reshape(NUM_LEVELS, 2, 1, 1)
    assert torch.equal(codec.decode(codec.encode(latent)), latent)


def test_a_value_on_a_threshold_takes_the_lower_level() -> None:
    codec = _fitted(ScalarTableCodec.Config(), sample=_sample())
    latent = midpoints(codec.table()["levels"])[:, 7].reshape(1, 2, 1, 1)
    assert codec.encode(latent).flatten().tolist() == [7, 7]


@pytest.mark.parametrize(
    "levels",
    [
        torch.linspace(1, 0, NUM_LEVELS).unsqueeze(0),
        torch.full((1, NUM_LEVELS), float("nan")),
        torch.zeros(1, NUM_LEVELS - 1),
    ],
)
def test_load_table_refuses_levels_it_cannot_code_with(levels: Tensor) -> None:
    codec = ScalarTableCodec.Config().make()
    with pytest.raises(ValueError, match="levels"):
        codec.load_table({"levels": levels})
    assert codec.levels is None


@pytest.mark.parametrize("fit", [GaussianFit.Config(), LinearFit.Config(clip_sigmas=4)])
def test_a_single_value_per_channel_cannot_fit_a_table(fit: Makeable[TableFit]) -> None:
    with pytest.raises(ValueError, match="two values per channel"):
        ScalarTableCodec.Config(fit=fit).make().fit(torch.ones(1, 1, 1, 1))


@pytest.mark.parametrize("fit", [GaussianFit.Config(), LinearFit.Config(clip_sigmas=4)])
def test_a_non_finite_sample_cannot_fit_a_table(fit: Makeable[TableFit]) -> None:
    sample = torch.tensor([0.0, float("nan")]).reshape(2, 1, 1, 1)
    with pytest.raises(ValueError, match="finite"):
        ScalarTableCodec.Config(fit=fit).make().fit(sample)


@pytest.mark.parametrize("clip_sigmas", [0.0, -1.0, float("nan"), float("inf")])
def test_linear_fit_needs_a_positive_finite_clip(clip_sigmas: float) -> None:
    with pytest.raises(ValueError, match="clip_sigmas"):
        LinearFit.Config(clip_sigmas=clip_sigmas).make()


def test_values_beyond_the_outer_levels_saturate() -> None:
    codec = _fitted(ScalarTableCodec.Config(), sample=_sample())
    latent = torch.tensor([1e3, -1e3]).reshape(1, 2, 1, 1)
    assert codec.encode(latent).flatten().tolist() == [NUM_LEVELS - 1, 0]
    assert bool(codec.saturated(latent).all())


def test_decode_reads_each_channel_from_its_own_row() -> None:
    codec = _fitted(ScalarTableCodec.Config(), sample=_sample())
    levels = codec.table()["levels"]
    stored = torch.full((1, 2, 1, 1), 100, dtype=torch.uint8)
    assert codec.decode(stored).flatten().tolist() == [
        levels[0, 100].item(),
        levels[1, 100].item(),
    ]


def test_per_channel_tables_beat_a_shared_table_when_scales_differ() -> None:
    """The measured RAE spread: channel standard deviations differ ~33x.

    A shared table spends its levels on the wide channel, leaving the narrow
    one with a handful -- the reason the default is one table per channel.
    """
    sample = _sample(scales=(1.0, 30.0))
    per_channel = _mse_per_channel(
        _fitted(ScalarTableCodec.Config(), sample=sample),
        latent=sample,
    )
    shared = _mse_per_channel(
        _fitted(ScalarTableCodec.Config(groups=SharedTable.Config()), sample=sample),
        latent=sample,
    )
    assert per_channel[0] * 10 < shared[0]


def test_shared_table_gives_every_channel_the_same_levels() -> None:
    codec = _fitted(
        ScalarTableCodec.Config(groups=SharedTable.Config()),
        sample=_sample(),
    )
    levels = codec.table()["levels"]
    assert torch.equal(levels[0], levels[1])


def test_scale_groups_pool_channels_of_similar_scale() -> None:
    groups = ScaleGroups.Config(num_groups=2).make()
    assert groups(torch.tensor([5.0, 0.1, 4.0, 0.2])).tolist() == [1, 0, 1, 0]


def test_lloyd_max_beats_a_uniform_table_on_a_normal_source() -> None:
    sample = _sample(scales=(1.0,), images=256)
    lloyd = _mse_per_channel(
        _fitted(ScalarTableCodec.Config(), sample=sample),
        latent=sample,
    )
    uniform = _mse_per_channel(
        _fitted(ScalarTableCodec.Config(fit=LinearFit.Config()), sample=sample),
        latent=sample,
    )
    assert lloyd[0] < uniform[0]


def test_linear_fit_spans_the_range_or_the_clipped_range() -> None:
    values = torch.linspace(-1.0, 1.0, 1001, dtype=torch.float64)
    full = LinearFit.Config().make()(values)
    assert full[0].item() == pytest.approx(-1 + 1 / NUM_LEVELS)
    assert full[-1].item() == pytest.approx(1 - 1 / NUM_LEVELS)
    clipped = LinearFit.Config(clip_sigmas=1.0).make()(values)
    assert clipped[-1] < full[-1]


def test_quantile_fit_places_levels_at_equal_probability() -> None:
    values = torch.arange(NUM_LEVELS * 4, dtype=torch.float64)
    levels = QuantileFit.Config().make()(values)
    assert levels[:3].tolist() == [2.0, 6.0, 10.0]


def test_gaussian_fit_scales_the_unit_table_to_the_sample() -> None:
    values = 3 + 2 * torch.randn(4096, generator=torch.Generator().manual_seed(1))
    levels = GaussianFit.Config().make()(values)
    assert levels.mean().item() == pytest.approx(
        values.double().mean().item(),
        abs=1e-6,
    )


def test_lloyd_max_fit_honours_its_iteration_budget() -> None:
    values = torch.randn(4096, generator=torch.Generator().manual_seed(2))
    fitted = LloydMaxFit.Config(max_iterations=0).make()(values)
    assert fitted.shape == (NUM_LEVELS,)


def test_table_round_trips_through_load_table() -> None:
    sample = _sample()
    first = _fitted(ScalarTableCodec.Config(), sample=sample)
    second = ScalarTableCodec.Config().make()
    second.load_table(first.table())
    assert torch.equal(first.encode(sample), second.encode(sample))


def test_unfitted_codec_refuses_to_encode() -> None:
    with pytest.raises(RuntimeError, match="no table"):
        _ = ScalarTableCodec.Config().make().encode(_sample())


def test_table_encoding_compiles_without_reading_a_python_scalar() -> None:
    codec = _fitted(ScalarTableCodec.Config(), sample=_sample())
    latent = _sample(images=1)
    compiled = torch.compile(codec.encode, backend="eager", fullgraph=True)
    assert torch.equal(compiled(latent), codec.encode(latent))


def test_entropy_of_uniform_indices_is_eight_bits() -> None:
    stored = torch.arange(NUM_LEVELS, dtype=torch.uint8).repeat(4)
    assert entropy_bits(stored) == pytest.approx(8.0)


def _sample(*, scales: tuple[float, ...] = (1.0, 30.0), images: int = 64) -> Tensor:
    """Return ``[images, C, 4, 4]`` normal latents, channel c scaled by scales[c]."""
    generator = torch.Generator().manual_seed(0)
    noise = torch.randn(images, len(scales), 4, 4, generator=generator)
    return noise * torch.tensor(scales).view(1, -1, 1, 1)


def _fitted(config: ScalarTableCodec.Config, sample: Tensor) -> ScalarTableCodec:
    codec = config.make()
    codec.fit(sample)
    return codec


def _mse_per_channel(codec: LatentCodec, latent: Tensor) -> Tensor:
    error = codec.decode(codec.encode(latent)) - latent
    return error.pow(2).mean(dim=(0, 2, 3))


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
