"""Tests for the INVAE autoencoder."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock

import math

from torch import Tensor, nn
from torch.nn import functional

import pytest
import torch

from priml.model import invae
from priml.model.invae import (
    AutoencoderKL,
    Decoder,
    DiagonalGaussianDistribution,
    Downsample,
    Encoder,
    ResnetBlock,
    SpatialSelfAttention,
    Upsample,
    decode_latents,
    encode_image,
    load_invae,
    vae_f8d4,
    vae_f16d32,
)
from priml.testing.golden import assert_text_golden


if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def _tiny_autoencoder() -> AutoencoderKL:
    return AutoencoderKL(channels_latent=2, channel_multipliers=(1, 2))


@pytest.mark.parametrize(
    ("factory", "golden"),
    [(vae_f8d4, "invae_f8d4_state"), (vae_f16d32, "invae_f16d32_state")],
)
def test_state_dict_keys_and_shapes_match_the_checkpoint(
    factory: Callable[[], AutoencoderKL],
    golden: str,
) -> None:
    """Recorded from the reference port, so published checkpoints still load."""
    listing = "\n".join(
        f"{key} {tuple(value.shape)}" for key, value in factory().state_dict().items()
    )
    assert_text_golden(test_file=__file__, name=golden, rendered=listing)


def test_resampling_halves_and_doubles_space() -> None:
    x = torch.randn(2, 32, 4, 6)
    assert Upsample(32)(x).shape == (2, 32, 8, 12)
    assert Downsample(32)(x).shape == (2, 32, 2, 3)


def test_downsample_pads_only_the_bottom_and_right() -> None:
    downsample = Downsample(32)
    x = torch.randn(2, 32, 4, 6)
    expected = downsample.conv(functional.pad(x, (0, 1, 0, 1)))
    assert torch.equal(downsample(x), expected)


@pytest.mark.parametrize(("channels_out", "projects"), [(32, False), (64, True)])
def test_resnet_block_projects_the_residual_only_on_a_width_change(
    channels_out: int,
    projects: bool,
) -> None:
    block = ResnetBlock(32, channels_out)
    assert isinstance(block.nin_shortcut, nn.Conv2d) is projects
    assert block(torch.randn(2, 32, 4, 6)).shape == (2, channels_out, 4, 6)


def test_spatial_attention_matches_the_explicit_softmax() -> None:
    attention = SpatialSelfAttention(32)
    x = torch.randn(2, 32, 3, 5, dtype=torch.float64)
    attention.double()
    h = attention.norm(x)
    q, k, v = (p(h).flatten(-2) for p in (attention.q, attention.k, attention.v))
    weights = torch.softmax(q.transpose(-1, -2) @ k / 32**0.5, dim=-1)
    expected = x + attention.proj_out(
        (v @ weights.transpose(-1, -2)).unflatten(-1, (3, 5)),
    )
    torch.testing.assert_close(attention(x), expected, rtol=1e-12, atol=1e-12)


def test_encoder_stages_and_attention_placement() -> None:
    encoder = Encoder(
        channels=32,
        channel_multipliers=(1, 2),
        blocks_per_stage=1,
        attention_resolutions=(4,),
        resolution=8,
        channels_latent=2,
    )
    assert [len(level.attn) for level in encoder.down] == [0, 1]
    assert encoder.down[0].downsample is not None
    assert encoder.down[1].downsample is None
    assert encoder(torch.randn(2, 3, 8, 10)).shape == (2, 4, 4, 5)


def test_decoder_mirrors_the_encoder_with_one_more_block() -> None:
    decoder = Decoder(
        channels=32,
        channel_multipliers=(1, 2),
        blocks_per_stage=1,
        attention_resolutions=(4, 8),
        resolution=8,
        channels_latent=2,
    )
    assert [len(level.block) for level in decoder.up] == [2, 2]
    assert decoder.up[0].upsample is None
    assert decoder.up[1].upsample is not None
    assert decoder(torch.randn(3, 2, 4, 5)).shape == (3, 3, 8, 10)


def test_autoencoder_round_trips_shapes() -> None:
    model = _tiny_autoencoder()
    posterior, reconstruction = model(torch.randn(2, 3, 8, 10))
    assert posterior.mean.shape == (2, 2, 4, 5)
    assert reconstruction.shape == (2, 3, 8, 10)


def test_gaussian_splits_moments_and_clamps_log_variance() -> None:
    mean = torch.randn(2, 3, 4, 5)
    logvar = torch.full_like(mean, 25.0)
    logvar[0, 0, 0, 0] = -31.0
    distribution = DiagonalGaussianDistribution(torch.cat((mean, logvar), dim=1))
    expected = logvar.clamp(-30.0, 20.0)
    assert torch.equal(distribution.mean, mean)
    assert torch.equal(distribution.logvar, expected)
    assert torch.equal(distribution.var, torch.exp(expected))
    assert torch.equal(distribution.std, torch.exp(0.5 * expected))


def test_gaussian_samples_by_reparameterization() -> None:
    mean = torch.randn(2, 3, 4, 5)
    logvar = torch.randn(2, 3, 4, 5)
    distribution = DiagonalGaussianDistribution(torch.cat((mean, logvar), dim=1))
    torch.manual_seed(29)
    noise = torch.randn_like(mean)
    torch.manual_seed(29)
    assert torch.equal(distribution.sample(), mean + torch.exp(0.5 * logvar) * noise)


def test_gaussian_losses_match_their_closed_forms() -> None:
    mean, logvar = torch.randn(2, 3, 4, 5), torch.randn(2, 3, 4, 5)
    other_mean, other_logvar = torch.randn(2, 3, 4, 5), torch.randn(2, 3, 4, 5)
    distribution = DiagonalGaussianDistribution(torch.cat((mean, logvar), dim=1))
    other = DiagonalGaussianDistribution(torch.cat((other_mean, other_logvar), dim=1))
    sample = torch.randn(2, 3, 4, 5)
    variance = logvar.exp()
    torch.testing.assert_close(
        distribution.kl(),
        0.5 * (mean.square() + variance - 1 - logvar).sum((1, 2, 3)),
    )
    torch.testing.assert_close(
        distribution.kl(other),
        0.5
        * (
            (mean - other_mean).square() / other_logvar.exp()
            + variance / other_logvar.exp()
            - 1
            - logvar
            + other_logvar
        ).sum((1, 2, 3)),
    )
    per_element = 0.5 * (
        math.log(2 * math.pi) + logvar + (sample - mean).square() / variance
    )
    torch.testing.assert_close(distribution.nll(sample), per_element.sum((1, 2, 3)))
    torch.testing.assert_close(
        distribution.nll(sample, dim=(1, 3)),
        per_element.sum((1, 3)),
    )


def test_gaussian_broadcasts_over_leading_axes() -> None:
    moments = torch.randn(2, 3, 6, 4, 5)
    distribution = DiagonalGaussianDistribution(moments)
    assert distribution.mean.shape == (2, 3, 3, 4, 5)
    assert distribution.kl().shape == (2, 3)


def test_image_codecs_scale_into_and_out_of_the_latent() -> None:
    model = _tiny_autoencoder()
    image = torch.randint(0, 256, (2, 3, 8, 10), dtype=torch.uint8)
    torch.manual_seed(3)
    latents = encode_image(model, image)
    torch.manual_seed(3)
    expected = model.encode(image.float() / 127.5 - 1).sample()
    assert torch.equal(latents, expected)
    decoded = decode_latents(model, latents)
    assert torch.equal(
        decoded,
        ((model.decode(latents / 0.3099) + 1) / 2).clamp(0, 1),
    )


def test_load_invae_local_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "state.pt"
    torch.save(_tiny_autoencoder().state_dict(), checkpoint)
    load_checkpoint = Mock(wraps=torch.load)
    monkeypatch.setattr(torch, "load", load_checkpoint)
    monkeypatch.setattr(invae, "vae_f16d32", _tiny_autoencoder)
    loaded = load_invae(checkpoint, device="meta")
    load_checkpoint.assert_called_once_with(
        checkpoint,
        map_location="cpu",
        weights_only=True,
    )
    assert not loaded.training
    assert all(not parameter.requires_grad for parameter in loaded.parameters())
    assert all(parameter.device.type == "meta" for parameter in loaded.parameters())


def test_load_invae_downloads_expected_hub_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "state.pt"
    torch.save(_tiny_autoencoder().state_dict(), checkpoint)

    class Hub:
        def hf_hub_download(self, *, repo_id: str, filename: str) -> Path:
            assert repo_id == "REPA-E/e2e-invae"
            assert filename == "e2e-invae-400k.pt"
            return checkpoint

    def import_hub(name: str) -> Hub:
        assert name == "huggingface_hub"
        return Hub()

    monkeypatch.setattr(invae, "vae_f16d32", _tiny_autoencoder)
    monkeypatch.setattr(invae, "import_module", import_hub)
    loaded = load_invae()
    assert not loaded.training
    assert all(not parameter.requires_grad for parameter in loaded.parameters())


def test_load_invae_rejects_invalid_hub_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Hub:
        def hf_hub_download(self, *, repo_id: str, filename: str) -> int:
            del repo_id, filename
            return 3

    def import_hub(name: str) -> Hub:
        del name
        return Hub()

    monkeypatch.setattr(invae, "import_module", import_hub)
    with pytest.raises(
        TypeError,
        match=r"^Hugging Face Hub returned an invalid checkpoint path\.$",
    ):
        load_invae()


def test_load_invae_missing_hub_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_import(name: str) -> Tensor:
        raise ImportError(name)

    monkeypatch.setattr(invae, "import_module", fail_import)
    with pytest.raises(
        ImportError,
        match=r"^Install priml\[hub\] or pass a local INVAE checkpoint path$",
    ):
        load_invae()


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
