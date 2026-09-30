"""Tests for the INVAE building blocks."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import Tensor, nn

import pytest
import torch

from priml.model import invae
from priml.model.invae import (
    AttnBlock,
    AutoencoderKL,
    Decoder,
    DiagonalGaussianDistribution,
    Downsample,
    Encoder,
    ResnetBlock,
    Upsample,
    load_invae,
)


if TYPE_CHECKING:
    from pathlib import Path


def test_basic_blocks_and_resnet_shortcuts() -> None:
    x = torch.randn(2, 32, 4, 6)
    assert Upsample(32, with_conv=True)(x).shape == (2, 32, 8, 12)
    assert Downsample(32, with_conv=True)(x).shape == (2, 32, 2, 3)
    assert Downsample(32, with_conv=False)(x).shape == (2, 32, 2, 3)
    temb = torch.randn(2, 7)
    block = ResnetBlock(
        in_channels=32,
        out_channels=64,
        conv_shortcut=True,
        dropout=0.0,
        temb_channels=7,
    )
    result = block.forward(x, temb)
    assert isinstance(result, Tensor)
    assert result.shape == (2, 64, 4, 6)
    block2 = ResnetBlock(
        in_channels=32,
        out_channels=64,
        conv_shortcut=False,
        dropout=0.0,
        temb_channels=0,
    )
    result = block2.forward(x, None)
    assert isinstance(result, Tensor)
    assert result.shape == (2, 64, 4, 6)
    attention = AttnBlock(32)
    result = attention.forward(x)
    assert isinstance(result, Tensor)
    assert result.shape == x.shape


def test_encoder_decoder_and_autoencoder_paths() -> None:
    encoder = Encoder(
        ch=32,
        ch_mult=(1, 2),
        num_res_blocks=1,
        attn_resolutions=(4,),
        resolution=8,
        z_channels=2,
        double_z=True,
    )
    encoded = encoder(torch.randn(2, 3, 8, 10))
    assert encoded.shape == (2, 4, 4, 5)
    decoder = Decoder(
        ch=32,
        ch_mult=(1, 2),
        num_res_blocks=1,
        attn_resolutions=(4,),
        resolution=8,
        z_channels=2,
        out_ch=3,
    )
    decoded = decoder(torch.randn(6, 2, 4, 5))
    assert decoded.shape == (6, 3, 8, 10)
    decoder.give_pre_end = True
    assert decoder(torch.randn(6, 2, 4, 5)).shape[1] == 32
    decoder.give_pre_end = False
    model = AutoencoderKL.__new__(AutoencoderKL)
    nn.Module.__init__(model)
    model.encoder = encoder
    model.decoder = decoder
    model.use_variational = True
    model.quant_conv = nn.Conv2d(4, 4, 1)
    model.post_quant_conv = nn.Conv2d(2, 2, 1)
    posterior, latent, reconstruction = model(torch.randn(2, 3, 8, 10))
    assert posterior.mean.shape == (2, 2, 4, 5)
    assert latent.shape == (2, 2, 4, 5)
    assert reconstruction is not None
    assert reconstruction.shape == (2, 3, 8, 10)
    assert model(torch.randn(2, 3, 8, 10), return_recon=False)[2] is None


def test_gaussian_distribution() -> None:
    parameters = torch.cat((torch.zeros(2, 3, 4, 5), torch.zeros(2, 3, 4, 5)), dim=1)
    distribution = DiagonalGaussianDistribution(parameters)
    sample = distribution.sample()
    assert sample.shape == (2, 3, 4, 5)
    assert distribution.kl().shape == (2,)
    other = DiagonalGaussianDistribution(parameters + 0.25)
    assert distribution.kl(other).shape == (2,)
    assert distribution.nll(sample).shape == (2,)
    assert torch.equal(distribution.mode(), distribution.mean)
    deterministic = DiagonalGaussianDistribution(
        parameters,
        deterministic=True,
    )
    assert deterministic.sample().shape == deterministic.mean.shape
    assert torch.equal(deterministic.kl(), torch.zeros(1))
    assert torch.equal(deterministic.nll(sample), torch.zeros(1))


def _tiny_autoencoder() -> AutoencoderKL:
    return AutoencoderKL(embed_dim=2, ch_mult=(1, 2))


def test_load_invae_local_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _tiny_autoencoder()
    checkpoint = tmp_path / "state.pt"
    torch.save(source.state_dict(), checkpoint)
    monkeypatch.setattr(invae, "VAE_F16D32", _tiny_autoencoder)
    loaded = load_invae(checkpoint)
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
    with pytest.raises(TypeError, match="invalid checkpoint"):
        load_invae()


def test_load_invae_missing_hub_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_import(name: str):
        raise ImportError(name)

    monkeypatch.setattr("priml.model.invae.import_module", fail_import)
    with pytest.raises(ImportError, match="Install priml"):
        load_invae()


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
