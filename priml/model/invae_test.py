"""Tests for the INVAE building blocks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable
from unittest.mock import Mock, call

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


@runtime_checkable
class _TensorFunction(Protocol):
    def __call__(self, value: Tensor) -> Tensor: ...


@runtime_checkable
class _Normalizer(Protocol):
    def __call__(self, in_channels: int, num_groups: int = 32) -> nn.GroupNorm: ...


@runtime_checkable
class _VAEFactory(Protocol):
    def __call__(self, **kwargs: object) -> AutoencoderKL: ...


@runtime_checkable
class _InVAEApi(Protocol):
    nonlinearity: _TensorFunction
    Normalize: _Normalizer
    VAE_F8D4: _VAEFactory
    VAE_F16D32: _VAEFactory


def _invae_api() -> _InVAEApi:
    assert isinstance(invae, _InVAEApi)
    return invae


@runtime_checkable
class _Posterior(Protocol):
    mean: Tensor
    logvar: Tensor


@runtime_checkable
class _EncodableAutoencoder(Protocol):
    def encode(self, x: Tensor) -> _Posterior: ...


def _encode_autoencoder(model: object, image: Tensor) -> _Posterior:
    assert isinstance(model, _EncodableAutoencoder)
    return model.encode(image)


@runtime_checkable
class _DecoderArchitecture(Protocol):
    ch: int
    num_resolutions: int
    num_res_blocks: int
    resolution: int
    in_channels: int
    give_pre_end: bool
    z_shape: tuple[int, int, int, int]
    last_z_shape: torch.Size | None


@runtime_checkable
class _ResnetBlock(Protocol):
    in_channels: int
    out_channels: int
    use_conv_shortcut: bool

    def forward(self, x: Tensor, temb: Tensor | None) -> Tensor: ...


@runtime_checkable
class _AttentionBlock(Protocol):
    in_channels: int

    def forward(self, x: Tensor) -> Tensor: ...


def _registered_module[ModuleT: nn.Module](
    parent: nn.Module,
    name: str,
    module_type: type[ModuleT],
) -> ModuleT:
    module = parent._modules[name]
    assert isinstance(module, module_type)
    return module


def _registered_modules(parent: nn.Module, name: str) -> tuple[nn.Module, ...]:
    modules = _registered_module(parent, name, nn.ModuleList)
    return tuple(
        _registered_module(modules, key, nn.Module) for key in modules._modules
    )


def _has_registered_module(parent: nn.Module, name: str) -> bool:
    return name in parent._modules


def _resnet_block(module: object) -> _ResnetBlock:
    assert isinstance(module, _ResnetBlock)
    return module


def _attention_block(module: object) -> _AttentionBlock:
    assert isinstance(module, _AttentionBlock)
    return module


def test_nonlinearity_is_swish() -> None:
    api = _invae_api()
    x = torch.tensor([-2.0, 0.5, 3.0])
    assert torch.equal(api.nonlinearity(x), x * torch.sigmoid(x))


def test_basic_blocks_and_resnet_shortcuts() -> None:
    x = torch.randn(2, 32, 4, 6)
    upsample = Upsample(32, with_conv=True)
    downsample = Downsample(32, with_conv=True)
    assert upsample.with_conv
    assert upsample.conv.kernel_size == (3, 3)
    assert upsample.conv.stride == (1, 1)
    assert upsample.conv.padding == (1, 1)
    assert downsample.with_conv
    assert downsample.conv.kernel_size == (3, 3)
    assert downsample.conv.stride == (2, 2)
    assert downsample.conv.padding == (0, 0)
    assert upsample(x).shape == (2, 32, 8, 12)
    assert downsample(x).shape == (2, 32, 2, 3)
    plain_upsample = Upsample(32, with_conv=False)
    plain_downsample = Downsample(32, with_conv=False)
    pattern = torch.arange(144, dtype=torch.float32).reshape(2, 3, 4, 6)
    assert torch.equal(
        plain_upsample(pattern),
        pattern.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3),
    )
    assert torch.equal(
        plain_downsample(pattern),
        # Downsample reshapes spatial axes by its fixed factor of two.
        pattern.reshape(2, 3, 2, 2, 3, 2).mean((3, 5)),
    )
    normalization = _invae_api().Normalize(32)
    assert normalization.num_groups == 32
    assert normalization.num_channels == 32
    assert normalization.eps == 1e-6
    assert normalization.affine
    temb = torch.randn(2, 7)
    block_module = ResnetBlock(
        in_channels=32,
        out_channels=64,
        conv_shortcut=True,
        dropout=0.0,
        temb_channels=7,
    )
    block = _resnet_block(block_module)
    conv1 = _registered_module(block_module, "conv1", nn.Conv2d)
    temb_proj = _registered_module(block_module, "temb_proj", nn.Linear)
    conv_shortcut = _registered_module(block_module, "conv_shortcut", nn.Conv2d)
    assert block.in_channels == 32
    assert block.out_channels == 64
    assert block.use_conv_shortcut
    assert conv1.in_channels == 32
    assert conv1.out_channels == 64
    assert conv1.kernel_size == (3, 3)
    assert conv1.stride == (1, 1)
    assert conv1.padding == (1, 1)
    assert temb_proj.in_features == 7
    assert temb_proj.out_features == 64
    assert conv_shortcut.kernel_size == (3, 3)
    assert conv_shortcut.in_channels == 32
    assert conv_shortcut.out_channels == 64
    result = block.forward(x, temb)
    assert result.shape == (2, 64, 4, 6)
    block2_module = ResnetBlock(
        in_channels=32,
        out_channels=64,
        conv_shortcut=False,
        dropout=0.0,
        temb_channels=0,
    )
    block2 = _resnet_block(block2_module)
    nin_shortcut = _registered_module(block2_module, "nin_shortcut", nn.Conv2d)
    assert block2.in_channels == 32
    assert block2.out_channels == 64
    assert not block2.use_conv_shortcut
    assert nin_shortcut.kernel_size == (1, 1)
    assert nin_shortcut.in_channels == 32
    assert nin_shortcut.out_channels == 64
    assert not _has_registered_module(block2_module, "temb_proj")
    result = block2.forward(x, None)
    assert result.shape == (2, 64, 4, 6)
    default_block_module = ResnetBlock(in_channels=32, dropout=0.0)
    default_block = _resnet_block(default_block_module)
    default_temb = _registered_module(default_block_module, "temb_proj", nn.Linear)
    assert default_block.out_channels == 32
    assert not default_block.use_conv_shortcut
    assert default_temb.in_features == 512
    assert default_temb.out_features == 32
    single_channel_module = ResnetBlock(
        in_channels=32,
        dropout=0.0,
        temb_channels=1,
    )
    single_channel_temb = _registered_module(
        single_channel_module,
        "temb_proj",
        nn.Linear,
    )
    assert single_channel_temb.in_features == 1
    attention_module = AttnBlock(32)
    attention = _attention_block(attention_module)
    assert attention.in_channels == 32
    for name in ("q", "k", "v", "proj_out"):
        projection = _registered_module(attention_module, name, nn.Conv2d)
        assert projection.in_channels == projection.out_channels == 32
        assert projection.kernel_size == (1, 1)
        assert projection.stride == (1, 1)
        assert projection.padding == (0, 0)
    result = attention.forward(x)
    assert result.shape == x.shape


def test_architecture_constructor_defaults() -> None:
    encoder = Encoder(ch=32)
    assert encoder.resolution == 256
    assert encoder.in_channels == 3
    assert encoder.num_resolutions == 5
    assert encoder.num_res_blocks == 2
    down = _registered_modules(encoder, "down")
    assert len(down) == 5
    down_blocks = [_registered_modules(level, "block") for level in down]
    down_attn = [_registered_modules(level, "attn") for level in down]
    assert all(len(blocks) == 2 for blocks in down_blocks)
    assert all(
        _registered_module(
            _registered_module(level, "downsample", Downsample),
            "conv",
            nn.Conv2d,
        ).stride
        == (2, 2)
        for level in down[:4]
    )
    assert [len(attn) for attn in down_attn] == [0, 0, 0, 0, 2]
    assert not _has_registered_module(down[-1], "downsample")
    assert (
        _registered_module(
            _registered_modules(down[0], "block")[0],
            "dropout",
            nn.Dropout,
        ).p
        == 0.0
    )
    assert all(
        not _has_registered_module(block, "temb_proj")
        for blocks in down_blocks
        for block in blocks
    )
    encoder_mid = _registered_module(encoder, "mid", nn.Module)
    assert not _has_registered_module(
        _registered_module(encoder_mid, "block_1", ResnetBlock),
        "temb_proj",
    )
    assert not _has_registered_module(
        _registered_module(encoder_mid, "block_2", ResnetBlock),
        "temb_proj",
    )
    assert _registered_module(encoder, "conv_in", nn.Conv2d).in_channels == 3
    assert _registered_module(encoder, "conv_out", nn.Conv2d).out_channels == 32

    decoder = Decoder(ch=32)
    assert isinstance(decoder, _DecoderArchitecture)
    assert decoder.resolution == 256
    assert decoder.in_channels == 3
    assert decoder.num_resolutions == 5
    assert decoder.num_res_blocks == 2
    assert decoder.z_shape == (1, 16, 16, 16)
    assert all(type(size) is int for size in decoder.z_shape)
    up = _registered_modules(decoder, "up")
    assert len(up) == 5
    up_blocks = [_registered_modules(level, "block") for level in up]
    up_attn = [_registered_modules(level, "attn") for level in up]
    assert all(len(blocks) == 3 for blocks in up_blocks)
    assert [len(attn) for attn in up_attn] == [0, 0, 0, 0, 3]
    upsample = _registered_module(up[4], "upsample", Upsample)
    assert _registered_module(upsample, "conv", nn.Conv2d).stride == (1, 1)
    assert _registered_module(up_blocks[4][0], "dropout", nn.Dropout).p == 0.0
    assert all(
        not _has_registered_module(block, "temb_proj")
        for blocks in up_blocks
        for block in blocks
    )
    decoder_mid = _registered_module(decoder, "mid", nn.Module)
    assert not _has_registered_module(
        _registered_module(decoder_mid, "block_1", ResnetBlock),
        "temb_proj",
    )
    assert not _has_registered_module(
        _registered_module(decoder_mid, "block_2", ResnetBlock),
        "temb_proj",
    )
    assert _registered_module(decoder, "conv_in", nn.Conv2d).in_channels == 16
    assert _registered_module(decoder, "conv_out", nn.Conv2d).out_channels == 3
    assert decoder.give_pre_end is False


def test_encoder_uses_floor_division_and_double_z() -> None:
    encoder = Encoder(
        ch=32,
        ch_mult=(1, 2),
        num_res_blocks=1,
        attn_resolutions=(3,),
        resolution=7,
        z_channels=2,
        double_z=False,
    )
    down = _registered_modules(encoder, "down")
    assert len(_registered_modules(down[1], "attn")) == 1
    assert _registered_module(encoder, "conv_out", nn.Conv2d).out_channels == 2


def test_encoder_without_resolutions_uses_input_width() -> None:
    encoder = Encoder(
        ch=32,
        ch_mult=(),
        num_res_blocks=0,
        attn_resolutions=(),
        resolution=8,
        z_channels=2,
        double_z=False,
    )

    assert encoder(torch.randn(2, 3, 8, 10)).shape == (2, 2, 8, 10)


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
    assert encoder.ch == 32
    assert encoder.temb_ch == 0
    assert encoder.num_resolutions == 2
    assert encoder.num_res_blocks == 1
    assert encoder.resolution == 8
    assert encoder.in_channels == 3
    conv_in = _registered_module(encoder, "conv_in", nn.Conv2d)
    assert conv_in.in_channels == 3
    assert conv_in.out_channels == 32
    down = _registered_modules(encoder, "down")
    down_blocks = [_registered_modules(level, "block") for level in down]
    down_attn = [_registered_modules(level, "attn") for level in down]
    assert len(down) == 2
    assert len(down_blocks[0]) == len(down_blocks[1]) == 1
    assert len(down_attn[0]) == 0
    assert len(down_attn[1]) == 1
    downsample = _registered_module(down[0], "downsample", Downsample)
    assert _registered_module(downsample, "conv", nn.Conv2d).stride == (2, 2)
    assert _registered_module(encoder, "conv_out", nn.Conv2d).out_channels == 4
    encoded = encoder(torch.randn(2, 3, 8, 10))
    assert encoded.shape == (2, 4, 4, 5)
    decoder = Decoder(
        ch=32,
        ch_mult=(1, 2),
        num_res_blocks=1,
        attn_resolutions=(4, 8),
        resolution=8,
        z_channels=2,
        out_ch=3,
    )
    assert isinstance(decoder, _DecoderArchitecture)
    assert decoder.ch == 32
    assert decoder.num_resolutions == 2
    assert decoder.num_res_blocks == 1
    assert decoder.resolution == 8
    assert decoder.in_channels == 3
    assert not decoder.give_pre_end
    assert decoder.last_z_shape is None
    assert decoder.z_shape == (1, 2, 4, 4)
    conv_in = _registered_module(decoder, "conv_in", nn.Conv2d)
    assert conv_in.in_channels == 2
    assert conv_in.out_channels == 64
    up = _registered_modules(decoder, "up")
    up_blocks = [_registered_modules(level, "block") for level in up]
    up_attn = [_registered_modules(level, "attn") for level in up]
    assert len(up) == 2
    assert len(up_blocks[0]) == len(up_blocks[1]) == 2
    assert len(up_attn[0]) == len(up_attn[1]) == 2
    upsample = _registered_module(up[1], "upsample", Upsample)
    assert _registered_module(upsample, "conv", nn.Conv2d).stride == (1, 1)
    assert (
        _registered_module(
            _registered_modules(up[0], "block")[0],
            "dropout",
            nn.Dropout,
        ).p
        == 0.0
    )
    decoder_mid = _registered_module(decoder, "mid", nn.Module)
    block_1 = _registered_module(decoder_mid, "block_1", ResnetBlock)
    block_2 = _registered_module(decoder_mid, "block_2", ResnetBlock)
    first_up_block = up_blocks[0][0]
    assert not _has_registered_module(block_1, "temb_proj")
    assert not _has_registered_module(block_2, "temb_proj")
    assert not _has_registered_module(first_up_block, "temb_proj")
    assert _resnet_block(block_1).out_channels == 64
    assert _resnet_block(block_2).out_channels == 64
    assert _registered_module(decoder, "conv_out", nn.Conv2d).out_channels == 3
    decoder_input = torch.randn(6, 2, 4, 5)
    decoded = decoder(decoder_input)
    assert decoder.last_z_shape == decoder_input.shape
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


def test_autoencoder_constructor_connects_variational_and_deterministic_channels() -> (
    None
):
    variational = AutoencoderKL(embed_dim=2, ch_mult=(1, 2))
    assert variational.use_variational
    assert variational.encoder.conv_out.out_channels == 4
    assert len(variational.encoder.down) == 2
    assert variational.decoder.conv_in.in_channels == 2
    assert len(variational.decoder.up) == 2
    assert variational.quant_conv.in_channels == 4
    assert variational.quant_conv.out_channels == 4
    assert variational.quant_conv.kernel_size == (1, 1)
    assert variational.post_quant_conv.in_channels == 2
    assert variational.post_quant_conv.out_channels == 2
    assert variational.post_quant_conv.kernel_size == (1, 1)

    deterministic = AutoencoderKL(embed_dim=2, ch_mult=(1, 2), use_variational=False)
    assert not deterministic.use_variational
    assert deterministic.quant_conv.in_channels == 4
    assert deterministic.quant_conv.out_channels == 2
    assert deterministic.encoder.conv_out.out_channels == 4


def test_nonvariational_encode_appends_unit_log_variance() -> None:
    model = AutoencoderKL.__new__(AutoencoderKL)
    nn.Module.__init__(model)
    model.encoder = Encoder(
        ch=32,
        ch_mult=(1, 2),
        num_res_blocks=1,
        attn_resolutions=(),
        resolution=8,
        z_channels=2,
    )
    model.quant_conv = nn.Conv2d(4, 2, 1)
    model.use_variational = False

    posterior = _encode_autoencoder(model, torch.zeros((2, 3, 8, 10)))

    assert posterior.mean.shape == (2, 2, 4, 5)
    assert torch.equal(posterior.logvar, torch.ones_like(posterior.mean))


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
    assert torch.equal(deterministic.sample(), deterministic.mean)
    assert torch.equal(deterministic.std, torch.zeros_like(deterministic.mean))
    assert torch.equal(deterministic.var, torch.zeros_like(deterministic.mean))
    assert torch.equal(deterministic.kl(), torch.zeros(1))
    assert torch.equal(deterministic.nll(sample), torch.zeros(1))


def test_gaussian_distribution_samples_from_its_mean_and_std() -> None:
    mean = torch.tensor([[[[1.0, -2.0], [0.5, 3.0]]], [[[2.0, 1.0], [-1.0, 0.0]]]])
    logvar = torch.tensor([[[[0.0, 1.0], [-1.0, 0.5]]], [[[1.0, -0.5], [0.0, 2.0]]]])
    distribution = DiagonalGaussianDistribution(torch.cat((mean, logvar), dim=1))
    torch.manual_seed(29)
    noise = torch.randn(mean.shape, device=mean.device)
    torch.manual_seed(29)

    assert torch.equal(distribution.sample(), mean + torch.exp(0.5 * logvar) * noise)


def test_gaussian_sample_stays_on_parameter_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def sample_noise(shape: torch.Size, *, device: torch.device) -> Tensor:
        return torch.empty(shape, device=device)

    random_normal = Mock(side_effect=sample_noise)
    monkeypatch.setattr(torch, "randn", random_normal)
    parameters = torch.empty(2, 8, 3, 5)
    distribution = DiagonalGaussianDistribution(parameters)
    sample = distribution.sample()
    random_normal.assert_called_once_with(
        distribution.mean.shape,
        device=parameters.device,
    )
    assert sample.shape == (2, 4, 3, 5)


def test_gaussian_distribution_clamps_log_variance() -> None:
    mean = torch.zeros(3, 2, 4, 5)
    logvar = torch.zeros_like(mean)
    logvar[0, 0, 0, :3] = torch.tensor([-31.0, -30.0, 0.0])
    logvar[0, 0, 1, :3] = torch.tensor([20.0, 21.0, 1.0])
    logvar[1, 0, 0, :4] = torch.arange(4.0)
    logvar[1, 0, 1, :2] = torch.tensor([4.0, 5.0])
    distribution = DiagonalGaussianDistribution(torch.cat((mean, logvar), dim=1))

    expected_logvar = logvar.clamp(-30.0, 20.0)
    assert torch.equal(distribution.logvar, expected_logvar)
    assert torch.equal(distribution.var, torch.exp(expected_logvar))
    assert torch.equal(distribution.std, torch.exp(0.5 * expected_logvar))


def test_gaussian_distribution_analytic_losses() -> None:
    mean = torch.tensor([[[[1.0, -2.0], [0.5, 3.0]]], [[[2.0, 1.0], [-1.0, 0.0]]]])
    logvar = torch.tensor([[[[0.0, 1.0], [-1.0, 0.5]]], [[[1.0, -0.5], [0.0, 2.0]]]])
    parameters = torch.cat((mean, logvar), dim=1)
    distribution = DiagonalGaussianDistribution(parameters)
    sample = torch.tensor([[[[2.0, 0.0], [-1.0, 4.0]]], [[[0.0, 3.0], [2.0, -2.0]]]])
    variance = torch.exp(logvar)
    expected_kl = 0.5 * (mean.square() + variance - 1.0 - logvar).sum((1, 2, 3))
    expected_nll = 0.5 * (
        torch.log(torch.tensor(2.0 * torch.pi))
        + logvar
        + (sample - mean).square() / variance
    ).sum((1, 2, 3))
    assert torch.allclose(distribution.kl(), expected_kl)
    assert torch.allclose(distribution.nll(sample), expected_nll)
    assert torch.allclose(
        distribution.nll(sample, dims=[1, 3]),
        0.5
        * (
            torch.log(torch.tensor(2.0 * torch.pi))
            + logvar
            + (sample - mean).square() / variance
        ).sum((1, 3)),
    )

    other_mean = torch.tensor([[[[0.0, 1.0], [2.0, -1.0]]], [[[1.0, 3.0], [2.0, 1.0]]]])
    other_logvar = torch.tensor(
        [[[[1.0, 0.0], [0.5, -1.0]]], [[[0.0, 1.0], [-0.5, 0.5]]]],
    )
    other = DiagonalGaussianDistribution(torch.cat((other_mean, other_logvar), dim=1))
    expected_pair_kl = 0.5 * (
        (mean - other_mean).square() / torch.exp(other_logvar)
        + variance / torch.exp(other_logvar)
        - 1.0
        - logvar
        + other_logvar
    ).sum((1, 2, 3))
    assert torch.allclose(distribution.kl(other), expected_pair_kl)


def test_predefined_vae_factories_set_their_latent_widths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory = Mock()
    monkeypatch.setattr(invae, "AutoencoderKL", factory)
    api = _invae_api()
    api.VAE_F8D4(ch=32)
    api.VAE_F16D32(ch=32)

    factory.assert_has_calls(
        [
            call(
                embed_dim=4,
                ch_mult=[1, 2, 4, 4],
                use_variational=True,
                ch=32,
            ),
            call(
                embed_dim=32,
                ch_mult=[1, 1, 2, 2, 4],
                use_variational=True,
                ch=32,
            ),
        ],
    )


def _tiny_autoencoder() -> AutoencoderKL:
    return AutoencoderKL(embed_dim=2, ch_mult=(1, 2))


def test_load_invae_local_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _tiny_autoencoder()
    checkpoint = tmp_path / "state.pt"
    torch.save(source.state_dict(), checkpoint)
    load_checkpoint = Mock(wraps=torch.load)
    monkeypatch.setattr(torch, "load", load_checkpoint)
    monkeypatch.setattr(invae, "VAE_F16D32", _tiny_autoencoder)
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
    source = _tiny_autoencoder()
    checkpoint = tmp_path / "state.pt"
    torch.save(source.state_dict(), checkpoint)

    class Hub:
        def hf_hub_download(self, *, repo_id: str, filename: str) -> Path:
            assert repo_id == "REPA-E/e2e-invae"
            assert filename == "e2e-invae-400k.pt"
            return checkpoint

    def import_hub(name: str) -> Hub:
        assert name == "huggingface_hub"
        return Hub()

    monkeypatch.setattr(invae, "VAE_F16D32", _tiny_autoencoder)
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
    def fail_import(name: str):
        raise ImportError(name)

    monkeypatch.setattr("priml.model.invae.import_module", fail_import)
    with pytest.raises(
        ImportError,
        match=r"^Install priml\[hub\] or pass a local INVAE checkpoint path$",
    ):
        load_invae()


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
