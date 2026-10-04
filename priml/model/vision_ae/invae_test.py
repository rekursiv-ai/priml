"""Tests for the INVAE wrapper.

``testdata/invae.pt`` was minted from the pre-protocol ``priml.model.invae``
(``encode_image`` plus the inline decode) before the module moved; replaying it
here is the proof that the move changed no arithmetic.

Parity with REG's own ``models/invae.py`` is ``scripts/reference_parity.py``'s,
not a golden's: REG hardcodes 32 GroupNorm groups, so the smallest model it
builds is far over the 32,768-byte golden ceiling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from configgle.testing import assert_pprint_golden
from torch import Tensor, nn

import pytest
import torch

from priml.cost import cost
from priml.model.vision_ae.custom_types import (
    Autoencoder,
    VariationalAutoencoder,
    posterior_mode,
)
from priml.model.vision_ae.invae import (
    INVAE,
    AttnBlock,
    Decoder,
    DiagonalGaussianDistribution,
    Downsample,
    Encoder,
    ResnetBlock,
    Upsample,
)
from priml.model.vision_ae.latent_norm import ScaleLatents
from priml.testing.bfb import assert_bfb_against_golden
from priml.testing.cost import assert_cost_matches_torch


_CWD: Final = Path(__file__).resolve().parent


def tiny() -> INVAE.Config:
    """Return the published architecture at golden size: widths 2 and 4, 16px."""
    config = INVAE.Config()
    config.channels_hidden = 2
    config.channel_multipliers = (1, 2)
    config.channels_latent = 2
    config.blocks_per_stage = 1
    config.num_groups = 2
    config.image_size = 16
    config.checkpoint = None
    return config


# The posterior draw reads the global generator, which the harness reseeds before
# building the module; an input drawn from it would shift that draw between mint and
# replay, since replay loads the stored input instead of rebuilding it.
def _image() -> Tensor:
    """Return a uint8 image from a private generator."""
    generator = torch.Generator().manual_seed(0)
    return torch.randint(0, 256, (1, 3, 16, 16), generator=generator, dtype=torch.uint8)


def _encode_decode(module: nn.Module, image: Tensor) -> Tensor:
    """Sample a latent, decode it, and join both."""
    assert isinstance(module, INVAE)
    latent = module.encode(image)
    return torch.cat([latent.flatten(), module.decode(latent).flatten()])


def test_invae_encode_decode_bfb() -> None:
    assert_bfb_against_golden(
        golden_dir=_CWD / "testdata",
        golden_name="invae",
        build_module=lambda: tiny().make(),
        build_input=_image,
        seed=0,
        run=_encode_decode,
    )


def test_invae_is_a_variational_autoencoder() -> None:
    model = tiny().make()
    assert isinstance(model, Autoencoder)
    assert isinstance(model, VariationalAutoencoder)


def test_encode_matches_the_configured_latent_shape() -> None:
    config = tiny()
    latent = config.make().encode(_image())
    assert tuple(latent.shape[1:]) == config.latent_shape()


def test_decode_returns_the_image_shape_in_the_unit_interval() -> None:
    model = tiny().make()
    image = model.decode(model.encode(_image()))
    assert image.shape == (1, 3, 16, 16)
    assert image.min() >= 0
    assert image.max() <= 1


def test_mode_latent_is_the_posterior_mean() -> None:
    config = tiny()
    config.latent_fn = posterior_mode
    model = config.make()
    assert torch.equal(model.encode(_image()), model.posterior(_image()).mean)


def test_sample_draws_from_an_explicit_generator() -> None:
    posterior = tiny().make().posterior(_image())
    first = posterior.sample(generator=torch.Generator().manual_seed(3))
    second = posterior.sample(generator=torch.Generator().manual_seed(3))
    assert torch.equal(first, second)
    assert not torch.equal(first, posterior.mode())


def test_train_leaves_the_frozen_model_in_eval_mode() -> None:
    model = tiny().make()
    _ = model.train()
    assert not model.training
    assert not any(parameter.requires_grad for parameter in model.parameters())


def test_published_architecture_keeps_the_checkpoint_parameter_names() -> None:
    """The e2e-invae-400k keys and shapes, pinned at the published defaults.

    342 entries with the reference module names: renaming a submodule would
    make ``load_state_dict`` reject the published checkpoint.
    """
    config = INVAE.Config()
    config.checkpoint = None
    with torch.device("meta"):
        state = config.make().state_dict()
    assert len(state) == 342
    assert tuple(state["encoder.conv_in.weight"].shape) == (128, 3, 3, 3)
    assert tuple(state["encoder.conv_out.weight"].shape) == (64, 512, 3, 3)
    assert tuple(state["quant_conv.weight"].shape) == (64, 64, 1, 1)
    assert tuple(state["post_quant_conv.weight"].shape) == (32, 32, 1, 1)
    assert tuple(state["decoder.conv_out.weight"].shape) == (3, 128, 3, 3)
    assert "encoder.down.4.attn.0.q.weight" in state


def test_published_latent_is_32_channels_at_16x16() -> None:
    assert INVAE.Config().latent_shape() == (32, 16, 16)


def test_latent_norm_is_the_published_scale() -> None:
    norm = INVAE.Config().latent_norm
    assert isinstance(norm, ScaleLatents.Config)
    assert norm.scale == 0.3099


def test_rejects_an_image_size_the_levels_cannot_halve() -> None:
    config = tiny()
    config.image_size = 15
    with pytest.raises(ValueError, match="divisible by 2"):
        _ = config.copy_tree().finalize()


def _round_trip(module: nn.Module, image: Tensor) -> Tensor:
    """Decode a sampled latent: the invocation ``cost`` describes."""
    assert isinstance(module, INVAE)
    return module.decode(module.encode(image))


def test_cost_matches_torch_for_a_forward_round_trip() -> None:
    """Convolution and attention FLOPs and bytes, and parameters, match torch.

    The module is frozen and fed uint8, so the harness's output carries no
    graph and it measures the forward alone -- the whole of what is costed.
    """
    generator = torch.Generator().manual_seed(0)
    analytical = assert_cost_matches_torch(
        tiny(),
        build_input=lambda: torch.randint(
            0,
            256,
            (2, 3, 16, 16),
            generator=generator,
            dtype=torch.uint8,
        ),
        run=_round_trip,
        batch_size=2,
        dtype=None,
    )
    assert analytical["flops", "matmul"].sum() == 4_583_424
    assert analytical["adjoint"].sum() == 0
    assert analytical["flops", "primal", "elementwise"].sum() > 0
    assert analytical["flops", "primal", "reduction"].sum() > 0


def test_cost_params_are_every_parameter() -> None:
    config = tiny()
    model = config.make()
    analytical = cost(config.copy_tree().finalize(), batch_size=1, dtype=None)
    assert analytical.params == sum(p.numel() for p in model.parameters())
    assert analytical.params_active == analytical.params


def test_published_cost_is_positive_and_linear_in_batch_size() -> None:
    """The pinned counts are torch's for one image through a meta-device build.

    ``FlopCounterMode`` over ``decode(encode(x))`` counts 390,131,613,696
    FLOPs, and the build owns 69,836,067 parameters; measuring them here would
    take the published forward, past this test's budget.
    """
    config = INVAE.Config().copy_tree().finalize()
    one = cost(config, batch_size=1, dtype=torch.bfloat16)
    two = cost(config, batch_size=2, dtype=torch.bfloat16)
    assert one["flops", "primal", "matmul", torch.bfloat16] == 390_131_613_696
    assert one.params == 69_836_067
    assert one["flops", "primal", "elementwise", torch.bfloat16] > 0
    assert one["adjoint"].sum() == 0
    assert two["flops"] == one["flops"].tile(2)
    assert two.params == one.params


def test_invae_config_pprint() -> None:
    assert_pprint_golden(test_file=__file__, name="invae_config", config=INVAE.Config())


def test_encode_refuses_a_float_image() -> None:
    """A ``[0, 1]`` float image would otherwise encode as near-black."""
    model = tiny().make()
    with pytest.raises(TypeError, match="uint8"):
        _ = model.encode(torch.rand(2, 3, 4, 5))


def test_resamplers_and_residual_shortcuts_run_every_branch() -> None:
    """The reference's options INVAE never sets still build and run.

    Resampling without a convolution, a time embedding, and a convolutional
    shortcut are the branches ``scripts/reference_parity.py`` allowlists as
    unreached on the published path.
    """
    x = torch.randn(2, 32, 4, 6)
    assert Upsample(32, with_conv=True)(x).shape == (2, 32, 8, 12)
    assert Upsample(32, with_conv=False)(x).shape == (2, 32, 8, 12)
    assert Downsample(32, with_conv=True)(x).shape == (2, 32, 2, 3)
    assert Downsample(32, with_conv=False)(x).shape == (2, 32, 2, 3)
    shortcut = ResnetBlock(
        in_channels=32,
        out_channels=64,
        conv_shortcut=True,
        dropout=0.0,
        temb_channels=7,
    )
    assert shortcut(x, torch.randn(2, 7)).shape == (2, 64, 4, 6)
    projected = ResnetBlock(
        in_channels=32,
        out_channels=64,
        conv_shortcut=False,
        dropout=0.0,
        temb_channels=0,
    )
    assert projected(x, None).shape == (2, 64, 4, 6)
    assert AttnBlock(32)(x).shape == x.shape


def test_encoder_and_decoder_keep_non_square_grids() -> None:
    encoder = Encoder(
        ch=32,
        ch_mult=(1, 2),
        num_res_blocks=1,
        attn_resolutions=(4,),
        resolution=8,
        z_channels=2,
        double_z=True,
    )
    assert encoder(torch.randn(2, 3, 8, 10)).shape == (2, 2 * 2, 4, 5)
    decoder = Decoder(
        ch=32,
        ch_mult=(1, 2),
        num_res_blocks=1,
        attn_resolutions=(4,),
        resolution=8,
        z_channels=2,
        out_ch=3,
    )
    latent = torch.randn(6, 2, 4, 5)
    assert decoder(latent).shape == (6, 3, 8, 10)
    decoder.give_pre_end = True
    assert decoder(latent).shape == (6, 32, 8, 10)


def test_posterior_draws_its_shape_and_its_mode_is_the_mean() -> None:
    moments = torch.randn(2, 2 * 3, 4, 5, generator=torch.Generator().manual_seed(0))
    posterior = DiagonalGaussianDistribution(moments)
    assert posterior.sample().shape == (2, 3, 4, 5)
    assert torch.equal(posterior.mode(), posterior.mean)


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
