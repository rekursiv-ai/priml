"""CausalVideoVAE: the model around a family's halves, on Wan 2.1's halves at test size."""

from __future__ import annotations

import pytest
import torch

from priml.model.ae import wan
from priml.model.ae.custom_types import Autoencoder
from priml.model.ae.video import CausalConv3d, CausalVideoVAE
from priml.testing.cost import assert_cost_matches_torch


def tiny() -> CausalVideoVAE.Config:
    """Return Wan 2.1's halves at widths 2 and 4: 2x in time, 4x in space."""
    return CausalVideoVAE.Config(
        encoder=wan.Encoder.Config(
            channels_hidden=(2, 4, 4),
            temporal=(1, 2),
            blocks_per_stage=1,
            channels_latent=3,
            checkpoint=None,
        ),
        decoder=wan.Decoder.Config(
            channels_hidden=(4, 4, 2),
            channels_upsampled=(2, 2),
            temporal=(2, 1),
            blocks_per_stage=1,
            channels_latent=3,
            checkpoint=None,
        ),
        num_frames=5,
        image_size=(8, 12),
    )


def test_satisfies_the_autoencoder_protocol() -> None:
    assert isinstance(tiny().make(), Autoencoder)


def test_leading_dimensions_pass_through() -> None:
    model = tiny().make()
    clip = torch.randint(
        0,
        256,
        (2, 4, 3, 5, 8, 12),
        generator=torch.Generator().manual_seed(1),
        dtype=torch.uint8,
    )
    flat = model.encode(clip.flatten(0, 1)).mode()
    nested = model.encode(clip).mode()
    assert torch.equal(nested.flatten(0, 1), flat)
    assert model.decode(nested).shape == clip.shape


@pytest.mark.parametrize(
    ("frames", "side"),
    [(6, 8), (5, 10)],
    ids=["frames", "sides"],
)
def test_encode_refuses_a_clip_the_resamplers_cannot_divide(
    frames: int,
    side: int,
) -> None:
    clip = torch.zeros(2, 3, frames, side, 12, dtype=torch.uint8)
    with pytest.raises(ValueError, match="1 \\+ 2k frames of sides divisible by 4"):
        _ = tiny().make().encode(clip)


def test_encode_refuses_a_float_clip() -> None:
    with pytest.raises(TypeError, match="uint8"):
        _ = tiny().make().encode(torch.rand(2, 3, 5, 8, 12))


def test_either_half_can_be_left_out() -> None:
    encode_only, decode_only = tiny(), tiny()
    encode_only.decoder = None
    decode_only.encoder = None
    with pytest.raises(ValueError, match="without a decoder"):
        _ = encode_only.make().decode(torch.zeros(2, 3, 2, 2, 3))
    with pytest.raises(ValueError, match="without an encoder"):
        _ = decode_only.make().encode(torch.zeros(2, 3, 5, 8, 12, dtype=torch.uint8))


def test_train_leaves_the_frozen_model_in_eval_mode() -> None:
    model = tiny().make()
    _ = model.train()
    assert not model.training
    assert not any(parameter.requires_grad for parameter in model.parameters())


def test_rejects_halves_that_disagree_on_the_latent() -> None:
    config = tiny()
    assert isinstance(config.decoder, wan.Decoder.Config)
    config.decoder.channels_latent = 5
    with pytest.raises(ValueError, match="channels_latent"):
        _ = config.make()


def test_rejects_halves_that_disagree_on_the_factors() -> None:
    config = tiny()
    assert isinstance(config.decoder, wan.Decoder.Config)
    config.decoder.temporal = (2, 2)
    with pytest.raises(ValueError, match="factors"):
        _ = config.make()


@pytest.mark.parametrize(
    ("num_frames", "image_size"),
    [(6, (8, 12)), (5, (8, 10))],
    ids=["frames", "sides"],
)
def test_rejects_a_stated_clip_the_model_cannot_encode(
    num_frames: int,
    image_size: tuple[int, int],
) -> None:
    """``latent_shape`` and ``cost`` would otherwise round a clip ``encode`` refuses."""
    config = tiny()
    config.num_frames, config.image_size = num_frames, image_size
    with pytest.raises(ValueError, match="must be 1 \\+ 2k and image_size"):
        _ = config.make()


def test_config_needs_a_half() -> None:
    config = tiny()
    config.encoder = config.decoder = None
    with pytest.raises(ValueError, match="needs an encoder or a decoder"):
        _ = config.latent_shape()
    with pytest.raises(ValueError, match="needs an encoder or a decoder"):
        _ = config.make()


def test_cost_needs_both_halves() -> None:
    config = tiny()
    config.decoder = None
    with pytest.raises(ValueError, match="prices both halves"):
        _ = config.cost(batch_size=1, dtype=None)


@pytest.mark.parametrize("padding_mode", ["constant", "replicate"])
def test_causal_conv3d_keeps_the_grid_and_sees_no_later_frame(
    padding_mode: str,
) -> None:
    conv = CausalConv3d.Config(2, 3, padding_mode=padding_mode).make()
    x = torch.randn(2, 2, 5, 4, 6)
    later = x.clone()
    later[:, :, 3:] = torch.randn(2, 2, 2, 4, 6)
    with torch.no_grad():
        y, y_later = conv(x), conv(later)
    assert y.shape == (2, 3, 5, 4, 6)
    assert torch.equal(y[:, :, :3], y_later[:, :, :3])
    assert not torch.equal(y[:, :, 3:], y_later[:, :, 3:])


def test_causal_conv3d_refuses_conv3d_padding_on_top_of_its_own() -> None:
    with pytest.raises(ValueError, match="padding must be 0"):
        _ = CausalConv3d.Config(2, 3, padding=1).make()


def test_causal_conv3d_cost_matches_torch() -> None:
    """A strided, unpadded time convolution, Wan's downsampler's."""
    _ = assert_cost_matches_torch(
        CausalConv3d.Config(2, 3, kernel_size=(3, 1, 1), stride=(2, 1, 1), past=0),
        build_input=lambda: torch.randn(2, 2, 5, 4, 6, requires_grad=True),
        input_grid=(5, 4, 6),
        batch_size=2,
        dtype=None,
    )


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
