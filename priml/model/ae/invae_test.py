"""INVAE: the published architecture, its checkpoint names, and its arithmetic.

``testdata/invae.pt`` freezes the round trip at golden size.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import json

from safetensors.torch import save_file
from torch import Tensor, nn
from treekle import json_loads

import pytest
import torch

from priml.cost import cost
from priml.model.ae.checkpoint import HubFile, LocalFile
from priml.model.ae.distribution import DiagonalGaussian
from priml.model.ae.invae import INVAE, remap_hf_state_dict
from priml.model.ae.latent_norm import ScaleLatents
from priml.testing.bfb import assert_bfb_against_golden
from priml.testing.cost import assert_cost_matches_torch
from priml.testing.golden import assert_pprint_golden


_CWD: Final = Path(__file__).resolve().parent

_TESTDATA: Final = _CWD / "testdata"


def tiny() -> INVAE.Config:
    """Return the published layout at golden size: widths 2 and 4, 16px."""
    config = INVAE.Config()
    assert config.encoder is not None
    assert config.decoder is not None
    for half in (config.encoder, config.decoder):
        half.channels_hidden = (2, 4)
        half.attention = (False, True)
        half.channels_latent = 2
        half.num_groups = 2
        half.checkpoint = None
    config.encoder.blocks_per_stage = 1
    config.decoder.blocks_per_stage = 2
    config.image_size = 16
    return config


def _published() -> INVAE.Config:
    return INVAE.Config.from_hf(
        json_loads(
            (_TESTDATA / "e2e_invae_hf_config.json").read_text(),
            dict[str, object],
        ),
    )


def _published_keys() -> dict[str, tuple[int, ...]]:
    """Every tensor in the published safetensors, by name, with its shape."""
    lines = (_TESTDATA / "e2e_invae_hf_keys.txt").read_text().splitlines()
    return {
        name: tuple(int(n) for n in shape.split("x"))
        for name, shape in (line.split() for line in lines)
    }


# The posterior draw reads the global generator, which the harness reseeds before
# building the module; an input drawn from it would shift that draw between mint and
# replay, since replay loads the stored input instead of rebuilding it.
def _image() -> Tensor:
    generator = torch.Generator().manual_seed(0)
    return torch.randint(0, 256, (2, 3, 16, 16), generator=generator, dtype=torch.uint8)


def _round_trip(module: nn.Module, image: Tensor) -> Tensor:
    assert isinstance(module, INVAE)
    latent = module.encode(image).sample()
    return torch.cat([latent.flatten(), module.decode(latent).flatten()])


def test_invae_round_trip_bfb() -> None:
    assert_bfb_against_golden(
        golden_dir=_TESTDATA,
        golden_name="invae",
        build_module=lambda: tiny().make(),
        build_input=_image,
        run=_round_trip,
    )


def test_satisfies_the_autoencoder_protocol() -> None:
    model = tiny().make()
    assert isinstance(model.encode(_image()), DiagonalGaussian)


def test_from_hf_parses_the_published_architecture_into_the_defaults() -> None:
    parsed, default = _published(), INVAE.Config()
    assert parsed.latent_shape() == (32, 16, 16)
    assert default.encoder is not None
    assert default.decoder is not None
    default.encoder.checkpoint = default.decoder.checkpoint = None
    assert parsed == default


def test_published_halves_load_exactly_the_published_tensors() -> None:
    """Remapped, the diffusers names and shapes are each half's state, strictly."""
    published = _published_keys()
    with torch.device("meta"):
        model = _published().make()
        state = {name: torch.empty(shape) for name, shape in published.items()}
    for half, module in (("encoder", model.encoder), ("decoder", model.decoder)):
        assert module is not None
        expected = {k: tuple(v.shape) for k, v in module.state_dict().items()}
        remapped = remap_hf_state_dict(state, half=half)
        assert {k: tuple(v.shape) for k, v in remapped.items()} == expected
    assert len(published) == 342


def test_the_published_checkpoint_is_pinned_on_both_halves() -> None:
    config = INVAE.Config()
    assert config.encoder is not None
    assert config.decoder is not None
    assert isinstance(config.encoder.checkpoint, HubFile.Config)
    assert config.encoder.checkpoint == config.decoder.checkpoint


def test_each_half_loads_from_a_file_holding_only_its_own_tensors(
    tmp_path: Path,
) -> None:
    """A half reads its own prefixes: the other half's absence never matters."""
    source = tiny().make()
    assert source.encoder is not None
    assert source.decoder is not None
    halves = {
        "encoder": _to_diffusers(source.encoder.state_dict(), half="encoder"),
        "decoder": _to_diffusers(source.decoder.state_dict(), half="decoder"),
    }
    config = tiny()
    assert config.encoder is not None
    assert config.decoder is not None
    for half, state in halves.items():
        save_file(state, str(tmp_path / f"{half}.safetensors"))
    config.encoder.checkpoint = LocalFile.Config(path=tmp_path / "encoder.safetensors")
    config.decoder.checkpoint = LocalFile.Config(path=tmp_path / "decoder.safetensors")
    loaded = config.make()
    image = _image()
    latent = source.encode(image).mode()
    assert torch.equal(loaded.encode(image).mode(), latent)
    assert torch.equal(loaded.decode(latent), source.decode(latent))


def test_either_half_can_be_left_out() -> None:
    encode_only, decode_only = tiny(), tiny()
    encode_only.decoder = None
    decode_only.encoder = None
    encoder = encode_only.make()
    decoder = decode_only.make()
    assert encoder.decoder is None
    assert decoder.encoder is None
    with pytest.raises(ValueError, match="without a decoder"):
        _ = encoder.decode(torch.zeros(3, 2, 8, 6))
    with pytest.raises(ValueError, match="without an encoder"):
        _ = decoder.encode(_image())


def test_leading_dimensions_pass_through() -> None:
    model = tiny().make()
    generator = torch.Generator().manual_seed(1)
    image = torch.randint(
        0,
        256,
        (2, 5, 3, 16, 12),
        generator=generator,
        dtype=torch.uint8,
    )
    flat = model.encode(image.flatten(0, 1)).mode()
    nested = model.encode(image).mode()
    assert nested.shape == (2, 5, *flat.shape[1:])
    assert torch.equal(nested.flatten(0, 1), flat)
    assert model.decode(nested).shape == (2, 5, 3, 16, 12)


def test_mode_is_the_mean_and_a_draw_follows_its_generator() -> None:
    posterior = tiny().make().encode(_image())
    first = posterior.sample(generator=torch.Generator().manual_seed(3))
    second = posterior.sample(generator=torch.Generator().manual_seed(3))
    assert torch.equal(first, second)
    assert not torch.equal(first, posterior.mode())
    assert torch.equal(posterior.mode(), posterior.mean)


def test_decode_returns_the_image_in_the_unit_interval() -> None:
    model = tiny().make()
    image = model.decode(model.encode(_image()).mode())
    assert image.shape == (2, 3, 16, 16)
    assert image.min() >= 0
    assert image.max() <= 1


def test_encode_refuses_a_float_image() -> None:
    with pytest.raises(TypeError, match="uint8"):
        _ = tiny().make().encode(torch.rand(2, 3, 16, 12))


def test_train_leaves_the_frozen_model_in_eval_mode() -> None:
    model = tiny().make()
    _ = model.train()
    assert not model.training
    assert not any(parameter.requires_grad for parameter in model.parameters())


def test_latent_norm_is_the_published_scale() -> None:
    norm = INVAE.Config().latent_norm
    assert isinstance(norm, ScaleLatents.Config)
    assert norm.scale == 0.3099


def test_rejects_an_image_size_the_stages_cannot_halve() -> None:
    config = tiny()
    config.image_size = 15
    with pytest.raises(ValueError, match="divisible by 2"):
        _ = config.make()


def test_rejects_halves_that_disagree_on_the_latent() -> None:
    config = tiny()
    assert config.decoder is not None
    config.decoder.channels_latent = 3
    with pytest.raises(ValueError, match="channels_latent"):
        _ = config.make()


def test_rejects_halves_that_disagree_on_the_grid() -> None:
    """A decoder with another stage count would return another image size."""
    config = tiny()
    assert config.decoder is not None
    config.decoder.channels_hidden = (2, 4, 4)
    config.decoder.attention = (False, True, True)
    with pytest.raises(ValueError, match="stages"):
        _ = config.make()


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("_class_name", "AutoencoderTiny", "AutoencoderKL"),
        ("act_fn", "gelu", "silu"),
        ("use_quant_conv", False, "use_quant_conv"),
        ("down_block_types", ["CrossAttnDownBlock2D"], "down_block_types"),
    ],
)
def test_from_hf_refuses_an_architecture_it_does_not_build(
    field: str,
    value: object,
    match: str,
) -> None:
    config = json_loads(
        (_TESTDATA / "e2e_invae_hf_config.json").read_text(),
        dict[str, object],
    )
    config[field] = value
    with pytest.raises(ValueError, match=match):
        _ = INVAE.Config.from_hf(config)


def test_load_reads_a_local_diffusers_directory(tmp_path: Path) -> None:
    """``load`` parses ``config.json`` and gives both halves the weights file."""
    source = tiny().make()
    assert source.encoder is not None
    assert source.decoder is not None
    hf = json_loads(
        (_TESTDATA / "e2e_invae_hf_config.json").read_text(),
        dict[str, object],
    )
    hf.update(
        block_out_channels=[2, 4],
        down_block_types=["DownEncoderBlock2D", "AttnDownEncoderBlock2D"],
        up_block_types=["AttnUpDecoderBlock2D", "UpDecoderBlock2D"],
        layers_per_block=1,
        latent_channels=2,
        norm_num_groups=2,
    )
    _ = (tmp_path / "config.json").write_text(json.dumps(hf))
    state = {
        **_to_diffusers(source.encoder.state_dict(), half="encoder"),
        **_to_diffusers(source.decoder.state_dict(), half="decoder"),
    }
    save_file(state, str(tmp_path / "diffusion_pytorch_model.safetensors"))
    loaded = INVAE.load(tmp_path)
    image = _image()
    assert torch.equal(loaded.encode(image).mode(), source.encode(image).mode())


def _to_diffusers(state: dict[str, Tensor], *, half: str) -> dict[str, Tensor]:
    """Invert :func:`remap_hf_state_dict` for one half, as diffusers saves it."""
    out: dict[str, Tensor] = {}
    for key, tensor in state.items():
        if key.startswith(("quant_conv.", "post_quant_conv.")):
            out[key] = tensor.contiguous()
            continue
        name = f"{half}.{key}".replace(".to_out.", ".to_out.0.")
        projection = any(f".to_{p}.weight" in name for p in ("q", "k", "v", "out.0"))
        out[name] = (tensor.flatten(1) if projection else tensor).contiguous()
    return out


def test_cost_matches_torch_for_a_forward_round_trip() -> None:
    analytical = assert_cost_matches_torch(
        tiny(),
        build_input=_image,
        run=_round_trip_decode,
        batch_size=2,
        dtype=None,
    )
    assert analytical["adjoint"].sum() == 0
    assert analytical["flops", "primal", "elementwise"].sum() > 0


def test_cost_prices_the_pixel_maps_at_the_configured_channels() -> None:
    """Grayscale in: the uint8 read and the float maps cover one channel, not three."""
    rgb, gray = tiny(), tiny()
    assert gray.encoder is not None
    gray.encoder.channels_in = 1
    rgb_cost, gray_cost = (
        cost(config, batch_size=2, dtype=None) for config in (rgb, gray)
    )
    pixels = 2 * 16 * 16
    key = ("bytes", "primal", "elementwise", torch.uint8)
    assert rgb_cost[key] - gray_cost[key] == 2 * pixels


def _round_trip_decode(module: nn.Module, image: Tensor) -> Tensor:
    assert isinstance(module, INVAE)
    return module.decode(module.encode(image).sample())


def test_cost_params_are_every_parameter() -> None:
    config = tiny()
    model = config.make()
    analytical = cost(config.copy_tree().finalize(), batch_size=2, dtype=None)
    assert analytical.params == sum(p.numel() for p in model.parameters())


def test_published_cost_is_linear_in_batch_size() -> None:
    """The pinned FLOPs are torch's for one 256px image through the published build."""
    config = INVAE.Config().copy_tree().finalize()
    one = cost(config, batch_size=1, dtype=torch.bfloat16)
    two = cost(config, batch_size=2, dtype=torch.bfloat16)
    assert one["flops", "primal", "matmul", torch.bfloat16] == 390_131_613_696
    assert one.params == 69_836_067
    assert two["flops"] == one["flops"].tile(2)


def test_invae_config_pprint() -> None:
    assert_pprint_golden(test_file=__file__, name="invae_config", config=INVAE.Config())


def test_config_needs_a_half_for_its_latent_shape() -> None:
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


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
