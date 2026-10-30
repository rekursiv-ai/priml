"""AutoencoderKL: the published architectures, their checkpoint names, and the arithmetic.

``testdata/kl.pt`` freezes the round trip at golden size, in INVAE's layout.
``testdata/kl_reference_{compvis,flux,flux2}.pt`` each hold a reference at its
pinned commit -- CompVis's latent-diffusion modules, BFL's FLUX.1 and FLUX.2
autoencoders -- at widths 2 and 4: its weights, an image, and its own moments,
normalized latent, and pixels. The references hardcode 32 GroupNorm groups,
so each was minted with its GroupNorms built at 2; their forward code ran
unmodified, under ``host_agnostic_numerics``. Replaying them here, through
``remap_ldm_state_dict``, proves the port against each reference's own
arithmetic: CompVis's attention for SD-1.x, SDPA for SDXL and FLUX. SDXL
differs from those two only in values.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final

import json

from safetensors.torch import save_file
from torch import Tensor, nn
from treekle import json_loads

import pytest
import torch

from priml.cost import cost
from priml.model.ae.checkpoint import HubFile, LocalFile
from priml.model.ae.distribution import DiagonalGaussian
from priml.model.ae.kl import (
    AutoencoderKL,
    LdmAttention,
    flux1,
    flux2,
    invae,
    remap_hf_state_dict,
    remap_ldm_state_dict,
    sd_ft_mse,
    sdxl,
    sdxl_fp16_fix,
)
from priml.model.ae.latent_norm import (
    BatchNormLatents,
    LatentNormalizer,
    ScaleLatents,
    ShiftScaleLatents,
)
from priml.model.attention.kernel import SdpaFused
from priml.model.conv import Conv2d
from priml.testing.bfb import assert_bfb_against_golden, host_agnostic_numerics
from priml.testing.cost import assert_cost_matches_torch
from priml.testing.golden import assert_pprint_golden, mismatches, read_tensors


if TYPE_CHECKING:
    from collections.abc import Callable


_CWD: Final = Path(__file__).resolve().parent

_TESTDATA: Final = _CWD / "testdata"


def tiny() -> AutoencoderKL.Config:
    """Return the published layout at golden size: widths 2 and 4, 16px."""
    config = invae()
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


def _published() -> AutoencoderKL.Config:
    return AutoencoderKL.Config.from_hf(
        json_loads(
            (_TESTDATA / "e2e_invae_hf_config.json").read_text(),
            dict[str, object],
        ),
    )


def _published_keys(listing: str) -> dict[str, tuple[int, ...]]:
    """Every tensor in a published file, by name, with its shape."""
    lines = (_TESTDATA / listing).read_text().splitlines()
    return {
        name: () if shape == "scalar" else tuple(int(n) for n in shape.split("x"))
        for name, shape in (line.split() for line in lines)
    }


# The posterior draw reads the global generator, which the harness reseeds before
# building the module; an input drawn from it would shift that draw between mint and
# replay, since replay loads the stored input instead of rebuilding it.
def _image() -> Tensor:
    generator = torch.Generator().manual_seed(0)
    return torch.randint(0, 256, (2, 3, 16, 16), generator=generator, dtype=torch.uint8)


def _round_trip(module: nn.Module, image: Tensor) -> Tensor:
    assert isinstance(module, AutoencoderKL)
    latent = module.encode(image).sample()
    return torch.cat([latent.flatten(), module.decode(latent).flatten()])


def test_round_trip_bfb() -> None:
    assert_bfb_against_golden(
        golden_dir=_TESTDATA,
        golden_name="kl",
        build_module=lambda: tiny().make(),
        build_input=_image,
        run=_round_trip,
    )


def test_satisfies_the_autoencoder_protocol() -> None:
    model = tiny().make()
    assert isinstance(model.encode(_image()), DiagonalGaussian)


def test_from_hf_parses_the_published_architecture_into_the_defaults() -> None:
    parsed, default = _published(), invae()
    assert parsed.latent_shape() == (32, 16, 16)
    assert default.encoder is not None
    assert default.decoder is not None
    default.encoder.checkpoint = default.decoder.checkpoint = None
    assert parsed == default


def test_published_halves_load_exactly_the_published_tensors() -> None:
    """Remapped, the diffusers names and shapes are each half's state, strictly."""
    published = _published_keys("e2e_invae_hf_keys.txt")
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
    config = invae()
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
    norm = invae().latent_norm
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
        ("mid_block_add_attention", False, "mid_block_add_attention"),
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
        _ = AutoencoderKL.Config.from_hf(config)


def test_from_hf_omits_the_projections_a_config_turns_off() -> None:
    config = json_loads(
        (_TESTDATA / "e2e_invae_hf_config.json").read_text(),
        dict[str, object],
    )
    config.update(use_quant_conv=False, use_post_quant_conv=False)
    parsed = AutoencoderKL.Config.from_hf(config)
    assert parsed.encoder is not None
    assert parsed.decoder is not None
    assert parsed.encoder.proj_out is None
    assert parsed.decoder.proj_in is None


def test_omitted_projections_build_no_modules_and_cost_nothing() -> None:
    """Without them the moments and the latent pass straight through."""
    with_proj, without = tiny(), tiny()
    assert without.encoder is not None
    assert without.decoder is not None
    without.encoder.proj_out = None
    without.decoder.proj_in = None
    model = without.make()
    assert model.encoder is not None
    assert model.decoder is not None
    assert model.encoder.quant_conv is None
    assert model.decoder.post_quant_conv is None
    # The two 1x1 convolutions, on the moments (2 * 2 channels) and the latent (2).
    removed = (4 * 4 + 4) + (2 * 2 + 2)
    full, bare = (
        cost(config.copy_tree().finalize(), batch_size=2, dtype=None)
        for config in (with_proj, without)
    )
    assert full.params - bare.params == removed
    assert bare.params == sum(p.numel() for p in model.parameters())
    image = _image()
    assert model.decode(model.encode(image).mode()).shape == (2, 3, 16, 16)


def test_projection_widths_follow_the_latent() -> None:
    config = tiny().copy_tree().finalize()
    assert config.encoder is not None
    assert config.decoder is not None
    proj_out, proj_in = config.encoder.proj_out, config.decoder.proj_in
    assert isinstance(proj_out, Conv2d.Config)
    assert isinstance(proj_in, Conv2d.Config)
    assert (proj_out.channels_in, proj_out.channels_out) == (4, 4)
    assert (proj_in.channels_in, proj_in.channels_out) == (2, 2)


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
    loaded = AutoencoderKL.load(tmp_path)
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
    assert isinstance(module, AutoencoderKL)
    return module.decode(module.encode(image).sample())


def test_cost_params_are_every_parameter() -> None:
    config = tiny()
    model = config.make()
    analytical = cost(config.copy_tree().finalize(), batch_size=2, dtype=None)
    assert analytical.params == sum(p.numel() for p in model.parameters())


def test_published_cost_is_linear_in_batch_size() -> None:
    """The pinned FLOPs are torch's for one 256px image through the published build."""
    config = invae().copy_tree().finalize()
    one = cost(config, batch_size=1, dtype=torch.bfloat16)
    two = cost(config, batch_size=2, dtype=torch.bfloat16)
    assert one["flops", "primal", "matmul", torch.bfloat16] == 390_131_613_696
    assert one.params == 69_836_067
    assert two["flops"] == one["flops"].tile(2)


def test_invae_config_pprint() -> None:
    assert_pprint_golden(test_file=__file__, name="invae_config", config=invae())


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


_REFERENCE_COMMITS: Final = {
    "compvis": "21f890f9da3cfbeaba8e2ac3c425ee9e998d5229",
    "flux": "802fb4713906133fcbd0d8dc5351620ca4773036",
    "flux2": "50fe5162777813d869182b139e83b10743caef15",
}


def _reference_layout(source: str) -> AutoencoderKL.Config:
    """Return the layout a reference golden was minted at: :func:`tiny`, 5 latents."""
    config = tiny()
    assert config.encoder is not None
    assert config.decoder is not None
    for half in (config.encoder, config.decoder):
        half.channels_latent = 5
        half.remap = remap_ldm_state_dict
        # BFL's stages never attend, and its attention is SDPA's.
        if source != "compvis":
            half.attention = (False, False)
            half.attn_kernel = SdpaFused.Config()
    if source == "flux":
        config.encoder.proj_out = None
        config.decoder.proj_in = None
    if source == "flux2":
        config.patch_size = 2
    return config


def _replay_reference(
    source: str,
    directory: Path,
    *,
    nudge: bool = False,
) -> list[str]:
    """Return how the port, loading a reference's tensors, misses its bits."""
    reference = read_tensors(_TESTDATA / f"kl_reference_{source}.pt")
    commit = reference.pop("source_commit").numpy().tobytes().decode()
    assert commit == _REFERENCE_COMMITS[source]
    # Inside (-1, 1), so decode's clamp leaves every compared bit visible.
    assert reference["decoded"].abs().max() < 1
    state = {
        key.removeprefix("state."): value
        for key, value in reference.items()
        if key.startswith("state.")
    }
    model = _reference_layout(source).make()
    assert model.encoder is not None
    assert model.decoder is not None
    model.encoder.load_state_dict(remap_ldm_state_dict(state, half="encoder"))
    model.decoder.load_state_dict(remap_ldm_state_dict(state, half="decoder"))
    if nudge:
        bias = model.decoder.conv_out.bias
        assert bias is not None
        bias[0] = torch.nextafter(bias[0], torch.tensor(torch.inf))
    # The normalizer each reference's ``encode`` applies; CompVis's applies none.
    norm: LatentNormalizer = ScaleLatents.Config().make()
    if source == "flux":
        norm = ShiftScaleLatents.Config(shift=0.1159, scale=0.3611).make()
    elif source == "flux2":
        stats = {key: state[key] for key in ("bn.running_mean", "bn.running_var")}
        save_file(stats, str(directory / "bn.safetensors"))
        norm = BatchNormLatents.Config(
            stats=LocalFile.Config(path=directory / "bn.safetensors"),
            eps=1e-4,
        ).make()
    image = reference["image"]
    with host_agnostic_numerics():
        actual = {
            "moments": model.encoder(image.float() / 127.5 - 1),
            "latent": norm.normalize(model.encode(image).mode()),
        }
        latent = reference.get("latent", actual["latent"])
        actual["decoded"] = model.decode(norm.denormalize(latent))
    expected = {
        "moments": reference["moments"],
        "latent": reference.get("latent", actual["latent"]),
        "decoded": ((reference["decoded"] + 1) / 2).clamp(0, 1),
    }
    return mismatches(expected, actual=actual)


@pytest.mark.parametrize("source", sorted(_REFERENCE_COMMITS))
def test_matches_the_reference_implementation(source: str, tmp_path: Path) -> None:
    report = _replay_reference(source, tmp_path)
    assert not report, report


def test_the_reference_replay_sees_one_ulp(tmp_path: Path) -> None:
    """One ULP on one decoder bias moves the pixels, and the comparison reports it."""
    report = _replay_reference("compvis", tmp_path, nudge=True)
    assert [line.split(":")[0] for line in report] == ["decoded"]


@pytest.mark.parametrize(
    ("factory", "listing", "count"),
    [
        (sd_ft_mse, "sd_vae_ft_mse_keys.txt", 250),
        # SDXL's and its float16 fix's files hold SD-1.x's names and shapes.
        (sdxl, "sd_vae_ft_mse_keys.txt", 250),
        (sdxl_fp16_fix, "sd_vae_ft_mse_keys.txt", 250),
        (flux1, "flux1_ae_keys.txt", 244),
        (flux2, "flux2_ae_keys.txt", 251),
    ],
    ids=["sd_ft_mse", "sdxl", "sdxl_fp16_fix", "flux1", "flux2"],
)
def test_published_ldm_halves_load_exactly_the_published_tensors(
    factory: Callable[[], AutoencoderKL.Config],
    listing: str,
    count: int,
) -> None:
    """Remapped, each file's names and shapes are each half's state, strictly."""
    published = _published_keys(listing)
    config = factory()
    assert config.encoder is not None
    assert config.decoder is not None
    config.encoder.checkpoint = config.decoder.checkpoint = None
    with torch.device("meta"):
        model = config.make()
        state = {name: torch.empty(shape) for name, shape in published.items()}
    for half, module in (("encoder", model.encoder), ("decoder", model.decoder)):
        assert module is not None
        expected = {k: tuple(v.shape) for k, v in module.state_dict().items()}
        remapped = remap_ldm_state_dict(state, half=half)
        assert {k: tuple(v.shape) for k, v in remapped.items()} == expected
    assert len(published) == count


@pytest.mark.parametrize(
    ("factory", "latent_shape"),
    [
        (sd_ft_mse, (4, 32, 32)),
        (sdxl, (4, 128, 128)),
        (sdxl_fp16_fix, (4, 128, 128)),
        (flux1, (16, 128, 128)),
        (flux2, (128, 64, 64)),
    ],
    ids=["sd_ft_mse", "sdxl", "sdxl_fp16_fix", "flux1", "flux2"],
)
def test_published_latents_and_pinned_checkpoints(
    factory: Callable[[], AutoencoderKL.Config],
    latent_shape: tuple[int, int, int],
) -> None:
    config = factory()
    assert config.latent_shape() == latent_shape
    assert config.encoder is not None
    assert config.decoder is not None
    assert isinstance(config.encoder.checkpoint, HubFile.Config)
    assert config.encoder.checkpoint == config.decoder.checkpoint
    assert config.encoder.checkpoint is not config.decoder.checkpoint


def test_published_latent_norms() -> None:
    norms = [factory().latent_norm for factory in (sd_ft_mse, sdxl, flux1, flux2)]
    sd, xl, one, two = norms
    assert sd == ScaleLatents.Config(scale=0.18215)
    assert xl == ScaleLatents.Config(scale=0.13025)
    assert one == ShiftScaleLatents.Config(shift=0.1159, scale=0.3611)
    assert isinstance(two, BatchNormLatents.Config)
    assert two.eps == 1e-4
    config = flux2()
    assert config.encoder is not None
    assert two.stats == config.encoder.checkpoint


_PUBLISHED_LDM: Final = {
    "sd_ft_mse": sd_ft_mse,
    "sdxl": sdxl,
    "sdxl_fp16_fix": sdxl_fp16_fix,
    "flux1": flux1,
    "flux2": flux2,
}


@pytest.mark.parametrize("name", sorted(_PUBLISHED_LDM))
def test_published_config_pprint(name: str) -> None:
    assert_pprint_golden(
        test_file=__file__,
        name=f"{name}_config",
        config=_PUBLISHED_LDM[name](),
    )


@pytest.mark.parametrize("source", ["flux", "flux2"])
def test_sdpa_and_folded_layouts_cost_matches_torch(source: str) -> None:
    analytical = assert_cost_matches_torch(
        _reference_layout(source),
        build_input=_image,
        run=_round_trip_decode,
        batch_size=2,
        dtype=None,
    )
    assert analytical["adjoint"].sum() == 0


def test_folding_needs_an_image_the_patch_divides() -> None:
    config = _reference_layout("flux2")
    config.image_size = 18
    with pytest.raises(ValueError, match="divisible by 4"):
        _ = config.make()


def test_rejects_a_patch_that_folds_nothing() -> None:
    config = _reference_layout("flux2")
    config.patch_size = 0
    with pytest.raises(ValueError, match="patch_size must be positive"):
        _ = config.make()


def test_ldm_attention_has_one_head() -> None:
    q = torch.randn(2, 5, 3, 4)
    with pytest.raises(ValueError, match="one head"):
        _ = LdmAttention()(q, q, q)


_QUERY: Final = torch.zeros(2, 5, 1, 4)


@pytest.mark.parametrize(
    "attend",
    [
        lambda: LdmAttention()(_QUERY, _QUERY, _QUERY, dropout_p=0.1),
        lambda: LdmAttention()(_QUERY, _QUERY, _QUERY, is_causal=True),
        lambda: LdmAttention()(_QUERY, _QUERY, _QUERY, attn_mask=torch.zeros(5, 5)),
        lambda: LdmAttention()(_QUERY, _QUERY, _QUERY, window=2),
        lambda: LdmAttention()(_QUERY, _QUERY, _QUERY, scale=0.5),
    ],
    ids=["dropout", "causal", "mask", "window", "scale"],
)
def test_ldm_attention_refuses_what_the_reference_does_not_compute(
    attend: Callable[[], Tensor],
) -> None:
    """A causal or scaled slot must not silently get full, fixed-scale attention."""
    with pytest.raises(ValueError, match="takes no"):
        _ = attend()


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
