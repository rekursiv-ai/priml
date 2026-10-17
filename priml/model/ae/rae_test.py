"""RAE: the published architectures, their checkpoint names, and the reference's bits.

``testdata/rae_reference.pt`` holds the reference at ``a4d18c4``: a tiny DINOv2-with-registers and decoder,
their weights, an input, and the reference's latent and unclamped pixels, the
reference's encoder pinned to HF's eager attention. Replaying it here proves
the port against the reference itself. ``testdata/rae.pt`` freezes the port's
own round trip.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import json

from safetensors.torch import save_file
from torch import Tensor, nn
from torch.nn import functional
from torch.utils.flop_counter import FlopCounterMode
from treekle import json_loads

import pytest
import torch

from priml.cost import cost
from priml.model.ae.checkpoint import HubFile, LocalFile
from priml.model.ae.latent_norm import ElementwiseLatentStats
from priml.model.ae.rae import (
    _DECODER_LAYER_NAMES,
    _ENCODER_LAYER_NAMES,
    _ENCODER_NAMES,
    RAE,
    Decoder,
    Encoder,
    rae_dinov2_base,
    remap_decoder_state_dict,
    remap_encoder_state_dict,
)
from priml.testing.bfb import assert_bfb_against_golden, host_agnostic_numerics
from priml.testing.cost import assert_cost_matches_torch
from priml.testing.golden import assert_pprint_golden, mismatches, read_tensors


_CWD: Final = Path(__file__).resolve().parent

_TESTDATA: Final = _CWD / "testdata"
_REFERENCE_COMMIT: Final = "a4d18c4db766419cbe7cb8c02cd9f7ceb0ec9041"


def tiny() -> RAE.Config:
    """Return the published recipe at golden size: width 8, 2x2 latent grid, 8px out.

    The encoder's position table is 3x3 against a 2x2 grid, and the input is
    16px against an 8px encoder side, so both resizes run as at 518 -> 224.
    """
    config = RAE.Config()
    assert config.encoder is not None
    assert config.decoder is not None
    config.encoder.channels_hidden = 8
    config.encoder.num_layers = 1
    config.encoder.heads = 2
    config.encoder.patch_size = 4
    config.encoder.image_size = 12
    config.encoder.checkpoint = None
    config.encoder_image_size = 8
    config.decoder.channels_in = 8
    config.decoder.num_patches = 4
    config.decoder.channels_hidden = 8
    config.decoder.channels_hidden_mlp = 16
    config.decoder.num_layers = 1
    config.decoder.heads = 2
    config.decoder.patch_size = 4
    config.decoder.checkpoint = None
    config.image_size = 8
    return config


def _json(name: str) -> dict[str, object]:
    return json_loads((_TESTDATA / name).read_text(), dict[str, object])


def _keys(listing: str) -> dict[str, tuple[int, ...]]:
    """Parse a ``name shape`` listing of a published checkpoint."""
    return {
        key: tuple(int(n) for n in shape.split("x"))
        for key, shape in (line.split() for line in listing.splitlines())
    }


# A private generator: the harness reseeds the global one before building the module,
# and replay loads the stored input rather than rebuilding it.
def _image() -> Tensor:
    """Return a uint8 image batch larger than the encoder side."""
    generator = torch.Generator().manual_seed(0)
    # ``testdata/rae.pt`` stores the upstream outputs for this input; a new shape
    # re-mints that golden on x86-64.
    return torch.randint(0, 256, (2, 3, 16, 16), generator=generator, dtype=torch.uint8)


def _round_trip(module: nn.Module, image: Tensor) -> Tensor:
    assert isinstance(module, RAE)
    latent = module.encode(image).mode()
    return torch.cat([latent.flatten(), module.decode(latent).flatten()])


def test_rae_round_trip_bfb() -> None:
    assert_bfb_against_golden(
        golden_dir=_TESTDATA,
        golden_name="rae",
        build_module=lambda: tiny().make(),
        build_input=_image,
        run=_round_trip,
    )


def test_matches_the_reference_implementation() -> None:
    reference = read_tensors(_TESTDATA / "rae_reference.pt")
    commit = reference.pop("source_commit").numpy().tobytes().decode()
    assert commit == _REFERENCE_COMMIT
    # Inside the unit interval, so the port's clamp leaves every compared bit visible.
    assert reference["decoded"].min() > 0
    assert reference["decoded"].max() < 1
    encoder = {
        key.removeprefix("state.encoder.encoder."): value
        for key, value in reference.items()
        if key.startswith("state.encoder.encoder.")
    }
    decoder = {
        key.removeprefix("state.decoder."): value
        for key, value in reference.items()
        if key.startswith("state.decoder.")
    }
    model = tiny().make()
    assert model.encoder is not None
    assert model.decoder is not None
    model.encoder.load_state_dict(remap_encoder_state_dict(encoder))
    model.decoder.load_state_dict(remap_decoder_state_dict(decoder))
    with host_agnostic_numerics():
        latent = model.encode(reference["image"]).mode()
        decoded = model.decode(latent)
    expected = {"latent": reference["latent"], "decoded": reference["decoded"]}
    report = mismatches(expected, actual={"latent": latent, "decoded": decoded})
    assert not report, report


def test_satisfies_the_autoencoder_protocol_with_a_point_mass() -> None:
    model = tiny().make()
    code = model.encode(_image())
    assert code.sample() is code.mode()


def test_from_hf_parses_the_published_architecture_into_the_defaults() -> None:
    parsed = RAE.Config.from_hf(
        _json("dinov2_with_registers_base_config.json"),
        _json("rae_vitxl_decoder_config.json"),
        decoder_patch_size=16,
    )
    default = rae_dinov2_base()
    assert default.encoder is not None
    assert default.decoder is not None
    default.encoder.checkpoint = default.decoder.checkpoint = None
    assert parsed == default
    assert parsed.latent_shape() == (768, 16, 16)


def test_published_halves_load_exactly_the_published_tensors() -> None:
    """Remapped, each published file's names and shapes are its half's state, strictly."""
    with torch.device("meta"):
        config = rae_dinov2_base()
        assert config.encoder is not None
        assert config.decoder is not None
        config.encoder.checkpoint = config.decoder.checkpoint = None
        model = config.make()
    for module, remap, listing, count in (
        (
            model.encoder,
            remap_encoder_state_dict,
            (_TESTDATA / "dinov2_with_registers_base_keys.txt").read_text(),
            224,
        ),
        (
            model.decoder,
            remap_decoder_state_dict,
            (_TESTDATA / "rae_vitxl_n08_keys.txt").read_text(),
            456,
        ),
    ):
        assert module is not None
        published = _keys(listing)
        assert len(published) == count
        state = {
            name: torch.empty(shape, device="meta") for name, shape in published.items()
        }
        expected = {k: tuple(v.shape) for k, v in module.state_dict().items()}
        assert {k: tuple(v.shape) for k, v in remap(state).items()} == expected


def test_encoder_remap_refuses_an_unknown_tensor() -> None:
    with pytest.raises(KeyError, match=r"pooler\.dense\.weight"):
        _ = remap_encoder_state_dict({"pooler.dense.weight": torch.zeros(2, 3)})


def test_decoder_remap_refuses_an_unknown_layer_tensor() -> None:
    with pytest.raises(KeyError, match=r"decoder_layers\.0\.unknown\.weight"):
        _ = remap_decoder_state_dict(
            {"decoder_layers.0.unknown.weight": torch.zeros(2)},
        )


def test_the_published_checkpoints_are_pinned() -> None:
    config = rae_dinov2_base()
    assert config.encoder is not None
    assert config.decoder is not None
    assert isinstance(config.encoder.checkpoint, HubFile.Config)
    assert isinstance(config.decoder.checkpoint, HubFile.Config)
    assert config.encoder.checkpoint.repo_id == "facebook/dinov2-with-registers-base"


def test_either_half_can_be_left_out() -> None:
    encode_only, decode_only = tiny(), tiny()
    encode_only.decoder = None
    decode_only.encoder = None
    encoder, decoder = encode_only.make(), decode_only.make()
    latent = encoder.encode(_image()).mode()
    assert decoder.decode(latent).shape == (2, 3, 8, 8)
    with pytest.raises(ValueError, match="without a decoder"):
        _ = encoder.decode(latent)
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
    nested = model.encode(image).mode()
    assert torch.equal(nested.flatten(0, 1), model.encode(image.flatten(0, 1)).mode())
    assert model.decode(nested).shape == (2, 5, 3, 8, 8)


def test_encode_matches_the_configured_latent_shape() -> None:
    config = tiny()
    latent = config.make().encode(_image()).mode()
    assert tuple(latent.shape[1:]) == config.latent_shape()


def test_decode_returns_the_decoder_size_in_the_unit_interval() -> None:
    model = tiny().make()
    image = model.decode(model.encode(_image()).mode())
    assert image.shape == (2, 3, 8, 8)
    assert image.min() >= 0
    assert image.max() <= 1


@pytest.mark.parametrize("shape", [(2, 8, 3, 3), (2, 8, 1, 4), (2, 16, 2, 2)])
def test_decode_rejects_a_latent_shape_the_model_was_not_built_for(
    shape: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError, match=r"shape \(8, 2, 2\), got"):
        _ = tiny().make().decode(torch.zeros(shape))


def test_decoder_rejects_a_token_count_it_was_not_built_for() -> None:
    decoder = tiny().make().decoder
    assert decoder is not None
    with pytest.raises(ValueError, match="built for 4 latent tokens, got 9"):
        _ = decoder(torch.zeros(2, 9, 8))


@pytest.mark.parametrize("num_patches", [0, -1, 8])
def test_decoder_refuses_a_patch_count_that_is_not_a_positive_square(
    num_patches: int,
) -> None:
    config = tiny().decoder
    assert config is not None
    config.num_patches = num_patches
    with pytest.raises(ValueError, match="must be a positive square"):
        _ = config.make()


def test_attention_refuses_heads_that_do_not_divide_the_width() -> None:
    config = tiny().encoder
    assert config is not None
    config.heads = 3
    with pytest.raises(ValueError, match="3 heads do not divide width 8"):
        _ = config.make()


def test_train_leaves_the_frozen_model_in_eval_mode() -> None:
    model = tiny().make()
    _ = model.train()
    assert not model.training
    assert not any(parameter.requires_grad for parameter in model.parameters())


def test_rejects_an_image_size_the_decoder_does_not_produce() -> None:
    config = tiny()
    config.image_size = 16
    with pytest.raises(ValueError, match="decoder patch 4 times the latent grid 2"):
        _ = config.make()


def test_rejects_an_encoder_side_the_patch_does_not_divide() -> None:
    config = tiny()
    config.encoder_image_size = 10
    with pytest.raises(ValueError, match="divisible by the encoder patch 4"):
        _ = config.make()


def test_rejects_a_decoder_built_for_another_encoder() -> None:
    config = tiny()
    assert config.decoder is not None
    config.decoder.channels_in = 16
    with pytest.raises(ValueError, match="16-wide tokens"):
        _ = config.make()


@pytest.mark.parametrize(
    ("mean", "std"),
    [
        ((0.5, 0.5, 0.5), (0.2, 0.0, 0.2)),
        ((0.5, 0.5, 0.5), (0.2, float("nan"), 0.2)),
        ((0.5, float("inf"), 0.5), (0.2, 0.2, 0.2)),
    ],
    ids=["zero-std", "nan-std", "inf-mean"],
)
def test_rejects_pixel_statistics_that_would_make_latents_nonfinite(
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> None:
    config = tiny()
    config.pixel_mean = mean
    config.pixel_std = std
    with pytest.raises(ValueError, match="finite pixel statistics"):
        _ = config.make()


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("model_type", "dinov2", "dinov2_with_registers"),
        ("use_swiglu_ffn", True, "SwiGLU"),
        ("qkv_bias", False, "Bias-free"),
        ("hidden_act", "silu", "exact GELU"),
    ],
)
def test_encoder_from_hf_refuses_an_architecture_it_does_not_build(
    field: str,
    value: object,
    match: str,
) -> None:
    config = _json("dinov2_with_registers_base_config.json")
    config[field] = value
    with pytest.raises(ValueError, match=match):
        _ = Encoder.Config.from_hf(config)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("model_type", "vit", "vit_mae"),
        ("hidden_act", "relu", "exact GELU"),
        ("qkv_bias", False, "Bias-free"),
    ],
)
def test_decoder_from_hf_refuses_an_architecture_it_does_not_build(
    field: str,
    value: object,
    match: str,
) -> None:
    config = _json("rae_vitxl_decoder_config.json")
    config[field] = value
    with pytest.raises(ValueError, match=match):
        _ = Decoder.Config.from_hf(config, patch_size=16)


def test_encoder_runs_the_injected_activation() -> None:
    config = tiny()
    reference = config.make()
    assert config.encoder is not None
    config.encoder.activation = functional.silu
    swapped = config.make()
    swapped.load_state_dict(reference.state_dict())
    image = _image()
    assert not torch.equal(swapped.encode(image).mode(), reference.encode(image).mode())


def test_encoder_cost_matches_torch() -> None:
    """Forward and adjoint, at an input that interpolates the 3x3 position table."""
    config = tiny().encoder
    assert config is not None
    analytical = assert_cost_matches_torch(
        config,
        build_input=lambda: torch.randn(2, 3, 8, 12, requires_grad=True),
        input_grid=(8, 12),
        batch_size=2,
        dtype=None,
    )
    assert analytical["adjoint"].sum() > 0
    # The mask token is owned but never read.
    assert analytical.params - analytical.params_active == 8


def test_encoder_cost_matches_torch_where_the_resize_is_skipped() -> None:
    """A 12px input is the 3x3 table's own grid, so no resize runs or is priced."""
    config = tiny().encoder
    assert config is not None
    # ``Encoder._positions`` skips the resize only on its square position grid.
    _ = assert_cost_matches_torch(
        config,
        build_input=lambda: torch.randn(2, 3, 12, 12, requires_grad=True),
        input_grid=12,
        batch_size=2,
        dtype=None,
    )
    bfloat16 = cost(config, input_grid=12, batch_size=2, dtype=torch.bfloat16)
    assert bfloat16[torch.float32].sum() == 0


def test_encoder_cost_resizes_the_position_table_in_float32() -> None:
    config = tiny().encoder
    assert config is not None
    analytical = cost(config, input_grid=8, batch_size=2, dtype=torch.bfloat16)
    # Antialiased 3 -> 2 reads three taps per axis, five operations per output:
    # a width pass over 8 x 3 x 2 values, then a height pass over 8 x 2 x 2.
    assert analytical["flops", "primal", "elementwise", torch.float32] == 5 * (48 + 32)
    # The resample moves 72 + 2 * 48 + 32 values; the casts write the 9-position
    # table and read the 4-patch grid, 8 channels each.
    assert analytical["bytes", "primal", "elementwise", torch.float32] == 4 * (
        200 + 13 * 8
    )


def test_encoder_cost_refuses_an_unpriced_activation() -> None:
    config = tiny().encoder
    assert config is not None
    config.activation = _unpriced
    with pytest.raises(TypeError):
        _ = cost(config, input_grid=8, batch_size=2, dtype=None)


def _unpriced(x: Tensor) -> Tensor:
    """Return ``x``: an activation with no cost model."""
    return x


def test_decoder_cost_matches_torch() -> None:
    config = tiny().decoder
    assert config is not None
    analytical = assert_cost_matches_torch(
        config,
        build_input=lambda: torch.randn(2, 4, 8, requires_grad=True),
        batch_size=2,
        dtype=None,
    )
    assert analytical["adjoint"].sum() > 0


# FlopCounterMode directly rather than ``assert_cost_matches_torch``: ``decode``
# hands the decoder a transposed latent, which torch's ``linear`` runs as a
# ``bmm`` against the weight expanded per image, so the dispatched matmul bytes
# re-read that weight once per image where the analytical convention reads it
# once. The FLOPs, what is gated here, are the same either way.
@pytest.mark.parametrize("decoder_patch", [4, 8], ids=["same-side", "resized"])
def test_rae_round_trip_cost_matches_torch(decoder_patch: int) -> None:
    config = tiny()
    assert config.decoder is not None
    config.decoder.patch_size = decoder_patch
    config.image_size = 2 * decoder_patch
    analytical = cost(config, batch_size=2, dtype=None)
    model = config.make()
    side = config.image_size
    image = torch.randint(
        0,
        256,
        (2, 3, side, side),
        generator=torch.Generator().manual_seed(0),
        dtype=torch.uint8,
    )
    with FlopCounterMode(display=False) as counter:
        _ = model.decode(model.encode(image).mode())
    assert analytical["flops", "matmul"].sum() == counter.get_total_flops()
    assert analytical.params == sum(p.numel() for p in model.parameters())
    assert analytical["adjoint"].sum() == 0


def test_latent_norm_is_the_published_imagenet_stats() -> None:
    norm = rae_dinov2_base().latent_norm
    assert isinstance(norm, ElementwiseLatentStats.Config)
    assert norm.eps == 1e-5
    assert isinstance(norm.stats, HubFile.Config)
    assert norm.stats.filename == "stats/dinov2/wReg_base/imagenet1k/stat.pt"


def test_rae_dinov2_base_config_pprint() -> None:
    assert_pprint_golden(
        test_file=__file__,
        name="rae_config",
        config=rae_dinov2_base(),
    )


def test_encode_refuses_a_float_image() -> None:
    with pytest.raises(TypeError, match="uint8"):
        _ = tiny().make().encode(torch.rand(2, 3, 4, 5))


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


def test_halves_load_their_checkpoints_and_load_reads_local_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Published names in, the same weights out, each half from its own file."""
    source = tiny().make()
    assert source.encoder is not None
    assert source.decoder is not None
    encoder_dir, decoder_dir = tmp_path / "encoder", tmp_path / "decoder"
    decoder_file = decoder_dir / "decoders/dinov2/wReg_base/ViTXL_n08/model.pt"
    decoder_file.parent.mkdir(parents=True)
    encoder_dir.mkdir()
    save_file(
        {
            **_published_encoder(source.encoder.state_dict()),
            "layernorm.weight": torch.ones(8),
            "layernorm.bias": torch.zeros(8),
        },
        str(encoder_dir / "model.safetensors"),
    )
    torch.save(_published_decoder(source.decoder.state_dict()), decoder_file)
    config = tiny()
    assert config.encoder is not None
    assert config.decoder is not None
    config.encoder.checkpoint = LocalFile.Config(path=encoder_dir / "model.safetensors")
    config.decoder.checkpoint = LocalFile.Config(path=decoder_file)
    loaded = config.make()
    assert mismatches(source.state_dict(), actual=loaded.state_dict()) == []

    _ = (encoder_dir / "config.json").write_text(
        json.dumps(_json("dinov2_with_registers_base_config.json")),
    )
    monkeypatch.setattr(RAE.Config, "make", _built_config)
    built = RAE.load(encoder_dir, decoder_dir)
    assert isinstance(built, RAE.Config)
    assert built.encoder is not None
    assert built.decoder is not None
    assert built.encoder.checkpoint == LocalFile.Config(
        path=encoder_dir / "model.safetensors",
    )
    assert built.decoder.checkpoint == LocalFile.Config(path=decoder_file)


def _published_encoder(state: dict[str, Tensor]) -> dict[str, Tensor]:
    """Rename an :class:`RaeEncoder` state to the published DINOv2 names."""
    inverse = {v: k for k, v in _ENCODER_NAMES.items()}
    layer = {v: k for k, v in _ENCODER_LAYER_NAMES.items() if "_proj" not in k}
    out: dict[str, Tensor] = {}
    for key, tensor in state.items():
        if key in inverse:
            out[inverse[key]] = tensor
            continue
        _, index, rest = key.split(".", 2)
        out[f"encoder.layer.{index}.{layer[rest]}"] = tensor
    return out


def _published_decoder(state: dict[str, Tensor]) -> dict[str, Tensor]:
    """Rename a :class:`Decoder` state to the published ViTMAE names."""
    layer = {v: k for k, v in _DECODER_LAYER_NAMES.items()}
    out: dict[str, Tensor] = {}
    for key, tensor in state.items():
        name = key
        if key.startswith("decoder_layers."):
            _, index, rest = key.split(".", 2)
            name = f"decoder_layers.{index}.{layer[rest]}"
        out[name] = tensor
    return out


def _built_config(config: RAE.Config) -> RAE.Config:
    """Stand in for ``make``: return the config ``load`` built."""
    return config


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
