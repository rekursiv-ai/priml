"""VTP: the published architectures, their checkpoint names, and the reference's bits.

``testdata/vtp_reference.pt`` holds MiniMax-AI/VTP's own outputs at commit
``5ce1eb6`` for the weights and input it stores, under
``host_agnostic_numerics``. Replaying it here proves the port against the reference. ``testdata/vtp.pt`` freezes the
port's own round trip.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import json

from safetensors.torch import save_file
from torch import Tensor, nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from treekle import json_loads

import pytest
import torch

from priml.cost import cost
from priml.model.ae.checkpoint import HubFile, LocalFile, UrlFile
from priml.model.ae.distribution import PointMass
from priml.model.ae.latent_norm import ChannelLatentStats
from priml.model.ae.vtp import (
    PUBLISHED,
    VTP,
    published_latent_norm,
    remap_hf_state_dict,
    vtp_base,
    vtp_large,
    vtp_small,
)
from priml.model.norm import RMSNorm
from priml.testing.bfb import assert_bfb_against_golden, host_agnostic_numerics
from priml.testing.cost import assert_cost_matches_torch
from priml.testing.golden import assert_pprint_golden, mismatches, read_tensors


_CWD: Final = Path(__file__).resolve().parent

_TESTDATA: Final = _CWD / "testdata"
_REFERENCE_COMMIT: Final = "5ce1eb67010fff3c1eed483352483be6a1838556"


def tiny() -> VTP.Config:
    """Return the published layout at golden size: 4px patches, 16 wide, 1 block."""
    config = VTP.Config()
    assert config.encoder is not None
    assert config.decoder is not None
    config.encoder.patch_size = 4
    config.encoder.channels_hidden = 16
    config.encoder.num_layers = 1
    config.encoder.heads = 2
    config.encoder.expansion = 1.0
    config.encoder.channels_out = 4
    config.encoder.checkpoint = None
    config.decoder.channels_in = 4
    config.decoder.channels_hidden = 16
    config.decoder.num_layers = 1
    config.decoder.heads = 2
    config.decoder.expansion = 1.0
    config.decoder.upscale_factor = 4
    config.decoder.checkpoint = None
    config.image_size = 8
    return config


def _json(name: str) -> dict[str, object]:
    return json_loads((_TESTDATA / name).read_text(), dict[str, object])


# A private generator keeps the input off the global one the harness reseeds. The
# sides differ, so a swapped height and width cannot pass.
def _image() -> Tensor:
    generator = torch.Generator().manual_seed(0)
    return torch.randint(0, 256, (2, 3, 8, 12), generator=generator, dtype=torch.uint8)


def _round_trip(module: nn.Module, image: Tensor) -> Tensor:
    assert isinstance(module, VTP)
    latent = module.encode(image).mode()
    return torch.cat([latent.flatten(), module.decode(latent).flatten()]).float()


def _halves(model: VTP) -> tuple[nn.Module, nn.Module]:
    assert model.encoder is not None
    assert model.decoder is not None
    return model.encoder, model.decoder


def _published_layout(state: dict[str, Tensor]) -> dict[str, Tensor]:
    """Rename a tiny model's state back to the published ``trunk.*``/``pixel_decoder.*``."""
    out: dict[str, Tensor] = {}
    renames = (
        ("rope.periods", "rope_embed.periods"),
        ("patch_embed.", "patch_embed.proj."),
        ("bottleneck.", "feature_bottleneck."),
    )
    for key, tensor in state.items():
        half, name = key.split(".", 1)
        prefix = {"encoder": "trunk.", "decoder": "pixel_decoder."}[half]
        for ours, published in renames:
            if name.startswith(ours):
                name = published + name.removeprefix(ours)
                break
        if ".mlp.up_proj." in name:
            gate, value = tensor.chunk(2)
            out[prefix + name.replace(".mlp.up_proj.", ".mlp.w1.")] = gate.contiguous()
            out[prefix + name.replace(".mlp.up_proj.", ".mlp.w2.")] = value.contiguous()
            continue
        out[prefix + name.replace(".mlp.down_proj.", ".mlp.w3.")] = tensor.contiguous()
    return out


def test_vtp_round_trip_bfb() -> None:
    assert_bfb_against_golden(
        golden_dir=_TESTDATA,
        golden_name="vtp",
        build_module=lambda: tiny().make(),
        build_input=_image,
        run=_round_trip,
    )


def test_matches_the_reference_implementation() -> None:
    """The raw decoder output is compared too: most golden pixels clamp."""
    reference = read_tensors(_TESTDATA / "vtp_reference.pt")
    commit = reference.pop("source_commit").numpy().tobytes().decode()
    assert commit == _REFERENCE_COMMIT
    published = {
        key.removeprefix("state."): value
        for key, value in reference.items()
        if key.startswith("state.")
    }
    model = tiny().make()
    encoder, decoder = _halves(model)
    encoder.load_state_dict(remap_hf_state_dict(published, half="encoder"))
    decoder.load_state_dict(remap_hf_state_dict(published, half="decoder"))
    with host_agnostic_numerics():
        latent = model.encode(reference["input"]).mode()
        actual = {
            "latent": latent,
            "pixels": decoder(latent),
            "image": model.decode(latent),
        }
    expected = {key: reference[key] for key in actual}
    assert mismatches(expected, actual=actual) == []


def test_satisfies_the_autoencoder_protocol_with_a_point_mass() -> None:
    model = tiny().make()
    assert isinstance(model.encode(_image()), PointMass)


@pytest.mark.parametrize(
    ("name", "factory"),
    [("small", vtp_small), ("base", vtp_base), ("large", vtp_large)],
)
def test_from_hf_parses_each_published_architecture_into_its_factory(
    name: str,
    factory: object,
) -> None:
    parsed = VTP.Config.from_hf(_json(f"vtp_{name}_config.json"))
    assert callable(factory)
    default = factory()
    assert isinstance(default, VTP.Config)
    assert default.encoder is not None
    assert default.decoder is not None
    default.encoder.checkpoint = default.decoder.checkpoint = None
    assert parsed == default
    assert parsed.latent_shape() == (64, 16, 16)


def test_from_hf_of_an_unpublished_architecture_has_no_statistics() -> None:
    hf = _json("vtp_small_config.json")
    hf["vision_depth"] = 3
    parsed = VTP.Config.from_hf(hf)
    with pytest.raises(ValueError, match="needs a stats file"):
        _ = parsed.latent_norm.make()


def test_published_halves_load_exactly_the_published_tensors() -> None:
    """VTP-Small's 477 tensors, text tower included: each half loads its own strictly."""
    lines = (_TESTDATA / "vtp_small_keys.txt").read_text().splitlines()
    dtypes = {"F32": torch.float32, "BF16": torch.bfloat16}
    published = {
        name: torch.empty(
            tuple(int(n) for n in shape.split("x") if n),
            dtype=dtypes[dtype],
            device="meta",
        )
        for name, shape, dtype in (line.split(" ") for line in lines)
    }
    assert len(published) == 477
    with torch.device("meta"):
        config = vtp_small()
        assert config.encoder is not None
        assert config.decoder is not None
        config.encoder.checkpoint = config.decoder.checkpoint = None
        encoder, decoder = _halves(config.make())
    for module, half in ((encoder, "encoder"), (decoder, "decoder")):
        expected = {
            k: (tuple(v.shape), v.dtype) for k, v in module.state_dict().items()
        }
        remapped = remap_hf_state_dict(published, half=half)
        assert {k: (tuple(v.shape), v.dtype) for k, v in remapped.items()} == expected


def test_remap_refuses_an_unknown_tensor_in_its_half() -> None:
    with pytest.raises(ValueError, match=r"trunk\.logit_bias"):
        _ = remap_hf_state_dict({"trunk.logit_bias": torch.ones(())}, half="encoder")


def test_each_half_loads_only_its_own_tensors(tmp_path: Path) -> None:
    """One published file; each half reads its prefix and ignores the text tower."""
    source = tiny().make()
    text_tower = {
        "text_transformer.resblocks.0.attn.in_proj_weight": torch.ones(2),
        "logit_scale": torch.ones(()),
    }
    path = tmp_path / "model.safetensors"
    save_file({**_published_layout(source.state_dict()), **text_tower}, str(path))
    config = tiny()
    assert config.encoder is not None
    assert config.decoder is not None
    config.encoder.checkpoint = LocalFile.Config(path=path)
    config.decoder.checkpoint = LocalFile.Config(path=path)
    loaded = config.make()
    assert mismatches(source.state_dict(), actual=loaded.state_dict()) == []


def test_load_reads_a_local_directory(tmp_path: Path) -> None:
    source = tiny().make()
    hf = _json("vtp_small_config.json")
    hf.update(
        vision_patch_size=4,
        vision_embed_dim=16,
        vision_depth=1,
        vision_num_heads=2,
        vision_mlp_ratio=1.0,
        vision_feature_bottleneck=4,
        decoder_embed_dim=16,
        decoder_depth=1,
        decoder_num_heads=2,
        decoder_ffn_ratio=1.0,
        image_size=8,
    )
    _ = (tmp_path / "config.json").write_text(json.dumps(hf))
    save_file(
        _published_layout(source.state_dict()),
        str(tmp_path / "model.safetensors"),
    )
    loaded = VTP.load(tmp_path)
    image = _image()
    assert torch.equal(loaded.encode(image).mode(), source.encode(image).mode())


def test_either_half_can_be_left_out() -> None:
    encode_only, decode_only = tiny(), tiny()
    encode_only.decoder = None
    decode_only.encoder = None
    encoder, decoder = encode_only.make(), decode_only.make()
    latent = encoder.encode(_image()).mode()
    assert decoder.decode(latent).shape == (2, 3, 8, 12)
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
        (2, 5, 3, 8, 12),
        generator=generator,
        dtype=torch.uint8,
    )
    nested = model.encode(image).mode()
    assert torch.equal(nested.flatten(0, 1), model.encode(image.flatten(0, 1)).mode())
    assert model.decode(nested).shape == (2, 5, 3, 8, 12)


def test_encode_matches_the_configured_latent_shape() -> None:
    config = tiny()
    # ``VTP.Config.latent_shape`` describes a square image of side ``image_size``.
    image = torch.zeros(2, 3, 8, 8, dtype=torch.uint8)
    latent = config.make().encode(image).mode()
    assert tuple(latent.shape[1:]) == config.latent_shape() == (4, 2, 2)


def test_decode_returns_the_image_shape_in_the_unit_interval() -> None:
    model = tiny().make()
    image = model.decode(model.encode(_image()).mode())
    assert image.shape == (2, 3, 8, 12)
    assert image.dtype == torch.float32
    assert image.min() >= 0
    assert image.max() <= 1


def test_dtype_autocast_reaches_the_encoder_and_decode_casts_back() -> None:
    config = tiny()
    config.dtype_autocast = torch.bfloat16
    model = config.make()
    latent = model.encode(_image()).mode()
    assert latent.dtype == torch.bfloat16
    assert tiny().make().encode(_image()).mode().dtype == torch.float32
    assert torch.equal(model.decode(latent), model.decode(latent.float()))


@pytest.mark.parametrize(
    ("mean", "std"),
    [
        ((0.5, 0.5, 0.5), (0.2, 0.0, 0.2)),
        ((0.5, 0.5, 0.5), (0.2, float("nan"), 0.2)),
        ((float("-inf"), 0.5, 0.5), (0.2, 0.2, 0.2)),
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


def test_encoder_refuses_a_bottleneck_as_wide_as_the_tokens() -> None:
    config = tiny().encoder
    assert config is not None
    config.channels_out = config.channels_hidden
    with pytest.raises(ValueError, match="channels_out 16 must differ"):
        _ = config.make()


def test_rotary_refuses_a_width_four_times_the_heads_does_not_divide() -> None:
    config = tiny().encoder
    assert config is not None
    config.heads = 3
    with pytest.raises(ValueError, match=r"divisible by 4 \* num_heads \(12\)"):
        _ = config.make()


def test_rejects_halves_that_disagree_on_the_latent() -> None:
    config = tiny()
    assert config.decoder is not None
    config.decoder.channels_in = 8
    with pytest.raises(ValueError, match="reads 8 channels"):
        _ = config.make()


def test_train_leaves_the_frozen_model_in_eval_mode() -> None:
    model = tiny().make()
    _ = model.train()
    assert not any(module.training for module in model.modules())
    assert not any(parameter.requires_grad for parameter in model.parameters())


def test_encode_rejects_a_side_off_the_patch_grid() -> None:
    with pytest.raises(ValueError, match="multiples of patch_size 4"):
        _ = tiny().make().encode(torch.zeros(2, 3, 8, 10, dtype=torch.uint8))


def test_rejects_an_image_size_off_the_patch_grid() -> None:
    config = tiny()
    config.image_size = 10
    with pytest.raises(ValueError, match="divisible by patch_size 4"):
        _ = config.make()


def test_encode_refuses_a_float_image() -> None:
    with pytest.raises(TypeError, match="uint8"):
        _ = tiny().make().encode(torch.rand(2, 3, 8, 12))


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("model_type", "clip", "'vtp'"),
        ("vision_ffn_layer", "mlp", "vision_ffn_layer"),
        ("decoder_norm_layer", "rmsnorm", "decoder_norm_layer"),
        ("vision_use_qk_norm", True, "vision_use_qk_norm"),
        ("decoder_init_values", 1e-5, "LayerScale"),
    ],
)
def test_from_hf_refuses_an_architecture_it_does_not_build(
    field: str,
    value: object,
    match: str,
) -> None:
    config = _json("vtp_large_config.json")
    config[field] = value
    with pytest.raises(ValueError, match=match):
        _ = VTP.Config.from_hf(config)


def test_latent_norm_standardizes_by_the_published_stats() -> None:
    norm = vtp_large().latent_norm
    assert isinstance(norm, ChannelLatentStats.Config)
    assert isinstance(norm.stats, UrlFile.Config)
    assert norm.stats.url.endswith(
        f"{_REFERENCE_COMMIT}/generation/latent_stats/vtp_l/latents_stats.pt",
    )


def test_each_size_pins_its_own_weights() -> None:
    repos: set[str] = set()
    for factory in (vtp_small, vtp_base, vtp_large):
        config = factory()
        assert config.encoder is not None
        assert isinstance(config.encoder.checkpoint, HubFile.Config)
        repos.add(config.encoder.checkpoint.repo_id)
    assert len(repos) == 3


def test_encoder_keeps_the_norms_in_the_weights_dtype_under_autocast() -> None:
    config = tiny().encoder
    assert config is not None
    analytical = config.cost(
        input_grid=8,
        batch_size=2,
        dtype=None,
        dtype_autocast=torch.bfloat16,
    )
    norm = RMSNorm.Config(16, eps=1e-5, elementwise_affine=True).cost(
        seq_len=5,
        batch_size=2,
        dtype=torch.float32,
    )
    # Two norms in the one block, then the final norm.
    assert (
        analytical["flops", "primal", "reduction", torch.float32]
        == 3 * norm["flops", "primal", "reduction", torch.float32]
    )
    assert analytical["matmul", torch.float32].sum() == 0


def test_encoder_cost_matches_torch() -> None:
    """The math backend runs attention as ``bmm``, which ``FlopCounterMode`` counts."""
    config = tiny().encoder
    assert config is not None
    # ``Encoder.Config.cost`` prices a square image of side ``input_grid``.
    with sdpa_kernel(SDPBackend.MATH):
        analytical = assert_cost_matches_torch(
            config,
            build_input=lambda: torch.randn(2, 3, 8, 8, requires_grad=True),
            input_grid=8,
            batch_size=2,
            dtype=None,
        )
    assert analytical["flops", "adjoint", "matmul"].sum() > 0


def test_decoder_cost_matches_torch() -> None:
    config = tiny().decoder
    assert config is not None
    # ``Decoder.Config.cost`` prices a square latent of side ``input_grid``.
    with sdpa_kernel(SDPBackend.MATH):
        analytical = assert_cost_matches_torch(
            config,
            build_input=lambda: torch.randn(3, 4, 2, 2, requires_grad=True),
            input_grid=2,
            batch_size=3,
            dtype=None,
        )
    assert analytical["flops", "adjoint", "matmul"].sum() > 0


def test_cost_is_one_frozen_round_trip() -> None:
    config = tiny()
    # ``VTP.Config.cost`` prices a square image of side ``image_size``.
    with sdpa_kernel(SDPBackend.MATH):
        analytical = assert_cost_matches_torch(
            config,
            build_input=lambda: torch.randint(0, 256, (2, 3, 8, 8), dtype=torch.uint8),
            run=_decode_encode,
            batch_size=2,
            dtype=None,
        )
    assert analytical["adjoint"].sum() == 0


def test_cost_charges_the_decoder_cast_under_autocast() -> None:
    config = tiny()
    config.dtype_autocast = torch.bfloat16
    assert config.encoder is not None
    assert config.decoder is not None
    analytical = cost(config, batch_size=2, dtype=None)
    parts = config.encoder.cost(
        input_grid=8,
        batch_size=2,
        dtype=None,
        dtype_autocast=torch.bfloat16,
    ) + config.decoder.cost(input_grid=2, batch_size=2, dtype=None)
    latents = 2 * 4 * 2 * 2
    assert (
        analytical["bytes", "primal", "elementwise", torch.bfloat16]
        == parts["bytes", "primal", "elementwise", torch.bfloat16] + 2 * latents
    )


def test_encoder_cost_rejects_a_side_off_the_patch_grid() -> None:
    config = tiny().encoder
    assert config is not None
    with pytest.raises(ValueError, match="divisible by patch_size 4"):
        _ = config.cost(input_grid=10, batch_size=2, dtype=None)


def _decode_encode(module: nn.Module, image: Tensor) -> Tensor:
    assert isinstance(module, VTP)
    return module.decode(module.encode(image).mode())


@pytest.mark.parametrize(
    ("name", "factory"),
    [("small", vtp_small), ("base", vtp_base), ("large", vtp_large)],
)
def test_vtp_config_pprint(name: str, factory: object) -> None:
    assert callable(factory)
    config = factory()
    assert isinstance(config, VTP.Config)
    assert_pprint_golden(test_file=__file__, name=f"vtp_{name}_pprint", config=config)


def test_config_needs_a_half() -> None:
    config = tiny()
    config.encoder = config.decoder = None
    with pytest.raises(ValueError, match="needs an encoder or a decoder"):
        _ = config.latent_shape()
    with pytest.raises(ValueError, match="needs an encoder or a decoder"):
        _ = config.make()


def test_a_decoder_alone_reads_its_latent_shape_from_the_upscale() -> None:
    config = tiny()
    config.encoder = None
    assert config.latent_shape() == (4, 2, 2)


def test_cost_needs_both_halves() -> None:
    config = tiny()
    config.decoder = None
    with pytest.raises(ValueError, match="prices both halves"):
        _ = config.cost(batch_size=1, dtype=None)


def test_load_of_a_published_repository_takes_its_statistics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, revision = PUBLISHED["small"][:2]

    monkeypatch.setattr(
        "priml.model.ae.vtp.read_json",
        _published_small_config,
    )
    monkeypatch.setattr(VTP.Config, "make", _built_config)
    loaded = VTP.load(repo, revision=revision)
    assert isinstance(loaded, VTP.Config)
    assert loaded.latent_norm == published_latent_norm("small")
    assert loaded.encoder is not None
    assert loaded.encoder.checkpoint == HubFile.Config(
        repo_id=repo,
        filename="model.safetensors",
        revision=revision,
    )


def _built_config(config: VTP.Config) -> VTP.Config:
    """Stand in for ``make``: return the config ``load`` built."""
    return config


def _published_small_config(source: object) -> dict[str, object]:
    """Stand in for ``read_json``: the pinned VTP-Small ``config.json``."""
    del source
    return _json("vtp_small_config.json")


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
