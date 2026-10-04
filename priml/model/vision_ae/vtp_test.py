"""Tests for the VTP port.

``testdata/vtp.pt`` is this port's own golden: randomized weights, input, and
output, each stored whole. ``testdata/vtp_reference.pt`` holds the reference's outputs for
those same weights and input, minted by ``scripts/reference_parity.py``
running MiniMax-AI/VTP at commit 5ce1eb6 (its ``VTPModel.get_reconstruction_latents`` and
``get_latents_decoded_images``, with torchvision's ``ToTensor``, ``Normalize``,
and the reference's inverse ``Normalize`` then clamp) under
``host_agnostic_numerics``; replaying it here needs nothing from the reference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from configgle.testing import assert_pprint_golden
from safetensors.torch import save_file
from torch import Tensor, nn
from torch.nn.attention import SDPBackend, sdpa_kernel

import pytest
import torch

from priml.cost import cost
from priml.model.norm import RMSNorm
from priml.model.vision_ae.checkpoint import LocalFile, UrlFile
from priml.model.vision_ae.custom_types import VariationalAutoencoder
from priml.model.vision_ae.latent_norm import ChannelLatentStats
from priml.model.vision_ae.vtp import (
    VTP,
    RopePositionEmbedding,
    init_weights_post,
    vtp_base,
    vtp_large,
    vtp_small,
)
from priml.testing.bfb import (
    assert_bfb_against_golden,
    host_agnostic_numerics,
    load_golden,
)
from priml.testing.cost import assert_cost_matches_torch
from priml.testing.golden import mismatches, read_tensors


_CWD: Final = Path(__file__).resolve().parent

_REFERENCE_COMMIT: Final = "5ce1eb67010fff3c1eed483352483be6a1838556"


def tiny() -> VTP.Config:
    """Return the published architecture at golden size: 4px patches, 16 wide, 1 block.

    Two heads of width 8 keep two rotary periods and the head split; the patch
    shrinks because the 16px patch embedding alone would overrun the golden.
    """
    config = VTP.Config()
    config.trunk.patch_size = 4
    config.trunk.channels_hidden = 16
    config.trunk.num_layers = 1
    config.trunk.heads = 2
    config.trunk.expansion = 1.0
    config.trunk.channels_out = 4
    config.pixel_decoder.channels_hidden = 16
    config.pixel_decoder.num_layers = 1
    config.pixel_decoder.heads = 2
    config.pixel_decoder.expansion = 1.0
    config.image_size = 8
    config.checkpoint = None
    return config


# A private generator keeps the input off the global one the harness reseeds. The
# sides differ, so a swapped height and width cannot pass.
def _image() -> Tensor:
    """Return a ``[2, 3, 8, 12]`` uint8 image from a private generator."""
    generator = torch.Generator().manual_seed(0)
    return torch.randint(0, 256, (2, 3, 8, 12), generator=generator, dtype=torch.uint8)


def _encode_decode(module: nn.Module, image: Tensor) -> Tensor:
    """Encode, decode, and join both."""
    assert isinstance(module, VTP)
    latent = module.encode(image)
    return torch.cat([latent.flatten(), module.decode(latent).flatten()]).float()


def test_vtp_encode_decode_bfb() -> None:
    assert_bfb_against_golden(
        golden_dir=_CWD / "testdata",
        golden_name="vtp",
        build_module=lambda: tiny().make(),
        build_input=_image,
        seed=0,
        run=_encode_decode,
    )


def test_vtp_matches_the_reference_bit_for_bit() -> None:
    """Replay the reference's outputs for the weights and input in ``vtp.pt``.

    The raw decoder output is compared besides the clamped image: with the
    golden's unit-variance weights most pixels clamp, which would hide the
    decoder's bits.
    """
    reference = read_tensors(_CWD / "testdata" / "vtp_reference.pt")
    assert (
        reference.pop("source_commit").numpy().tobytes().decode() == _REFERENCE_COMMIT
    )
    golden = load_golden(_CWD / "testdata" / "vtp.pt")
    model = tiny().make()
    model.load_state_dict(golden["state_dict"])
    image = golden["input"]
    assert isinstance(image, Tensor)
    with host_agnostic_numerics():
        latent = model.encode(image)
        port = {
            "latent": latent,
            "pixels": model.pixel_decoder(latent),
            "image": model.decode(latent),
        }
    assert mismatches(reference, actual=port) == []


def test_vtp_is_a_deterministic_autoencoder() -> None:
    model = tiny().make()
    assert not isinstance(model, VariationalAutoencoder)


def test_encode_matches_the_configured_latent_shape() -> None:
    config = tiny()
    image = torch.zeros(1, 3, config.image_size, config.image_size, dtype=torch.uint8)
    latent = config.make().encode(image)
    assert tuple(latent.shape[1:]) == config.latent_shape() == (4, 2, 2)


def test_decode_returns_the_image_shape_in_the_unit_interval() -> None:
    model = tiny().make()
    image = model.decode(model.encode(_image()))
    assert image.shape == (2, 3, 8, 12)
    assert image.dtype == torch.float32
    assert image.min() >= 0
    assert image.max() <= 1


def test_dtype_autocast_reaches_the_encoder() -> None:
    config = tiny()
    config.dtype_autocast = torch.bfloat16
    assert config.make().encode(_image()).dtype == torch.bfloat16
    assert tiny().make().encode(_image()).dtype == torch.float32


def test_decode_takes_the_latent_encode_returns_under_autocast() -> None:
    """The protocol's round trip holds: ``decode`` casts back to the weights' dtype."""
    config = tiny()
    config.dtype_autocast = torch.bfloat16
    model = config.make()
    latent = model.encode(_image())
    image = model.decode(latent)
    assert image.dtype == torch.float32
    assert torch.equal(image, model.decode(latent.float()))


def test_vtp_cost_charges_the_decoder_cast_under_autocast() -> None:
    config = tiny()
    config.dtype_autocast = torch.bfloat16
    finalized = config.copy_tree().finalize()
    analytical = cost(finalized, batch_size=2, dtype=None)
    parts = finalized.trunk.cost(
        image_size=8,
        batch_size=2,
        dtype=None,
        dtype_autocast=torch.bfloat16,
    ) + finalized.pixel_decoder.cost(latent_size=2, batch_size=2, dtype=None)
    latents = 2 * 4 * 2 * 2
    # The cast reads the bfloat16 latent once; its float32 write joins the pixel maps.
    assert (
        analytical["bytes", "primal", "elementwise", torch.bfloat16]
        == parts["bytes", "primal", "elementwise", torch.bfloat16] + 2 * latents
    )


def test_trunk_cost_keeps_the_norms_in_the_weights_dtype_under_autocast() -> None:
    """The reference's RMSNorm promotes to its float32 weight, as does the stream."""
    trunk = tiny().trunk
    analytical = trunk.cost(
        image_size=8,
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


def test_trunk_refuses_a_bottleneck_as_wide_as_the_tokens() -> None:
    config = tiny()
    config.trunk.channels_out = config.trunk.channels_hidden
    with pytest.raises(ValueError, match="channels_out 16 must differ"):
        _ = config.copy_tree().finalize()


def test_rotary_embedding_refuses_a_width_four_times_the_heads_does_not_divide() -> (
    None
):
    with pytest.raises(ValueError, match=r"divisible by 4 \* num_heads \(12\)"):
        _ = RopePositionEmbedding(16, num_heads=3)


def test_train_leaves_the_frozen_model_in_eval_mode() -> None:
    model = tiny().make()
    _ = model.train()
    assert not model.training
    assert not any(module.training for module in model.modules())
    assert not any(parameter.requires_grad for parameter in model.parameters())


def test_encode_rejects_a_side_off_the_patch_grid() -> None:
    image = torch.zeros(1, 3, 8, 10, dtype=torch.uint8)
    with pytest.raises(ValueError, match="multiples of patch_size 4"):
        _ = tiny().make().encode(image)


def test_rejects_an_image_size_off_the_patch_grid() -> None:
    config = tiny()
    config.image_size = 10
    with pytest.raises(ValueError, match="divisible by patch_size 4"):
        _ = config.copy_tree().finalize()


def test_registers_parameters_in_the_reference_order() -> None:
    """The order ``VTPModel`` registers, printed from the reference at this size."""
    trunk_block = [
        "norm1.weight",
        "attn.qkv.weight",
        "attn.qkv.bias",
        "attn.proj.weight",
        "attn.proj.bias",
        "norm2.weight",
        *(f"mlp.w{i}.{kind}" for i in (1, 2, 3) for kind in ("weight", "bias")),
    ]
    decoder_block = [
        "norm1.weight",
        "norm1.bias",
        *trunk_block[1:5],
        "norm2.weight",
        "norm2.bias",
        *trunk_block[6:],
    ]
    expected = [
        "trunk.cls_token",
        "trunk.mask_token",
        "trunk.patch_embed.proj.weight",
        "trunk.patch_embed.proj.bias",
        "trunk.rope_embed.periods",
        *(f"trunk.blocks.0.{key}" for key in trunk_block),
        "trunk.norm.weight",
        "trunk.feature_bottleneck.weight",
        "pixel_decoder.proj_in.weight",
        "pixel_decoder.proj_in.bias",
        "pixel_decoder.rope_embed.periods",
        *(f"pixel_decoder.blocks.0.{key}" for key in decoder_block),
        "pixel_decoder.norm.weight",
        "pixel_decoder.norm.bias",
        "pixel_decoder.proj_out.weight",
        "pixel_decoder.proj_out.bias",
    ]
    assert list(tiny().make().state_dict()) == expected


def test_published_architecture_keeps_the_checkpoint_parameter_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """VTP-Large-f16d64's ``trunk.*`` and ``pixel_decoder.*`` entries, pinned.

    The safetensors header lists 789 tensors: 295 trunk, 343 pixel decoder, and
    151 text-tower entries the loader discards. Every name, shape, and dtype of
    the 638 kept ones was checked equal to this build when the port landed;
    renaming a submodule would make ``load_state_dict`` reject the checkpoint.
    """

    # On meta tensors ``trunc_normal_``'s clamp runs through Python references,
    # 90% of this build; the names and shapes pinned here do not depend on it.
    def keep(tensor: Tensor, **_: object) -> Tensor:
        return tensor

    monkeypatch.setattr(nn.init, "trunc_normal_", keep)
    config = vtp_large()
    config.checkpoint = None
    with torch.device("meta"):
        state = config.make().state_dict()
    assert sum(key.startswith("trunk.") for key in state) == 295
    assert sum(key.startswith("pixel_decoder.") for key in state) == 343
    assert len(state) == 638
    shapes = {
        "trunk.cls_token": (1, 1, 1024),
        "trunk.mask_token": (1, 1024),
        "trunk.patch_embed.proj.weight": (1024, 3, 16, 16),
        "trunk.blocks.23.attn.qkv.weight": (3072, 1024),
        "trunk.blocks.23.mlp.w1.weight": (2736, 1024),
        "trunk.feature_bottleneck.weight": (64, 1024),
        "pixel_decoder.proj_in.weight": (1024, 64, 1, 1),
        "pixel_decoder.blocks.23.norm2.bias": (1024,),
        "pixel_decoder.proj_out.weight": (768, 1024, 1, 1),
    }
    assert {key: tuple(state[key].shape) for key in shapes} == shapes
    for key in ("trunk.rope_embed.periods", "pixel_decoder.rope_embed.periods"):
        assert state[key].dtype == torch.bfloat16
        assert tuple(state[key].shape) == (16,)


def _checkpoint(tmp_path: Path, extra: dict[str, Tensor]) -> tuple[VTP, VTP.Config]:
    """Save a tiny model plus ``extra`` as safetensors; return it and a config loading it."""
    source = tiny().make()
    path = tmp_path / "model.safetensors"
    save_file({**source.state_dict(), **extra}, path)
    config = tiny()
    config.checkpoint = LocalFile.Config(path=path)
    return source, config


def test_loader_discards_the_text_tower(tmp_path: Path) -> None:
    text_tower = {
        "text_transformer.resblocks.0.attn.in_proj_weight": torch.ones(2),
        "token_embedding.weight": torch.ones(2),
        "positional_embedding": torch.ones(2),
        "ln_final.weight": torch.ones(2),
        "ln_final.bias": torch.ones(2),
        "text_projection": torch.ones(2),
        "visual_proj.weight": torch.ones(2),
        "logit_scale": torch.ones(()),
    }
    source, config = _checkpoint(tmp_path, extra=text_tower)
    loaded = config.make().state_dict()
    # Each build draws its own initialization, so equality means it was loaded.
    assert not torch.equal(loaded["trunk.cls_token"], tiny().make().trunk.cls_token)
    assert mismatches(source.state_dict(), actual=loaded) == []
    assert loaded["trunk.rope_embed.periods"].dtype == torch.bfloat16


def test_loader_refuses_a_key_nothing_explains(tmp_path: Path) -> None:
    _, config = _checkpoint(tmp_path, extra={"logit_bias": torch.ones(())})
    with pytest.raises(ValueError, match="logit_bias"):
        _ = config.make()


def test_published_latent_is_64_channels_at_16x16() -> None:
    assert vtp_large().latent_shape() == (64, 16, 16)


def test_latent_norm_standardizes_by_the_published_stats() -> None:
    norm = vtp_large().latent_norm
    assert isinstance(norm, ChannelLatentStats.Config)
    assert norm.multiplier == 1.0
    assert isinstance(norm.stats, UrlFile.Config)
    assert norm.stats.url.endswith(
        f"{_REFERENCE_COMMIT}/generation/latent_stats/vtp_l/latents_stats.pt",
    )


def test_trunk_cost_matches_torch() -> None:
    """Forward and backward of the trunk at 8px: every product torch counts.

    The port calls ``scaled_dot_product_attention`` directly, and on CPU that
    dispatches to ``_scaled_dot_product_flash_attention_for_cpu``, which
    ``FlopCounterMode`` does not register. The math backend runs the same two
    products as ``bmm``, which it does, so the attention the cost counts is
    measured rather than excused.
    """
    config = tiny().copy_tree().finalize()
    with sdpa_kernel(SDPBackend.MATH):
        analytical = assert_cost_matches_torch(
            config.trunk,
            build_input=lambda: torch.randn(2, 3, 8, 8, requires_grad=True),
            image_size=8,
            batch_size=2,
            dtype=None,
        )
    assert analytical["flops", "adjoint", "matmul"].sum() > 0


def test_pixel_decoder_cost_matches_torch() -> None:
    config = tiny().copy_tree().finalize()
    with sdpa_kernel(SDPBackend.MATH):
        analytical = assert_cost_matches_torch(
            config.pixel_decoder,
            build_input=lambda: torch.randn(2, 4, 2, 2, requires_grad=True),
            latent_size=2,
            batch_size=2,
            dtype=None,
        )
    assert analytical["flops", "adjoint", "matmul"].sum() > 0


def test_vtp_cost_is_one_frozen_round_trip() -> None:
    """``decode(encode(x))`` forward only; the parameters are still all owned."""
    config = tiny()
    with sdpa_kernel(SDPBackend.MATH):
        analytical = assert_cost_matches_torch(
            config,
            build_input=lambda: torch.randint(0, 256, (2, 3, 8, 8), dtype=torch.uint8),
            run=_round_trip,
            batch_size=2,
            dtype=None,
        )
    assert analytical["adjoint"].sum() == 0
    finalized = config.copy_tree().finalize()
    parts = finalized.trunk.cost(
        image_size=8,
        batch_size=2,
        dtype=None,
    ) + finalized.pixel_decoder.cost(latent_size=2, batch_size=2, dtype=None)
    assert analytical["matmul"] == parts.only("primal")["matmul"]
    assert analytical.params == analytical.params_active == parts.params
    pixels = 2 * 3 * 8 * 8
    # Normalize is three maps and denormalize four, the clamp two-sided.
    assert (
        analytical["flops", "primal", "elementwise", torch.float32]
        - parts["flops", "primal", "elementwise", torch.float32]
        == 7 * pixels
    )


def test_trunk_cost_rejects_a_side_off_the_patch_grid() -> None:
    with pytest.raises(ValueError, match="divisible by patch_size 4"):
        _ = tiny().trunk.cost(image_size=10, batch_size=1, dtype=None)


def test_pixel_decoder_cost_needs_the_widths_vtp_propagates() -> None:
    with pytest.raises(ValueError, match="channels_in and upscale_factor"):
        _ = tiny().pixel_decoder.cost(latent_size=2, batch_size=1, dtype=None)


def _round_trip(module: nn.Module, image: Tensor) -> Tensor:
    """Reconstruct ``image``."""
    assert isinstance(module, VTP)
    return module.decode(module.encode(image))


def test_vtp_large_config_pprint() -> None:
    assert_pprint_golden(
        test_file=__file__,
        name="vtp_large_config",
        config=vtp_large(),
    )


def test_vtp_small_config_pprint() -> None:
    assert_pprint_golden(
        test_file=__file__,
        name="vtp_small_config",
        config=vtp_small(),
    )


def test_vtp_base_config_pprint() -> None:
    assert_pprint_golden(test_file=__file__, name="vtp_base_config", config=vtp_base())


@pytest.mark.network_huggingface
@pytest.mark.network_github
@pytest.mark.compute_large_fixture
def test_published_large_checkpoint_round_trips() -> None:
    """Reconstruct gray at the published checkpoint's trained resolution."""
    config = vtp_large()
    model = config.make()
    norm = config.latent_norm.make()
    image = torch.full(
        (1, 3, config.image_size, config.image_size),
        128,
        dtype=torch.uint8,
    )
    latent = model.encode(image)
    assert latent.shape == (1, *config.latent_shape())
    assert torch.isfinite(norm.normalize(latent)).all()
    decoded = model.decode(latent)
    assert (decoded - 128 / 255).abs().mean() < 0.05


def test_encode_refuses_a_float_image() -> None:
    """A ``[0, 1]`` float image would otherwise encode as near-black."""
    model = tiny().make()
    with pytest.raises(TypeError, match="uint8"):
        _ = model.encode(torch.rand(2, 3, 4, 5))


def test_initialization_ends_with_vtp_models_post_init_pass() -> None:
    """VTPModel's ``post_init`` redraws every linear weight after the modules draw theirs.

    ``scripts/reference_parity.py`` proves the result equal to a seeded
    ``VTPModel``; this pins the order without the reference.
    """
    config = tiny().copy_tree().finalize()
    torch.manual_seed(0)
    built = config.make()
    torch.manual_seed(0)
    expected = nn.Module()
    expected.trunk = config.trunk.make()
    expected.pixel_decoder = config.pixel_decoder.make()
    for module in expected.modules():
        init_weights_post(module)
    assert list(built.state_dict()) == list(expected.state_dict())
    report = mismatches(expected.state_dict(), actual=built.state_dict())
    assert not report, report


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
