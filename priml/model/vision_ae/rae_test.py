"""Tests for the RAE wrapper.

``testdata/rae_reference.pt`` is minted by ``scripts/reference_parity.py`` from
the reference clone at ``a4d18c4``, which that run also checks function by
function: its ``stage1.RAE`` loaded a tiny
DINOv2-with-registers through ``from_pretrained`` and a tiny decoder through its
own ``GeneralDecoder``, while the port loaded the same two files strictly, and
the reference's ``encode``/``decode`` ran under ``host_agnostic_numerics``. The
record holds the tiny weights, the input, and the reference's latent and
(unclamped) pixels. The reference's encoder attention was pinned to HF's eager
kernel, the port's; its ``from_pretrained`` default, ``sdpa``, moves 6 of the 64
latent values by up to 3 float32 ULP.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Final

from configgle.testing import assert_pprint_golden
from torch import Tensor, nn
from torch.nn import functional
from torch.utils.flop_counter import FlopCounterMode

import pytest
import torch

from priml.cost import cost
from priml.model.vision_ae.checkpoint import HubFile
from priml.model.vision_ae.custom_types import Autoencoder, VariationalAutoencoder
from priml.model.vision_ae.latent_norm import ElementwiseLatentStats
from priml.model.vision_ae.rae import RAE, Dinov2WithRegisters, rae_dinov2_base
from priml.testing.bfb import assert_bfb_against_golden, host_agnostic_numerics
from priml.testing.cost import assert_cost_matches_torch
from priml.testing.golden import mismatches, read_tensors


# Imported at collection: the model class costs ~1 s on first import, which would
# otherwise land inside whichever test builds the encoder first.
pytest.importorskip(
    "transformers.models.dinov2_with_registers.modeling_dinov2_with_registers",
)

_CWD: Final = Path(__file__).resolve().parent


def tiny() -> RAE.Config:
    """Return the published recipe at golden size: width 8, 2x2 latent grid, 8px out.

    The encoder's position table is 3x3 against a 2x2 grid, and the input is
    16px against an 8px encoder side, so both interpolations run as they do at
    518 -> 224.
    """
    config = RAE.Config()
    encoder = config.encoder
    assert isinstance(encoder, Dinov2WithRegisters.Config)
    encoder.channels_hidden = 8
    encoder.num_layers = 1
    encoder.heads = 2
    encoder.patch_size = 4
    encoder.image_size = 12
    encoder.checkpoint = None
    config.encoder_image_size = 8
    config.decoder.channels_hidden = 8
    config.decoder.channels_hidden_mlp = 16
    config.decoder.num_layers = 1
    config.decoder.heads = 2
    config.decoder.patch_size = 4
    config.decoder.checkpoint = None
    config.image_size = 8
    return config


# A private generator: the harness reseeds the global one before building the module,
# and replay loads the stored input rather than rebuilding it.
def _image() -> Tensor:
    """Return a uint8 image batch larger than the encoder side."""
    generator = torch.Generator().manual_seed(0)
    return torch.randint(0, 256, (2, 3, 16, 16), generator=generator, dtype=torch.uint8)


def _encode_decode(module: nn.Module, image: Tensor) -> Tensor:
    """Encode, decode, and join both."""
    assert isinstance(module, RAE)
    latent = module.encode(image)
    return torch.cat([latent.flatten(), module.decode(latent).flatten()]).float()


def test_rae_encode_decode_bfb() -> None:
    assert_bfb_against_golden(
        golden_dir=_CWD / "testdata",
        golden_name="rae",
        build_module=lambda: tiny().make(),
        build_input=_image,
        seed=0,
        run=_encode_decode,
    )


def test_matches_the_reference_implementation() -> None:
    reference = read_tensors(_CWD / "testdata" / "rae_reference.pt")
    # Inside the unit interval, so the port's clamp leaves every compared bit visible.
    assert reference["decoded"].min() > 0
    assert reference["decoded"].max() < 1
    model = tiny().make()
    model.load_state_dict(
        {
            key.removeprefix("state."): value
            for key, value in reference.items()
            if key.startswith("state.")
        },
    )
    with host_agnostic_numerics():
        latent = model.encode(reference["image"])
        decoded = model.decode(latent)
    expected = {"latent": reference["latent"], "decoded": reference["decoded"]}
    report = mismatches(expected, {"latent": latent, "decoded": decoded})
    assert not report, report


def test_rae_is_a_deterministic_autoencoder() -> None:
    model = tiny().make()
    assert isinstance(model, Autoencoder)
    assert not isinstance(model, VariationalAutoencoder)


def test_encode_matches_the_configured_latent_shape() -> None:
    config = tiny()
    latent = config.make().encode(_image())
    assert tuple(latent.shape[1:]) == config.latent_shape()


def test_decode_returns_the_decoder_size_in_the_unit_interval() -> None:
    model = tiny().make()
    image = model.decode(model.encode(_image()))
    assert image.shape == (2, 3, 8, 8)
    assert image.min() >= 0
    assert image.max() <= 1


def test_decode_rejects_a_grid_the_decoder_was_not_built_for() -> None:
    model = tiny().make()
    with pytest.raises(ValueError, match="built for 4 latent tokens, got 9"):
        _ = model.decode(torch.zeros(1, 8, 3, 3))


def test_train_leaves_the_frozen_model_in_eval_mode() -> None:
    model = tiny().make()
    _ = model.train()
    assert not model.training
    assert not any(parameter.requires_grad for parameter in model.parameters())


def test_rejects_an_image_size_the_decoder_does_not_produce() -> None:
    config = tiny()
    config.image_size = 16
    with pytest.raises(ValueError, match="decoder patch 4 times the latent grid 2"):
        _ = config.copy_tree().finalize()


def test_rejects_an_encoder_side_the_patch_does_not_divide() -> None:
    config = tiny()
    config.encoder_image_size = 10
    with pytest.raises(ValueError, match="divisible by the encoder patch 4"):
        _ = config.copy_tree().finalize()


def test_decoder_takes_its_width_and_token_count_from_the_encoder() -> None:
    config = tiny()
    config.decoder.channels_in = 16
    finalized = config.copy_tree().finalize()
    assert finalized.decoder.channels_in == 8
    assert finalized.decoder.num_patches == 4


def test_encoder_cost_matches_torch() -> None:
    """Forward and adjoint, at an input that interpolates the 3x3 position table."""
    config = tiny().copy_tree().finalize()
    assert isinstance(config.encoder, Dinov2WithRegisters.Config)
    analytical = assert_cost_matches_torch(
        config.encoder,
        build_input=lambda: torch.randn(2, 3, 8, 8, requires_grad=True),
        input_grid=8,
        batch_size=2,
        dtype=None,
    )
    assert analytical["adjoint"].sum() > 0
    # The mask token is owned but never read.
    assert analytical.params - analytical.params_active == 8


def test_encoder_cost_refuses_an_unpriced_activation() -> None:
    config = tiny().copy_tree().finalize()
    assert isinstance(config.encoder, Dinov2WithRegisters.Config)
    config.encoder.activation = _unpriced
    with pytest.raises(TypeError):
        _ = cost(config.encoder, input_grid=8, batch_size=2, dtype=None)


def test_encoder_runs_the_injected_activation() -> None:
    """HF's string schema is bypassed: the configured function is what every MLP calls."""
    config = tiny()
    assert isinstance(config.encoder, Dinov2WithRegisters.Config)
    reference = config.make()
    config.encoder.activation = functional.silu
    swapped = config.make()
    swapped.load_state_dict(reference.state_dict())
    assert list(swapped.state_dict()) == list(reference.state_dict())
    image = _image()
    assert not torch.equal(swapped.encode(image), reference.encode(image))


def _unpriced(x: Tensor) -> Tensor:
    """Return ``x``: an activation with no cost model."""
    return x


def test_decoder_cost_matches_torch() -> None:
    config = tiny().copy_tree().finalize()
    analytical = assert_cost_matches_torch(
        config.decoder,
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
    config.decoder.patch_size = decoder_patch
    config.image_size = 2 * decoder_patch
    config = config.copy_tree().finalize()
    analytical = cost(config, batch_size=2, dtype=None)
    model = config.make()
    generator = torch.Generator().manual_seed(0)
    side = config.image_size
    image = torch.randint(
        0,
        256,
        (2, 3, side, side),
        generator=generator,
        dtype=torch.uint8,
    )
    with FlopCounterMode(display=False) as counter:
        _ = model.decode(model.encode(image))
    assert analytical["flops", "matmul"].sum() == counter.get_total_flops()
    assert analytical.params == sum(p.numel() for p in model.parameters())
    assert analytical["adjoint"].sum() == 0


# The ``ViTXL_n08`` decoder's state-dict order, read from the ``data.pkl`` of the
# published ``model.pt`` (HTTP range requests on the zip, 456 entries): the
# reference's registration order, which the port must keep.
_DECODER_HEAD: Final = (
    ("decoder_pos_embed", (1, 257, 1152)),
    ("trainable_cls_token", (1, 1, 1152)),
    ("decoder_embed.weight", (1152, 768)),
    ("decoder_embed.bias", (1152,)),
)
_DECODER_LAYER: Final = (
    ("attention.attention.query.weight", (1152, 1152)),
    ("attention.attention.query.bias", (1152,)),
    ("attention.attention.key.weight", (1152, 1152)),
    ("attention.attention.key.bias", (1152,)),
    ("attention.attention.value.weight", (1152, 1152)),
    ("attention.attention.value.bias", (1152,)),
    ("attention.output.dense.weight", (1152, 1152)),
    ("attention.output.dense.bias", (1152,)),
    ("intermediate.dense.weight", (4096, 1152)),
    ("intermediate.dense.bias", (4096,)),
    ("output.dense.weight", (1152, 4096)),
    ("output.dense.bias", (1152,)),
    ("layernorm_before.weight", (1152,)),
    ("layernorm_before.bias", (1152,)),
    ("layernorm_after.weight", (1152,)),
    ("layernorm_after.bias", (1152,)),
)
_DECODER_TAIL: Final = (
    ("decoder_norm.weight", (1152,)),
    ("decoder_norm.bias", (1152,)),
    ("decoder_pred.weight", (768, 1152)),
    ("decoder_pred.bias", (768,)),
)

# The pinned ``model.safetensors`` header (224 entries, read by range request) minus
# ``layernorm.weight`` and ``layernorm.bias``, which the port loads and then strips.
_ENCODER_EMBEDDINGS: Final = (
    ("embeddings.cls_token", (1, 1, 768)),
    ("embeddings.mask_token", (1, 768)),
    ("embeddings.register_tokens", (1, 4, 768)),
    ("embeddings.position_embeddings", (1, 1370, 768)),
    ("embeddings.patch_embeddings.projection.weight", (768, 3, 14, 14)),
    ("embeddings.patch_embeddings.projection.bias", (768,)),
)
_ENCODER_LAYER: Final = (
    ("norm1.weight", (768,)),
    ("norm1.bias", (768,)),
    ("attention.attention.query.weight", (768, 768)),
    ("attention.attention.query.bias", (768,)),
    ("attention.attention.key.weight", (768, 768)),
    ("attention.attention.key.bias", (768,)),
    ("attention.attention.value.weight", (768, 768)),
    ("attention.attention.value.bias", (768,)),
    ("attention.output.dense.weight", (768, 768)),
    ("attention.output.dense.bias", (768,)),
    ("layer_scale1.lambda1", (768,)),
    ("norm2.weight", (768,)),
    ("norm2.bias", (768,)),
    ("mlp.fc1.weight", (3072, 768)),
    ("mlp.fc1.bias", (3072,)),
    ("mlp.fc2.weight", (768, 3072)),
    ("mlp.fc2.bias", (768,)),
    ("layer_scale2.lambda1", (768,)),
)


@cache
def _published_state() -> dict[str, tuple[int, ...]]:
    """Build the published architecture on the meta device; return key shapes."""
    config = rae_dinov2_base()
    config.decoder.checkpoint = None
    assert isinstance(config.encoder, Dinov2WithRegisters.Config)
    config.encoder.checkpoint = None
    with torch.device("meta"):
        state = config.make().state_dict()
    return {key: tuple(value.shape) for key, value in state.items()}


def test_published_decoder_keeps_the_checkpoint_names_and_order() -> None:
    state = _published_state()
    decoder = [
        (key.removeprefix("decoder."), shape)
        for key, shape in state.items()
        if key.startswith("decoder.")
    ]
    layers = [
        (f"decoder_layers.{index}.{name}", shape)
        for index in range(28)
        for name, shape in _DECODER_LAYER
    ]
    assert decoder == [*_DECODER_HEAD, *layers, *_DECODER_TAIL]
    assert len(decoder) == 456


def test_published_encoder_keeps_the_checkpoint_names() -> None:
    state = _published_state()
    encoder = {
        key.removeprefix("encoder.encoder."): shape
        for key, shape in state.items()
        if key.startswith("encoder.")
    }
    layers = {
        f"encoder.layer.{index}.{name}": shape
        for index in range(12)
        for name, shape in _ENCODER_LAYER
    }
    assert encoder == {**dict(_ENCODER_EMBEDDINGS), **layers}
    assert len(encoder) == 224 - 2


def test_encoder_registers_before_decoder() -> None:
    keys = list(tiny().make().state_dict())
    prefixes = [key.split(".", 1)[0] for key in keys]
    first_decoder = prefixes.index("decoder")
    assert set(prefixes[:first_decoder]) == {"encoder"}
    assert set(prefixes[first_decoder:]) == {"decoder"}


def test_published_latent_is_768_channels_at_16x16() -> None:
    assert rae_dinov2_base().latent_shape() == (768, 16, 16)


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
    """A ``[0, 1]`` float image would otherwise encode as near-black."""
    model = tiny().make()
    with pytest.raises(TypeError, match="uint8"):
        _ = model.encode(torch.rand(2, 3, 4, 5))


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
