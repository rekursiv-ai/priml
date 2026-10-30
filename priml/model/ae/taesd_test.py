"""TAESD: the published architecture, its checkpoint names, and the reference's bits.

``testdata/taesd_reference.pt`` holds the reference at ``fdbae2d``: its halves
with every convolution and block built at width 2 rather than its hardcoded 64,
their weights, an image, and the reference's own latent and unclamped pixels,
minted under ``host_agnostic_numerics``; its forward code ran unmodified. The
encoder is deterministic, so replaying it here freezes every bit the port
computes, against the reference itself.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import math

from safetensors.torch import save_file
from torch import Tensor, nn

import pytest
import torch

from priml.cost import cost
from priml.model.ae.checkpoint import HubFile, LocalFile
from priml.model.ae.distribution import PointMass
from priml.model.ae.latent_norm import ScaleLatents
from priml.model.ae.taesd import (
    TAESD,
    Decoder,
    Encoder,
    remap_taesd_state_dict,
    taesd,
    taesdxl,
)
from priml.model.conv import Conv2d
from priml.testing.bfb import assert_bfb_against_golden, host_agnostic_numerics
from priml.testing.cost import assert_cost_matches_torch
from priml.testing.golden import assert_pprint_golden, mismatches, read_tensors


_CWD: Final = Path(__file__).resolve().parent

_TESTDATA: Final = _CWD / "testdata"

_REFERENCE_COMMIT: Final = "fdbae2df989db95761f482efb8977e1182fb68d2"


def tiny() -> TAESD.Config:
    """Return the published layout at golden size: width 2, stated at 16x16 images."""
    config = taesd()
    assert config.encoder is not None
    assert config.decoder is not None
    for half in (config.encoder, config.decoder):
        half.channels_hidden = 2
        half.checkpoint = None
    config.image_size = 16
    return config


def _image(batch: int = 2) -> Tensor:
    generator = torch.Generator().manual_seed(0)
    return torch.randint(
        0,
        256,
        (batch, 3, 16, 24),
        generator=generator,
        dtype=torch.uint8,
    )


_PUBLISHED: Final = {"taesd": taesd, "taesdxl": taesdxl}


def _keys(listing: str) -> dict[str, tuple[int, ...]]:
    """Every tensor in a published half, by name, with its shape."""
    lines = (_TESTDATA / listing).read_text().splitlines()
    return {
        name: tuple(int(n) for n in shape.split("x"))
        for name, shape in (line.split() for line in lines)
    }


def test_matches_the_reference_implementation() -> None:
    reference = read_tensors(_TESTDATA / "taesd_reference.pt")
    commit = reference.pop("source_commit").numpy().tobytes().decode()
    assert commit == _REFERENCE_COMMIT
    model = tiny().make()
    assert isinstance(model.encoder, Encoder)
    assert isinstance(model.decoder, Decoder)
    for half, module in (("encoder", model.encoder), ("decoder", model.decoder)):
        state = {
            key.removeprefix(f"state.{half}."): value
            for key, value in reference.items()
            if key.startswith(f"state.{half}.")
        }
        module.load_state_dict(remap_taesd_state_dict(state, half=half))
    # Batch 2 is the width 2: the reference's fixed depth leaves width 2 under the
    # golden ceiling, and its own bits catch any transposed axis.
    image = reference["image"]
    with host_agnostic_numerics():
        latent = model.encode(image).mode()
        decoded = model.decoder.forward(reference["latent"])
        clamped = model.decode(reference["latent"])
    expected = {
        "latent": reference["latent"],
        "decoded": reference["decoded"],
        "clamped": reference["decoded"].clamp(0, 1),
    }
    actual = {"latent": latent, "decoded": decoded, "clamped": clamped}
    report = mismatches(expected, actual=actual)
    assert not report, report


def test_the_reference_replay_sees_one_ulp() -> None:
    reference = read_tensors(_TESTDATA / "taesd_reference.pt")
    model = tiny().make()
    assert isinstance(model.decoder, Decoder)
    state = {
        key.removeprefix("state.decoder."): value
        for key, value in reference.items()
        if key.startswith("state.decoder.")
    }
    model.decoder.load_state_dict(remap_taesd_state_dict(state, half="decoder"))
    conv_out = model.decoder.layers[-1]
    assert isinstance(conv_out, Conv2d)
    bias = conv_out.bias
    assert bias is not None
    bias[0] = torch.nextafter(bias[0], torch.tensor(torch.inf))
    with host_agnostic_numerics():
        decoded = model.decoder.forward(reference["latent"])
    report = mismatches({"decoded": reference["decoded"]}, actual={"decoded": decoded})
    assert [line.split(":")[0] for line in report] == ["decoded"]


@pytest.mark.parametrize("name", sorted(_PUBLISHED))
def test_published_halves_load_exactly_the_published_tensors(name: str) -> None:
    """Each published file's names and shapes are its half's state, strictly.

    TAESDXL's files hold TAESD's names and shapes.
    """
    config = _PUBLISHED[name]()
    assert config.encoder is not None
    assert config.decoder is not None
    config.encoder.checkpoint = config.decoder.checkpoint = None
    with torch.device("meta"):
        model = config.make()
    for half, module in (("encoder", model.encoder), ("decoder", model.decoder)):
        assert module is not None
        published = _keys(f"taesd_{half}_keys.txt")
        with torch.device("meta"):
            state = {name: torch.empty(shape) for name, shape in published.items()}
        expected = {k: tuple(v.shape) for k, v in module.state_dict().items()}
        remapped = remap_taesd_state_dict(state, half=half)
        assert {k: tuple(v.shape) for k, v in remapped.items()} == expected
        assert len(published) == 67


def test_the_published_checkpoints_are_pinned_per_half() -> None:
    for config, size in ((taesd(), 512), (taesdxl(), 1024)):
        assert config.encoder is not None
        assert config.decoder is not None
        assert isinstance(config.encoder.checkpoint, HubFile.Config)
        assert isinstance(config.decoder.checkpoint, HubFile.Config)
        assert config.encoder.checkpoint.filename.endswith("_encoder.safetensors")
        assert config.decoder.checkpoint.filename.endswith("_decoder.safetensors")
        assert config.latent_shape() == (4, size // 8, size // 8)
        assert config.latent_norm == ScaleLatents.Config()


def test_each_half_loads_its_own_file(tmp_path: Path) -> None:
    source = tiny().make()
    assert source.encoder is not None
    assert source.decoder is not None
    config = tiny()
    assert config.encoder is not None
    assert config.decoder is not None
    for half, module in (("encoder", source.encoder), ("decoder", source.decoder)):
        # As published: ``nn.Sequential`` indices, no ``layers.`` prefix.
        state = {
            key.removeprefix("layers."): value.contiguous()
            for key, value in module.state_dict().items()
        }
        save_file(state, str(tmp_path / f"{half}.safetensors"))
    config.encoder.checkpoint = LocalFile.Config(path=tmp_path / "encoder.safetensors")
    config.decoder.checkpoint = LocalFile.Config(path=tmp_path / "decoder.safetensors")
    loaded = config.make()
    image = _image()
    latent = source.encode(image).mode()
    assert torch.equal(loaded.encode(image).mode(), latent)
    assert torch.equal(loaded.decode(latent), source.decode(latent))


def test_encode_is_a_point_mass_at_the_latent_shape() -> None:
    posterior = tiny().make().encode(_image())
    assert isinstance(posterior, PointMass)
    assert posterior.mode().shape == (2, 4, 2, 3)


def test_leading_dimensions_pass_through() -> None:
    model = tiny().make()
    image = torch.randint(
        0,
        256,
        (2, 5, 3, 16, 24),
        generator=torch.Generator().manual_seed(1),
        dtype=torch.uint8,
    )
    flat = model.encode(image.flatten(0, 1)).mode()
    nested = model.encode(image).mode()
    assert torch.equal(nested.flatten(0, 1), flat)
    assert model.decode(nested).shape == (2, 5, 3, 16, 24)


def test_decode_returns_the_image_in_the_unit_interval() -> None:
    model = tiny().make()
    image = model.decode(model.encode(_image()).mode())
    assert image.min() >= 0
    assert image.max() <= 1


def test_either_half_can_be_left_out() -> None:
    encode_only, decode_only = tiny(), tiny()
    encode_only.decoder = None
    decode_only.encoder = None
    with pytest.raises(ValueError, match="without a decoder"):
        _ = encode_only.make().decode(torch.zeros(2, 4, 2, 3))
    with pytest.raises(ValueError, match="without an encoder"):
        _ = decode_only.make().encode(_image())


def test_encode_refuses_a_float_image() -> None:
    with pytest.raises(TypeError, match="uint8"):
        _ = tiny().make().encode(torch.rand(2, 3, 16, 24))


def test_train_leaves_the_frozen_model_in_eval_mode() -> None:
    model = tiny().make()
    _ = model.train()
    assert not model.training
    assert not any(parameter.requires_grad for parameter in model.parameters())


def test_rejects_an_image_size_the_stages_cannot_halve() -> None:
    config = tiny()
    config.image_size = 20
    with pytest.raises(ValueError, match="divisible by 8"):
        _ = config.make()


def test_rejects_halves_that_disagree_on_the_latent() -> None:
    config = tiny()
    assert config.decoder is not None
    config.decoder.channels_latent = 5
    with pytest.raises(ValueError, match="channels_latent"):
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


def _round_trip(module: nn.Module, image: Tensor) -> Tensor:
    assert isinstance(module, TAESD)
    return module.decode(module.encode(image).mode())


def test_round_trip_bfb() -> None:
    """The port's own round trip, as ``kl.pt`` freezes AutoencoderKL's."""
    assert_bfb_against_golden(
        golden_dir=_TESTDATA,
        golden_name="taesd",
        build_module=tiny().make,
        build_input=_image,
        run=_round_trip,
    )


def test_cost_matches_torch_for_a_forward_round_trip() -> None:
    """Batch 4 and 5 latent channels keep every axis distinct from width 2."""
    config = tiny()
    assert config.encoder is not None
    assert config.decoder is not None
    config.encoder.channels_latent = config.decoder.channels_latent = 5
    # A square image: ``cost`` prices ``image_size``.
    config.image_size = 24
    analytical = assert_cost_matches_torch(
        config,
        build_input=lambda: torch.randint(0, 256, (4, 3, 24, 24), dtype=torch.uint8),
        run=_round_trip,
        batch_size=4,
        dtype=None,
    )
    assert analytical["adjoint"].sum() == 0
    assert analytical.params == sum(p.numel() for p in config.make().parameters())


def test_published_cost_counts_every_parameter() -> None:
    config = taesd().copy_tree().finalize()
    analytical = cost(config, batch_size=1, dtype=None)
    assert config.encoder is not None
    assert config.decoder is not None
    config.encoder.checkpoint = config.decoder.checkpoint = None
    with torch.device("meta"):
        model = config.make()
    assert analytical.params == sum(p.numel() for p in model.parameters())
    published = sum(
        math.prod(shape)
        for half in ("encoder", "decoder")
        for shape in _keys(f"taesd_{half}_keys.txt").values()
    )
    assert analytical.params == published


@pytest.mark.parametrize("name", sorted(_PUBLISHED))
def test_published_config_pprint(name: str) -> None:
    assert_pprint_golden(
        test_file=__file__,
        name=f"{name}_config",
        config=_PUBLISHED[name](),
    )


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
