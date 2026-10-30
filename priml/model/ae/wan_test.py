"""Wan: the published architectures, their checkpoint names, and the references' bits.

``testdata/wan_reference_{21,22}.pt`` each hold a reference at its pinned
commit -- Wan 2.1's ``vae.py`` and Wan 2.2's ``vae2_2.py`` -- built small
through its own arguments (width 2, two stages, Wan 2.2 with no residual block
in its encoder stages), its weights, a clip, and its own moments and unclamped
pixels, minted under ``host_agnostic_numerics`` through its chunked ``encode``
and its frame-at-a-time ``decode``. Replaying them proves the port's
whole-clip arithmetic against the references' cached chunks.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final

from torch import Tensor, nn
from torch.nn.attention import SDPBackend, sdpa_kernel

import pytest
import torch

from priml.model.ae.checkpoint import HubFile, LocalFile
from priml.model.ae.distribution import DiagonalGaussian
from priml.model.ae.latent_norm import InverseStdLatents
from priml.model.ae.video import CausalConv3d, CausalVideoVAE
from priml.model.ae.wan import (
    Decoder,
    Encoder,
    average_shortcut,
    duplicate_shortcut,
    remap_wan_state_dict,
    wan21,
    wan21_comfy,
    wan22,
    wan22_comfy,
)
from priml.testing.bfb import assert_bfb_against_golden, host_agnostic_numerics
from priml.testing.cost import assert_cost_matches_torch
from priml.testing.golden import assert_pprint_golden, mismatches, read_tensors


if TYPE_CHECKING:
    from collections.abc import Callable


_CWD: Final = Path(__file__).resolve().parent

_TESTDATA: Final = _CWD / "testdata"

_REFERENCE_COMMITS: Final = {
    "21": "9737cba9c1c3c4d04b33fcad41c111989865d315",
    "22": "1ea34ff48f87168174e12956e200b1d908b1c5ff",
}

_PUBLISHED: Final = {
    "wan21": wan21,
    "wan21_comfy": wan21_comfy,
    "wan22": wan22,
    "wan22_comfy": wan22_comfy,
}


def tiny(version: str = "21") -> CausalVideoVAE.Config:
    """Return a published layout at test size: three stages, widths 2 and 4.

    Both a spatial-only and a spatiotemporal resampler run, and Wan 2.2's both
    shortcuts.
    """
    config = wan21() if version == "21" else wan22()
    assert isinstance(config.encoder, Encoder.Config)
    assert isinstance(config.decoder, Decoder.Config)
    encoder, decoder = config.encoder, config.decoder
    encoder.channels_hidden = (2, 4, 4)
    decoder.channels_hidden = (4, 4, 2)
    decoder.channels_upsampled = (2, 2) if version == "21" else (4, 4)
    encoder.temporal = (1, 2)
    decoder.temporal = (2, 1)
    encoder.blocks_per_stage, decoder.blocks_per_stage = 1, 2
    encoder.channels_latent = decoder.channels_latent = 3
    encoder.checkpoint = decoder.checkpoint = None
    config.num_frames = 5
    config.image_size = (8, 12) if version == "21" else (16, 24)
    return config


def _clip(config: CausalVideoVAE.Config, *, frames: int = 5) -> Tensor:
    generator = torch.Generator().manual_seed(0)
    shape = (2, 3, frames, *config.image_size)
    return torch.randint(0, 256, shape, generator=generator, dtype=torch.uint8)


def _reference_layout(version: str) -> CausalVideoVAE.Config:
    """Return the layout the reference goldens were minted at, on a 3-frame 4x8 clip."""
    config = tiny(version)
    assert isinstance(config.encoder, Encoder.Config)
    assert isinstance(config.decoder, Decoder.Config)
    encoder, decoder = config.encoder, config.decoder
    encoder.channels_hidden = decoder.channels_hidden = (2, 2)
    encoder.temporal = decoder.temporal = (2,)
    # Wan 2.1's decoder halves its width at each upsampling, 2 to 1 here.
    decoder.channels_upsampled = (1,) if version == "21" else (2,)
    if version == "22":
        encoder.blocks_per_stage, decoder.blocks_per_stage = 0, 1
    config.num_frames, config.image_size = 3, (4, 8)
    return config


def _replay_reference(version: str, *, nudge: bool = False) -> list[str]:
    """Return how the port, loading a reference's tensors, misses its bits."""
    reference = read_tensors(_TESTDATA / f"wan_reference_{version}.pt")
    commit = reference.pop("source_commit").numpy().tobytes().decode()
    assert commit == _REFERENCE_COMMITS[version]
    state = {
        key.removeprefix("state."): value
        for key, value in reference.items()
        if key.startswith("state.")
    }
    config = _reference_layout(version)
    model = config.make()
    assert isinstance(model.encoder, Encoder)
    assert isinstance(model.decoder, Decoder)
    model.encoder.load_state_dict(remap_wan_state_dict(state, half="encoder"))
    model.decoder.load_state_dict(remap_wan_state_dict(state, half="decoder"))
    if nudge:
        conv_out = model.decoder.head[2]
        assert isinstance(conv_out, CausalConv3d)
        bias = conv_out.bias
        assert bias is not None
        bias[0] = torch.nextafter(bias[0], torch.tensor(torch.inf))
    # Batch, frames, and sides tie at 2 and 3: the references' sizes at the golden
    # ceiling, and their own bits catch any transposed axis.
    image = reference["image"]
    with host_agnostic_numerics():
        moments = model.encoder.forward(image.float() / 127.5 - 1)
        mean = model.encode(image).mode()
        decoded = model.decoder.forward(mean)
    expected = {
        "moments": reference["moments"],
        "mean": reference["moments"].chunk(2, dim=1)[0],
        "decoded": reference["decoded"],
    }
    actual = {"moments": moments, "mean": mean, "decoded": decoded}
    return mismatches(expected, actual=actual)


@pytest.mark.parametrize("version", sorted(_REFERENCE_COMMITS))
def test_matches_the_reference_implementation(version: str) -> None:
    report = _replay_reference(version)
    assert not report, report


def test_the_reference_replay_sees_one_ulp() -> None:
    report = _replay_reference("21", nudge=True)
    assert [line.split(":")[0] for line in report] == ["decoded"]


@pytest.mark.parametrize("name", sorted(_PUBLISHED))
def test_published_halves_load_exactly_the_published_tensors(name: str) -> None:
    """Each release's names and shapes are each half's state, strictly.

    The Comfy repackagings hold the releases' names and shapes, rounded.
    """
    version = "21" if name.startswith("wan21") else "22"
    lines = (_TESTDATA / f"wan{version}_vae_keys.txt").read_text().splitlines()
    published = {
        key: tuple(int(n) for n in shape.split("x"))
        for key, shape in (line.split() for line in lines)
    }
    config = _PUBLISHED[name]()
    assert isinstance(config.encoder, Encoder.Config)
    assert isinstance(config.decoder, Decoder.Config)
    assert isinstance(config.encoder.checkpoint, HubFile.Config)
    assert config.encoder.checkpoint == config.decoder.checkpoint
    assert config.encoder.checkpoint is not config.decoder.checkpoint
    config.encoder.checkpoint = config.decoder.checkpoint = None
    with torch.device("meta"):
        model = config.make()
        state = {key: torch.empty(shape) for key, shape in published.items()}
    for half, module in (("encoder", model.encoder), ("decoder", model.decoder)):
        assert module is not None
        expected = {k: tuple(v.shape) for k, v in module.state_dict().items()}
        remapped = remap_wan_state_dict(state, half=half)
        assert {k: tuple(v.shape) for k, v in remapped.items()} == expected


@pytest.mark.parametrize(
    ("factory", "latent_shape", "channels"),
    [(wan21, (16, 21, 60, 104), 16), (wan22, (48, 31, 44, 80), 48)],
    ids=["wan21", "wan22"],
)
def test_published_latents_and_statistics(
    factory: Callable[[], CausalVideoVAE.Config],
    latent_shape: tuple[int, int, int, int],
    channels: int,
) -> None:
    config = factory()
    assert config.latent_shape() == latent_shape
    norm = config.latent_norm
    assert isinstance(norm, InverseStdLatents.Config)
    assert len(norm.mean) == len(norm.std) == channels


@pytest.mark.parametrize("version", ["21", "22"])
def test_round_trip_shapes(version: str) -> None:
    config = tiny(version)
    model = config.make()
    clip = _clip(config)
    posterior = model.encode(clip)
    assert isinstance(posterior, DiagonalGaussian)
    assert posterior.mode().shape == (2, *config.latent_shape())
    video = model.decode(posterior.mode())
    assert video.shape == clip.shape
    assert video.min() >= 0
    assert video.max() <= 1


@pytest.mark.parametrize("version", ["21", "22"])
def test_the_first_frame_encodes_as_the_image_it_is(version: str) -> None:
    """Causal in time: no later frame reaches the first latent frame."""
    config = tiny(version)
    model = config.make()
    clip = _clip(config, frames=9)
    with host_agnostic_numerics():
        whole = model.encode(clip).mode()
        first = model.encode(clip[:, :, :1]).mode()
        decoded = model.decode(whole)
        decoded_first = model.decode(whole[:, :, :1])
    assert torch.equal(whole[:, :, :1], first)
    assert torch.equal(decoded[:, :, :1], decoded_first)


def test_both_halves_load_one_release_file(tmp_path: Path) -> None:
    """Saved as Wan releases it -- one ``.pth``, projections at the top -- and read back."""
    source_config = tiny("22")
    source = source_config.make()
    assert source.encoder is not None
    assert source.decoder is not None
    released: dict[str, Tensor] = {}
    for half, module, projection, named in (
        ("encoder", source.encoder, "quant_conv.", "conv1."),
        ("decoder", source.decoder, "post_quant_conv.", "conv2."),
    ):
        for key, value in module.state_dict().items():
            name = (
                named + key.removeprefix(projection)
                if key.startswith(projection)
                else f"{half}.{key}"
            )
            released[name] = value
    torch.save(released, tmp_path / "vae.pth")
    config = tiny("22")
    assert isinstance(config.encoder, Encoder.Config)
    assert isinstance(config.decoder, Decoder.Config)
    config.encoder.checkpoint = LocalFile.Config(path=tmp_path / "vae.pth")
    config.decoder.checkpoint = LocalFile.Config(path=tmp_path / "vae.pth")
    loaded = config.make()
    clip = _clip(source_config)
    latent = source.encode(clip).mode()
    assert torch.equal(loaded.encode(clip).mode(), latent)
    assert torch.equal(loaded.decode(latent), source.decode(latent))


@pytest.mark.parametrize(
    ("half", "message"),
    [
        (
            Encoder.Config(channels_hidden=(2, 4, 4), temporal=(2,)),
            "one factor, 1 or 2, per resampling stage",
        ),
        (
            Encoder.Config(channels_hidden=(2, 4, 4), temporal=(1, 3)),
            "one factor, 1 or 2, per resampling stage",
        ),
        (
            Decoder.Config(channels_hidden=(4, 4, 2), temporal=(2, 1, 1)),
            "one factor, 1 or 2, per resampling stage",
        ),
        (
            Decoder.Config(channels_hidden=(4, 4, 2), temporal=(2, 1)),
            "one width per upsampling stage",
        ),
    ],
    ids=["encoder_short", "encoder_thirds", "decoder_long", "decoder_widths"],
)
def test_halves_refuse_resampling_that_misses_a_stage_or_is_not_wans(
    half: Encoder.Config | Decoder.Config,
    message: str,
) -> None:
    half.checkpoint = None
    with pytest.raises(ValueError, match=message):
        _ = half.make()


def _round_trip(module: nn.Module, clip: Tensor) -> Tensor:
    assert isinstance(module, CausalVideoVAE)
    return module.decode(module.encode(clip).sample())


@pytest.mark.parametrize("version", ["21", "22"])
def test_round_trip_bfb(version: str) -> None:
    """The port's own round trip, ``sample()`` included, as ``kl.pt`` freezes AutoencoderKL's."""
    config = _reference_layout(version)
    assert_bfb_against_golden(
        golden_dir=_TESTDATA,
        golden_name=f"wan{version}",
        build_module=config.make,
        build_input=lambda: _clip(config, frames=3),
        run=_round_trip,
    )


@pytest.mark.parametrize("version", ["21", "22"])
def test_cost_matches_torch_for_a_forward_round_trip(version: str) -> None:
    config = tiny(version)
    # Torch's FLOP counter has no entry for CPU flash attention; its math kernel
    # counts the same products as ``bmm``.
    with sdpa_kernel(SDPBackend.MATH):
        analytical = assert_cost_matches_torch(
            config,
            build_input=lambda: _clip(config),
            run=_round_trip,
            batch_size=2,
            dtype=None,
        )
    assert analytical["adjoint"].sum() == 0
    assert analytical.params == sum(p.numel() for p in config.make().parameters())


@pytest.mark.parametrize("name", ["wan21", "wan22"])
def test_published_config_pprint(name: str) -> None:
    assert_pprint_golden(
        test_file=__file__,
        name=f"{name}_config",
        config=_PUBLISHED[name](),
    )


def test_shortcuts_resample_parameter_free() -> None:
    """Averaging pads the first frame with a zero frame; duplicating drops its copy."""
    x = torch.arange(2 * 2 * 3 * 4 * 6, dtype=torch.float32).reshape(2, 2, 3, 4, 6)
    down = average_shortcut(x, channels_out=4, factor_t=2, factor_s=2)
    assert down.shape == (2, 4, 2, 2, 3)
    up = duplicate_shortcut(x, channels_out=4, factor_t=2, factor_s=2)
    assert up.shape == (2, 4, 5, 8, 12)
    # A group of one is a pure rearrangement: channel 0 of the first frame is the
    # zero frame padded before it.
    moved = average_shortcut(x, channels_out=16, factor_t=2, factor_s=2)
    assert torch.equal(moved[:, 0, 0], torch.zeros(2, 2, 3))
    assert torch.equal(moved[:, 0, 1], x[:, 0, 1, ::2, ::2])


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
