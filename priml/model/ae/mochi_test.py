"""Mochi: the published architecture, its checkpoint names, and the reference's bits.

``testdata/mochi_reference.pt`` holds the reference at ``15c9bad`` built small
through its own arguments -- width 2, two stages, time downsampled 3x -- with
its GroupNorms and attention heads, hardcoded at 32, built at 2: its weights,
a clip, and its own moments and unclamped pixels, minted under
``host_agnostic_numerics``; its forward code ran unmodified.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from safetensors.torch import save_file
from torch import Tensor, nn
from torch.nn.attention import SDPBackend, sdpa_kernel

import pytest
import torch

from priml.model.ae.checkpoint import HubFile, LocalFile
from priml.model.ae.distribution import DiagonalGaussian
from priml.model.ae.latent_norm import StdLatents
from priml.model.ae.mochi import (
    Decoder,
    Encoder,
    mochi,
    mochi_comfy,
    remap_mochi_state_dict,
)
from priml.model.ae.video import CausalVideoVAE
from priml.testing.bfb import assert_bfb_against_golden, host_agnostic_numerics
from priml.testing.cost import assert_cost_matches_torch
from priml.testing.golden import assert_pprint_golden, mismatches, read_tensors


_CWD: Final = Path(__file__).resolve().parent

_TESTDATA: Final = _CWD / "testdata"

_REFERENCE_COMMIT: Final = "15c9bad662df1fa3e7b53136423695b286484c08"


def tiny() -> CausalVideoVAE.Config:
    """Return the published layout at the reference golden's size: widths 2 and 4."""
    config = mochi()
    assert isinstance(config.encoder, Encoder.Config)
    assert isinstance(config.decoder, Decoder.Config)
    encoder, decoder = config.encoder, config.decoder
    encoder.channels_hidden, decoder.channels_hidden = (2, 4), (4, 2)
    encoder.blocks_per_stage = decoder.blocks_per_stage = (1, 1, 1)
    encoder.attention = (False, True, True)
    encoder.temporal = decoder.temporal = (3,)
    encoder.spatial = decoder.spatial = (2,)
    encoder.channels_head = 2
    for half in (encoder, decoder):
        half.channels_latent = 3
        half.num_groups = 2
        half.checkpoint = None
    config.num_frames = 4
    config.image_size = (4, 6)
    return config


def _clip(*, frames: int = 4) -> Tensor:
    generator = torch.Generator().manual_seed(0)
    return torch.randint(
        0,
        256,
        (2, 3, frames, 4, 6),
        generator=generator,
        dtype=torch.uint8,
    )


def _replay_reference(*, nudge: bool = False) -> list[str]:
    """Return how the port, loading the reference's tensors, misses its bits."""
    reference = read_tensors(_TESTDATA / "mochi_reference.pt")
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
        module.load_state_dict(remap_mochi_state_dict(state, half=half))
    if nudge:
        bias = model.decoder.output_proj.bias
        assert isinstance(bias, Tensor)
        bias[0] = torch.nextafter(bias[0], torch.tensor(torch.inf))
    # Batch, frames, and sides tie at 2 and 4: the reference's sizes at the golden
    # ceiling, and its own bits catch any transposed axis.
    image = reference["image"]
    assert reference["decoded"].abs().max() < 1
    with host_agnostic_numerics():
        moments = model.encoder.forward(image.float() / 127.5 - 1)
        mean = model.encode(image).mode()
        decoded = model.decoder.forward(mean)
        clamped = model.decode(mean)
    expected = {
        "moments": reference["moments"],
        "decoded": reference["decoded"],
        "clamped": ((reference["decoded"] + 1) / 2).clamp(0, 1),
    }
    actual = {"moments": moments, "decoded": decoded, "clamped": clamped}
    return mismatches(expected, actual=actual)


def test_matches_the_reference_implementation() -> None:
    report = _replay_reference()
    assert not report, report


def test_the_reference_replay_sees_one_ulp() -> None:
    report = _replay_reference(nudge=True)
    assert [line.split(":")[0] for line in report] == ["clamped", "decoded"]


def test_the_draw_is_mochis_inside_the_log_variance_clamp() -> None:
    """Mochi draws ``noise * exp(logvar * 0.5) + mean``, unclamped; ours clamps.

    The clamp, ``[-30, 20]``, never binds on the reference's own moments, and
    there the two draws are the same bits.
    """
    moments = read_tensors(_TESTDATA / "mochi_reference.pt")["moments"]
    mean, logvar = moments.chunk(2, dim=1)
    assert logvar.min() > -30
    assert logvar.max() < 20
    noise = torch.randn(mean.shape, generator=torch.Generator().manual_seed(0))
    mochis = noise * torch.exp(logvar * 0.5) + mean
    ours = DiagonalGaussian(mean, logvar).sample(
        generator=torch.Generator().manual_seed(0),
    )
    assert torch.equal(ours, mochis)


def test_round_trip_bfb() -> None:
    """The port's own round trip, ``sample()`` included, as ``kl.pt`` freezes AutoencoderKL's."""
    assert_bfb_against_golden(
        golden_dir=_TESTDATA,
        golden_name="mochi",
        build_module=tiny().make,
        build_input=_clip,
        run=_round_trip,
    )


@pytest.mark.parametrize("comfy", [False, True], ids=["released", "comfy"])
def test_published_halves_load_exactly_the_published_tensors(comfy: bool) -> None:
    """Each released half's names and shapes are its state, strictly.

    Comfy's one file holds both halves, each under its prefix.
    """
    config = mochi_comfy() if comfy else mochi()
    assert isinstance(config.encoder, Encoder.Config)
    assert isinstance(config.decoder, Decoder.Config)
    assert isinstance(config.encoder.checkpoint, HubFile.Config)
    assert isinstance(config.decoder.checkpoint, HubFile.Config)
    config.encoder.checkpoint = config.decoder.checkpoint = None
    with torch.device("meta"):
        model = config.make()
    for half, module in (("encoder", model.encoder), ("decoder", model.decoder)):
        assert module is not None
        lines = (_TESTDATA / f"mochi_{half}_keys.txt").read_text().splitlines()
        prefix = f"{half}." if comfy else ""
        published = {
            prefix + name: tuple(int(n) for n in shape.split("x"))
            for name, shape in (line.split() for line in lines)
        }
        with torch.device("meta"):
            state = {key: torch.empty(shape) for key, shape in published.items()}
        expected = {k: tuple(v.shape) for k, v in module.state_dict().items()}
        remapped = remap_mochi_state_dict(state, half=half)
        assert {k: tuple(v.shape) for k, v in remapped.items()} == expected


def test_published_latent_and_statistics() -> None:
    config = mochi()
    assert config.latent_shape() == (12, 28, 60, 106)
    norm = config.latent_norm
    assert isinstance(norm, StdLatents.Config)
    assert len(norm.mean) == len(norm.std) == 12


def test_round_trip_shapes() -> None:
    config = tiny()
    model = config.make()
    clip = _clip()
    posterior = model.encode(clip)
    assert isinstance(posterior, DiagonalGaussian)
    assert posterior.mode().shape == (2, *config.latent_shape())
    video = model.decode(posterior.mode())
    assert video.shape == clip.shape
    assert video.min() >= 0
    assert video.max() <= 1


def test_the_first_frame_encodes_as_the_image_it_is() -> None:
    """Causal in time: no later frame reaches the first latent frame."""
    model = tiny().make()
    clip = _clip(frames=7)
    with host_agnostic_numerics():
        whole = model.encode(clip).mode()
        first = model.encode(clip[:, :, :1]).mode()
        decoded = model.decode(whole)
        decoded_first = model.decode(whole[:, :, :1])
    assert torch.equal(whole[:, :, :1], first)
    assert torch.equal(decoded[:, :, :1], decoded_first)


def test_both_halves_load_their_own_files(tmp_path: Path) -> None:
    source = tiny().make()
    config = tiny()
    for half, module in (("encoder", source.encoder), ("decoder", source.decoder)):
        assert module is not None
        state = {key: value.contiguous() for key, value in module.state_dict().items()}
        save_file(state, str(tmp_path / f"{half}.safetensors"))
    assert isinstance(config.encoder, Encoder.Config)
    assert isinstance(config.decoder, Decoder.Config)
    config.encoder.checkpoint = LocalFile.Config(path=tmp_path / "encoder.safetensors")
    config.decoder.checkpoint = LocalFile.Config(path=tmp_path / "decoder.safetensors")
    loaded = config.make()
    clip = _clip()
    latent = source.encode(clip).mode()
    assert torch.equal(loaded.encode(clip).mode(), latent)
    assert torch.equal(loaded.decode(latent), source.decode(latent))


def test_rejects_per_stage_settings_that_miss_a_stage() -> None:
    encoder_short, decoder_short = tiny(), tiny()
    assert isinstance(encoder_short.encoder, Encoder.Config)
    assert isinstance(decoder_short.decoder, Decoder.Config)
    encoder_short.encoder.attention = (False, True)
    decoder_short.decoder.blocks_per_stage = (1, 1)
    for config in (encoder_short, decoder_short):
        with pytest.raises(ValueError, match="2 stages need"):
            _ = config.make()


def _round_trip(module: nn.Module, clip: Tensor) -> Tensor:
    assert isinstance(module, CausalVideoVAE)
    return module.decode(module.encode(clip).sample())


def test_cost_matches_torch_for_a_forward_round_trip() -> None:
    config = tiny()
    # Torch's FLOP counter has no entry for CPU flash attention; its math kernel
    # counts the same products as ``bmm``.
    with sdpa_kernel(SDPBackend.MATH):
        analytical = assert_cost_matches_torch(
            config,
            build_input=_clip,
            run=_round_trip,
            batch_size=2,
            dtype=None,
        )
    assert analytical["adjoint"].sum() == 0
    assert analytical.params == sum(p.numel() for p in config.make().parameters())


def test_published_config_pprint() -> None:
    assert_pprint_golden(test_file=__file__, name="mochi_config", config=mochi())


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
