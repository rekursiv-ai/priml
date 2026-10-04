"""Stored latents round-trip, and a receipt pins the corpus's producers."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import json

import pytest
import torch

from priml.baselines.speedrundit.corpus import (
    CorpusMismatchError,
    autoencoder_identity,
    load_stored,
    load_table,
    save_stored,
    save_table,
    verify_receipt,
    write_atomically,
    write_receipt,
)
from priml.baselines.speedrundit.latent_codec import (
    NUM_LEVELS,
    FloatCodec,
    ScalarTableCodec,
)
from priml.lib.custom_json import DictCodec, loads
from priml.model.vision_ae.custom_types import posterior_mode
from priml.model.vision_ae.invae import INVAE
from priml.model.vision_ae.latent_norm import ScaleLatents
from priml.model.vision_ae.rae import RAE, rae_dinov2_base
from priml.model.vision_ae.vtp import VTP


if TYPE_CHECKING:
    from pathlib import Path
    from typing import BinaryIO


@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.float16, torch.bfloat16, torch.uint8],
)
def test_every_stored_dtype_round_trips(tmp_path: Path, dtype: torch.dtype) -> None:
    """bfloat16 has no NumPy dtype and rides as its int16 bit pattern."""
    stored = (torch.arange(24).reshape(1, 2, 3, 4) % 7).to(dtype)
    path = tmp_path / "latent.npy"
    save_stored(path, stored=stored)
    loaded = load_stored(path, dtype=dtype)
    assert loaded.dtype == dtype
    assert torch.equal(loaded, stored)


def test_a_file_of_another_dtype_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "latent.npy"
    save_stored(path, stored=torch.zeros(2, dtype=torch.float32))
    with pytest.raises(ValueError, match="the codec stores"):
        _ = load_stored(path, dtype=torch.float16)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_non_finite_latents_are_refused_before_publication(
    tmp_path: Path,
    value: float,
) -> None:
    path = tmp_path / "latent.npy"
    with pytest.raises(ValueError, match="finite"):
        save_stored(path, stored=torch.tensor([value]))
    assert not path.exists()


def test_identity_collects_only_encoding_checkpoints() -> None:
    """Raw encoding depends on the encoder, not on decoding or normalization."""
    identity = autoencoder_identity(rae_dinov2_base())
    encoding = DictCodec.coerce(identity["encoding"], default=None)
    assert "decoder" not in encoding
    assert "latent_norm" not in encoding
    assert "model.safetensors" in json.dumps(encoding)
    assert identity["latent_shape"] == [768, 16, 16]


def test_encoding_policy_is_part_of_identity() -> None:
    assert autoencoder_identity(INVAE.Config()) != autoencoder_identity(
        INVAE.Config(latent_fn=posterior_mode),
    )


def test_normalization_and_decoding_do_not_change_raw_identity() -> None:
    config = RAE.Config()
    before = autoencoder_identity(config)
    config.latent_norm = ScaleLatents.Config()
    config.decoder.checkpoint = None
    assert autoencoder_identity(config) == before


def test_vtp_pixel_decoder_does_not_change_raw_identity() -> None:
    config = VTP.Config()
    before = autoencoder_identity(config)
    config.pixel_decoder.num_layers = 1
    assert autoencoder_identity(config) == before
    config.trunk.patch_size = 8
    assert autoencoder_identity(config) != before


def test_interrupted_save_does_not_publish_partial_latent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "latent.npy"
    monkeypatch.setattr("priml.baselines.speedrundit.corpus.np.save", _interrupted_save)
    with pytest.raises(OSError, match="interrupted"):
        save_stored(path, stored=torch.zeros(1))
    assert not any(tmp_path.iterdir())


def _interrupted_save(stream: BinaryIO, array: object) -> None:
    del array
    _ = stream.write(b"\x93NUMPY")
    raise OSError("interrupted")


def test_a_matching_receipt_verifies(tmp_path: Path) -> None:
    autoencoder, codec_config = INVAE.Config(), FloatCodec.Config()
    codec = codec_config.make()
    _ = write_receipt(
        tmp_path,
        autoencoder=autoencoder,
        codec_config=codec_config,
        codec=codec,
        table_sha256=None,
        details={},
    )
    verify_receipt(
        tmp_path,
        autoencoder=autoencoder,
        codec_config=codec_config,
        codec=codec,
        table_sha256=None,
    )


def test_every_mismatched_identity_is_named(tmp_path: Path) -> None:
    codec_config = FloatCodec.Config()
    _ = write_receipt(
        tmp_path,
        autoencoder=INVAE.Config(),
        codec_config=codec_config,
        codec=codec_config.make(),
        table_sha256=None,
        details={},
    )
    other = FloatCodec.Config(dtype=torch.float16)
    with pytest.raises(CorpusMismatchError) as caught:
        verify_receipt(
            tmp_path,
            autoencoder=INVAE.Config(image_size=128),
            codec_config=other,
            codec=other.make(),
            table_sha256=None,
        )
    message = str(caught.value)
    assert "autoencoder.latent_shape" in message
    assert "codec.stored_dtype" in message


def test_a_table_is_pinned_by_its_digest(tmp_path: Path) -> None:
    codec = ScalarTableCodec.Config().make()
    codec.fit(torch.randn(8, 2, 2, 2, generator=torch.Generator().manual_seed(0)))
    digest = save_table(tmp_path, codec=codec)
    restored = ScalarTableCodec.Config().make()
    assert load_table(tmp_path, codec=restored) == digest
    assert torch.equal(restored.table()["levels"], codec.table()["levels"])


def test_load_table_refuses_decreasing_persisted_levels(tmp_path: Path) -> None:
    levels = torch.linspace(1, 0, NUM_LEVELS).unsqueeze(0)
    torch.save({"levels": levels}, tmp_path / "codec.pt")
    with pytest.raises(ValueError, match="non-decreasing"):
        load_table(tmp_path, codec=ScalarTableCodec.Config().make())


def test_a_receipt_of_another_format_is_refused(tmp_path: Path) -> None:
    codec_config = FloatCodec.Config()
    path = write_receipt(
        tmp_path,
        autoencoder=INVAE.Config(),
        codec_config=codec_config,
        codec=codec_config.make(),
        table_sha256=None,
        details={},
    )
    receipt = DictCodec.coerce(loads(path.read_text()), default=None)
    _ = path.write_text(json.dumps({**receipt, "format": 1}))
    with pytest.raises(CorpusMismatchError, match="format 1"):
        verify_receipt(
            tmp_path,
            autoencoder=INVAE.Config(),
            codec_config=codec_config,
            codec=codec_config.make(),
            table_sha256=None,
        )


def test_a_lambda_cannot_identify_a_corpus() -> None:
    config = INVAE.Config(latent_fn=lambda posterior: posterior.mode())
    with pytest.raises(ValueError, match="module-level function"):
        _ = autoencoder_identity(config)


def test_concurrent_writers_never_share_a_staging_file(tmp_path: Path) -> None:
    """A second writer finishing inside the first must not move the first's file."""
    path = tmp_path / "shared.png"
    write_atomically(path, write=partial(_write_around, path=path))
    assert path.read_bytes() == b"outer"
    assert [entry.name for entry in tmp_path.iterdir()] == ["shared.png"]


def _write_around(stream: BinaryIO, *, path: Path) -> None:
    write_atomically(path, write=lambda inner: inner.write(b"inner"))
    _ = stream.write(b"outer")


def test_a_fitted_codec_without_its_table_is_refused(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="table"):
        _ = load_table(tmp_path, codec=ScalarTableCodec.Config().make())


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
