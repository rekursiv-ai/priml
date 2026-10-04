"""Stored latents round-trip, and a receipt pins the corpus's producers."""

from __future__ import annotations

from typing import TYPE_CHECKING

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
    write_receipt,
)
from priml.baselines.speedrundit.latent_codec import FloatCodec, ScalarTableCodec
from priml.lib.custom_json import ListCodec
from priml.model.vision_ae.invae import INVAE
from priml.model.vision_ae.rae import rae_dinov2_base


if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.float16, torch.bfloat16, torch.uint8],
)
def test_every_stored_dtype_round_trips(tmp_path: Path, dtype: torch.dtype) -> None:
    """bfloat16 has no NumPy dtype and rides as its int16 bit pattern."""
    stored = (torch.arange(24).reshape(1, 2, 3, 4) % 7).to(dtype)
    path = tmp_path / "latent.npy"
    save_stored(path, stored)
    loaded = load_stored(path, dtype)
    assert loaded.dtype == dtype
    assert torch.equal(loaded, stored)


def test_a_file_of_another_dtype_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "latent.npy"
    save_stored(path, torch.zeros(2, dtype=torch.float32))
    with pytest.raises(ValueError, match="the codec stores"):
        _ = load_stored(path, torch.float16)


def test_identity_collects_every_nested_checkpoint() -> None:
    """RAE names its encoder, decoder, and statistics files: all three are pinned."""
    identity = autoencoder_identity(rae_dinov2_base())
    files = ListCodec.mappings(identity["checkpoints"])
    names = {str(entry["filename"]) for entry in files}
    assert names == {
        "model.safetensors",
        "decoders/dinov2/wReg_base/ViTXL_n08/model.pt",
        "stats/dinov2/wReg_base/imagenet1k/stat.pt",
    }
    assert identity["latent_shape"] == [768, 16, 16]


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
    digest = save_table(tmp_path, codec)
    restored = ScalarTableCodec.Config().make()
    assert load_table(tmp_path, restored) == digest
    assert torch.equal(restored.table()["levels"], codec.table()["levels"])


def test_a_fitted_codec_without_its_table_is_refused(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="table"):
        _ = load_table(tmp_path, ScalarTableCodec.Config().make())


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
