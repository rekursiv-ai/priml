"""Local ImageNet/INVAE preparation without the REG repository at runtime."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

from PIL import Image

import numpy as np
import torch

from priml.baselines.speedrundit.data import PairedImageLatentDataset
from priml.baselines.speedrundit.scripts import prepare_data
from priml.lib.custom_json import DictCodec, loads


if TYPE_CHECKING:
    from numpy.typing import NDArray

    import pytest


_CWD: Final = Path(__file__).resolve().parent


class _FakeVAE:
    def encode(self, image: torch.Tensor) -> object:
        class Posterior:
            def sample(self) -> torch.Tensor:
                return torch.ones(image.shape[0], 32, 16, 16)

        return Posterior()


def _fake_load_invae(*_args: object, **_kwargs: object) -> _FakeVAE:
    return _FakeVAE()


def test_preparer_writes_paired_source_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    synsets = _CWD.parents[2] / "data/processors/labels_imagenet.txt"
    first_class = synsets.read_text(encoding="utf-8").splitlines()[0]
    image_dir = tmp_path / "raw" / "train" / first_class
    image_dir.mkdir(parents=True)
    Image.new("RGB", (320, 280), (128, 64, 32)).save(
        image_dir / f"{first_class}_1.JPEG",
    )
    monkeypatch.setattr(prepare_data, "load_invae", _fake_load_invae)
    output = tmp_path / "prepared"
    assert prepare_data.prepare(tmp_path / "raw", output, device="cpu") == 1
    image_path = output / "images/00000/img00000000.png"
    latent_path = output / "vae-in/00000/img-latents-00000000.npy"
    assert image_path.exists()
    assert latent_path.exists()
    assert Image.open(image_path).size == (256, 256)
    latent = cast("NDArray[np.float32]", np.load(latent_path))
    assert latent.shape == (1, 32, 16, 16)
    manifest = DictCodec.coerce(
        loads((output / "vae-in/dataset.json").read_text()),
        default=None,
    )
    assert manifest["labels"] == [["00000/img-latents-00000000.npy", 0]]
    dataset = PairedImageLatentDataset.Config(working_dir=output).make()
    assert len(dataset) == 1
    assert dataset[0]["label"] == 0


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
