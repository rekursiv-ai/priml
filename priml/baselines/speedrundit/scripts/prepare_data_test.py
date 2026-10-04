"""The corpus preparer builds exactly what the experiment declares."""

from __future__ import annotations

from dataclasses import field
from pathlib import Path
from typing import Final, cast, override

from configgle import Fig, Makeable
from PIL import Image
from torch import Tensor, nn
from torch.nn import functional

import pytest
import torch

from priml.baselines.speedrundit.corpus import RECEIPT, table_path
from priml.baselines.speedrundit.data import PairedImageLatentDataset
from priml.baselines.speedrundit.latent_codec import ScalarTableCodec
from priml.baselines.speedrundit.scripts import prepare_data
from priml.lib.custom_json import DictCodec, loads
from priml.model.vision_ae.custom_types import LatentNormalizer
from priml.model.vision_ae.latent_norm import ScaleLatents


_CWD: Final = Path(__file__).resolve().parent
_SIZE: Final = 16
"""Crop side the fake autoencoder asks for; square because center_crop cuts squares."""

_GRID: Final = (2, 3)
"""The fake latent's height and width."""


class _MeanAutoencoder(nn.Module):
    """Encodes an image as its pooled channel means, the first repeated as a fourth."""

    class Config(Fig["_MeanAutoencoder"]):
        image_size: int = _SIZE
        """Crop side."""

        latent_norm: Makeable[LatentNormalizer] = field(
            default_factory=ScaleLatents.Config,
        )
        """The identity scale."""

        def latent_shape(self) -> tuple[int, int, int]:
            """Return four channels on the fake's grid."""
            return 4, *_GRID

    def __init__(self, config: Config) -> None:
        super().__init__()
        del config

    def encode(self, image: Tensor, /) -> Tensor:
        """Return ``[B, 4, 2, 3]`` pooled channel means."""
        means = functional.adaptive_avg_pool2d(image.float(), _GRID)
        return torch.cat([means, means[:, :1]], dim=1)

    def decode(self, latent: Tensor, /) -> Tensor:
        """Return a flat image of the first three channels' means."""
        means = latent[:, :3].mean(dim=(2, 3), keepdim=True)
        return (means / 255).expand(-1, -1, _SIZE, _SIZE)

    @override
    def forward(self, image: Tensor) -> Tensor:
        return self.encode(image)


def _imagenet(root: Path, count: int) -> Path:
    """Write ``count`` distinct training JPEGs under the first synset."""
    synsets = _CWD.parents[2] / "data/processors/labels_imagenet.txt"
    synset = synsets.read_text(encoding="utf-8").splitlines()[0]
    directory = root / "raw" / "train" / synset
    directory.mkdir(parents=True)
    for index in range(count):
        colour = (40 * index, 64, 32)
        Image.new("RGB", (40, 30), colour).save(directory / f"{synset}_{index}.JPEG")
    return root / "raw"


def _source(root: Path, **fields: object) -> PairedImageLatentDataset.Config:
    config = PairedImageLatentDataset.Config(working_dir=root / "corpus")
    config.autoencoder = _MeanAutoencoder.Config()
    for name, value in fields.items():
        setattr(config, name, value)
    return config


def _receipt(config: PairedImageLatentDataset.Config) -> dict[str, object]:
    path = Path(config.working_dir) / config.latent_subdir / RECEIPT
    return DictCodec.coerce(loads(path.read_text()), default=None)


def test_prepared_corpus_loads_back_as_the_encoder_wrote_it(tmp_path: Path) -> None:
    raw = _imagenet(tmp_path, 3)
    config = _source(tmp_path)
    assert prepare_data.prepare(config, raw, device="cpu", batch_size=2) == 3
    dataset = config.make()
    assert len(dataset) == 3
    sample = dataset[1]
    expected = _MeanAutoencoder.Config().make().encode(sample["image"][None])[0]
    assert torch.equal(sample["latent"], expected)


def test_images_are_center_cropped_to_the_autoencoder_size(tmp_path: Path) -> None:
    config = _source(tmp_path)
    _ = prepare_data.prepare(config, _imagenet(tmp_path, 1), device="cpu")
    image = Path(config.working_dir) / "images/00000/img00000000.png"
    assert Image.open(image).size == (_SIZE, _SIZE)


def test_a_rerun_keeps_what_is_already_encoded(tmp_path: Path) -> None:
    raw = _imagenet(tmp_path, 2)
    config = _source(tmp_path)
    _ = prepare_data.prepare(config, raw, device="cpu")
    _ = prepare_data.prepare(config, raw, device="cpu")
    encoding = DictCodec.coerce(_receipt(config)["details"], default=None)["encoding"]
    assert DictCodec.coerce(encoding, default=None)["newly_encoded"] == 0


def test_a_second_corpus_reuses_the_shared_images(tmp_path: Path) -> None:
    raw = _imagenet(tmp_path, 2)
    _ = prepare_data.prepare(_source(tmp_path), raw, device="cpu")
    image = tmp_path / "corpus/images/00000/img00000001.png"
    before = image.stat().st_mtime_ns
    other = _source(tmp_path, latent_subdir="other")
    _ = prepare_data.prepare(other, raw, device="cpu")
    assert image.stat().st_mtime_ns == before
    assert len(other.make()) == 2


def test_receipt_records_the_producers_and_error(tmp_path: Path) -> None:
    config = _source(tmp_path)
    _ = prepare_data.prepare(config, _imagenet(tmp_path, 2), device="cpu")
    receipt = _receipt(config)
    autoencoder = DictCodec.coerce(receipt["autoencoder"], default=None)
    assert autoencoder["latent_shape"] == [4, *_GRID]
    error = DictCodec.coerce(
        DictCodec.coerce(receipt["details"], default=None)["error"],
        default=None,
    )
    assert error["mse"] == 0.0


def test_fitted_codec_writes_its_table_before_the_latents(tmp_path: Path) -> None:
    config = _source(tmp_path, codec=ScalarTableCodec.Config(num_fit_images=2))
    _ = prepare_data.prepare(config, _imagenet(tmp_path, 3), device="cpu")
    assert table_path(Path(config.working_dir) / "vae-in").is_file()
    latent = config.make()[0]["latent"]
    assert latent.dtype == torch.float32


def test_fit_sample_is_distinct_sorted_and_in_range() -> None:
    chosen = prepare_data.fit_sample_indices(1_000, 37)
    assert len(chosen) == 37
    assert chosen == sorted(set(chosen))
    assert chosen[0] >= 0
    assert chosen[-1] < 1_000
    assert prepare_data.fit_sample_indices(1_000, 37) == chosen


def test_fit_sample_takes_every_image_of_a_small_corpus() -> None:
    assert prepare_data.fit_sample_indices(5, 37) == [0, 1, 2, 3, 4]


def test_fit_sample_reaches_across_the_synset_sorted_order() -> None:
    """Images arrive grouped by class; a prefix would fit on the first classes only."""
    chosen = prepare_data.fit_sample_indices(1_000, 10)
    assert chosen[-1] >= 500


def test_receipt_only_admits_an_existing_reg_corpus(tmp_path: Path) -> None:
    raw = _imagenet(tmp_path, 2)
    config = _source(tmp_path)
    _ = prepare_data.prepare(config, raw, device="cpu")
    (Path(config.working_dir) / "vae-in" / RECEIPT).unlink()
    assert prepare_data.record_receipt(config) == 2
    assert len(config.make()) == 2


def test_dataset_config_is_the_experiments_resolved_corpus() -> None:
    config = prepare_data.dataset_config("exp000")
    assert Path(config.working_dir) == Path("/opt/scratch/datasets/speedrundit")
    assert config.latent_subdir == "vae-in"


def test_main_requires_a_source_unless_recording_a_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("sys.argv", ["prepare_data", "--experiment", "exp000"])
    with pytest.raises(SystemExit):
        _ = prepare_data.main()


def test_main_hands_the_experiment_corpus_to_the_preparer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called: list[tuple[PairedImageLatentDataset.Config, Path]] = []

    def fake(config: PairedImageLatentDataset.Config, source: Path, **_: object) -> int:
        called.append((config, source))
        return 0

    monkeypatch.setattr(prepare_data, "prepare", fake)
    monkeypatch.setattr(
        "sys.argv",
        ["prepare_data", "--source", str(tmp_path), "--directory", str(tmp_path / "c")],
    )
    assert prepare_data.main() == 0
    config, source = called[0]
    assert source == tmp_path
    assert Path(config.working_dir) == tmp_path / "c"
    assert cast(object, config.autoencoder) is not None


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
