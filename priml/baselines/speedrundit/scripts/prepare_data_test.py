"""The corpus preparer builds exactly what the experiment declares."""

from __future__ import annotations

from dataclasses import field
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Final, cast, override
from unittest.mock import Mock

import shutil

from configgle import Fig, Makeable
from PIL import Image
from torch import Tensor, nn
from torch.nn import functional
from treekle import from_plain, json_loads

import pytest
import torch

from priml.baselines.speedrundit.corpus import (
    RECEIPT,
    CorpusMismatchError,
    save_stored,
    table_path,
)
from priml.baselines.speedrundit.data import PairedImageLatentDataset
from priml.baselines.speedrundit.latent_codec import FloatCodec, ScalarTableCodec
from priml.baselines.speedrundit.scripts import prepare_data
from priml.model.ae.distribution import PointMass
from priml.model.ae.kl_test import tiny
from priml.model.ae.latent_norm import LatentNormalizer, ScaleLatents


_CWD: Final = Path(__file__).resolve().parent
_SIZE: Final = 16
"""Crop side the fake autoencoder asks for; square because center_crop cuts squares."""

_GRID: Final = (2, 3)
"""The fake latent's height and width."""


def test_prepared_corpus_loads_back_as_the_encoder_wrote_it(tmp_path: Path) -> None:
    raw = _imagenet(tmp_path, count=3)
    config = _source(tmp_path)
    assert prepare_data.prepare(config, imagenet=raw, device="cpu", batch_size=2) == 3
    dataset = config.make()
    assert len(dataset) == 3
    sample = dataset[1]
    expected = _MeanAutoencoder.Config().make().encode(sample["image"][None]).mode()[0]
    assert torch.equal(sample["latent"], expected)


def test_center_crop_box_downsamples_before_center_crop() -> None:
    image = _gradient((12, 8))

    cropped = prepare_data.center_crop(image, size=4)
    expected = image.resize((6, 4), Image.Resampling.BOX).crop((1, 0, 5, 4))

    assert cropped.size == (4, 4)
    assert cropped.tobytes() == expected.tobytes()


def test_center_crop_resizes_short_axis_and_centers_both_axes() -> None:
    image = Image.new("RGB", (7, 5))
    for x in range(image.width):
        for y in range(image.height):
            image.putpixel((x, y), (x * 30, y * 40, 0))

    cropped = prepare_data.center_crop(image, size=4)
    expected = image.resize((6, 4), Image.Resampling.BICUBIC).crop((1, 0, 5, 4))

    assert cropped.size == (4, 4)
    assert cropped.tobytes() == expected.tobytes()


@pytest.mark.parametrize(
    ("source_size", "resized_size", "crop_box"),
    [
        ((14, 8), (7, 4), (1, 0, 5, 4)),
        ((8, 14), (4, 7), (0, 1, 4, 5)),
    ],
)
def test_center_crop_floors_odd_center_offsets(
    source_size: tuple[int, int],
    resized_size: tuple[int, int],
    crop_box: tuple[int, int, int, int],
) -> None:
    image = _gradient(source_size)
    resized = image.resize(resized_size, Image.Resampling.BOX)
    expected = resized.crop(crop_box)

    assert prepare_data.center_crop(image, size=4).tobytes() == expected.tobytes()


def test_images_are_center_cropped_to_the_autoencoder_size(tmp_path: Path) -> None:
    config = _source(tmp_path)
    _ = prepare_data.prepare(
        config,
        imagenet=_imagenet(tmp_path, count=1),
        device="cpu",
    )
    with Image.open(Path(config.working_dir) / "images/00000/img00000000.png") as image:
        assert image.size == (_SIZE, _SIZE)


def test_a_rerun_keeps_what_is_already_encoded(tmp_path: Path) -> None:
    raw = _imagenet(tmp_path, count=2)
    config = _source(tmp_path)
    _ = prepare_data.prepare(config, imagenet=raw, device="cpu")
    _ = prepare_data.prepare(config, imagenet=raw, device="cpu")
    encoding = from_plain(_receipt(config)["details"], dict[str, object])["encoding"]
    assert from_plain(encoding, dict[str, object])["newly_encoded"] == 0


def test_a_second_corpus_reuses_the_shared_images(tmp_path: Path) -> None:
    raw = _imagenet(tmp_path, count=2)
    _ = prepare_data.prepare(_source(tmp_path), imagenet=raw, device="cpu")
    image = tmp_path / "corpus/images/00000/img00000001.png"
    before = image.stat().st_mtime_ns
    other = _source(tmp_path, latent_subdir="other")
    _ = prepare_data.prepare(other, imagenet=raw, device="cpu")
    assert image.stat().st_mtime_ns == before
    assert len(other.make()) == 2


def test_receipt_records_the_producers_and_error(tmp_path: Path) -> None:
    config = _source(tmp_path)
    _ = prepare_data.prepare(
        config,
        imagenet=_imagenet(tmp_path, count=2),
        device="cpu",
    )
    receipt = _receipt(config)
    identity = from_plain(receipt["identity"], dict[str, object])
    autoencoder = from_plain(identity["autoencoder"], dict[str, object])
    assert autoencoder["latent_shape"] == [4, *_GRID]
    error = from_plain(
        from_plain(receipt["details"], dict[str, object])["error"],
        dict[str, object],
    )
    assert error["mse"] == 0.0


def test_fitted_codec_writes_its_table_before_the_latents(tmp_path: Path) -> None:
    config = _source(tmp_path, codec=ScalarTableCodec.Config(num_fit_images=2))
    _ = prepare_data.prepare(
        config,
        imagenet=_imagenet(tmp_path, count=3),
        device="cpu",
    )
    assert table_path(Path(config.working_dir) / "vae-in").is_file()
    latent = config.make()[0]["latent"]
    assert latent.dtype == torch.float32


def test_a_table_left_without_a_receipt_is_refitted_for_its_new_producer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _source(tmp_path, codec=ScalarTableCodec.Config(num_fit_images=2))
    config.autoencoder = tiny()
    raw = _imagenet(tmp_path, count=3)
    with monkeypatch.context() as scoped:
        scoped.setattr(prepare_data, "write_receipt", _interrupted)
        with pytest.raises(OSError, match="interrupted"):
            prepare_data.prepare(config, imagenet=raw, device="cpu")
    directory = Path(config.working_dir) / config.latent_subdir
    orphan = table_path(directory).read_bytes()
    config.sample = False
    _ = prepare_data.prepare(config, imagenet=raw, device="cpu")
    assert table_path(directory).read_bytes() != orphan
    assert len(config.make()) == 3


def _interrupted(*args: object, **kwargs: object) -> None:
    del args, kwargs
    raise OSError("interrupted")


def test_an_interrupted_preparation_never_loads_as_a_finished_corpus(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _imagenet(tmp_path, count=6)
    config = _source(tmp_path)
    _ = prepare_data.prepare(config, imagenet=raw, device="cpu", limit=2)
    saves: list[Path] = []
    with monkeypatch.context() as scoped:
        scoped.setattr(prepare_data, "save_stored", partial(_save_twice, saves=saves))
        with pytest.raises(OSError, match="interrupted"):
            prepare_data.prepare(config, imagenet=raw, device="cpu", batch_size=2)
    with pytest.raises(CorpusMismatchError, match="unfinished"):
        _ = config.make()
    _ = prepare_data.prepare(config, imagenet=raw, device="cpu", batch_size=2)
    assert len(config.make()) == 6


def _save_twice(path: Path, stored: Tensor, *, saves: list[Path]) -> None:
    if len(saves) == 2:
        raise OSError("interrupted")
    saves.append(path)
    save_stored(path, stored=stored)


def test_latents_without_a_receipt_are_refused(tmp_path: Path) -> None:
    config = _source(tmp_path)
    latent = Path(config.working_dir) / config.latent_subdir / "00000/x.npy"
    latent.parent.mkdir(parents=True)
    save_stored(latent, stored=torch.zeros(1))
    with pytest.raises(CorpusMismatchError, match="no receipt"):
        prepare_data.prepare(
            config,
            imagenet=_imagenet(tmp_path, count=1),
            device="cpu",
        )


@pytest.mark.parametrize("limits", [{"batch_size": 0}, {"limit": 0}])
def test_preparation_needs_positive_batch_size_and_limit(
    tmp_path: Path,
    limits: dict[str, int],
) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        prepare_data.prepare(
            _source(tmp_path),
            imagenet=tmp_path,
            device="cpu",
            **limits,
        )


def test_a_limited_run_fits_the_table_on_the_whole_source(tmp_path: Path) -> None:
    raw = _imagenet(tmp_path, count=4)
    codec = ScalarTableCodec.Config(num_fit_images=4)
    limited = _source(tmp_path / "limited", codec=codec)
    full = _source(tmp_path / "full", codec=codec.copy_tree())
    _ = prepare_data.prepare(limited, imagenet=raw, device="cpu", limit=1)
    _ = prepare_data.prepare(full, imagenet=raw, device="cpu")
    tables = [
        table_path(Path(c.working_dir) / c.latent_subdir).read_bytes()
        for c in (limited, full)
    ]
    assert tables[0] == tables[1]


def test_a_rerun_keeps_the_fit_record(tmp_path: Path) -> None:
    raw = _imagenet(tmp_path, count=3)
    config = _source(tmp_path, codec=ScalarTableCodec.Config(num_fit_images=2))
    _ = prepare_data.prepare(config, imagenet=raw, device="cpu")
    fit = from_plain(_receipt(config)["details"], dict[str, object])["fit"]
    _ = prepare_data.prepare(config, imagenet=raw, device="cpu")
    assert from_plain(_receipt(config)["details"], dict[str, object])["fit"] == fit
    assert from_plain(fit, dict[str, object])["num_images"] == 2


def test_latents_of_the_wrong_shape_are_refused_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_MeanAutoencoder, "encode", _three_channel_encode)
    config = _source(tmp_path)
    with pytest.raises(ValueError, match="declares"):
        prepare_data.prepare(
            config,
            imagenet=_imagenet(tmp_path, count=1),
            device="cpu",
        )
    assert not any((Path(config.working_dir) / config.latent_subdir).rglob("*.npy"))


def _three_channel_encode(model: nn.Module, images: Tensor) -> PointMass:
    del model
    return PointMass(torch.zeros(images.shape[0], 3, *_GRID))


def test_fit_sample_is_distinct_sorted_and_in_range() -> None:
    chosen = prepare_data.fit_sample_indices(1_000, num_images=37)
    assert len(chosen) == 37
    assert chosen == sorted(set(chosen))
    assert chosen[0] >= 0
    assert chosen[-1] < 1_000
    assert prepare_data.fit_sample_indices(1_000, num_images=37) == chosen


def test_fit_sample_takes_every_image_of_a_small_corpus() -> None:
    assert prepare_data.fit_sample_indices(5, num_images=37) == [0, 1, 2, 3, 4]


def test_fit_sample_reaches_across_the_synset_sorted_order() -> None:
    """Images arrive grouped by class; a prefix would fit on the first classes only."""
    chosen = prepare_data.fit_sample_indices(1_000, num_images=10)
    assert chosen[-1] >= 500


def test_receipt_only_admits_an_existing_reg_corpus(tmp_path: Path) -> None:
    raw = _imagenet(tmp_path, count=2)
    config = _source(tmp_path)
    _ = prepare_data.prepare(config, imagenet=raw, device="cpu")
    (Path(config.working_dir) / "vae-in" / RECEIPT).unlink()
    assert prepare_data.record_receipt(config) == 2
    assert len(config.make()) == 2


def test_rerun_refuses_to_relabel_another_producers_bytes(tmp_path: Path) -> None:
    raw = _imagenet(tmp_path, count=2)
    config = _source(tmp_path)
    prepare_data.prepare(config, imagenet=raw, device="cpu")
    receipt = Path(config.working_dir) / config.latent_subdir / RECEIPT
    before = receipt.read_bytes()
    config.codec = FloatCodec.Config(dtype=torch.float16)
    with pytest.raises(CorpusMismatchError):
        prepare_data.prepare(config, imagenet=raw, device="cpu")
    assert receipt.read_bytes() == before


def test_receipt_only_never_replaces_a_receipt(tmp_path: Path) -> None:
    config = _source(tmp_path)
    _ = prepare_data.prepare(
        config,
        imagenet=_imagenet(tmp_path, count=2),
        device="cpu",
    )
    receipt = Path(config.working_dir) / config.latent_subdir / RECEIPT
    before = receipt.read_bytes()
    config.codec = FloatCodec.Config(dtype=torch.float16)
    with pytest.raises(CorpusMismatchError):
        prepare_data.record_receipt(config)
    assert receipt.read_bytes() == before


def test_rerun_refuses_changed_encoding_seed(tmp_path: Path) -> None:
    raw = _imagenet(tmp_path, count=2)
    config = _source(tmp_path)
    config.autoencoder = tiny()
    config.seed = 1
    prepare_data.prepare(config, imagenet=raw, device="cpu", limit=1)
    config.seed = 2
    with pytest.raises(CorpusMismatchError, match=r"other settings:\n  seed"):
        prepare_data.prepare(config, imagenet=raw, device="cpu")


def test_resume_ignores_device_and_an_unseeded_batch_size(tmp_path: Path) -> None:
    raw = _imagenet(tmp_path, count=3)
    config = _source(tmp_path)
    _ = prepare_data.prepare(config, imagenet=raw, device="cpu", batch_size=2, limit=1)
    _ = prepare_data.prepare(config, imagenet=raw, device="cpu:0", batch_size=3)
    assert len(config.make()) == 3


def test_shared_images_refuse_a_different_source(tmp_path: Path) -> None:
    first, second = (
        _imagenet(tmp_path / "first", count=2),
        _imagenet(tmp_path / "second", count=3),
    )
    prepare_data.prepare(_source(tmp_path), imagenet=first, device="cpu")
    other = _source(tmp_path, latent_subdir="other")
    with pytest.raises(CorpusMismatchError, match="records_sha256"):
        prepare_data.prepare(other, imagenet=second, device="cpu")


def test_shared_images_accept_the_same_source_at_another_path(tmp_path: Path) -> None:
    first = _imagenet(tmp_path / "first", count=2)
    second = Path(shutil.copytree(first, tmp_path / "second"))
    prepare_data.prepare(_source(tmp_path), imagenet=first, device="cpu")
    other = _source(tmp_path, latent_subdir="other")
    assert prepare_data.prepare(other, imagenet=second, device="cpu") == 2


def test_crops_that_record_no_source_are_refused_until_adopted(tmp_path: Path) -> None:
    raw = _imagenet(tmp_path, count=2)
    config = _source(tmp_path)
    _ = prepare_data.prepare(config, imagenet=raw, device="cpu")
    (Path(config.working_dir) / "images" / prepare_data.IMAGE_SOURCE).unlink()
    other = _source(tmp_path, latent_subdir="other")
    with pytest.raises(CorpusMismatchError, match="record no source"):
        prepare_data.prepare(other, imagenet=raw, device="cpu")
    assert prepare_data.record_receipt(config, imagenet=raw) == 2
    assert prepare_data.prepare(other, imagenet=raw, device="cpu") == 2


def test_lower_limit_preserves_labels_for_existing_latents(tmp_path: Path) -> None:
    raw = _imagenet(tmp_path, count=3)
    config = _source(tmp_path)
    prepare_data.prepare(config, imagenet=raw, device="cpu")
    prepare_data.prepare(config, imagenet=raw, device="cpu", limit=2)
    assert len(config.make()) == 3


def test_non_finite_encoder_output_is_refused_before_coding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_MeanAutoencoder, "encode", _non_finite_encode)
    config = _source(tmp_path, codec=ScalarTableCodec.Config(num_fit_images=1))
    with pytest.raises(ValueError, match="non-finite"):
        prepare_data.prepare(
            config,
            imagenet=_imagenet(tmp_path, count=1),
            device="cpu",
        )
    assert not table_path(Path(config.working_dir) / config.latent_subdir).exists()


def _non_finite_encode(model: nn.Module, images: Tensor) -> PointMass:
    del model
    return PointMass(torch.full((images.shape[0], 4, *_GRID), float("nan")))


def test_seeded_posterior_is_stable_after_resume(tmp_path: Path) -> None:
    """A limit inside a window still encodes the whole window, as a full run does.

    At 4px the latent has 8 values, below the size where torch's CPU normal
    sampler draws a batch's first rows identically to a smaller batch.
    """
    raw = _imagenet(tmp_path, count=3)
    full, resumed = _source(tmp_path / "full"), _source(tmp_path / "resumed")
    full.autoencoder = tiny()
    full.autoencoder.image_size = 4
    resumed.autoencoder = full.autoencoder.copy_tree()
    full.seed = resumed.seed = 7
    prepare_data.prepare(full, imagenet=raw, device="cpu", batch_size=2)
    prepare_data.prepare(resumed, imagenet=raw, device="cpu", batch_size=2, limit=1)
    prepare_data.prepare(resumed, imagenet=raw, device="cpu", batch_size=2)
    for index in range(3):
        assert torch.equal(
            full.make()[index]["latent"],
            resumed.make()[index]["latent"],
        )


def test_an_unseeded_resume_encodes_with_the_same_random_weights(
    tmp_path: Path,
) -> None:
    """The mode is deterministic, so only the weights can differ."""
    raw = _imagenet(tmp_path, count=2)
    full, resumed = _source(tmp_path / "full"), _source(tmp_path / "resumed")
    full.autoencoder = tiny()
    full.sample = resumed.sample = False
    resumed.autoencoder = full.autoencoder.copy_tree()
    prepare_data.prepare(full, imagenet=raw, device="cpu", batch_size=1)
    prepare_data.prepare(resumed, imagenet=raw, device="cpu", batch_size=1, limit=1)
    _ = torch.rand(1)
    prepare_data.prepare(resumed, imagenet=raw, device="cpu", batch_size=1)
    assert torch.equal(full.make()[1]["latent"], resumed.make()[1]["latent"])


def test_resume_keeps_latents_already_saved_in_a_reencoded_window(
    tmp_path: Path,
) -> None:
    """Every save renames a new file into place, so a rewrite changes the inode."""
    raw = _imagenet(tmp_path, count=2)
    config = _source(tmp_path)
    _ = prepare_data.prepare(config, imagenet=raw, device="cpu", batch_size=2, limit=1)
    latent = Path(config.working_dir) / config.latent_subdir / "00000"
    before = (latent / "img-latents-00000000.npy").stat().st_ino
    _ = prepare_data.prepare(config, imagenet=raw, device="cpu", batch_size=2)
    assert (latent / "img-latents-00000000.npy").stat().st_ino == before
    assert (latent / "img-latents-00000001.npy").is_file()


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

    def fake(
        config: PairedImageLatentDataset.Config,
        imagenet: Path,
        **_: object,
    ) -> int:
        called.append((config, imagenet))
        return 0

    raw = tmp_path / "raw"
    for synset in _synsets():
        (raw / "train" / synset).mkdir(parents=True)
    monkeypatch.setattr(prepare_data, "prepare", fake)
    monkeypatch.setattr(
        "sys.argv",
        ["prepare_data", "--source", str(raw), "--directory", str(tmp_path / "c")],
    )
    assert prepare_data.main() == 0
    config, source = called[0]
    assert source == raw
    assert Path(config.working_dir) == tmp_path / "c"
    assert cast(object, config.autoencoder) is not None


def test_main_refuses_a_partial_imagenet_before_preparing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train = tmp_path / "raw" / "train"
    for synset in _synsets()[1:]:
        (train / synset).mkdir(parents=True)
    (train / "n00000000").mkdir()
    prepare = Mock()
    monkeypatch.setattr(prepare_data, "prepare", prepare)
    monkeypatch.setattr(
        "sys.argv",
        ["prepare_data", "--source", str(tmp_path / "raw"), "--directory", "out"],
    )

    with pytest.raises(ValueError, match="1 are missing and 1 are unexpected"):
        _ = prepare_data.main()
    prepare.assert_not_called()


def test_main_refuses_a_corpus_directory_over_its_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    prepare = Mock()
    monkeypatch.setattr(prepare_data, "prepare", prepare)
    monkeypatch.setattr(
        "sys.argv",
        ["prepare_data", "--source", str(raw), "--directory", str(raw)],
    )

    with pytest.raises(ValueError, match="aliases protected input"):
        _ = prepare_data.main()
    prepare.assert_not_called()
    assert list(raw.iterdir()) == []


def test_main_records_a_receipt_without_encoding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    called: list[tuple[Path, Path | None]] = []

    def fake(
        config: PairedImageLatentDataset.Config,
        imagenet: Path | None = None,
    ) -> int:
        called.append((Path(config.working_dir), imagenet))
        return 5

    prepare = Mock()
    monkeypatch.setattr(prepare_data, "record_receipt", fake)
    monkeypatch.setattr(prepare_data, "prepare", prepare)
    monkeypatch.setattr(
        "sys.argv",
        ["prepare_data", "--receipt-only", "--directory", str(tmp_path / "c")],
    )
    assert prepare_data.main() == 0
    assert called == [(tmp_path / "c", None)]
    assert capsys.readouterr().out == "5\n"
    prepare.assert_not_called()


def test_receipt_only_needs_the_latent_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="vae-in does not exist"):
        prepare_data.record_receipt(_source(tmp_path))


def test_a_source_without_training_images_is_refused(tmp_path: Path) -> None:
    (tmp_path / "raw" / "train").mkdir(parents=True)
    with pytest.raises(ValueError, match="holds no training images"):
        prepare_data.prepare(_source(tmp_path), imagenet=tmp_path / "raw")


@pytest.mark.parametrize(
    ("sample", "message"),
    [
        ({"label": "n00000000", "file_path": "a.JPEG"}, "integer ImageNet label"),
        ({"label": 3, "file_path": None}, "ImageNet file path"),
    ],
)
def test_records_refuse_an_untyped_label_or_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sample: dict[str, object],
    message: str,
) -> None:
    monkeypatch.setattr(
        prepare_data,
        "ExtractedImageNetSource",
        SimpleNamespace(Config=partial(_FakeSourceConfig, sample=sample)),
    )
    with pytest.raises(TypeError, match=message):
        prepare_data.records(tmp_path)


class _FakeSourceConfig:
    """Builds a source yielding one fixed sample."""

    def __init__(self, *, sample: dict[str, object], **_: object) -> None:
        self.sample = sample

    def make(self) -> list[dict[str, object]]:
        return [self.sample]


def test_a_stored_crop_of_the_wrong_size_is_refused(tmp_path: Path) -> None:
    stored = tmp_path / "images" / "00000" / "img00000000.png"
    stored.parent.mkdir(parents=True)
    Image.new("RGB", (_SIZE, _SIZE + 1)).save(stored)
    record = prepare_data.Record(source="unused.JPEG", label=0, stem="00000000")
    with pytest.raises(ValueError, match=rf"autoencoder needs {_SIZE}x{_SIZE}"):
        list(prepare_data.batches([record], root=tmp_path, size=_SIZE, batch_size=2))


def test_an_empty_error_tally_reports_nothing_measured() -> None:
    assert prepare_data._ErrorTally().summary() == {
        "mse": None,
        "max_abs": None,
        "saturated_fraction": None,
    }


def test_main_help_keeps_every_description_line(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(prepare_data, "__doc__", "a\nb\nfirst line\nsecond line\n")
    monkeypatch.setattr("sys.argv", ["prepare_data", "--help"])

    with pytest.raises(SystemExit):
        _ = prepare_data.main()

    assert "first line\nsecond line\n" in capsys.readouterr().out


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

    def encode(self, image: Tensor, /) -> PointMass:
        """Return ``[B, 4, 2, 3]`` pooled channel means."""
        means = functional.adaptive_avg_pool2d(image.float(), _GRID)
        return PointMass(torch.cat([means, means[:, :1]], dim=1))

    def decode(self, latent: Tensor, /) -> Tensor:
        """Return a flat image of the first three channels' means."""
        means = latent[:, :3].mean(dim=(2, 3), keepdim=True)
        return (means / 255).expand(-1, -1, _SIZE, _SIZE)

    @override
    def forward(self, image: Tensor) -> Tensor:
        return self.encode(image).mode()


def _gradient(size: tuple[int, int]) -> Image.Image:
    """Return an RGB image whose every pixel differs, so a misplaced crop shows."""
    image = Image.new("RGB", size)
    for x in range(image.width):
        for y in range(image.height):
            image.putpixel((x, y), (x * 18, y * 18, (x + y) * 9))
    return image


def _synsets() -> list[str]:
    """Return the 1,000 ImageNet synsets in the shipped label order."""
    labels = _CWD.parents[2] / "data/processors/labels_imagenet.txt"
    return labels.read_text(encoding="utf-8").splitlines()


def _imagenet(root: Path, count: int) -> Path:
    """Write ``count`` distinct training JPEGs under the first synset."""
    synset = _synsets()[0]
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
    return json_loads(path.read_text(), dict[str, object])


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
