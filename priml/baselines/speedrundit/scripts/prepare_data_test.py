"""Local ImageNet/INVAE preparation without the REG repository at runtime."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, cast

import argparse

from PIL import Image

import numpy as np
import pytest
import torch

from priml.baselines.speedrundit.data import PairedImageLatentDataset
from priml.baselines.speedrundit.scripts import prepare_data
from priml.data.processors.labels import ImagenetSynsetToIndex
from priml.data.sources.extracted_imagenet import ExtractedImageNetSource
from priml.lib.custom_json import convert, loads


if TYPE_CHECKING:
    from numpy.typing import NDArray


_CWD: Final = Path(__file__).resolve().parent


class _Flags(Protocol):
    source: Path
    output: Path
    resolution: int
    checkpoint: Path | None
    device: str
    limit: int | None


class _FakeVAE:
    def encode(self, image: torch.Tensor) -> object:
        class Posterior:
            def sample(self) -> torch.Tensor:
                # encode_image returns the fixed INVAE latent geometry.
                return torch.ones(image.shape[0], 32, 16, 16)

        return Posterior()


def _fake_load_invae(*_args: object, **_kwargs: object) -> _FakeVAE:
    return _FakeVAE()


def test_center_crop_box_downsamples_before_center_crop() -> None:
    image = Image.new("RGB", (768, 512))
    for x in range(768):
        for y in range(512):
            image.putpixel((x, y), (x % 256, y % 256, (x + y) % 256))

    cropped = prepare_data.center_crop(image, 256)
    expected = image.resize((384, 256), Image.Resampling.BOX).crop(
        (64, 0, 320, 256),
    )

    assert cropped.size == (256, 256)
    assert cropped.tobytes() == expected.tobytes()


def test_center_crop_resizes_short_axis_and_centers_both_axes() -> None:
    image = Image.new("RGB", (7, 5))
    for x in range(image.width):
        for y in range(image.height):
            image.putpixel((x, y), (x * 30, y * 40, 0))

    cropped = prepare_data.center_crop(image, 4)
    expected = image.resize((6, 4), Image.Resampling.BICUBIC).crop((1, 0, 5, 4))

    assert cropped.size == (4, 4)
    assert cropped.tobytes() == expected.tobytes()


@pytest.mark.parametrize(
    ("source_size", "resized_size", "crop_box"),
    [
        ((774, 512), (387, 256), (65, 0, 321, 256)),
        ((512, 774), (256, 387), (0, 65, 256, 321)),
    ],
)
def test_center_crop_floors_odd_center_offsets(
    source_size: tuple[int, int],
    resized_size: tuple[int, int],
    crop_box: tuple[int, int, int, int],
) -> None:
    image = Image.new("RGB", source_size)
    for x in range(image.width):
        for y in range(image.height):
            image.putpixel((x, y), (x % 256, y % 256, (x + y) % 256))
    resized = image.resize(resized_size, Image.Resampling.BOX)
    expected = resized.crop(crop_box)

    assert prepare_data.center_crop(image, 256).tobytes() == expected.tobytes()


def test_prepare_forwards_checkpoint_device_and_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    synsets = _CWD.parents[2] / "data/processors/labels_imagenet.txt"
    first_class = synsets.read_text(encoding="utf-8").splitlines()[0]
    image_dir = tmp_path / "raw" / "train" / first_class
    image_dir.mkdir(parents=True)
    for image_id in (1, 2):
        Image.new("RGB", (320, 280), (image_id, 64, 32)).save(
            image_dir / f"{first_class}_{image_id}.JPEG",
        )
    checkpoint = tmp_path / "checkpoint.pt"
    captured: dict[str, object] = {}

    def load(checkpoint: Path | None, *, device: str) -> _FakeVAE:
        captured["checkpoint"] = checkpoint
        captured["device"] = device
        del checkpoint
        return _FakeVAE()

    monkeypatch.setattr(prepare_data, "load_invae", load)
    output = tmp_path / "prepared"

    assert (
        prepare_data.prepare(
            tmp_path / "raw",
            output,
            checkpoint=checkpoint,
            device="cpu",
            limit=1,
        )
        == 1
    )
    assert captured == {"checkpoint": checkpoint, "device": "cpu"}
    assert (output / "images/00000/img00000000.png").exists()
    assert not (output / "images/00000/img00000001.png").exists()


@pytest.mark.parametrize("resolution", [255, 257, 511, 513])
def test_prepare_rejects_unsupported_resolution(
    tmp_path: Path,
    resolution: int,
) -> None:
    with pytest.raises(ValueError, match=r"^resolution must be 256 or 512$"):
        prepare_data.prepare(tmp_path, tmp_path / "out", resolution=resolution)


def test_add_arguments_parses_paths_choices_and_defaults(tmp_path: Path) -> None:
    parser = argparse.ArgumentParser()
    prepare_data._add_arguments(parser)

    source = tmp_path / "source"
    output = tmp_path / "out"
    flags = cast(
        _Flags,
        parser.parse_args(["--source", str(source), "--output", str(output)]),
    )
    assert isinstance(flags.source, Path)
    assert isinstance(flags.output, Path)
    assert flags.source == source
    assert flags.output == output
    assert isinstance(flags.source, Path)
    assert isinstance(flags.output, Path)
    assert isinstance(flags.resolution, int)
    assert flags.resolution == 256
    assert flags.checkpoint is None
    assert isinstance(flags.device, str)
    assert flags.device == "cuda"
    assert flags.limit is None
    with pytest.raises(SystemExit):
        parser.parse_args(["--source", str(source)])
    with pytest.raises(SystemExit):
        parser.parse_args(["--output", str(output)])
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["--source", str(source), "--output", str(output), "--resolution", "257"],
        )


def test_main_forwards_arguments_and_reports_prepared_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "out"
    checkpoint = tmp_path / "invae.pt"
    monkeypatch.setattr(
        "sys.argv",
        [
            "prepare_data",
            "--source",
            str(source),
            "--output",
            str(output),
            "--resolution",
            "512",
            "--checkpoint",
            str(checkpoint),
            "--device",
            "cpu",
            "--limit",
            "3",
        ],
    )
    captured: dict[str, object] = {}

    def prepare(
        actual_source: Path,
        actual_output: Path,
        *,
        resolution: int,
        checkpoint: Path | None,
        device: str,
        limit: int | None,
    ) -> int:
        captured.update(
            source=actual_source,
            output=actual_output,
            resolution=resolution,
            checkpoint=checkpoint,
            device=device,
            limit=limit,
        )
        return 7

    monkeypatch.setattr(prepare_data, "prepare", prepare)

    assert prepare_data.main() == 0
    assert captured == {
        "source": source,
        "output": output,
        "resolution": 512,
        "checkpoint": checkpoint,
        "device": "cpu",
        "limit": 3,
    }
    assert capsys.readouterr().out == "7\n"


@pytest.mark.parametrize(
    ("record", "message"),
    [
        (
            {"label": "bad", "file_path": "unused"},
            "expected an integer ImageNet label: {'label': 'bad', 'file_path': 'unused'}",
        ),
        (
            {"label": 3, "file_path": 7},
            "expected an ImageNet file path: {'label': 3, 'file_path': 7}",
        ),
    ],
)
def test_prepare_rejects_untyped_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    record: dict[str, object],
    message: str,
) -> None:
    def source_make(config: ExtractedImageNetSource.Config) -> object:
        del config
        return [record]

    def labels_make(config: ImagenetSynsetToIndex.Config) -> object:
        del config

        def identity(records: object) -> object:
            return records

        return identity

    monkeypatch.setattr(ExtractedImageNetSource.Config, "make", source_make)
    monkeypatch.setattr(ImagenetSynsetToIndex.Config, "make", labels_make)
    monkeypatch.setattr(prepare_data, "load_invae", _fake_load_invae)

    with pytest.raises(TypeError) as error:
        prepare_data.prepare(tmp_path, tmp_path / "out", device="cpu")
    assert str(error.value) == message


def test_prepare_creates_manifest_for_empty_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "raw"
    (source / "train").mkdir(parents=True)
    captured: dict[str, object] = {}

    def load(checkpoint: Path | None, *, device: str) -> _FakeVAE:
        captured.update(checkpoint=checkpoint, device=device)
        return _FakeVAE()

    monkeypatch.setattr(prepare_data, "load_invae", load)
    output = tmp_path / "prepared"

    assert prepare_data.prepare(source, output) == 0
    assert captured == {"checkpoint": None, "device": "cuda"}
    assert (output / "vae-in/dataset.json").read_bytes() == b'{"labels": []}'


def test_main_help_keeps_every_description_line(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(prepare_data, "__doc__", "a\nb\nfirst line\nsecond line\n")
    monkeypatch.setattr("sys.argv", ["prepare_data", "--help"])

    with pytest.raises(SystemExit):
        prepare_data.main()

    assert "first line\nsecond line\n" in capsys.readouterr().out


def test_main_help_includes_module_description(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("sys.argv", ["prepare_data", "--help"])

    with pytest.raises(SystemExit) as error:
        prepare_data.main()

    assert error.value.code == 0
    assert capsys.readouterr().out == (
        "usage: prepare_data [-h] --source SOURCE --output OUTPUT\n"
        "                    [--resolution {256,512}] [--checkpoint CHECKPOINT]\n"
        "                    [--device DEVICE] [--limit LIMIT]\n\n"
        "Crop extracted ImageNet and encode paired 32-channel INVAE latents.\n\n"
        "options:\n"
        "  -h, --help            show this help message and exit\n"
        "  --source SOURCE\n"
        "  --output OUTPUT\n"
        "  --resolution {256,512}\n"
        "  --checkpoint CHECKPOINT\n"
        "  --device DEVICE\n"
        "  --limit LIMIT\n"
    )


def test_preparer_writes_paired_source_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    synsets = _CWD.parents[2] / "data/processors/labels_imagenet.txt"
    first_class = synsets.read_text(encoding="utf-8").splitlines()[0]
    image_dir = tmp_path / "raw" / "train" / first_class
    image_dir.mkdir(parents=True)
    Image.new("L", (320, 280), 128).save(image_dir / f"{first_class}_1.JPEG")
    monkeypatch.setattr(prepare_data, "load_invae", _fake_load_invae)
    devices: list[torch.device] = []

    def encode(vae: object, image: torch.Tensor) -> torch.Tensor:
        del vae
        devices.append(image.device)
        # `prepare` encodes one image; INVAE latents are [1, 32, 16, 16] at 256px.
        return torch.ones(1, 32, 16, 16)

    monkeypatch.setattr(prepare_data, "encode_image", encode)
    output = tmp_path / "prepared"
    assert prepare_data.prepare(tmp_path / "raw", output, device="meta") == 1
    assert prepare_data.prepare(tmp_path / "raw", output, device="meta") == 1
    assert devices == [torch.device("meta"), torch.device("meta")]
    image_path = output / "images/00000/img00000000.png"
    latent_path = output / "vae-in/00000/img-latents-00000000.npy"
    assert image_path.exists()
    assert latent_path.exists()
    with Image.open(image_path) as prepared_image:
        assert prepared_image.size == (256, 256)
        assert prepared_image.mode == "RGB"
        assert prepared_image.getpixel((128, 128)) == (128, 128, 128)
    latent = cast("NDArray[np.float32]", np.load(latent_path))
    assert latent.shape == (1, 32, 16, 16)
    manifest = convert(
        loads((output / "vae-in/dataset.json").read_text()),
        dict[str, object],
    )
    assert manifest["labels"] == [["00000/img-latents-00000000.npy", 0]]
    dataset = PairedImageLatentDataset.Config(working_dir=output).make()
    assert len(dataset) == 1
    assert dataset[0]["label"] == 0


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
