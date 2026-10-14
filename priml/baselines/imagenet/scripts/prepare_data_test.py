from __future__ import annotations

from pathlib import Path
from typing import Final, Self
from unittest.mock import patch
from urllib import request

import argparse
import hashlib
import logging
import sys

import pytest

from priml.baselines.imagenet.data import NUM_CLASSES, NUM_TRAIN_SAMPLES
from priml.baselines.imagenet.scripts import prepare_data


SYNSETS: Final = ("n00000000", "n00000001", "n00000002")
VAL_IMAGES: Final = 5
TRAIN_IMAGES: Final = 4
LABELS_TEXT: Final = "".join(
    f"{SYNSETS[index % len(SYNSETS)]}\n" for index in range(VAL_IMAGES)
)


@pytest.fixture
def image_net(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Build the layout at shrunk counts; the real 1.3M entries cannot be a fixture."""
    monkeypatch.setattr(prepare_data, "NUM_VAL_IMAGES", VAL_IMAGES)
    monkeypatch.setattr(prepare_data, "NUM_TRAIN_SAMPLES", TRAIN_IMAGES)
    monkeypatch.setattr(
        prepare_data,
        "VALIDATION_LABELS_SHA256",
        hashlib.sha256(LABELS_TEXT.encode()).hexdigest(),
    )
    monkeypatch.setattr(
        prepare_data,
        "_download_text",
        {prepare_data.VALIDATION_LABELS_URL: LABELS_TEXT}.__getitem__,
    )
    for index, synset in enumerate(SYNSETS):
        (tmp_path / "train" / synset).mkdir(parents=True)
        # Two images in the first class and one in each other: four in all.
        for image in range(2 if index == 0 else 1):
            (tmp_path / "train" / synset / f"{synset}_{image}.JPEG").touch()
    val = tmp_path / "val"
    val.mkdir()
    for index in range(VAL_IMAGES):
        (val / f"ILSVRC2012_val_{index + 1:08d}.JPEG").touch()
    return tmp_path


def test_counts_are_the_published_ilsvrc2012_sizes() -> None:
    assert (NUM_CLASSES, NUM_TRAIN_SAMPLES, prepare_data.NUM_VAL_IMAGES) == (
        1000,
        1_281_167,
        50_000,
    )


def test_prepare_writes_validation_labels_atomically(
    image_net: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    with (
        patch.object(
            prepare_data,
            "_download_text",
            return_value=LABELS_TEXT,
        ) as download,
        patch.object(Path, "replace", autospec=True, wraps=Path.replace) as replace,
    ):
        prepare_data.prepare(image_net)

    labels = image_net / "validation_labels.txt"
    assert labels.read_text() == LABELS_TEXT
    assert not (image_net / "validation_labels.partial").exists()
    download.assert_called_once_with(prepare_data.VALIDATION_LABELS_URL)
    assert replace.call_args.args == (
        image_net / "validation_labels.partial",
        labels,
    )
    assert caplog.messages == [f"{image_net} is ready."]


def test_prepare_verifies_and_keeps_existing_labels(image_net: Path) -> None:
    labels = image_net / "validation_labels.txt"
    labels.write_text(LABELS_TEXT)
    with patch.object(prepare_data, "_download_text") as download:
        prepare_data.prepare(image_net)

    download.assert_not_called()
    assert labels.read_text() == LABELS_TEXT


def test_prepare_rejects_existing_labels_failing_the_checksum(image_net: Path) -> None:
    labels = image_net / "validation_labels.txt"
    labels.write_text("existing labels\n")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        prepare_data.prepare(image_net)
    assert labels.read_text() == "existing labels\n"


def test_prepare_rejects_downloaded_labels_failing_the_checksum(
    image_net: Path,
) -> None:
    with (
        patch.object(prepare_data, "_download_text", return_value="one label"),
        pytest.raises(ValueError, match="SHA-256 mismatch"),
    ):
        prepare_data.prepare(image_net)
    assert not (image_net / "validation_labels.txt").exists()


def test_prepare_rejects_missing_split(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="train is missing"):
        prepare_data.prepare(tmp_path)


def test_prepare_rejects_a_misnamed_synset(image_net: Path) -> None:
    (image_net / "train" / SYNSETS[-1]).rename(image_net / "train" / "n99999999")

    with pytest.raises(ValueError, match=r"Missing: \['n00000002'\]"):
        prepare_data.prepare(image_net)
    assert not (image_net / "validation_labels.txt").exists()


def test_prepare_rejects_a_missing_train_image(image_net: Path) -> None:
    (image_net / "train" / SYNSETS[0] / f"{SYNSETS[0]}_1.JPEG").unlink()

    with pytest.raises(ValueError, match=f"holds {TRAIN_IMAGES - 1} images"):
        prepare_data.prepare(image_net)


def test_prepare_rejects_an_extra_validation_image(image_net: Path) -> None:
    (image_net / "val" / "extra.JPEG").touch()

    with pytest.raises(ValueError, match=f"holds {VAL_IMAGES + 1} images"):
        prepare_data.prepare(image_net)


def test_prepare_rejects_a_misnamed_validation_image(image_net: Path) -> None:
    val = image_net / "val"
    (val / "ILSVRC2012_val_00000001.JPEG").rename(val / "ILSVRC2012_val_00000099.JPEG")

    with pytest.raises(ValueError, match=r"unexpected: \['ILSVRC2012_val_00000099"):
        prepare_data.prepare(image_net)


def test_default_directory_is_from_default_train_loop() -> None:
    directory = prepare_data.default_directory()

    assert directory == Path("/opt/scratch/datasets/imagenet")


def test_download_text_decodes_utf8() -> None:
    class Response:
        def __enter__(self) -> Self:
            return self

        def __exit__(self, *args: object) -> None:
            del args

        def read(self) -> bytes:
            return "synset é".encode()

    with patch.object(
        request,
        "urlopen",
        return_value=Response(),
    ) as urlopen:
        assert prepare_data._download_text("https://example.test") == "synset é"

    urlopen.assert_called_once_with("https://example.test")


def test_add_arguments_registers_directory(tmp_path: Path) -> None:
    parser = argparse.ArgumentParser()
    with patch.object(prepare_data, "default_directory", return_value=tmp_path):
        prepare_data._add_arguments(parser)

    assert vars(parser.parse_args([])) == {"directory": tmp_path}
    assert vars(parser.parse_args(["--directory", str(tmp_path / "other")])) == {
        "directory": tmp_path / "other",
    }


def test_main_prepares_the_selected_directory(tmp_path: Path) -> None:
    with (
        patch.object(sys, "argv", ["prepare_data"]),
        patch.object(prepare_data, "default_directory", return_value=tmp_path),
        patch.object(prepare_data, "prepare") as prepare,
        patch.object(logging, "basicConfig") as basic_config,
    ):
        assert prepare_data.main() == 0

    prepare.assert_called_once_with(tmp_path)
    basic_config.assert_called_once_with(level=logging.INFO, format="%(message)s")


def test_main_help_shows_description(capsys: pytest.CaptureFixture[str]) -> None:
    with (
        patch.object(sys, "argv", ["prepare_data", "--help"]),
        pytest.raises(SystemExit, match="0"),
    ):
        prepare_data.main()

    assert (
        "Verify an extracted ImageNet and write its validation labels beside it."
        in capsys.readouterr().out.splitlines()
    )


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
