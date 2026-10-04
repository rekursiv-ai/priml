from __future__ import annotations

from pathlib import Path
from typing import Self
from unittest.mock import patch
from urllib import request

import argparse
import logging
import sys

import pytest

from priml.baselines.imagenet.data import NUM_CLASSES
from priml.baselines.imagenet.scripts import prepare_data


_CLASSES = 3
_VAL_IMAGES = 5


@pytest.fixture
def image_net(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Build the layout at shrunk counts; the real 51k entries took 1.7s per test."""
    monkeypatch.setattr(prepare_data, "NUM_CLASSES", _CLASSES)
    monkeypatch.setattr(prepare_data, "NUM_VAL_IMAGES", _VAL_IMAGES)
    (tmp_path / "train").mkdir()
    for index in range(_CLASSES):
        (tmp_path / "train" / f"n{index:08d}").mkdir()
    val = tmp_path / "val"
    val.mkdir()
    for index in range(_VAL_IMAGES):
        (val / f"ILSVRC2012_val_{index + 1:08d}.JPEG").touch()
    return tmp_path


def test_counts_are_the_published_ilsvrc2012_sizes() -> None:
    assert (NUM_CLASSES, prepare_data.NUM_VAL_IMAGES) == (1000, 50_000)


def test_prepare_writes_validation_labels_atomically(
    image_net: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    labels_text = "n01440764\n" * _VAL_IMAGES
    with (
        patch.object(
            prepare_data,
            "_download_text",
            return_value=labels_text,
        ) as download,
        patch.object(Path, "replace", autospec=True, wraps=Path.replace) as replace,
    ):
        prepare_data.prepare(image_net)

    labels = image_net / "validation_labels.txt"
    assert labels.read_text() == labels_text
    assert not (image_net / "validation_labels.partial").exists()
    download.assert_called_once_with(prepare_data.VALIDATION_LABELS_URL)
    assert replace.call_args.args == (
        image_net / "validation_labels.partial",
        labels,
    )
    assert caplog.messages == [f"{image_net} is ready."]


def test_prepare_leaves_existing_labels_untouched(image_net: Path) -> None:
    labels = image_net / "validation_labels.txt"
    labels.write_text("existing labels\n")
    with patch.object(prepare_data, "_download_text") as download:
        prepare_data.prepare(image_net)

    download.assert_not_called()
    assert labels.read_text() == "existing labels\n"


def test_prepare_rejects_missing_split(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="train is missing"):
        prepare_data.prepare(tmp_path)


def test_prepare_rejects_wrong_class_count(tmp_path: Path) -> None:
    (tmp_path / "train").mkdir()
    (tmp_path / "train" / "n00000001").mkdir()
    (tmp_path / "val").mkdir()

    with pytest.raises(ValueError, match="train/ holds 1 synsets"):
        prepare_data.prepare(tmp_path)


def test_prepare_rejects_wrong_validation_count(image_net: Path) -> None:
    (image_net / "val" / "extra.JPEG").touch()

    with pytest.raises(ValueError, match=f"val/ holds {_VAL_IMAGES + 1} images"):
        prepare_data.prepare(image_net)


def test_prepare_rejects_wrong_download_count(image_net: Path) -> None:
    with (
        patch.object(prepare_data, "_download_text", return_value="one label"),
        pytest.raises(ValueError, match=f"is not {_VAL_IMAGES} lines"),
    ):
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
    from priml.lib.testing.main import test_main

    test_main(__file__)
