"""Tests for staging the throughput image set."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Final
from unittest.mock import patch

import io
import os
import sys

from PIL import Image

import numpy as np
import pytest
import torch

from priml.baselines.imagenet.throughput.scripts import (
    prepare_throughput_data,
)
from priml.testing.golden import assert_tensor_golden


if TYPE_CHECKING:
    from numpy.typing import NDArray


_CWD: Final = Path(__file__).resolve().parent
GOLDEN: Final = _CWD / "testdata" / "synthetic_pixels.pt"
SYNSETS: Final = ("n00000000", "n00000001", "n00000002")
SIDES: Final = (5, 9)
"""Small synthetic images: the recipe is the same at any size."""


def _source(root: Path, counts: tuple[int, ...]) -> Path:
    """Write ``counts[i]`` distinct fake JPEGs under synset ``i``."""
    for synset, count in zip(SYNSETS, counts, strict=False):
        (root / "train" / synset).mkdir(parents=True)
        for index in range(count):
            image = root / "train" / synset / f"{synset}_{index}.JPEG"
            _ = image.write_bytes(f"{synset}/{index}".encode())
    return root


def _staged(train: Path) -> dict[str, list[bytes]]:
    """Return each staged synset's image bytes, read through the links."""
    return {
        synset.name: [image.read_bytes() for image in sorted(synset.iterdir())]
        for synset in sorted(train.iterdir())
    }


def test_link_subset_takes_images_round_robin_over_every_synset(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path / "imagenet", (3, 3, 3))

    prepare_throughput_data.link_subset(
        source / "train",
        tmp_path / "train",
        num_images=4,
    )

    assert _staged(tmp_path / "train") == {
        "n00000000": [b"n00000000/0", b"n00000000/1"],
        "n00000001": [b"n00000001/0"],
        "n00000002": [b"n00000002/0"],
    }


def test_link_subset_fills_a_short_synset_from_the_others(tmp_path: Path) -> None:
    source = _source(tmp_path / "imagenet", (1, 3))

    prepare_throughput_data.link_subset(
        source / "train",
        tmp_path / "train",
        num_images=4,
    )

    assert _staged(tmp_path / "train") == {
        "n00000000": [b"n00000000/0"],
        "n00000001": [b"n00000001/0", b"n00000001/1", b"n00000001/2"],
    }


def test_link_subset_refuses_a_source_with_too_few_images(tmp_path: Path) -> None:
    source = _source(tmp_path / "imagenet", (1, 1, 1))

    with pytest.raises(ValueError, match=r"holds 3 images; 4 requested\."):
        prepare_throughput_data.link_subset(
            source / "train",
            tmp_path / "train",
            num_images=4,
        )


def test_link_subset_resolves_a_relative_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ = _source(tmp_path / "imagenet", (1,))
    monkeypatch.chdir(tmp_path)

    prepare_throughput_data.link_subset(
        Path("imagenet/train"),
        tmp_path / "staged" / "train",
        num_images=1,
    )

    assert _staged(tmp_path / "staged" / "train") == {
        "n00000000": [b"n00000000/0"],
    }


def test_synthetic_pixels_are_pinned_by_the_seed() -> None:
    rng = np.random.default_rng(0)

    record = {
        f"pixels{i}": torch.from_numpy(
            prepare_throughput_data.synthetic_pixels(rng, sides=SIDES),
        )
        for i in range(2)
    }

    assert_tensor_golden(GOLDEN, record)


REAL_SYNTHETIC_PIXELS: Final = prepare_throughput_data.synthetic_pixels


def _recording_pixels(
    sides_seen: list[tuple[int, int]],
    rng: np.random.Generator,
    *,
    sides: tuple[int, int],
) -> NDArray[np.uint8]:
    sides_seen.append(sides)
    return REAL_SYNTHETIC_PIXELS(rng, sides=SIDES)


def test_write_synthetic_defaults_to_imagenet_sizes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sides_seen: list[tuple[int, int]] = []
    monkeypatch.setattr(
        prepare_throughput_data,
        "synthetic_pixels",
        partial(_recording_pixels, sides_seen),
    )

    prepare_throughput_data.write_synthetic(tmp_path, num_images=2, num_classes=1)

    assert sides_seen == [(300, 500), (300, 500)]


def test_write_synthetic_writes_seeded_jpegs_in_label_order(tmp_path: Path) -> None:
    prepare_throughput_data.write_synthetic(
        tmp_path,
        num_images=3,
        num_classes=2,
        sides=SIDES,
    )

    rng = np.random.default_rng(0)
    for name in (
        "n01440764/n01440764_0.JPEG",
        "n01443537/n01443537_1.JPEG",
        "n01440764/n01440764_2.JPEG",
    ):
        want = io.BytesIO()
        pixels = prepare_throughput_data.synthetic_pixels(rng, sides=SIDES)
        Image.fromarray(pixels).save(
            want,
            format="JPEG",
            quality=90,
        )
        assert (tmp_path / name).read_bytes() == want.getvalue(), name
    assert len(list(tmp_path.rglob("*.JPEG"))) == 3


def test_stage_writes_the_synthetic_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        prepare_throughput_data,
        "synthetic_pixels",
        partial(_recording_pixels, []),
    )

    train = prepare_throughput_data.stage(
        tmp_path,
        source=None,
        num_images=3,
        num_classes=2,
    )

    assert train == tmp_path / "train"
    assert [len(images) for images in _staged(train).values()] == [2, 1]
    assert list(tmp_path.iterdir()) == [train]
    assert train.stat().st_mode & 0o777 == 0o755


def _write_one_then_fail(train: Path, *, num_images: int, num_classes: int) -> None:
    del num_images, num_classes
    (train / "n00000000").mkdir(parents=True)
    _ = (train / "n00000000" / "partial.JPEG").write_bytes(b"partial")
    raise KeyboardInterrupt


def test_stage_leaves_no_train_directory_when_interrupted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(tmp_path / "imagenet", (1,))
    staging = tmp_path / "staging"
    with monkeypatch.context() as patched:
        patched.setattr(
            prepare_throughput_data,
            "write_synthetic",
            _write_one_then_fail,
        )
        with pytest.raises(KeyboardInterrupt):
            _ = prepare_throughput_data.stage(
                staging,
                source=None,
                num_images=1,
                num_classes=1,
            )
    assert list(staging.iterdir()) == []

    train = prepare_throughput_data.stage(
        staging,
        source=source,
        num_images=1,
        num_classes=1,
    )

    assert _staged(train) == {"n00000000": [b"n00000000/0"]}


def test_stage_never_deletes_what_another_run_is_building(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    other = tmp_path / "train.partial"
    (other / "n00000000").mkdir(parents=True)
    marker = other / "n00000000" / "theirs.JPEG"
    _ = marker.write_bytes(b"theirs")
    monkeypatch.setattr(
        prepare_throughput_data,
        "write_synthetic",
        _write_one_then_fail,
    )

    with pytest.raises(KeyboardInterrupt):
        _ = prepare_throughput_data.stage(
            tmp_path,
            source=None,
            num_images=1,
            num_classes=1,
        )

    assert list(tmp_path.iterdir()) == [other]
    assert marker.read_bytes() == b"theirs"


def _stage_synthetic(directory: Path, num_images: int) -> Path:
    return prepare_throughput_data.stage(
        directory,
        source=None,
        num_images=num_images,
        num_classes=1,
    )


def test_stage_leaves_a_finished_set_alone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        prepare_throughput_data,
        "synthetic_pixels",
        partial(_recording_pixels, []),
    )
    train = _stage_synthetic(tmp_path, 2)
    before = _staged(train)
    monkeypatch.setattr(
        prepare_throughput_data,
        "write_synthetic",
        _write_one_then_fail,
    )

    assert _stage_synthetic(tmp_path, 2) == train
    assert _staged(train) == before


@pytest.mark.parametrize(
    ("source", "num_images", "match"),
    [
        (False, 3, r"holds 2 images, not 3 synthetic ones; move it aside"),
        (True, 2, r"holds 2 images, not 2 linked ones; move it aside"),
    ],
)
def test_stage_refuses_a_finished_set_of_another_kind_or_size(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source: bool,
    num_images: int,
    match: str,
) -> None:
    monkeypatch.setattr(
        prepare_throughput_data,
        "synthetic_pixels",
        partial(_recording_pixels, []),
    )
    _ = _stage_synthetic(tmp_path / "set", 2)
    imagenet = _source(tmp_path / "imagenet", (3,))

    with pytest.raises(ValueError, match=match):
        _ = prepare_throughput_data.stage(
            tmp_path / "set",
            source=imagenet if source else None,
            num_images=num_images,
            num_classes=1,
        )


@pytest.mark.parametrize(
    ("num_images", "num_classes", "match"),
    [
        (0, 1, r"^num_images must be positive; got 0\.$"),
        (1, 0, r"^num_classes must lie in \[1, 1000\]; got 0\.$"),
        (1, 1001, r"^num_classes must lie in \[1, 1000\]; got 1001\.$"),
    ],
)
def test_stage_refuses_a_count_out_of_range_before_writing(
    tmp_path: Path,
    num_images: int,
    num_classes: int,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        _ = prepare_throughput_data.stage(
            tmp_path,
            source=None,
            num_images=num_images,
            num_classes=num_classes,
        )
    assert list(tmp_path.iterdir()) == []


def test_stage_links_regardless_of_the_synthetic_class_count(tmp_path: Path) -> None:
    source = _source(tmp_path / "imagenet", (1,))

    train = prepare_throughput_data.stage(
        tmp_path / "set",
        source=source,
        num_images=1,
        num_classes=0,
    )

    assert _staged(train) == {"n00000000": [b"n00000000/0"]}


def test_synsets_are_imagenets_in_label_order() -> None:
    synsets = prepare_throughput_data._synsets()

    assert (len(synsets), synsets[:2]) == (1000, ["n01440764", "n01443537"])


def test_stage_refuses_an_unusable_directory() -> None:
    with pytest.raises(ValueError, match="must not be root"):
        _ = prepare_throughput_data.stage(
            Path("/"),
            source=None,
            num_images=1,
            num_classes=1,
        )


def test_main_stages_the_selected_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        prepare_throughput_data,
        "synthetic_pixels",
        partial(_recording_pixels, []),
    )
    argv = [
        "prepare_throughput_data",
        "--directory",
        os.fspath(tmp_path),
        "--num-images",
        "1",
        "--num-classes",
        "1",
    ]
    with patch.object(sys, "argv", argv):
        assert prepare_throughput_data.main() == 0

    assert [len(images) for images in _staged(tmp_path / "train").values()] == [1]


def test_main_help_shows_description(capsys: pytest.CaptureFixture[str]) -> None:
    with (
        patch.object(sys, "argv", ["prepare_throughput_data", "--help"]),
        pytest.raises(SystemExit, match="0"),
    ):
        _ = prepare_throughput_data.main()

    assert (
        "Stage the fixed image set the throughput experiments time."
        in capsys.readouterr().out.splitlines()
    )


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
