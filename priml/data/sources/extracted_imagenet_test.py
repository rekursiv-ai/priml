"""Tests for the extracted-archive ImageNet source."""

from __future__ import annotations

from pathlib import Path

import pytest

from priml.data.sources.extracted_imagenet import ExtractedImageNetSource


def test_extracted_imagenet_default_is_opinionated() -> None:
    assert ExtractedImageNetSource.Config().working_dir == "/datasets/imagenet"


def _write_extracted_imagenet(root: Path) -> None:
    """Lay out two train synsets and three val images with a labels file."""
    for synset, count in (("n01440764", 2), ("n01443537", 1)):
        class_dir = root / "train" / synset
        class_dir.mkdir(parents=True)
        for i in range(count):
            (class_dir / f"{synset}_{i}.JPEG").write_bytes(b"jpeg")
    (root / "train" / "stray.txt").write_text("not a class dir")
    val_dir = root / "val"
    val_dir.mkdir()
    for i in range(3):
        (val_dir / f"ILSVRC2012_val_{i:08d}.JPEG").write_bytes(b"jpeg")
    (root / "validation_labels.txt").write_text("n01440764\nn01443537\nn01484850\n")


def test_extracted_imagenet_train_walks_sorted_synsets(tmp_path: Path) -> None:
    _write_extracted_imagenet(tmp_path)

    source = ExtractedImageNetSource.Config(working_dir=tmp_path).make()
    samples = list(source)

    assert [s.get("key") for s in samples] == [
        "n01440764_0",
        "n01440764_1",
        "n01443537_0",
    ]
    assert samples[0] == {
        "key": "n01440764_0",
        "file_path": str(tmp_path / "train" / "n01440764" / "n01440764_0.JPEG"),
        "format": "jpg",
        "label": "n01440764",
        "frames": 1,
    }
    assert len(source) == 1_281_167


def test_extracted_imagenet_train_shards_and_shuffles_synsets(tmp_path: Path) -> None:
    _write_extracted_imagenet(tmp_path)
    config = ExtractedImageNetSource.Config(working_dir=tmp_path)
    config.shuffle = True

    config.worker_slice = (0, 2)
    config.epoch_seed = 0
    first = [s.get("label") for s in config.make()]
    config.worker_slice = (1, 2)
    second = [s.get("label") for s in config.make()]

    # Both workers share one permutation, so their slices partition the classes.
    assert sorted({str(label) for label in (*first, *second)}) == [
        "n01440764",
        "n01443537",
    ]
    assert set(first).isdisjoint(second)
    # A later epoch reshuffles: with two classes, seed 1 flips their order.
    config.worker_slice = None
    config.epoch_seed = 1
    reshuffled = [s.get("label") for s in config.make()]
    assert reshuffled == ["n01443537", "n01440764", "n01440764"]


def test_extracted_imagenet_val_reads_labels_relative_to_the_dataset(
    tmp_path: Path,
) -> None:
    _write_extracted_imagenet(tmp_path)
    config = ExtractedImageNetSource.Config(working_dir=tmp_path)
    config.split = "val"
    config.validation_labels_file = Path("validation_labels.txt")

    source = config.make()
    samples = list(source)

    assert [s.get("label") for s in samples] == ["n01440764", "n01443537", "n01484850"]
    assert samples[0].get("key") == "ILSVRC2012_val_00000000"
    assert samples[0].get("format") == "jpg"
    assert len(source) == 50_000


def test_extracted_imagenet_val_without_labels_reports_unknown(tmp_path: Path) -> None:
    _write_extracted_imagenet(tmp_path)
    config = ExtractedImageNetSource.Config(working_dir=tmp_path)
    config.split = "val"
    config.worker_slice = (1, 2)

    samples = list(config.make())

    # Val is never shuffled: worker 1 of 2 takes the sorted tail.
    assert [s.get("key") for s in samples] == [
        "ILSVRC2012_val_00000001",
        "ILSVRC2012_val_00000002",
    ]
    assert {s.get("label") for s in samples} == {"unknown"}


def test_extracted_imagenet_val_labels_absent_for_a_file_fall_back_to_unknown(
    tmp_path: Path,
) -> None:
    _write_extracted_imagenet(tmp_path)
    config = ExtractedImageNetSource.Config(working_dir=tmp_path)
    config.split = "val"
    config.validation_labels_file = tmp_path / "validation_labels.txt"
    source = config.make()
    (tmp_path / "val" / "ILSVRC2012_val_00000009.JPEG").write_bytes(b"jpeg")

    labels = [s.get("label") for s in source]

    assert labels == ["n01440764", "n01443537", "n01484850", "unknown"]


def test_extracted_imagenet_rejects_a_label_count_mismatch(tmp_path: Path) -> None:
    _write_extracted_imagenet(tmp_path)
    labels = tmp_path / "short.txt"
    labels.write_text("n01440764\n")
    config = ExtractedImageNetSource.Config(working_dir=tmp_path)
    config.split = "val"
    config.validation_labels_file = labels

    with pytest.raises(ValueError, match="Mismatch: 3 files but 1 labels"):
        _ = config.make()


def test_extracted_imagenet_rejects_missing_directories_and_unknown_splits(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="Dataset directory does not exist"):
        _ = ExtractedImageNetSource.Config(working_dir=tmp_path / "nope").make()

    (tmp_path / "train").mkdir()
    val_only = ExtractedImageNetSource.Config(working_dir=tmp_path)
    val_only.split = "val"
    with pytest.raises(ValueError, match="Split directory does not exist"):
        _ = val_only.make()

    bogus = ExtractedImageNetSource.Config(working_dir=tmp_path)
    bogus.split = "test"  # ty: ignore[invalid-assignment] -- The runtime guard exists for unchecked config text.  # pyright: ignore[reportAttributeAccessIssue] -- Same.
    with pytest.raises(ValueError, match="Unknown split: test"):
        _ = bogus.make()


def test_extracted_imagenet_accepts_resolved_working_dir(tmp_path: Path) -> None:
    (tmp_path / "train").mkdir(parents=True)

    source = ExtractedImageNetSource.Config(working_dir=tmp_path).make()

    assert source.dataset_dir == tmp_path


def test_extracted_imagenet_explicit_path_is_literal(tmp_path: Path) -> None:
    working_dir = tmp_path / "{scratch_dir}" / "imagenet"
    config = ExtractedImageNetSource.Config(working_dir=working_dir).finalize()

    assert config.working_dir == working_dir


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
