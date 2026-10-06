"""Tests for ImageNet data source."""

from __future__ import annotations

from collections.abc import Sized
from functools import partialmethod
from pathlib import Path
from typing import TYPE_CHECKING, cast

import io
import logging
import tarfile
import tempfile

from PIL import Image

import pytest

from priml.data.sources.imagenet import (
    ImageNetSource,
    _read_class_tar,
)


if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import IO, Literal


def test_default_working_dir_is_opinionated() -> None:
    assert ImageNetSource.Config().working_dir == "/datasets/imagenet"


def test_owner_resolves_default_working_dir(tmp_path: Path) -> None:
    config = ImageNetSource.Config()
    config.base_dir = tmp_path

    with pytest.raises(ValueError, match="Dataset directory does not exist") as error:
        config.make()

    assert str(tmp_path / "datasets" / "imagenet") in str(error.value)


def test_explicit_path_working_dir_is_literal(tmp_path: Path) -> None:
    working_dir = tmp_path / "{scratch_dir}" / "imagenet"
    config = ImageNetSource.Config(working_dir=working_dir).finalize()

    assert config.working_dir == working_dir


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_image():
    """Create a mock PIL image."""
    return Image.new("RGB", (100, 101), color="red")


def create_test_tar(path: Path, files: dict[str, bytes]):
    """Create a tar file with given files."""
    with tarfile.open(path, "w") as tar:
        for name, content in files.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))


def create_nested_tar(path: Path, class_name: str, num_images: int = 2):
    """Create a nested tar structure for ImageNet training."""
    # Create class tar in memory.
    class_tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=class_tar_buffer, mode="w") as class_tar:
        for i in range(num_images):
            # Create a simple image.
            img = Image.new("RGB", (10, 12), color="red")
            img_buffer = io.BytesIO()
            img.save(img_buffer, format="JPEG")
            img_bytes = img_buffer.getvalue()

            info = tarfile.TarInfo(name=f"{class_name}_{i}.JPEG")
            info.size = len(img_bytes)
            class_tar.addfile(info, io.BytesIO(img_bytes))

    # Create main tar with class tar inside.
    class_tar_bytes = class_tar_buffer.getvalue()
    with tarfile.open(path, "w") as main_tar:
        info = tarfile.TarInfo(name=f"{class_name}.tar")
        info.size = len(class_tar_bytes)
        main_tar.addfile(info, io.BytesIO(class_tar_bytes))


def create_flat_tar(path: Path, num_images: int = 2):
    """Create a flat tar structure for ImageNet val/test."""
    with tarfile.open(path, "w") as tar:
        for i in range(num_images):
            img = Image.new("RGB", (10, 12), color="blue")
            img_buffer = io.BytesIO()
            img.save(img_buffer, format="JPEG")
            img_bytes = img_buffer.getvalue()

            info = tarfile.TarInfo(name=f"ILSVRC2012_val_{i:08d}.JPEG")
            info.size = len(img_bytes)
            tar.addfile(info, io.BytesIO(img_bytes))


def _extractfile_skipping(
    archive: tarfile.TarFile,
    member: tarfile.TarInfo | str,
    *,
    skip_name: str,
    original: Callable[
        [tarfile.TarFile, tarfile.TarInfo | str],
        IO[bytes] | None,
    ],
) -> IO[bytes] | None:
    if isinstance(member, tarfile.TarInfo) and member.name == skip_name:
        return None
    return original(archive, member)


def _skip_extractfile(
    monkeypatch: pytest.MonkeyPatch,
    *,
    skip_name: str,
) -> None:
    monkeypatch.setattr(
        tarfile.TarFile,
        "extractfile",
        partialmethod(
            _extractfile_skipping,
            skip_name=skip_name,
            original=tarfile.TarFile.extractfile,
        ),
    )


class TestImageNetSourceInit:
    """Test ImageNetSource initialization."""

    def test_init_train_split(
        self,
        temp_dir: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Initialization records the selected split and archive."""
        caplog.set_level(logging.INFO)
        train_tar = temp_dir / "ILSVRC2012_img_train.tar"
        create_nested_tar(train_tar, "n01440764")

        config = ImageNetSource.Config(working_dir=temp_dir, split="train")
        source = ImageNetSource(config)

        assert caplog.messages == [
            "ImageNetSource initialized: split=train, tar=ILSVRC2012_img_train.tar",
        ]
        assert source.split == "train"
        assert source.tar_path == train_tar
        assert source.shuffle is False
        assert source.worker_slice is None

    def test_init_val_split(self, temp_dir: Path) -> None:
        """Test initialization with val split."""
        val_tar = temp_dir / "ILSVRC2012_img_val.tar"
        create_flat_tar(val_tar, num_images=2)
        (temp_dir / "validation_labels.txt").write_text("n01440764\nn01443537\n")

        config = ImageNetSource.Config(working_dir=temp_dir, split="val")
        source = ImageNetSource(config)

        assert source.split == "val"
        assert source.tar_path == val_tar
        assert source.validation_labels is not None

    def test_init_val_with_labels(self, temp_dir: Path) -> None:
        """Test initialization with validation labels."""
        val_tar = temp_dir / "ILSVRC2012_img_val.tar"
        create_flat_tar(val_tar, num_images=2)

        labels_file = temp_dir / "explicit_labels.txt"
        labels_file.write_text("n01440764\nn01443537\n")

        config = ImageNetSource.Config(
            working_dir=temp_dir,
            split="val",
            validation_labels_file=labels_file,
        )
        source = ImageNetSource(config)

        assert source.validation_labels is not None
        assert len(source.validation_labels) == 2

    def test_val_split_finds_companion_labels_file(self, temp_dir: Path) -> None:
        """An unset override falls back to the labels file beside the archive.

        Without it every sample was labelled ``"unknown"``, so eval scored
        against no ground truth.
        """
        create_flat_tar(temp_dir / "ILSVRC2012_img_val.tar", num_images=2)
        (temp_dir / "validation_labels.txt").write_text("n01440764\nn01443537\n")

        source = ImageNetSource(
            ImageNetSource.Config(working_dir=temp_dir, split="val"),
        )

        assert source.validation_labels is not None
        assert next(iter(source)).get("label") == "n01440764"

    def test_val_split_without_companion_labels_is_refused(
        self,
        temp_dir: Path,
    ) -> None:
        """No labels file at all fails loudly rather than labelling "unknown"."""
        create_flat_tar(temp_dir / "ILSVRC2012_img_val.tar", num_images=2)
        labels_file = temp_dir / "validation_labels.txt"

        with pytest.raises(
            ValueError,
            match=r"^Validation labels not found at",
        ) as error:
            ImageNetSource(
                ImageNetSource.Config(working_dir=temp_dir, split="val"),
            )

        assert str(error.value) == (
            f"Validation labels not found at {labels_file}. Set "
            "validation_labels_file, or place validation_labels.txt "
            "beside the archive."
        )

    def test_init_test_split(self, temp_dir: Path) -> None:
        """Test initialization with test split."""
        test_tar = temp_dir / "ILSVRC2012_img_test.tar"
        create_flat_tar(test_tar)

        config = ImageNetSource.Config(working_dir=temp_dir, split="test")
        source = ImageNetSource(config)

        assert source.split == "test"
        assert source.tar_path == test_tar

    def test_init_test_split_v1(self, temp_dir: Path) -> None:
        """Test initialization with test_v1 tar file."""
        test_tar = temp_dir / "ILSVRC2012_img_test_v1.tar"
        create_flat_tar(test_tar)

        config = ImageNetSource.Config(working_dir=temp_dir, split="test")
        source = ImageNetSource(config)

        assert source.tar_path == test_tar

    def test_init_missing_dataset_dir(self):
        """Test error when dataset directory doesn't exist."""
        config = ImageNetSource.Config(working_dir="/nonexistent/path", split="train")
        with pytest.raises(ValueError, match="Dataset directory does not exist"):
            ImageNetSource(config)

    def test_init_missing_train_tar(self, temp_dir: Path) -> None:
        """Test error when train tar doesn't exist."""
        config = ImageNetSource.Config(working_dir=temp_dir, split="train")
        with pytest.raises(ValueError, match="Tar file not found"):
            ImageNetSource(config)

    def test_init_missing_val_tar(self, temp_dir: Path) -> None:
        """Test error when val tar doesn't exist."""
        config = ImageNetSource.Config(working_dir=temp_dir, split="val")
        with pytest.raises(ValueError, match="Tar file not found"):
            ImageNetSource(config)

    def test_init_missing_test_tar(self, temp_dir: Path) -> None:
        """Test error when test tar doesn't exist."""
        config = ImageNetSource.Config(working_dir=temp_dir, split="test")
        with pytest.raises(ValueError, match="No test tar found"):
            ImageNetSource(config)

    def test_init_unknown_split(self, temp_dir: Path) -> None:
        """A split outside the declared set is refused at construction.

        The annotation rules this out statically, so both checkers reject the
        assignment: that rejection is the point. A config arriving from JSON or
        a ``--override`` is unchecked text, and the runtime guard catches it.
        """
        config = ImageNetSource.Config(working_dir=temp_dir)
        config.split = "unknown"  # ty: ignore[invalid-assignment] -- The fixture uses the PIL test double's runtime-only attribute.  # pyright: ignore[reportAttributeAccessIssue] -- The fixture uses the PIL test double's runtime-only attribute.
        with pytest.raises(ValueError, match="Unknown split"):
            ImageNetSource(config)

    def test_init_with_shuffle(self, temp_dir: Path) -> None:
        """Test initialization with shuffle enabled."""
        train_tar = temp_dir / "ILSVRC2012_img_train.tar"
        create_nested_tar(train_tar, "n01440764")

        config = ImageNetSource.Config(
            working_dir=temp_dir,
            split="train",
            shuffle=True,
        )
        source = ImageNetSource(config)

        assert source.shuffle is True

    def test_init_with_slice(self, temp_dir: Path) -> None:
        """Test initialization with worker slicing."""
        train_tar = temp_dir / "ILSVRC2012_img_train.tar"
        create_nested_tar(train_tar, "n01440764")

        config = ImageNetSource.Config(
            working_dir=temp_dir,
            split="train",
            worker_slice=(0, 2),
        )
        source = ImageNetSource(config)

        assert source.worker_slice == (0, 2)

    def test_init_with_concurrency(self, temp_dir: Path) -> None:
        """Test initialization with concurrent shard reading."""
        train_tar = temp_dir / "ILSVRC2012_img_train.tar"
        create_nested_tar(train_tar, "n01440764")

        config = ImageNetSource.Config(
            working_dir=temp_dir,
            split="train",
            num_concurrently_read_shards=8,
        )
        source = ImageNetSource(config)

        assert source.num_concurrently_read_shards == 8


class TestImageNetValidationLabels:
    """Test validation label loading."""

    def test_load_validation_labels(self, temp_dir: Path) -> None:
        """Test loading validation labels from file."""
        val_tar = temp_dir / "ILSVRC2012_img_val.tar"
        create_flat_tar(val_tar, num_images=2)

        labels_file = temp_dir / "labels.txt"
        labels_file.write_text("n01440764\nn01443537\n")

        config = ImageNetSource.Config(
            working_dir=temp_dir,
            split="val",
            validation_labels_file=labels_file,
        )
        source = ImageNetSource(config)

        assert source.validation_labels is not None
        assert "ILSVRC2012_val_00000000.JPEG" in source.validation_labels
        assert source.validation_labels["ILSVRC2012_val_00000000.JPEG"] == "n01440764"
        assert source.validation_labels["ILSVRC2012_val_00000001.JPEG"] == "n01443537"

    def test_load_validation_labels_mismatch(self, temp_dir: Path) -> None:
        """Test error when number of labels doesn't match files."""
        val_tar = temp_dir / "ILSVRC2012_img_val.tar"
        create_flat_tar(val_tar, num_images=2)

        labels_file = temp_dir / "labels.txt"
        labels_file.write_text("n01440764\n")  # Only 1 label for 2 images.

        config = ImageNetSource.Config(
            working_dir=temp_dir,
            split="val",
            validation_labels_file=labels_file,
        )

        with pytest.raises(
            ValueError,
            match=r"^zip\(\) argument 2 is shorter than argument 1$",
        ) as error:
            ImageNetSource(config)

        assert str(error.value) == "zip() argument 2 is shorter than argument 1"


class TestImageNetIteration:
    """Test ImageNet iteration."""

    def test_iter_train_basic(self, temp_dir: Path) -> None:
        """Test basic training iteration."""
        train_tar = temp_dir / "ILSVRC2012_img_train.tar"
        create_nested_tar(train_tar, "n01440764", num_images=2)

        config = ImageNetSource.Config(working_dir=temp_dir, split="train")
        source = ImageNetSource(config)

        samples = list(source)
        assert len(samples) == 2
        assert samples[0].get("label") == "n01440764"
        assert samples[0].get("file_name") == "n01440764_0.JPEG"
        assert "image" in samples[0]

    def test_iter_train_multiple_classes(self, temp_dir: Path) -> None:
        """Test training iteration with multiple class tars."""
        train_tar = temp_dir / "ILSVRC2012_img_train.tar"

        # Create nested tar with multiple classes.
        class_tars: dict[str, bytes] = {}
        for class_id in ["n01440764", "n01443537"]:
            class_tar_buffer = io.BytesIO()
            with tarfile.open(fileobj=class_tar_buffer, mode="w") as class_tar:
                for i in range(2):
                    img = Image.new("RGB", (10, 12))
                    img_buffer = io.BytesIO()
                    img.save(img_buffer, format="JPEG")
                    info = tarfile.TarInfo(name=f"{class_id}_{i}.JPEG")
                    info.size = len(img_buffer.getvalue())
                    class_tar.addfile(info, io.BytesIO(img_buffer.getvalue()))

            class_tars[f"{class_id}.tar"] = class_tar_buffer.getvalue()

        create_test_tar(train_tar, class_tars)

        config = ImageNetSource.Config(working_dir=temp_dir, split="train")
        source = ImageNetSource(config)

        samples = list(source)
        assert len(samples) == 4

    def test_iter_train_with_shuffle(self, temp_dir: Path) -> None:
        """Test training iteration with shuffle."""
        train_tar = temp_dir / "ILSVRC2012_img_train.tar"
        create_nested_tar(train_tar, "n01440764")

        config = ImageNetSource.Config(
            working_dir=temp_dir,
            split="train",
            shuffle=True,
        )
        source = ImageNetSource(config)

        samples = list(source)
        assert len(samples) > 0

    def test_iter_train_with_slice(self, temp_dir: Path) -> None:
        """Test training iteration with worker slicing."""
        train_tar = temp_dir / "ILSVRC2012_img_train.tar"

        # Create multiple class tars.
        class_tars: dict[str, bytes] = {}
        for i in range(4):
            class_id = f"n0144{i:04d}"
            class_tar_buffer = io.BytesIO()
            with tarfile.open(fileobj=class_tar_buffer, mode="w") as class_tar:
                img = Image.new("RGB", (10, 12))
                img_buffer = io.BytesIO()
                img.save(img_buffer, format="JPEG")
                info = tarfile.TarInfo(name=f"{class_id}_0.JPEG")
                info.size = len(img_buffer.getvalue())
                class_tar.addfile(info, io.BytesIO(img_buffer.getvalue()))

            class_tars[f"{class_id}.tar"] = class_tar_buffer.getvalue()

        create_test_tar(train_tar, class_tars)

        # Worker 0 of 2 should get first 2 class tars.
        config = ImageNetSource.Config(
            working_dir=temp_dir,
            split="train",
            worker_slice=(0, 2),
        )
        source = ImageNetSource(config)

        samples = list(source)
        assert len(samples) == 2

    def test_iter_train_with_slice_last_worker(self, temp_dir: Path) -> None:
        """Test training iteration with last worker getting remainder."""
        train_tar = temp_dir / "ILSVRC2012_img_train.tar"

        # Create 5 class tars.
        class_tars: dict[str, bytes] = {}
        for i in range(5):
            class_id = f"n0144{i:04d}"
            class_tar_buffer = io.BytesIO()
            with tarfile.open(fileobj=class_tar_buffer, mode="w") as class_tar:
                img = Image.new("RGB", (10, 12))
                img_buffer = io.BytesIO()
                img.save(img_buffer, format="JPEG")
                info = tarfile.TarInfo(name=f"{class_id}_0.JPEG")
                info.size = len(img_buffer.getvalue())
                class_tar.addfile(info, io.BytesIO(img_buffer.getvalue()))

            class_tars[f"{class_id}.tar"] = class_tar_buffer.getvalue()

        create_test_tar(train_tar, class_tars)

        # Worker 1 of 2 should get 3 class tars (last worker gets remainder)
        config = ImageNetSource.Config(
            working_dir=temp_dir,
            split="train",
            worker_slice=(1, 2),
        )
        source = ImageNetSource(config)

        samples = list(source)
        assert len(samples) == 3

    def test_iter_train_parallel(self, temp_dir: Path) -> None:
        """Test training iteration with parallel shard reading."""
        train_tar = temp_dir / "ILSVRC2012_img_train.tar"

        # Create multiple class tars.
        class_tars: dict[str, bytes] = {}
        for i in range(3):
            class_id = f"n0144{i:04d}"
            class_tar_buffer = io.BytesIO()
            with tarfile.open(fileobj=class_tar_buffer, mode="w") as class_tar:
                for j in range(2):
                    img = Image.new("RGB", (10, 12))
                    img_buffer = io.BytesIO()
                    img.save(img_buffer, format="JPEG")
                    info = tarfile.TarInfo(name=f"{class_id}_{j}.JPEG")
                    info.size = len(img_buffer.getvalue())
                    class_tar.addfile(info, io.BytesIO(img_buffer.getvalue()))

            class_tars[f"{class_id}.tar"] = class_tar_buffer.getvalue()

        create_test_tar(train_tar, class_tars)

        config = ImageNetSource.Config(
            working_dir=temp_dir,
            split="train",
            num_concurrently_read_shards=2,
        )
        source = ImageNetSource(config)

        samples = list(source)
        file_names: list[str] = []
        for sample in samples:
            assert "file_name" in sample
            file_names.append(sample["file_name"])
        assert file_names == [
            "n01440000_0.JPEG",
            "n01440001_0.JPEG",
            "n01440000_1.JPEG",
            "n01440001_1.JPEG",
            "n01440002_0.JPEG",
            "n01440002_1.JPEG",
        ]

    def test_iter_train_parallel_more_shards_than_classes(self, temp_dir: Path) -> None:
        """Test parallel reading with more shards requested than classes."""
        train_tar = temp_dir / "ILSVRC2012_img_train.tar"

        # Create only 2 class tars.
        class_tars: dict[str, bytes] = {}
        for i in range(2):
            class_id = f"n0144{i:04d}"
            class_tar_buffer = io.BytesIO()
            with tarfile.open(fileobj=class_tar_buffer, mode="w") as class_tar:
                img = Image.new("RGB", (10, 12))
                img_buffer = io.BytesIO()
                img.save(img_buffer, format="JPEG")
                info = tarfile.TarInfo(name=f"{class_id}_0.JPEG")
                info.size = len(img_buffer.getvalue())
                class_tar.addfile(info, io.BytesIO(img_buffer.getvalue()))

            class_tars[f"{class_id}.tar"] = class_tar_buffer.getvalue()

        create_test_tar(train_tar, class_tars)

        # Request 5 concurrent shards but only 2 exist.
        config = ImageNetSource.Config(
            working_dir=temp_dir,
            split="train",
            num_concurrently_read_shards=5,
        )
        source = ImageNetSource(config)

        samples = list(source)
        assert len(samples) == 2

    def test_iter_train_sequential(self, temp_dir: Path) -> None:
        """Test sequential reading path (num_concurrently_read_shards=1)."""
        train_tar = temp_dir / "ILSVRC2012_img_train.tar"

        # Create multiple class tars.
        class_tars: dict[str, bytes] = {}
        for i in range(3):
            class_id = f"n0144{i:04d}"
            class_tar_buffer = io.BytesIO()
            with tarfile.open(fileobj=class_tar_buffer, mode="w") as class_tar:
                img = Image.new("RGB", (10, 12))
                img_buffer = io.BytesIO()
                img.save(img_buffer, format="JPEG")
                info = tarfile.TarInfo(name=f"{class_id}_0.JPEG")
                info.size = len(img_buffer.getvalue())
                class_tar.addfile(info, io.BytesIO(img_buffer.getvalue()))

            class_tars[f"{class_id}.tar"] = class_tar_buffer.getvalue()

        create_test_tar(train_tar, class_tars)

        # Sequential reading with num_concurrently_read_shards <= 1.
        config = ImageNetSource.Config(
            working_dir=temp_dir,
            split="train",
            num_concurrently_read_shards=1,
        )
        source = ImageNetSource(config)

        samples = list(source)
        assert len(samples) == 3

    def test_iter_train_zero_concurrency_uses_sequential_path(
        self,
        temp_dir: Path,
    ) -> None:
        train_tar = temp_dir / "ILSVRC2012_img_train.tar"
        create_nested_tar(train_tar, "n01440764", num_images=2)
        with pytest.raises(ValueError, match="num_concurrently_read_shards"):
            ImageNetSource(
                ImageNetSource.Config(
                    working_dir=temp_dir,
                    split="train",
                    num_concurrently_read_shards=0,
                ),
            )

    def test_iter_val_basic(self, temp_dir: Path) -> None:
        """Test basic validation iteration."""
        val_tar = temp_dir / "ILSVRC2012_img_val.tar"
        create_flat_tar(val_tar, num_images=3)
        (temp_dir / "validation_labels.txt").write_text(
            "n01440764\nn01443537\nn01484850\n",
        )

        config = ImageNetSource.Config(working_dir=temp_dir, split="val")
        source = ImageNetSource(config)

        samples = list(source)
        assert len(samples) == 3
        assert samples[0].get("label") == "n01440764"
        assert "image" in samples[0]

    def test_iter_val_with_labels(self, temp_dir: Path) -> None:
        """Test validation iteration with labels."""
        val_tar = temp_dir / "ILSVRC2012_img_val.tar"
        create_flat_tar(val_tar, num_images=2)

        labels_file = temp_dir / "labels.txt"
        labels_file.write_text("n01440764\nn01443537\n")

        config = ImageNetSource.Config(
            working_dir=temp_dir,
            split="val",
            validation_labels_file=labels_file,
        )
        source = ImageNetSource(config)

        samples = list(source)
        assert len(samples) == 2
        assert samples[0].keys() == {"file_name", "image", "label"}
        assert "file_name" in samples[0]
        assert "label" in samples[0]
        assert "image" in samples[0]
        assert samples[0]["file_name"] == "ILSVRC2012_val_00000000.JPEG"
        assert samples[0]["label"] == "n01440764"
        assert samples[0]["image"].size == (10, 12)
        assert "file_name" in samples[1]
        assert "label" in samples[1]
        assert samples[1]["file_name"] == "ILSVRC2012_val_00000001.JPEG"
        assert samples[1]["label"] == "n01443537"

    def test_iter_val_with_slice(self, temp_dir: Path) -> None:
        """Test validation iteration with worker slicing."""
        val_tar = temp_dir / "ILSVRC2012_img_val.tar"
        create_flat_tar(val_tar, num_images=4)
        (temp_dir / "validation_labels.txt").write_text("n01440764\n" * 4)

        config = ImageNetSource.Config(
            working_dir=temp_dir,
            split="val",
            worker_slice=(0, 2),
        )
        source = ImageNetSource(config)

        samples = list(source)
        assert len(samples) == 2

    def test_iter_val_with_slice_last_worker(self, temp_dir: Path) -> None:
        """Test validation iteration with last worker."""
        val_tar = temp_dir / "ILSVRC2012_img_val.tar"
        create_flat_tar(val_tar, num_images=5)
        (temp_dir / "validation_labels.txt").write_text("n01440764\n" * 5)

        config = ImageNetSource.Config(
            working_dir=temp_dir,
            split="val",
            worker_slice=(1, 2),
        )
        source = ImageNetSource(config)

        samples = list(source)
        assert len(samples) == 3

    def test_iter_test_basic(self, temp_dir: Path) -> None:
        """Test basic test iteration."""
        test_tar = temp_dir / "ILSVRC2012_img_test.tar"
        create_flat_tar(test_tar, num_images=3)

        config = ImageNetSource.Config(working_dir=temp_dir, split="test")
        source = ImageNetSource(config)

        samples = list(source)
        assert len(samples) == 3
        assert samples[0].keys() == {"file_name", "image", "label"}
        assert "file_name" in samples[0]
        assert "label" in samples[0]
        assert "image" in samples[0]
        assert samples[0]["file_name"] == "ILSVRC2012_val_00000000.JPEG"
        assert samples[0]["label"] == -1
        assert samples[0]["image"].size == (10, 12)

    def test_iter_test_with_slice(self, temp_dir: Path) -> None:
        """Test test iteration with worker slicing."""
        test_tar = temp_dir / "ILSVRC2012_img_test.tar"
        create_flat_tar(test_tar, num_images=4)

        config = ImageNetSource.Config(
            working_dir=temp_dir,
            split="test",
            worker_slice=(0, 2),
        )
        source = ImageNetSource(config)

        samples = list(source)
        assert len(samples) == 2

    @pytest.mark.parametrize(
        ("split", "expected_label"),
        [("val", "n01443537"), ("test", -1)],
    )
    def test_iter_flat_splits_continue_after_extractfile_returns_none(
        self,
        temp_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
        split: Literal["val", "test"],
        expected_label: str | int,
    ) -> None:
        archive_name = f"ILSVRC2012_img_{split}.tar"
        create_flat_tar(temp_dir / archive_name, num_images=2)
        if split == "val":
            (temp_dir / "validation_labels.txt").write_text(
                "n01440764\nn01443537\n",
            )
        _skip_extractfile(
            monkeypatch,
            skip_name="ILSVRC2012_val_00000000.JPEG",
        )
        source = ImageNetSource(
            ImageNetSource.Config(working_dir=temp_dir, split=split),
        )

        samples = list(source)

        assert [sample.get("file_name") for sample in samples] == [
            "ILSVRC2012_val_00000001.JPEG",
        ]
        assert [sample.get("label") for sample in samples] == [expected_label]

    def test_iter_val_extractfile_none(self, temp_dir: Path) -> None:
        """Test val iteration when extractfile returns None."""
        val_tar = temp_dir / "ILSVRC2012_img_val.tar"

        # Create tar with directory entry.
        with tarfile.open(val_tar, "w") as tar:
            info = tarfile.TarInfo(name="subdir")
            info.type = tarfile.DIRTYPE
            tar.addfile(info)
        (temp_dir / "validation_labels.txt").write_text("")

        config = ImageNetSource.Config(working_dir=temp_dir, split="val")
        source = ImageNetSource(config)

        samples = list(source)
        # Directory should be skipped.
        assert len(samples) == 0

    def test_iter_test_extractfile_none(self, temp_dir: Path) -> None:
        """Test test iteration when extractfile returns None."""
        test_tar = temp_dir / "ILSVRC2012_img_test.tar"

        # Create tar with directory entry.
        with tarfile.open(test_tar, "w") as tar:
            info = tarfile.TarInfo(name="subdir")
            info.type = tarfile.DIRTYPE
            tar.addfile(info)

        config = ImageNetSource.Config(working_dir=temp_dir, split="test")
        source = ImageNetSource(config)

        samples = list(source)
        # Directory should be skipped.
        assert len(samples) == 0

    def test_iter_handles_corrupt_images(
        self,
        temp_dir: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that corrupt images are skipped with warning."""
        val_tar = temp_dir / "ILSVRC2012_img_val.tar"

        # Create tar with corrupt image.
        with tarfile.open(val_tar, "w") as tar:
            info = tarfile.TarInfo(name="corrupt.JPEG")
            info.size = 10
            tar.addfile(info, io.BytesIO(b"notanimage"))
        (temp_dir / "validation_labels.txt").write_text("n01440764\n")

        config = ImageNetSource.Config(working_dir=temp_dir, split="val")
        caplog.set_level(logging.WARNING, logger="priml.data.sources.imagenet")
        source = ImageNetSource(config)

        samples = list(source)
        # Corrupt image should be skipped.
        assert len(samples) == 0
        assert len(caplog.records) == 1
        assert isinstance(caplog.records[0].args, tuple)
        assert caplog.records[0].args[0] == "corrupt.JPEG"
        assert isinstance(caplog.records[0].args[1], OSError)
        assert caplog.messages[0].startswith("Failed to load corrupt.JPEG: ")

    def test_iter_test_handles_corrupt_images(
        self,
        temp_dir: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that corrupt images in test split are skipped."""
        test_tar = temp_dir / "ILSVRC2012_img_test.tar"

        # Create tar with corrupt image.
        with tarfile.open(test_tar, "w") as tar:
            info = tarfile.TarInfo(name="corrupt.JPEG")
            info.size = 10
            tar.addfile(info, io.BytesIO(b"notanimage"))

        config = ImageNetSource.Config(working_dir=temp_dir, split="test")
        caplog.set_level(logging.WARNING, logger="priml.data.sources.imagenet")
        source = ImageNetSource(config)

        samples = list(source)
        # Corrupt image should be skipped.
        assert len(samples) == 0
        assert len(caplog.records) == 1
        assert isinstance(caplog.records[0].args, tuple)
        assert caplog.records[0].args[0] == "corrupt.JPEG"
        assert isinstance(caplog.records[0].args[1], OSError)
        assert caplog.messages[0].startswith("Failed to load corrupt.JPEG: ")


class TestImageNetReshuffle:
    """Test per-epoch reshuffling of the train split (epoch_seed folded in)."""

    @classmethod
    def _make_multiclass_train_tar(cls, path: Path, num_classes: int) -> None:
        class_tars: dict[str, bytes] = {}
        for i in range(num_classes):
            class_id = f"n0144{i:04d}"
            class_tar_buffer = io.BytesIO()
            with tarfile.open(fileobj=class_tar_buffer, mode="w") as class_tar:
                img = Image.new("RGB", (10, 12))
                img_buffer = io.BytesIO()
                img.save(img_buffer, format="JPEG")
                info = tarfile.TarInfo(name=f"{class_id}_0.JPEG")
                info.size = len(img_buffer.getvalue())
                class_tar.addfile(info, io.BytesIO(img_buffer.getvalue()))
            class_tars[f"{class_id}.tar"] = class_tar_buffer.getvalue()
        create_test_tar(path, class_tars)

    def test_train_reshuffles_across_epochs(self, temp_dir: Path) -> None:
        """Distinct epoch seeds give different class order, same class set.

        Epoch state originates outside the source (the loader injects
        ``epoch_seed`` per epoch), so reshuffling is driven by the seed.
        """
        train_tar = temp_dir / "ILSVRC2012_img_train.tar"
        self._make_multiclass_train_tar(train_tar, num_classes=8)

        config = ImageNetSource.Config(
            working_dir=temp_dir,
            split="train",
            shuffle=True,
            epoch_seed=0,
            num_concurrently_read_shards=1,
        )
        source = ImageNetSource(config)

        order_a = [cast(str, s.get("label")) for s in source]
        source.epoch_seed = 1
        order_b = [cast(str, s.get("label")) for s in source]

        assert order_a != order_b  # Reshuffled across epochs.
        assert sorted(order_a) == sorted(order_b)  # Same class set.
        assert len(order_a) == 8

    def test_per_epoch_order_identical_across_workers(self, temp_dir: Path) -> None:
        """All workers of one epoch share the seed: union is an exact partition."""
        train_tar = temp_dir / "ILSVRC2012_img_train.tar"
        num_classes = 8
        self._make_multiclass_train_tar(train_tar, num_classes)

        num_workers = 4
        union: list[object] = []
        for w in range(num_workers):
            config = ImageNetSource.Config(
                working_dir=temp_dir,
                split="train",
                shuffle=True,
                worker_slice=(w, num_workers),
                num_concurrently_read_shards=1,
            )
            worker = ImageNetSource(config)
            union.extend(s.get("label") for s in worker)

        # Every worker shuffles the same first-epoch permutation, so slices
        # partition the classes exactly: no gap, no duplicate.
        assert len(union) == num_classes
        assert len(set(union)) == num_classes


class TestImageNetLength:
    """Test __len__ method."""

    def test_len_train(self, temp_dir: Path) -> None:
        """Test length for train split."""
        train_tar = temp_dir / "ILSVRC2012_img_train.tar"
        create_nested_tar(train_tar, "n01440764")

        config = ImageNetSource.Config(working_dir=temp_dir, split="train")
        source = ImageNetSource(config)

        assert not isinstance(source, Sized)

    def test_len_val(self, temp_dir: Path) -> None:
        """Test length for val split."""
        val_tar = temp_dir / "ILSVRC2012_img_val.tar"
        create_flat_tar(val_tar, num_images=2)
        (temp_dir / "validation_labels.txt").write_text("n01440764\nn01443537\n")

        config = ImageNetSource.Config(working_dir=temp_dir, split="val")
        source = ImageNetSource(config)

        assert not isinstance(source, Sized)

    def test_len_test(self, temp_dir: Path) -> None:
        """Test length for test split."""
        test_tar = temp_dir / "ILSVRC2012_img_test.tar"
        create_flat_tar(test_tar)

        config = ImageNetSource.Config(working_dir=temp_dir, split="test")
        source = ImageNetSource(config)

        assert not isinstance(source, Sized)

    def test_iter_train_parallel_exhausted_shard(self, temp_dir: Path) -> None:
        """Test parallel reading when shards get exhausted (lines 141-142)."""
        train_tar = temp_dir / "ILSVRC2012_img_train.tar"

        # Create multiple class tars with varying number of images.
        class_tars: dict[str, bytes] = {}
        for i in range(4):
            class_id = f"n0144{i:04d}"
            class_tar_buffer = io.BytesIO()
            with tarfile.open(fileobj=class_tar_buffer, mode="w") as class_tar:
                # First tar has 1 image, others have 2.
                num_images = 1 if i == 0 else 2
                for j in range(num_images):
                    img = Image.new("RGB", (10, 12))
                    img_buffer = io.BytesIO()
                    img.save(img_buffer, format="JPEG")
                    info = tarfile.TarInfo(name=f"{class_id}_{j}.JPEG")
                    info.size = len(img_buffer.getvalue())
                    class_tar.addfile(info, io.BytesIO(img_buffer.getvalue()))

            class_tars[f"{class_id}.tar"] = class_tar_buffer.getvalue()

        create_test_tar(train_tar, class_tars)

        # Use parallel reading with 2 concurrent shards.
        config = ImageNetSource.Config(
            working_dir=temp_dir,
            split="train",
            num_concurrently_read_shards=2,
        )
        source = ImageNetSource(config)

        samples = list(source)
        # Should get all samples: 1 + 2 + 2 + 2 = 7.
        assert len(samples) == 7


class TestReadClassTar:
    """Test _read_class_tar function."""

    def test_read_class_tar_basic(self, temp_dir: Path) -> None:
        """Test reading a class tar."""
        train_tar = temp_dir / "ILSVRC2012_img_train.tar"
        create_nested_tar(train_tar, "n01440764", num_images=2)

        with tarfile.open(train_tar, "r") as tar:
            class_tar_member = tar.getmembers()[0]
            samples = list(_read_class_tar(tar, class_tar_member))

        assert len(samples) == 2
        assert samples[0].get("label") == "n01440764"

    def test_read_class_tar_continues_after_extractfile_returns_none(
        self,
        temp_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        train_tar = temp_dir / "ILSVRC2012_img_train.tar"
        create_nested_tar(train_tar, "n01440764", num_images=2)
        _skip_extractfile(monkeypatch, skip_name="n01440764_0.JPEG")

        with tarfile.open(train_tar) as tar:
            samples = list(_read_class_tar(tar, tar.getmembers()[0]))

        assert [sample.get("file_name") for sample in samples] == [
            "n01440764_1.JPEG",
        ]
        assert [sample.get("label") for sample in samples] == ["n01440764"]

    def test_read_class_tar_continues_after_directory(self, temp_dir: Path) -> None:
        """A directory member does not hide later images in the class tar."""
        train_tar = temp_dir / "ILSVRC2012_img_train.tar"
        class_tar_buffer = io.BytesIO()
        image_buffer = io.BytesIO()
        Image.new("RGB", (2, 3)).save(image_buffer, format="JPEG")
        image_bytes = image_buffer.getvalue()
        with tarfile.open(fileobj=class_tar_buffer, mode="w") as class_tar:
            directory = tarfile.TarInfo(name="subdir")
            directory.type = tarfile.DIRTYPE
            class_tar.addfile(directory)
            image = tarfile.TarInfo(name="later.JPEG")
            image.size = len(image_bytes)
            class_tar.addfile(image, io.BytesIO(image_bytes))

        with tarfile.open(train_tar, "w") as tar:
            member = tarfile.TarInfo(name="n01440764.tar")
            member.size = len(class_tar_buffer.getvalue())
            tar.addfile(member, io.BytesIO(class_tar_buffer.getvalue()))

        with tarfile.open(train_tar, "r") as tar:
            samples = list(_read_class_tar(tar, tar.getmembers()[0]))

        assert len(samples) == 1
        assert "file_name" in samples[0]
        assert "label" in samples[0]
        assert samples[0].get("file_name") == "later.JPEG"
        assert samples[0].get("label") == "n01440764"

    def test_read_class_tar_no_extractfile(
        self,
        temp_dir: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test handling when extractfile returns None."""
        train_tar = temp_dir / "ILSVRC2012_img_train.tar"

        # Create empty tar info.
        with tarfile.open(train_tar, "w") as tar:
            info = tarfile.TarInfo(name="test.tar")
            info.type = tarfile.DIRTYPE  # Directory, can't be extracted.
            tar.addfile(info)

        with tarfile.open(train_tar, "r") as tar:
            member = tar.getmembers()[0]
            samples = list(_read_class_tar(tar, member))

        assert len(samples) == 0
        assert caplog.messages == ["Could not extract test.tar"]

    def test_read_class_tar_corrupt_image(
        self,
        temp_dir: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test handling corrupt images in class tar."""
        train_tar = temp_dir / "ILSVRC2012_img_train.tar"

        # Create class tar with corrupt image.
        class_tar_buffer = io.BytesIO()
        with tarfile.open(fileobj=class_tar_buffer, mode="w") as class_tar:
            info = tarfile.TarInfo(name="corrupt.JPEG")
            info.size = 10
            class_tar.addfile(info, io.BytesIO(b"notanimage"))

        with tarfile.open(train_tar, "w") as tar:
            info = tarfile.TarInfo(name="n01440764.tar")
            info.size = len(class_tar_buffer.getvalue())
            tar.addfile(info, io.BytesIO(class_tar_buffer.getvalue()))

        with tarfile.open(train_tar, "r") as tar:
            member = tar.getmembers()[0]
            samples = list(_read_class_tar(tar, member))

        # Corrupt image should be skipped.
        assert len(samples) == 0
        assert len(caplog.records) == 1
        assert isinstance(caplog.records[0].args, tuple)
        assert caplog.records[0].args[0] == "corrupt.JPEG"
        assert isinstance(caplog.records[0].args[1], OSError)
        assert caplog.messages[0].startswith("Failed to load corrupt.JPEG: ")

    def test_read_class_tar_image_extractfile_none(self, temp_dir: Path) -> None:
        """Test handling when image extractfile returns None."""
        train_tar = temp_dir / "ILSVRC2012_img_train.tar"

        # Create class tar with directory entry.
        class_tar_buffer = io.BytesIO()
        with tarfile.open(fileobj=class_tar_buffer, mode="w") as class_tar:
            info = tarfile.TarInfo(name="subdir")
            info.type = tarfile.DIRTYPE
            class_tar.addfile(info)

        with tarfile.open(train_tar, "w") as tar:
            info = tarfile.TarInfo(name="n01440764.tar")
            info.size = len(class_tar_buffer.getvalue())
            tar.addfile(info, io.BytesIO(class_tar_buffer.getvalue()))

        with tarfile.open(train_tar, "r") as tar:
            member = tar.getmembers()[0]
            samples = list(_read_class_tar(tar, member))

        assert len(samples) == 0


def test_validation_directory_members_do_not_consume_labels(tmp_path: Path) -> None:
    archive = tmp_path / "ILSVRC2012_img_val.tar"
    create_flat_tar(archive)
    with tarfile.open(archive, "a") as tar:
        directory = tarfile.TarInfo("subdir")
        directory.type = tarfile.DIRTYPE
        tar.addfile(directory)
    (tmp_path / "validation_labels.txt").write_text("first\nsecond\n")
    source = ImageNetSource.Config(working_dir=tmp_path, split="val").make()
    assert [sample.get("label") for sample in source] == ["first", "second"]


def test_multiple_test_archives_are_rejected(tmp_path: Path) -> None:
    create_flat_tar(tmp_path / "ILSVRC2012_img_test_a.tar")
    create_flat_tar(tmp_path / "ILSVRC2012_img_test_b.tar")
    with pytest.raises(ValueError, match="Multiple test archives"):
        ImageNetSource.Config(working_dir=tmp_path, split="test").make()


@pytest.mark.parametrize("shards", [0, -1])
def test_imagenet_rejects_nonpositive_concurrency(tmp_path: Path, shards: int) -> None:
    create_nested_tar(tmp_path / "ILSVRC2012_img_train.tar", "class")
    config = ImageNetSource.Config(working_dir=tmp_path)
    config.num_concurrently_read_shards = shards
    with pytest.raises(ValueError, match="num_concurrently_read_shards"):
        config.make()


def test_string_labels_resolve_beneath_owner_and_members_are_cached(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    create_flat_tar(dataset / "ILSVRC2012_img_val.tar")
    (tmp_path / "labels.txt").write_text("first\nsecond\n")
    source = ImageNetSource.Config(
        base_dir=tmp_path,
        working_dir="/dataset",
        validation_labels_file="/labels.txt",
        split="val",
    ).make()

    def reject_scan(_: tarfile.TarFile) -> list[tarfile.TarInfo]:
        raise AssertionError("Validation members must reuse the construction scan.")

    monkeypatch.setattr(tarfile.TarFile, "getmembers", reject_scan)
    assert [sample.get("label") for sample in source] == ["first", "second"]


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
